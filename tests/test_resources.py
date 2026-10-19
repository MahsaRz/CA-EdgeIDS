import copy
import json

import numpy as np
import pytest
import torch
from scipy import sparse

from ca_edgeids.cli import demo_config
from ca_edgeids.data import prepare
from ca_edgeids.experiment import run, infer
from ca_edgeids.graph import FlowGraph, build_graph, halo_batch
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.profile import measure, profile_run
from ca_edgeids.worker import save_graph, load_graph


def star():
    s=np.array([0,0,0]); d=np.array([1,2,3])
    ms=np.r_[s,d]; md=np.r_[d,s]
    incoming=sparse.csr_matrix((np.arange(6),(md,np.arange(6))),shape=(4,6))
    return FlowGraph(s,d,np.ones((3,1),dtype=np.float32),np.ones((4,2),dtype=np.float32),
                     np.zeros(3,dtype=np.int64),np.arange(3),np.array(["a","b","c","d"]),
                     incoming,ms,md,np.tile(np.arange(3),2))


def test_packed_input_limit_matches_actual_tensor_storage():
    g=star(); q=np.array([0])
    batch=halo_batch(g,q,2)
    size=sum(v.numel()*v.element_size() for v in vars(batch).values() if isinstance(v,torch.Tensor))
    halo_batch(g,q,2,max_tensor_bytes=size)
    with pytest.raises(MemoryError,match="tensor limit exceeded"):
        halo_batch(g,q,2,max_tensor_bytes=size-1)
    with pytest.raises(ValueError,match="positive"):
        halo_batch(g,q,2,max_tensor_bytes=0)


def test_small_query_batches_can_repeat_the_entire_star_context():
    g=star(); owner=np.array([0,1,1,1])
    full,_=measure(g,np.zeros(4,dtype=int),1,2,1,full=True)
    chunked,rows=measure(g,owner,2,2,1)
    assert full["message_work_factor"]==1
    assert chunked["message_work_factor"]==3
    assert chunked["batches"]==3
    assert chunked["empty_target_partitions"]==1
    assert chunked["cut_flow_fraction"]==1
    assert chunked["coverage_min"]==chunked["coverage_max"]==1
    assert sum(r["target_flows"] for r in rows)==3
    changed=copy.deepcopy(g); changed.y[:]=99
    other,_=measure(changed,owner,2,2,1)
    assert {k:v for k,v in other.items() if k!="packing_profile_s"}=={k:v for k,v in chunked.items() if k!="packing_profile_s"}


def test_mapped_graph_preserves_logits_and_is_read_only(tmp_path):
    g=star(); save_graph(g,tmp_path/"graph")
    restored=load_graph(tmp_path/"graph")
    assert isinstance(restored.edge_x,np.memmap)
    assert not restored.edge_x.flags.writeable
    np.testing.assert_array_equal(restored.incoming.indices,g.incoming.indices)
    model=EdgeSAGE(1,2,8,2,0).eval()
    with torch.inference_mode():
        a=model(halo_batch(g,np.arange(3),2))
        b=model(halo_batch(restored,np.arange(3),2))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    with pytest.raises(FileExistsError): save_graph(g,tmp_path/"graph")


def test_csv_file_limit_stops_before_loading(tmp_path):
    cfg=demo_config(tmp_path/"fixture")
    with pytest.raises(MemoryError,match="file limit"):
        prepare(dict(cfg,max_input_bytes=1))


def test_new_graph_storage_runs_in_fresh_workers_and_profiles(tmp_path):
    cfg=demo_config(tmp_path/"fixture")
    cfg.update(seeds=[7],epochs=1,methods=["full","random"],partition_counts=[2],repeats=1,
               baselines=False,isolated_benchmarks=True)
    out=tmp_path/"run"
    run(cfg,out)
    assert (out/"test_graph/format.json").exists()
    assert not (out/"test_graph.npz").exists()
    assert json.loads((out/"unavailable_runs.json").read_text())==[]
    profile=profile_run(out,tmp_path/"profile")
    manifest=json.loads((profile/"profile_manifest.json").read_text())
    assert manifest["graph_storage"]=="memory-mapped arrays"
    assert not manifest["unavailable"]
    assert manifest["max_batch_tensor_bytes"]==256*1024**2
    import pandas as pd
    work=pd.read_csv(profile/"work_summary.csv")
    assert set(work.method)=={"full","chunked","random"}
    assert work.loc[work.method=="full","message_work_factor"].item()==1
    assert not list(profile.glob("*.npy"))
    from ca_edgeids.batch_study import run_study
    study=run_study(out,tmp_path/"batch-study",batch_sizes=[64],partition_counts=[2],
                    methods=["chunked","random"],repeats=1)
    completion=json.loads((study/"completion.json").read_text())
    assert completion["requested_conditions"]==completion["completed_conditions"]==3
    assert completion["timing_rows"]==3 and not completion["model_retrained"]
    timings=pd.read_csv(study/"timing_repeats.csv")
    assert timings.loc[timings.method=="full","batches"].item()==1
    assert timings.loc[timings.method=="chunked","batches"].item()>1
    assert set(timings.warmup_full_passes)=={1}
    assert (timings.first_prediction_process_s>0).all()
    assert (timings.first_prediction_process_s<=timings.worker_process_wall_s).all()
    assert (timings.first_prediction_in_worker_s>=timings.graph_load_s+timings.checkpoint_load_s+timings.model_setup_s).all()
    assert all(r["same_class_predictions"] for r in json.loads((study/"prediction_checks.json").read_text()))


def test_batch_plan_keeps_full_pass_separate_from_batch_cap():
    from ca_edgeids.batch_study import conditions
    jobs=conditions([256,512],[2,4],["chunked","random"])
    assert jobs.count(("full",1,0))==1
    assert ("chunked",1,256) in jobs and ("random",4,512) in jobs
    assert len(jobs)==7
    with pytest.raises(ValueError): conditions([0],[2],["chunked"])
    with pytest.raises(ValueError): conditions([256,256],[2],["chunked"])


def test_partition_average_includes_empty_trailing_partitions():
    g=star(); model=EdgeSAGE(1,2,8,2,0).eval()
    cfg={"device":"cpu","classes":2,"max_edges_per_batch":2}
    _,timing=infer(model,g,np.zeros(4,dtype=int),cfg,partition_count=4)
    assert timing["requested_partition_count"]==4
    assert timing["nonempty_target_partitions"]==1
    assert timing["average_partition_forward_s"]==timing["model_forward_s"]/4
