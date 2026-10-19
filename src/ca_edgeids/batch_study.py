"""Compare inference batch sizes with frozen models and fixed test rows."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .data import sha256_file
from .experiment import dump
from .graph import partition_nodes
from .runtime import environment
from .worker import load_graph, save_graph


def conditions(batch_sizes,partition_counts,methods):
    if not batch_sizes or any(b<1 for b in batch_sizes) or len(set(batch_sizes))!=len(batch_sizes):
        raise ValueError("Batch sizes must be unique positive integers")
    if not partition_counts or any(p<1 for p in partition_counts) or len(set(partition_counts))!=len(partition_counts):
        raise ValueError("Partition counts must be unique positive integers")
    if not methods or len(set(methods))!=len(methods) or set(methods)-{"chunked","random","metis"}:
        raise ValueError("Use unique chunked, random or metis methods")
    return [("full",1,0)]+[(method,parts,batch) for method in methods
        for parts in ([1] if method=="chunked" else partition_counts) for batch in batch_sizes]


def run_study(run_dir,out,batch_sizes=(256,512,1024,2048),partition_counts=(2,4,8),
              methods=("chunked","random","metis"),repeats=3,order_seed=20261019,timeout_s=120):
    jobs=conditions(batch_sizes,partition_counts,methods)
    if repeats<1 or timeout_s<=0: raise ValueError("Repeat count and timeout must be positive")
    run_dir=Path(run_dir); out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite {out}")
    cfg=json.loads((run_dir/"resolved_config.json").read_text())
    if cfg["task"]!="binary": raise ValueError("This pilot study is defined for binary runs")
    data=json.loads((run_dir/"dataset_manifest.json").read_text())
    if len(cfg["seeds"])!=1: raise ValueError("Supply a separate frozen one-seed run for each timing study")
    seed=cfg["seeds"][0]; checkpoint=run_dir/f"seed-{seed}/model.pt"
    source_graph=run_dir/"test_graph"
    if not source_graph.is_dir(): source_graph=run_dir/"test_graph.npz"
    graph=load_graph(source_graph)
    row_hash=hashlib.sha256(graph.row_ids.astype("<i8").tobytes()).hexdigest()
    if row_hash!=data["test_row_ids_sha256"]: raise ValueError("Stored graph does not match the frozen test rows")
    anchor=run_dir/f"seed-{seed}/full-1-predictions.npz"
    with np.load(anchor,allow_pickle=False) as saved:
        if not np.array_equal(saved["row_ids"],graph.row_ids): raise ValueError("Reference row order differs")
        reference=saved["probabilities"].copy()
    out.mkdir(parents=True,exist_ok=True)
    np.random.default_rng(order_seed).shuffle(jobs)
    dump(out/"study_plan.json",{
        "dataset":cfg["dataset"],"scope":"fixed-checkpoint binary pilot; inference scheduling only",
        "source_run":run_dir.name,"model_seed":seed,"batch_sizes":list(batch_sizes),"partition_counts":list(partition_counts),
        "methods":list(methods),"repeats":repeats,"order_seed":order_seed,"timeout_s":timeout_s,
        "execution_order":[{"method":m,"partitions":p,"batch_size":b} for m,p,b in jobs],
        "full_batch_size_zero_means":"All test targets in a single forward call; batch-size cap is ignored",
        "checkpoint_sha256":sha256_file(checkpoint),"source_csv_sha256":data["csv_sha256"],
        "target_row_ids_sha256":row_hash,"reference_predictions_sha256":sha256_file(anchor),
        "environment":environment(),"package_source_sha256":{p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "timing":"Fresh process per condition; one complete warm-up pass; repeated whole-graph inference. Loading, graph conversion, partitioning, warm-up and process startup are outside total_inference_s.",
        "memory":"Sampled process RSS with resident input graph; activations from warm-up may remain in allocator caches. Packed graph arrays use read-only memory mapping. No claim of a cold-start or end-to-end online latency."})
    t=time.perf_counter(); save_graph(graph,out/"test_graph")
    dump(out/"graph_storage.json",{"conversion_s":time.perf_counter()-t,
        "array_sha256":{p.name:sha256_file(p) for p in sorted((out/"test_graph").glob("*.npy"))}})
    rows=[]; failures=[]; checks=[]
    for index,(method,parts,batch_size) in enumerate(jobs):
        tag=f"{method}-p{parts}-b{batch_size}"
        identity={"method":method,"partitions":parts,"batch_size":batch_size,"order_index":index}
        try:
            partition_nodes(graph,"full" if method=="chunked" else method,parts,seed)
        except (ValueError,RuntimeError) as exc:
            if isinstance(exc,RuntimeError) and "METIS unavailable" not in str(exc): raise
            failures.append(dict(identity,stage="feasibility",error=str(exc)))
            continue
        command=[sys.executable,"-m","ca_edgeids.worker","--graph",str((out/"test_graph").resolve()),
                 "--checkpoint",str(checkpoint.resolve()),"--method",method,"--parts",str(parts),
                 "--repeats",str(repeats),"--out",str((out/tag).resolve())]
        if batch_size: command.extend(["--batch-size",str(batch_size)])
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=timeout_s)
        except subprocess.TimeoutExpired:
            failures.append(dict(identity,stage="worker",error=f"Timed out after {timeout_s} seconds")); continue
        if result.returncode:
            failures.append(dict(identity,stage="worker",error=result.stderr))
            dump(out/"unavailable_runs.json",failures)
            if not any(x in result.stderr.lower() for x in ["out of memory","tensor limit exceeded","memoryerror"]):
                raise RuntimeError(result.stderr)
            continue
        with np.load(out/(tag+".npz"),allow_pickle=False) as saved: probabilities=saved["probabilities"]
        difference=float(np.max(np.abs(probabilities-reference)))
        if not np.allclose(probabilities,reference,atol=1e-5,rtol=1e-5) or not np.array_equal(probabilities.argmax(1),reference.argmax(1)):
            raise AssertionError(f"Frozen model predictions changed for {tag}")
        checks.append(dict(identity,max_probability_difference=difference,same_class_predictions=True))
        records=json.loads((out/(tag+".json")).read_text())
        if len(records)!=repeats: raise AssertionError("Worker did not return every timing repeat")
        for record in records:
            if record["target_row_ids_sha256"]!=row_hash or record["coverage_min"]!=1 or record["coverage_max"]!=1:
                raise AssertionError("Timing condition changed target coverage")
            rows.append(dict(dataset=cfg["dataset"],model_seed=seed,**identity,**record))
        pd.DataFrame(rows).to_csv(out/"timing_repeats.csv",index=False)
        print(f"{cfg['dataset']}: {tag} checked",flush=True)
    pd.DataFrame(rows).to_csv(out/"timing_repeats.csv",index=False)
    dump(out/"prediction_checks.json",checks)
    dump(out/"unavailable_runs.json",failures)
    if len(checks)+len(failures)!=len(jobs) or len(rows)!=repeats*len(checks):
        raise AssertionError("Study conditions are missing from the final record")
    dump(out/"completion.json",{"requested_conditions":len(jobs),"completed_conditions":len(checks),
        "unavailable_conditions":len(failures),"timing_rows":len(rows),"model_retrained":False})
    return out
