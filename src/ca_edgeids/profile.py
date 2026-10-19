"""Count exact-context work without fitting or scoring a classifier."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .experiment import dump
from .graph import partition_nodes, query_batches, halo_batch
from .runtime import seed_all, environment
from .worker import load_graph


def measure(g, owner, parts, depth, batch_size, full=False, tensor_limit=256*1024**2):
    """Integer workload counts; elapsed packing time is not model latency."""
    seen=np.zeros(len(g.src),dtype=np.int64)
    rows=[]
    start=time.perf_counter()
    for part,q in query_batches(g,owner,len(g.src) if full else batch_size):
        b=halo_batch(g,q,depth,"cpu",tensor_limit)
        endpoints=len(np.unique(np.r_[g.src[q],g.dst[q]]))
        size=sum(v.numel()*v.element_size() for v in vars(b).values() if isinstance(v,torch.Tensor))
        rows.append({"partition":part,"target_flows":len(q),"query_endpoint_nodes":endpoints,
                     "context_nodes":len(b.node_x),"extra_context_nodes":len(b.node_x)-endpoints,
                     "message_arcs":len(b.msg_src),"context_flow_features":len(b.edge_x),"packed_tensor_bytes":size})
        seen[q]+=1
    if not np.all(seen==1): raise AssertionError("Profile missed or repeated prediction targets")
    node_load=np.bincount(owner,minlength=parts)
    flow_load=np.bincount(owner[g.src],minlength=parts)
    result={"requested_partitions":parts,"target_flows":len(g.src),"graph_nodes":len(g.hosts),
            "graph_message_arcs":len(g.message_src),"batches":len(rows),"coverage_min":int(seen.min()),"coverage_max":int(seen.max()),
            "cut_flow_fraction":float(np.mean(owner[g.src]!=owner[g.dst])),
            "empty_node_partitions":int(np.sum(node_load==0)),"empty_target_partitions":int(np.sum(flow_load==0)),
            "min_partition_target_flows":int(flow_load.min()),"max_partition_target_flows":int(flow_load.max()),
            "total_context_node_visits":sum(r["context_nodes"] for r in rows),
            "total_message_arc_visits":sum(r["message_arcs"] for r in rows),
            "message_work_factor":sum(r["message_arcs"] for r in rows)/len(g.message_src),
            "node_work_factor":sum(r["context_nodes"] for r in rows)/len(g.hosts),
            "max_packed_tensor_bytes":max(r["packed_tensor_bytes"] for r in rows),
            "total_extra_context_node_visits":sum(r["extra_context_nodes"] for r in rows),
            "packing_profile_s":time.perf_counter()-start}
    return result,rows


def profile_run(run_dir,out):
    run_dir=Path(run_dir); out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite {out}")
    cfg=json.loads((run_dir/"resolved_config.json").read_text())
    manifest=json.loads((run_dir/"dataset_manifest.json").read_text())
    seed_all(cfg["seeds"][0],cfg.get("threads",1))
    graph_path=run_dir/"test_graph"
    if not graph_path.is_dir(): graph_path=run_dir/"test_graph.npz"
    g=load_graph(graph_path)
    target_hash=hashlib.sha256(g.row_ids.astype("<i8").tobytes()).hexdigest()
    if target_hash!=manifest["test_row_ids_sha256"]: raise ValueError("Stored graph differs from the frozen test rows")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; batches=[]; failures=[]
    limit=cfg.get("max_batch_tensor_bytes",256*1024**2)
    for method in ["chunked"]+cfg["methods"]:
        for parts in ([1] if method in {"full","chunked"} else cfg["partition_counts"]):
            try:
                t=time.perf_counter()
                owner=partition_nodes(g,"full" if method=="chunked" else method,parts,cfg["seeds"][0])
                partition_s=time.perf_counter()-t
                result,detail=measure(g,owner,parts,cfg["layers"],cfg["max_edges_per_batch"],method=="full",limit)
                rows.append(dict(method=method,partitioning_profile_s=partition_s,**result))
                batches.extend(dict(method=method,requested_partitions=parts,**r) for r in detail)
            except (ValueError,MemoryError,RuntimeError) as exc:
                if isinstance(exc,RuntimeError) and "METIS unavailable" not in str(exc): raise
                failures.append({"method":method,"partitions":parts,"error":str(exc)})
    pd.DataFrame(rows).to_csv(out/"work_summary.csv",index=False)
    pd.DataFrame(batches).to_csv(out/"batch_work.csv",index=False)
    arrays=[v for v in vars(g).values() if isinstance(v,np.ndarray)]
    array_bytes=sum(v.nbytes for v in arrays)+sum(v.nbytes for v in [g.incoming.data,g.incoming.indices,g.incoming.indptr])
    dump(out/"profile_manifest.json",{
        "dataset":cfg["dataset"],"scope":cfg.get("data_scope","unspecified"),"source_csv_sha256":manifest["csv_sha256"],
        "target_row_ids_sha256":target_hash,"source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "graph_storage":"memory-mapped arrays" if graph_path.is_dir() else "legacy NPZ loaded into memory",
        "graph_array_bytes":array_bytes,"layers":cfg["layers"],"batch_size":cfg["max_edges_per_batch"],
        "partition_seed":cfg["seeds"][0],"max_batch_tensor_bytes":limit,"unavailable":failures,"environment":environment(),
        "meaning":"Exact integer workload counts, not model performance. Factors sum context visits across all batches relative to one full graph. Repeated arcs are repeated work, not duplicated prediction targets. Packing times use a shared process and are not a controlled inference benchmark.",
        "limits":"Packed tensors exclude graph storage, index temporaries, model activations, gradients and optimizer state. File and packed-tensor limits do not guarantee a total RAM bound."})
    return out
