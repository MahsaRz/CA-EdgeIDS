"""Fresh-process inference worker for controlled configuration-local memory."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
import torch
from .graph import FlowGraph,partition_nodes,query_batches,halo_batch
from .model import EdgeSAGE
from .runtime import seed_all,sync
from .experiment import infer,dump


def save_graph(g,path):
    path=Path(path)
    if path.exists(): raise FileExistsError(f"Refusing to overwrite saved graph: {path}")
    arrays={k:getattr(g,k) for k in ["src","dst","edge_x","node_x","y","row_ids","hosts","message_src","message_dst","message_flow"]}
    if path.suffix==".npz":
        np.savez(path,**arrays)
        return
    path.mkdir(parents=True,exist_ok=False)
    arrays.update(incoming_data=g.incoming.data,incoming_indices=g.incoming.indices,incoming_indptr=g.incoming.indptr)
    for name,value in arrays.items(): np.save(path/(name+".npy"),value,allow_pickle=False)
    dump(path/"format.json",{"version":1,"storage":"read-only memory-mapped NumPy arrays","nodes":len(g.hosts),"message_arcs":len(g.message_src)})


def load_graph(path):
    path=Path(path)
    if path.is_dir():
        meta=json.loads((path/"format.json").read_text())
        if meta["version"]!=1: raise ValueError("Unknown graph storage version")
        a={k:np.load(path/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in
           ["src","dst","edge_x","node_x","y","row_ids","hosts","message_src","message_dst","message_flow"]}
        csr=[np.load(path/("incoming_"+k+".npy"),mmap_mode="r",allow_pickle=False) for k in ["data","indices","indptr"]]
        incoming=sparse.csr_matrix(tuple(csr),shape=(meta["nodes"],meta["message_arcs"]),copy=False)
        if len(a["hosts"])!=meta["nodes"] or len(a["message_src"])!=meta["message_arcs"]:
            raise ValueError("Graph arrays do not match stored dimensions")
        return FlowGraph(**a,incoming=incoming)
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    n=len(a["hosts"]); m=len(a["message_src"])
    incoming=sparse.csr_matrix((np.arange(m),(a["message_dst"],np.arange(m))),shape=(n,m))
    return FlowGraph(**a,incoming=incoming)


def benchmark(graph_path,checkpoint,method,parts,out,batch_size=None,repeats=None):
    if Path(str(out)+".json").exists() or Path(str(out)+".npz").exists():
        raise FileExistsError("Refusing to overwrite prior worker output")
    # Only a checkpoint written by this project is loaded, never an external model.
    checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=copy.deepcopy(checkpoint["config"]); seed=checkpoint["seed"]
    if batch_size is not None: cfg["max_edges_per_batch"]=batch_size
    if repeats is not None: cfg["repeats"]=repeats
    if cfg["max_edges_per_batch"]<1 or cfg["repeats"]<1:
        raise ValueError("Batch size and repeat count must be positive")
    seed_all(seed,cfg.get("threads",1))
    g=load_graph(graph_path)
    model=EdgeSAGE(g.edge_x.shape[1],cfg["classes"],cfg["hidden"],cfg["layers"],cfg["dropout"]).to(cfg["device"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    t=time.perf_counter(); owner=partition_nodes(g,"full" if method=="chunked" else method,parts,seed); partition_s=time.perf_counter()-t
    warm=time.perf_counter()
    infer(model,g,owner,cfg,full_graph=method=="full",partition_count=parts)
    sync(cfg["device"]); warm_s=time.perf_counter()-warm
    records=[]
    ref=None
    for repeat in range(cfg["repeats"]):
        p,timing=infer(model,g,owner,cfg,full_graph=method=="full",partition_count=parts)
        if ref is None: ref=p
        elif not np.allclose(ref,p,atol=1e-6,rtol=1e-6): raise AssertionError("Repeated deterministic inference changed probabilities")
        records.append(dict(repeat=repeat,partitioning_s=partition_s,warmup_s=warm_s,warmup_full_passes=1,
                            max_edges_per_batch=cfg["max_edges_per_batch"],memory_process="fresh per method/partition/batch-size configuration",**timing))
    np.savez_compressed(str(out)+".npz",probabilities=ref,owner=owner)
    dump(str(out)+".json",records)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--graph",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--method",required=True); parser.add_argument("--parts",type=int,required=True); parser.add_argument("--out",required=True)
    parser.add_argument("--batch-size",type=int); parser.add_argument("--repeats",type=int)
    a=parser.parse_args(); benchmark(a.graph,a.checkpoint,a.method,a.parts,a.out,a.batch_size,a.repeats)
