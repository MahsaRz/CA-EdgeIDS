"""Fresh-process inference worker for controlled configuration-local memory."""
import argparse
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
    np.savez(path,**{k:getattr(g,k) for k in ["src","dst","edge_x","node_x","y","row_ids","hosts","message_src","message_dst","message_flow"]})


def load_graph(path):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    n=len(a["hosts"]); m=len(a["message_src"])
    incoming=sparse.csr_matrix((np.arange(m),(a["message_dst"],np.arange(m))),shape=(n,m))
    return FlowGraph(**a,incoming=incoming)


def benchmark(graph_path,checkpoint,method,parts,out):
    # Only a checkpoint written by this project is loaded, never an external model.
    checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=checkpoint["config"]; seed=checkpoint["seed"]
    seed_all(seed,cfg.get("threads",1))
    g=load_graph(graph_path)
    model=EdgeSAGE(g.edge_x.shape[1],cfg["classes"],cfg["hidden"],cfg["layers"],cfg["dropout"]).to(cfg["device"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    t=time.perf_counter(); owner=partition_nodes(g,method,parts,seed); partition_s=time.perf_counter()-t
    warm=time.perf_counter()
    first=next(query_batches(g,owner,len(g.src) if method=="full" else cfg["max_edges_per_batch"]))[1]
    with torch.inference_mode(): model(halo_batch(g,first,model.depth,cfg["device"]))
    sync(cfg["device"]); warm_s=time.perf_counter()-warm
    records=[]
    ref=None
    for repeat in range(cfg["repeats"]):
        p,timing=infer(model,g,owner,cfg,full_graph=method=="full")
        if ref is None: ref=p
        elif not np.allclose(ref,p,atol=1e-6,rtol=1e-6): raise AssertionError("Repeated deterministic inference changed probabilities")
        records.append(dict(repeat=repeat,partitioning_s=partition_s,warmup_s=warm_s,memory_process="fresh per method/partition configuration",**timing))
    np.savez_compressed(str(out)+".npz",probabilities=ref,owner=owner)
    dump(str(out)+".json",records)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--graph",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--method",required=True); parser.add_argument("--parts",type=int,required=True); parser.add_argument("--out",required=True)
    a=parser.parse_args(); benchmark(a.graph,a.checkpoint,a.method,a.parts,a.out)
