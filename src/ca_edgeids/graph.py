"""Directed flow graph, label-free structural features, exact query halos."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import torch
from .data import SOURCE, DEST


@dataclass
class FlowGraph:
    src: np.ndarray
    dst: np.ndarray
    edge_x: np.ndarray
    node_x: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    hosts: np.ndarray
    incoming: object
    message_src: np.ndarray
    message_dst: np.ndarray
    message_flow: np.ndarray


def build_graph(prepared, part, bidirectional=True):
    ids = np.flatnonzero(prepared.split == part)
    frame = prepared.frame.iloc[ids]
    hosts = np.asarray(np.sort(np.unique(np.concatenate([frame[SOURCE].astype(str),frame[DEST].astype(str)]))),dtype=str)
    s = np.searchsorted(hosts, frame[SOURCE].astype(str)).astype(np.int64)
    d = np.searchsorted(hosts, frame[DEST].astype(str)).astype(np.int64)
    # One prediction target per original record; reverse message arcs never become extra targets.
    ms = np.concatenate([s,d]) if bidirectional else s.copy()
    md = np.concatenate([d,s]) if bidirectional else d.copy()
    mf = np.tile(np.arange(len(ids),dtype=np.int64),2) if bidirectional else np.arange(len(ids),dtype=np.int64)
    # CSR rows are destination nodes, with message arc ids stored as data.
    incoming = sparse.csr_matrix((np.arange(len(ms)),(md,np.arange(len(ms)))),shape=(len(hosts),len(ms)))
    nx = np.stack([np.log1p(np.bincount(s,minlength=len(hosts))),np.log1p(np.bincount(d,minlength=len(hosts)))],axis=1).astype(np.float32)
    source_ids=frame["source_row_id"].to_numpy(np.int64) if "source_row_id" in frame else ids
    return FlowGraph(s,d,prepared.features[ids],nx,prepared.labels[ids],source_ids,hosts,incoming,ms,md,mf)


def partition_nodes(g, method, parts, seed):
    n = len(g.hosts)
    if method == "full":
        if parts != 1:
            raise ValueError("Full graph requires exactly one partition")
        return np.zeros(n,dtype=np.int64)
    if not 1 <= parts <= n:
        raise ValueError(f"Requested {parts} partitions for {n} nodes")
    if method == "random":
        owner = np.empty(n,dtype=np.int64)
        owner[np.random.default_rng(seed).permutation(n)] = np.arange(n) % parts
        return owner
    if method != "metis":
        raise ValueError(method)
    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError("METIS unavailable. Install the metis extra on a supported platform; no substitute is called METIS.") from exc
    # METIS receives a simple undirected topology, with no labels and no self loops.
    adj = sparse.coo_matrix((np.ones(2*len(g.src)),(np.r_[g.src,g.dst],np.r_[g.dst,g.src])),shape=(n,n)).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.sort_indices()
    options = pymetis.Options(seed=int(seed))
    _, membership = pymetis.part_graph(parts,xadj=adj.indptr.tolist(),adjncy=adj.indices.tolist(),options=options)
    result = np.asarray(membership,dtype=np.int64)
    if result.shape != (n,) or result.min()<0 or result.max()>=parts:
        raise RuntimeError("Invalid METIS assignment")
    return result


@dataclass
class Batch:
    node_x: torch.Tensor
    edge_x: torch.Tensor
    msg_src: torch.Tensor
    msg_dst: torch.Tensor
    msg_flow: torch.Tensor
    query_src: torch.Tensor
    query_dst: torch.Tensor
    query_x: torch.Tensor
    query_ids: np.ndarray
    y: torch.Tensor


def halo_batch(g, query_ids, layers, device="cpu"):
    """K-hop incoming closure of BOTH query endpoints preserves full-graph logits.

    Context can cross partition boundaries. This deliberately trades extra work for
    coverage/correctness; it is not isolated Cluster-GCN sampling or distributed execution.
    """
    q = np.asarray(query_ids,dtype=np.int64)
    if len(q)==0:
        raise ValueError("Empty query batch")
    nodes = np.unique(np.r_[g.src[q],g.dst[q]])
    frontier = nodes
    for _ in range(layers):
        arc_ids = g.incoming[frontier].indices
        expanded = np.unique(g.message_src[arc_ids])
        frontier = np.setdiff1d(expanded,nodes,assume_unique=True)
        nodes = np.union1d(nodes,expanded)
        if len(frontier)==0:
            break
    arcs = g.incoming[nodes].indices
    arcs = arcs[np.isin(g.message_src[arcs],nodes,assume_unique=False)]
    flow_ids, inverse = np.unique(g.message_flow[arcs],return_inverse=True)
    tensor = lambda x, dtype: torch.as_tensor(x,dtype=dtype,device=device)
    return Batch(
        tensor(g.node_x[nodes],torch.float32), tensor(g.edge_x[flow_ids],torch.float32),
        tensor(np.searchsorted(nodes,g.message_src[arcs]),torch.long),
        tensor(np.searchsorted(nodes,g.message_dst[arcs]),torch.long),tensor(inverse,torch.long),
        tensor(np.searchsorted(nodes,g.src[q]),torch.long),tensor(np.searchsorted(nodes,g.dst[q]),torch.long),
        tensor(g.edge_x[q],torch.float32),q,tensor(g.y[q],torch.long))


def query_batches(g, owner, max_edges=4096):
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    # Stable sort once rather than scanning all target edges for every partition.
    partition = owner[g.src]
    order = np.argsort(partition,kind="stable")
    boundaries = np.r_[0,np.flatnonzero(np.diff(partition[order]))+1,len(order)]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        for start in range(a,b,max_edges):
            yield int(partition[order[start]]), order[start:min(start+max_edges,b)]
