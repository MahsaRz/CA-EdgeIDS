"""Label-free input summaries for linear mean messages; learned states stay live."""
from dataclasses import dataclass,field
import hashlib

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F


@dataclass
class InputSummary:
    adjacency: object
    mean_flow: np.ndarray
    has_incoming: np.ndarray
    cache_enabled: bool = True
    closure_cache_bytes: int = 64*1024**2
    _memo: dict = field(default_factory=dict,repr=False)
    _packed: object = field(default=None,repr=False)


def summarize(g,max_bytes=256*1024**2,chunk_size=4096):
    n=len(g.node_x); d=g.edge_x.shape[1]; m=len(g.message_src)
    estimate=n*d*12+m*40+min(m,chunk_size)*d*4
    if max_bytes<=0 or chunk_size<1: raise ValueError('Positive summary limits required')
    if estimate>max_bytes: raise MemoryError('Input-summary storage estimate exceeds the configured limit')
    degree=np.bincount(g.message_dst,minlength=n)
    denominator=degree.clip(1)
    adjacency=sparse.coo_matrix((np.ones(m),(g.message_dst,g.message_src)),shape=(n,n)).tocsr()
    adjacency=sparse.diags(1/denominator)@adjacency
    total=np.zeros((n,d),dtype=np.float64)
    for a in range(0,m,chunk_size):
        b=min(a+chunk_size,m)
        np.add.at(total,g.message_dst[a:b],g.edge_x[g.message_flow[a:b]])
    mean=(total/denominator[:,None]).astype(np.float32)
    return InputSummary(adjacency.astype(np.float32),mean,(degree>0).astype(np.float32))


def query_nodes(g,q,depth):
    nodes=np.unique(np.r_[g.src[q],g.dst[q]]); frontier=nodes
    for _ in range(depth):
        arcs=g.incoming[frontier].indices
        expanded=np.unique(g.message_src[arcs])
        frontier=np.setdiff1d(expanded,nodes,assume_unique=True)
        nodes=np.union1d(nodes,expanded)
        if not len(frontier): break
    return nodes


def summary_query_nodes(g,summary,q,depth):
    """Exact closure using unique neighbor pairs, without rescanning multiarcs."""
    nodes=np.unique(np.r_[g.src[q],g.dst[q]]); frontier=nodes
    for _ in range(depth):
        expanded=np.unique(summary.adjacency[frontier].indices)
        frontier=np.setdiff1d(expanded,nodes,assume_unique=True)
        nodes=np.union1d(nodes,expanded)
        if not len(frontier): break
    return nodes


def fixed_inputs(g,summary,q,depth,device):
    """Reuse immutable graph inputs only; never store weights or hidden states.

    A summary belongs to one fixed snapshot. Build a new summary after changing
    graph features/context. Closure memoization counts array/key bytes, not total
    Python RAM. Only one packed tensor context is retained.
    """
    memo=summary._memo
    key=(id(g),depth,q.tobytes())
    nodes=memo.get('queries',{}).get(key) if summary.cache_enabled else None
    if nodes is None:
        nodes=summary_query_nodes(g,summary,q,depth)
        if summary.cache_enabled and summary.closure_cache_bytes>0:
            digest=hashlib.sha256(nodes.tobytes()).digest()
            intern=memo.get('sets',{}).get(digest)
            if intern is not None and not np.array_equal(intern,nodes): raise AssertionError('Closure digest collision')
            cost=len(key[2])+16+(nodes.nbytes+len(digest) if intern is None else 0)
            if memo.get('bytes',0)+cost<=summary.closure_cache_bytes:
                if intern is None: memo.setdefault('sets',{})[digest]=nodes
                else: nodes=intern
                memo.setdefault('queries',{})[key]=nodes; memo['bytes']=memo.get('bytes',0)+cost
    packed=summary._packed if summary.cache_enabled else None
    if packed is not None and packed[0]==id(g) and packed[1]==str(device) and np.array_equal(packed[2],nodes):
        return nodes,packed[3]
    # Cached constants must also be safe for later autograd use.
    with torch.inference_mode(False):
        matrix=summary.adjacency[nodes][:,nodes].tocoo()
        indices=torch.tensor(np.stack([matrix.row,matrix.col]),dtype=torch.long,device=device)
        adjacency=torch.sparse_coo_tensor(indices,torch.tensor(matrix.data,device=device),matrix.shape).coalesce()
        values=(adjacency,torch.tensor(g.node_x[nodes],device=device),torch.tensor(summary.mean_flow[nodes],device=device),
            torch.tensor(summary.has_incoming[nodes],device=device).unsqueeze(1))
    if summary.cache_enabled: summary._packed=(id(g),str(device),nodes,values)
    return nodes,values


def forward_summary(model,g,summary,q,ablation='full'):
    """Same K-hop node set/dropout draws; linear message terms are factored.

    Boundary-node layer states may differ from induced-halo states, but are
    outside the computation paths reaching the queries at the declared depth.
    Only graph inputs are summarized; no learned embeddings are cached.
    """
    if ablation not in {'full','no_degree','no_message_features','no_messages'}:
        raise ValueError('Unknown summary ablation')
    q=np.asarray(q,dtype=np.int64)
    if not len(q): raise ValueError('Empty query batch')
    device=next(model.parameters()).device
    nodes,(adjacency,h,features,mask)=fixed_inputs(g,summary,q,model.depth,device)
    if ablation=='no_degree': h=torch.zeros_like(h)
    if ablation=='no_message_features': features=torch.zeros_like(features)
    for msg,update in zip(model.messages,model.updates):
        width=h.shape[1]
        if ablation=='no_messages': neighbor=torch.zeros((len(nodes),msg.out_features),device=device)
        else:
            neighbor=F.linear(torch.sparse.mm(adjacency,h),msg.weight[:,:width])
            neighbor=neighbor+F.linear(features,msg.weight[:,width:])+mask*msg.bias
        h=F.dropout(F.relu(update(torch.cat([h,neighbor],dim=1))),p=model.dropout,training=model.training)
    src=torch.tensor(np.searchsorted(nodes,g.src[q]),device=device)
    dst=torch.tensor(np.searchsorted(nodes,g.dst[q]),device=device)
    query_x=torch.tensor(g.edge_x[q],device=device)
    return model.decoder(torch.cat([h[src],h[dst],query_x],dim=1))
