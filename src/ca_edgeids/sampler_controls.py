"""Explicit edge-label adaptations of cluster and normalized node sampling."""
from dataclasses import dataclass,replace
import copy
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .graph import partition_nodes
from .statistics_backend import InputSummary,forward_summary,summarize
from .controlled_training import predict
from .metrics import evaluate
from .model import EdgeSAGE
from .runtime import seed_all


def node_probabilities(n,b,src,dst):
    if not 1<=b<=n or (b==1 and np.any(src!=dst)): raise ValueError('Sampling must give every target positive probability')
    node=b/n
    pair=b*(b-1)/(n*(n-1)) if n>1 else 1.
    return node,np.where(src==dst,node,pair)


def cluster_probabilities(parts,selected,owner,src,dst):
    if not 2<=selected<=parts: raise ValueError('At least two clusters required to include cut targets')
    same=selected/parts; cross=selected*(selected-1)/(parts*(parts-1))
    return np.where(owner[src]==owner[dst],same,cross)


@dataclass
class PairInputs:
    src: np.ndarray
    dst: np.ndarray
    counts: np.ndarray
    flow_sums: np.ndarray
    full_degree: np.ndarray


def pair_inputs(g,max_bytes=256*1024**2):
    n=len(g.node_x); m=len(g.message_src); d=g.edge_x.shape[1]
    unique,inverse=np.unique(g.message_dst*n+g.message_src,return_inverse=True)
    estimate=len(unique)*d*8+m*16+4096*d*4
    if estimate>max_bytes: raise MemoryError('Pair-input storage estimate exceeds the configured limit')
    sums=np.zeros((len(unique),d),dtype=np.float64)
    for a in range(0,m,4096): np.add.at(sums,inverse[a:a+4096],g.edge_x[g.message_flow[a:a+4096]])
    return PairInputs(unique%n,unique//n,np.bincount(inverse),sums,np.bincount(g.message_dst,minlength=n))


def induced_inputs(g,pairs,nodes,normalization):
    n=len(g.node_x); selected=np.zeros(n,dtype=bool); selected[nodes]=True
    keep=selected[pairs.src]&selected[pairs.dst]
    src=pairs.src[keep]; dst=pairs.dst[keep]; count=pairs.counts[keep]
    if normalization=='local':
        degree=np.bincount(dst,weights=count,minlength=n).clip(1)
        coefficient=1/degree[dst]
    elif normalization=='saint':
        p_node,p_edge=node_probabilities(n,len(nodes),src,dst)
        coefficient=p_node/p_edge/pairs.full_degree[dst].clip(1)
    else: raise ValueError('Unknown sampler normalization')
    adjacency=sparse.coo_matrix((count*coefficient,(dst,src)),shape=(n,n)).tocsr().astype(np.float32)
    features=np.zeros((n,g.edge_x.shape[1]),dtype=np.float64)
    np.add.at(features,dst,pairs.flow_sums[keep]*coefficient[:,None])
    bias=np.bincount(dst,weights=count*coefficient,minlength=n).astype(np.float32)
    arcs=selected[g.message_src]&selected[g.message_dst]
    ms=g.message_src[arcs]; md=g.message_dst[arcs]; mf=g.message_flow[arcs]
    incoming=sparse.csr_matrix((np.arange(len(ms)),(md,np.arange(len(ms)))),shape=(n,len(ms)))
    sub=replace(g,incoming=incoming,message_src=ms,message_dst=md,message_flow=mf)
    targets=np.flatnonzero(selected[g.src]&selected[g.dst])
    return sub,InputSummary(adjacency,features.astype(np.float32),bias),targets


def edge_risk(loss,target_weights,q,eligible,pi,total_targets):
    """HT risk for fixed per-edge losses, including uniform target subsampling."""
    if len(q)==0: raise ValueError('Handle empty samples as zero-gradient updates')
    weights=torch.as_tensor(target_weights,dtype=loss.dtype,device=loss.device)
    probability=torch.as_tensor(pi[q],dtype=loss.dtype,device=loss.device)
    return (loss*weights/probability).sum()*(len(eligible)/(len(q)*total_targets))


class SubgraphSampler:
    def __init__(self,g,method,parts=4,clusters_per_batch=2,node_fraction=.25,ownership_seed=7):
        if method not in {'cluster_gcn_edge','graphsaint_node_edge'}: raise ValueError('Unknown sampling control')
        self.g=g; self.method=method; self.pairs=pair_inputs(g); self.cache={}
        self.parts=parts; self.clusters_per_batch=clusters_per_batch
        if method=='cluster_gcn_edge':
            self.owner=partition_nodes(g,'metis',parts,ownership_seed)
            self.pi=cluster_probabilities(parts,clusters_per_batch,self.owner,g.src,g.dst)
        else:
            self.b=min(len(g.node_x),max(2,int(np.ceil(len(g.node_x)*node_fraction))))
            _,self.pi=node_probabilities(len(g.node_x),self.b,g.src,g.dst)

    def sample(self,rng,batch_size):
        if self.method=='cluster_gcn_edge':
            key=tuple(sorted(rng.choice(self.parts,self.clusters_per_batch,replace=False).tolist()))
            if key not in self.cache:
                self.cache[key]=induced_inputs(self.g,self.pairs,np.flatnonzero(np.isin(self.owner,key)),'local')
            graph,summary,eligible=self.cache[key]
        else:
            nodes=np.sort(rng.choice(len(self.g.node_x),self.b,replace=False))
            graph,summary,eligible=induced_inputs(self.g,self.pairs,nodes,'saint')
        q=rng.choice(eligible,min(batch_size,len(eligible)),replace=False) if len(eligible) else np.array([],dtype=np.int64)
        return graph,summary,q,eligible,self.pi


METHODS=['unpartitioned','random_exact','metis_exact','cluster_gcn_edge','graphsaint_node_edge']


def fit_sampler(g,val_graph,cfg,seed,method):
    if method not in METHODS: raise ValueError('Unknown training control')
    seed_all(seed,cfg.get('threads',1)); start=time.perf_counter()
    counts=np.bincount(g.y,minlength=cfg['classes'])
    if np.any(counts==0) or any(not np.any(val_graph.y==i) for i in range(cfg['classes'])): raise ValueError('Missing class')
    weight=torch.tensor(len(g.y)/(len(counts)*counts),dtype=torch.float32)
    model=EdgeSAGE(g.edge_x.shape[1],cfg['classes'],cfg['hidden'],cfg['layers'],cfg['dropout'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    validation=summarize(val_graph)
    sampled=method in {'cluster_gcn_edge','graphsaint_node_edge'}
    sampler=SubgraphSampler(g,method,parts=4,clusters_per_batch=2,node_fraction=.25,ownership_seed=7) if sampled else None
    if not sampled:
        owner=partition_nodes(g,'full' if method=='unpartitioned' else 'random' if method=='random_exact' else 'metis',1 if method=='unpartitioned' else 4,7)
        ordered=np.argsort(owner[g.src],kind='stable')
        batches=[ordered[a:a+cfg['max_edges_per_batch']] for a in range(0,len(ordered),cfg['max_edges_per_batch'])]
        full=summarize(g)
    steps=int(np.ceil(len(g.src)/cfg['max_edges_per_batch'])); prep=time.perf_counter()-start
    rng=np.random.default_rng(seed); coverage=np.zeros(len(g.src),dtype=np.int64)
    best=-1.; stale=0; state=None; history=[]; updates=0; empty=0; exposures=0
    for epoch in range(cfg['epochs']):
        if not sampled: order=list(batches); rng.shuffle(order)
        model.train(); loss_total=0.
        for step in range(steps):
            if sampled: graph,summary,q,eligible,pi=sampler.sample(rng,cfg['max_edges_per_batch'])
            else: graph=g; summary=full; q=order[step]
            optimizer.zero_grad(set_to_none=True)
            if len(q):
                logits=forward_summary(model,graph,summary,q)
                losses=F.cross_entropy(logits,torch.tensor(g.y[q]),reduction='none')
                if sampled: loss=edge_risk(losses,weight[g.y[q]],q,eligible,pi,len(g.src))
                else: loss=(losses*weight[g.y[q]]).sum()*(steps/len(g.src))
                coverage[q]+=1; exposures+=len(q)
            else:
                loss=sum(p.sum()*0 for p in model.parameters()); empty+=1
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite sampled training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); updates+=1; loss_total+=float(loss.item())
        score=evaluate(val_graph.y,predict(model,val_graph,validation,'full'),cfg['label_names'])['macro_f1']
        history.append(dict(epoch=epoch+1,mean_update_loss=loss_total/steps,validation_macro_f1=score))
        if score>best:
            best=score; best_epoch=epoch+1; stale=0; state=copy.deepcopy(model.state_dict())
        else: stale+=1
        print(f'{method} seed {seed} epoch {epoch+1}: validation macro F1 {score:.5f}',flush=True)
        if stale>=cfg['patience']: break
    model.load_state_dict(state); model.eval()
    summary=dict(validation_macro_f1=best,best_epoch=best_epoch,epochs_run=len(history),gradient_updates=updates,
        target_exposures=exposures,unique_training_targets_seen=int(np.sum(coverage>0)),unseen_training_targets=int(np.sum(coverage==0)),
        min_target_exposures=int(coverage.min()),max_target_exposures=int(coverage.max()),empty_samples=empty,
        sampler_and_validation_preparation_s=prep,full_graph_targets=len(g.src),
        class_targets_seen={cfg['label_names'][i]:int(np.sum((coverage>0)&(g.y==i))) for i in range(cfg['classes'])},
        objective='global class-balanced risk; fixed per-update scaling; sampling uses analytic edge inclusion and target-subsample corrections')
    return model,history,summary
