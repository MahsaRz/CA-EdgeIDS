"""Paired graph controls with fixed targets and validation-only checkpointing."""
import copy
from dataclasses import replace

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .metrics import evaluate
from .model import EdgeSAGE
from .runtime import seed_all
from .statistics_backend import forward_summary,summarize

VARIANTS=['full','no_degree','no_message_features','no_messages','rewired','directed']


def graph_variant(g,variant,seed):
    if variant not in VARIANTS: raise ValueError('Unknown graph variant')
    if variant not in {'rewired','directed'}: return g
    if variant=='rewired':
        src=g.message_src.copy(); dst=np.random.default_rng(seed).permutation(g.message_dst); flow=g.message_flow.copy()
        for values,original in [(src,g.message_src),(dst,g.message_dst)]:
            if not np.array_equal(np.bincount(values,minlength=len(g.hosts)),np.bincount(original,minlength=len(g.hosts))):
                raise AssertionError('Context rewiring changed degree counts')
    else: src=g.src.copy(); dst=g.dst.copy(); flow=np.arange(len(g.src),dtype=np.int64)
    arcs=np.arange(len(src))
    incoming=sparse.csr_matrix((arcs,(dst,arcs)),shape=(len(g.hosts),len(src)))
    return replace(g,incoming=incoming,message_src=src,message_dst=dst,message_flow=flow)


def predict(model,g,summary,variant,batch_size=2048):
    model.eval(); output=[]
    flag=variant if variant in {'no_degree','no_message_features','no_messages'} else 'full'
    with torch.inference_mode():
        for a in range(0,len(g.src),batch_size):
            q=np.arange(a,min(a+batch_size,len(g.src)))
            output.append(forward_summary(model,g,summary,q,flag).softmax(1).numpy())
    return np.concatenate(output)


def fit(g,val_graph,summary,val_summary,batches,cfg,seed,variant):
    seed_all(seed,cfg.get('threads',1))
    counts=np.bincount(g.y,minlength=cfg['classes'])
    if np.any(counts==0) or any(not np.any(val_graph.y==i) for i in range(cfg['classes'])):
        raise ValueError('Missing training/validation class')
    model=EdgeSAGE(g.edge_x.shape[1],cfg['classes'],cfg['hidden'],cfg['layers'],cfg['dropout'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    weights=torch.tensor(len(g.y)/(len(counts)*counts),dtype=torch.float32)
    rng=np.random.default_rng(seed); best=-1.; stale=0; state=None; history=[]
    flag=variant if variant in {'no_degree','no_message_features','no_messages'} else 'full'
    for epoch in range(cfg['epochs']):
        order=list(batches); rng.shuffle(order); model.train(); total=0.
        for step,q in enumerate(order,1):
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(forward_summary(model,g,summary,q,flag),torch.tensor(g.y[q]),weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); total+=float(loss.item())*len(q)
            if cfg.get('progress_every',0)>0 and step%cfg['progress_every']==0:
                print(f'{variant} seed {seed} epoch {epoch+1}: {step}/{len(order)} updates complete',flush=True)
        score=evaluate(val_graph.y,predict(model,val_graph,val_summary,variant),cfg['label_names'])['macro_f1']
        history.append(dict(epoch=epoch+1,train_loss=total/len(g.src),validation_macro_f1=score))
        if score>best:
            best=score; best_epoch=epoch+1; stale=0; state=copy.deepcopy(model.state_dict())
        else: stale+=1
        print(f'{variant} seed {seed} epoch {epoch+1}: validation macro F1 {score:.5f}',flush=True)
        if stale>=cfg['patience']: break
    model.load_state_dict(state); model.eval()
    return model,history,dict(validation_macro_f1=best,best_epoch=best_epoch,epochs_run=len(history),
        gradient_updates=len(history)*len(batches),parameters=sum(p.numel() for p in model.parameters()),class_weights=weights.tolist())
