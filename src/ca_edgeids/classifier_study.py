"""Validation-only selection helpers for the fixed binary development study."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import evaluate
from .runtime import seed_all


class FlowMLP(nn.Module):
    def __init__(self,width,hidden=16,classes=2,dropout=.2):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,classes))

    def forward(self,x): return self.network(x)


def predict_mlp(model,x,batch_size=2048):
    if batch_size<1: raise ValueError('Batch size must be positive')
    model.eval(); rows=[]
    with torch.inference_mode():
        for a in range(0,len(x),batch_size):
            rows.append(model(torch.tensor(x[a:a+batch_size],dtype=torch.float32)).softmax(1).numpy())
    return np.concatenate(rows)


def select_candidate(records):
    """Only declared validation scores determine selection; first wins exact ties."""
    if not records or len({r['candidate_id'] for r in records})!=len(records):
        raise ValueError('Need unique candidates')
    if any(not np.isfinite(r['validation_macro_f1']) for r in records):
        raise ValueError('Invalid validation score')
    if len({r['validation_row_ids_sha256'] for r in records})!=1:
        raise ValueError('Validation populations differ')
    return max(records,key=lambda r:r['validation_macro_f1'])['candidate_id']


def fit_mlp(x,y,val_x,val_y,batches,config,seed):
    seed_all(seed,config.get('threads',1))
    classes=config['classes']; counts=np.bincount(y,minlength=classes)
    if np.any(counts==0) or any(not np.any(val_y==i) for i in range(classes)):
        raise ValueError('Training and validation must contain every class')
    model=FlowMLP(x.shape[1],config['hidden'],classes,config['dropout'])
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    weights=torch.tensor(len(y)/(classes*counts),dtype=torch.float32)
    rng=np.random.default_rng(seed); best=-1.; stale=0; state=None; history=[]; updates=0
    for epoch in range(config['epochs']):
        order=list(batches); rng.shuffle(order); model.train(); loss_total=0.
        for q in order:
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(torch.tensor(x[q],dtype=torch.float32)),torch.tensor(y[q],dtype=torch.long),weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step(); updates+=1; loss_total+=float(loss.item())*len(q)
        score=evaluate(val_y,predict_mlp(model,val_x),config['label_names'])['macro_f1']
        history.append(dict(epoch=epoch+1,train_loss=loss_total/len(y),validation_macro_f1=score))
        if score>best:
            best=score; best_epoch=epoch+1; stale=0; state=copy.deepcopy(model.state_dict())
        else: stale+=1
        print(f'MLP epoch {epoch+1}: validation macro F1 {score:.5f}',flush=True)
        if stale>=config['patience']: break
    model.load_state_dict(state); model.eval()
    return model,history,dict(best_epoch=best_epoch,epochs_run=len(history),validation_macro_f1=best,
        gradient_updates=updates,class_weights=weights.tolist(),parameters=sum(p.numel() for p in model.parameters()))
