"""GraphSAGE-style edge-conditioned mean aggregation and explicit edge decoder."""
import torch
from torch import nn
from torch.nn import functional as F


class EdgeSAGE(nn.Module):
    def __init__(self,edge_dim,classes,hidden=64,layers=2,dropout=0.2):
        super().__init__()
        if layers<1:
            raise ValueError("At least one message-passing layer required")
        self.depth = layers
        self.dropout = dropout
        self.messages = nn.ModuleList()
        self.updates = nn.ModuleList()
        width = 2  # log1p(outgoing flow count), log1p(incoming flow count)
        for _ in range(layers):
            self.messages.append(nn.Linear(width+edge_dim,hidden))
            self.updates.append(nn.Linear(width+hidden,hidden))
            width = hidden
        self.decoder = nn.Sequential(nn.Linear(2*hidden+edge_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,classes))

    def forward(self,b):
        h = b.node_x
        for msg,update in zip(self.messages,self.updates):
            values = msg(torch.cat([h[b.msg_src],b.edge_x[b.msg_flow]],dim=1))
            total = torch.zeros((len(h),values.shape[1]),device=h.device,dtype=h.dtype)
            total.index_add_(0,b.msg_dst,values)
            degree = torch.bincount(b.msg_dst,minlength=len(h)).clamp_min(1).unsqueeze(1)
            neighbor = total / degree
            h = F.dropout(F.relu(update(torch.cat([h,neighbor],dim=1))),p=self.dropout,training=self.training)
        return self.decoder(torch.cat([h[b.query_src],h[b.query_dst],b.query_x],dim=1))
