"""Layer-wise snapshot inference with one message traversal per layer.

Node states stay resident; message tensors and edge decoding are chunked. The
cache exists only within one call, so another snapshot/model is always recomputed.
"""
import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import PeakRSS, sync


def infer_cached(model,g,config,message_batch_size=4096):
    device=config['device']; query_size=config['max_edges_per_batch']
    if message_batch_size<1 or query_size<1:
        raise ValueError('Chunk sizes must be positive')
    n=len(g.node_x); m=len(g.message_src); targets=len(g.src)
    if not n or not targets: raise ValueError('Empty graph')
    hidden=model.updates[-1].out_features
    # Planned tensor-storage screening estimate, not an allocation/RAM bound.
    estimate=(n*(2+12*hidden)*4+n*8
              +min(m,message_batch_size)*(g.edge_x.shape[1]+3*hidden)*4
              +min(m,message_batch_size)*24
              +min(targets,query_size)*(g.edge_x.shape[1]+4*hidden+config['classes'])*4)
    limit=config.get('max_batch_tensor_bytes',256*1024**2)
    if limit<=0: raise ValueError('Tensor limit must be positive')
    if estimate>limit:
        raise MemoryError(f'Configured cached tensor limit exceeded: estimated {estimate} bytes > {limit}. Node states must remain resident.')
    model.eval()
    probabilities=np.empty((targets,config['classes']),dtype=np.float32)
    seen=np.zeros(targets,dtype=np.int64)
    tensor=lambda a,dtype:torch.tensor(a,dtype=dtype,device=device)
    if str(device).startswith('cuda'):
        sync(device); torch.cuda.reset_peak_memory_stats(device)
    sync(device); start=time.perf_counter()
    packing_s=0.; message_s=0.; decoder_s=0.; message_chunks=0; query_chunks=0
    with PeakRSS() as mem,torch.inference_mode():
        t=time.perf_counter()
        h=tensor(g.node_x,torch.float32)
        degree=tensor(np.bincount(g.message_dst,minlength=n).clip(1),torch.float32).unsqueeze(1)
        sync(device); packing_s+=time.perf_counter()-t
        for msg,update in zip(model.messages,model.updates):
            t=time.perf_counter()
            total=torch.zeros((n,msg.out_features),device=device,dtype=h.dtype)
            sync(device); message_s+=time.perf_counter()-t
            for a in range(0,m,message_batch_size):
                b=min(a+message_batch_size,m); t=time.perf_counter()
                src=tensor(g.message_src[a:b],torch.long)
                dst=tensor(g.message_dst[a:b],torch.long)
                features=tensor(g.edge_x[g.message_flow[a:b]],torch.float32)
                sync(device); packing_s+=time.perf_counter()-t
                t=time.perf_counter()
                values=msg(torch.cat([h[src],features],dim=1))
                total.index_add_(0,dst,values)
                sync(device); message_s+=time.perf_counter()-t
                message_chunks+=1
                del src,dst,features,values
            t=time.perf_counter()
            h=F.relu(update(torch.cat([h,total/degree],dim=1)))
            sync(device); message_s+=time.perf_counter()-t
            del total
        for a in range(0,targets,query_size):
            b=min(a+query_size,targets); t=time.perf_counter()
            src=tensor(g.src[a:b],torch.long); dst=tensor(g.dst[a:b],torch.long)
            features=tensor(g.edge_x[a:b],torch.float32)
            sync(device); packing_s+=time.perf_counter()-t
            t=time.perf_counter()
            logits=model.decoder(torch.cat([h[src],h[dst],features],dim=1))
            sync(device); decoder_s+=time.perf_counter()-t
            probabilities[a:b]=logits.softmax(1).cpu().numpy(); seen[a:b]+=1
            query_chunks+=1
            del src,dst,features,logits
    sync(device); elapsed=time.perf_counter()-start
    if not np.all(seen==1): raise AssertionError('Missing or duplicated flow predictions')
    return probabilities,dict(total_inference_s=elapsed,model_forward_s=message_s+decoder_s,
        halo_and_transfer_s=packing_s,message_passing_s=message_s,decoder_s=decoder_s,
        message_arc_visits=m*model.depth,message_chunks=message_chunks,batches=query_chunks,
        node_state_rows=n,estimated_working_tensor_bytes=estimate,
        cache_scope='one complete prediction; no cache reused between calls',
        peak_cpu_rss_mb_sampled=mem.peak/2**20,baseline_cpu_rss_mb=mem.baseline/2**20,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith('cuda') else None,
        rss_sampling_interval_ms=5,coverage_min=int(seen.min()),coverage_max=int(seen.max()),
        target_row_ids_sha256=hashlib.sha256(g.row_ids.astype('<i8').tobytes()).hexdigest())
