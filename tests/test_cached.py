import numpy as np
import pytest
import torch

from ca_edgeids.cached import infer_cached
from ca_edgeids.cli import demo_config
from ca_edgeids.data import prepare
from ca_edgeids.graph import build_graph,halo_batch
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import seed_all


@pytest.mark.parametrize('depth',[1,2,3])
@pytest.mark.parametrize('bidirectional',[False,True])
def test_cached_matches_full_and_recomputes_each_call(tmp_path,depth,bidirectional):
    cfg=demo_config(tmp_path); p=prepare(cfg); seed_all(123)
    g=build_graph(p,'test',bidirectional)
    cfg.update(classes=2,max_edges_per_batch=17)
    model=EdgeSAGE(g.edge_x.shape[1],2,hidden=8,layers=depth,dropout=.4).eval()
    def full():
        with torch.inference_mode(): return model(halo_batch(g,np.arange(len(g.src)),depth)).softmax(1).numpy()
    expected=full()
    for chunk in [1,31,len(g.message_src)+1]:
        actual,t=infer_cached(model,g,cfg,chunk)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        assert t['message_arc_visits']==depth*len(g.message_src)
        assert t['coverage_min']==t['coverage_max']==1
        if chunk==31: reference=actual.copy()
    labels=g.y
    g.y=None  # Inference must not read labels at all.
    np.testing.assert_array_equal(infer_cached(model,g,cfg,31)[0],reference)
    g.y=labels
    g.edge_x=g.edge_x+1
    changed,_=infer_cached(model,g,cfg,31)
    np.testing.assert_allclose(changed,full(),atol=2e-6,rtol=2e-6)
    assert not np.allclose(changed,expected)
    with torch.no_grad(): model.decoder[-1].bias[0]+=1
    np.testing.assert_allclose(infer_cached(model,g,cfg,31)[0],full(),atol=2e-6,rtol=2e-6)


def test_cached_limits_stop_before_model_work(tmp_path):
    cfg=demo_config(tmp_path); p=prepare(cfg); g=build_graph(p,'test')
    cfg.update(classes=2); model=EdgeSAGE(g.edge_x.shape[1],2,8)
    with pytest.raises(ValueError,match='positive'): infer_cached(model,g,cfg,0)
    with pytest.raises(MemoryError,match='tensor limit exceeded'):
        infer_cached(model,g,dict(cfg,max_batch_tensor_bytes=1))


def test_cached_handles_self_loops_parallel_flows_and_isolated_node():
    from scipy import sparse
    from ca_edgeids.graph import FlowGraph
    seed_all(12)
    src=np.array([0,0,0,1,2]); dst=np.array([0,1,1,0,2]); ids=np.arange(5)
    g=FlowGraph(src,dst,np.arange(10,dtype=np.float32).reshape(5,2)/10,
        np.ones((4,2),dtype=np.float32),np.zeros(5,dtype=np.int64),ids,
        np.array(['a','b','c','isolated']),
        sparse.csr_matrix((ids,(dst,ids)),shape=(4,5)),src,dst,ids)
    model=EdgeSAGE(2,2,8,3,.5).eval()
    cfg=dict(device='cpu',classes=2,max_edges_per_batch=2)
    with torch.inference_mode(): expected=model(halo_batch(g,ids,3)).softmax(1).numpy()
    visits=[0,0,0]
    def record(layer):
        def hook(module,args): visits[layer]+=len(args[0])
        return hook
    handles=[module.register_forward_pre_hook(record(i)) for i,module in enumerate(model.messages)]
    actual,_=infer_cached(model,g,cfg,2)
    for handle in handles: handle.remove()
    assert visits==[5,5,5]
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
