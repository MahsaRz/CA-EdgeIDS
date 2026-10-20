import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ca_edgeids.cli import demo_config
from ca_edgeids.data import prepare
from ca_edgeids.graph import build_graph,halo_batch
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import seed_all
from ca_edgeids.statistics_backend import forward_summary,summarize


@pytest.mark.parametrize('depth',[1,2,3])
@pytest.mark.parametrize('bidirectional',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_summary_matches_query_outputs_and_parameter_gradients(tmp_path,depth,bidirectional,training):
    seed_all(13); data=prepare(demo_config(tmp_path)); g=build_graph(data,'test',bidirectional)
    a=EdgeSAGE(g.edge_x.shape[1],2,8,depth,.2); b=copy.deepcopy(a)
    a.train(training); b.train(training)
    q=np.arange(0,min(31,len(g.src)),2); summary=summarize(g)
    torch.manual_seed(29); expected=a(halo_batch(g,q,depth))
    torch.manual_seed(29); actual=forward_summary(b,g,summary,q)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    F.cross_entropy(expected,torch.tensor(g.y[q])).backward()
    F.cross_entropy(actual,torch.tensor(g.y[q])).backward()
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-5)
    g.y=None
    torch.manual_seed(29); independent=forward_summary(b,g,summary,q)
    torch.testing.assert_close(independent,actual,atol=0,rtol=0)


def test_summary_limits():
    from test_resources import star
    with pytest.raises(MemoryError): summarize(star(),max_bytes=1)


def test_summary_recomputes_learned_values_after_weight_changes():
    from test_resources import star
    g=star(); summary=summarize(g); seed_all(8)
    model=EdgeSAGE(1,2,8,2,0).eval(); q=np.array([0,2])
    a=forward_summary(model,g,summary,q)
    with torch.no_grad(): model.messages[0].bias.add_(.3)
    b=forward_summary(model,g,summary,q)
    expected=model(halo_batch(g,q,2))
    assert not torch.allclose(a,b)
    torch.testing.assert_close(b,expected,atol=1e-5,rtol=1e-5)


def test_graph_controls_preserve_targets_and_rewiring_degrees():
    from test_resources import star
    from ca_edgeids.controlled_training import graph_variant
    g=star()
    for variant in ['rewired','directed']:
        changed=graph_variant(g,variant,77)
        for name in ['src','dst','edge_x','node_x','y','row_ids','hosts']:
            np.testing.assert_array_equal(getattr(g,name),getattr(changed,name))
    a=graph_variant(g,'rewired',77); b=graph_variant(g,'rewired',77)
    np.testing.assert_array_equal(a.message_dst,b.message_dst)
    np.testing.assert_array_equal(np.bincount(a.message_dst,minlength=4),np.bincount(g.message_dst,minlength=4))
def test_compressed_neighbor_closure_is_identical_with_repeated_flows():
    from test_resources import star
    from ca_edgeids.statistics_backend import query_nodes,summary_query_nodes,summarize
    from scipy import sparse
    import numpy as np
    g=star()
    g.message_src=np.tile(g.message_src,31); g.message_dst=np.tile(g.message_dst,31); g.message_flow=np.tile(g.message_flow,31)
    g.incoming=sparse.csr_matrix((np.arange(len(g.message_src)),(g.message_dst,np.arange(len(g.message_src)))),shape=(4,len(g.message_src)))
    summary=summarize(g)
    for depth in [1,2,3]:
        for q in [np.array([0]),np.array([1,2]),np.arange(3)]:
            np.testing.assert_array_equal(query_nodes(g,q,depth),summary_query_nodes(g,summary,q,depth))


def test_compressed_closure_preserves_fitted_weights_exactly(monkeypatch):
    from test_resources import star
    from ca_edgeids.controlled_training import fit
    import ca_edgeids.statistics_backend as backend
    import numpy as np
    import torch
    g=star(); g.y=np.array([0,1,0]); summary=backend.summarize(g)
    cfg=dict(classes=2,hidden=4,layers=2,dropout=.2,learning_rate=.003,weight_decay=.0001,epochs=3,patience=3,label_names=['Benign','Attack'],threads=1)
    batches=[np.array([0,1]),np.array([2])]
    new,history,_=fit(g,g,summary,summary,batches,cfg,7,'full')
    monkeypatch.setattr(backend,'summary_query_nodes',lambda graph,summary,q,depth:backend.query_nodes(graph,q,depth))
    old,old_history,_=fit(g,g,summary,summary,batches,cfg,7,'full')
    assert history==old_history
    for key,value in new.state_dict().items(): assert torch.equal(value,old.state_dict()[key])


def test_input_tensor_reuse_preserves_complete_fits_bitwise():
    from test_resources import star
    from ca_edgeids.controlled_training import fit
    from ca_edgeids.statistics_backend import summarize
    import numpy as np
    import torch
    g=star(); g.y=np.array([0,1,0])
    cfg=dict(classes=2,hidden=4,layers=2,dropout=.2,learning_rate=.003,weight_decay=.0001,epochs=4,patience=3,label_names=['Benign','Attack'],threads=1)
    batches=[np.array([0,1]),np.array([2])]
    cached=summarize(g); plain=summarize(g); plain.cache_enabled=False
    new,history,_=fit(g,g,cached,cached,batches,cfg,7,'full')
    old,old_history,_=fit(g,g,plain,plain,batches,cfg,7,'full')
    assert history==old_history
    for key,value in new.state_dict().items(): assert torch.equal(value,old.state_dict()[key])


def test_input_cache_is_bounded_and_contains_only_fixed_inputs():
    from test_resources import star
    from ca_edgeids.statistics_backend import fixed_inputs,summarize
    import numpy as np
    import torch
    g=star(); summary=summarize(g)
    _,a=fixed_inputs(g,summary,np.array([0]),2,'cpu')
    _,b=fixed_inputs(g,summary,np.array([1]),2,'cpu')
    assert a is b  # Same node closure across different flow batches.
    assert all(not t.requires_grad for t in a)
    assert summary._memo['bytes']<=summary.closure_cache_bytes
    tiny=summarize(g); tiny.closure_cache_bytes=1
    fixed_inputs(g,tiny,np.arange(3),2,'cpu')
    assert not tiny._memo


def test_inference_warmed_summary_preserves_later_training_gradients():
    from test_resources import star
    g=star(); seed_all(17)
    actual=EdgeSAGE(g.edge_x.shape[1],2,4,2,.2)
    expected=copy.deepcopy(actual); summary=summarize(g); q=np.arange(3)
    actual.eval()
    with torch.inference_mode(): forward_summary(actual,g,summary,q)
    assert all(not torch.is_inference(t) for t in summary._packed[3])
    actual.train(); expected.train()
    torch.manual_seed(29); a=forward_summary(actual,g,summary,q)
    torch.manual_seed(29); b=forward_summary(expected,g,summarize(g),q)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    labels=torch.tensor([0,1,0])
    F.cross_entropy(a,labels).backward(); F.cross_entropy(b,labels).backward()
    for p,r in zip(actual.parameters(),expected.parameters()):
        torch.testing.assert_close(p.grad,r.grad,rtol=0,atol=0)
