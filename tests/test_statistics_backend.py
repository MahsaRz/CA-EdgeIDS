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
