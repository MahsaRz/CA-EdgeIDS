from itertools import combinations

import numpy as np
import pytest
import torch

from ca_edgeids.sampler_controls import cluster_probabilities,edge_risk,induced_inputs,node_probabilities,pair_inputs
from ca_edgeids.statistics_backend import summarize


def test_uniform_node_probability_normalization_including_self_loops():
    from test_resources import star
    from scipy import sparse
    g=star(); g.message_src=np.r_[g.message_src,0]; g.message_dst=np.r_[g.message_dst,0]; g.message_flow=np.r_[g.message_flow,0]
    g.incoming=sparse.csr_matrix((np.arange(7),(g.message_dst,np.arange(7))),shape=(4,7))
    pairs=pair_inputs(g); complete=summarize(g)
    node_hits=np.zeros(4); averaged=np.zeros((4,4)); features=np.zeros_like(complete.mean_flow); bias=np.zeros(4)
    for subset in combinations(range(4),2):
        nodes=np.array(subset); _,summary,_=induced_inputs(g,pairs,nodes,'saint')
        averaged+=summary.adjacency.toarray(); features+=summary.mean_flow; bias+=summary.has_incoming; node_hits[nodes]+=1
    np.testing.assert_allclose(averaged/node_hits[:,None],complete.adjacency.toarray(),atol=1e-7)
    np.testing.assert_allclose(features/node_hits[:,None],complete.mean_flow,atol=1e-7)
    np.testing.assert_allclose(bias/node_hits,complete.has_incoming,atol=1e-7)
    _,pi=node_probabilities(4,2,np.array([0,0]),np.array([0,1]))
    np.testing.assert_allclose(pi,[.5,1/6])


def test_edge_loss_estimator_matches_full_risk_with_target_subsampling():
    src=np.array([0,0,1,2]); dst=np.array([0,1,2,3]); values=np.array([1.,2.,4.,7.]); weights=np.array([2.,1.,3.,1.])
    _,pi=node_probabilities(4,2,src,dst); estimates=[]
    for nodes in combinations(range(4),2):
        eligible=np.flatnonzero(np.isin(src,nodes)&np.isin(dst,nodes))
        if not len(eligible): estimates.append(0.); continue
        # Enumerate the uniform one-target subsample conditional on these nodes.
        estimates.append(np.mean([edge_risk(torch.tensor(values[[q]]),weights[[q]],np.array([q]),eligible,pi,len(src)).item() for q in eligible]))
    assert np.mean(estimates)==pytest.approx(np.mean(values*weights))


def test_cluster_probability_includes_between_cluster_targets():
    owner=np.array([0,0,1,2]); src=np.array([0,0,2]); dst=np.array([1,2,3])
    pi=cluster_probabilities(4,2,owner,src,dst)
    np.testing.assert_allclose(pi,[.5,1/6,1/6])
    observed=np.mean([np.isin(owner[src],c)&np.isin(owner[dst],c) for c in combinations(range(4),2)],axis=0)
    np.testing.assert_allclose(pi,observed)


def test_pair_inputs_reproduce_full_summary_and_do_not_read_labels():
    from test_resources import star
    g=star(); g.y=None
    pairs=pair_inputs(g); expected=summarize(g)
    sub,actual,targets=induced_inputs(g,pairs,np.arange(4),'local')
    np.testing.assert_allclose(actual.adjacency.toarray(),expected.adjacency.toarray())
    np.testing.assert_allclose(actual.mean_flow,expected.mean_flow)
    np.testing.assert_allclose(actual.has_incoming,expected.has_incoming)
    np.testing.assert_array_equal(targets,np.arange(3))


def test_empty_sample_is_recorded_as_a_zero_gradient_update(monkeypatch):
    from test_resources import star
    from ca_edgeids.sampler_controls import SubgraphSampler,fit_sampler
    g=star(); g.y=np.array([0,1,0]); summary=summarize(g)
    def empty(self,rng,batch_size): return g,summary,np.array([],dtype=np.int64),np.array([],dtype=np.int64),np.ones(3)
    monkeypatch.setattr(SubgraphSampler,'sample',empty)
    cfg=dict(classes=2,hidden=4,layers=2,dropout=0,learning_rate=.01,weight_decay=0,epochs=2,patience=1,
             max_edges_per_batch=2,label_names=['Benign','Attack'],threads=1)
    _,_,result=fit_sampler(g,g,cfg,7,'graphsaint_node_edge')
    assert result['gradient_updates']==result['empty_samples']==4
    assert result['unseen_training_targets']==3 and result['target_exposures']==0
