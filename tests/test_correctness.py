import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from ca_edgeids.cli import demo_config
from ca_edgeids.data import prepare,split_rows,normalize_labels
from ca_edgeids.graph import build_graph,partition_nodes,query_batches,halo_batch
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.metrics import evaluate
from ca_edgeids.experiment import infer
from ca_edgeids.runtime import seed_all,current_rss


@pytest.fixture
def dataset(tmp_path):
    cfg=demo_config(tmp_path)
    return cfg,prepare(cfg)


def test_labels_and_no_target_features(dataset):
    cfg,p=dataset
    assert p.names==["Benign","Attack"]
    assert not set(["Label","Attack","IPV4_SRC_ADDR"]) & set(cfg["features"])
    bad=copy.deepcopy(cfg); bad["features"].append("Label")
    with pytest.raises(ValueError,match="Forbidden"): prepare(bad)
    f=p.frame.copy(); f.loc[0,"Label"]=1-int(f.loc[0,"Label"])
    with pytest.raises(ValueError,match="disagree"): normalize_labels(f,"bot","binary")


def test_multiclass_uses_attack_names(dataset):
    _,p=dataset
    y,names,_=normalize_labels(p.frame,"bot","multiclass")
    assert names.index("DDoS") in y
    assert np.all(y[p.frame.Attack=="DDoS"]==names.index("DDoS"))
    f=p.frame.copy(); f["Attack"]=f.Attack.replace({"DDoS":"ddos"})
    _,_,original=normalize_labels(f,"bot","multiclass")
    assert "ddos" in original


def test_split_pair_isolation_and_reverse_direction(dataset):
    cfg,p=dataset
    pairs=[tuple(sorted((a,b))) for a,b in zip(p.frame.IPV4_SRC_ADDR,p.frame.IPV4_DST_ADDR)]
    groups={}
    for key,part in zip(pairs,p.split): groups.setdefault(key,set()).add(part)
    assert all(len(v)==1 for v in groups.values())
    reversed_frame=p.frame.rename(columns={"IPV4_SRC_ADDR":"IPV4_DST_ADDR","IPV4_DST_ADDR":"IPV4_SRC_ADDR"})
    np.testing.assert_array_equal(p.split,split_rows(reversed_frame,cfg["split_seed"]))


def test_preprocessing_is_fit_on_train_only(dataset):
    _,p=dataset
    transformer=p.preprocessor.named_transformers_["numeric"]
    columns=[c for c in p.manifest["feature_columns"] if c not in p.manifest["categorical_columns"]]
    expected=p.frame.loc[p.split=="train",columns].mean().to_numpy()
    np.testing.assert_allclose(transformer.named_steps["scale"].mean_,expected)


def test_edges_and_identity_preserved(dataset):
    _,p=dataset
    g=build_graph(p,"test")
    assert len(g.src)==np.sum(p.split=="test")
    assert len(g.message_src)==2*len(g.src)
    np.testing.assert_array_equal(g.hosts[g.src],p.frame.loc[p.split=="test","IPV4_SRC_ADDR"])
    owner=partition_nodes(g,"random",8,7)
    ids=np.concatenate([q for _,q in query_batches(g,owner,19)])
    np.testing.assert_array_equal(np.sort(ids),np.arange(len(g.src)))


@pytest.mark.parametrize("depth",[1,2,3])
@pytest.mark.parametrize("bidirectional",[True,False])
def test_exact_halo_matches_full_including_boundary_edges(dataset,depth,bidirectional):
    cfg,p=dataset; seed_all(7)
    g=build_graph(p,"test",bidirectional)
    model=EdgeSAGE(g.edge_x.shape[1],2,hidden=8,layers=depth,dropout=.3).eval()
    with torch.inference_mode(): expected=model(halo_batch(g,np.arange(len(g.src)),depth)).numpy()
    owner=partition_nodes(g,"random",8,7)
    actual=np.empty_like(expected)
    assert np.any(owner[g.src]!=owner[g.dst])
    with torch.inference_mode():
        for _,q in query_batches(g,owner,17): actual[q]=model(halo_batch(g,q,depth)).numpy()
    np.testing.assert_allclose(expected,actual,atol=2e-6,rtol=2e-6)


def test_test_labels_cannot_change_probabilities(dataset):
    cfg,p=dataset; seed_all(19)
    cfg.update(classes=2,label_names=p.names)
    g=build_graph(p,"test")
    model=EdgeSAGE(g.edge_x.shape[1],2,hidden=8)
    owner=partition_nodes(g,"random",4,7)
    a,t=infer(model,g,owner,cfg)
    g.y=1-g.y
    b,_=infer(model,g,owner,cfg)
    np.testing.assert_array_equal(a,b)
    assert t["coverage_min"]==t["coverage_max"]==1


def test_metrics_probabilities_and_minority_failures():
    y=np.array([0,0,1,1,1,1]); p=np.array([[.1,.9]]*6)
    m=evaluate(y,p,["Benign","Attack"])
    assert m["attack_false_positive_rate"]==1
    assert m["per_class"]["Benign"]["recall"]==0
    assert m["average_precision_ovr"]["Attack"]==pytest.approx(4/6)
    with pytest.raises(ValueError): evaluate(y,p*2,["Benign","Attack"])


def test_host_split_is_host_disjoint(dataset):
    _,p=dataset
    split=split_rows(p.frame,42,"host")
    hosts=[]
    for part in ["train","validation","test"]:
        f=p.frame.loc[split==part]
        hosts.append(set(f.IPV4_SRC_ADDR)|set(f.IPV4_DST_ADDR))
    assert not hosts[0]&hosts[1] and not hosts[0]&hosts[2] and not hosts[1]&hosts[2]
    assert np.any(split=="excluded_cross_host")


def test_temporal_requires_real_time_and_preserves_ties(dataset):
    _,p=dataset
    with pytest.raises(ValueError,match="timestamp"): split_rows(p.frame,protocol="temporal")
    f=p.frame.copy(); f["time"]=pd.date_range("2020-01-01",periods=len(f),freq="s")
    s=split_rows(f,protocol="temporal",timestamp="time")
    assert f.loc[s=="train","time"].max()<f.loc[s=="validation","time"].min()
    assert f.loc[s=="validation","time"].max()<f.loc[s=="test","time"].min()


def test_metis_never_silently_falls_back(dataset):
    _,p=dataset; g=build_graph(p,"test")
    try:
        owner=partition_nodes(g,"metis",4,7)
    except RuntimeError as exc:
        assert "METIS unavailable" in str(exc)
    else:
        assert len(owner)==len(g.hosts) and owner.min()>=0 and owner.max()<4


def test_memory_measurement_is_available():
    assert current_rss()>0


def test_cse_family_mapping_preserves_original_labels():
    from ca_edgeids.data import CSE_V2_LABEL_MAP
    raw=list(CSE_V2_LABEL_MAP)
    f=pd.DataFrame({"IPV4_SRC_ADDR":["10.0.0.1"]*len(raw),"IPV4_DST_ADDR":["10.0.0.2"]*len(raw),
                    "Label":[int(x!="Benign") for x in raw],"Attack":raw})
    y,names,original=normalize_labels(f,"cse","multiclass")
    assert list(original)==raw
    assert [names[i] for i in y]==[CSE_V2_LABEL_MAP[x] for x in raw]


def test_graph_roundtrip_without_pickle(dataset,tmp_path):
    from ca_edgeids.worker import save_graph,load_graph
    _,p=dataset
    g=build_graph(p,"test")
    save_graph(g,tmp_path/"graph.npz")
    restored=load_graph(tmp_path/"graph.npz")
    np.testing.assert_array_equal(restored.hosts,g.hosts)
    np.testing.assert_array_equal(restored.src,g.src)
    np.testing.assert_array_equal(restored.incoming.indices,g.incoming.indices)


def test_full_inference_is_one_batch(dataset):
    cfg,p=dataset; cfg.update(classes=2,label_names=p.names); seed_all(7)
    g=build_graph(p,"test")
    model=EdgeSAGE(g.edge_x.shape[1],2,hidden=8)
    _,timing=infer(model,g,partition_nodes(g,"full",1,7),cfg,full_graph=True)
    assert timing["batches"]==1


def test_missing_validation_class_stops_before_training(dataset,tmp_path):
    from ca_edgeids.experiment import run
    cfg,p=dataset
    f=p.frame.copy()
    f.loc[p.split=="validation","Attack"]="DDoS"
    f.loc[p.split=="validation","Label"]=1
    path=tmp_path/"invalid-validation.csv"; f.to_csv(path,index=False)
    cfg=dict(cfg,csv=str(path))
    with pytest.raises(ValueError,match="Validation split lacks"):
        run(cfg,tmp_path/"guard-run")
    assert not list((tmp_path/"guard-run").rglob("model.pt"))
