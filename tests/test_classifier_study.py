import numpy as np
import pytest

from ca_edgeids.classifier_study import FlowMLP,fit_mlp,predict_mlp,select_candidate


def test_selection_uses_validation_and_stable_ties():
    records=[dict(candidate_id='a',validation_macro_f1=.8,validation_row_ids_sha256='rows',test_macro_f1=0),
             dict(candidate_id='b',validation_macro_f1=.7,validation_row_ids_sha256='rows',test_macro_f1=1)]
    assert select_candidate(records)=='a'
    records[1]['validation_macro_f1']=.8
    assert select_candidate(records)=='a'
    records[1]['validation_row_ids_sha256']='different'
    with pytest.raises(ValueError,match='populations'): select_candidate(records)


def test_mlp_rejects_missing_validation_class():
    x=np.ones((4,2),dtype=np.float32); y=np.array([0,0,1,1])
    with pytest.raises(ValueError,match='every class'):
        fit_mlp(x,y,x,np.zeros(4,dtype=int),[np.arange(4)],dict(classes=2),7)


def test_mlp_training_is_deterministic_and_has_full_batch_coverage():
    rng=np.random.default_rng(4); x=rng.normal(size=(80,3)).astype(np.float32)
    y=(x[:,0]>0).astype(np.int64)
    batches=[np.arange(a,min(a+13,len(x))) for a in range(0,len(x),13)]
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(len(x)))
    cfg=dict(classes=2,hidden=8,dropout=.2,learning_rate=.01,weight_decay=.0001,epochs=4,patience=3,label_names=['Benign','Attack'])
    a,history,summary=fit_mlp(x,y,x,y,batches,cfg,7)
    b,other,second=fit_mlp(x,y,x,y,batches,cfg,7)
    assert history==other and summary==second
    assert summary['gradient_updates']==len(batches)*summary['epochs_run']
    np.testing.assert_array_equal(predict_mlp(a,x,17),predict_mlp(b,x,17))
    np.testing.assert_allclose(predict_mlp(a,x,17),predict_mlp(a,x,80),atol=1e-6,rtol=1e-6)
