import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from novel_host_protocol import assign
from larger_data import transform_frame
from ca_edgeids.data import stable_bucket,prepare
from ca_edgeids.cli import demo_config


def test_novel_host_protocol_excludes_crossings_and_keeps_development_in_train():
    known={'10.0.0.1','10.0.0.2'}
    hosts=[f'10.0.1.{i}' for i in range(1,200)]
    val=next(h for h in hosts if .6<=stable_bucket(h,42)<.8)
    test=next(h for h in hosts if stable_bucket(h,42)>=.8)
    frame=pd.DataFrame({'IPV4_SRC_ADDR':['10.0.0.1',val,test,val,test],
        'IPV4_DST_ADDR':['10.0.0.2','10.0.0.1','10.0.0.1',test,test]})
    split,held,novel=assign(frame,known)
    assert split.tolist()==['train','validation','test','excluded_validation_test','test']
    assert held=={'validation':{val},'test':{test}}
    assert novel=={val,test}
    # Reversing flows and changing row order cannot change the assignment.
    reverse=frame.rename(columns={'IPV4_SRC_ADDR':'IPV4_DST_ADDR','IPV4_DST_ADDR':'IPV4_SRC_ADDR'})
    np.testing.assert_array_equal(assign(reverse,known)[0],split)
    np.testing.assert_array_equal(assign(frame.iloc[::-1],known)[0],split[::-1])


def test_chunked_larger_transform_matches_existing_train_only_preprocessing(tmp_path):
    cfg=demo_config(tmp_path); original=prepare(cfg)
    frame=original.frame.copy(); frame['source_row_id']=np.arange(len(frame))
    fresh=transform_frame(frame,cfg,original.split)
    np.testing.assert_array_equal(fresh.features,original.features)
    fixed=transform_frame(frame,cfg,original.split,original.preprocessor)
    np.testing.assert_array_equal(fixed.features,original.features)
    heldout=original.split!='train'
    numeric=next(c for c in cfg['features'] if c not in cfg['categorical'])
    frame.loc[heldout,numeric]=1e9
    changed=transform_frame(frame,cfg,original.split)
    np.testing.assert_array_equal(changed.features[~heldout],original.features[~heldout])
    assert not fixed.manifest['preprocessor_fit_on_current_training']
    assert fresh.manifest['preprocessor_fit_on_current_training']
