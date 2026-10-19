"""Shared checked preprocessing for larger offline workloads."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from ca_edgeids.data import SOURCE,DEST,LABEL_COLUMNS,Prepared,normalize_labels,sha256_file


def transform_frame(frame,cfg,split,pre=None):
    features=cfg['features']; categorical=cfg['categorical']; numeric=[c for c in features if c not in categorical]
    if not features or len(features)!=len(set(features)) or set(features)&(LABEL_COLUMNS|{SOURCE,DEST,'source_row_id','split'}):
        raise ValueError('Invalid feature allowlist')
    if not set(categorical).issubset(features): raise ValueError('Invalid categorical fields')
    y,names,original=normalize_labels(frame,cfg['dataset'],'binary')
    if len(np.unique(frame.source_row_id))!=len(frame): raise ValueError('Duplicate source row IDs')
    if not all(np.any(split==part) for part in ['train','validation','test']): raise ValueError('Empty required split')
    x=frame[features].copy()
    for c in numeric: x[c]=pd.to_numeric(x[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    for c in categorical: x[c]=x[c].fillna('<missing>').astype(str)
    fitted=pre is None
    if fitted:
        pre=ColumnTransformer([
            ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
        pre.fit(x.loc[split=='train'])
    # Transform in chunks to avoid a full-size intermediate float64 matrix.
    first=pre.transform(x.iloc[:1]); width=first.shape[1]
    if len(frame)*width*4>3*1024**3: raise MemoryError('Transformed feature storage exceeds the declared 3 GiB screening limit')
    z=np.empty((len(frame),width),dtype=np.float32)
    for start in range(0,len(frame),16384): z[start:start+16384]=pre.transform(x.iloc[start:start+16384]).astype(np.float32)
    if not np.isfinite(z).all(): raise ValueError('Non-finite features')
    fp=pd.util.hash_pandas_object(frame[[SOURCE,DEST]+features].astype(str),index=False).to_numpy()
    fingerprints={p:set(fp[split==p]) for p in ['train','validation','test']}
    overlap={f'{a}/{b}':len(fingerprints[a]&fingerprints[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(overlap.values()): raise ValueError('Duplicate content crosses splits')
    manifest=dict(dataset=cfg['dataset'],csv_sha256=sha256_file(cfg['csv']),rows_read=len(frame),feature_columns=features,
        transformed_width=width,label_names=names,original_labels=sorted(set(original)),preprocessor_fit_on_current_training=fitted,
        class_counts={p:dict(zip(names,np.bincount(y[split==p],minlength=2).tolist())) for p in ['train','validation','test']},
        split_counts={p:int(np.sum(split==p)) for p in np.unique(split)},duplicate_overlap=overlap,
        test_row_ids_sha256=hashlib.sha256(frame.loc[split=='test','source_row_id'].to_numpy(dtype='<i8').tobytes()).hexdigest())
    return Prepared(frame,z,y,names,split,manifest,pre)
