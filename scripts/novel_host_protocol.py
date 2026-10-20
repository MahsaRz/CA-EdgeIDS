"""A prespecified incident-flow holdout on hosts absent from development."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ca_edgeids.data import SOURCE,DEST,stable_bucket,normalize_labels,sha256_file
from ca_edgeids.experiment import dump


def assign(frame,development_hosts,seed=42):
    hosts=set(frame[SOURCE].astype(str))|set(frame[DEST].astype(str))
    novel=hosts-set(development_hosts)
    membership={h:('train' if stable_bucket(h,seed)<.6 else 'validation' if stable_bucket(h,seed)<.8 else 'test') for h in novel}
    membership.update({h:'train' for h in hosts-novel})
    a=frame[SOURCE].astype(str).map(membership).to_numpy(); b=frame[DEST].astype(str).map(membership).to_numpy()
    va=(a=='validation')|(b=='validation'); te=(a=='test')|(b=='test')
    split=np.where(va&te,'excluded_validation_test',np.where(te,'test',np.where(va,'validation','train')))
    held={part:{h for h in novel if membership[h]==part} for part in ['validation','test']}
    for part,forbidden in [('train',held['validation']|held['test']),('validation',held['test']),('test',held['validation'])]:
        present=set(frame.loc[split==part,SOURCE])|set(frame.loc[split==part,DEST])
        if present&forbidden: raise AssertionError('Held-out novel hosts leaked across the declared boundary')
    return split,held,novel


def audit(output='artifacts/novel-host-preflight-1000000'):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    reports=[]; rows=[]
    for d in ['ton','cse','bot']:
        cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())
        old=pd.read_csv(cfg['csv'],usecols=[SOURCE,DEST,'source_row_id'])
        oldhosts=set(old[SOURCE].astype(str))|set(old[DEST].astype(str))
        path=Path(f'data/larger-uniform/{d}/{d}-uniform-1000000.csv')
        source=json.loads(path.with_name('source_audit.json').read_text())
        frozen=next(s for s in source['samples'] if s['rows']==1000000)
        if sha256_file(path)!=frozen['csv_sha256']: raise ValueError('Larger sample changed')
        frame=pd.read_csv(path,low_memory=False); y,names,_=normalize_labels(frame,d,'binary')
        split,held,novel=assign(frame,oldhosts)
        old_ids=frame.source_row_id.isin(old.source_row_id).to_numpy()
        if int(old_ids.sum())!=100000 or not np.all(split[old_ids]=='train'): raise AssertionError('Development rows entered a new holdout')
        counts={part:dict(zip(names,np.bincount(y[split==part],minlength=2).tolist())) for part in ['train','validation','test','excluded_validation_test']}
        missing={part:[label for label,count in counts[part].items() if count==0] for part in ['train','validation','test']}
        nodes={p:set(frame.loc[split==p,SOURCE])|set(frame.loc[split==p,DEST]) for p in ['train','validation','test']}
        report=dict(dataset=d,rows=len(frame),csv_sha256=frozen['csv_sha256'],development_csv_sha256=sha256_file(cfg['csv']),
            development_nodes=len(oldhosts),new_nodes=len(novel),split_seed=42,
            class_counts=counts,missing_classes=missing,feasible=not any(missing.values()),
            held_out_novel_nodes={p:len(v) for p,v in held.items()},nodes={p:len(v) for p,v in nodes.items()},
            shared_all_nodes={f'{a}/{b}':len(nodes[a]&nodes[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
            held_out_test_nodes_seen_in_development=0,held_out_test_nodes_seen_in_training=0,
            original_development_rows_in_new_holdouts=0,
            test_row_ids_sha256=hashlib.sha256(frame.loc[split=='test','source_row_id'].to_numpy(dtype='<i8').tobytes()).hexdigest(),
            scope='new-host incident-flow holdout; known infrastructure may be shared; offline context; no strict all-host-disjoint, temporal, episode or causal claim')
        reports.append(report)
        for part,counts_part in counts.items():
            for label,count in counts_part.items(): rows.append(dict(dataset=d,split=part,label=label,flows=count))
        print(f'{d}: {report["new_nodes"]} new hosts; {counts}; feasible={report["feasible"]}',flush=True)
    dump(out/'protocols.json',reports); pd.DataFrame(rows).to_csv(out/'class_support.csv',index=False)
    (out/'SCOPE.txt').write_text('Protocol frozen before larger-sample support inspection. No model fitting or test predictions.\nHeld-out novel nodes were absent from all original 100000 development flows.\nKnown infrastructure may be shared. Validation-test incident crossings excluded.\nNo raw flows, host identities or row IDs published.\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='artifacts/novel-host-preflight-1000000'); a=p.parse_args(); audit(a.out)
