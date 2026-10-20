"""Count independent identity support without exposing identities or fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ca_edgeids.data import SOURCE,DEST,normalize_labels,sha256_file
from novel_host_protocol import assign


def execute(output):
    out=Path(output); out.mkdir(parents=True,exist_ok=False); rows=[]; families=[]
    protocols=Path('artifacts/novel-host-preflight-1000000/protocols.json'); frozen=json.loads(protocols.read_text())
    for d in ['ton','cse','bot']:
        cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text()); old=pd.read_csv(cfg['csv'],usecols=[SOURCE,DEST])
        development=set(old[SOURCE])|set(old[DEST]); path=Path(f'data/larger-uniform/{d}/{d}-uniform-1000000.csv')
        expected=next(r for r in frozen if r['dataset']==d)
        if sha256_file(path)!=expected['csv_sha256']: raise ValueError('Sample changed')
        frame=pd.read_csv(path,usecols=[SOURCE,DEST,'Label','Attack','source_row_id'],low_memory=False)
        y,names,original=normalize_labels(frame,d,'binary'); split,held,novel=assign(frame,development)
        for part in ['train','validation','test']:
            for label_id,label in enumerate(names):
                keep=(split==part)&(y==label_id); subset=frame.loc[keep]
                nodes=set(subset[SOURCE])|set(subset[DEST]); pairs={tuple(sorted((s,t))) for s,t in zip(subset[SOURCE],subset[DEST])}
                rows.append(dict(dataset=d,split=part,label=label,flows=int(keep.sum()),distinct_pairs=len(pairs),
                    distinct_all_nodes=len(nodes),distinct_novel_nodes=len(nodes&novel),
                    active_held_out_novel_nodes=len(nodes&held[part]) if part in held else 0))
            counts=pd.Series(original[split==part]).value_counts()
            for label,count in counts.items(): families.append(dict(dataset=d,split=part,original_attack_name=label,flows=int(count)))
    pd.DataFrame(rows).to_csv(out/'class_identity_support.csv',index=False)
    pd.DataFrame(families).to_csv(out/'original_attack_support.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(protocols_sha256=sha256_file(protocols),model_fits=0,
        scope='descriptive counts on the already declared holdout; no identities published; host/pair counts do not prove independent attack episodes'),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[lambda x:x.split=='test'].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='artifacts/novel-host-preflight-1000000/group-support'); a=p.parse_args(); execute(a.out)
