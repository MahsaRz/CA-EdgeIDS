"""Report host-disjoint and temporal feasibility without fitting a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ca_edgeids.data import SOURCE,DEST,normalize_labels,sha256_file,split_rows,stable_bucket
from ca_edgeids.experiment import dump


def inspect():
    out=Path('artifacts/generalization-preflight-100000'); out.mkdir(parents=True,exist_ok=False)
    reports=[]; supports=[]
    for dataset in ['ton','cse','bot']:
        cfg=json.loads(Path(f'configs/{dataset}_binary_linux_100000.json').read_text())
        frame=pd.read_csv(cfg['csv'],low_memory=False); y,names,_=normalize_labels(frame,dataset,'binary')
        hosts=set(frame[SOURCE])|set(frame[DEST])
        def assignment(host):
            u=stable_bucket(host,cfg['split_seed'])
            return 'train' if u<.6 else 'validation' if u<.8 else 'test'
        membership={h:assignment(h) for h in hosts}
        split=np.array([membership[s] if membership[s]==membership[d] else 'excluded_cross_host' for s,d in zip(frame[SOURCE],frame[DEST])])
        if all(np.any(split==part) for part in ['train','validation','test']):
            np.testing.assert_array_equal(split,split_rows(frame,cfg['split_seed'],'host'))
        counts={}
        node_sets={}
        for part in ['train','validation','test','excluded_cross_host']:
            ids=np.flatnonzero(split==part)
            counts[part]=dict(zip(names,np.bincount(y[ids],minlength=2).tolist()))
            if part!='excluded_cross_host': node_sets[part]=set(frame.loc[ids,SOURCE])|set(frame.loc[ids,DEST])
            for label in names: supports.append(dict(dataset=dataset,split=part,label=label,flows=counts[part][label]))
        shared={f'{a}/{b}':len(node_sets[a]&node_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        if any(shared.values()): raise AssertionError('Host protocol leaks identities')
        missing={part:[label for label,count in counts[part].items() if count==0] for part in ['train','validation','test']}
        provenance=json.loads(Path(f'docs/provenance/nf-{dataset}-v2.json').read_text())
        columns=provenance['source_columns']
        timestamp_candidates=[c for c in columns if any(k in c.upper() for k in ['TIMESTAMP','DATETIME','START_TIME','FIRST_SWITCHED','LAST_SWITCHED'])]
        report=dict(dataset=dataset,sample_rows=len(frame),source_csv_sha256=sha256_file(cfg['csv']),split_seed=cfg['split_seed'],
            split_counts={part:int(np.sum(split==part)) for part in counts},class_counts=counts,missing_classes=missing,
            shared_hosts=shared,nodes_per_split={p:len(v) for p,v in node_sets.items()},
            feasible_binary_all_class=not any(missing.values()),timestamp_candidates=timestamp_candidates,
            temporal_protocol='requires genuine timestamp metadata; row order and duration are not event time',
            interpretation='host-disjoint feasibility is not accuracy; cross-host flows are explicitly excluded')
        reports.append(report)
    dump(out/'protocols.json',reports)
    pd.DataFrame(supports).to_csv(out/'class_support.csv',index=False)
    table=pd.DataFrame([dict(dataset=r['dataset'],**r['split_counts'],feasible=r['feasible_binary_all_class'],
        training_nodes=r['nodes_per_split']['train'],validation_nodes=r['nodes_per_split']['validation'],test_nodes=r['nodes_per_split']['test']) for r in reports])
    table.to_csv(out/'protocol_summary.csv',index=False); table.to_latex(out/'protocol_summary.tex',index=False,escape=True)
    (out/'SCOPE.txt').write_text('Host-disjoint feasibility on the frozen 100000-flow samples; no fitted models or accuracy measurements.\nAll excluded cross-host flows counted; no seed search. No timestamps invented from source row IDs or durations.\nRaw flows and host identities excluded.\n',encoding='utf-8')
    print(table.to_string(index=False))


if __name__=='__main__': inspect()
