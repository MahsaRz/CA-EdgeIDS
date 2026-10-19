"""Fixed-setting models on a novel-host incident-flow holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from ca_edgeids.classifier_study import FlowMLP,fit_mlp,predict_mlp
from ca_edgeids.controlled_training import fit,predict
from ca_edgeids.data import SOURCE,DEST,sha256_file
from ca_edgeids.experiment import dump
from ca_edgeids.graph import build_graph,partition_nodes,query_batches
from ca_edgeids.metrics import evaluate
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import environment,seed_all
from ca_edgeids.statistics_backend import summarize
from larger_data import transform_frame
from novel_host_protocol import assign

ROOT=Path('runs/novel-host-1000000'); PUBLIC=Path('artifacts/novel-host-1000000')
SEEDS=[7,19,31,43,59]


def sources():
    paths=list(Path('src/ca_edgeids').glob('*.py'))+[Path('scripts')/name for name in ['run_novel_host_study.py','larger_data.py','novel_host_protocol.py']]
    paths+=[Path('docs/NOVEL_HOST_PROTOCOL.md')]+[Path(f'configs/{d}_binary_linux_100000.json') for d in ['ton','cse','bot']]
    return {str(p).replace('\\','/'):sha256_file(p) for p in paths}


def check(plan):
    if sources()!=plan['sources']: raise ValueError('Sources changed during frozen experiment')
    for path,digest in plan['sources'].items():
        content=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Uncommitted source')


def inputs(d,pre=None):
    cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())
    old=pd.read_csv(cfg['csv'],usecols=[SOURCE,DEST])
    hosts=set(old[SOURCE])|set(old[DEST]); cfg.update(csv=f'data/larger-uniform/{d}/{d}-uniform-1000000.csv',classes=2,
        label_names=['Benign','Attack'],learning_rate=.003,epochs=10,patience=3,hidden=16,layers=2,dropout=.2,weight_decay=.0001,threads=1,max_edges_per_batch=512)
    frame=pd.read_csv(cfg['csv'],low_memory=False); split,_,_=assign(frame,hosts)
    expected=next(r for r in json.loads(Path('artifacts/novel-host-preflight-1000000/protocols.json').read_text()) if r['dataset']==d)
    if sha256_file(cfg['csv'])!=expected['csv_sha256'] or not expected['feasible']: raise ValueError('Infeasible or changed preflight')
    data=transform_frame(frame,cfg,split,pre)
    if data.manifest['class_counts']!={p:expected['class_counts'][p] for p in ['train','validation','test']}: raise ValueError('Class support changed')
    if data.manifest['test_row_ids_sha256']!=expected['test_row_ids_sha256']: raise ValueError('Target IDs changed')
    return cfg,data


def train_all():
    if ROOT.exists(): raise FileExistsError(ROOT)
    protocols=json.loads(Path('artifacts/novel-host-preflight-1000000/protocols.json').read_text())
    plan=dict(datasets=[r['dataset'] for r in protocols if r['feasible']],infeasible=[r['dataset'] for r in protocols if not r['feasible']],
        source_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),sources=sources(),seeds=SEEDS,
        environment=environment(),preflight_sha256=sha256_file('artifacts/novel-host-preflight-1000000/protocols.json'),hyperparameter_search=False)
    check(plan); ROOT.mkdir(parents=True); dump(ROOT/'frozen_plan.json',plan); selected=[]; failures=[]
    for d in plan['datasets']:
        cfg,data=inputs(d); folder=ROOT/d; folder.mkdir()
        joblib.dump(data.preprocessor,folder/'preprocessor.joblib'); dump(folder/'dataset_manifest.json',data.manifest); dump(folder/'resolved_config.json',cfg)
        graphs={p:build_graph(data,p,True) for p in ['train','validation']}
        owner=partition_nodes(graphs['train'],'random',4,7); batches=[q for _,q in query_batches(graphs['train'],owner,512)]
        try: summaries={p:summarize(g) for p,g in graphs.items()}
        except MemoryError as e: summaries=None; failures.append(dict(dataset=d,family='GNN',error=str(e)))
        tr=data.split=='train'; va=data.split=='validation'
        for family in ['GNN','MLP','RF']:
            if family=='GNN' and summaries is None: continue
            for seed in SEEDS:
                dest=folder/f'{family.lower()}-{seed}'; dest.mkdir(); seed_all(seed,1); start=time.perf_counter()
                print(f'{d} {family} seed {seed}: training',flush=True)
                if family=='GNN': model,history,metrics=fit(graphs['train'],graphs['validation'],summaries['train'],summaries['validation'],batches,cfg,seed,'full')
                elif family=='MLP': model,history,metrics=fit_mlp(data.features[tr],data.labels[tr],data.features[va],data.labels[va],batches,cfg,seed)
                else:
                    model=RandomForestClassifier(n_estimators=100,max_depth=None if d=='ton' else 16,class_weight='balanced',random_state=seed,n_jobs=1)
                    model.fit(data.features[tr],data.labels[tr]); history=None
                    metrics=dict(validation_macro_f1=evaluate(data.labels[va],model.predict_proba(data.features[va]),data.names)['macro_f1'])
                elapsed=time.perf_counter()-start
                if family=='RF': file=dest/'model.joblib'; joblib.dump(model,file)
                else:
                    file=dest/'model.pt'; torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed),file)
                    pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False)
                row=dict(dataset=d,family=family,seed=seed,fit_s=elapsed,model_file=str(file).replace('\\','/'),model_sha256=sha256_file(file),**metrics)
                selected.append(row); dump(dest/'selection.json',row); dump(ROOT/'selected_models.json',selected)
        dump(ROOT/'failures.json',failures)
    check(plan); out=PUBLIC/'selection'; out.mkdir(parents=True)
    for name in ['frozen_plan.json','selected_models.json','failures.json']: (out/name).write_bytes((ROOT/name).read_bytes())
    for d in plan['datasets']:
        (out/d).mkdir()
        for name in ['dataset_manifest.json','resolved_config.json']: (out/d/name).write_bytes((ROOT/d/name).read_bytes())
    for r in selected:
        src=Path(r['model_file']).parent; dest=out/r['dataset']/src.name; dest.mkdir()
        for name in ['training_history.csv','selection.json']:
            if (src/name).exists(): (dest/name).write_bytes((src/name).read_bytes())
    dump(ROOT/'test_gate.json',dict(selection_sha256=sha256_file(ROOT/'selected_models.json'),models=len(selected),test_predictions=0))
    print('Checkpoint hashes exported for publication before test evaluation.',flush=True)


def evaluate_all():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); check(plan)
    selected=json.loads((ROOT/'selected_models.json').read_text()); gate=json.loads((ROOT/'test_gate.json').read_text())
    if sha256_file(ROOT/'selected_models.json')!=gate['selection_sha256'] or sha256_file(PUBLIC/'selection/selected_models.json')!=gate['selection_sha256']: raise ValueError('Selection gate changed')
    out=ROOT/'test-evaluation'; out.mkdir(exist_ok=False); rows=[]
    dump(out/'gate.json',gate)
    for d in plan['datasets']:
        cfg,data=inputs(d,joblib.load(ROOT/d/'preprocessor.joblib')); g=build_graph(data,'test',True); summary=summarize(g)
        te=data.split=='test'
        for r in [r for r in selected if r['dataset']==d]:
            if sha256_file(r['model_file'])!=r['model_sha256']: raise ValueError('Checkpoint changed')
            seed_all(r['seed'],1); family=r['family']
            if family=='RF': probability=joblib.load(r['model_file']).predict_proba(data.features[te])
            else:
                model=EdgeSAGE(g.edge_x.shape[1],2,16,2,.2) if family=='GNN' else FlowMLP(g.edge_x.shape[1],16,2,.2)
                model.load_state_dict(torch.load(r['model_file'],map_location='cpu',weights_only=False)['state_dict'])
                probability=predict(model,g,summary,'full') if family=='GNN' else predict_mlp(model,data.features[te])
            metric=evaluate(g.y,probability,data.names); tag=f'{d}-{family.lower()}-{r["seed"]}'
            dump(out/(tag+'-metrics.json'),metric); np.savez_compressed(out/(tag+'-predictions.npz'),row_ids=g.row_ids,y=g.y,probabilities=probability)
            rows.append(dict(dataset=d,family=family,seed=r['seed'],macro_f1=metric['macro_f1'],weighted_f1=metric['weighted_f1'],
                attack_fpr=metric['attack_false_positive_rate'],attack_recall=metric['per_class']['Attack']['recall'],attack_ap=metric['average_precision_ovr']['Attack'],test_flows=len(g.src)))
            pd.DataFrame(rows).to_csv(out/'model_results.csv',index=False)
    dump(out/'completion.json',dict(models=len(rows),retuned=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['train','evaluate'],required=True); a=p.parse_args()
    train_all() if a.stage=='train' else evaluate_all()
