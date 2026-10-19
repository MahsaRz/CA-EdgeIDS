"""Validation-only sampler selection, five seeds, then a gated test stage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from ca_edgeids.controlled_training import predict
from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump
from ca_edgeids.graph import build_graph
from ca_edgeids.metrics import evaluate
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import environment,seed_all
from ca_edgeids.sampler_controls import fit_sampler
from ca_edgeids.statistics_backend import summarize
from run_classifier_comparison import common

ROOT=Path('runs/sampler-binary-100000')
PUBLIC=Path('artifacts/sampler-binary-100000')
PLAN=Path('configs/sampler_binary_100000.json')


def sources():
    paths=list(Path('src/ca_edgeids').glob('*.py'))+[PLAN,Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),Path('scripts/run_classifier_comparison.py')]
    return {str(p).replace('\\','/'):sha256_file(p) for p in paths}


def check_sources(plan):
    if sources()!=plan['source_sha256']: raise ValueError('Frozen numerical sources changed')
    for path,digest in plan['source_sha256'].items():
        content=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Sources were not committed before fitting')


def dataset_inputs(dataset,plan):
    _,cfg,data=common(dataset)
    cfg.update(epochs=plan['epochs'],patience=plan['patience'],hidden=plan['hidden'],layers=plan['layers'],
        dropout=plan['dropout'],weight_decay=plan['weight_decay'],max_edges_per_batch=plan['batch_size'],threads=plan['threads'])
    folder=ROOT/dataset
    if folder.exists():
        if data.manifest!=json.loads((folder/'dataset_manifest.json').read_text()): raise ValueError('Data manifest changed')
    else:
        folder.mkdir(); dump(folder/'dataset_manifest.json',data.manifest); dump(folder/'resolved_config.json',cfg)
    graphs={part:build_graph(data,part,cfg['bidirectional_messages']) for part in ['train','validation']}
    return cfg,data,graphs


def fit_one(dataset,method,seed,lr,index,cfg,graphs):
    dest=ROOT/dataset/f'{method}-seed-{seed}-candidate-{index}'
    dest.mkdir(); cfg=dict(cfg,learning_rate=lr)
    start=time.perf_counter()
    model,history,summary=fit_sampler(graphs['train'],graphs['validation'],cfg,seed,method)
    fit_s=time.perf_counter()-start
    file=dest/'model.pt'; torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,method=method),file)
    pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False)
    record=dict(dataset=dataset,method=method,seed=seed,learning_rate=lr,candidate_index=index,fit_s=fit_s,
        model_file=str(file).replace('\\','/'),model_sha256=sha256_file(file),**summary)
    dump(dest/'selection.json',record)
    return record


def export(stage,records,plan):
    out=PUBLIC/stage; out.mkdir(parents=True,exist_ok=False)
    dump(out/'frozen_plan.json',plan); dump(out/'records.json',records)
    flat=[{k:v for k,v in row.items() if k not in ['model_file','class_targets_seen']} for row in records]
    pd.DataFrame(flat).to_csv(out/'training_summary.csv',index=False)
    for r in records:
        source=Path(r['model_file']).parent; dest=out/r['dataset']/source.name; dest.mkdir(parents=True)
        for name in ['selection.json','training_history.csv']: shutil.copyfile(source/name,dest/name)
    for d in plan['datasets']:
        for name in ['dataset_manifest.json','resolved_config.json']: shutil.copyfile(ROOT/d/name,out/d/name)
    return out


def select():
    if ROOT.exists(): raise FileExistsError(ROOT)
    plan=json.loads(PLAN.read_text())
    plan.update(source_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        source_sha256=sources(),environment=environment())
    check_sources(plan); ROOT.mkdir(parents=True); dump(ROOT/'frozen_plan.json',plan)
    candidates=[]; choices=[]
    for d in plan['datasets']:
        cfg,_,graphs=dataset_inputs(d,plan)
        for method in plan['methods']:
            rows=[]
            for index,lr in enumerate(plan['learning_rates']):
                record=fit_one(d,method,plan['selection_seed'],lr,index,cfg,graphs)
                rows.append(record); candidates.append(record); dump(ROOT/'validation_candidates.json',candidates)
            choices.append(max(rows,key=lambda r:r['validation_macro_f1']))
            dump(ROOT/'method_choices.json',choices)
    check_sources(plan)
    out=export('validation-selection',candidates,plan); dump(out/'method_choices.json',choices)
    dump(ROOT/'validation_gate.json',dict(candidates=30,choices=15,choices_sha256=sha256_file(ROOT/'method_choices.json')))
    print('All 30 validation candidates exported; publish before the remaining seeds. No test predictions.',flush=True)


def seeds():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); check_sources(plan)
    if (ROOT/'selected_models.json').exists(): raise FileExistsError('Five-seed selection already exists')
    gate=json.loads((ROOT/'validation_gate.json').read_text())
    if gate['choices_sha256']!=sha256_file(ROOT/'method_choices.json'): raise ValueError('Choice gate changed')
    if sha256_file(PUBLIC/'validation-selection/method_choices.json')!=gate['choices_sha256']: raise ValueError('Public choice gate changed')
    choices=json.loads((ROOT/'method_choices.json').read_text()); records=list(choices)
    for d in plan['datasets']:
        cfg,_,graphs=dataset_inputs(d,plan)
        for chosen in [r for r in choices if r['dataset']==d]:
            if sha256_file(chosen['model_file'])!=chosen['model_sha256']: raise ValueError('Selected seed-7 checkpoint changed')
            for seed in plan['model_seeds']:
                if seed==plan['selection_seed']: continue
                records.append(fit_one(d,chosen['method'],seed,chosen['learning_rate'],chosen['candidate_index'],cfg,graphs))
                dump(ROOT/'partial_selections.json',records)
    check_sources(plan)
    if len(records)!=75: raise ValueError('Incomplete seed grid')
    dump(ROOT/'selected_models.json',records)
    out=export('five-seed-selection',records,plan)
    shutil.copyfile(ROOT/'selected_models.json',out/'selected_models.json')
    dump(ROOT/'test_gate.json',dict(selected_models=75,selection_sha256=sha256_file(ROOT/'selected_models.json'),test_predictions=0))
    print('All 75 checkpoint hashes exported; publish before evaluating. No test predictions.',flush=True)


def evaluate_all():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); check_sources(plan)
    records=json.loads((ROOT/'selected_models.json').read_text()); gate=json.loads((ROOT/'test_gate.json').read_text())
    if gate['selection_sha256']!=sha256_file(ROOT/'selected_models.json') or gate['selected_models']!=75: raise ValueError('Invalid test gate')
    if sha256_file(PUBLIC/'five-seed-selection/selected_models.json')!=gate['selection_sha256']: raise ValueError('Public test gate differs')
    expected={(d,m,s) for d in plan['datasets'] for m in plan['methods'] for s in plan['model_seeds']}
    if len(records)!=75 or {(r['dataset'],r['method'],r['seed']) for r in records}!=expected: raise ValueError('Incomplete grid')
    out=ROOT/'test-evaluation'; out.mkdir(exist_ok=False); rows=[]
    dump(out/'gate.json',gate)
    for d in plan['datasets']:
        _,cfg,data=common(d); cfg=json.loads((ROOT/d/'resolved_config.json').read_text())
        g=build_graph(data,'test',cfg['bidirectional_messages']); summary=summarize(g)
        for r in [r for r in records if r['dataset']==d]:
            if sha256_file(r['model_file'])!=r['model_sha256']: raise ValueError('Checkpoint changed')
            seed_all(r['seed'],plan['threads']); saved=torch.load(r['model_file'],map_location='cpu',weights_only=False)
            model=EdgeSAGE(g.edge_x.shape[1],2,cfg['hidden'],cfg['layers'],cfg['dropout']); model.load_state_dict(saved['state_dict'])
            probabilities=predict(model,g,summary,'full'); score=evaluate(g.y,probabilities,data.names)
            tag=f"{d}-{r['method']}-{r['seed']}"
            dump(out/(tag+'-metrics.json'),score)
            np.savez_compressed(out/(tag+'-predictions.npz'),row_ids=g.row_ids,y=g.y,probabilities=probabilities)
            row=dict(dataset=d,method=r['method'],seed=r['seed'],macro_f1=score['macro_f1'],weighted_f1=score['weighted_f1'],
                attack_fpr=score['attack_false_positive_rate'],attack_ap=score['average_precision_ovr']['Attack'],
                attack_recall=score['per_class']['Attack']['recall'],test_flows=len(g.src),target_row_ids_sha256=data.manifest['test_row_ids_sha256'])
            rows.append(row); pd.DataFrame(rows).to_csv(out/'model_results.csv',index=False)
    dump(out/'completion.json',dict(models=len(rows),retuned=False))
    print('All 75 frozen models evaluated on identical original test targets.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True,choices=['select','seeds','evaluate'])
    p.add_argument('--run-root',default=str(ROOT)); p.add_argument('--public-root',default=str(PUBLIC)); a=p.parse_args()
    ROOT=Path(a.run_root); PUBLIC=Path(a.public_root)
    {'select':select,'seeds':seeds,'evaluate':evaluate_all}[a.stage]()
