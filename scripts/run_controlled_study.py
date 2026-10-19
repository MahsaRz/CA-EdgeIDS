"""Five paired training seeds and explicit graph controls with frozen settings."""
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
from ca_edgeids.controlled_training import fit,graph_variant,predict
from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump
from ca_edgeids.graph import build_graph,partition_nodes,query_batches
from ca_edgeids.metrics import evaluate
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import environment,seed_all
from ca_edgeids.statistics_backend import summarize
from run_classifier_comparison import common

ROOT=Path('runs/controlled-binary-100000')
PLAN=Path('configs/controlled_binary_100000.json')


def source_snapshot():
    paths=list(Path('src/ca_edgeids').glob('*.py'))+[PLAN,Path('scripts/run_controlled_study.py'),Path('scripts/run_classifier_comparison.py')]
    return {str(p).replace('\\','/'):sha256_file(p) for p in paths}


def train_all():
    if ROOT.exists(): raise FileExistsError(ROOT)
    plan=json.loads(PLAN.read_text()); commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    sources=source_snapshot()
    for path,digest in sources.items():
        content=subprocess.run(['git','show',f'{commit}:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Commit numerical sources and plan before training')
    # Verify the settings were chosen earlier using validation, before this study.
    previous=json.loads(Path(plan['parameter_source']).read_text())
    for chosen in previous['models']:
        candidate=Path(plan['parameter_source']).parent/chosen['dataset']/chosen['candidate_id']/'candidate.json'
        settings=json.loads(candidate.read_text())['settings']
        if chosen['family'] in ['GNN','MLP'] and settings['learning_rate']!=plan['learning_rate']: raise ValueError('Neural settings changed')
        if chosen['family']=='RF' and settings['max_depth']!=plan['rf_depths'][chosen['dataset']]: raise ValueError('Forest settings changed')
    ROOT.mkdir(parents=True)
    dump(ROOT/'frozen_plan.json',dict(plan,source_commit=commit,source_sha256=sources,environment=environment(),
        earlier_selection_sha256=sha256_file(plan['parameter_source']),
        backend='factored linear mean messages; fresh node states per optimizer update; floating-point trajectories may differ from historical fits',
        uncertainty='training-seed variation on one fixed data split; paired ablation deltas; not population or data-split uncertainty'))
    records=[]
    for dataset in plan['datasets']:
        _,cfg,data=common(dataset); folder=ROOT/dataset; folder.mkdir()
        cfg.update(learning_rate=plan['learning_rate'],epochs=plan['epochs'],patience=plan['patience'],hidden=plan['hidden'],
            layers=plan['layers'],dropout=plan['dropout'],weight_decay=plan['weight_decay'],max_edges_per_batch=plan['batch_size'],threads=plan['threads'])
        dump(folder/'dataset_manifest.json',data.manifest); dump(folder/'resolved_config.json',cfg)
        base={part:build_graph(data,part,cfg['bidirectional_messages']) for part in ['train','validation']}
        owner=partition_nodes(base['train'],cfg['train_partition_method'],cfg['train_parts'],plan['ownership_seed'])
        batches=[q for _,q in query_batches(base['train'],owner,plan['batch_size'])]
        if not np.array_equal(np.sort(np.concatenate(batches)),np.arange(len(base['train'].src))): raise AssertionError('Training targets changed')
        tr=data.split=='train'; va=data.split=='validation'
        for variant in plan['variants']:
            graphs={part:graph_variant(graph,variant,plan['topology_seeds'][part]) for part,graph in base.items()}
            start=time.perf_counter(); summaries={part:summarize(graph) for part,graph in graphs.items()}; summary_s=time.perf_counter()-start
            for seed in plan['model_seeds']:
                dest=folder/f'gnn-{variant}-{seed}'; dest.mkdir()
                print(f'{dataset} GNN {variant} seed {seed}: starting',flush=True)
                start=time.perf_counter()
                model,history,metrics=fit(graphs['train'],graphs['validation'],summaries['train'],summaries['validation'],batches,cfg,seed,variant)
                fit_s=time.perf_counter()-start
                torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,variant=variant),dest/'model.pt')
                pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False)
                record=dict(dataset=dataset,family='GNN',variant=variant,seed=seed,fit_s=fit_s,summary_preparation_s=summary_s,
                    model_file=str(dest/'model.pt').replace('\\','/'),model_sha256=sha256_file(dest/'model.pt'),
                    training_message_arcs=len(graphs['train'].message_src),validation_message_arcs=len(graphs['validation'].message_src),**metrics)
                dump(dest/'selection.json',record); records.append(record); dump(ROOT/'selected_models.json',records)
        for family in ['MLP','RF']:
            for seed in plan['model_seeds']:
                dest=folder/f'{family.lower()}-full-{seed}'; dest.mkdir(); seed_all(seed,plan['threads'])
                print(f'{dataset} {family} seed {seed}: starting',flush=True); start=time.perf_counter()
                if family=='MLP':
                    model,history,metrics=fit_mlp(data.features[tr],data.labels[tr],data.features[va],data.labels[va],batches,cfg,seed)
                    fit_s=time.perf_counter()-start
                    torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed),dest/'model.pt')
                    pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False); file=dest/'model.pt'
                else:
                    model=RandomForestClassifier(n_estimators=plan['rf_trees'],max_depth=plan['rf_depths'][dataset],class_weight='balanced',random_state=seed,n_jobs=plan['threads'])
                    model.fit(data.features[tr],data.labels[tr]); fit_s=time.perf_counter()-start
                    if model.classes_.tolist()!=[0,1]: raise AssertionError('Forest classes changed')
                    score=evaluate(data.labels[va],model.predict_proba(data.features[va]),data.names)
                    metrics=dict(validation_macro_f1=score['macro_f1'],trees=len(model.estimators_),tree_nodes=sum(t.tree_.node_count for t in model.estimators_))
                    file=dest/'model.joblib'; joblib.dump(model,file)
                    dump(dest/'estimator_parameters.json',model.get_params())
                record=dict(dataset=dataset,family=family,variant='full',seed=seed,fit_s=fit_s,model_file=str(file).replace('\\','/'),model_sha256=sha256_file(file),**metrics)
                dump(dest/'selection.json',record); records.append(record); dump(ROOT/'selected_models.json',records)
    if len(records)!=120: raise AssertionError('Missing fit')
    if source_snapshot()!=sources: raise AssertionError('Numerical sources changed during fitting')
    dump(ROOT/'training_completion.json',dict(selected_models=120,test_evaluations=0))
    print('All 120 model selections are complete; no test predictions computed.',flush=True)


def evaluate_all():
    frozen=json.loads((ROOT/'frozen_plan.json').read_text()); chosen=json.loads((ROOT/'selected_models.json').read_text())
    if len(chosen)!=120: raise ValueError('Incomplete model selection')
    if source_snapshot()!=frozen['source_sha256']: raise ValueError('Numerical sources changed before evaluation')
    out=ROOT/'test-evaluation'; out.mkdir(exist_ok=False)
    dump(out/'evaluation_gate.json',dict(selection_sha256=sha256_file(ROOT/'selected_models.json'),models=120))
    records=[]
    for dataset in frozen['datasets']:
        _,cfg,data=common(dataset); cfg=json.loads((ROOT/dataset/'resolved_config.json').read_text())
        base=build_graph(data,'test',cfg['bidirectional_messages']); te=data.split=='test'
        summary_cache={}
        for selected in [r for r in chosen if r['dataset']==dataset]:
            family=selected['family']; variant=selected['variant']; seed=selected['seed']; seed_all(seed,frozen['threads'])
            file=Path(selected['model_file'])
            if sha256_file(file)!=selected['model_sha256']: raise ValueError('Selected checkpoint changed')
            if family=='GNN':
                if variant not in summary_cache:
                    graph=graph_variant(base,variant,frozen['topology_seeds']['test']); summary_cache[variant]=(graph,summarize(graph))
                graph,summary=summary_cache[variant]
                model=EdgeSAGE(base.edge_x.shape[1],2,cfg['hidden'],cfg['layers'],cfg['dropout'])
                model.load_state_dict(torch.load(file,map_location='cpu',weights_only=False)['state_dict'])
                probabilities=predict(model,graph,summary,variant)
            elif family=='MLP':
                model=FlowMLP(base.edge_x.shape[1],cfg['hidden'],2,cfg['dropout'])
                model.load_state_dict(torch.load(file,map_location='cpu',weights_only=False)['state_dict'])
                probabilities=predict_mlp(model,data.features[te])
            else: probabilities=joblib.load(file).predict_proba(data.features[te])
            metric=evaluate(data.labels[te],probabilities,data.names); tag=f'{dataset}-{family.lower()}-{variant}-{seed}'
            dump(out/(tag+'-metrics.json'),metric)
            np.savez_compressed(out/(tag+'-predictions.npz'),row_ids=base.row_ids,y=data.labels[te],probabilities=probabilities)
            row=dict(dataset=dataset,family=family,variant=variant,seed=seed,validation_macro_f1=selected['validation_macro_f1'],fit_s=selected['fit_s'],
                macro_f1=metric['macro_f1'],weighted_f1=metric['weighted_f1'],attack_fpr=metric['attack_false_positive_rate'],
                attack_recall=metric['per_class']['Attack']['recall'],benign_recall=metric['per_class']['Benign']['recall'],
                benign_precision=metric['per_class']['Benign']['precision'],attack_ap=metric['average_precision_ovr']['Attack'],
                test_flows=len(probabilities),target_row_ids_sha256=data.manifest['test_row_ids_sha256'])
            records.append(row); pd.DataFrame(records).to_csv(out/'model_results.csv',index=False)
            print(f'{tag}: evaluated',flush=True)
    dump(out/'completion.json',dict(evaluated_models=len(records),retuned=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    args=parser.parse_args(); train_all() if args.stage=='train' else evaluate_all()
