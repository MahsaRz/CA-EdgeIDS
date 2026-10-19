"""Separate validation-only model selection from selected-model test evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from ca_edgeids.cached import infer_cached
from ca_edgeids.classifier_study import FlowMLP,fit_mlp,predict_mlp,select_candidate
from ca_edgeids.data import prepare,sha256_file
from ca_edgeids.experiment import dump,train
from ca_edgeids.graph import build_graph,partition_nodes,query_batches
from ca_edgeids.metrics import evaluate
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import environment,seed_all

ROOT=Path('runs/classifier-comparison-100000')
PLAN=Path('configs/classifier_comparison_100000.json')


def common(dataset):
    original=Path('runs/linux-scaling-100000/training')/dataset
    cfg=json.loads((original/'resolved_config.json').read_text())
    data=prepare(cfg)
    if data.manifest!=json.loads((original/'dataset_manifest.json').read_text()):
        raise ValueError('Prepared data differs from the frozen pilot')
    for split in ['train','validation']:
        if any(count==0 for count in data.manifest['class_counts'][split].values()):
            raise ValueError('Missing selection class')
    return original,cfg,data


def gnn_from(path,width,cfg):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=EdgeSAGE(width,cfg['classes'],cfg['hidden'],cfg['layers'],cfg['dropout'])
    model.load_state_dict(saved['state_dict']); model.eval()
    return model


def select():
    if ROOT.exists(): raise FileExistsError(ROOT)
    plan=json.loads(PLAN.read_text()); seed_all(plan['seed'],plan['threads'])
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    snapshots={str(p).replace('\\','/'):sha256_file(p) for p in sorted(Path('src/ca_edgeids').glob('*.py'))}
    snapshots.update({str(PLAN):sha256_file(PLAN),'scripts/run_classifier_comparison.py':sha256_file(__file__)})
    for path,digest in snapshots.items():
        content=subprocess.run(['git','show',f'{commit}:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Commit the protocol and code before selection')
    ROOT.mkdir(parents=True,exist_ok=False)
    dump(ROOT/'frozen_plan.json',dict(plan,source_commit=commit,source_sha256=snapshots,environment=environment(),
        test_scope='previously inspected development holdout; not an untouched final test',
        budget='two candidates per family; neural maximum 10 epochs with patience 3; RF 100 trees per candidate; unequal compute reported'))
    all_records=[]; selections=[]
    for dataset in plan['datasets']:
        original,cfg,data=common(dataset); folder=ROOT/dataset; folder.mkdir()
        dump(folder/'dataset_manifest.json',data.manifest)
        cfg.update(epochs=plan['neural_epochs'],patience=plan['patience'],hidden=plan['hidden'],dropout=plan['dropout'],
                   weight_decay=plan['weight_decay'],max_edges_per_batch=plan['batch_size'],threads=plan['threads'])
        train_graph=build_graph(data,'train',cfg['bidirectional_messages'])
        val_graph=build_graph(data,'validation',cfg['bidirectional_messages'])
        owner=partition_nodes(train_graph,cfg['train_partition_method'],cfg['train_parts'],plan['seed'])
        batches=[q for _,q in query_batches(train_graph,owner,cfg['max_edges_per_batch'])]
        if not np.array_equal(np.sort(np.concatenate(batches)),np.arange(len(train_graph.src))): raise AssertionError('Training batch coverage changed')
        row_hash=hashlib.sha256(val_graph.row_ids.astype('<i8').tobytes()).hexdigest()
        tr=data.split=='train'; va=data.split=='validation'
        for family in ['GNN','MLP','RF']:
            records=[]
            for index in range(plan['candidates_per_family']):
                tag=f'{family.lower()}-{index}'; dest=folder/tag; dest.mkdir()
                fitcfg=copy.deepcopy(cfg); fitcfg['learning_rate']=plan['neural_learning_rates'][index]
                print(f'{dataset} {tag}: fitting or checking frozen candidate',flush=True)
                started=time.perf_counter(); reused=False
                if family=='GNN':
                    if index==0 and plan['reuse_original_gnn_candidate']:
                        original_code=json.loads((original/'code_manifest.json').read_text())['package_source_sha256']
                        for name in ['model.py','experiment.py','graph.py','data.py']:
                            if sha256_file(Path('src/ca_edgeids')/name)!=original_code[name]: raise ValueError('Reused fit numerical source differs')
                        prior_inputs=json.loads(Path('artifacts/cached-confirmation-100000/block-1/inputs.json').read_text())
                        if sha256_file(original/'seed-7/model.pt')!=prior_inputs[dataset]['checkpoint_sha256']:
                            raise ValueError('Original checkpoint differs from its earlier frozen hash')
                        old=torch.load(original/'seed-7/model.pt',map_location='cpu',weights_only=False)
                        for key in ['epochs','patience','hidden','layers','dropout','learning_rate','weight_decay','max_edges_per_batch','train_parts','train_partition_method','features','categorical']:
                            if old['config'][key]!=fitcfg[key]: raise ValueError('Reused GNN settings differ')
                        if old['seed']!=plan['seed']: raise ValueError('Reused seed differs')
                        for name in ['model.pt','training_history.csv','training_summary.json']:
                            shutil.copyfile(original/'seed-7'/name,dest/name)
                        summary=json.loads((dest/'training_summary.json').read_text())
                        model=gnn_from(dest/'model.pt',data.features.shape[1],fitcfg)
                        fit_s=float(pd.read_csv(original/'benchmark_runs.csv').iloc[0].training_s); reused=True
                    else:
                        model,summary=train(train_graph,val_graph,fitcfg,plan['seed'],dest)
                        fit_s=time.perf_counter()-started
                    val_p,_=infer_cached(model,val_graph,fitcfg)
                    summary.update(gradient_updates=len(batches)*summary['epochs_run'],parameters=sum(p.numel() for p in model.parameters()))
                    model_file=dest/'model.pt'; settings=dict(learning_rate=fitcfg['learning_rate'])
                elif family=='MLP':
                    model,history,summary=fit_mlp(data.features[tr],data.labels[tr],data.features[va],data.labels[va],batches,fitcfg,plan['seed'])
                    fit_s=time.perf_counter()-started
                    val_p=predict_mlp(model,data.features[va])
                    torch.save(dict(state_dict=model.state_dict(),config=fitcfg),dest/'model.pt')
                    pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False)
                    model_file=dest/'model.pt'; settings=dict(learning_rate=fitcfg['learning_rate'])
                else:
                    model=RandomForestClassifier(n_estimators=plan['rf_trees'],max_depth=plan['rf_depths'][index],
                        min_samples_leaf=1,max_features='sqrt',bootstrap=True,class_weight='balanced',random_state=plan['seed'],n_jobs=plan['threads'])
                    model.fit(data.features[tr],data.labels[tr]); fit_s=time.perf_counter()-started
                    if not np.array_equal(model.classes_,np.array([0,1])): raise AssertionError('RF class order differs')
                    val_p=model.predict_proba(data.features[va]); joblib.dump(model,dest/'model.joblib')
                    summary=dict(trees=len(model.estimators_),tree_nodes=sum(tree.tree_.node_count for tree in model.estimators_))
                    model_file=dest/'model.joblib'; settings=dict(max_depth=plan['rf_depths'][index],trees=plan['rf_trees'])
                score=evaluate(data.labels[va],val_p,data.names)
                if family in ['GNN','MLP'] and abs(score['macro_f1']-summary['validation_macro_f1'])>1e-12:
                    raise AssertionError('Stored best checkpoint differs from validation selection')
                dump(dest/'validation_metrics.json',score); dump(dest/'training_summary.json',summary)
                dump(dest/'resolved_config.json',fitcfg)
                record=dict(summary)
                record.update(dataset=dataset,family=family,candidate_id=tag,settings=settings,
                    validation_macro_f1=score['macro_f1'],validation_row_ids_sha256=row_hash,
                    validation_flows=len(val_graph.src),training_flows=len(train_graph.src),fit_s=fit_s,
                    reused_original_fit=reused,model_sha256=sha256_file(model_file),model_file=str(model_file).replace('\\','/'))
                # Summary contains the same validation score for neural models.
                dump(dest/'candidate.json',record); records.append(record); all_records.append(record)
                dump(ROOT/'candidate_records.json',all_records)
            winner=select_candidate(records)
            selected=next(r for r in records if r['candidate_id']==winner)
            selections.append(dict(dataset=dataset,family=family,candidate_id=winner,
                validation_macro_f1=selected['validation_macro_f1'],model_sha256=selected['model_sha256'],model_file=selected['model_file']))
    dump(ROOT/'selection.json',dict(rule=plan['selection'],models=selections,
         test_predictions_computed_in_selection=False,source_commit=commit))
    dump(ROOT/'selection_completion.json',dict(candidates=len(all_records),selected_models=len(selections),test_evaluations=0))
    print('Selection complete; selected models are frozen before test evaluation.',flush=True)


def test_selected():
    selection_path=ROOT/'selection.json'; selection=json.loads(selection_path.read_text())
    output=ROOT/'test-evaluation'
    if output.exists(): raise FileExistsError(output)
    output.mkdir()
    dump(output/'evaluation_gate.json',dict(selection_sha256=sha256_file(selection_path),models=selection['models'],
        evaluation_scope='selected candidates only; previously inspected development test splits; no retuning'))
    rows=[]
    for dataset in ['ton','cse','bot']:
        _,cfg,data=common(dataset); graph=build_graph(data,'test',cfg['bidirectional_messages']); te=data.split=='test'
        for chosen in [r for r in selection['models'] if r['dataset']==dataset]:
            seed_all(7,1); family=chosen['family']; path=Path(chosen['model_file'])
            if sha256_file(path)!=chosen['model_sha256']: raise ValueError('Selected model changed')
            saved_cfg=json.loads((path.parent/'resolved_config.json').read_text())
            started=time.perf_counter()
            if family=='GNN':
                model=gnn_from(path,data.features.shape[1],saved_cfg); load_s=time.perf_counter()-started
                p,timing=infer_cached(model,graph,saved_cfg); elapsed=timing['total_inference_s']
            elif family=='MLP':
                model=FlowMLP(data.features.shape[1],saved_cfg['hidden'],2,saved_cfg['dropout'])
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict']); load_s=time.perf_counter()-started
                started=time.perf_counter(); p=predict_mlp(model,data.features[te]); elapsed=time.perf_counter()-started
            else:
                model=joblib.load(path); load_s=time.perf_counter()-started
                started=time.perf_counter(); p=model.predict_proba(data.features[te]); elapsed=time.perf_counter()-started
            score=evaluate(data.labels[te],p,data.names)
            tag=f'{dataset}-{family.lower()}'
            dump(output/(tag+'-metrics.json'),score)
            np.savez_compressed(output/(tag+'-predictions.npz'),row_ids=graph.row_ids,probabilities=p,y=data.labels[te])
            rows.append(dict(dataset=dataset,family=family,candidate_id=chosen['candidate_id'],validation_macro_f1=chosen['validation_macro_f1'],
                macro_f1=score['macro_f1'],weighted_f1=score['weighted_f1'],attack_fpr=score['attack_false_positive_rate'],
                benign_recall=score['per_class']['Benign']['recall'],benign_precision=score['per_class']['Benign']['precision'],
                attack_average_precision=score['average_precision_ovr']['Attack'],test_flows=len(p),
                target_row_ids_sha256=data.manifest['test_row_ids_sha256'],model_load_s=load_s,
                diagnostic_shared_process_inference_s=elapsed))
            pd.DataFrame(rows).to_csv(output/'selected_results.csv',index=False)
            print(f'{tag}: selected-model test evaluation complete',flush=True)
    if len(rows)!=9: raise AssertionError('Incomplete selected-model evaluation')
    dump(output/'completion.json',dict(selected_models_evaluated=9,retuned=False,selection_sha256=sha256_file(selection_path)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    args=parser.parse_args(); select() if args.stage=='select' else test_selected()
