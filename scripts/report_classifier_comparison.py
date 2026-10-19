"""Publish selection evidence before exporting the selected test results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_edgeids.classifier_study import select_candidate
from ca_edgeids.data import sha256_file
from ca_edgeids.metrics import evaluate

PRIVATE=Path('runs/classifier-comparison-100000')
PUBLIC=Path('artifacts/classifier-comparison-100000')


def selections():
    frozen=json.loads((PRIVATE/'frozen_plan.json').read_text())
    for path,digest in frozen['source_sha256'].items():
        content=subprocess.run(['git','show',f"{frozen['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Frozen source differs from its commit')
        if sha256_file(path)!=digest: raise ValueError('Current numerical source or protocol differs from selection')
    records=json.loads((PRIVATE/'candidate_records.json').read_text())
    selected=json.loads((PRIVATE/'selection.json').read_text())
    if len(records)!=18 or len(selected['models'])!=9: raise ValueError('Incomplete selection matrix')
    expected={(d,f) for d in ['ton','cse','bot'] for f in ['GNN','MLP','RF']}
    if {(r['dataset'],r['family']) for r in selected['models']}!=expected: raise ValueError('Missing selected family')
    for chosen in selected['models']:
        candidates=[r for r in records if (r['dataset'],r['family'])==(chosen['dataset'],chosen['family'])]
        if len(candidates)!=2 or select_candidate(candidates)!=chosen['candidate_id']: raise ValueError('Selection differs from validation rule')
        winner=next(r for r in candidates if r['candidate_id']==chosen['candidate_id'])
        if winner['model_sha256']!=chosen['model_sha256'] or winner['validation_macro_f1']!=chosen['validation_macro_f1']: raise ValueError('Selected checkpoint differs')
        if sha256_file(chosen['model_file'])!=chosen['model_sha256']: raise ValueError('Selected model changed')
    return records,selected


def export_selection():
    records,selected=selections(); out=PUBLIC/'selection'; out.mkdir(parents=True,exist_ok=False)
    for name in ['frozen_plan.json','candidate_records.json','selection.json','selection_completion.json']:
        shutil.copyfile(PRIVATE/name,out/name)
    shutil.copyfile('artifacts/linux-scaling-100000/requirements.txt',out/'requirements.txt')
    rows=[]
    for record in records:
        dataset=record['dataset']; tag=record['candidate_id']; source=PRIVATE/dataset/tag; dest=out/dataset/tag
        dest.mkdir(parents=True,exist_ok=False)
        for name in ['candidate.json','resolved_config.json','validation_metrics.json','training_summary.json','training_history.csv']:
            if (source/name).exists(): shutil.copyfile(source/name,dest/name)
        if record['family']=='RF':
            parameters=joblib.load(record['model_file']).get_params()
            (dest/'estimator_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n',encoding='utf-8')
        rows.append({k:v for k,v in record.items() if k not in ['class_weights','model_file','settings']}|{'settings':json.dumps(record['settings'],sort_keys=True)})
    for dataset in ['ton','cse','bot']: shutil.copyfile(PRIVATE/dataset/'dataset_manifest.json',out/dataset/'dataset_manifest.json')
    table=pd.DataFrame(rows); table.to_csv(out/'candidate_summary.csv',index=False)
    table.to_latex(out/'candidate_summary.tex',index=False,float_format='%.6f',na_rep='--',escape=True)
    (out/'SCOPE.txt').write_text('Validation-only selection from two candidates per family and dataset.\nNeural epoch/update upper limits matched; forest tree budget separate. Candidate count is not equal computing cost.\nOriginal GNN candidate fits reused with source/config/checkpoint and validation checks.\nOne model seed; previously inspected pilot splits. Test predictions are not computed during selection.\nNo checkpoints, raw flows or per-flow predictions exported.\n',encoding='utf-8')
    print(table[['dataset','family','candidate_id','validation_macro_f1','fit_s','reused_original_fit']].to_string(index=False))


def export_test():
    records,selected=selections(); selection_sha=sha256_file(PRIVATE/'selection.json')
    if sha256_file(PUBLIC/'selection/selection.json')!=selection_sha: raise ValueError('Public selection differs')
    source=PRIVATE/'test-evaluation'; gate=json.loads((source/'evaluation_gate.json').read_text())
    if gate['selection_sha256']!=selection_sha or gate['models']!=selected['models']: raise ValueError('Evaluation selection gate differs')
    table=pd.read_csv(source/'selected_results.csv')
    if len(table)!=9 or table.duplicated(['dataset','family']).any(): raise ValueError('Missing or duplicate selected results')
    confusion={}; per_class=[]; labels_by_dataset={}
    for chosen in selected['models']:
        dataset=chosen['dataset']; family=chosen['family']; tag=f'{dataset}-{family.lower()}'
        manifest=json.loads((PRIVATE/dataset/'dataset_manifest.json').read_text())
        with np.load(source/(tag+'-predictions.npz'),allow_pickle=False) as saved:
            row_ids=saved['row_ids']; y=saved['y']; probabilities=saved['probabilities']
            digest=hashlib.sha256(row_ids.astype('<i8').tobytes()).hexdigest()
            if digest!=manifest['test_row_ids_sha256'] or len(row_ids)!=manifest['split_counts']['test'] or len(np.unique(row_ids))!=len(row_ids):
                raise ValueError('Test population differs')
            if dataset in labels_by_dataset and not np.array_equal(labels_by_dataset[dataset],y): raise ValueError('Labels differ across models')
            labels_by_dataset[dataset]=y.copy()
            metric=evaluate(y,probabilities,['Benign','Attack'])
        if metric!=json.loads((source/(tag+'-metrics.json')).read_text()): raise ValueError('Metrics differ from saved predictions')
        row=table[(table.dataset==dataset)&(table.family==family)]
        if len(row)!=1 or row.iloc[0].candidate_id!=chosen['candidate_id'] or row.iloc[0].target_row_ids_sha256!=digest:
            raise ValueError('Results differ from selected models')
        confusion[(dataset,family)]=np.asarray(metric['confusion_matrix'])
        for label in ['Benign','Attack']:
            item=metric['per_class'][label]
            per_class.append(dict(dataset=dataset,family=family,label=label,precision=item['precision'],recall=item['recall'],f1=item['f1-score'],
                                  support=int(item['support']),average_precision=metric['average_precision_ovr'][label],ovr_fpr=metric['fpr_ovr'][label]))
    out=PUBLIC/'test-evaluation'; out.mkdir(parents=True,exist_ok=False)
    for file in source.glob('*.json'): shutil.copyfile(file,out/file.name)
    shutil.copyfile(source/'selected_results.csv',out/'selected_results.csv')
    table.to_latex(out/'selected_results.tex',index=False,float_format='%.6f',escape=True)
    per_class=pd.DataFrame(per_class); per_class.to_csv(out/'per_class_metrics.csv',index=False)
    per_class.to_latex(out/'per_class_metrics.tex',index=False,float_format='%.6f',escape=True)
    (out/'verification.json').write_text(json.dumps(dict(selection_sha256=selection_sha,selected_models=9,
        same_target_population=True,metrics_recomputed_from_private_predictions=True,validation_only_candidate_selection=True),indent=2)+'\n',encoding='utf-8')
    (out/'SCOPE.txt').write_text('Selected-model evaluation on previously inspected 100000-flow pilot splits.\nOne initialization seed; no model-seed confidence intervals or final unseen-host claim.\nTiming values in selected_results are shared-process single observations, not a latency or memory comparison.\nTest results do not authorize further candidate or threshold tuning on this holdout.\n',encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharey='row',layout='constrained')
    for col,dataset in enumerate(['ton','cse','bot']):
        d=table[table.dataset==dataset].set_index('family').loc[['GNN','MLP','RF']]
        x=np.arange(3)
        axes[0,col].bar(x-.18,d.macro_f1,width=.36,label='Macro F1',color='#3274a1')
        axes[0,col].bar(x+.18,d.weighted_f1,width=.36,label='Weighted F1',color='#d98832')
        axes[1,col].bar(x,d.attack_fpr,color='#b34c48',width=.6)
        axes[0,col].set_title(dataset.upper())
        for ax in axes[:,col]:
            ax.set_xticks(x,['GraphSAGE','Flow MLP','Forest'],fontsize=9); ax.set_ylim(0,1.05)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0,0].set_ylabel('F1'); axes[1,0].set_ylabel('Attack false-positive rate')
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Validation-selected classifiers on the same pilot test flows\nOne model seed · two candidates per family · no test-driven candidate selection',fontsize=12)
    for extension in ['png','svg','pdf']: fig.savefig(out/('classifier_comparison.'+extension),dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(10,9),layout='constrained')
    for row,family in enumerate(['GNN','MLP','RF']):
        for col,dataset in enumerate(['ton','cse','bot']):
            ax=axes[row,col]; counts=confusion[(dataset,family)]; rates=counts/counts.sum(axis=1,keepdims=True)
            ax.imshow(rates,cmap='Blues',vmin=0,vmax=1)
            for i in range(2):
                for j in range(2): ax.text(j,i,f'{counts[i,j]:,}\n{rates[i,j]:.1%}',ha='center',va='center',fontsize=9,color='white' if rates[i,j]>.6 else '#15324a')
            ax.set_xticks([0,1],['Benign','Attack']); ax.set_yticks([0,1],['Benign','Attack'])
            if row==0: ax.set_title(dataset.upper())
            if col==0: ax.set_ylabel(f'{family}\nTrue class')
            if row==2: ax.set_xlabel('Predicted class')
    fig.suptitle('Selected-model test confusion matrices\nCounts and percentages within each true class · one model seed',fontsize=12)
    for extension in ['png','svg','pdf']: fig.savefig(out/('confusion_matrices.'+extension),dpi=300)
    plt.close(fig)
    print(table.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    args=parser.parse_args(); export_selection() if args.stage=='select' else export_test()
