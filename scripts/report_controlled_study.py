"""Audit every seed and target before publishing paired graph-control results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file
from ca_edgeids.metrics import evaluate

ROOT=Path('runs/controlled-binary-100000'); PUBLIC=Path('artifacts/controlled-binary-100000')


def verify_selection():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); rows=json.loads((ROOT/'selected_models.json').read_text())
    done=json.loads((ROOT/'training_completion.json').read_text())
    if done!={'selected_models':120,'test_evaluations':0}: raise ValueError('Incomplete training stage')
    expected={(d,'GNN',v,s) for d in plan['datasets'] for v in plan['variants'] for s in plan['model_seeds']}|{(d,f,'full',s) for d in plan['datasets'] for f in ['MLP','RF'] for s in plan['model_seeds']}
    keys=[(r['dataset'],r['family'],r['variant'],r['seed']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Missing or duplicate selected models')
    for path,digest in plan['source_sha256'].items():
        content=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Source snapshot differs from commit')
    for r in rows:
        if sha256_file(r['model_file'])!=r['model_sha256']: raise ValueError('Selected checkpoint changed')
        history=Path(r['model_file']).parent/'training_history.csv'
        if history.exists():
            h=pd.read_csv(history)
            if not np.isclose(h.validation_macro_f1.max(),r['validation_macro_f1'],rtol=0,atol=1e-12): raise ValueError('Wrong best checkpoint score')
            if int(h.loc[h.validation_macro_f1.idxmax(),'epoch'])!=r['best_epoch']: raise ValueError('Wrong best epoch')
    return plan,rows


def selection():
    plan,rows=verify_selection(); out=PUBLIC/'selection'; out.mkdir(parents=True,exist_ok=False)
    for name in ['frozen_plan.json','selected_models.json','training_completion.json']: shutil.copyfile(ROOT/name,out/name)
    shutil.copyfile('artifacts/linux-scaling-100000/requirements.txt',out/'requirements.txt')
    for d in plan['datasets']:
        (out/d).mkdir()
        for name in ['dataset_manifest.json','resolved_config.json']: shutil.copyfile(ROOT/d/name,out/d/name)
    for r in rows:
        source=Path(r['model_file']).parent; dest=out/r['dataset']/source.name; dest.mkdir()
        for name in ['selection.json','training_history.csv','estimator_parameters.json']:
            if (source/name).exists(): shutil.copyfile(source/name,dest/name)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['model_file','class_weights']} for r in rows]).to_csv(out/'training_summary.csv',index=False)
    (out/'SCOPE.txt').write_text('All 120 selected checkpoints frozen before new test predictions.\nFive training seeds; one fixed previously inspected data split.\nGNN arithmetic is factored for all controls; hidden states recomputed each update.\nNo raw data, checkpoints, per-flow predictions or host identities exported.\n',encoding='utf-8')
    print('Verified and exported 120 pre-test selections')


def verify_test():
    plan,selected=verify_selection(); source=ROOT/'test-evaluation'
    gate=json.loads((source/'evaluation_gate.json').read_text())
    digest=sha256_file(ROOT/'selected_models.json')
    if gate['selection_sha256']!=digest or sha256_file(PUBLIC/'selection/selected_models.json')!=digest: raise ValueError('Published selections differ')
    table=pd.read_csv(source/'model_results.csv')
    keys=['dataset','family','variant','seed']
    if len(table)!=120 or table.duplicated(keys).any(): raise ValueError('Incomplete or duplicated test results')
    classes=[]; label_hashes={}
    for r in selected:
        tag=f"{r['dataset']}-{r['family'].lower()}-{r['variant']}-{r['seed']}"
        manifest=json.loads((ROOT/r['dataset']/'dataset_manifest.json').read_text())
        with np.load(source/(tag+'-predictions.npz'),allow_pickle=False) as saved:
            ids=saved['row_ids']; y=saved['y']; p=saved['probabilities']
            row_hash=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
            if len(np.unique(ids))!=len(ids) or row_hash!=manifest['test_row_ids_sha256']: raise ValueError('Test population differs')
            labels=hashlib.sha256(y.astype('<i8').tobytes()).hexdigest()
            if r['dataset'] in label_hashes and labels!=label_hashes[r['dataset']]: raise ValueError('Test labels differ')
            label_hashes[r['dataset']]=labels
            metrics=evaluate(y,p,['Benign','Attack'])
        if metrics!=json.loads((source/(tag+'-metrics.json')).read_text()): raise ValueError('Metrics differ from predictions')
        row=table[(table.dataset==r['dataset'])&(table.family==r['family'])&(table.variant==r['variant'])&(table.seed==r['seed'])]
        expected=dict(macro_f1=metrics['macro_f1'],weighted_f1=metrics['weighted_f1'],attack_fpr=metrics['attack_false_positive_rate'],
            attack_recall=metrics['per_class']['Attack']['recall'],benign_recall=metrics['per_class']['Benign']['recall'],
            benign_precision=metrics['per_class']['Benign']['precision'],attack_ap=metrics['average_precision_ovr']['Attack'],test_flows=metrics['test_n'])
        if len(row)!=1 or row.iloc[0].target_row_ids_sha256!=row_hash or any(not np.isclose(row.iloc[0][k],v,atol=1e-12,rtol=1e-12) for k,v in expected.items()):
            raise ValueError('Comparison row differs from recomputed metrics')
        for name in ['Benign','Attack']:
            item=metrics['per_class'][name]
            classes.append(dict(dataset=r['dataset'],family=r['family'],variant=r['variant'],seed=r['seed'],label=name,
                support=int(item['support']),precision=item['precision'],recall=item['recall'],f1=item['f1-score'],average_precision=metrics['average_precision_ovr'][name]))
    return table,pd.DataFrame(classes)


def test():
    table,classes=verify_test(); source=ROOT/'test-evaluation'; out=PUBLIC/'test-evaluation'; out.mkdir(parents=True,exist_ok=False)
    for file in source.glob('*.json'): shutil.copyfile(file,out/file.name)
    table.to_csv(out/'model_results.csv',index=False); classes.to_csv(out/'per_class_results.csv',index=False)
    summary=table.groupby(['dataset','family','variant'],as_index=False).agg(seeds=('seed','size'),mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),
        min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),mean_weighted_f1=('weighted_f1','mean'),sd_weighted_f1=('weighted_f1','std'),
        mean_attack_fpr=('attack_fpr','mean'),sd_attack_fpr=('attack_fpr','std'),mean_attack_recall=('attack_recall','mean'),mean_attack_ap=('attack_ap','mean'),mean_fit_s=('fit_s','mean'))
    if not summary.seeds.eq(5).all(): raise ValueError('Five seeds required for every setting')
    full=table[(table.family=='GNN')&(table.variant=='full')][['dataset','seed','macro_f1','attack_fpr']]
    pairs=table[(table.family=='GNN')&(table.variant!='full')].merge(full,on=['dataset','seed'],suffixes=('','_full'),validate='many_to_one')
    pairs['delta_macro_f1']=pairs.macro_f1-pairs.macro_f1_full; pairs['delta_attack_fpr']=pairs.attack_fpr-pairs.attack_fpr_full
    paired=pairs.groupby(['dataset','variant'],as_index=False).agg(seeds=('seed','size'),mean_delta_macro_f1=('delta_macro_f1','mean'),sd_delta_macro_f1=('delta_macro_f1','std'),
        min_delta_macro_f1=('delta_macro_f1','min'),max_delta_macro_f1=('delta_macro_f1','max'),mean_delta_attack_fpr=('delta_attack_fpr','mean'))
    for name,frame in [('summary',summary),('paired_seed_differences',pairs),('paired_ablation_summary',paired)]:
        frame.to_csv(out/(name+'.csv'),index=False); frame.to_latex(out/(name+'.tex'),index=False,float_format='%.6f',escape=True)
    (out/'verification.json').write_text(json.dumps(dict(models=120,five_seeds_per_setting=True,unchanged_targets=True,unchanged_labels=True,
        metrics_recomputed=True,comparison_tables_verified=True,paired_differences_match_training_seeds=True),indent=2)+'\n',encoding='utf-8')
    (out/'SCOPE.txt').write_text('Five trained seeds on the same fixed pilot split; seed variation is not independent dataset uncertainty.\nAll predefined variants retained. No test-based choice of seed, graph setting or threshold.\nAblation changes use identical target flows and paired model seeds.\nFit times are diagnostic; no fresh-process speed or memory claim is inferred.\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True,layout='constrained')
    methods=[('GNN','full','GraphSAGE'),('MLP','full','Flow MLP'),('RF','full','Forest')]
    for ax,d in zip(axes,['ton','cse','bot']):
        for x,(family,variant,label) in enumerate(methods):
            r=summary[(summary.dataset==d)&(summary.family==family)&(summary.variant==variant)].iloc[0]
            values=table[(table.dataset==d)&(table.family==family)&(table.variant==variant)].sort_values('seed').macro_f1.to_numpy()
            ax.scatter(x+np.linspace(-.1,.1,5),values,color='#3274a1',s=20)
            ax.errorbar(x,r.mean_macro_f1,yerr=r.sd_macro_f1,color='#222222',marker='_',capsize=4)
        ax.set_xticks(range(3),[m[2] for m in methods]); ax.set_title(d.upper()); ax.set_ylim(0,1.03)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Test macro F1'); fig.suptitle('Frozen classifiers across five training seeds\nDots: individual models · black: mean ± sample SD · fixed development split',fontsize=12)
    for ext in ['png','svg','pdf']: fig.savefig(out/('five_seed_classifiers.'+ext),dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True,layout='constrained')
    order=['no_degree','no_message_features','no_messages','rewired','directed']; labels=['No degree','No message\nfeatures','No messages','Rewired','Directed']
    for ax,d in zip(axes,['ton','cse','bot']):
        for x,variant in enumerate(order):
            values=pairs[(pairs.dataset==d)&(pairs.variant==variant)].sort_values('seed').delta_macro_f1.to_numpy()
            ax.scatter(x+np.linspace(-.12,.12,5),values,color='#3274a1',s=18)
            ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),marker='_',color='#222222',capsize=3)
        ax.axhline(0,color='#777777',linestyle='--'); ax.set_xticks(range(5),labels,rotation=35,ha='right',fontsize=8)
        ax.set_title(d.upper()); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Macro F1 difference from Full at the same seed')
    fig.suptitle('Paired graph controls\nDots: five seed differences · black: mean ± sample SD · all original test targets retained',fontsize=12)
    for ext in ['png','svg','pdf']: fig.savefig(out/('paired_graph_controls.'+ext),dpi=300)
    plt.close(fig)
    print(summary.to_string(index=False)); print(paired.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    args=parser.parse_args(); selection() if args.stage=='train' else test()
