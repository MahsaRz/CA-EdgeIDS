"""Verify sampled-training selections, saved predictions and matched targets."""
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

ROOT=Path('runs/sampler-binary-100000'); PUBLIC=Path('artifacts/sampler-binary-100000')


def execute():
    plan=json.loads((ROOT/'frozen_plan.json').read_text())
    for path,digest in plan['source_sha256'].items():
        content=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Frozen source digest differs')
    candidates=json.loads((ROOT/'validation_candidates.json').read_text()); choices=json.loads((ROOT/'method_choices.json').read_text())
    selected=json.loads((ROOT/'selected_models.json').read_text())
    if len(candidates)!=30 or len(choices)!=15 or len(selected)!=75: raise ValueError('Incomplete selection grid')
    for chosen in choices:
        rows=[r for r in candidates if r['dataset']==chosen['dataset'] and r['method']==chosen['method']]
        if [r['learning_rate'] for r in rows]!=plan['learning_rates'] or chosen!=max(rows,key=lambda r:r['validation_macro_f1']): raise ValueError('Validation selection changed')
    for r in candidates+selected:
        if sha256_file(r['model_file'])!=r['model_sha256']: raise ValueError('Checkpoint changed')
        h=pd.read_csv(Path(r['model_file']).parent/'training_history.csv')
        if not np.isclose(h.validation_macro_f1.max(),r['validation_macro_f1'],rtol=0,atol=1e-12): raise ValueError('Wrong checkpoint score')
        if int(h.loc[h.validation_macro_f1.idxmax(),'epoch'])!=r['best_epoch']: raise ValueError('Wrong checkpoint epoch')
        if r['gradient_updates']!=r['epochs_run']*int(np.ceil(r['full_graph_targets']/plan['batch_size'])): raise ValueError('Update budget changed')
    source=ROOT/'test-evaluation'; gate=json.loads((source/'gate.json').read_text())
    if gate['selection_sha256']!=sha256_file(ROOT/'selected_models.json'): raise ValueError('Selection gate changed')
    out=PUBLIC/'test-evaluation'; out.mkdir(parents=True,exist_ok=False)
    rows=[]; target_identity={}
    csv=pd.read_csv(source/'model_results.csv')
    for r in selected:
        d=r['dataset']; method=r['method']; seed=r['seed']; tag=f'{d}-{method}-{seed}'
        with np.load(source/(tag+'-predictions.npz'),allow_pickle=False) as saved:
            ids=saved['row_ids']; y=saved['y']; probability=saved['probabilities']
            if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate test target')
            identity=(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),hashlib.sha256(y.astype('<i8').tobytes()).hexdigest())
            if d in target_identity and identity!=target_identity[d]: raise ValueError('Test population changed across methods or seeds')
            target_identity[d]=identity
            metric=evaluate(y,probability,['Benign','Attack'])
        prior=json.loads((source/(tag+'-metrics.json')).read_text())
        if prior!=metric: raise ValueError('Saved metrics disagree with predictions')
        match=csv[(csv.dataset==d)&(csv.method==method)&(csv.seed==seed)]
        if len(match)!=1 or match.iloc[0].target_row_ids_sha256!=identity[0]: raise ValueError('Missing CSV model or target mismatch')
        for name in ['macro_f1','weighted_f1']:
            if not np.isclose(match.iloc[0][name],metric[name],rtol=0,atol=1e-12): raise ValueError('CSV metric differs')
        dump_path=out/(tag+'-metrics.json'); shutil.copyfile(source/dump_path.name,dump_path)
        rows.append(dict(dataset=d,method=method,seed=seed,macro_f1=metric['macro_f1'],weighted_f1=metric['weighted_f1'],
            attack_fpr=metric['attack_false_positive_rate'],attack_ap=metric['average_precision_ovr']['Attack'],
            attack_recall=metric['per_class']['Attack']['recall'],fit_s=r['fit_s'],epochs=r['epochs_run'],
            updates=r['gradient_updates'],target_exposures=r['target_exposures'],unseen_training_targets=r['unseen_training_targets'],
            empty_samples=r['empty_samples'],test_flows=len(ids)))
    result=pd.DataFrame(rows); result.to_csv(out/'model_results.csv',index=False)
    summary=result.groupby(['dataset','method']).agg(seeds=('seed','count'),mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),
        mean_weighted_f1=('weighted_f1','mean'),mean_attack_fpr=('attack_fpr','mean'),mean_attack_recall=('attack_recall','mean'),
        mean_attack_ap=('attack_ap','mean'),mean_fit_s=('fit_s','mean'),mean_target_exposures=('target_exposures','mean'),
        mean_unseen_training_targets=('unseen_training_targets','mean'),mean_empty_samples=('empty_samples','mean')).reset_index()
    if not (summary.seeds==5).all(): raise ValueError('Missing seed')
    summary.to_csv(out/'summary.csv',index=False); summary.to_latex(out/'summary.tex',index=False,float_format='%.5f')
    deltas=[]
    for d in plan['datasets']:
        pivot=result[result.dataset==d].pivot(index='seed',columns='method',values='macro_f1')
        for m in plan['methods'][1:]:
            for seed,value in (pivot[m]-pivot.unpartitioned).items(): deltas.append(dict(dataset=d,method=m,seed=seed,macro_f1_delta=value))
    pd.DataFrame(deltas).to_csv(out/'paired_differences.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),sharey=True)
    labels=['Full','Random\nexact','METIS\nexact','Cluster\nsampled','Node\nsampled']
    for ax,d in zip(axes,plan['datasets']):
        for i,m in enumerate(plan['methods']):
            a=result[(result.dataset==d)&(result.method==m)].macro_f1.to_numpy()
            ax.scatter(i+np.linspace(-.13,.13,len(a)),a,s=10)
            ax.errorbar(i,a.mean(),yerr=a.std(ddof=1),color='black',capsize=3,linewidth=1)
        ax.set_title(d.upper()); ax.set_xticks(range(5),labels,rotation=45,ha='right'); ax.set_ylim(0,1.02); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Test macro F1'); fig.suptitle('Training procedures: five seeds, full test context',fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(out/f'sampler_quality.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    verification=dict(models=len(result),candidates=len(candidates),target_population_sha256=target_identity,
        selection_sha256=gate['selection_sha256'],checkpoint_and_saved_prediction_metrics_checked=True,retuned=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (out/'SCOPE.txt').write_text('Edge-task adaptations of cluster-induced and normalized uniform-node sampling.\nAll 75 selected models, five seeds and all original test targets retained.\nSampling only affects training; full context used for validation/test.\nVariation across training seeds on reused development holdouts; not population uncertainty.\nRaw data, host identities, checkpoints and per-flow predictions excluded.\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__': execute()
