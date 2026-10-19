"""Verify all new-host models against saved predictions and fixed supports."""
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
from ca_edgeids.experiment import dump
from ca_edgeids.metrics import evaluate

ROOT=Path('runs/novel-host-1000000'); PUBLIC=Path('artifacts/novel-host-1000000')


def execute():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); selected=json.loads((ROOT/'selected_models.json').read_text())
    failures=json.loads((ROOT/'failures.json').read_text()); source=ROOT/'test-evaluation'
    gate=json.loads((source/'gate.json').read_text())
    if gate['selection_sha256']!=sha256_file(ROOT/'selected_models.json') or gate['models']!=len(selected): raise ValueError('Selection gate mismatch')
    if sha256_file('artifacts/novel-host-preflight-1000000/protocols.json')!=plan['preflight_sha256']: raise ValueError('Preflight changed')
    for path,digest in plan['sources'].items():
        data=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('Frozen source mismatch')
    expected={(d,f,s) for d in plan['datasets'] for f in ['GNN','MLP','RF'] for s in plan['seeds']}
    for fail in failures:
        expected-={(fail['dataset'],fail['family'],s) for s in plan['seeds']}
    if {(r['dataset'],r['family'],r['seed']) for r in selected}!=expected or len(selected)!=len(expected): raise ValueError('Missing model')
    out=PUBLIC/'test-evaluation'; out.mkdir(parents=True,exist_ok=False); rows=[]; targets={}
    table=pd.read_csv(source/'model_results.csv')
    if len(table)!=len(selected): raise ValueError('CSV grid incomplete')
    for r in selected:
        d=r['dataset']; family=r['family']; seed=r['seed']; file=Path(r['model_file']); tag=f'{d}-{family.lower()}-{seed}'
        if sha256_file(file)!=r['model_sha256']: raise ValueError('Checkpoint mismatch')
        history=file.parent/'training_history.csv'
        if history.exists():
            h=pd.read_csv(history)
            if not np.isclose(h.validation_macro_f1.max(),r['validation_macro_f1'],rtol=0,atol=1e-12): raise ValueError('Best epoch mismatch')
        with np.load(source/(tag+'-predictions.npz'),allow_pickle=False) as saved:
            ids=saved['row_ids']; labels=saved['y']; p=saved['probabilities']; score=evaluate(labels,p,['Benign','Attack'])
            identity=(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest())
            if len(np.unique(ids))!=len(ids) or d in targets and identity!=targets[d]: raise ValueError('Target population mismatch')
            targets[d]=identity
        if score!=json.loads((source/(tag+'-metrics.json')).read_text()): raise ValueError('Metric mismatch')
        prior=table[(table.dataset==d)&(table.family==family)&(table.seed==seed)]
        if len(prior)!=1 or not np.isclose(prior.iloc[0].macro_f1,score['macro_f1'],rtol=0,atol=1e-12): raise ValueError('CSV mismatch')
        shutil.copyfile(source/(tag+'-metrics.json'),out/(tag+'-metrics.json'))
        rows.append(dict(dataset=d,family=family,seed=seed,macro_f1=score['macro_f1'],weighted_f1=score['weighted_f1'],attack_fpr=score['attack_false_positive_rate'],
            attack_recall=score['per_class']['Attack']['recall'],attack_ap=score['average_precision_ovr']['Attack'],
            benign_support=score['per_class']['Benign']['support'],attack_support=score['per_class']['Attack']['support'],fit_s=r['fit_s']))
    frame=pd.DataFrame(rows); frame.to_csv(out/'model_results.csv',index=False)
    summary=frame.groupby(['dataset','family']).agg(seeds=('seed','count'),mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),
        mean_weighted_f1=('weighted_f1','mean'),mean_attack_fpr=('attack_fpr','mean'),mean_attack_recall=('attack_recall','mean'),mean_attack_ap=('attack_ap','mean'),
        benign_support=('benign_support','first'),attack_support=('attack_support','first')).reset_index()
    if not (summary.seeds==5).all(): raise ValueError('Missing seed')
    summary.to_csv(out/'summary.csv',index=False); summary.to_latex(out/'summary.tex',index=False,float_format='%.4f')
    dump(out/'verification.json',dict(models=len(rows),target_id_and_label_hashes=targets,metrics_recomputed=True,retuned=False))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,len(plan['datasets']),figsize=(7.1,2.8),sharey=True,squeeze=False)
    for ax,d in zip(axes.flat,plan['datasets']):
        for i,f in enumerate(['GNN','MLP','RF']):
            a=frame[(frame.dataset==d)&(frame.family==f)].macro_f1.to_numpy()
            if len(a): ax.scatter(i+np.linspace(-.13,.13,len(a)),a,s=10); ax.errorbar(i,a.mean(),yerr=a.std(ddof=1),color='black',capsize=3)
        support=summary[summary.dataset==d].iloc[0]
        ax.set_title(f'{d.upper()}: {int(support.attack_support)} attack / {int(support.benign_support)} benign')
        ax.set_xticks(range(3),['Edge GNN','Flow MLP','Forest']); ax.set_ylim(0,1.02); ax.grid(axis='y',alpha=.2)
    axes.flat[0].set_ylabel('New-host test macro F1'); fig.tight_layout()
    for ext in ['png','svg','pdf']: fig.savefig(out/f'novel_host_quality.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (out/'SCOPE.txt').write_text('New-host incident-flow classification; known infrastructure may be shared.\nVery small attack supports limit generalization evidence. BoT fails all-class feasibility and is not fitted.\nFive training seeds are not independent data splits or population confidence intervals.\nNo raw flows, host identities, checkpoints or per-flow predictions exported.\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__': execute()
