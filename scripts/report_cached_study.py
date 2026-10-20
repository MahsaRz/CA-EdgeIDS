"""Verify and export aggregate layer-wise inference measurements."""
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


def build(source='runs/cached-inference-100000',public='artifacts/cached-inference-100000',draw=True):
    source=Path(source); public=Path(public)
    plan=json.loads((source/'frozen_plan.json').read_text())
    inputs=json.loads((source/'inputs.json').read_text())
    done=json.loads((source/'completion.json').read_text())
    checks=json.loads((source/'prediction_checks.json').read_text())
    failures=json.loads((source/'failures.json').read_text())
    rows=pd.read_csv(source/'timing_repeats.csv')
    keys=['dataset','method','message_batch_size','query_batch_size']
    key=lambda r:tuple(r[k] for k in keys)
    expected={key(job) for job in plan['execution_order']}
    valid={key(row) for row in checks}; failed={key(row) for row in failures}
    if len(expected)!=18 or valid&failed or valid|failed!=expected or len(valid)!=len(checks) or len(failed)!=len(failures):
        raise ValueError('Missing or duplicated conditions')
    if done['completed_conditions']!=len(checks) or done['failed_conditions']!=len(failures) or done['timing_measurements']!=len(rows):
        raise ValueError('Completion counts differ')
    snapshots=[(f'src/ca_edgeids/{name}',digest) for name,digest in plan['package_source_sha256'].items()]
    snapshots.append(('scripts/run_cached_study.py',plan['driver_sha256']))
    for path,digest in snapshots:
        content=subprocess.run(['git','show',f"{plan['git_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Experimental source differs from commit')
    for check in checks:
        item=inputs[check['dataset']]
        if check['target_row_ids_sha256']!=item['target_row_ids_sha256'] or not check['same_class_predictions'] or check['maximum_probability_difference']>2e-5:
            raise ValueError('Reference agreement check failed')
    actual=set()
    for identity,group in rows.groupby(keys):
        actual.add(identity)
        if len(group)!=3 or set(group['repeat'])!={0,1,2}: raise ValueError('Repeat count differs')
        item=inputs[identity[0]]
        if not group.target_row_ids_sha256.eq(item['target_row_ids_sha256']).all(): raise ValueError('Population changed')
        if not group.coverage_min.eq(1).all() or not group.coverage_max.eq(1).all(): raise ValueError('Coverage changed')
        if not ((group.first_prediction_process_s>0)&(group.first_prediction_process_s<=group.worker_process_wall_s)).all(): raise ValueError('Process timing invalid')
        if identity[1]=='cached' and not group.message_arc_visits.eq(2*item['message_arcs']).all(): raise ValueError('Message visits changed')
    if actual!=valid: raise ValueError('Timing conditions differ')
    summary=rows.groupby(keys,as_index=False).agg(repeats=('repeat','size'),mean_inference_s=('total_inference_s','mean'),
        sd_inference_s=('total_inference_s','std'),peak_rss_mib=('peak_cpu_rss_mb_sampled','max'),
        first_prediction_process_s=('first_prediction_process_s','first'),mean_forward_s=('model_forward_s','mean'),
        mean_packing_s=('halo_and_transfer_s','mean'),message_arc_visits=('message_arc_visits','first'))
    for dataset in ['ton','cse','bot']:
        d=summary[summary.dataset==dataset]
        full=d[d.method=='full']; halo=d[d.method=='chunked']
        if len(full)!=1 or len(halo)!=1: raise ValueError('Required controls missing')
        index=summary.dataset==dataset
        summary.loc[index,'time_relative_to_full']=summary.loc[index,'mean_inference_s']/full.iloc[0].mean_inference_s
        summary.loc[index,'time_relative_to_halo512']=summary.loc[index,'mean_inference_s']/halo.iloc[0].mean_inference_s
    public.mkdir(parents=True,exist_ok=False)
    for name in ['frozen_plan.json','inputs.json','completion.json','prediction_checks.json','failures.json','timing_repeats.csv']:
        shutil.copyfile(source/name,public/name)
    shutil.copyfile('artifacts/linux-scaling-100000/requirements.txt',public/'requirements.txt')
    summary.to_csv(public/'condition_summary.csv',index=False)
    summary.to_latex(public/'condition_summary.tex',index=False,float_format='%.6f',na_rep='--',escape=True)
    verification=dict(source_snapshots_match_commit=True,conditions_accounted_for=len(expected),
        same_target_coverage=True,same_class_predictions=True,
        maximum_probability_difference=max(row['maximum_probability_difference'] for row in checks),
        message_visits_once_per_layer=True,cache_recomputed_each_repeat=True)
    (public/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    (public/'SCOPE.txt').write_text('Fixed seed-7 models from three 100000-flow samples; no retraining or test-driven tuning.\nEach timing includes recomputing node states, then all target predictions.\nWarm timing repeats are not independent models. One process-first-prediction observation per condition; file caches need not be cold.\nProcess RSS sampled every 5 ms; global node states remain resident. No total RAM bound or distributed claim.\nRaw graph arrays, checkpoints and per-flow predictions are excluded.\n',encoding='utf-8')
    if draw: plot(summary,public,'Three warm timing repeats; error bars show their standard deviation')
    print(summary.to_string(index=False))
    print(json.dumps(verification,indent=2))
    return summary


def plot(summary,public,caption):
    order=[('full',0,0),('chunked',0,512)]+[('cached',m,q) for m in [4096,16384] for q in [512,2048]]
    labels=['Full','Halo\n512','4096\n512','4096\n2048','16384\n512','16384\n2048']
    colors=['#555555','#c97d30']+['#3274a1']*4
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey='row',layout='constrained')
    for col,dataset in enumerate(['ton','cse','bot']):
        d=summary[summary.dataset==dataset].set_index(['method','message_batch_size','query_batch_size'])
        d=d.reindex(order)
        for row,field in [(0,'mean_inference_s'),(1,'peak_rss_mib')]:
            ax=axes[row,col]; values=d[field].to_numpy()*(1000 if row==0 else 1)
            errors=d.sd_inference_s.to_numpy()*1000 if row==0 else None
            ax.bar(np.arange(6),values,color=colors,yerr=errors,capsize=3,width=.7)
            ax.set_xticks(np.arange(6),labels,fontsize=8)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
            if row==0: ax.set_yscale('log'); ax.set_title(dataset.upper())
            else: ax.set_ylim(0,summary.peak_rss_mib.max()*1.1)
        axes[1,col].set_xlabel('Schedule / message arcs and target flows per chunk',fontsize=8)
    axes[0,0].set_ylabel('Whole-test-graph inference (ms, log scale)')
    axes[1,0].set_ylabel('Peak process RSS (MiB)')
    fig.suptitle('Layer-wise inference on the same frozen models and test flows\nGray: Full · orange: exact halos · blue: node states computed once per prediction\n'+caption,fontsize=12)
    for extension in ['png','svg','pdf']: fig.savefig(public/('inference_comparison.'+extension),dpi=300)
    plt.close(fig)


if __name__=='__main__': build()
