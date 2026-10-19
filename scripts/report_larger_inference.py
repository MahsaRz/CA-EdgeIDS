"""Verify every larger workload, including rejected resource conditions."""
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
from ca_edgeids.experiment import dump

ROOT=Path('runs/larger-inference'); PUBLIC=Path('artifacts/larger-inference')


def execute():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); complete=json.loads((ROOT/'completion.json').read_text())
    results=json.loads((ROOT/'condition_results.json').read_text()); failures=json.loads((ROOT/'failures.json').read_text())
    for path,digest in plan['source_sha256'].items():
        content=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Source commit mismatch')
    key=lambda r:(r['block'],r['dataset'],r['sample_size'],r['method'])
    expected={key(r) for r in plan['execution_order']}; actual=[key(r) for r in results+failures]
    if len(actual)!=len(set(actual)) or set(actual)!=expected: raise ValueError('Missing or duplicate condition')
    if complete!=dict(requested=len(expected),completed=len(results),failures=len(failures)): raise ValueError('Completion counts differ')
    PUBLIC.mkdir(parents=True,exist_ok=False); rows=[]; checks=[]; inputs={}
    for d in plan['datasets']:
        original=Path('runs/linux-scaling-100000/training')/d
        inputs[d]=dict(checkpoint_sha256=sha256_file(original/'seed-7/model.pt'),preprocessor_sha256=sha256_file(original/'preprocessor.joblib'))
        earlier=json.loads(Path(f'runs/linux-scaling-100000/batch-study/{d}/study_plan.json').read_text())
        if earlier['checkpoint_sha256']!=inputs[d]['checkpoint_sha256']: raise ValueError('Systems checkpoint differs from its own prior baseline study')
        for n in plan['sample_sizes']:
            group=[r for r in results if r['dataset']==d and r['sample_size']==n]
            if not group: continue
            if len({r['csv_sha256'] for r in group})!=1: raise ValueError('Different source CSVs within one workload')
            anchor=next((r for r in group if r['method']=='full'),next((r for r in group if r['method']=='cached'),None))
            if anchor is None: raise ValueError('No independent full or layer-wise reference')
            def path(r): return ROOT/f"block-{r['block']}-{d}-{n}-{r['method']}.npz"
            with np.load(path(anchor),allow_pickle=False) as saved: reference=saved['probabilities'].copy(); ids=saved['row_ids'].copy()
            digest=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
            if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate target')
            methods=set()
            for r in group:
                methods.add(r['method'])
                if any(r[name]!=value for name,value in inputs[d].items()): raise ValueError('Checkpoint/preprocessor changed across sizes')
                if r['target_row_ids_sha256']!=digest: raise ValueError('Test target digest changed')
                with np.load(path(r),allow_pickle=False) as saved:
                    p=saved['probabilities']; np.testing.assert_array_equal(saved['row_ids'],ids)
                    if p.shape!=(len(ids),2) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid probability outputs')
                    probability_equivalent=bool(np.allclose(p,reference,rtol=1e-5,atol=1e-5))
                    disagreements=int(np.sum(p.argmax(1)!=reference.argmax(1)))
                    delta=float(np.abs(p-reference).max())
                if len(r['repeats'])!=3 or {a['repeat'] for a in r['repeats']}!={0,1,2}: raise ValueError('Missing warm repeat')
                if any(a['coverage_min']!=1 or a['coverage_max']!=1 for a in r['repeats']): raise ValueError('Target coverage changed')
                checks.append(dict(block=r['block'],dataset=d,sample_size=n,method=r['method'],anchor_method=anchor['method'],
                    anchor_block=anchor['block'],maximum_probability_difference=delta,probability_equivalent=probability_equivalent,
                    class_disagreements=disagreements,same_class_predictions=disagreements==0,target_row_ids_sha256=digest))
                flat={k:v for k,v in r.items() if k!='repeats'}
                flat.update(probability_equivalent=probability_equivalent,class_disagreements=disagreements,maximum_probability_difference=delta)
                flat.update(warm_median_s=float(np.median([a['total_inference_s'] for a in r['repeats']])),
                    warm_min_s=min(a['total_inference_s'] for a in r['repeats']),warm_max_s=max(a['total_inference_s'] for a in r['repeats']))
                rows.append(flat)
            if len(methods)<2: raise ValueError('No second implementation for workload verification')
    frame=pd.DataFrame(rows); frame.to_csv(PUBLIC/'condition_results.csv',index=False)
    summary=frame.groupby(['dataset','sample_size','method']).agg(blocks=('block','count'),test_flows=('test_flows','first'),test_nodes=('test_nodes','first'),
        transformed_width=('transformed_width','first'),mean_first_process_s=('first_prediction_process_s','mean'),sd_first_process_s=('first_prediction_process_s','std'),
        mean_warm_s=('warm_median_s','mean'),sd_warm_s=('warm_median_s','std'),mean_whole_peak_rss_mb=('whole_process_peak_rss_mb_sampled','mean'),
        all_probabilities_equivalent=('probability_equivalent','all'),maximum_class_disagreements=('class_disagreements','max'),
        maximum_probability_difference=('maximum_probability_difference','max')).reset_index()
    summary.to_csv(PUBLIC/'summary.csv',index=False); summary.to_latex(PUBLIC/'summary.tex',index=False,float_format='%.4f')
    for name in ['frozen_plan.json','completion.json']: shutil.copyfile(ROOT/name,PUBLIC/name)
    # Keep complete diagnostics privately; publish errors without workstation paths.
    dump(PUBLIC/'failures.json',[dict(**{k:v for k,v in r.items() if k!='error'},error=r['error'].strip().splitlines()[-1],
        private_diagnostic_sha256=hashlib.sha256(r['error'].encode()).hexdigest()) for r in failures])
    dump(PUBLIC/'prediction_checks.json',checks); dump(PUBLIC/'inputs.json',inputs)
    dump(PUBLIC/'verification.json',dict(requested=len(expected),audited_conditions=len(checks),
        equivalent_conditions=sum(c['probability_equivalent'] and c['same_class_predictions'] for c in checks),
        numerical_mismatch_conditions=sum(not(c['probability_equivalent'] and c['same_class_predictions']) for c in checks),
        failures=len(failures),same_frozen_models_across_sizes=True,
        uncertainty='sample SD across three process-block medians; not three independent model fits or a population confidence interval'))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    labels={'full':'Full','random':'Random exact','metis':'METIS exact','cached':'Layer-wise','factored':'Factored mean'}
    for field,ylabel,name in [('mean_warm_s','Complete warm inference (s)','warm_scaling'),('mean_first_process_s','Process launch to first prediction (s)','pipeline_scaling'),('mean_whole_peak_rss_mb','Sampled process peak RSS (MiB)','memory_scaling')]:
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.6))
        for ax,d in zip(axes,plan['datasets']):
            for m in plan['methods']:
                a=summary[(summary.dataset==d)&(summary.method==m)].sort_values('sample_size')
                if len(a): ax.plot(a.sample_size/1e6,a[field],marker='o',markersize=3,label=labels[m],linewidth=1)
            ax.set_title(d.upper()); ax.set_xlabel('Uniform sample flows (million)'); ax.set_yscale('log'); ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel); handles,names=axes[0].get_legend_handles_labels()
        fig.legend(handles,names,loc='upper center',ncol=5,fontsize=7); fig.tight_layout(rect=(0,0,1,.9))
        for ext in ['png','svg','pdf']: fig.savefig(PUBLIC/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    (PUBLIC/'SCOPE.txt').write_text('Fixed original pilot preprocessor/checkpoint across three flow sample sizes.\nFresh process per condition; warm medians use three complete predictions.\nWhole-pipeline first prediction includes parse/transform/audits/graph/partition/load/inference/imports.\nKnown resource failures retained. No raw data, model files, identities or per-flow predictions exported.\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-root',default=str(ROOT)); p.add_argument('--public-root',default=str(PUBLIC)); a=p.parse_args()
    ROOT=Path(a.run_root); PUBLIC=Path(a.public_root); execute()
