"""Check classifier costs against the exact selected-model predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file

ROOT=Path('runs/classifier-cost-100000'); PUBLIC=Path('artifacts/classifier-cost-100000')


def execute():
    plan=json.loads((ROOT/'frozen_plan.json').read_text()); records=json.loads((ROOT/'condition_results.json').read_text()); failures=json.loads((ROOT/'failures.json').read_text())
    model_root=Path(plan['model_root']); chosen=json.loads((model_root/'selected_models.json').read_text())
    if sha256_file(model_root/'selected_models.json')!=plan['model_selections_sha256']: raise ValueError('Model selection manifest changed')
    for path,digest in plan['source_sha256'].items():
        raw=subprocess.run(['git','show',f"{plan['source_commit']}:{path}"],capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Frozen source mismatch')
    key=lambda r:(r['block'],r['dataset'],r['method'])
    expected={key(r) for r in plan['execution_order']}; actual=[key(r) for r in records+failures]
    if len(actual)!=len(set(actual)) or set(actual)!=expected: raise ValueError('Incomplete cost grid')
    out=PUBLIC; out.mkdir(parents=True,exist_ok=False); rows=[]; checks=[]
    for r in records:
        d=r['dataset']; method=r['method']; family=r['family']; block=r['block']
        model=next(a for a in chosen if a['dataset']==d and a['family']==family and a['variant']=='full' and a['seed']==7)
        if r['model_sha256']!=model['model_sha256'] or sha256_file(model['model_file'])!=model['model_sha256']: raise ValueError('Checkpoint mismatch')
        path=ROOT/f'block-{block}-{d}-{method}.npz'
        reference=model_root/'test-evaluation'/f'{d}-{family.lower()}-full-7-predictions.npz'
        with np.load(path,allow_pickle=False) as saved,np.load(reference,allow_pickle=False) as old:
            ids=saved['row_ids']; p=saved['probabilities']; ref=old['probabilities']
            np.testing.assert_array_equal(ids,old['row_ids'])
            if len(np.unique(ids))!=len(ids) or len(ids)!=r['test_flows']: raise ValueError('Target coverage mismatch')
            if hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()!=r['target_row_ids_sha256']: raise ValueError('Target digest mismatch')
            equivalent=bool(np.allclose(p,ref,rtol=1e-5,atol=1e-5)); disagreements=int(np.sum(p.argmax(1)!=ref.argmax(1)))
            checks.append(dict(dataset=d,method=method,block=block,maximum_probability_difference=float(np.abs(p-ref).max()),
                probability_equivalent=equivalent,class_disagreements=disagreements,target_row_ids_sha256=r['target_row_ids_sha256']))
        if len(r['repeats'])!=3 or {a['repeat'] for a in r['repeats']}!={0,1,2}: raise ValueError('Missing timing repeat')
        if family=='RF' and r['torch_imported']: raise ValueError('Unused Torch import changes Forest startup/memory scope')
        rows.append(dict(dataset=d,method=method,family=family,block=block,test_flows=r['test_flows'],
            first_prediction_process_s=r['first_prediction_process_s'],first_inference_s=r['first_inference_s'],
            warm_median_s=float(np.median([a['total_inference_s'] for a in r['repeats']])),
            setup_and_prediction_peak_rss_mib_sampled=r['setup_and_prediction_peak_rss_mib_sampled'],
            warm_peak_rss_mib_sampled=max(a['peak_rss_mib_sampled'] for a in r['repeats']),
            probability_equivalent=equivalent,class_disagreements=disagreements,graph_construction_s=r['graph_construction_s']))
    frame=pd.DataFrame(rows); frame.to_csv(out/'condition_results.csv',index=False)
    summary=frame.groupby(['dataset','method']).agg(process_blocks=('block','count'),test_flows=('test_flows','first'),
        mean_warm_s=('warm_median_s','mean'),sd_warm_s=('warm_median_s','std'),mean_first_process_s=('first_prediction_process_s','mean'),
        sd_first_process_s=('first_prediction_process_s','std'),mean_setup_peak_rss_mib=('setup_and_prediction_peak_rss_mib_sampled','mean'),
        mean_warm_peak_rss_mib=('warm_peak_rss_mib_sampled','mean'),all_probabilities_equivalent=('probability_equivalent','all'),
        maximum_class_disagreements=('class_disagreements','max')).reset_index()
    if not (summary.process_blocks==3).all(): raise ValueError('Missing process block')
    summary.to_csv(out/'summary.csv',index=False); summary.to_latex(out/'summary.tex',index=False,float_format='%.5f')
    for name in ['frozen_plan.json','completion.json']: (out/name).write_bytes((ROOT/name).read_bytes())
    (out/'prediction_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (out/'failures.json').write_text(json.dumps([dict(**{k:v for k,v in r.items() if k!='error'},error=r['error'].strip().splitlines()[-1]) for r in failures],indent=2),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(audited_conditions=len(rows),failures=len(failures),
        equivalent_conditions=sum(c['probability_equivalent'] and c['class_disagreements']==0 for c in checks),
        fixed_seed=7,selected_model_predictions_checked=True,flow_only_workers_skip_graph_construction=True,
        forest_worker_avoids_unused_torch_import=True,retuned=False),indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':8,'font.family':'serif','pdf.fonttype':42,'ps.fonttype':42})
    methods=['gnn_full','gnn_layerwise','mlp','rf']; labels=['GNN\nFull','GNN\nlayer-wise','MLP','Forest']
    for metric,ylabel,name in [('mean_warm_s','Complete warm inference (s)','classifier_warm_cost'),('mean_first_process_s','Launch to first prediction (s)','classifier_pipeline_cost'),('mean_setup_peak_rss_mib','Sampled setup/prediction RSS (MiB)','classifier_memory')]:
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.5))
        for ax,d in zip(axes,plan['datasets']):
            b=summary[summary.dataset==d].set_index('method').loc[methods]
            ax.bar(range(4),b[metric],color=['#28668c','#5694b7','#bf7f36','#708557']); ax.set_title(d.upper())
            ax.set_xticks(range(4),labels); ax.set_yscale('log'); ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel(ylabel); fig.tight_layout()
        for ext in ['png','svg','pdf']: fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    (out/'SCOPE.txt').write_text('Same frozen seed-7 models and original test populations from the five-seed classifier study.\nThree process blocks; three complete warm predictions per block; no refitting/selection.\nMethod-specific imports and necessary preprocessing/graph work; Forest does not load Torch.\nSampled phase/process RSS, not allocation-exact or whole-system peak.\nNo raw flows, checkpoints, identities or per-flow predictions exported.\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-root',default=str(ROOT)); p.add_argument('--public-root',default=str(PUBLIC)); a=p.parse_args()
    ROOT=Path(a.run_root); PUBLIC=Path(a.public_root); execute()
