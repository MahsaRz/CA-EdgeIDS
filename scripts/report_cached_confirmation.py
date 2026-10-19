"""Keep process repeats separate from trained-model repetitions."""
import json
from pathlib import Path

import pandas as pd

from report_cached_study import build,plot


def report():
    root=Path('artifacts/cached-confirmation-100000')
    if root.exists(): raise FileExistsError(root)
    all_rows=[]; inputs=None; sources=None
    for block in [1,2,3]:
        private=Path('runs/cached-confirmation-100000')/f'block-{block}'
        plan=json.loads((private/'frozen_plan.json').read_text())
        frozen_inputs=json.loads((private/'inputs.json').read_text())
        if plan['order_seed']!=20261020+block: raise ValueError('Unexpected confirmation order seed')
        if inputs is not None and inputs!=frozen_inputs: raise ValueError('Input graph or checkpoint changed between blocks')
        if sources is not None and sources!=plan['package_source_sha256']: raise ValueError('Numerical code changed between blocks')
        inputs=frozen_inputs; sources=plan['package_source_sha256']
        rows=build(private,root/f'block-{block}',draw=False)
        rows.insert(0,'process_block',block); all_rows.append(rows)
    processes=pd.concat(all_rows,ignore_index=True)
    processes.to_csv(root/'process_means.csv',index=False)
    keys=['dataset','method','message_batch_size','query_batch_size']
    summary=processes.groupby(keys,as_index=False).agg(processes=('process_block','size'),
        mean_inference_s=('mean_inference_s','mean'),sd_inference_s=('mean_inference_s','std'),
        min_process_mean_s=('mean_inference_s','min'),max_process_mean_s=('mean_inference_s','max'),
        peak_rss_mib=('peak_rss_mib','max'),mean_process_peak_rss_mib=('peak_rss_mib','mean'),
        first_prediction_process_s=('first_prediction_process_s','mean'),sd_first_prediction_process_s=('first_prediction_process_s','std'),
        mean_forward_s=('mean_forward_s','mean'),mean_packing_s=('mean_packing_s','mean'),message_arc_visits=('message_arc_visits','first'))
    for dataset in ['ton','cse','bot']:
        data=summary[summary.dataset==dataset]
        full=data[data.method=='full'].iloc[0]; halo=data[data.method=='chunked'].iloc[0]
        index=summary.dataset==dataset
        summary.loc[index,'time_relative_to_full']=summary.loc[index,'mean_inference_s']/full.mean_inference_s
        summary.loc[index,'time_relative_to_halo512']=summary.loc[index,'mean_inference_s']/halo.mean_inference_s
    summary.to_csv(root/'condition_summary.csv',index=False)
    summary.to_latex(root/'condition_summary.tex',index=False,float_format='%.6f',na_rep='--',escape=True)
    plot(summary,root,'Three fresh-process means per condition; error bars show their standard deviation')
    (root/'SCOPE.txt').write_text('Confirmation stage only; exploratory runs are kept in a separate bundle.\nThree independently started processes per condition, each with one complete warm-up and three warm whole-graph timing repeats.\nAverage within each process before computing the across-process mean and sample standard deviation.\nOne trained seed-7 model per dataset; process repetitions are not model-seed uncertainty.\nNode states recomputed for each prediction. Resident node states and original graph remain required.\nProcess-first-prediction times include imports and loading; file caches need not be cold. Source preparation is excluded.\nAll predefined chunk settings are retained; no test-label-based selection.\n',encoding='utf-8')
    failures=sum(len(json.loads((root/f'block-{block}'/'failures.json').read_text())) for block in [1,2,3])
    checks=[item for block in [1,2,3] for item in json.loads((root/f'block-{block}'/'prediction_checks.json').read_text())]
    (root/'completion.json').write_text(json.dumps(dict(process_conditions=len(processes),warm_timing_measurements=int(processes.repeats.sum()),
        failed_conditions=failures,maximum_probability_difference=max(item['maximum_probability_difference'] for item in checks),
        same_class_predictions=all(item['same_class_predictions'] for item in checks)),indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__': report()
