"""Report AP and trapezoidal PR area separately for every frozen model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,auc,precision_recall_curve

from ca_edgeids.data import sha256_file


def execute(output,require_novel=False,verification_root='artifacts',run_base='runs'):
    studies=[('controlled','controlled-binary-100000'),('sampler','sampler-binary-100000')]
    if require_novel: studies.append(('novel-host','novel-host-1000000'))
    out=Path(output); out.mkdir(parents=True,exist_ok=False); rows=[]; sources={}
    for name,folder in studies:
        verified=Path(verification_root)/folder/'test-evaluation/verification.json'
        if not verified.exists(): raise ValueError('Model/target verification must finish first')
        sources[str(verified)]=sha256_file(verified)
        root=Path(run_base)/folder; selected=json.loads((root/'selected_models.json').read_text())
        verification=json.loads(verified.read_text())
        digest=verification.get('selection_sha256')
        if digest is None and name=='controlled':
            gate=verified.with_name('evaluation_gate.json'); digest=json.loads(gate.read_text())['selection_sha256']; sources[str(gate)]=sha256_file(gate)
        if digest!=sha256_file(root/'selected_models.json'): raise ValueError('Verification belongs to a different selection')
        for r in selected:
            d=r['dataset']; seed=r['seed']
            if name=='sampler': method=r['method']; family='GNN'; tag=f'{d}-{method}-{seed}'
            elif name=='controlled': family=r['family']; method=r['variant']; tag=f'{d}-{family.lower()}-{method}-{seed}'
            else: family=r['family']; method='full'; tag=f'{d}-{family.lower()}-{seed}'
            if sha256_file(r['model_file'])!=r['model_sha256']: raise ValueError('Selected checkpoint changed')
            path=root/'test-evaluation'/(tag+'-predictions.npz')
            with np.load(path,allow_pickle=False) as saved:
                y=saved['y']; p=saved['probabilities']; ids=saved['row_ids']
                if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate target')
                metrics=json.loads(path.with_name(tag+'-metrics.json').read_text())
                for i,label in enumerate(['Benign','Attack']):
                    truth=(y==i).astype(int)
                    if not 0<truth.sum()<len(truth): raise ValueError('Both binary classes required')
                    precision,recall,_=precision_recall_curve(truth,p[:,i])
                    ap=float(average_precision_score(truth,p[:,i])); area=float(auc(recall,precision))
                    if not np.isclose(ap,metrics['average_precision_ovr'][label],rtol=0,atol=1e-12): raise ValueError('Original AP disagrees with saved predictions')
                    rows.append(dict(study=name,dataset=d,family=family,method=method,seed=seed,label=label,
                        support=int(truth.sum()),test_flows=len(y),average_precision=ap,trapezoidal_pr_auc=area,
                        target_row_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()))
    frame=pd.DataFrame(rows); frame.to_csv(out/'per_model_class_areas.csv',index=False)
    summary=frame.groupby(['study','dataset','family','method','label']).agg(seeds=('seed','count'),support=('support','first'),
        mean_ap=('average_precision','mean'),sd_ap=('average_precision','std'),mean_trapezoidal_pr_auc=('trapezoidal_pr_auc','mean'),
        sd_trapezoidal_pr_auc=('trapezoidal_pr_auc','std')).reset_index()
    if not (summary.seeds==5).all(): raise ValueError('Incomplete seed group')
    summary.to_csv(out/'class_summary.csv',index=False); summary.to_latex(out/'class_summary.tex',index=False,float_format='%.5f')
    macro=frame.groupby(['study','dataset','family','method','seed']).agg(macro_ap=('average_precision','mean'),
        macro_trapezoidal_pr_auc=('trapezoidal_pr_auc','mean')).reset_index()
    macro.to_csv(out/'per_model_macro_areas.csv',index=False)
    (out/'definitions.json').write_text(json.dumps(dict(source_verification_sha256=sources,model_class_rows=len(rows),
        ap='sklearn non-interpolated average precision',pr_auc='sklearn auc(recall,precision) on precision_recall_curve; trapezoidal interpolation',
        averaging='one-versus-rest area for each binary class; unweighted mean over both classes for macro area; five seed means/SD',
        decisions='no threshold selection, refitting or checkpoint changes'),indent=2),encoding='utf-8')
    print(f'{len(rows)//2} frozen models; AP and trapezoidal PR-AUC separately verified/exported.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='artifacts/precision-recall-areas'); p.add_argument('--require-novel',action='store_true')
    p.add_argument('--verification-root',default='artifacts'); p.add_argument('--run-base',default='runs'); a=p.parse_args()
    execute(a.out,a.require_novel,a.verification_root,a.run_base)
