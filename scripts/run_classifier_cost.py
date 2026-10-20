"""Fresh-process costs for frozen GNN, MLP and Forest classifiers."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time
from hardware_context import hardware_context

ROOT=Path('runs/classifier-cost-100000'); MODEL_ROOT=Path('runs/controlled-binary-100000'); PLAN=Path('configs/classifier_cost_100000.json')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


class Memory:
    def __enter__(self):
        self.read=lambda:int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        self.baseline=self.peak=self.read(); self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.005): self.peak=max(self.peak,self.read())
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.read()); self.stop.set(); self.thread.join()


def sources():
    paths=list(Path('src/ca_edgeids').glob('*.py'))+[PLAN,Path('scripts/run_classifier_cost.py'),Path('scripts/larger_data.py'),Path('scripts/hardware_context.py')]
    return {str(p).replace('\\','/'):sha(p) for p in paths}


def worker(job,out,start):
    plan=json.loads(PLAN.read_text()); d=job['dataset']; method=job['method']; family={'gnn_full':'GNN','gnn_layerwise':'GNN','mlp':'MLP','rf':'RF'}[method]
    with Memory() as whole:
        t=time.perf_counter()
        import joblib
        import numpy as np
        import pandas as pd
        from ca_edgeids.data import split_rows
        from larger_data import transform_frame
        selected=next(r for r in json.loads((MODEL_ROOT/'selected_models.json').read_text()) if r['dataset']==d and r['family']==family and r['variant']=='full' and r['seed']==7)
        if sha(selected['model_file'])!=selected['model_sha256']: raise ValueError('Selected checkpoint changed')
        cfg=json.loads((MODEL_ROOT/d/'resolved_config.json').read_text()); original=Path('runs/linux-scaling-100000/training')/d
        prefile=original/'preprocessor.joblib'; pre=joblib.load(prefile)
        imports_and_preprocessor_s=time.perf_counter()-t
        t=time.perf_counter(); frame=pd.read_csv(cfg['csv'],usecols=['IPV4_SRC_ADDR','IPV4_DST_ADDR']+cfg['features']+['Label','Attack','source_row_id'],low_memory=False)
        csv_parse_s=time.perf_counter()-t
        t=time.perf_counter(); split=split_rows(frame,42,'conversation'); data=transform_frame(frame,cfg,split,pre)
        split_transform_and_audit_s=time.perf_counter()-t
        manifest=json.loads((MODEL_ROOT/d/'dataset_manifest.json').read_text())
        if data.manifest['csv_sha256']!=manifest['csv_sha256'] or data.manifest['test_row_ids_sha256']!=manifest['test_row_ids_sha256']: raise ValueError('Original test population changed')
        te=split=='test'; x=data.features[te]; ids=frame.loc[te,'source_row_id'].to_numpy(dtype=np.int64)
        graph_s=0.; t=time.perf_counter()
        if family=='RF':
            model=joblib.load(selected['model_file'])
            if model.classes_.tolist()!=[0,1] or model.n_jobs!=1: raise ValueError('Forest class/thread configuration changed')
            def call(): return np.concatenate([model.predict_proba(x[a:a+plan['query_batch_size']]) for a in range(0,len(x),plan['query_batch_size'])])
        else:
            import torch
            from ca_edgeids.runtime import seed_all
            seed_all(7,1)
            saved=torch.load(selected['model_file'],map_location='cpu',weights_only=False)
            if family=='MLP':
                from ca_edgeids.classifier_study import FlowMLP,predict_mlp
                model=FlowMLP(x.shape[1],cfg['hidden'],2,cfg['dropout']); model.load_state_dict(saved['state_dict']); model.eval()
                def call(): return predict_mlp(model,x,plan['query_batch_size'])
            else:
                from ca_edgeids.graph import build_graph
                from ca_edgeids.model import EdgeSAGE
                from ca_edgeids.cached import infer_cached
                from ca_edgeids.experiment import infer
                a=time.perf_counter(); g=build_graph(data,'test',True); graph_s=time.perf_counter()-a
                model=EdgeSAGE(x.shape[1],2,cfg['hidden'],cfg['layers'],cfg['dropout']); model.load_state_dict(saved['state_dict']); model.eval()
                cfg=dict(cfg,max_edges_per_batch=plan['query_batch_size']); owner=np.zeros(len(g.node_x),dtype=np.int64)
                def call():
                    if method=='gnn_layerwise': return infer_cached(model,g,cfg,plan['message_batch_size'])[0]
                    return infer(model,g,owner,cfg,full_graph=True,partition_count=1)[0]
        method_imports_model_and_graph_s=time.perf_counter()-t
        t=time.perf_counter(); probability=call(); first_inference_s=time.perf_counter()-t; first_process_s=time.perf_counter()-start
        timings=[]
        for repeat in range(plan['warm_repeats']):
            t=time.perf_counter()
            with Memory() as mem: p=call()
            elapsed=time.perf_counter()-t
            if not np.allclose(p,probability,rtol=1e-5,atol=1e-5) or not np.array_equal(p.argmax(1),probability.argmax(1)): raise ValueError('Repeated predictions changed')
            timings.append(dict(repeat=repeat,total_inference_s=elapsed,peak_rss_mib_sampled=mem.peak/2**20,baseline_rss_mib=mem.baseline/2**20))
        record=dict(**job,family=family,seed=7,imports_and_preprocessor_s=imports_and_preprocessor_s,csv_parse_s=csv_parse_s,
            split_transform_and_audit_s=split_transform_and_audit_s,graph_construction_s=graph_s,method_imports_model_and_graph_s=method_imports_model_and_graph_s,
            first_inference_s=first_inference_s,first_prediction_process_s=first_process_s,repeats=timings,test_flows=len(x),
            model_sha256=selected['model_sha256'],preprocessor_sha256=sha(prefile),source_csv_sha256=data.manifest['csv_sha256'],
            target_row_ids_sha256=data.manifest['test_row_ids_sha256'],torch_imported='torch' in sys.modules,coverage_min=1,coverage_max=1)
    record.update(setup_and_prediction_peak_rss_mib_sampled=whole.peak/2**20,initial_worker_rss_mib=whole.baseline/2**20)
    np.savez_compressed(str(out)+'.npz',row_ids=ids,probabilities=probability); dump(str(out)+'.json',record)


def execute():
    if platform.system()!='Linux': raise RuntimeError('Linux process measurements required')
    if ROOT.exists(): raise FileExistsError(ROOT)
    plan=json.loads(PLAN.read_text()); snapshot=sources()
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    for path,digest in snapshot.items():
        raw=subprocess.run(['git','show',f'{commit}:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Commit protocol and sources before timing')
    jobs=[dict(block=b,dataset=d,method=m) for b in plan['blocks'] for d in plan['datasets'] for m in plan['methods']]
    random.Random(plan['order_seed']).shuffle(jobs); ROOT.mkdir(parents=True)
    package_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','joblib']}
    dump(ROOT/'frozen_plan.json',dict(plan,source_commit=commit,source_sha256=snapshot,execution_order=jobs,
        model_root=str(MODEL_ROOT),model_selections_sha256=sha(MODEL_ROOT/'selected_models.json'),
        environment=dict(python=platform.python_version(),platform=platform.platform(),packages=package_versions,hardware=hardware_context())))
    records=[]; failures=[]
    for i,job in enumerate(jobs):
        tag=f"block-{job['block']}-{job['dataset']}-{job['method']}"; out=ROOT/tag; start=time.perf_counter()
        command=[sys.executable,'scripts/run_classifier_cost.py','--worker-job',json.dumps(job),'--out',str(out),'--process-started',str(start),'--model-root',str(MODEL_ROOT)]
        try: result=subprocess.run(command,capture_output=True,text=True,timeout=plan['worker_timeout_seconds'])
        except subprocess.TimeoutExpired:
            failures.append(dict(**job,error='Worker timeout at frozen 120-second limit')); dump(ROOT/'failures.json',failures); continue
        if result.returncode:
            failures.append(dict(**job,error=result.stderr)); dump(ROOT/'failures.json',failures)
            if 'memoryerror' not in result.stderr.lower(): raise RuntimeError(result.stderr)
            continue
        record=json.loads(Path(str(out)+'.json').read_text()); records.append(record); dump(ROOT/'condition_results.json',records)
        print(f'{i+1}/{len(jobs)} {tag}: complete',flush=True)
    if sources()!=snapshot: raise ValueError('Numerical sources changed during timing')
    dump(ROOT/'failures.json',failures); dump(ROOT/'completion.json',dict(requested=len(jobs),completed=len(records),failures=len(failures)))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--worker-job'); p.add_argument('--out'); p.add_argument('--process-started',type=float)
    p.add_argument('--run-root',default=str(ROOT)); p.add_argument('--model-root',default=str(MODEL_ROOT)); a=p.parse_args()
    ROOT=Path(a.run_root); MODEL_ROOT=Path(a.model_root)
    worker(json.loads(a.worker_job),a.out,a.process_started) if a.worker_job else execute()
