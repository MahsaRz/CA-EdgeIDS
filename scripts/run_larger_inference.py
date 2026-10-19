"""Fixed-model scaling with fresh-process end-to-end and warm measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

from ca_edgeids.cached import infer_cached
from ca_edgeids.controlled_training import predict
from ca_edgeids.data import split_rows,sha256_file
from ca_edgeids.experiment import infer,dump
from ca_edgeids.graph import build_graph,partition_nodes
from ca_edgeids.model import EdgeSAGE
from ca_edgeids.runtime import PeakRSS,environment,seed_all
from ca_edgeids.statistics_backend import summarize
from larger_data import transform_frame
from hardware_context import hardware_context

ROOT=Path('runs/larger-inference'); PLAN=Path('configs/larger_inference.json')


def sources():
    paths=list(Path('src/ca_edgeids').glob('*.py'))+[PLAN,Path('scripts/run_larger_inference.py'),Path('scripts/larger_data.py'),Path('scripts/hardware_context.py')]
    return {str(p).replace('\\','/'):sha256_file(p) for p in paths}


def worker(job,out,start):
    plan=json.loads(PLAN.read_text()); d=job['dataset']; method=job['method']; size=job['sample_size']
    original=Path('runs/linux-scaling-100000/training')/d
    cfg=json.loads((original/'resolved_config.json').read_text()); seed_all(7,1)
    checkpoint=original/'seed-7/model.pt'; preprocessing=original/'preprocessor.joblib'
    input_path=Path(cfg['csv']) if size==100000 else Path(f'data/larger-uniform/{d}/{d}-uniform-{size}.csv')
    if input_path.stat().st_size>plan['input_limit_bytes']: raise MemoryError('Input exceeds the frozen CSV limit')
    with PeakRSS() as whole:
        t=time.perf_counter()
        frame=pd.read_csv(input_path,usecols=['IPV4_SRC_ADDR','IPV4_DST_ADDR']+cfg['features']+['Label','Attack','source_row_id'],low_memory=False)
        csv_parse_s=time.perf_counter()-t
        if len(frame)!=size: raise ValueError('Sample size changed')
        t=time.perf_counter(); pre=joblib.load(preprocessing); preprocessor_load_s=time.perf_counter()-t
        t=time.perf_counter(); split=split_rows(frame,42,'conversation'); split_s=time.perf_counter()-t
        cfg.update(csv=str(input_path),max_edges_per_batch=plan['query_batch_size'],max_batch_tensor_bytes=plan['tensor_screen_bytes'])
        t=time.perf_counter(); data=transform_frame(frame,cfg,split,pre); transform_and_audit_s=time.perf_counter()-t
        if size==100000:
            expected_csv=json.loads((original/'dataset_manifest.json').read_text())['csv_sha256']
        else:
            audit=json.loads(Path(f'artifacts/larger-uniform-source-audits/{d}.json').read_text())
            expected_csv=next(r['csv_sha256'] for r in audit['samples'] if r['rows']==size)
        if data.manifest['csv_sha256']!=expected_csv: raise ValueError('Workload differs from its verified sample')
        t=time.perf_counter(); g=build_graph(data,'test',True); graph_s=time.perf_counter()-t
        t=time.perf_counter(); saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=EdgeSAGE(g.edge_x.shape[1],2,cfg['hidden'],cfg['layers'],cfg['dropout']); model.load_state_dict(saved['state_dict']); model.eval()
        checkpoint_and_model_s=time.perf_counter()-t
        t=time.perf_counter()
        owner=partition_nodes(g,method,4,7) if method in ['random','metis'] else np.zeros(len(g.node_x),dtype=np.int64)
        partition_s=time.perf_counter()-t
        def call():
            if method=='cached': return infer_cached(model,g,cfg,plan['message_batch_size'])
            if method=='factored':
                a=time.perf_counter()
                with PeakRSS() as mem:
                    summary=summarize(g,max_bytes=plan['tensor_screen_bytes'])
                    probabilities=predict(model,g,summary,'full',batch_size=plan['query_batch_size'])
                return probabilities,dict(total_inference_s=time.perf_counter()-a,peak_cpu_rss_mb_sampled=mem.peak/2**20,
                    baseline_cpu_rss_mb=mem.baseline/2**20,coverage_min=1,coverage_max=1)
            return infer(model,g,owner,cfg,full_graph=method=='full',partition_count=4 if method in ['random','metis'] else 1)
        t=time.perf_counter(); probability,first=call(); first_inference_s=time.perf_counter()-t
        first_process_s=time.perf_counter()-start
        repeats=[]
        for repeat in range(plan['warm_repeats']):
            p,timing=call()
            if not np.allclose(p,probability,rtol=1e-5,atol=1e-5) or not np.array_equal(p.argmax(1),probability.argmax(1)): raise AssertionError('Repeated predictions changed')
            repeats.append(dict(repeat=repeat,**timing))
        record=dict(**job,csv_parse_s=csv_parse_s,preprocessor_load_s=preprocessor_load_s,split_s=split_s,
            transform_and_audit_s=transform_and_audit_s,graph_construction_s=graph_s,checkpoint_and_model_s=checkpoint_and_model_s,
            partitioning_s=partition_s,first_inference_s=first_inference_s,first_prediction_process_s=first_process_s,
            test_flows=len(g.src),test_nodes=len(g.node_x),message_arcs=len(g.message_src),transformed_width=g.edge_x.shape[1],
            checkpoint_sha256=sha256_file(checkpoint),preprocessor_sha256=sha256_file(preprocessing),csv_sha256=data.manifest['csv_sha256'],
            target_row_ids_sha256=data.manifest['test_row_ids_sha256'],repeats=repeats)
    record.update(whole_process_peak_rss_mb_sampled=whole.peak/2**20,whole_process_baseline_rss_mb=whole.baseline/2**20)
    np.savez_compressed(str(out)+'.npz',row_ids=g.row_ids,probabilities=probability)
    dump(str(out)+'.json',record)


def execute():
    if ROOT.exists(): raise FileExistsError(ROOT)
    plan=json.loads(PLAN.read_text()); snapshot=sources()
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    for path,digest in snapshot.items():
        content=subprocess.run(['git','show',f'{commit}:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Commit the protocol before timing')
    jobs=[dict(block=b,dataset=d,sample_size=n,method=m) for b in plan['blocks'] for d in plan['datasets'] for n in plan['sample_sizes'] for m in plan['methods']]
    np.random.default_rng(plan['order_seed']).shuffle(jobs)
    ROOT.mkdir(parents=True); dump(ROOT/'frozen_plan.json',dict(plan,source_commit=commit,source_sha256=snapshot,environment=environment(),hardware=hardware_context(),execution_order=jobs))
    failures=[]; records=[]
    for index,job in enumerate(jobs):
        tag=f"block-{job['block']}-{job['dataset']}-{job['sample_size']}-{job['method']}"; dest=ROOT/tag
        started=time.perf_counter()
        command=[sys.executable,'scripts/run_larger_inference.py','--worker-job',json.dumps(job),'--out',str(dest),'--process-started',str(started)]
        try: result=subprocess.run(command,capture_output=True,text=True,timeout=plan['worker_timeout_seconds'])
        except subprocess.TimeoutExpired:
            failures.append(dict(**job,error='worker timeout at the frozen 300-second limit'))
            dump(ROOT/'failures.json',failures); continue
        if result.returncode:
            failures.append(dict(**job,error=result.stderr)); dump(ROOT/'failures.json',failures)
            if not any(s in result.stderr.lower() for s in ['memoryerror','out of memory','requested 4 partitions','tensor limit']): raise RuntimeError(result.stderr)
            print(f'{index+1}/{len(jobs)} {tag}: resource/infeasibility guard retained',flush=True); continue
        record=json.loads(Path(str(dest)+'.json').read_text()); records.append(record)
        dump(ROOT/'condition_results.json',records)
        print(f'{index+1}/{len(jobs)} {tag}: {record["first_prediction_process_s"]:.2f} s through first prediction',flush=True)
    if sources()!=snapshot: raise ValueError('Numerical sources changed while timing')
    dump(ROOT/'failures.json',failures); dump(ROOT/'completion.json',dict(requested=len(jobs),completed=len(records),failures=len(failures)))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--worker-job'); p.add_argument('--out'); p.add_argument('--process-started',type=float)
    p.add_argument('--run-root',default=str(ROOT)); a=p.parse_args(); ROOT=Path(a.run_root)
    worker(json.loads(a.worker_job),a.out,a.process_started) if a.worker_job else execute()
