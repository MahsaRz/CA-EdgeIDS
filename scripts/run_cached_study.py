"""Frozen-model comparison of layer-wise inference with existing controls."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump
from ca_edgeids.runtime import environment
from ca_edgeids.worker import load_graph


def execute(output='runs/cached-inference-100000',order_seed=20261020):
    if platform.system()!='Linux': raise RuntimeError('Use the recorded Linux environment')
    out=Path(output)
    if out.exists(): raise FileExistsError(out)
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    sources={p.name:sha256_file(p) for p in sorted(Path('src/ca_edgeids').glob('*.py'))}
    # Ensure the experiment uses committed package and driver bytes.
    for path,digest in [(f'src/ca_edgeids/{name}',sha) for name,sha in sources.items()]+[('scripts/run_cached_study.py',sha256_file(__file__))]:
        content=subprocess.run(['git','show',f'{commit}:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Commit experimental sources before running')
    jobs=[dict(dataset=d,method=method,message_batch_size=message,query_batch_size=query)
          for d in ['ton','cse','bot']
          for method,message,query in [('full',0,0),('chunked',0,512)]+
          [('cached',message,query) for message in [4096,16384] for query in [512,2048]]]
    np.random.default_rng(order_seed).shuffle(jobs)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'frozen_plan.json',dict(git_commit=commit,package_source_sha256=sources,
        driver_sha256=sha256_file(__file__),environment=environment(),threads=1,
        execution_order=jobs,timing_repeats=3,order_seed=order_seed,
        cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        logical_cpus=os.cpu_count(),linux_memory_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))),
        scope='same saved seed-7 binary models and 100000-flow sample test populations; no fitting or tuning',
        cache='recomputed for every full prediction including each timing repeat',
        timings='fresh worker per condition; one complete warm-up followed by three complete predictions; warm total includes encoding and decoding; source preparation and graph conversion excluded; first-process time has one observation and OS cache need not be cold',
        storage='global resident node states; bounded message and query chunks; no claim of bounded total RAM'))
    rows=[]; checks=[]; failures=[]; inputs={}
    for index,job in enumerate(jobs):
        dataset=job['dataset']; method=job['method']
        run=Path('runs/linux-scaling-100000/training')/dataset
        graph=run/'test_graph'; checkpoint=run/'seed-7/model.pt'; anchor=run/'seed-7/full-1-predictions.npz'
        g=load_graph(graph)
        row_hash=hashlib.sha256(g.row_ids.astype('<i8').tobytes()).hexdigest()
        manifest=json.loads((run/'dataset_manifest.json').read_text())
        if row_hash!=manifest['test_row_ids_sha256']: raise ValueError('Test rows changed')
        with np.load(anchor,allow_pickle=False) as saved:
            if not np.array_equal(saved['row_ids'],g.row_ids): raise ValueError('Reference row order changed')
            reference=saved['probabilities'].copy()
        identity=dict(order_index=index,**job)
        entry=dict(checkpoint_sha256=sha256_file(checkpoint),reference_predictions_sha256=sha256_file(anchor),
                   target_row_ids_sha256=row_hash,test_flows=len(g.src),test_nodes=len(g.node_x),message_arcs=len(g.message_src),
                   graph_array_sha256={p.name:sha256_file(p) for p in sorted(graph.glob('*.npy'))})
        if dataset in inputs and inputs[dataset]!=entry: raise ValueError('Frozen inputs changed during study')
        inputs[dataset]=entry; dump(out/'inputs.json',inputs)
        tag=f"{dataset}-{method}-m{job['message_batch_size']}-q{job['query_batch_size']}"
        command=[sys.executable,'-m','ca_edgeids.worker','--graph',str(graph.resolve()),'--checkpoint',str(checkpoint.resolve()),
                 '--method',method,'--parts','1','--repeats','3','--out',str((out/tag).resolve())]
        if job['query_batch_size']: command+=['--batch-size',str(job['query_batch_size'])]
        if job['message_batch_size']: command+=['--message-batch-size',str(job['message_batch_size'])]
        started=time.perf_counter(); command+=['--process-started-at',str(started)]
        try: result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired:
            failures.append(dict(identity,error='Worker timeout after 120 s')); dump(out/'failures.json',failures); continue
        wall=time.perf_counter()-started
        if result.returncode:
            failures.append(dict(identity,error=result.stderr)); dump(out/'failures.json',failures)
            if not any(term in result.stderr.lower() for term in ['memoryerror','out of memory','tensor limit exceeded']): raise RuntimeError(result.stderr)
            continue
        with np.load(out/(tag+'.npz'),allow_pickle=False) as saved: probabilities=saved['probabilities']
        if not np.allclose(probabilities,reference,atol=1e-5,rtol=1e-5) or not np.array_equal(probabilities.argmax(1),reference.argmax(1)):
            raise AssertionError('Frozen-model predictions changed')
        records=json.loads((out/(tag+'.json')).read_text())
        if len(records)!=3 or {r['repeat'] for r in records}!={0,1,2}: raise AssertionError('Missing repeat')
        for r in records:
            if r['target_row_ids_sha256']!=row_hash or r['coverage_min']!=1 or r['coverage_max']!=1: raise AssertionError('Coverage changed')
            if not 0<r['first_prediction_process_s']<=wall: raise AssertionError('Invalid process timing')
            if method=='cached' and r['message_arc_visits']!=len(g.message_src)*2: raise AssertionError('Repeated message work')
            rows.append(dict(identity,worker_process_wall_s=wall,**r))
        checks.append(dict(identity,maximum_probability_difference=float(np.abs(probabilities-reference).max()),same_class_predictions=True,
                           target_row_ids_sha256=row_hash))
        pd.DataFrame(rows).to_csv(out/'timing_repeats.csv',index=False)
        dump(out/'prediction_checks.json',checks)
        print(f'{tag}: checked',flush=True)
    dump(out/'failures.json',failures)
    if len(checks)+len(failures)!=len(jobs): raise AssertionError('Missing conditions')
    # Check the numerical implementation did not change during execution.
    if sources!={p.name:sha256_file(p) for p in sorted(Path('src/ca_edgeids').glob('*.py'))}: raise AssertionError('Sources changed')
    dump(out/'completion.json',dict(requested_conditions=len(jobs),completed_conditions=len(checks),failed_conditions=len(failures),timing_measurements=len(rows)))
    print(out,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/cached-inference-100000')
    parser.add_argument('--order-seed',type=int,default=20261020)
    args=parser.parse_args(); execute(args.out,args.order_seed)
