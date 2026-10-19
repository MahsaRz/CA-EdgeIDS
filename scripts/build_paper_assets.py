"""Build compact paper assets from verified aggregate evidence only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump

OUT=Path('artifacts/paper-assets')
DATASETS=['ton','cse','bot']


def table(name,columns,rows):
    frame=pd.DataFrame(rows,columns=columns); frame.to_csv(OUT/(name+'.csv'),index=False)
    frame.to_latex(OUT/(name+'.tex'),index=False,escape=False,column_format='l'+'r'*(len(columns)-1))


def figure(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=300)
    plt.close(fig)


def execute(complete=False):
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=[Path('artifacts/controlled-binary-100000/test-evaluation/verification.json')]
    if complete:
        evidence += [Path(f'artifacts/{s}/test-evaluation/verification.json') for s in ['sampler-binary-100000','novel-host-1000000']]
        evidence += [Path('artifacts/larger-inference/verification.json')]
    if not all(p.exists() for p in evidence): raise ValueError('Required verified evidence is incomplete')
    plt.rcParams.update({'font.size':8,'font.family':'serif','pdf.fonttype':42,'ps.fonttype':42})
    rows=[]; support=[]
    for d in DATASETS:
        source=json.loads(Path(f'docs/provenance/nf-{d}-v2.json').read_text())
        manifest=json.loads(Path(f'artifacts/controlled-binary-100000/selection/{d}/dataset_manifest.json').read_text())
        counts=manifest['class_counts']
        rows.append([d.upper(),f"{source['source_rows']:,}",f"{source['unique_ip_nodes']:,}",
            *[f"{sum(counts[p].values()):,}" for p in ['train','validation','test']]])
        for part in ['train','validation','test']: support.append([d.upper(),part,*[f'{counts[part][c]:,}' for c in ['Benign','Attack']]])
    table('table_A_source_and_splits',['Dataset','Source flows','Source IPs','Train','Validation','Test'],rows)
    table('table_A_binary_support',['Dataset','Split','Benign','Attack'],support)
    # Five-seed families: use the frozen full model, never select a seed/control by test.
    study=pd.read_csv('artifacts/controlled-binary-100000/test-evaluation/model_results.csv')
    a=study[study.variant=='full']; rows=[]
    for d in DATASETS:
        for family in ['GNN','MLP','RF']:
            b=a[(a.dataset==d)&(a.family==family)]
            rows.append([d.upper(),family,*[f'{b[c].mean():.4f} $\\pm$ {b[c].std(ddof=1):.4f}' for c in ['macro_f1','weighted_f1','attack_fpr','attack_recall','attack_ap']]])
    table('table_classifier_quality',['Dataset','Model','Macro $F_1$','Weighted $F_1$','FPR','Attack recall','Attack AP'],rows)
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.7),sharey=True)
    for ax,d in zip(axes,DATASETS):
        for i,f in enumerate(['GNN','MLP','RF']):
            b=a[(a.dataset==d)&(a.family==f)].macro_f1
            ax.scatter(i+np.linspace(-.12,.12,len(b)),b,s=10,color='#28668c')
            ax.errorbar(i,b.mean(),yerr=b.std(ddof=1),color='black',capsize=3,linewidth=1)
        ax.set_title(d.upper()); ax.set_xticks(range(3),['Edge GNN','MLP','Forest']); ax.set_ylim(0,1.02); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Test macro $F_1$'); fig.tight_layout(); figure(fig,'five_seed_classifiers')
    # Complete per-class report means; support is identical across seeds.
    per=[]
    for d in DATASETS:
        for family in ['gnn','mlp','rf']:
            files=[Path(f'artifacts/controlled-binary-100000/test-evaluation/{d}-{family}-full-{s}-metrics.json') for s in [7,19,31,43,59]]
            reports=[json.loads(p.read_text())['per_class'] for p in files]
            for label in ['Benign','Attack','macro avg','weighted avg']:
                values={c:np.array([r[label][c] for r in reports]) for c in ['precision','recall','f1-score','support']}
                if len(set(values['support']))!=1: raise ValueError('Per-class support changed across seeds')
                per.append([d.upper(),family.upper(),label,*[f'{values[c].mean():.4f} $\\pm$ {values[c].std(ddof=1):.4f}' for c in ['precision','recall','f1-score']],str(int(values['support'][0]))])
    table('table_E_per_class',['Dataset','Model','Class','Precision','Recall','$F_1$','Support'],per)
    # Fixed-model partition counts from the original Linux pilot, batch size 512.
    rows=[]; systems=[]
    for d in DATASETS:
        b=pd.read_csv(f'artifacts/linux-scaling-100000/batch-study/{d}/timing_repeats.csv')
        b=b[(b.method=='full')|(b.batch_size==512)].copy()
        old=pd.read_csv(f'artifacts/linux-scaling-100000/training/{d}/benchmark_runs.csv').iloc[0]
        for column in ['test_n','macro_f1','weighted_f1','macro_precision','macro_recall']: b[column]=old[column]
        for (method,parts),group in b.groupby(['method','partitions']):
            record=dict(dataset=d,method=method,partitions=parts,test_n=int(group.test_n.iloc[0]),macro_f1=group.macro_f1.iloc[0],
                weighted_f1=group.weighted_f1.iloc[0],macro_precision=group.macro_precision.iloc[0],macro_recall=group.macro_recall.iloc[0],
                part_forward_ms=group.average_partition_forward_s.mean()*1000,total_ms=group.total_inference_s.mean()*1000,rss_mib=group.peak_cpu_rss_mb_sampled.max())
            systems.append(record)
            rows.append([d.upper(),method,str(parts),str(record['test_n']),f"{record['macro_f1']:.4f}",f"{record['weighted_f1']:.4f}",
                f"{record['macro_precision']:.4f}",f"{record['macro_recall']:.4f}",f"{record['part_forward_ms']:.2f}",f"{record['total_ms']:.2f}",f"{record['rss_mib']:.1f}"])
    table('table_C_partition_counts',['Dataset','Method','$P$','Test $n$','Macro $F_1$','Weighted $F_1$','Macro P','Macro R','Part. forward ms','Total ms','RSS MiB'],rows)
    systems=pd.DataFrame(systems)
    for kind in ['quality','time']:
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.9))
        for ax,d in zip(axes,DATASETS):
            for method,style in [('random','--'),('metis','-')]:
                b=systems[(systems.dataset==d)&(systems.method==method)].sort_values('partitions')
                if kind=='quality':
                    ax.plot(b.partitions,b.macro_f1,marker='o',markersize=3,linestyle=style,label=method+' macro')
                    ax.plot(b.partitions,b.weighted_f1,marker='s',markersize=3,linestyle=style,label=method+' weighted')
                else:
                    ax.plot(b.partitions,b.total_ms,marker='o',markersize=3,linestyle=style,label=method+' total')
                    ax.plot(b.partitions,b.part_forward_ms,marker='s',markersize=3,linestyle=style,label=method+' part. forward')
            ax.set_title(d.upper()); ax.set_xlabel('Requested partitions'); ax.set_xticks([2,4,8]); ax.grid(alpha=.2)
            if kind=='quality': ax.set_ylim(0,1.02)
            else: ax.set_yscale('log')
        axes[0].set_ylabel('$F_1$' if kind=='quality' else 'Time (ms)')
        handles,names=axes[0].get_legend_handles_labels(); fig.legend(handles,names,loc='upper center',ncol=4,fontsize=7)
        fig.tight_layout(rect=(0,0,1,.89)); figure(fig,'partition_'+kind)
    # A pipeline diagram keeps labels away from inference inputs.
    fig,ax=plt.subplots(figsize=(7.1,2.6)); ax.set_xlim(0,4); ax.set_ylim(0,3); ax.axis('off')
    boxes=[['Audit original\nflow records','Hash unordered\nIP pairs','Fit preprocessing\non training only','Transform each\nsplit separately'],
        ['Training graph\nlabels on flows','Full or sampled\ntraining context','Validation macro F1\ncheckpoint choice','Frozen model\nand preprocessing'],
        ['Unlabeled test\nflow graph','Target ownership\nFull / Random / METIS','Exact context or\nlayer-wise inference','One prediction per flow\nthen evaluate labels']]
    for row,texts in enumerate(boxes):
        y=2.65-row
        for col,label in enumerate(texts):
            x=col+.05
            ax.add_patch(FancyBboxPatch((x,y-.28),.88,.56,boxstyle='round,pad=.025',facecolor='#eef4f8',edgecolor='#28668c',linewidth=.8))
            ax.text(x+.44,y,label,ha='center',va='center',fontsize=7)
            if col<3: ax.annotate('',xy=(col+1.02,y),xytext=(col+.96,y),arrowprops=dict(arrowstyle='->',linewidth=.8))
    fig.tight_layout(pad=.2); figure(fig,'pipeline')
    metadata=dict(scope='verified binary research evidence; pilot systems checkpoint distinct from five-seed classifier fits',
        complete_evidence=complete,evidence_sha256={str(p).replace('\\','/'):sha256_file(p) for p in evidence},
        no_test_selection=True,plot_width_inches=7.1,font_points=8,
        tables='A: original source and split counts; C: historical fixed-model batch-512 partition sweep; E: per-class five-seed reports',
        limitations='five seeds on reused development split; one-seed historical partition sweep; unsupported multiclass/causal/distributed claims excluded')
    dump(OUT/'asset_manifest.json',metadata)
    print(OUT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--complete',action='store_true'); a=p.parse_args(); execute(a.complete)
