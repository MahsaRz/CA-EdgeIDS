"""Build compact paper assets from verified aggregate evidence only."""
import argparse
import json
from pathlib import Path
import shutil

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


def mean_sd(mean,sd,digits):
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}' if pd.notna(sd) else f'{mean:.{digits}f} (SD --)'


def figure(fig,name):
    for ext in ['pdf','svg','png']:
        path=OUT/(name+'.'+ext); fig.savefig(path,bbox_inches='tight',dpi=300)
        if ext=='svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def extra_figures(complete):
    pair=pd.read_csv('artifacts/controlled-binary-100000/test-evaluation/paired_seed_differences.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.9),sharey=True)
    variants=['no_degree','no_message_features','no_messages','rewired','directed']
    labels=['No degree','No message\nfeatures','No messages','Rewired','Directed']
    for ax,d in zip(axes,DATASETS):
        for i,v in enumerate(variants):
            b=pair[(pair.dataset==d)&(pair.variant==v)].delta_macro_f1.to_numpy()
            ax.scatter(i+np.linspace(-.12,.12,len(b)),b,s=9,color='#28668c')
            ax.errorbar(i,b.mean(),yerr=b.std(ddof=1),color='black',capsize=3,linewidth=1)
        ax.axhline(0,linestyle='--',color='gray',linewidth=.8); ax.set_title(d.upper())
        ax.set_xticks(range(5),labels,rotation=40,ha='right'); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Paired macro $F_1$ change from Full'); fig.tight_layout(); figure(fig,'graph_controls')
    sample=Path('artifacts/sampler-binary-100000/test-evaluation/model_results.csv')
    if sample.exists():
        methods=['unpartitioned','random_exact','metis_exact','cluster_gcn_edge','graphsaint_node_edge']
        frame=pd.read_csv(sample); fig,axes=plt.subplots(1,3,figsize=(7.1,2.9),sharey=True)
        for ax,d in zip(axes,DATASETS):
            for i,m in enumerate(methods):
                b=frame[(frame.dataset==d)&(frame.method==m)].macro_f1.to_numpy()
                ax.scatter(i+np.linspace(-.12,.12,len(b)),b,s=9,color='#28668c')
                ax.errorbar(i,b.mean(),yerr=b.std(ddof=1),color='black',capsize=3,linewidth=1)
            ax.set_title(d.upper()); ax.set_xticks(range(5),['Full','Random\nexact','METIS\nexact','Cluster\nsampled','Node\nsampled'],rotation=40,ha='right')
            ax.set_ylim(0,1.02); ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Test macro $F_1$'); fig.tight_layout(); figure(fig,'sampler_quality')
    if complete:
        frame=pd.read_csv('artifacts/novel-host-1000000/test-evaluation/model_results.csv')
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.5),sharey=True)
        for ax,d in zip(axes,['ton','cse']):
            for i,f in enumerate(['GNN','MLP','RF']):
                b=frame[(frame.dataset==d)&(frame.family==f)].macro_f1.to_numpy()
                ax.scatter(i+np.linspace(-.12,.12,len(b)),b,s=10,color='#28668c')
                ax.errorbar(i,b.mean(),yerr=b.std(ddof=1),color='black',capsize=3,linewidth=1)
            support=frame[frame.dataset==d].iloc[0]
            ax.set_title(f'{d.upper()}: {int(support.attack_support)} attack / {int(support.benign_support)} benign')
            ax.set_xticks(range(3),['Edge GNN','MLP','Forest']); ax.set_ylim(0,1.02); ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('New-host test macro $F_1$'); fig.tight_layout(); figure(fig,'novel_host_quality')
        scale=pd.read_csv('artifacts/larger-inference/summary.csv')
        methods=['full','random','metis','cached','factored']; labels=['Full','Random exact','METIS exact','Layer-wise','Factored mean']
        for metric,label,name in [('mean_warm_s','Complete warm inference (s)','warm_scaling'),('mean_first_process_s','Process launch to first prediction (s)','pipeline_scaling'),('mean_whole_peak_rss_mb','Sampled process peak RSS (MiB)','memory_scaling')]:
            fig,axes=plt.subplots(1,3,figsize=(7.1,2.7))
            for ax,d in zip(axes,DATASETS):
                for m,display in zip(methods,labels):
                    b=scale[(scale.dataset==d)&(scale.method==m)].sort_values('sample_size')
                    if len(b): ax.plot(b.sample_size/1e6,b[metric],marker='o',markersize=3,label=display,linewidth=1)
                ax.set_title(d.upper()); ax.set_yscale('log'); ax.grid(alpha=.2); ax.set_xlabel('Uniform sample flows (million)')
            axes[0].set_ylabel(label); handles,names=axes[0].get_legend_handles_labels()
            fig.legend(handles,names,loc='upper center',ncol=5,fontsize=7); fig.tight_layout(rect=(0,0,1,.9)); figure(fig,name)


def final_tables():
    areas=pd.read_csv('artifacts/precision-recall-areas/class_summary.csv')
    scores=pd.read_csv('artifacts/controlled-binary-100000/test-evaluation/model_results.csv')
    rows=[]
    for d in DATASETS:
        for family in ['GNN','MLP','RF']:
            b=scores[(scores.dataset==d)&(scores.family==family)&(scores.variant=='full')]
            p=areas[(areas.study=='controlled')&(areas.dataset==d)&(areas.family==family)&(areas.method=='full')&(areas.label=='Attack')]
            if len(b)!=5 or len(p)!=1: raise ValueError('Incomplete baseline metrics')
            p=p.iloc[0]
            rows.append([d.upper(),family,*[f'{b[c].mean():.4f} $\\pm$ {b[c].std(ddof=1):.4f}' for c in ['weighted_f1','attack_fpr','attack_recall']],
                f'{p.mean_ap:.4f} $\\pm$ {p.sd_ap:.4f}',f'{p.mean_trapezoidal_pr_auc:.4f} $\\pm$ {p.sd_trapezoidal_pr_auc:.4f}'])
    table('table_baseline_metrics',['Dataset','Model','Weighted $F_1$','FPR','Attack recall','Attack AP','Attack PR-AUC'],rows)
    frame=pd.read_csv('artifacts/novel-host-1000000/test-evaluation/summary.csv')
    rows=[]
    for d in ['ton','cse']:
        for family in ['GNN','MLP','RF']:
            b=frame[(frame.dataset==d)&(frame.family==family)].iloc[0]
            rows.append([d.upper(),family,f'{b.mean_macro_f1:.4f} $\\pm$ {b.sd_macro_f1:.4f}',
                f'{b.mean_attack_fpr:.4f}',f'{b.mean_attack_recall:.4f}',f'{b.mean_attack_ap:.4f}',str(int(b.attack_support))])
    table('table_new_host',['Dataset','Model','Macro $F_1$','FPR','Attack recall','Attack AP','Attack $n$'],rows)
    frame=pd.read_csv('artifacts/classifier-cost-100000/summary.csv'); rows=[]
    for d in DATASETS:
        for method,label in [('gnn_full','GNN Full'),('gnn_layerwise','GNN layer-wise'),('mlp','MLP'),('rf','Forest')]:
            selected=frame[(frame.dataset==d)&(frame.method==method)]
            if len(selected)==0:
                rows.append([d.upper(),label,'--','0','--','--','--'])
                continue
            b=selected.iloc[0]
            rows.append([d.upper(),label,str(int(b.test_flows)),str(int(b.process_blocks)),mean_sd(1000*b.mean_warm_s,1000*b.sd_warm_s,2),
                mean_sd(b.mean_first_process_s,b.sd_first_process_s,2),f'{b.mean_setup_peak_rss_mib:.1f}'])
    table('table_classifier_cost',['Dataset','Method','Test $n$','Blocks','Warm ms','First prediction s','RSS MiB'],rows)
    for name in ['classifier_warm_cost','classifier_pipeline_cost','classifier_memory']:
        for ext in ['pdf','svg','png']: shutil.copyfile(Path('artifacts/classifier-cost-100000')/(name+'.'+ext),OUT/(name+'.'+ext))
    frame=pd.read_csv('artifacts/larger-inference/summary.csv'); rows=[]
    for d in DATASETS:
        for method,label in [('full','Full'),('random','Random exact'),('metis','METIS exact'),('cached','Layer-wise'),('factored','Factored mean')]:
            selected=frame[(frame.dataset==d)&(frame.method==method)&(frame.sample_size==1000000)]
            if len(selected)==0:
                rows.append([d.upper(),label,'--','0','--','--','--'])
                continue
            b=selected.iloc[0]
            rows.append([d.upper(),label,str(int(b.test_flows)),str(int(b.blocks)),
                mean_sd(b.mean_warm_s,b.sd_warm_s,3),mean_sd(b.mean_first_process_s,b.sd_first_process_s,2),f'{b.mean_whole_peak_rss_mb:.1f}'])
    table('table_million_flow_cost',['Dataset','Method','Test $n$','Blocks','Warm s','First prediction s','RSS MiB'],rows)


def execute(complete=False):
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=[Path('artifacts/controlled-binary-100000/test-evaluation/verification.json')]
    if complete:
        evidence += [Path(f'artifacts/{s}/test-evaluation/verification.json') for s in ['sampler-binary-100000','novel-host-1000000']]
        evidence += [Path('artifacts/larger-inference/verification.json'),Path('artifacts/classifier-cost-100000/verification.json'),
            Path('artifacts/precision-recall-areas/definitions.json')]
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
    mapping=[]
    for d in DATASETS:
        source=json.loads(Path(f'docs/provenance/nf-{d}-v2.json').read_text())
        mapping.append([d.upper(),', '.join(source['original_attack_counts']),'Label=0: Benign; Label=1: Attack','Not fitted: missing validation family support'])
    frame=pd.DataFrame(mapping,columns=['Dataset','Original Attack names','Binary mapping','Multiclass status'])
    frame.to_csv(OUT/'table_A_label_mapping.csv',index=False)
    frame.to_latex(OUT/'table_A_label_mapping.tex',index=False,column_format=r'p{.10\textwidth}p{.40\textwidth}p{.18\textwidth}p{.18\textwidth}')
    configuration=[
        ('Task / target','Binary intrusion classification; one prediction per original directed flow record'),
        ('Node / context graph','Original IP identity; reverse arcs supply context only; multiedges and self-loops retain multiplicity'),
        ('Node features','Log one plus outgoing and incoming original-flow counts, independently in each split'),
        ('Flow features','Ten declared fields; eight numerical, PROTOCOL and L7 PROTO categorical; all preprocessing fitted on training only'),
        ('Network','Edge-conditioned mean aggregation; two layers, hidden width 16, ReLU; edge decoder uses endpoint states and flow attributes'),
        ('Training / prediction','Dropout 0.2; class-balanced cross entropy; Adam, weight decay 0.0001, norm-five clipping; softmax and fixed argmax'),
        ('Classifier / graph-control cohort','Learning rate 0.003; four fixed random ownership groups, at most 512 targets per update, full context; ten-epoch maximum, patience three; validation macro F1; seeds 7/19/31/43/59'),
        ('Flow-only baselines','MLP: two hidden layers of width 16, ReLU and dropout 0.2; Forest: 100 balanced bootstrap trees, square-root feature subsampling, selected depth'),
        ('Sampler cohort','Two rates 0.001/0.003 per method at seed 7, validation-only choice, then four remaining seeds; global target-risk scaling differs from per-batch weighted mean'),
        ('Systems cohort','Original frozen 100000-flow pilot seed-7 weights, learning rate 0.001, and its fitted preprocessing; no fitting during scaling; distinct from classifier refits'),
        ('Partition / target batch','Exact-context controls: Full, balanced random or real METIS; ownership seed 7; flow targets never dropped at cuts'),
        ('Framework / hardware','PyTorch 2.9 CPU; local Ubuntu/WSL2; one Torch thread; no multi-worker deployment')]
    frame=pd.DataFrame(configuration,columns=['Component','Actual implementation']); frame.to_csv(OUT/'table_B_configuration.csv',index=False)
    frame.to_latex(OUT/'table_B_configuration.tex',index=False,column_format=r'p{.22\textwidth}p{.70\textwidth}')
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
    compact=[]
    for d in DATASETS:
        reports=[json.loads(Path(f'artifacts/controlled-binary-100000/test-evaluation/{d}-gnn-full-{s}-metrics.json').read_text())['per_class'] for s in [7,19,31,43,59]]
        for label in ['Benign','Attack']:
            compact.append([d.upper(),label,*[f'{np.mean([r[label][metric] for r in reports]):.4f}' for metric in ['precision','recall','f1-score']],str(int(reports[0][label]['support']))])
    table('table_E_main_gnn',['Dataset','Class','Precision','Recall','$F_1$','Support'],compact)
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
    comparison=systems[(systems.method=='full')|((systems.method.isin(['random','metis']))&(systems.partitions==4))]
    if len(comparison)!=9: raise ValueError('Missing fixed four-partition method comparison')
    comparison.to_csv(OUT/'table_D_fixed_methods.csv',index=False)
    printable=[]
    for r in comparison.itertuples():
        printable.append([r.dataset.upper(),r.method,str(r.partitions),str(r.test_n),f'{r.macro_f1:.4f}',f'{r.weighted_f1:.4f}',f'{r.total_ms:.2f}',f'{r.rss_mib:.1f}'])
    table('table_D_fixed_methods',['Dataset','Method','$P$','Test $n$','Macro $F_1$','Weighted $F_1$','Total ms','RSS MiB'],printable)
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
    extra_figures(complete)
    if complete: final_tables()
    metadata=dict(scope='verified binary research evidence; pilot systems checkpoint distinct from five-seed classifier fits',
        complete_evidence=complete,evidence_sha256={str(p).replace('\\','/'):sha256_file(p) for p in evidence},
        no_test_selection=True,plot_width_inches=7.1,font_points=8,
        tables='A: original source and split counts; C: historical fixed-model batch-512 partition sweep; E: per-class five-seed reports',
        limitations='five seeds on reused development split; one-seed historical partition sweep; unsupported multiclass/causal/distributed claims excluded')
    dump(OUT/'asset_manifest.json',metadata)
    print(OUT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--complete',action='store_true'); a=p.parse_args(); execute(a.complete)
