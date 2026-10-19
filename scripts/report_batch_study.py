"""Publish aggregate batch-size timings and rebuild their figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export(run,destination):
    run=Path(run); destination=Path(destination)
    completion=json.loads((run/"completion.json").read_text())
    if completion["completed_conditions"]+completion["unavailable_conditions"]!=completion["requested_conditions"]:
        raise ValueError("Incomplete study")
    checks=json.loads((run/"prediction_checks.json").read_text())
    ownership={}
    for check in checks:
        tag=f"{check['method']}-p{check['partitions']}-b{check['batch_size']}"
        with np.load(run/(tag+".npz"),allow_pickle=False) as saved:
            digest=hashlib.sha256(saved["owner"].astype("<i8").tobytes()).hexdigest()
        key=f"{check['method']}-{check['partitions']}"
        if key in ownership and ownership[key]!=digest:
            raise ValueError("Node ownership changed between batch sizes")
        ownership[key]=digest
    destination.mkdir(parents=True,exist_ok=False)
    for name in ["study_plan.json","graph_storage.json","timing_repeats.csv","prediction_checks.json","unavailable_runs.json","completion.json"]:
        shutil.copyfile(run/name,destination/name)
    (destination/"ownership_checks.json").write_text(json.dumps({"same_ownership_across_batch_sizes":True,"ownership_sha256":ownership},indent=2)+"\n",encoding="utf-8")
    (destination/"SCOPE.txt").write_text("30,000-flow sample; one frozen model per dataset; no model fitting or tuning.\nThree timing repeats are measurements of the same model, not independent training seeds.\nTotal inference excludes cold data preparation, graph construction, graph loading, partitioning, warm-up and process startup.\nFresh process per condition; one complete warm-up pass; memory is sampled process RSS.\nNo raw flows, host identities, per-flow probabilities or checkpoints are included.\n",encoding="utf-8")


def report(root="artifacts/batch-study"):
    root=Path(root)
    all_rows=[]
    for dataset in ["ton","cse","bot"]:
        rows=pd.read_csv(root/dataset/"timing_repeats.csv")
        if rows.dataset.unique().tolist()!=[dataset]: raise ValueError("Unexpected dataset in timing table")
        all_rows.append(rows)
    rows=pd.concat(all_rows,ignore_index=True)
    keys=["dataset","method","partitions","batch_size"]
    summary=rows.groupby(keys,as_index=False).agg(
        repeats=("repeat","size"),test_row_ids_sha256=("target_row_ids_sha256","first"),
        mean_inference_s=("total_inference_s","mean"),sd_inference_s=("total_inference_s","std"),
        peak_rss_mib=("peak_cpu_rss_mb_sampled","max"),mean_forward_s=("model_forward_s","mean"),
        mean_packing_s=("halo_and_transfer_s","mean"),batches=("batches","first"))
    summary.to_csv(root/"condition_summary.csv",index=False)
    summary[["dataset","method","partitions","batch_size","repeats","mean_inference_s","sd_inference_s","peak_rss_mib","batches"]].to_latex(root/"condition_summary.tex",index=False,float_format="%.6f")
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained",sharey="row")
    series=[("chunked",1,"One partition","#3274a1"),("random",2,"Random 2","#d98832"),
            ("random",4,"Random 4","#47966a"),("random",8,"Random 8","#8a64ad")]
    for col,dataset in enumerate(["ton","cse","bot"]):
        data=summary[summary.dataset==dataset]
        full=data[data.method=="full"].iloc[0]
        for method,parts,label,color in series:
            selected=data[(data.method==method)&(data.partitions==parts)].sort_values("batch_size")
            if selected.empty: continue
            axes[0,col].errorbar(selected.batch_size,selected.mean_inference_s*1000,
                yerr=selected.sd_inference_s.fillna(0)*1000,label=label,color=color,marker="o",capsize=3)
            axes[1,col].plot(selected.batch_size,selected.peak_rss_mib,color=color,marker="o")
        axes[0,col].axhline(full.mean_inference_s*1000,color="#333333",linestyle="--",label="Full graph")
        axes[0,col].axhspan((full.mean_inference_s-full.sd_inference_s)*1000,
                           (full.mean_inference_s+full.sd_inference_s)*1000,color="#333333",alpha=.08)
        axes[1,col].axhline(full.peak_rss_mib,color="#333333",linestyle="--")
        axes[0,col].set_yscale("log")
        axes[0,col].set_title(dataset.upper())
        axes[1,col].set_ylim(0,summary.peak_rss_mib.max()*1.1)
        for ax in axes[:,col]:
            ax.set_xscale("log",base=2); ax.set_xticks([256,512,1024,2048],["256","512","1024","2048"])
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.2)
        axes[1,col].set_xlabel("Maximum target flows per batch")
    axes[0,0].set_ylabel("Whole-test-graph inference (ms)")
    axes[1,0].set_ylabel("Peak process RSS (MiB)")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Batch-size comparison on 30,000-flow samples\nSame saved models and test rows · three timing repeats · no retraining",fontsize=13)
    for extension in ["png","svg","pdf"]: fig.savefig(root/("batch_size_comparison."+extension),dpi=300)
    plt.close(fig)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--export",nargs=2,metavar=("RUN","DESTINATION"))
    args=parser.parse_args()
    if args.export: export(*args.export)
    else: print(report().drop(columns="test_row_ids_sha256").to_string(index=False))
