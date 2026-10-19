"""Export aggregate evidence from the completed larger Linux pilot."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from report_batch_study import export, report
from verify_linux_scaling import verify


def copy_files(source,dest,names):
    dest.mkdir(parents=True,exist_ok=True)
    for name in names: shutil.copyfile(source/name,dest/name)


def build():
    verify()
    source=Path("runs/linux-scaling-100000")
    completion=json.loads((source/"completion.json").read_text())
    if {r["dataset"] for r in completion["datasets"]}!={"ton","cse","bot"}:
        raise ValueError("The three-dataset study is incomplete")
    public=Path("artifacts/linux-scaling-100000")
    public.mkdir(parents=True,exist_ok=False)
    copy_files(source,public,["frozen_plan.json","completion.json","requirements.txt","verification.json"])
    metrics=[]; stages=[]; profiles=[]; per_class=[]; confusion={}
    for dataset in ["ton","cse","bot"]:
        train=source/"training"/dataset
        export(source/"batch-study"/dataset,public/"batch-study"/dataset)
        copy_files(train,public/"training"/dataset,["requested_config.json","resolved_config.json","environment.json", "code_manifest.json",
            "dataset_manifest.json","protocol_checks.json","graph_statistics.json","benchmark_runs.csv","unavailable_runs.json"])
        copy_files(train/"seed-7",public/"training"/dataset/"seed-7",["training_history.csv","training_summary.json","full-1-metrics.json"])
        copy_files(source/"preflight"/dataset,public/"preflight"/dataset,["requested_config.json","dataset_manifest.json","preflight.json","class_split_support.csv"])
        copy_files(source/"profiles"/dataset,public/"profiles"/dataset,["work_summary.csv","batch_work.csv","profile_manifest.json"])
        manifest=json.loads((train/"dataset_manifest.json").read_text())
        graph=json.loads((train/"graph_statistics.json").read_text())["test"]
        quality=json.loads((train/"seed-7/full-1-metrics.json").read_text())
        training=json.loads((train/"seed-7/training_summary.json").read_text())
        confusion[dataset]=np.asarray(quality["confusion_matrix"])
        for label in ["Benign","Attack"]:
            item=quality["per_class"][label]
            per_class.append(dict(dataset=dataset,label=label,support=int(item["support"]),
                precision=item["precision"],recall=item["recall"],f1=item["f1-score"],
                average_precision=quality["average_precision_ovr"][label],ovr_fpr=quality["fpr_ovr"][label]))
        if int(quality["per_class"]["Benign"]["support"])!=manifest["class_counts"]["test"]["Benign"]:
            raise ValueError("Test supports disagree")
        metrics.append(dict(dataset=dataset,source_sample_flows=manifest["rows_read"],
            train_flows=manifest["split_counts"]["train"],validation_flows=manifest["split_counts"]["validation"],
            test_flows=quality["test_n"],test_ip_nodes=graph["nodes"],test_message_arcs=graph["message_arcs"],
            transformed_feature_width=manifest["transformed_width"],
            epochs_run=training["epochs_run"],best_epoch=training["best_epoch"],validation_macro_f1=training["validation_macro_f1"],
            test_macro_f1=quality["macro_f1"],test_weighted_f1=quality["weighted_f1"],
            test_benign_recall=quality["per_class"]["Benign"]["recall"],test_attack_fpr=quality["attack_false_positive_rate"],
            test_attack_average_precision=quality["average_precision_ovr"]["Attack"]))
        first=pd.read_csv(train/"benchmark_runs.csv").iloc[0]
        stages.append(dict(dataset=dataset,sample_read_and_preprocessing_s=first.preprocessing_s,
            graph_construction_s=first.graph_construction_s,training_and_validation_s=first.training_s,
            study_graph_copy_s=json.loads((source/"batch-study"/dataset/"graph_storage.json").read_text())["conversion_s"]))
        profile=pd.read_csv(source/"profiles"/dataset/"work_summary.csv")
        profile.insert(0,"dataset",dataset); profiles.append(profile)
        if "metis" not in set(profile.method): raise ValueError("No real-data METIS profile was measured")
    summary=report(public/"batch-study")
    quality=pd.DataFrame(metrics); preparation=pd.DataFrame(stages); work=pd.concat(profiles,ignore_index=True)
    for name,table in [("classification_and_counts",quality),("per_class_metrics",pd.DataFrame(per_class)),("preparation_stages",preparation),("graph_work",work)]:
        table.to_csv(public/(name+".csv"),index=False)
        table.to_latex(public/(name+".tex"),index=False,float_format="%.6f",na_rep="--",escape=True)
    rows=[]
    for dataset in ["ton","cse","bot"]:
        data=summary[summary.dataset==dataset]; row={"dataset":dataset}
        full=data[data.method=="full"].iloc[0]
        row.update(full_warm_ms=full.mean_inference_s*1000,full_first_prediction_process_s=full.first_prediction_process_s,
                   full_peak_rss_mib=full.peak_rss_mib,full_graph_load_s=full.graph_load_s,full_checkpoint_load_s=full.checkpoint_load_s)
        for method,parts in [("chunked",1),("random",2),("random",4),("random",8),("metis",2),("metis",4),("metis",8)]:
            selected=data[(data.method==method)&(data.partitions==parts)&(data.batch_size==512)]
            row[f"{method}_{parts}_batch512_warm_ms"]=None if selected.empty else selected.iloc[0].mean_inference_s*1000
        rows.append(row)
    timings=pd.DataFrame(rows); timings.to_csv(public/"fixed_batch512_comparison.csv",index=False)
    timings.to_latex(public/"fixed_batch512_comparison.tex",index=False,float_format="%.6f",na_rep="--",escape=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(9,3.4),layout="constrained")
    for ax,dataset in zip(axes,["ton","cse","bot"]):
        counts=confusion[dataset]; normalized=counts/counts.sum(axis=1,keepdims=True)
        ax.imshow(normalized,vmin=0,vmax=1,cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{counts[i,j]:,}\n{normalized[i,j]:.1%}",ha="center",va="center",
                        color="white" if normalized[i,j]>.6 else "#15324a",fontsize=10)
        ax.set_xticks([0,1],["Benign","Attack"]); ax.set_yticks([0,1],["Benign","Attack"])
        ax.set_title(dataset.upper()); ax.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    fig.suptitle("Binary test confusion matrices\nCounts and percentages within each true class · one model per dataset",fontsize=12)
    for extension in ["png","svg","pdf"]: fig.savefig(public/("binary_confusion_matrices."+extension),dpi=300)
    plt.close(fig)
    (public/"SCOPE.txt").write_text("100,000 source flows per dataset; actual test populations are reported separately.\nOne binary model seed per dataset; unchanged initial training settings.\nReal-data Full, one-partition, Random and METIS inference on local Ubuntu under WSL2.\nWarm timing repeats are not independent trained models. First-process timings have one observation per condition and may use a warm OS file cache.\nNo raw flows, model files, host addresses, per-flow probabilities or node assignments are included.\nNo claim of a full-dataset study, distributed deployment, topology adaptation or causal inference.\n",encoding="utf-8")
    print(quality.to_string(index=False))
    print(timings.to_string(index=False))
    print(public)


if __name__=="__main__": build()
