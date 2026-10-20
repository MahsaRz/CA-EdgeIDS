"""Tables A-E and vector/PNG figures from actual run artifacts only."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t as student_t


def save_figure(fig,path):
    fig.savefig(str(path)+".svg",bbox_inches="tight")
    svg=Path(str(path)+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    fig.savefig(str(path)+".png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def table(frame,path):
    frame.to_csv(str(path)+".csv",index=False)
    Path(str(path)+".tex").write_text(frame.to_latex(index=False,float_format=lambda x:f"{x:.5f}",escape=True),encoding="utf-8")


def summary(runs):
    # Timing repetitions are NOT independent training seeds. Collapse within seed first.
    numeric=["macro_f1","weighted_f1","macro_precision","macro_recall","weighted_precision","weighted_recall",
             "average_partition_forward_s","model_forward_s","total_inference_s","peak_cpu_rss_mb_sampled","partitioning_s","halo_and_transfer_s"]
    per_seed=runs.groupby(["method","partitions","seed"],as_index=False)[numeric].mean()
    rows=[]
    for (method,parts),sub in per_seed.groupby(["method","partitions"]):
        row={"method":method,"partitions":parts,"test_n":int(runs.test_n.iloc[0]),"seeds":len(sub)}
        for metric in numeric:
            vals=sub[metric].to_numpy()
            row[metric]=float(vals.mean())
            row[metric+"_ci95_halfwidth"]=float(student_t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pipeline_figure(out):
    fig,ax=plt.subplots(figsize=(13,3.0))
    ax.set(xlim=(0,12.5),ylim=(0,3)); ax.axis("off")
    boxes=[(.2,1.35,"NetFlow-v2 records\nIP identities preserved"),(2.7,1.35,"Split by host pair\nFit preprocessing\non training data"),(5.2,1.35,"Directed flow graph\nEdge features\nNode degree features"),(7.7,1.35,"Own every target edge\nFull / Random / METIS"),(10.2,1.35,"GraphSAGE-style GNN\nEdge classifier")]
    for x,y,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.1,1.05,boxstyle="round,pad=.10",facecolor="#e9f1f8",edgecolor="#215a86"))
        ax.text(x+1.05,y+.525,text,ha="center",va="center",fontsize=8)
        if x<10:
            ax.annotate("",xy=(x+2.39,y+.525),xytext=(x+2.23,y+.525),arrowprops={"arrowstyle":"->","color":"#215a86"})
    ax.text(6.25,.78,"Cross-partition K-hop context retained; every original flow predicted exactly once",ha="center",fontsize=10)
    ax.text(6.25,.28,"Training: edge labels -> weighted cross-entropy     |     Inference: softmax -> Benign / Attack or attack type",ha="center",fontsize=9)
    save_figure(fig,out/"figure_1_pipeline")


def generate(run_dir):
    root=Path(run_dir)
    out=root/"paper_outputs"
    out.mkdir(exist_ok=True)
    m=json.loads((root/"dataset_manifest.json").read_text())
    cfg=json.loads((root/"resolved_config.json").read_text())
    runs=pd.read_csv(root/"benchmark_runs.csv")
    if runs.empty:
        raise ValueError("No successful benchmark rows")
    if runs.test_n.nunique()!=1 or runs.target_row_ids_sha256.nunique()!=1:
        raise AssertionError("Fixed test population invariant violated")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    banner="SYNTHETIC SOFTWARE CHECK - NOT PAPER RESULTS" if cfg.get("synthetic") else "PILOT SUBSET - NOT FULL DATASET" if cfg.get("max_rows") or cfg.get("data_scope","full_dataset")!="full_dataset" else "Fixed checkpoint inference benchmark"
    table(pd.DataFrame([{"dataset":m["dataset"],"prediction_unit":"original flow / edge","original_labels":"; ".join(m["original_labels"]),"task":m["task"],"classes":"; ".join(m["label_names"]),"train_n":m["split_counts"]["train"],"validation_n":m["split_counts"]["validation"],"test_n":m["split_counts"]["test"],"scope":banner}]),out/"table_A_dataset_labels")
    b={"task":cfg["task"],"node_definition":"original IP address","target_edge":"one original directed flow record","message_direction":"both directions" if cfg.get("bidirectional_messages",True) else "original direction",
       "node_features":"log1p(outgoing/incoming flow counts), computed separately per split snapshot","edge_features":"; ".join(cfg["features"]),"label_location":"original target edges only; no label in messages",
       "model":"edge-conditioned GraphSAGE-style mean aggregator + edge MLP","layers":cfg["layers"],"hidden":cfg["hidden"],"activation":"ReLU","dropout":cfg["dropout"],"loss":"weighted cross entropy; training frequencies only","output":"softmax probabilities; argmax decision",
       "optimizer":"Adam","learning_rate":cfg["learning_rate"],"weight_decay":cfg["weight_decay"],"maximum_epochs":cfg["epochs"],"selection":"validation macro-F1; patience "+str(cfg["patience"]),"framework":"PyTorch","training_partitioning":cfg["train_partition_method"],"training_partitions":cfg["train_parts"],"seeds":str(cfg["seeds"]),"split_seed":cfg["split_seed"],"split_protocol":cfg["split_protocol"],"test_context":"K-hop incoming halo; no boundary edges discarded","comparison":"same trained weights across inference partition counts","scope":banner}
    table(pd.DataFrame({"component":list(b),"actual_implementation":list(b.values())}),out/"table_B_configuration")
    s=summary(runs)
    s.to_csv(out/"complete_seed_summary.csv",index=False)
    compact={"method":"Method","partitions":"C","test_n":"Test n","macro_f1":"Macro F1","weighted_f1":"Weighted F1",
             "macro_precision":"Macro precision","macro_recall":"Macro recall","average_partition_forward_s":"Avg part forward (s)",
             "total_inference_s":"Total inference (s)","peak_cpu_rss_mb_sampled":"Peak CPU RSS (MiB)"}
    table(s.loc[s.method!="full",list(compact)].rename(columns=compact),out/"table_C_partition_benchmark")
    # This is an inference comparison, not three independently trained methods.
    nonfull=s[s.method!="full"]
    compare_parts=int(nonfull.partitions.min()) if len(nonfull) else 1
    comp=s[(s.method=="full")|(s.partitions==compare_parts)]
    columns=["method","partitions","test_n","macro_f1","weighted_f1","total_inference_s","peak_cpu_rss_mb_sampled"]
    table(comp[columns].rename(columns=compact),out/"table_D_full_random_metis")
    class_rows=[]
    seed_metrics=[]
    for seed in cfg["seeds"]:
        choices=sorted((root/f"seed-{seed}").glob("*-metrics.json"))
        if not choices: continue
        metric=json.loads(choices[0].read_text())
        seed_metrics.append(metric)
        for name in m["label_names"]+["macro avg","weighted avg"]:
            class_rows.append({"seed":seed,"class":name,**metric["per_class"][name]})
    table(pd.DataFrame(class_rows),out/"table_E_per_class")
    pipeline_figure(out)
    for method,sub in s[s.method!="full"].groupby("method"):
        sub=sub.sort_values("partitions")
        fig,ax=plt.subplots(figsize=(6.4,4))
        for metric,label in [("macro_f1","Macro-F1"),("weighted_f1","Weighted-F1")]:
            ax.errorbar(sub.partitions,sub[metric],yerr=sub[metric+"_ci95_halfwidth"].fillna(0),marker="o",label=label,capsize=3)
        ax.set(xlabel="Number of partitions",ylabel="F1 score",ylim=(0,1.02),title=f"{method.upper()}: fixed test population")
        ax.legend(); ax.grid(alpha=.2); fig.suptitle(banner,fontsize=8); fig.tight_layout()
        save_figure(fig,out/f"figure_2_partition_accuracy_{method}")
        fig,axes=plt.subplots(1,2,figsize=(9,3.7))
        axes[0].plot(sub.partitions,sub.average_partition_forward_s,"o-")
        axes[0].set(ylabel="Mean forward time per partition (s)",xlabel="Partitions")
        axes[1].plot(sub.partitions,sub.total_inference_s,"o-",label="Total inference")
        axes[1].plot(sub.partitions,sub.model_forward_s,"s--",label="All forward passes")
        axes[1].set(ylabel="Time for fixed test population (s)",xlabel="Partitions"); axes[1].legend(fontsize=8)
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle(banner,fontsize=8); fig.tight_layout(); save_figure(fig,out/f"figure_3_partition_latency_{method}")
    # Pick the smallest common non-full partition count, declared in the figure data.
    fig,axes=plt.subplots(1,3,figsize=(10,3.8))
    for ax,(metric,label) in zip(axes,[("macro_f1","Macro-F1"),("total_inference_s","Total inference (s)"),("peak_cpu_rss_mb_sampled","Sampled peak CPU RSS (MiB)")]):
        ax.bar(comp.method,comp[metric],color=["#255b84","#d8882a","#3c8b6b"][:len(comp)])
        ax.set_ylabel(label)
        ax.tick_params(axis="x",rotation=20)
    fig.suptitle(banner+f"\nNon-full methods: {compare_parts} partitions; fixed checkpoint",fontsize=8); fig.tight_layout()
    save_figure(fig,out/"figure_4_method_comparison")
    # Row-normalized confusion matrix supplements classification-report heatmaps.
    if seed_metrics:
        cm=np.array(seed_metrics[0]["confusion_matrix"],dtype=float)
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(max(5,len(m["label_names"])*.65),max(4,len(m["label_names"])*.6)))
        im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(range(len(m["label_names"])),m["label_names"],rotation=60,ha="right")
        ax.set_yticks(range(len(m["label_names"])),m["label_names"])
        ax.set(xlabel="Predicted class",ylabel="True class",title="Row-normalized confusion matrix, first seed")
        fig.colorbar(im,ax=ax); fig.suptitle(banner,fontsize=8); fig.tight_layout(); save_figure(fig,out/"figure_5_confusion_matrix")
    failures=json.loads((root/"unavailable_runs.json").read_text())
    (out/"report_manifest.json").write_text(json.dumps({"report_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_csv_sha256":hashlib.sha256((root/"benchmark_runs.csv").read_bytes()).hexdigest(),
        "scope":banner},indent=2),encoding="utf-8")
    (out/"READ_ME_FIRST.txt").write_text(banner+"\n\nTables C and D use the same checkpoint per seed. Exact K-hop halos should keep F1 invariant. This is an inference-system comparison, not evidence of superior training accuracy.\n\n95% intervals are Student-t intervals over training seeds after averaging timing repetitions within seed. Missing intervals indicate one seed. Memory is sampled process RSS, including resident input graphs; not an allocator-exact CPU peak.\n\nUnavailable runs (never replaced with invented values):\n"+json.dumps(failures,indent=2),encoding="utf-8")
    return out
