"""Plot measured graph-work counts; no classifier scores are used."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(root="artifacts/profiles"):
    root=Path(root); tables=[]
    fig,axes=plt.subplots(1,3,figsize=(11,4.6),layout="constrained",sharex=True)
    methods=[("full",1),("chunked",1),("random",2),("random",4),("random",8)]
    labels=["Full graph","1 part, 512-flow batches","Random: 2 parts","Random: 4 parts","Random: 8 parts"]
    for ax,dataset in zip(axes,["ton","cse","bot"]):
        table=pd.read_csv(root/(dataset+"-30000")/"work_summary.csv")
        table.insert(0,"dataset",dataset); tables.append(table)
        values=[]
        for method,parts in methods:
            selected=table[(table.method==method)&(table.requested_partitions==parts)]
            values.append(float(selected.iloc[0].message_work_factor) if len(selected) else np.nan)
        ax.barh(range(5),np.nan_to_num(values),color=["#3274a1","#777777"]+["#d98832"]*3,height=.65)
        for i,value in enumerate(values):
            ax.text((value+.25) if np.isfinite(value) else .25,i,f"{value:.2f}×" if np.isfinite(value) else "Not feasible",va="center",fontsize=9)
        ax.set_yticks(range(5),labels,fontsize=9); ax.invert_yaxis()
        ax.axvline(1,color="#3274a1",linestyle=":",linewidth=1)
        ax.set_xlim(0,21); ax.set_xlabel("Message-edge visits / full-graph visits")
        ax.set_title(dataset.upper()); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Repeated graph work in the 30,000-flow pilots\nTwo layers · same test targets · each target scored once",fontsize=13)
    for extension in ["png","svg","pdf"]: fig.savefig(root/("message_work."+extension),dpi=300)
    plt.close(fig)
    combined=pd.concat(tables,ignore_index=True)
    combined.to_csv(root/"combined_work_summary.csv",index=False)
    combined[["dataset","method","requested_partitions","target_flows","batches","message_work_factor","max_packed_tensor_bytes"]].to_latex(root/"work_summary.tex",index=False,float_format="%.3f")
    return combined


if __name__=="__main__": print(report()[["dataset","method","requested_partitions","message_work_factor"]].to_string(index=False))
