"""Rebuild aggregate feasibility tables and publication-quality audit figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def report(root="artifacts/preflight"):
    root=Path(root)
    fig, axes=plt.subplots(2,3,figsize=(12,8),layout="constrained")
    tables=[]
    for col,dataset in enumerate(["ton","bot","cse"]):
        for row,scope in enumerate(["30000-flow uniform pilot","Complete source"]):
            path=root/(f"{dataset}-multiclass-30000/preflight.json" if row==0 else f"{dataset}-complete-source.json")
            result=json.loads(path.read_text())
            table=pd.DataFrame(result["class_support"])
            table.insert(0,"dataset",dataset); table.insert(1,"scope",scope); tables.append(table)
            counts=table[[f"{p}_flows" for p in ["train","validation","test"]]].to_numpy()
            ax=axes[row,col]
            cmap=plt.get_cmap("Blues").copy(); cmap.set_bad("#fbe3df")
            ax.imshow(np.ma.masked_equal(counts,0),cmap=cmap,norm=LogNorm(vmin=1,vmax=2e7),aspect="auto")
            for i in range(len(table)):
                for j in range(3):
                    n=counts[i,j]
                    # Exact counts remain in CSV; abbreviated labels keep the figure legible.
                    text=str(n) if n<1000 else (f"{n/1e6:.2f}M" if n>=1e6 else f"{n/1000:.1f}k")
                    ax.text(j,i,text,ha="center",va="center",fontsize=8,
                            color="#a32f21" if n==0 else ("white" if n>300000 else "#102a43"),
                            fontweight="bold" if n==0 else "normal")
            ax.set_xticks(range(3),["Train","Validation","Test"])
            ax.set_yticks(range(len(table)),table["class"],fontsize=8)
            ax.set_title(f"{dataset.upper()} · {scope}",fontsize=11)
    fig.suptitle("Multiclass split feasibility: frozen host-pair split, seed 42\nRed cells: zero flows · Blue cells: shared logarithmic count scale",fontsize=13)
    for ext in ["png","svg","pdf"]:
        fig.savefig(root/f"multiclass_split_coverage.{ext}",dpi=300)
    plt.close(fig)
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(root/"all_class_split_support.csv",index=False)
    table.to_latex(root/"all_class_split_support.tex",index=False,escape=True)
    return table


if __name__=="__main__": report()
