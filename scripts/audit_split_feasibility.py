"""Stream complete official archives to audit a frozen pair split, without training.

No raw host addresses or pair identifiers are written to the published report.
Memory is proportional to unique pairs, not the number of flow records.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile

import pandas as pd

from ca_edgeids.data import SOURCE, DEST, SCHEMAS, normalize_labels, sha256_file, stable_bucket


def audit(archive, dataset, provenance, out, seed=42):
    out=Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    source=json.loads(Path(provenance).read_text())
    archive_hash=sha256_file(archive)
    if archive_hash!=source["archive_sha256"]:
        raise ValueError("Archive differs from previously checksum-verified provenance")
    counts=Counter(); originals=Counter(); pair_classes=defaultdict(set); cache={}; total=0
    with zipfile.ZipFile(archive) as z:
        files=[n for n in z.namelist() if n.endswith(source["dataset"]+".csv")]
        if len(files)!=1: raise ValueError("Ambiguous source CSV")
        for frame in pd.read_csv(z.open(files[0]),usecols=[SOURCE,DEST,"Label","Attack"],chunksize=200000):
            labels,names,raw=normalize_labels(frame,dataset,"multiclass")
            originals.update(pd.Series(raw).value_counts().to_dict())
            left=frame[SOURCE].astype(str); right=frame[DEST].astype(str)
            ordered=left<=right
            keys=left.where(ordered,right)+"|"+right.where(ordered,left)
            # Hash each distinct pair only once; a pair's assignment ignores labels.
            for key in keys.unique():
                if key not in cache:
                    u=stable_bucket(key,seed)
                    cache[key]="train" if u<.6 else "validation" if u<.8 else "test"
            grouped=pd.DataFrame({"pair":keys,"class_id":labels}).groupby(["pair","class_id"]).size()
            for (key,class_id),n in grouped.items():
                part=cache[key]; name=names[class_id]
                counts[name,part]+=int(n); pair_classes[name,part].add(key)
            total+=len(frame)
            if total%2000000==0: print(f"{dataset}: {total:,} flows checked",flush=True)
    if total!=source["source_rows"] or dict(originals)!=source["original_attack_counts"]:
        raise ValueError("Complete-source counts disagree with verified provenance")
    rows=[]
    for name in SCHEMAS[dataset]:
        row={"class":name}
        for part in ["train","validation","test"]:
            row[f"{part}_flows"]=counts[name,part]
            row[f"{part}_host_pairs"]=len(pair_classes[name,part])
        row["all_host_pairs"]=sum(row[f"{p}_host_pairs"] for p in ["train","validation","test"])
        rows.append(row)
    missing={p:[r["class"] for r in rows if not r[f"{p}_flows"]] for p in ["train","validation","test"]}
    report={"dataset":dataset,"scope":"complete_source_split_feasibility_not_model_results",
            "archive_sha256":archive_hash,"csv_sha256_from_verified_provenance":source["csv_sha256"],
            "source_rows":total,"split_seed":seed,"split_protocol":"conversation",
            "fractions":[.6,.2,.2],"unique_host_pairs":len(cache),"missing_classes":missing,
            "can_train_and_select_all_classes":not missing["train"] and not missing["validation"],
            "can_evaluate_all_classes":not missing["test"],"class_support":rows,
            "interpretation":"Frozen split audited before model fitting. Counts alone do not demonstrate statistical power or generalization; no seeds searched, no class merging, no test scores used."}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"),index=False)
    print(json.dumps({k:v for k,v in report.items() if k!="class_support"},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("archive"); p.add_argument("--dataset",choices=list(SCHEMAS),required=True)
    p.add_argument("--provenance",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); audit(a.archive,a.dataset,a.provenance,a.out)
