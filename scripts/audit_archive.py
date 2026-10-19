"""Audit an official BagIt ZIP and generate a deterministic uniform pilot.

Never publish the raw/sample records. The archive is read incrementally, avoiding
an in-memory full CSV. Full class counts and the random-priority sample are exact
for the complete scan. Sample rows retain their original CSV row number.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def audit(archive,dataset,out,sample_size=30000,seed=2026):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    h=hashlib.sha256()
    with open(archive,"rb") as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b""): h.update(b)
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith(dataset+".csv")]
        if len(names)!=1: raise ValueError(f"Expected one matching CSV; found {names}")
        name=names[0]
        manifest_name=next(n for n in z.namelist() if n.endswith("/manifest-sha1.txt"))
        manifest=z.read(manifest_name).decode()
        expected=None
        for line in manifest.splitlines():
            parts=line.split(None,1)
            if len(parts)==2 and parts[1].strip().endswith(dataset+".csv"): expected=parts[0]
        class_counts=Counter(); label_counts=Counter(); sample=None; total=0; hosts=set()
        reader=pd.read_csv(z.open(name),chunksize=200000,low_memory=False)
        for chunk in reader:
            class_counts.update(chunk.Attack.astype(str).value_counts().to_dict())
            label_counts.update({str(k):int(v) for k,v in chunk.Label.value_counts().items()})
            hosts.update(chunk.IPV4_SRC_ADDR.astype(str).unique())
            hosts.update(chunk.IPV4_DST_ADDR.astype(str).unique())
            chunk["source_row_id"]=np.arange(total,total+len(chunk))
            chunk["_priority"]=rng.random(len(chunk))
            total+=len(chunk)
            # Retaining the globally smallest uniform keys gives a uniform sample.
            chunk=chunk.nsmallest(min(sample_size,len(chunk)),"_priority")
            sample=chunk if sample is None else pd.concat([sample,chunk],ignore_index=True).nsmallest(sample_size,"_priority")
            if total % 2000000==0: print(f"{dataset}: {total:,} source rows audited",flush=True)
        sample=sample.sort_values("source_row_id").drop(columns="_priority")
        sample.to_csv(out/(dataset+"-uniform-pilot.csv"),index=False)
        csv_hash=hashlib.sha256(); sha1=hashlib.sha1()
        with z.open(name) as stream:
            for b in iter(lambda:stream.read(8*1024*1024),b""):
                csv_hash.update(b); sha1.update(b)
        if expected is None or sha1.hexdigest()!=expected:
            raise ValueError("Archive BagIt checksum missing or mismatched")
        result={"dataset":dataset,"archive_sha256":h.hexdigest(),"csv_sha256":csv_hash.hexdigest(),"bagit_sha1_verified":True,
            "source_rows":total,"unique_ip_nodes":len(hosts),"source_columns":[c for c in sample if c!="source_row_id"],"original_attack_counts":dict(class_counts),"binary_counts":dict(label_counts),
            "pilot_rows":len(sample),"pilot_seed":seed,"pilot_sampling":"uniform random priorities across full CSV; no label conditioning",
            "pilot_attack_counts":sample.Attack.value_counts().to_dict(),"pilot_source_row_ids_sha256":hashlib.sha256(sample.source_row_id.to_numpy(dtype="<i8").tobytes()).hexdigest()}
        (out/"source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("archive"); p.add_argument("--dataset",required=True); p.add_argument("--out",required=True); p.add_argument("--sample-size",type=int,default=30000)
    a=p.parse_args(); print(json.dumps(audit(a.archive,a.dataset,a.out,a.sample_size),indent=2))
