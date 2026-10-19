"""Create a small, explicitly marked pilot from a fully audited uniform sample."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def create(dataset,rows=5000,task="binary"):
    directory=Path("data")/dataset
    audit=json.loads((directory/"source_audit.json").read_text())
    name=audit["dataset"]
    frame=pd.read_csv(directory/(name+"-uniform-pilot.csv"))
    if not 0<rows<=len(frame): raise ValueError("Requested sample exceeds audited uniform sample")
    sample=frame.sample(n=rows,random_state=17).sort_values("source_row_id")
    csv=directory/f"{name}-uniform-{rows}.csv"
    sample.to_csv(csv,index=False)
    audit.update(pilot_rows=rows,pilot_sampling=f"uniform {len(frame)} from full CSV, then uniform {rows} with seed 17",
        pilot_attack_counts=sample.Attack.value_counts().to_dict(),
        pilot_source_row_ids_sha256=hashlib.sha256(sample.source_row_id.to_numpy(dtype="<i8").tobytes()).hexdigest())
    audit_path=directory/f"pilot{rows}_audit.json"
    audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    cfg=json.loads(Path(f"configs/nf_{dataset}_{task}.json").read_text())
    cfg.update(csv=str(csv),source_audit=str(audit_path),data_scope="uniform_pilot",seeds=[7],hidden=16,
        epochs=10,patience=3,train_parts=4,max_edges_per_batch=512,partition_counts=[2,4,8],
        repeats=2,threads=1,isolated_benchmarks=True)
    output=Path(f"configs/{dataset}_{task}_pilot.json")
    output.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    print(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["ton","bot","cse"],required=True); p.add_argument("--rows",type=int,default=5000); p.add_argument("--task",choices=["binary","multiclass"],default="binary")
    a=p.parse_args(); create(a.dataset,a.rows,a.task)
