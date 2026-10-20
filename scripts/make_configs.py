"""Generate the six explicit task/dataset configurations from the protocol base."""
import json
from pathlib import Path

base=json.loads(Path("configs/nf_ton_binary.json").read_text())
for dataset,file in [("ton","NF-ToN-IoT-v2.csv"),("bot","NF-BoT-IoT-v2.csv"),("cse","NF-CSE-CIC-IDS2018-v2.csv")]:
    for task in ["binary","multiclass"]:
        cfg=dict(base,csv="data/"+file,dataset=dataset,task=task)
        if dataset=="bot":
            cfg.update(train_parts=8,partition_counts=[2,4,8,16,32])
        Path(f"configs/nf_{dataset}_{task}.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
