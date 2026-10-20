"""Add exact original-IP node counts to an existing full-file source audit."""
import argparse
import json
from pathlib import Path
import zipfile
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument("archive"); p.add_argument("--audit",required=True); a=p.parse_args()
path=Path(a.audit); report=json.loads(path.read_text()); hosts=set(); rows=0
with zipfile.ZipFile(a.archive) as z:
    name=next(n for n in z.namelist() if n.endswith(report["dataset"]+".csv"))
    for chunk in pd.read_csv(z.open(name),usecols=["IPV4_SRC_ADDR","IPV4_DST_ADDR"],chunksize=1000000):
        hosts.update(chunk.IPV4_SRC_ADDR.unique()); hosts.update(chunk.IPV4_DST_ADDR.unique()); rows+=len(chunk)
if rows!=report["source_rows"]: raise ValueError("Row count disagrees with original audit")
report["unique_ip_nodes"]=len(hosts)
path.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"dataset":report["dataset"],"rows":rows,"unique_ip_nodes":len(hosts)}))
