import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import run
from .report import generate


def demo_config(folder):
    """Artificial flow records for software correctness only; not a benchmark."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1234)
    n=900
    frame=pd.DataFrame({
        "IPV4_SRC_ADDR":[f"10.0.0.{x}" for x in rng.integers(1,81,n)],
        "IPV4_DST_ADDR":[f"10.0.0.{x}" for x in rng.integers(1,81,n)],
        "L4_SRC_PORT":rng.integers(1024,65000,n),"L4_DST_PORT":rng.choice([80,443,502],n),
        "PROTOCOL":rng.choice([6,17],n),"IN_BYTES":rng.lognormal(6,1,n),"OUT_BYTES":rng.lognormal(5,1,n),
        "FLOW_DURATION_MILLISECONDS":rng.lognormal(3,1,n)})
    attack=(frame.IN_BYTES>450).to_numpy()
    frame["Label"]=attack.astype(int)
    frame["Attack"]=np.where(attack,"DDoS","Benign")
    path=folder/"synthetic_flows.csv"; frame.to_csv(path,index=False)
    return {"csv":str(path.resolve()),"dataset":"bot","task":"binary","features":["IN_BYTES","OUT_BYTES","FLOW_DURATION_MILLISECONDS","L4_SRC_PORT","L4_DST_PORT","PROTOCOL"],"categorical":["PROTOCOL"],
        "split_seed":42,"split_protocol":"conversation","seeds":[7,19],"device":"cpu","threads":1,"hidden":16,"layers":2,"dropout":.2,"learning_rate":.01,"weight_decay":.0001,
        "epochs":5,"patience":3,"train_partition_method":"random","train_parts":4,"max_edges_per_batch":128,"methods":["full","random","metis"],"partition_counts":[2,4,8],"repeats":2,"baselines":True,"synthetic":True}


def main():
    p=argparse.ArgumentParser(description="CA-EdgeIDS reproducible edge-classification experiments")
    sub=p.add_subparsers(dest="cmd",required=True)
    demo=sub.add_parser("demo"); demo.add_argument("--out",required=True)
    real=sub.add_parser("run"); real.add_argument("--config",required=True); real.add_argument("--out",required=True)
    real.add_argument("--no-report",action="store_true",help="Save the run without building paper tables")
    report=sub.add_parser("report"); report.add_argument("run_dir")
    preflight=sub.add_parser("preflight"); preflight.add_argument("--config",required=True); preflight.add_argument("--out",required=True)
    profile=sub.add_parser("profile"); profile.add_argument("--run-dir",required=True); profile.add_argument("--out",required=True)
    study=sub.add_parser("batch-study"); study.add_argument("--run-dir",required=True); study.add_argument("--out",required=True)
    args=p.parse_args()
    if args.cmd=="batch-study":
        from .batch_study import run_study
        print(run_study(args.run_dir,args.out)); return
    if args.cmd=="profile":
        from .profile import profile_run
        print(profile_run(args.run_dir,args.out)); return
    if args.cmd=="preflight":
        from .preflight import inspect
        print(json.dumps(inspect(json.loads(Path(args.config).read_text()),args.out),indent=2)); return
    if args.cmd=="report":
        print(generate(args.run_dir)); return
    config=demo_config(Path(args.out).parent/(Path(args.out).name+"-fixture")) if args.cmd=="demo" else json.loads(Path(args.config).read_text())
    result=run(config,args.out)
    if args.cmd=="run" and args.no_report:
        print(result); return
    print(generate(result))


if __name__=="__main__": main()
