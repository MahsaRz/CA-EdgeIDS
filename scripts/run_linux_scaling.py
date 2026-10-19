"""Run the frozen 100,000-flow study locally on Linux."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from ca_edgeids.batch_study import run_study
from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump
from ca_edgeids.preflight import inspect
from ca_edgeids.profile import profile_run
from ca_edgeids.runtime import environment


def execute():
    if platform.system()!="Linux": raise RuntimeError("Run this study in Linux")
    env=environment()
    if env["pymetis"] is None: raise RuntimeError("The real METIS package must be installed")
    plan_path=Path("configs/linux_scaling_100000_plan.json")
    plan=json.loads(plan_path.read_text())
    out=Path("runs/linux-scaling-100000")
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    configurations={}
    for dataset in plan["datasets"]:
        item=plan["inputs"][dataset]; config_path=Path(item["config"])
        if sha256_file(config_path)!=item["config_sha256"]: raise ValueError("Frozen config changed")
        cfg=json.loads(config_path.read_text())
        if sha256_file(cfg["csv"])!=item["sample_csv_sha256"]: raise ValueError("Frozen sample changed")
        check=inspect(cfg,out/"preflight"/dataset)
        if not check["can_train_and_select_all_classes"] or not check["can_evaluate_all_classes"]:
            raise ValueError(f"Binary class coverage failed for {dataset}")
        configurations[dataset]=cfg
        print(f"{dataset}: binary split checks passed",flush=True)
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    cpu_model=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),"unavailable")
    memory_kib=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))
    dump(out/"frozen_plan.json",dict(plan,environment=env,git_commit=commit,
        hardware={"cpu_model":cpu_model,"logical_cpus":os.cpu_count(),"linux_memory_kib":memory_kib,
                  "torch_threads":1,"virtualization":"local Ubuntu under WSL2"},
        driver_source_sha256=sha256_file(__file__),
        package_source_sha256={p.name:sha256_file(p) for p in sorted(Path("src/ca_edgeids").glob("*.py"))}))
    versions={d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata["Name"].lower()!="ca-edgeids"}
    (out/"requirements.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n"+
        "\n".join(f"{name}=={version}" for name,version in sorted(versions.items(),key=lambda x:x[0].lower()))+"\n",encoding="utf-8")
    results=[]
    for dataset in plan["datasets"]:
        train_out=out/"training"/dataset
        print(f"{dataset}: starting training with the frozen settings",flush=True)
        log=out/(dataset+"-training.log")
        command=[sys.executable,"-u","-m","ca_edgeids","run","--config",plan["inputs"][dataset]["config"],
                 "--out",str(train_out),"--no-report"]
        with log.open("w",encoding="utf-8") as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line,end="",flush=True)
            code=process.wait()
        if code:
            dump(out/"failure.json",{"dataset":dataset,"stage":"training","exit_code":code,"local_log":log.name})
            raise RuntimeError(f"Training failed for {dataset}; inspect the local log")
        print(f"{dataset}: starting the fixed-model timing sweep",flush=True)
        study=run_study(train_out,out/"batch-study"/dataset,batch_sizes=plan["batch_sizes"],
            partition_counts=plan["partition_counts"],methods=plan["methods"],repeats=plan["timing_repeats"],
            order_seed=plan["order_seed"],timeout_s=plan["worker_timeout_s"])
        profile_run(train_out,out/"profiles"/dataset,methods=["full","random","metis"])
        completion=json.loads((study/"completion.json").read_text())
        results.append(dict(dataset=dataset,**completion))
        dump(out/"progress.json",results)
        print(f"{dataset}: study complete",flush=True)
    dump(out/"completion.json",{"datasets":results,"scope":"100,000-flow binary systems pilots; one model seed per dataset"})
    print(out,flush=True)


if __name__=="__main__": execute()
