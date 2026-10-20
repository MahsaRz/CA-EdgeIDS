"""Publish aggregate evidence only: never raw flows, predictions, or checkpoints."""
import argparse
import json
from pathlib import Path
import shutil


def export(run,tag):
    source=Path(run)
    dest=Path("artifacts")/"pilots"/tag
    dest.mkdir(parents=True,exist_ok=True)
    for name in ["requested_config.json","resolved_config.json","dataset_manifest.json","environment.json","code_manifest.json","protocol_checks.json","graph_statistics.json","benchmark_runs.csv","baseline_runs.csv","unavailable_runs.json"]:
        if (source/name).is_file(): shutil.copyfile(source/name,dest/name)
    shutil.copytree(source/"paper_outputs",dest/"paper_outputs",dirs_exist_ok=True)
    for seed_dir in source.glob("seed-*"):
        sub=dest/seed_dir.name; sub.mkdir(exist_ok=True)
        for path in seed_dir.iterdir():
            if path.name.endswith("-metrics.json") or path.name in ["training_summary.json","training_history.csv"]:
                shutil.copyfile(path,sub/path.name)
    for path in source.glob("baseline-*.json"): shutil.copyfile(path,dest/path.name)
    (dest/"SCOPE.txt").write_text("REAL-DATA PILOT, NOT FINAL PAPER RESULTS.\nSmall uniform sample; one training seed; initial untuned settings.\nGraphs and classifier are under development. Exact inference coverage is the validated property; a speedup or state-of-the-art accuracy is not established.\nMissing METIS configurations are explicitly listed. Compare baselines only as diagnostic results; tuning budgets and baseline process isolation are not yet matched.\nNo raw flows, per-flow predictions, or model checkpoint is included in this published bundle.\n",encoding="utf-8")
    return dest


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--tag",required=True); a=p.parse_args(); print(export(a.run,a.tag))
