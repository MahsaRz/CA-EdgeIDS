"""Check study coverage and source snapshots before publishing aggregates."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def verify():
    root=Path("runs/linux-scaling-100000")
    frozen=json.loads((root/"frozen_plan.json").read_text())
    completion=json.loads((root/"completion.json").read_text())
    commit=frozen["git_commit"]
    for name,expected in frozen["package_source_sha256"].items():
        content=subprocess.run(["git","show",f"{commit}:src/ca_edgeids/{name}"],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError("Package source snapshot differs from its recorded commit")
    driver=subprocess.run(["git","show",f"{commit}:scripts/run_linux_scaling.py"],capture_output=True,check=True).stdout
    if hashlib.sha256(driver).hexdigest()!=frozen["driver_source_sha256"]: raise ValueError("Runner snapshot differs")
    expected_conditions={("full",1,0)}|{(method,part,batch) for method in frozen["methods"]
        for part in ([1] if method=="chunked" else frozen["partition_counts"]) for batch in frozen["batch_sizes"]}
    report=[]
    for dataset in frozen["datasets"]:
        train=root/"training"/dataset; study=root/"batch-study"/dataset
        manifest=json.loads((train/"dataset_manifest.json").read_text())
        training_code=json.loads((train/"code_manifest.json").read_text())["package_source_sha256"]
        study_plan=json.loads((study/"study_plan.json").read_text())
        if training_code!=frozen["package_source_sha256"] or study_plan["package_source_sha256"]!=training_code:
            raise ValueError("Model code changed during the study")
        checks=json.loads((study/"prediction_checks.json").read_text())
        failures=json.loads((study/"unavailable_runs.json").read_text())
        key=lambda r:(r["method"],r["partitions"],r["batch_size"])
        checked={key(r) for r in checks}; failed={key(r) for r in failures}
        if checked&failed or checked|failed!=expected_conditions or len(checked)!=len(checks) or len(failed)!=len(failures):
            raise ValueError("Missing or duplicate study conditions")
        if not {"full","chunked","random","metis"}.issubset({r["method"] for r in checks}):
            raise ValueError("A required real-data method is missing")
        with (study/"timing_repeats.csv").open(newline="") as f: rows=list(csv.DictReader(f))
        counts={condition:set() for condition in checked}
        for row in rows:
            condition=(row["method"],int(row["partitions"]),int(row["batch_size"]))
            if condition not in counts: raise ValueError("Unexpected timing condition")
            repeat=int(row["repeat"])
            if repeat in counts[condition]: raise ValueError("Duplicate timing repeat")
            counts[condition].add(repeat)
            if row["target_row_ids_sha256"]!=manifest["test_row_ids_sha256"] or int(row["coverage_min"])!=1 or int(row["coverage_max"])!=1:
                raise ValueError("Prediction population changed")
            if not 0<float(row["first_prediction_process_s"])<=float(row["worker_process_wall_s"]):
                raise ValueError("Invalid process timing")
        if any(repeats!=set(range(frozen["timing_repeats"])) for repeats in counts.values()):
            raise ValueError("Missing timing repeats")
        if not all(r["same_class_predictions"] for r in checks): raise ValueError("Class predictions changed")
        worst=max(r["max_probability_difference"] for r in checks)
        if worst>2e-5: raise ValueError("Probability difference exceeds the allowed bound")
        report.append(dict(dataset=dataset,completed_conditions=len(checks),unavailable_conditions=len(failures),
                           timing_rows=len(rows),maximum_probability_difference=worst))
    result={"source_commit":commit,"source_snapshots_match":True,"fixed_test_coverage":True,
            "same_class_predictions":True,"real_data_metis_in_each_dataset":True,"datasets":report}
    (root/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__": verify()
