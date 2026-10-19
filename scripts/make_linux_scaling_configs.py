"""Freeze the larger binary pilot before collecting model results."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""): h.update(block)
    return h.hexdigest()


def create():
    inputs={}
    for dataset in ["ton","cse","bot"]:
        source=Path("data/scaling100000")/dataset
        audit=json.loads((source/"source_audit.json").read_text())
        original=json.loads(Path(f"docs/provenance/nf-{dataset}-v2.json").read_text())
        for key in ["archive_sha256","csv_sha256","source_rows","original_attack_counts"]:
            if audit[key]!=original[key]: raise ValueError(f"Source mismatch: {dataset} {key}")
        if audit["pilot_rows"]!=100000 or not audit["bagit_sha1_verified"]:
            raise ValueError("Expected a checksum-verified 100,000-flow sample")
        sample=source/(audit["dataset"]+"-uniform-pilot.csv")
        config=json.loads(Path("configs/ton_binary_scaling_30000.json").read_text())
        config.update(dataset=dataset,csv=sample.as_posix(),source_audit=(source/"source_audit.json").as_posix(),
            methods=["full"],baselines=False,log_progress=True,require_validation_class_coverage=True,
            max_input_bytes=512*1024**2,max_batch_tensor_bytes=256*1024**2)
        path=Path(f"configs/{dataset}_binary_linux_100000.json")
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8",newline="\n")
        inputs[dataset]={"config":path.as_posix(),"config_sha256":digest(path),"sample_csv_sha256":digest(sample),
            "sample_rows":100000,"sample_row_ids_sha256":audit["pilot_source_row_ids_sha256"],
            "archive_sha256":audit["archive_sha256"]}
    plan={"name":"linux-binary-scaling-100000","datasets":["ton","cse","bot"],"inputs":inputs,
          "sampling_seed":2026,"split_seed":42,"model_seed":7,"batch_sizes":[256,512,1024,2048],
          "partition_counts":[2,4,8],"methods":["chunked","random","metis"],"timing_repeats":3,
          "order_seed":20261019,"worker_timeout_s":120,
          "scope":"One new binary model per dataset. Freeze that checkpoint across all inference conditions. No test-driven tuning; no five-seed or full-dataset claim.",
          "training":"Same initial 10-epoch, patience-3, hidden-16, 2-layer, batch-512 settings as the 30,000-flow pilots. Training uses random source ownership and exact context. No RF/MLP fitting in this systems pilot.",
          "timing":"Fresh process per condition; one complete warm-up pass; three whole-test-graph repeats. Record total inference, forward, packing, partitioning and sampled RSS separately. Loading and preparation are outside warm inference.",
          "failures":"Retain infeasible partition counts, configured limits, timeouts and memory failures. Stop on a prediction or coverage mismatch."}
    target=Path("configs/linux_scaling_100000_plan.json")
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(target)


if __name__=="__main__": create()
