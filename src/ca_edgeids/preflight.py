"""Aggregate split feasibility before training; never search seeds with test scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import prepare, SOURCE, DEST


def inspect(config, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior preflight: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(config)
    pairs = [tuple(sorted((str(s), str(d)))) for s, d in zip(data.frame[SOURCE], data.frame[DEST])]
    rows = []
    for label_id, name in enumerate(data.names):
        row = {"class": name}
        for part in ["train", "validation", "test"]:
            ids = np.flatnonzero((data.labels == label_id) & (data.split == part))
            row[f"{part}_flows"] = len(ids)
            row[f"{part}_host_pairs"] = len({pairs[i] for i in ids})
        row["all_host_pairs"] = len({pairs[i] for i in np.flatnonzero(data.labels == label_id)})
        rows.append(row)
    missing = {p: [r["class"] for r in rows if r[f"{p}_flows"] == 0]
               for p in ["train", "validation", "test"]}
    host_sets = {}
    for part in ["train", "validation", "test"]:
        f = data.frame.loc[data.split == part]
        host_sets[part] = set(f[SOURCE]) | set(f[DEST])
    host_overlap = {f"{a}/{b}": len(host_sets[a] & host_sets[b])
                    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    report = {
        "dataset": config["dataset"], "task": config["task"],
        "scope": config.get("data_scope", "unspecified"),
        "missing_classes": missing,
        "can_train_and_select_all_classes": not missing["train"] and not missing["validation"],
        "can_evaluate_all_classes": not missing["test"],
        "nodes_per_split": {p: len(h) for p, h in host_sets.items()},
        "shared_hosts_between_splits": host_overlap,
        "interpretation": "Host-pair isolation permits shared hosts. Counts indicate feasibility, not adequate statistical power. Do not choose seeds or merge classes to improve test outcomes.",
        "class_support": rows,
    }
    for name, value in [("requested_config.json", config), ("dataset_manifest.json", data.manifest), ("preflight.json", report)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "class_split_support.csv", index=False)
    return report
