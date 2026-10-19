"""Explicit NetFlow-v2 schemas and deterministic, label-free grouped splits."""
from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SOURCE = "IPV4_SRC_ADDR"
DEST = "IPV4_DST_ADDR"
LABEL_COLUMNS = {"Label", "Attack", "label", "attack", "category", "subcategory", "type"}
SCHEMAS = {
    "ton": ["Benign", "Backdoor", "DDoS", "DoS", "Injection", "MITM", "Password", "Ransomware", "Scanning", "XSS"],
    "bot": ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"],
    "cse": ["Benign", "Bot", "BruteForce", "DDoS", "DoS", "Infiltration", "Web Attacks"],
}
# Explicit mapping verified against the official CSE v2 archive and UQ's seven-family summary.
# The original CSV names remain in manifests; this is family classification, not 15-way classification.
CSE_V2_LABEL_MAP = {
    "Benign":"Benign", "Bot":"Bot", "Infilteration":"Infiltration",
    "SSH-Bruteforce":"BruteForce", "FTP-BruteForce":"BruteForce",
    "DDOS attack-HOIC":"DDoS", "DDoS attacks-LOIC-HTTP":"DDoS", "DDOS attack-LOIC-UDP":"DDoS",
    "DoS attacks-Hulk":"DoS", "DoS attacks-GoldenEye":"DoS", "DoS attacks-SlowHTTPTest":"DoS", "DoS attacks-Slowloris":"DoS",
    "Brute Force -Web":"Web Attacks", "Brute Force -XSS":"Web Attacks", "SQL Injection":"Web Attacks",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_bucket(value, seed):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big") / 2**64


def normalize_labels(frame, dataset, task):
    required = {SOURCE, DEST, "Label", "Attack"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected NetFlow-v2 columns; missing {sorted(required - set(frame.columns))}. No legacy schema substitution.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing endpoints or labels are not silently imputed.")
    for col in [SOURCE, DEST]:
        for value in frame[col].unique():
            ipaddress.ip_address(str(value))
    canonical = {x.casefold(): x for x in SCHEMAS[dataset]}
    canonical.update({"normal": "Benign", "benign": "Benign"})
    if dataset=="cse":
        canonical.update({k.casefold():v for k,v in CSE_V2_LABEL_MAP.items()})
    raw = frame["Attack"].astype(str).str.strip()
    mapped = raw.str.casefold().map(canonical)
    if mapped.isna().any():
        raise ValueError(f"Unmapped attack labels: {sorted(raw[mapped.isna()].unique())}. Review mapping explicitly.")
    binary = pd.to_numeric(frame["Label"], errors="raise").to_numpy()
    if not np.isin(binary, [0, 1]).all() or not np.array_equal(binary, (mapped != "Benign").astype(int)):
        raise ValueError("Binary Label and Attack annotations disagree.")
    names = ["Benign", "Attack"] if task == "binary" else SCHEMAS[dataset]
    labels = binary.astype(np.int64) if task == "binary" else mapped.map(dict(zip(names, range(len(names))))).to_numpy(np.int64)
    return labels, names, raw.to_numpy()


def split_rows(frame, seed=42, protocol="conversation", train=0.6, validation=0.2, timestamp=None):
    """No stratification or repeated draws using test labels; report absent classes."""
    if not (0 < train < 1 and 0 < validation < 1 and train + validation < 1):
        raise ValueError("Invalid split fractions")
    if protocol == "temporal":
        if timestamp is None or timestamp not in frame:
            raise ValueError("Temporal splitting requires a real timestamp column; row order is not time.")
        t = pd.to_datetime(frame[timestamp], utc=True, errors="raise")
        if t.isna().any() or t.nunique() < 3:
            raise ValueError("At least three distinct, valid timestamps required")
        ordered = np.sort(t.unique())
        a = ordered[max(1, int(len(ordered) * train))]
        b = ordered[min(len(ordered)-1, max(2, int(len(ordered)*(train+validation))))]
        out = np.where(t < a, "train", np.where(t < b, "validation", "test"))
    elif protocol in {"conversation", "host"}:
        def bucket(key):
            z = stable_bucket(key, seed)
            return "train" if z < train else "validation" if z < train + validation else "test"
        if protocol == "conversation":
            # Group ALL traffic between the same unordered IP pair, regardless of port/protocol.
            # This is a pair-held-out protocol, NOT an unseen-host or episode claim.
            keys = ["|".join(sorted((str(s), str(d)))) for s,d in zip(frame[SOURCE], frame[DEST])]
            cache = {k: bucket(k) for k in set(keys)}
            out = np.array([cache[k] for k in keys])
        else:
            hosts = set(frame[SOURCE]) | set(frame[DEST])
            cache = {h: bucket(h) for h in hosts}
            out = np.array([cache[s] if cache[s] == cache[d] else "excluded_cross_host" for s,d in zip(frame[SOURCE],frame[DEST])])
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")
    if not all(np.any(out == part) for part in ["train", "validation", "test"]):
        raise ValueError("Empty split. Do not silently relax group isolation; review dataset suitability.")
    return out


@dataclass
class Prepared:
    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    names: list
    split: np.ndarray
    manifest: dict
    preprocessor: object


def prepare(config):
    path = Path(config["csv"])
    file_limit=config.get("max_input_bytes",512*1024**2)
    if file_limit <= 0:
        raise ValueError("max_input_bytes must be positive")
    if path.stat().st_size > file_limit:
        raise MemoryError("Input CSV exceeds the configured in-memory file limit. Run the streaming source audit and a scaling pilot before increasing max_input_bytes. File bytes are not a RAM estimate.")
    frame = pd.read_csv(path, low_memory=False, nrows=config.get("max_rows"))
    frame.columns = frame.columns.str.strip()
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names")
    y, names, original = normalize_labels(frame, config["dataset"], config["task"])
    features = config["features"]
    if not features or len(set(features)) != len(features):
        raise ValueError("An explicit unique feature allowlist is required")
    forbidden = LABEL_COLUMNS | {SOURCE, DEST, config.get("timestamp"), "row_id", "source_row_id", "split"}
    if set(features) & forbidden:
        raise ValueError(f"Forbidden features: {set(features) & forbidden}")
    if not set(features).issubset(frame):
        raise ValueError(f"Missing selected features: {set(features)-set(frame)}")
    split = split_rows(frame, config["split_seed"], config["split_protocol"], timestamp=config.get("timestamp"))
    categorical = config.get("categorical", [])
    if not set(categorical).issubset(features):
        raise ValueError("Categorical columns must be selected features")
    numeric = [c for c in features if c not in categorical]
    x = frame[features].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="raise").replace([np.inf,-np.inf], np.nan)
    for col in categorical:
        x[col] = x[col].fillna("<missing>").astype(str)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())]),numeric))
    if categorical:
        transformers.append(("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical))
    pre = ColumnTransformer(transformers)
    pre.fit(x.loc[split == "train"])
    z = pre.transform(x).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite transformed features")
    # Strict duplicate-content audit, including endpoint identity, excludes labels from fingerprints.
    fp = pd.util.hash_pandas_object(frame[[SOURCE,DEST]+features].astype(str), index=False).to_numpy()
    fps = {part: set(fp[split==part]) for part in ["train","validation","test"]}
    overlap = {f"{a}/{b}":len(fps[a]&fps[b]) for a,b in [("train","test"),("train","validation"),("validation","test")]}
    if any(overlap.values()):
        raise ValueError(f"Duplicate feature records cross splits: {overlap}. Deduplicate/group before this protocol.")
    original_ids = frame["source_row_id"].to_numpy(np.int64) if "source_row_id" in frame else np.arange(len(frame),dtype=np.int64)
    if len(np.unique(original_ids))!=len(frame):
        raise ValueError("Original source row IDs must be unique")
    manifest = {
        "dataset":config["dataset"], "task":config["task"], "csv_sha256":sha256_file(path),
        "rows_read":len(frame), "max_rows":config.get("max_rows"), "feature_columns":features,
        "categorical_columns":categorical, "transformed_width":z.shape[1], "label_names":names,
        "original_labels":sorted(set(original)), "split_protocol":config["split_protocol"],
        "explicit_family_mapping":CSE_V2_LABEL_MAP if config["dataset"]=="cse" else "case normalization only; original names preserved",
        "split_seed":config["split_seed"], "duplicate_overlap":overlap,
        "split_counts":{p:int(np.sum(split==p)) for p in np.unique(split)},
        "class_counts":{p:dict(zip(names,np.bincount(y[split==p],minlength=len(names)).tolist())) for p in ["train","validation","test"]},
        "test_row_ids_sha256":hashlib.sha256(original_ids[split=="test"].astype("<i8").tobytes()).hexdigest(),
        "data_scope":config.get("data_scope","full_dataset"),
        "max_input_bytes":file_limit,
        "scope":"offline joint snapshot classification; no online, causal, or unseen-host claim unless separately tested",
    }
    if config.get("source_audit"):
        audit=json.loads(Path(config["source_audit"]).read_text())
        if audit["pilot_rows"]!=len(frame) or hashlib.sha256(original_ids.astype("<i8").tobytes()).hexdigest()!=audit["pilot_source_row_ids_sha256"]:
            raise ValueError("Pilot source row IDs do not match source audit")
        manifest["source_audit"]=audit
    return Prepared(frame,z,y,names,split,manifest,pre)
