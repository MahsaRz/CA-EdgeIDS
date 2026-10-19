"""Train on train, select by validation, evaluate a frozen test population."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from .data import prepare
from .graph import build_graph,partition_nodes,query_batches,halo_batch
from .model import EdgeSAGE
from .metrics import evaluate
from .runtime import seed_all,sync,PeakRSS,environment


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def infer(model,g,owner,config,full_graph=False):
    """All original targets, including cut edges, exactly once. No label enters model."""
    device = config["device"]
    model.eval()
    probabilities = np.empty((len(g.src),config["classes"]),dtype=np.float32)
    seen = np.zeros(len(g.src),dtype=np.int64)
    forward_s = 0.0
    packing_s = 0.0
    partition_s = {}
    batch_count = 0
    if str(device).startswith("cuda"):
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    with PeakRSS() as mem, torch.inference_mode():
        for part,q in query_batches(g,owner,len(g.src) if full_graph else config["max_edges_per_batch"]):
            a = time.perf_counter()
            b = halo_batch(g,q,model.depth,device,config.get("max_batch_tensor_bytes",256*1024**2))
            sync(device)
            packing_s += time.perf_counter()-a
            a = time.perf_counter()
            logits = model(b)
            sync(device)
            dt = time.perf_counter()-a
            forward_s += dt
            partition_s[part] = partition_s.get(part,0)+dt
            probabilities[q] = logits.softmax(1).cpu().numpy()
            seen[q] += 1
            batch_count += 1
    sync(device)
    total = time.perf_counter()-start
    if not np.all(seen == 1):
        raise AssertionError("Missing or duplicated flow predictions")
    return probabilities,{
        "model_forward_s":forward_s,"halo_and_transfer_s":packing_s,"total_inference_s":total,
        "average_partition_forward_s":forward_s/(int(owner.max())+1),
        "nonempty_target_partitions":len(partition_s),"batches":batch_count,
        "peak_cpu_rss_mb_sampled":mem.peak/2**20,"baseline_cpu_rss_mb":mem.baseline/2**20,
        "peak_gpu_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith("cuda") else None,
        "rss_sampling_interval_ms":5,"coverage_min":int(seen.min()),"coverage_max":int(seen.max()),
        "target_row_ids_sha256":hashlib.sha256(g.row_ids.astype("<i8").tobytes()).hexdigest(),
        "empty_partitions_in_average":True,
    }


def train(train_graph,val_graph,config,seed,out):
    seed_all(seed,config.get("threads",1))
    owner = partition_nodes(train_graph,config["train_partition_method"],config["train_parts"],seed)
    val_owner = partition_nodes(val_graph,"random",min(config["train_parts"],len(val_graph.hosts)),seed)
    model = EdgeSAGE(train_graph.edge_x.shape[1],config["classes"],config["hidden"],config["layers"],config["dropout"]).to(config["device"])
    opt = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    counts = np.bincount(train_graph.y,minlength=config["classes"])
    if np.any(counts==0):
        raise ValueError("Training split lacks a declared class. Report this split limitation, do not select a seed using test labels.")
    weights = torch.tensor(len(train_graph.y)/(len(counts)*counts),dtype=torch.float32,device=config["device"])
    best,best_epoch,stale,state = -1,-1,0,None
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(config["epochs"]):
        model.train()
        batches = list(query_batches(train_graph,owner,config["max_edges_per_batch"]))
        rng.shuffle(batches)
        total_loss = 0.0
        for _,q in batches:
            batch = halo_batch(train_graph,q,model.depth,config["device"],config.get("max_batch_tensor_bytes",256*1024**2))
            opt.zero_grad(set_to_none=True)
            # Weights really enter the loss; unlike the legacy notebooks.
            loss = F.cross_entropy(model(batch),batch.y,weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            total_loss += loss.item()*len(q)
        p,_ = infer(model,val_graph,val_owner,config)
        score = evaluate(val_graph.y,p,config["label_names"])["macro_f1"]
        history.append({"epoch":epoch+1,"train_loss":total_loss/len(train_graph.y),"validation_macro_f1":score})
        if score > best:
            best,best_epoch,stale = score,epoch+1,0
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(state)
    torch.save({"state_dict":state,"config":config,"seed":seed,"best_epoch":best_epoch,"validation_macro_f1":best},out/"model.pt")
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    return model,{"best_epoch":best_epoch,"epochs_run":len(history),"validation_macro_f1":best,"class_weights":weights.cpu().tolist()}


def run(config,out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior run: {out}")
    out.mkdir(parents=True,exist_ok=True)
    config = copy.deepcopy(config)
    seed_all(config["seeds"][0],config.get("threads",1))
    dump(out/"requested_config.json",config)
    config.setdefault("max_input_bytes",512*1024**2)
    config.setdefault("max_batch_tensor_bytes",256*1024**2)
    dump(out/"environment.json",environment())
    # Snapshot hashes of the actual Python sources, including uncommitted edits.
    source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    dump(out/"code_manifest.json",{"package_source_sha256":source_files,
        "combined_sha256":hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()})
    t = time.perf_counter()
    data = prepare(config)
    prep_s = time.perf_counter()-t
    config.update(classes=len(data.names),label_names=data.names)
    dump(out/"resolved_config.json",config)
    dump(out/"dataset_manifest.json",data.manifest)
    missing_training=[name for name,count in data.manifest["class_counts"]["train"].items() if count==0]
    missing_validation=[name for name,count in data.manifest["class_counts"]["validation"].items() if count==0]
    missing_test=[name for name,count in data.manifest["class_counts"]["test"].items() if count==0]
    dump(out/"protocol_checks.json",{"missing_training_classes":missing_training,
        "missing_validation_classes":missing_validation,"missing_test_classes":missing_test,
        "valid_for_all_class_validation_selection":not missing_training and not missing_validation,
        "valid_for_all_class_test_evaluation":not missing_test})
    if missing_training:
        raise ValueError(f"Training split lacks {missing_training}. Cannot fit a declared all-class classifier; review split feasibility before building graphs.")
    if missing_validation and config.get("require_validation_class_coverage",True):
        raise ValueError(f"Validation split lacks {missing_validation}. All-class model selection is unsupported; increase the label-independent pilot sample or review the split protocol. No test-driven seed search.")
    import joblib
    joblib.dump(data.preprocessor,out/"preprocessor.joblib")
    pd.DataFrame({"row_id":data.frame["source_row_id"] if "source_row_id" in data.frame else np.arange(len(data.frame)),"split":data.split,"class_id":data.labels}).to_csv(out/"split_manifest.csv.gz",index=False)
    t = time.perf_counter()
    graphs = {part:build_graph(data,part,config.get("bidirectional_messages",True)) for part in ["train","validation","test"]}
    graph_s = time.perf_counter()-t
    dump(out/"graph_statistics.json",{p:{"nodes":len(g.hosts),"target_flows":len(g.src),"message_arcs":len(g.message_src)} for p,g in graphs.items()})
    all_rows, failures = [],[]
    g = graphs["test"]
    if config.get("isolated_benchmarks",False):
        from .worker import save_graph
        save_graph(g,out/"test_graph")
    for seed in config["seeds"]:
        seed_out = out/f"seed-{seed}"
        seed_out.mkdir()
        t = time.perf_counter()
        model,training = train(graphs["train"],graphs["validation"],config,seed,seed_out)
        train_s = time.perf_counter()-t
        dump(seed_out/"training_summary.json",training)
        reference = None
        for method in config["methods"]:
            for parts in ([1] if method=="full" else config["partition_counts"]):
                tag = f"{method}-{parts}"
                try:
                    t = time.perf_counter()
                    owner = partition_nodes(g,method,parts,seed)
                    partition_s = time.perf_counter()-t
                    np.save(seed_out/f"{tag}-ownership.npy",owner)
                    isolated_records=None
                    if config.get("isolated_benchmarks",False):
                        worker_out=seed_out/f"{tag}-worker"
                        child=subprocess.run([sys.executable,"-m","ca_edgeids.worker","--graph",str((out/"test_graph").resolve()),"--checkpoint",str((seed_out/"model.pt").resolve()),"--method",method,"--parts",str(parts),"--out",str(worker_out.resolve())],capture_output=True,text=True)
                        if child.returncode:
                            raise RuntimeError(child.stderr)
                        isolated_records=json.loads(Path(str(worker_out)+".json").read_text())
                        with np.load(str(worker_out)+".npz",allow_pickle=False) as result:
                            isolated_p=result["probabilities"]
                    else:
                        # Warm-up costs are recorded separately from steady-state timing.
                        warm = time.perf_counter()
                        first_q = next(query_batches(g,owner,len(g.src) if method=="full" else config["max_edges_per_batch"]))[1]
                        with torch.inference_mode():
                            model.eval()
                            model(halo_batch(g,first_q,model.depth,config["device"],config.get("max_batch_tensor_bytes",256*1024**2)))
                        sync(config["device"])
                        warm_s = time.perf_counter()-warm
                    for repeat in range(config["repeats"]):
                        if isolated_records is not None:
                            p=isolated_p
                            timing=isolated_records[repeat].copy()
                            timing.pop("repeat")
                            partition_s=timing.pop("partitioning_s")
                            warm_s=timing.pop("warmup_s")
                        else:
                            p,timing = infer(model,g,owner,config,full_graph=method=="full")
                            timing["memory_process"]="shared benchmark process; exploratory memory only"
                        if reference is None:
                            reference = p.copy()
                        if not np.allclose(p,reference,atol=1e-5,rtol=1e-5):
                            raise AssertionError("Exact-halo inference changed scores across partition configurations")
                        scores = evaluate(g.y,p,data.names)
                        record = dict(seed=seed,method=method,partitions=parts,repeat=repeat,
                            scope="synthetic_smoke" if config.get("synthetic") else config.get("data_scope","prefix_pilot" if config.get("max_rows") else "full_dataset"),
                            comparison="fixed-checkpoint inference; not a partition-training ablation",
                            preprocessing_s=prep_s,graph_construction_s=graph_s,training_s=train_s,
                            partitioning_s=partition_s,warmup_s=warm_s,**timing,
                            offline_pipeline_excluding_training_s=prep_s+graph_s+partition_s+warm_s+timing["total_inference_s"],
                            **{k:v for k,v in scores.items() if k not in {"per_class","confusion_matrix","average_precision_ovr","fpr_ovr","missing_test_classes","pr_metric_definition"}})
                        # Pipeline cost above includes reading/splitting ALL data; it is not online latency.
                        all_rows.append(record)
                        if repeat==0:
                            dump(seed_out/f"{tag}-metrics.json",scores)
                            np.savez_compressed(seed_out/f"{tag}-predictions.npz",row_ids=g.row_ids,y=g.y,probabilities=p)
                except (RuntimeError,ValueError,MemoryError) as exc:
                    if isinstance(exc,RuntimeError) and not any(marker in str(exc) for marker in ["out of memory", "METIS unavailable", "Configured batch tensor limit exceeded"]):
                        raise
                    failures.append({"seed":seed,"method":method,"partitions":parts,"error":str(exc)})
                    if str(config["device"]).startswith("cuda"):
                        torch.cuda.empty_cache()
    pd.DataFrame(all_rows).to_csv(out/"benchmark_runs.csv",index=False)
    dump(out/"unavailable_runs.json",failures)
    baseline_rows=[]
    if config.get("baselines",True):
        for seed in config["seeds"]:
            for name,estimator in [
                ("RF",RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=seed,n_jobs=config.get("threads",1))),
                ("MLP",MLPClassifier(hidden_layer_sizes=(64,64),max_iter=config["epochs"],random_state=seed,early_stopping=False)),
            ]:
                tr = data.split=="train"
                te = data.split=="test"
                t=time.perf_counter()
                estimator.fit(data.features[tr],data.labels[tr])
                fit_s=time.perf_counter()-t
                with PeakRSS() as mem:
                    t=time.perf_counter()
                    raw=estimator.predict_proba(data.features[te])
                    dt=time.perf_counter()-t
                p=np.zeros((np.sum(te),len(data.names)))
                p[:,estimator.classes_.astype(int)]=raw
                metric=evaluate(data.labels[te],p,data.names)
                dump(out/f"baseline-{name}-{seed}.json",metric)
                baseline_rows.append({"seed":seed,"method":name,"macro_f1":metric["macro_f1"],"weighted_f1":metric["weighted_f1"],"fit_s":fit_s,"total_inference_s":dt,"peak_cpu_rss_mb_sampled":mem.peak/2**20,"test_n":metric["test_n"],"selection":"fixed settings; no test tuning"})
    pd.DataFrame(baseline_rows).to_csv(out/"baseline_runs.csv",index=False)
    return out
