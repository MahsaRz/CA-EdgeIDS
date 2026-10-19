# CA-EdgeIDS - research rebuild

**Working title:** *Cluster-Aware GraphSAGE for Flow-Level Intrusion Detection: Accuracy, Coverage, and End-to-End Cost*.

This branch rebuilds the original DC-RCA experiments as **flow/edge intrusion classification**. It does not claim causal root-cause identification, physical fault localization, distributed execution, or proven real-time operation. The six original notebooks and `Readme.txt` remain historical reference material; they are not the revised implementation.

The new package is a tested starting point, **not a completed journal evaluation**. A successful synthetic demonstration is software evidence only. Publication requires real-data schema verification, complete experiments, scientific contribution assessment, and the gates in [the phased research plan](docs/RESEARCH_PLAN.md).

## What is implemented

- Explicit standardized NetFlow-v2 schema and label checks. Binary targets come from `Label`; attack-type targets come from `Attack`. No target fields are allowed as features.
- Original IP identities; one target edge per original directed flow record. Optional reverse arcs are message-passing context, not additional labeled samples.
- Label-free node degree features, numerical flow features, and train-fitted categorical encoders, imputers, and scaling.
- Deterministic 60/20/20 host-pair-grouped splits. Separate host-held-out and true timestamp-based protocols are available. Ratios are approximate for group isolation; class supports are reported without selecting seeds using test outcomes.
- Edge-conditioned GraphSAGE-style mean message passing and an edge decoder, with weighted cross entropy actually applied, validation macro-F1 checkpoint selection, and seeded initialization.
- Full-context, balanced random, and optional METIS partition inference. Every target flow is owned by its source-node partition. Both endpoints receive exact incoming K-hop context, including across partition boundaries.
- Fixed test row IDs; assertions for exactly-once coverage and partition-invariant probabilities for a frozen checkpoint.
- Repeated synchronized timings; complete inference loop versus forward calls; partition/preprocessing/graph costs; sampled CPU RSS and GPU allocation peaks.
- Per-class/macro/weighted metrics, probability-based average precision, false-positive rates, confusion matrices, RF and MLP reference baselines, machine-readable manifests, checkpoints, and CSV/LaTeX/SVG/PNG output generation.

Exact halos may grow to much of the graph and can repeat work. They prioritize correctness; they **do not guarantee a speedup or bounded memory**, and are not presented as a new algorithm. METIS is never silently replaced with random partitioning.

## Quick start

Use Python 3.11+ in a dedicated environment. PyTorch CPU is sufficient for correctness tests. Linux is recommended for METIS and full dataset work.

```text
python -m pip install -e ".[test,metis]"
python -m pytest -q
python -m ca_edgeids demo --out runs/synthetic-check
```

If Windows has no compatible PyMETIS package, install `.[test]` and run Full/Random locally; METIS runs are explicitly recorded as unavailable. The Linux GitHub workflow installs the METIS extra and exercises it.

The demo generates artificial flow records and marked plots. **Never copy demo values into the manuscript.** Run directories refuse to overwrite previous experiments.

## Real datasets

**Multiclass feasibility update:** complete-source checks of the frozen host-pair split found missing validation classes in all three datasets; CSE BruteForce has only two host pairs in total. Multiclass fitting is paused at this protocol gate. See [the measured audit and next-step decision](docs/MULTICLASS_FEASIBILITY.md). Binary experiments remain a separate task.

Check a proposed configuration before training:

```text
python -m ca_edgeids preflight --config configs/ton_multiclass_pilot.json --out runs/ton-multiclass-preflight
```

The older README links to the original dataset families. The standardized v2 files must be obtained from the dataset authors' [University of Queensland release page](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets). See [data provenance](docs/DATA_SOURCES.md). Keep raw data outside Git history. Never substitute original BoT/ToN schemas under the NetFlow-v2 name.

`configs/nf_ton_binary.json` is an initial **protocol configuration**, not a tuned final model. It uses an explicit 10-column feature subset; it does not claim to use all 43 extended fields. Confirm the actual header, original attack labels, file hashes, and memory needs first. The same pipeline supports `dataset` values `ton`, `bot`, `cse` and `task` values `binary`, `multiclass`; use separate named configurations and run directories for each.

```text
python -m ca_edgeids run --config configs/nf_ton_binary.json --out runs/ton-binary-seed-study
python -m ca_edgeids report runs/ton-binary-seed-study
```

Configs can use `max_rows` for **explicitly marked prefix pilots only**. Prefix pilots are not representative random samples and must not be reported as full-dataset results. If label spelling differs, stop and review the schema/mapping rather than discarding classes.

Runs also stop if a declared class is absent from validation, because checkpoint selection could not assess that class. The retained 5,000-row BoT pilot predates this guard and has no Benign validation samples; it is a documented protocol failure, not a valid study result.

## Interpretation and limitations

[Graph-work measurements](docs/GRAPH_WORK.md) now separate small-batch repetition from partitioning. New graph files use read-only memory-mapped arrays. Default file-size and packed-tensor limits stop oversized in-memory jobs, but do not guarantee a total RAM bound. Older pilot results keep their original storage and timing definitions.

The present partition sweep holds the checkpoint fixed. Exact halo inference should preserve accuracy; Tables C/D therefore examine **inference scheduling**, not accuracy improvements due to training partitions. Separate training-partition experiments, topology/edge-feature ablations, a direct Cluster-GCN training baseline, and GraphSAINT remain research tasks in the plan. The RF/MLP baselines have fixed initial settings and are not a final equal-budget tuned comparison.

Graphs are built independently for train/validation/test and use all features of a split snapshot jointly. Test labels never enter representations, but test features participate in message passing. This is an **offline snapshot evaluation**. It is not causal streaming inference. No timestamp is invented where the NetFlow release lacks one.

The mean per-partition forward time divides total forward time by the requested number of partitions, including empty target partitions. `max_edges_per_batch` may create several microbatches within a partition; total inference includes their packing and result collection. A genuine Full baseline performs one forward pass over the entire test graph and can run out of memory; failures are retained.

CPU memory is process RSS sampled at 5 ms, including resident data, not an allocator-exact transient peak. `isolated_benchmarks: true` runs each method/partition configuration in a fresh process, including separate warm-up and repeated inference. The synthetic demo uses a shared process for speed and its memory comparisons are exploratory. Process startup and checkpoint loading are outside the inference timer; resident input graph memory is included. Profiling remains necessary before journal-level systems claims. The recorded offline pipeline cost includes reading/splitting the entire CSV and is **not online detection latency**. GPU deterministic operations may be unsupported on some builds; such errors must be resolved explicitly, not silently relaxed.

Confidence intervals collapse timing repeats within each training seed first. Synthetic intervals do not establish real-world generalization. Tables E preserve individual seed supports. ROC/PR metrics use probabilities; average precision is not mislabeled as trapezoidal PR-AUC.

## Layout

```text
src/ca_edgeids/       data, graph, model, training, metrics, reporting, CLI
configs/             auditable experiment protocols
tests/               leakage, label, coverage, halo equivalence, and metric checks
docs/                phased plan, experiment registry, data sources, status
.github/workflows/   Linux correctness and synthetic smoke run
runs/                ignored experiment artifacts and generated figures
data/                ignored raw datasets
```

No new license is assigned to the existing research code or third-party datasets by this rebuild. Follow the authors' academic-use and citation conditions for each dataset.
