# 30,000-flow binary scaling pilots

These runs extend the software and scaling checks to the previously audited 30,000-flow uniform samples. They use the same initial architecture, settings, fixed split seed 42, and model seed 7. No test-driven tuning was performed. They are single-seed pilots, not final manuscript results or evidence of superiority.

| Dataset | Test flows | Macro F1 | Weighted F1 | Attack false-positive rate |
|---|---:|---:|---:|---:|
| ToN | 6,197 | 0.731324 | 0.754601 | 0.333795 |
| CSE | 6,592 | 0.985918 | 0.991486 | 0.003914 |

Training/validation contain both classes. The same retained test IDs were scored exactly once under Full and Random 2/4/8 partition inference. Probabilities match the frozen Full reference within the configured tolerance. METIS is explicitly unavailable on this Windows environment; previous Linux synthetic verification is not a substitute for real-data METIS results.

| Dataset | Full total inference (s) | Random 2 (s) | Random 4 (s) | Random 8 (s) |
|---|---:|---:|---:|---:|
| ToN | 0.015090 | 0.111802 | 0.104966 | 0.172658 |
| CSE | 0.028824 | 0.180792 | 0.204710 | 0.306775 |

Times are the mean of two synchronized repeats for one checkpoint, measured in fresh per-configuration processes on CPU. They include the entire inference loop, halo packing, forward calls, and result collection; they exclude cold preprocessing, graph construction, partitioning, checkpoint loading and warm-up. The separate component and offline pipeline costs are retained in `benchmark_runs.csv`; these values must not be described as end-to-end online detection latency. Full uses a single full-test-graph forward call.

Partition inference is slower in these pilots because exact context introduces repeated work. Its sampled RSS is roughly 315–323 MiB across these configurations, and these small differences do not establish a material memory saving. Training took approximately 20 s for ToN and 33 s for CSE, but full-dataset memory/time cannot be linearly extrapolated from these small graphs. RF/MLP are diagnostic baselines with unmatched tuning budgets; both MLP runs reached their 10-iteration limit with convergence warnings.

The 5,000-flow and 30,000-flow pilots have different retained test populations and transformed feature widths can differ. Their quality scores are not a paired improvement experiment. The high CSE score does not resolve the multiclass feasibility problem, shared-host generalization limits, or the need for controlled baselines and five seeds.

Aggregate manifests, learning histories, class reports, timing tables, and SVG/PNG figures are in `artifacts/pilots/ton-binary-30000/` and `artifacts/pilots/cse-binary-30000/`. Raw flows, predictions, and checkpoints remain local and ignored. `docs/BINARY_SCALING_SUMMARY.csv` contains the exact scalar metrics.

Next engineering work is to measure halo expansion and use efficient persisted graph arrays before loading the complete sources. The scientific work remains a documented multiclass protocol revision, controlled sampler/feature ablations, validation-only tuning, and appropriately scoped generalization experiments.
