# Larger fixed-model inference study

The frozen grid uses three independently ordered process blocks, three datasets, uniform sample sizes 100,000 / 300,000 / 1,000,000, and five methods: Full, four random partitions with exact context, four METIS partitions with exact context, layer-wise chunked inference, and factored linear-mean inference. This requests 135 fresh-process conditions, each with a first complete prediction and three further complete warm predictions. The grid is fixed before execution.

Within each dataset, retain the original 100,000-flow pilot preprocessor and seed-7 checkpoint for all sizes. This is an inference workload study, not a new model-quality experiment. Every method at a given size uses the same test flow IDs, original IP identities and full context. Uniform sample priorities are nested and match the old development IDs. Only unused source columns are omitted in the larger stored CSVs; all workers parse the same required columns.

The original 100,000-flow CSV still retains unused fields, while the larger files store the required projection only. Input byte sizes are recorded. First-process costs across sizes therefore reflect file layout as well as record volume; they are descriptive checked-pipeline costs, not an isolated model-scaling curve. Method comparisons at each workload use the identical input file. Complete warm inference excludes parsing in all sizes.

The nested-observable audit also checks all 100,000 retained development records in each larger file: endpoints, original labels, every selected numerical value and the exact categorical strings seen by the encoder. All six dataset/size checks pass with zero changes. This catches representation changes from CSV type inference as well as row-identity errors. The aggregate audit is `artifacts/larger-uniform-source-audits/nested-observables.json`.

Warm totals include complete node encoding and edge decoding across every target. The layer-wise method recomputes node states on every call, using 4,096 message arcs and 2,048 query flows per chunk. Factored inference recomputes its label-free input summaries within each call before applying the model; it is linear algebra reuse, not a new message-passing architecture. Random/METIS use 2,048-flow batches and retain cross-partition context. Full uses one complete target batch and can hit the declared tensor guard.

Fresh-process time starts before process launch and ends after the first complete prediction. It includes interpreter/import startup, CSV parsing, preprocessor loading, pair splitting, fixed feature transformation and integrity audits, graph construction, checkpoint/model loading, partitioning and inference. CSV checksum and feature-duplicate audits are included in transformation time. This is a checked research pipeline time, not production-network packet latency. Process memory is sampled every 5 ms from the start of input setup, after module imports, through all predictions. It includes resident imported libraries but does not observe transient import-time peaks. Warm inference memory is also reported where available. OS caches are not forced cold.

Retain the original 512 MiB CSV and 256 MiB inference tensor/input-summary screening limits. Dense transformed features have a declared 3 GiB storage screen and are transformed in chunks. These are storage estimates, not hard bounds on total RAM. Retain timeout and resource-guard failures without increasing limits after seeing their results. Compare successful predictions against Full where feasible and the independent layer-wise baseline when Full is rejected, report numerical differences and class agreement, and never call an unverified condition equivalent.

Run `python scripts/run_larger_inference.py` after training and full-source scans finish. Do not run competing training/data-scan workloads during latency measurement. Verify and export the private `runs/larger-inference/` results before using any number in the manuscript.

Prediction checks use absolute and relative tolerances 0.00001 plus exact class agreement. Any numerical mismatch is retained with its maximum probability difference and number of changed decisions; it is not relabeled equivalent or removed from the declared grid. Invalid/missing targets or corrupted outputs fail the audit. A pre-grid functional software check reproduced the old ToN target IDs and predictions through the fixed-preprocessor worker; its concurrent-job timing is excluded from the study.

Memory values use MiB (2^20 bytes), including historical CSV field names ending in `_mb`. Inputs and run files reside on the Windows-mounted workspace under WSL2. These storage/cache conditions affect process and preparation times; the study does not treat them as native edge-device deployment measurements.

The frozen manifest also records the CPU model, logical CPU count, Linux-visible memory/swap and single-thread environment. The current workstation exposes 16 logical CPUs and about 13.5 GiB to Linux; this is VM-visible total memory, not a guarantee of available memory during a call.

## Completed results

Numerical source: `83747d5`. All 135 requests are retained: 129 complete and six time out. Every successful condition passes probability tolerances and exact class agreement; maximum probability difference is 0.000007033348 and no decisions change. The six failures are all CSE one-million-flow random/METIS conditions, across all three process blocks. Their 300-second limit covers the entire worker, including first prediction and three warm predictions; it is not an inferred forward time.

At one million sampled flows, actual test populations are 206,110 / 219,226 / 262,028 for ToN / CSE / BoT, containing 810 / 18,846 / 12 IP nodes. Increasing flow volume does not establish large-node-topology scaling.

| Dataset | Method | Complete warm seconds, mean ± SD | First prediction seconds, mean | Process peak MiB, mean |
|---|---|---:|---:|---:|
| ToN | Full | 0.6563 ± 0.0157 | 8.739 | 1918.1 |
| ToN | Layer-wise | 0.1770 ± 0.0080 | 8.268 | 1923.8 |
| ToN | Factored | 1.8088 ± 0.0044 | 10.005 | 1918.2 |
| ToN | Random exact | 48.2408 ± 0.2794 | 56.942 | 1924.0 |
| ToN | METIS exact | 50.5391 ± 0.1954 | 59.066 | 1929.8 |
| CSE | Full | 1.0706 ± 0.0355 | 10.262 | 2071.6 |
| CSE | Layer-wise | 0.2316 ± 0.0016 | 9.323 | 2049.4 |
| CSE | Factored | 4.8829 ± 0.0466 | 13.895 | 2053.9 |
| CSE | Random / METIS exact | unavailable: all blocks timed out | unavailable | unavailable |
| BoT | Full | 0.5944 ± 0.0205 | 8.795 | 1753.6 |
| BoT | Layer-wise | 0.1597 ± 0.0043 | 8.311 | 1759.3 |
| BoT | Factored | 1.4138 ± 0.0317 | 9.548 | 1753.7 |
| BoT | Random exact | 62.2192 ± 0.1645 | 71.165 | 1759.6 |
| BoT | METIS exact | 63.3554 ± 0.8651 | 72.656 | 1754.1 |

Layer-wise is the fastest complete warm GNN schedule in all nine dataset/size workloads. Rebuilding factored summaries every prediction is slower than Full; reuse during training answers a different question. Exact halos repeat substantial work: one-million-flow METIS inference is about 77 times Full on ToN and 107 times Full on BoT. The tested partition schedules do not offer a speed benefit.

Whole-pipeline peaks at one million flows are similar across methods, about 1.7–2.0 GiB. These measurements do not establish substantial whole-pipeline memory savings from partitioning or layer-wise processing. Preparation and resident transformed features matter. The smaller classifier-cost grid uses another model cohort and memory sampling window; its memory comparisons must keep that scope.

Verified aggregate records, failures, block counts, hardware and input hashes are in `artifacts/larger-inference/`; the complete tables and figures are in `artifacts/paper-assets/`.
