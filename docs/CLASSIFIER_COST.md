# Classifier processing costs

This protocol is frozen before timing. It uses the already selected seed-7 GNN, flow MLP and Random Forest models from the five-seed classifier study. Choosing seed 7 does not choose the best test seed. There are four inference conditions: GNN Full, GNN layer-wise (4,096 message arcs / 2,048 target flows per chunk), MLP and Forest (2,048-flow prediction chunks). Three independently ordered fresh-process blocks produce 36 conditions, each with a complete first prediction and three additional complete warm predictions.

Every condition retains the same original test population and fixed training-fitted preprocessing. Flow-only methods skip graph construction. The Forest worker avoids importing unused Torch code, so its startup and RSS are not inflated by the GNN framework. Method-specific imports, preprocessing, checkpoint loading, required graph construction and the first complete prediction enter worker-launch-to-first-prediction time. Serialization, metric/report generation, the parent driver and WSL startup remain outside that timer. Warm totals include every target's probabilities becoming available in memory.

RSS is sampled every 5 ms through imports/setup/prediction and separately through warm prediction; it is not an allocator-exact peak or a measurement of the whole machine. Collapse warm repetitions within each block before computing mean/sample SD over the three blocks. These are fixed-weight process repetitions, not extra training seeds. Never pool them into the five-seed quality uncertainty.

Check output row IDs, model hashes, probability differences and class decisions against the selected-model predictions. Keep every declared condition and any failures. These models differ in capacity and compute; the table reports that difference rather than asserting a matched parameter or execution budget. This classifier-cost cohort is distinct from the older 0.001-learning-rate systems checkpoint used by the larger inference study.

Run `python scripts/run_classifier_cost.py` only after training/source scans finish, then `python scripts/report_classifier_cost.py`. The latter exports verified aggregate evidence without raw records or model files. Private runs are under `runs/classifier-cost-100000/`; publication outputs use `artifacts/classifier-cost-100000/`.

The frozen manifest records the CPU model, logical CPU count, Linux-visible memory/swap, thread environment and whether the workspace is on a Windows mount. The present platform has an AMD Ryzen 7 PRO 7840U, 16 logical CPUs and about 13.5 GiB Linux-visible memory; compute uses one thread. These are local WSL2 research measurements rather than a hardware-independent deployment limit.

## Completed comparison

All 36 declared fresh-process conditions completed, with three complete warm predictions per condition. Every output matches the selected model's original test IDs, probabilities within tolerance and exact class decisions; there are no failures or disagreements. Numerical sources are frozen at `83747d5`. No weights or thresholds were changed.

| Dataset | Method | Complete warm ms, mean +/- sample SD | First prediction s | Sampled process RSS MiB |
|---|---|---:|---:|---:|
| ToN | GNN Full | 49.41 +/- 1.16 | 2.570 | 483.8 |
| ToN | GNN layer-wise | 16.54 +/- 0.29 | 2.573 | 427.7 |
| ToN | MLP | 3.99 +/- 0.13 | 2.445 | 409.0 |
| ToN | Forest | 149.03 +/- 2.52 | 1.946 | 335.2 |
| CSE | GNN Full | 72.89 +/- 0.29 | 2.704 | 527.1 |
| CSE | GNN layer-wise | 24.81 +/- 4.31 | 2.645 | 447.8 |
| CSE | MLP | 4.65 +/- 0.40 | 2.539 | 424.7 |
| CSE | Forest | 144.32 +/- 2.44 | 1.949 | 355.3 |
| BoT | GNN Full | 41.56 +/- 7.12 | 2.598 | 457.3 |
| BoT | GNN layer-wise | 15.38 +/- 0.28 | 2.556 | 408.5 |
| BoT | MLP | 3.91 +/- 0.23 | 2.472 | 390.4 |
| BoT | Forest | 83.23 +/- 1.41 | 1.758 | 306.5 |

Warm uncertainty is sample SD over the three process-block medians, not over nine independent model fits. First prediction and RSS columns are block means; the complete export retains their individual measurements and first-prediction SD. The original populations contain 20,691 / 21,870 / 26,304 ToN/CSE/BoT test flows.

MLP is fastest for warm inference here. Forest has lower startup/process memory but takes longer for warm prediction than these GNN conditions. The chosen layer-wise GNN condition improves warm time and sampled process RSS over Full on all three datasets in this grid. These are fixed-seed implementation measurements, not a claim that one classifier is best in both quality and cost. Quality uncertainty comes from the separate five-seed study; capacity and fitting compute remain unequal. See `artifacts/classifier-cost-100000/` for all blocks and prediction checks.
