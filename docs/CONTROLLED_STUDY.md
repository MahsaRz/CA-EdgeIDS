# Five training seeds and graph controls

## Frozen scope

The plan in `configs/controlled_binary_100000.json` uses seeds 7, 19, 31, 43 and 59 on the existing three 100,000-flow development samples. All three families retain the settings chosen in the earlier validation study: neural learning rate 0.003, forest depth unrestricted on ToN and 16 on CSE/BoT, 100 trees, neural hidden width 16, maximum ten epochs and patience three. There is no new hyperparameter or threshold search. The original split, class mapping, train-fitted preprocessing and test target IDs remain fixed.

For each dataset, fit GraphSAGE at five seeds for each of six settings, plus MLP and Forest at five seeds: 120 fits in total. Every model selects its epoch on validation macro F1. Freeze all selected checkpoint hashes before new test predictions. The train/validation phase builds no test graph and computes no test predictions. The test split has already been inspected in development, so this study measures training-seed variation and input sensitivity on that split; it is not new independent data evidence.

Training ownership is fixed at seed 7 across model seeds. Neural families share the resulting target-batch membership and maximum size 512; their seeded shuffles, initialization and dropout vary with training seed. This avoids mixing an uncontrolled change in partition assignment into initialization comparisons. Actual epochs, updates, fit times and capacities remain recorded. These upper budgets do not equate forest compute with neural compute.

## Graph settings

| Setting | Change | Preserved |
|---|---|---|
| Full | Original directed flows with reverse message arcs | All original data and targets |
| No degree | Set the two initial node features to zero | Flow features, messages, targets, architecture |
| No message features | Zero flow-feature terms inside messages | Decoder flow features, node features, topology, targets |
| No messages | Set the aggregated neighbor vector to zero | Node degree and decoder features, target batch membership, architecture |
| Rewired | Permute directed message destinations within each split | Original prediction endpoints/flow features/labels/IDs, message count, message in/out degree counts |
| Directed | Use only original flow-direction message arcs | Original targets, flow/node features and identities |

Rewiring uses fixed topology seeds 104729/104730/104731 for train/validation/test across all training seeds. It is an explicitly corrupted-context control, not a new network dataset: node identities and target endpoints are never randomized. It may create self loops and does not preserve reverse-arc pairing. Each split is transformed independently. No labels enter the transformation. The no-message control retains degree features, so it is not identical to the flow-only MLP.

## Faster training arithmetic

The model's message is linear in the concatenation of the source state and flow features, followed by a mean at the destination. Linearity allows the same mean to be computed from two label-free inputs: a sparse matrix of normalized source-to-destination multiplicities and the mean incoming flow-feature vector. Model weights and hidden states are still applied anew on every optimizer update. No learned state is cached across updates.

The implementation retains the original query batch's sorted K-hop node closure, preserving the shape/order of dropout draws. Boundary-node layer states need not equal states of an induced halo, but cannot reach the query output at the specified depth. Tests compare logits and every parameter gradient at depths 1-3 in both message directions, in evaluation and training with the same dropout seed. They also check that changing weights recomputes outputs and that rewiring preserves targets and degree counts. All 59 tests passed locally in Linux before the frozen study.

This is algebraic factorization of the existing linear mean message, not a new GNN method. Floating-point summation order changes, so fitted trajectories can differ from historical fits. Refit every seed/control with the same new backend; do not splice historical GNN weights into the five-seed table. Original earlier results remain unchanged. Input summaries have a storage-screening limit, not a bound on total process RAM.

## Execution and reporting

Run `python scripts/run_controlled_study.py --stage train`, publish/check the frozen selections, then run `--stage evaluate`. Private outputs are under `runs/controlled-binary-100000/`; raw flows, checkpoints and per-flow predictions remain outside Git. Run folders refuse to overwrite earlier results.

Report every training seed, mean/sample standard deviation/minimum/maximum across the five models, complete per-class supports and paired ablation differences against Full at the same seed. Keep initialization variation distinct from data-split or population uncertainty. The same sample/split is reused; rewired graphs are not extra independent topology replicates. Do not select seeds, graph settings or thresholds using the resulting test metrics. Later generalization, sampler and scaling studies require separately frozen protocols.

## Completed results

All 120 fits and all 120 test evaluations completed. Numerical sources were frozen at `31d515c`; all checkpoint selections were published at `9e1da4b` before evaluation. The verified aggregate bundle is `artifacts/controlled-binary-100000/test-evaluation/`. Every comparison retains the same original target IDs and labels within each dataset. Both result figures were visually inspected.

Test macro F1 below is mean plus/minus sample standard deviation over five training seeds on one fixed split.

| Dataset | Edge-conditioned GNN | Flow MLP | Random Forest |
|---|---:|---:|---:|
| ToN | 0.7405 +/- 0.0780 | 0.8755 +/- 0.0120 | 0.9745 +/- 0.0004 |
| CSE | 0.9874 +/- 0.0021 | 0.9375 +/- 0.0323 | 0.7436 +/- 0.0015 |
| BoT | 0.9449 +/- 0.0175 | 0.8228 +/- 0.0732 | 0.9785 +/- 0.0025 |

The GNN performs best on CSE; Forest performs best on ToN and BoT. The graph controls do not support a universal message-passing benefit. Relative to the full GNN at the same seed, removing messages changes mean macro F1 by +0.1290 on ToN, -0.0076 on CSE and +0.0083 on BoT. This control retains degree features. Removing degree features reduces BoT macro F1 by 0.2810, while removing message flow features improves ToN by 0.1349. Directed-only context is unstable on CSE (mean change -0.1744, SD 0.1505). Rewired context can improve ToN while hurting BoT; this is evidence of dataset-dependent sensitivity, not reliable causal information.

Keep the predefined full model and every negative result in the report. These observations do not authorize choosing a better-looking ablation on test data. BoT's small benign support and the reused development holdouts limit the interpretation. No result establishes causal localization, online adaptation or distributed execution.
