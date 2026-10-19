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
