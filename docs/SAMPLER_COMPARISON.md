# Training sampler comparison

This separate study compares training procedures for the same edge-conditioned GNN on the existing three binary development samples. It is an edge-task adaptation of published sampling ideas, not a replication of their original node-classification experiments. The five-seed graph controls remain unchanged.

## Frozen design

The methods are unpartitioned target batching, random ownership with exact context, METIS ownership with exact context, a Cluster-GCN-style induced cluster sampler, and a GraphSAINT-style uniform-node sampler. Exact controls expose every training target once per epoch: stable source-ownership ordering is cut into consecutive batches, which may straddle an ownership boundary, then batches are shuffled. Sampling controls use the same maximum 512 target flows per update and ceil(E/512) updates per logical epoch. Sampled target exposure and unique target coverage are reported, so equal update budgets must not be described as equal data exposure or equal compute.

Each dataset/method receives the same two learning-rate candidates (0.001 and 0.003) at seed 7. Architecture, optimizer, ten-epoch maximum and patience three match the earlier neural budget. Choose by validation macro F1 only, publish all 30 candidates and 15 choices, then train seeds 19, 31, 43 and 59 at each chosen setting. Publish all 75 selected checkpoint hashes before the test stage. All methods use full-snapshot validation and test context, and score all the original test flows once. No threshold search is performed. Test data are reused development holdouts.

## Sampling and normalization

[Cluster-GCN](https://arxiv.org/abs/1905.07953) motivates selecting several partitions and training on their induced context. Here four fixed METIS clusters are formed without labels, and two are selected uniformly without replacement. A target within one cluster has inclusion probability 1/2; a cut target has probability 1/6. All targets have positive probability, and none are permanently discarded at cluster boundaries. Context means use degrees inside the induced graph. This is an explicit edge-label adaptation.

[GraphSAINT](https://arxiv.org/abs/1907.04931) motivates sampled-subgraph normalization. Here a uniform node subset of size b=ceil(N/4), at least two where possible, has node probability b/N and distinct-endpoint pair probability b(b-1)/(N(N-1)); a self-loop has node probability. Context coefficients multiply full-graph mean weights by the conditional correction pi_node/pi_pair, including flow-feature and bias terms. This sampler differs from the official project's nonuniform edge sampler. The implementation and small exhaustive tests include multiedges and self-loops.

The full target objective is mean class-balanced cross entropy, with training-only class weights. Sampling loss uses inverse edge inclusion probability and, when eligible targets exceed 512, the additional uniform target-subsampling correction. This gives the correct expected risk for fixed per-edge losses. Nonlinear sampled hidden states differ from full-graph states, so no unbiased full-network gradient claim is made. Empty samples receive zero data-loss gradients and a counted Adam step, including its usual momentum and weight-decay behavior; they are never silently redrawn. All learned states are recomputed after updates. Fixed input summaries contain no labels.

This study refits every model. Its global-risk scaling differs from the earlier per-batch weighted-mean objective, so historical GNN checkpoints are not reused. Report actual epochs, updates, fit times, empty samples, class coverage and target exposures alongside quality. Memory results require separate fresh-process measurement.

## Reproduction

Run `python scripts/run_sampler_study.py --stage select`, publish the validation-only selection bundle, then `--stage seeds`. Publish that checkpoint bundle before `--stage evaluate`. Private outputs are in `runs/sampler-binary-100000/`. The driver refuses overwrites and checks the committed numerical sources, input manifests and checkpoint hashes at each gate.

## Completed results

All 30 validation candidates, 15 method choices and 75 five-seed selected-model evaluations completed. Numerical sources are frozen at `f1574fa`; validation choices were published at `4863cbf` and all 75 checkpoints at `c5143cc` before their test stage. Scores were independently recomputed from private predictions and all original target IDs/labels were checked. The paper-size result figure was visually inspected.

| Dataset | Unpartitioned | Random exact | METIS exact | Cluster sampled | Node sampled |
|---|---:|---:|---:|---:|---:|
| ToN | 0.8085 +/- 0.0836 | 0.8230 +/- 0.0806 | 0.7714 +/- 0.0948 | 0.8027 +/- 0.0827 | 0.8810 +/- 0.0115 |
| CSE | 0.9872 +/- 0.0010 | 0.9883 +/- 0.0017 | 0.9892 +/- 0.0011 | 0.9880 +/- 0.0016 | 0.9882 +/- 0.0034 |
| BoT | 0.7617 +/- 0.1975 | 0.8377 +/- 0.1647 | 0.8547 +/- 0.1745 | 0.8708 +/- 0.0629 | 0.8548 +/- 0.0344 |

Entries are test macro F1 mean plus/minus sample SD across the five seeds on the reused development split. METIS does not give a consistent quality improvement. The uniform-node adaptation performs best on ToN within this study; CSE differences are small; BoT exact controls are unstable. Different objectives and new validation selections mean these rows must not be substituted for the earlier graph-control/classifier table.

Sampling exposure is not full coverage. Mean numbers of training targets never selected during the entire fit are 1,297/293 on ToN, 2,848/742 on CSE and 2,038/1,843 on BoT for cluster/node sampling, respectively. Every target has positive inclusion probability; finite training still leaves some unseen. Mean empty node samples are 5.8 on ToN, zero on CSE and 328.2 on BoT. They count against the fixed update budget and are never redrawn. Exposure/coverage counts describe the entire fitting run, including patience epochs after its best checkpoint, rather than guaranteed exposure by the selected checkpoint epoch.

All test targets are retained in every condition. No new seed, learning rate or sampler setting is selected after these observations. Fit wall times are diagnostics because some training/source preparation jobs overlapped, not isolated training-speed evidence. Complete metrics, paired same-seed differences and exposure counts are in `artifacts/sampler-binary-100000/test-evaluation/`.
