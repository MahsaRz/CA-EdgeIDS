# Training sampler comparison

This separate study compares training procedures for the same edge-conditioned GNN on the existing three binary development samples. It is an edge-task adaptation of published sampling ideas, not a replication of their original node-classification experiments. The five-seed graph controls remain unchanged.

## Frozen design

The methods are unpartitioned target batching, random ownership with exact context, METIS ownership with exact context, a Cluster-GCN-style induced cluster sampler, and a GraphSAINT-style uniform-node sampler. Exact controls expose every training target once per epoch; ordering differs with ownership. Sampling controls use the same maximum 512 target flows per update and ceil(E/512) updates per logical epoch. Sampled target exposure and unique target coverage are reported, so equal update budgets must not be described as equal data exposure or equal compute.

Each dataset/method receives the same two learning-rate candidates (0.001 and 0.003) at seed 7. Architecture, optimizer, ten-epoch maximum and patience three match the earlier neural budget. Choose by validation macro F1 only, publish all 30 candidates and 15 choices, then train seeds 19, 31, 43 and 59 at each chosen setting. Publish all 75 selected checkpoint hashes before the test stage. All methods use full-snapshot validation and test context, and score all the original test flows once. No threshold search is performed. Test data are reused development holdouts.

## Sampling and normalization

[Cluster-GCN](https://arxiv.org/abs/1905.07953) motivates selecting several partitions and training on their induced context. Here four fixed METIS clusters are formed without labels, and two are selected uniformly without replacement. A target within one cluster has inclusion probability 1/2; a cut target has probability 1/6. All targets have positive probability, and none are permanently discarded at cluster boundaries. Context means use degrees inside the induced graph. This is an explicit edge-label adaptation.

[GraphSAINT](https://arxiv.org/abs/1907.04931) motivates sampled-subgraph normalization. Here a uniform node subset of size b=ceil(N/4), at least two where possible, has node probability b/N and distinct-endpoint pair probability b(b-1)/(N(N-1)); a self-loop has node probability. Context coefficients multiply full-graph mean weights by the conditional correction pi_node/pi_pair, including flow-feature and bias terms. This sampler differs from the official project's nonuniform edge sampler. The implementation and small exhaustive tests include multiedges and self-loops.

The full target objective is mean class-balanced cross entropy, with training-only class weights. Sampling loss uses inverse edge inclusion probability and, when eligible targets exceed 512, the additional uniform target-subsampling correction. This gives the correct expected risk for fixed per-edge losses. Nonlinear sampled hidden states differ from full-graph states, so no unbiased full-network gradient claim is made. Empty samples receive zero-gradient optimizer steps and are counted; they are never silently redrawn. All learned states are recomputed after updates. Fixed input summaries contain no labels.

This study refits every model. Its global-risk scaling differs from the earlier per-batch weighted-mean objective, so historical GNN checkpoints are not reused. Report actual epochs, updates, fit times, empty samples, class coverage and target exposures alongside quality. Memory results require separate fresh-process measurement.

## Reproduction

Run `python scripts/run_sampler_study.py --stage select`, publish the validation-only selection bundle, then `--stage seeds`. Publish that checkpoint bundle before `--stage evaluate`. Private outputs are in `runs/sampler-binary-100000/`. The driver refuses overwrites and checks the committed numerical sources, input manifests and checkpoint hashes at each gate.
