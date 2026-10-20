# Claim and implementation checks

## Completed evidence update

The declared experiments are complete: 225 selected-model evaluations (120 classifier/graph controls, 75 training-procedure controls and 30 new-host models), 36 matched classifier-cost conditions and 135 larger-workload requests. All 36 classifier-cost and 129 successful larger-workload conditions preserve class decisions within the declared probability tolerance; six CSE one-million-flow partition conditions time out. The current numerical package passes 71 Linux correctness tests. AP and trapezoidal PR-AUC are separately reported for all 225 models.

Forest leads the original ToN/BoT classifier comparison and the full GNN leads CSE. Removing neighbor messages helps ToN/BoT. Layer-wise inference is the fastest tested GNN schedule in all nine larger workloads; exact partition processing is substantially slower. New-host detection is weak: CSE GNN attack recall is 1.8% despite weighted F1 of 0.9951. ToN/CSE attack test supports are only 12/22; BoT is infeasible. These results support a conditional empirical comparison, not reliable adaptation, causal diagnosis or a new partitioning algorithm.

The current execution record and contribution assessment supersede earlier planned or pending milestones below. Historical protocols and exclusions remain preserved.


| Issue | What the revised work does | Evidence or remaining limit |
|---|---|---|
| Causal root-cause labels | Predicts original Benign/Attack flow labels; removes initiating-node and propagation claims | The source provides intrusion annotations, not causal ground truth; this work is intrusion classification |
| Node versus edge prediction | One original directed record is one target; incident flows never require a single node label | Explicit endpoint identities, flow labels and exactly-once coverage tests |
| Features and framework | PyTorch edge-conditioned mean GNN; two degree features plus ten selected flow fields; training-only preprocessing | Resolved configurations, source hashes, model equations and Table B |
| Method novelty | Treats partitioning, mean aggregation and linear factorization as inherited techniques | Empirical contribution and mixed/negative results; no new-algorithm claim |
| Cluster/sampling controls | Separately trains unpartitioned, random/METIS exact-context, cluster-induced and normalized uniform-node procedures | 30 validation candidates; 75 frozen evaluations; explicit edge-task adaptations and analytic inclusion tests |
| Inference cost | Scores a fixed complete target population and reports total processing across all partitions | Historical Linux pilot and layer-wise confirmation complete; larger fixed-model grid complete: 129 successful conditions and six retained CSE partition timeouts |
| Complexity | Gives arc-linear full-message-passing cost at fixed depth/dimensions | Repeated query-neighborhood work is separate from full-graph complexity; no exponential full-graph claim |
| Distributed/online deployment | Describes the actual single-workstation offline snapshot experiment | No multi-worker, communication-scaling, streaming deadline or deployment claim |
| Generalization | Pair-grouped split plus a separately frozen new-host incident-flow protocol | Known infrastructure can be shared; ToN/CSE new-host attack test supports only 12/22; BoT infeasible; all 30 ToN/CSE model evaluations complete, with weak attack detection |
| Minority performance | Reports macro/weighted metrics, per-class supports, FPR, AP and separate trapezoidal PR-AUC | All seeds retained; no threshold or model rollback using test scores; no population CI inferred from five seeds |
| Counts and result consistency | Keeps source-audit, fitted sample, per-split and per-graph counts separate | Original source IDs and source/model/target hashes; systems cohort is distinct from classifier refits |
| Time windows | Uses explicitly offline split snapshots | No genuine timestamp available for temporal claims; duration and row order are not treated as time |
| Graph direction | Original flows are directed; reverse arcs supply context only | Directed-only ablation retained; no reverse target duplication |
| Prediction and training rules | Weighted cross entropy, validation macro F1 checkpoint, fixed softmax/argmax | Sigmoid-threshold claims removed; all candidates and actual epoch/update counts retained |
| Symbols and figure meaning | Uses L for layer count and P for partition count; figures state their populations and timing boundaries | Final manuscript cross-reference and render audit remains before upload |
| Classification plots | Uses actual true/predicted count matrices when called confusion matrices | Per-class report tables remain separately named; no classification-report heatmap relabeled as a confusion matrix |
| References and prose | Rewrites around the actual task and checks cited primary sources | No unresolved publication placeholders; original manuscript/authorship preserved in a private backup |

Completed experiments do not by themselves establish journal-level methodological novelty or broad deployment validity. The completed contribution assessment includes generalization and larger-workload results. The private Overleaf revision is compiled, with the original files preserved.

## Private manuscript release

The private Overleaf project now uses `ca-edgeids-main.tex` and is named **CA-EdgeIDS: Graph Context and Processing Cost**. Its compiled paper is **Graph Context and Processing Cost in Flow-Level Intrusion Detection**. The original source files, authors, affiliations and funding acknowledgment are preserved. Live compilation reports zero errors; an unused inherited document-class option and underfull typesetting notices are non-blocking. The actual downloaded Overleaf PDF is reviewed separately from the local compiler output. No journal submission or merge to main has been made.
