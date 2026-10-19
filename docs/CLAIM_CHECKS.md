# Claim and implementation checks

| Issue | What the revised work does | Evidence or remaining limit |
|---|---|---|
| Causal root-cause labels | Predicts original Benign/Attack flow labels; removes initiating-node and propagation claims | The source provides intrusion annotations, not causal ground truth; this work is intrusion classification |
| Node versus edge prediction | One original directed record is one target; incident flows never require a single node label | Explicit endpoint identities, flow labels and exactly-once coverage tests |
| Features and framework | PyTorch edge-conditioned mean GNN; two degree features plus ten selected flow fields; training-only preprocessing | Resolved configurations, source hashes, model equations and Table B |
| Method novelty | Treats partitioning, mean aggregation and linear factorization as inherited techniques | Empirical contribution and mixed/negative results; no new-algorithm claim |
| Cluster/sampling controls | Separately trains unpartitioned, random/METIS exact-context, cluster-induced and normalized uniform-node procedures | 30 validation candidates; 75 frozen evaluations; explicit edge-task adaptations and analytic inclusion tests |
| Inference cost | Scores a fixed complete target population and reports total processing across all partitions | Historical Linux pilot and layer-wise confirmation complete; larger fixed-model grid frozen, pending execution |
| Complexity | Gives arc-linear full-message-passing cost at fixed depth/dimensions | Repeated query-neighborhood work is separate from full-graph complexity; no exponential full-graph claim |
| Distributed/online deployment | Describes the actual single-workstation offline snapshot experiment | No multi-worker, communication-scaling, streaming deadline or deployment claim |
| Generalization | Pair-grouped split plus a separately frozen new-host incident-flow protocol | Known infrastructure can be shared; ToN/CSE new-host attack test supports only 12/22; BoT infeasible; model fits in progress |
| Minority performance | Reports macro/weighted metrics, per-class supports, FPR and AP | All seeds retained; no threshold or model rollback using test scores; no population CI inferred from five seeds |
| Counts and result consistency | Keeps source-audit, fitted sample, per-split and per-graph counts separate | Original source IDs and source/model/target hashes; systems cohort is distinct from classifier refits |
| Time windows | Uses explicitly offline split snapshots | No genuine timestamp available for temporal claims; duration and row order are not treated as time |
| Graph direction | Original flows are directed; reverse arcs supply context only | Directed-only ablation retained; no reverse target duplication |
| Prediction and training rules | Weighted cross entropy, validation macro F1 checkpoint, fixed softmax/argmax | Sigmoid-threshold claims removed; all candidates and actual epoch/update counts retained |
| Symbols and figure meaning | Uses L for layer count and P for partition count; figures state their populations and timing boundaries | Final manuscript cross-reference and render audit remains before upload |
| Classification plots | Uses actual true/predicted count matrices when called confusion matrices | Per-class report tables remain separately named; no classification-report heatmap relabeled as a confusion matrix |
| References and prose | Rewrites around the actual task and checks cited primary sources | No unresolved publication placeholders; original manuscript/authorship preserved in a private backup |

Completed experiments do not by themselves establish journal-level methodological novelty or broad deployment validity. The final scientific assessment must follow the remaining generalization and larger-workload evidence. The live Overleaf manuscript is unchanged until the private revised draft is complete and checked.
