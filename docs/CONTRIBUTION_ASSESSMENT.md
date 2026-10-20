# What the evidence supports

The declared experiments are complete: 225 selected-model evaluations (120 classifier/graph controls, 75 training-procedure controls and 30 new-host models), 36 matched classifier-cost conditions and 135 larger-workload requests. All 36 classifier-cost and 129 successful larger-workload conditions preserve class decisions within the declared probability tolerance; six CSE one-million-flow partition conditions time out. The current numerical package passes 71 Linux correctness tests. AP and trapezoidal PR-AUC are separately reported for all 225 models.

Forest leads the original ToN/BoT classifier comparison and the full GNN leads CSE. Removing neighbor messages helps ToN/BoT. Layer-wise inference is the fastest tested GNN schedule in all nine larger workloads; exact partition processing is substantially slower. New-host detection is weak: CSE GNN attack recall is 1.8% despite weighted F1 of 0.9951. ToN/CSE attack test supports are only 12/22; BoT is infeasible. These results support a conditional empirical comparison, not reliable adaptation, causal diagnosis or a new partitioning algorithm.

The contribution is a reproducible empirical analysis of graph context, target-preserving inference and processing cost on three standardized flow datasets. METIS, GraphSAGE, cluster sampling and node sampling are established components. The edge-task adaptations and checks are described explicitly without claiming their integration is a new algorithm.

The observed classifier ranking depends on the dataset and protocol. Flow-only MLP is the fastest warm classifier in the matched cost grid; Forest uses less process memory/startup time but slower warm prediction. Models have unequal capacities. Cost uses fixed seed 7; quality uses five seeds. Previously inspected development holdouts are not a pristine final benchmark.

New-host results show poor operating points, not merely uncertain positive gains. Known infrastructure is shared, the attack supports are small, and CSE holdout attacks are all Infiltration. Multiclass split support and genuine event timestamps remain unavailable. No initiating causes, failure propagation, physical edge deployment, multi-worker operation or online latency were measured.

The remaining scientific decision is whether these conditional and negative findings form a suitable empirical paper. Author review and venue selection are separate from completion of the declared engineering and experiments.
