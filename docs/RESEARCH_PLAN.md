# Rebuild and resubmission plan

## Decision and scope

The supervisor's advice identifies the correct order: change the research task, align the code, then regenerate the evidence and manuscript. More experiments cannot establish RCA without initiating-cause ground truth. The revised prediction is a label on an original flow edge: Benign/Attack or the original attack category. A malicious flow is not automatically a causal root or a faulty physical device.

**Provisional project name:** CA-EdgeIDS. **Provisional title:** Cluster-Aware GraphSAGE for Flow-Level Intrusion Detection: Accuracy, Coverage, and End-to-End Cost. Avoid "distributed", "real-time", "causal", and "unseen-topology adaptation" unless separately demonstrated. Naming is provisional and is not a claim of trademark or methodological novelty.

The attached tables/figures checklist is useful and substantially appropriate after reframing. It is a minimum evidence list, not evidence that acceptance is assured. In particular, Table D cannot establish novelty merely by showing METIS versus random; the paper must distinguish existing Cluster-GCN ideas and justify what the study contributes.

**Observed feasibility issue:** the 30,000-flow BoT pilot produces a test graph with just four original IP nodes. The brief's 100-2000 partition schedule cannot be applied indiscriminately. Establish actual node counts first; use valid per-dataset counts, report infeasible conditions, and separate large flow volume from large node topology. Do not randomize identities or change the graph definition merely to recover the former large-graph claim.

The completed full-file IP audit found **29,245 ToN nodes, 293 BoT nodes, and 255,042 CSE nodes**, before splitting. In particular, 1000/2000 IP-node partitions are impossible for BoT even before splitting. The proposed full BoT configurations now use a smaller 2/4/8/16/32 inference sweep and eight training partitions, with feasibility checks on each actual split. These settings were adjusted using topology counts, not test classification performance.

## Phases, gates, and deliverables

| Phase | Work | Exit gate | Planning allowance |
|---|---|---|---|
| 0. Reframe and freeze protocol | Define task, node/edge identities, graph direction, label map, offline/online scope, contribution hypothesis. Preserve original repository state. | Written protocol with no unsupported RCA or distributed claim. | 1-2 working days |
| 1. Rebuild and validate | Modular implementation; train-only preprocessing; grouped splits; validation checkpointing; exactly-once boundary-edge inference; synthetic tests; environment capture. | Tests pass, training completes from a clean process, logits match full graph for exact halos. | 3-5 working days |
| 2. Acquire and audit real data | Official v2 downloads; checksum/header/count audit; original and canonical label counts; duplicate and host-pair overlap; memory pilot; complete mapping. | Table A reconciles source counts and actual split supports; every exclusion is explained. | 2-4 working days, download access permitting |
| 3. Pilot and profile | Binary and multiclass pilots on all three datasets; fresh-process CPU/GPU peak memory; stage timings; validation-only settings; diagnose halo expansion. | Stable runs, plausible learning curves, fixed targets, a measured extrapolation of runtime and memory. | 3-5 working days |
| 4. Full controlled experiments | Full/Random/METIS, fixed-workload partition sweep, independently trained partition ablations, Cluster-GCN and edge-feature/graph ablations, RF/MLP equal-budget tuning; five seeds; uncertainty. | All requested configurations complete or explicit OOM/unavailable results; no test tuning; no changing support. | 5-10 working days plus compute queue |
| 5. Manuscript and release | Replace structure, claims, equations, figures and tables; trace every value to a run; limitations and citations; independent clean reproduction. | No stale RCA text or reused legacy result; Tables A-E and Figures 1-4 reproducible from immutable artifacts. | 4-6 working days |

**Defensible estimate for the supervisor:** approximately **4-6 weeks** for a complete, carefully checked revision with stable dataset access and suitable compute. A working code/pilot milestone is approximately **1-2 weeks**. These are engineering estimates, not measured experiment durations or an automatic completion promise. Full-scale timing must be updated after Phase 3. This workstation currently exposes CPU-only PyTorch; full dataset repetitions may need a larger Linux/GPU machine. GPU access is not assumed or purchased.

## Proposed manuscript structure

1. Introduction: intrusion-detection problem, practical graph-processing cost, precise hypothesis and supported contributions.
2. Related work: graph IDS, E-GraphSAGE, GraphSAGE, Cluster-GCN, GraphSAINT, partitioning, evaluation leakage. Explain inherited components.
3. Task and data protocol: original flows, node identity, labels, direction, available features, splits, duplicate controls, offline snapshot assumptions.
4. Method: actual edge-conditioned message function, node update, edge decoder, ownership, cross-boundary context, training/inference distinction.
5. Experimental design: dataset audit; validation-only tuning; seeds; baselines; fixed populations; timing boundaries; memory definitions; limitations of the machine.
6. Results: correctness/coverage, per-class quality, partition tradeoffs, full/random/METIS systems comparison, training/feature/topology ablations, false positives and uncertainty.
7. Discussion and limitations: loss of causal interpretation; joint snapshot test features; boundary cost; minority classes; limits of generalization.
8. Conclusion and reproducibility: only claims supported by new experiments; code/data provenance and exact commands.

Do not write a results-based abstract until Phase 4. The existing Overleaf manuscript has not been edited during this rebuild.

## Mandatory experimental discipline

- Freeze source file SHA-256, row IDs, label map, feature allowlist, splits, checkpoint rules, and environment. Use the same split seed for all model seeds.
- Preserve original IP identities. Ports may be features but do not create fictitious physical devices. No random address replacement.
- Never infer causal labels from intrusion labels. Do not silently merge rare classes or relabel model outputs by table order.
- Conversation/host-pair grouping is not unseen-host evaluation. Run host-held-out separately; record all cross-host excluded flows. Use time splits only when genuine timestamps exist.
- Exact halos preserve fixed-model scores. A cluster-only method that discards inter-partition targets is not an acceptable fixed-population comparison.
- Record graph and target-edge counts separately. Reverse message arcs do not double test support.
- Separate one-time cold preprocessing/partitioning, warm-up, halo packing/transfer, model forward, output collection, complete inference, and training costs. If reuse is claimed, report cold and amortized cost with explicit reuse count.
- Count every partition and microbatch. Include empty partition counts, cut-edge fraction, halo duplication, and per-partition workload distribution.
- Run five paired training seeds. Average timing repeats within seed before intervals; do not treat repeats as extra independent models. Add paired seed differences and appropriate uncertainty before journal claims.
- Treat OOM/unavailable as outcomes. Do not fill tables with estimates or legacy values. Never optimize on test metrics.
- Report per-class support, macro/weighted precision/recall/F1, attack false-positive rate, probability-based AP, and confusion matrices. Include seed spread and failure cases.

## Contribution gate

METIS plus GraphSAGE alone is established engineering integration. A defensible submission needs evidence for a narrower contribution, potentially an auditable coverage-preserving partition study and quantitative boundary-context/cost analysis. Whether that is sufficient for a particular journal remains an open scientific judgment. If exact halos provide no useful systems benefit, report that result and reconsider the design; do not change test populations to create a speedup.

## Near-term execution order

1. Finish correctness tests and synthetic smoke artifacts; publish the isolated branch.
2. Resolve v2 download delivery, verify files and original label spellings.
3. Run a small real binary pilot; inspect false positives and model inputs.
4. Run multiclass pilots and verify every label is represented and mapped correctly.
5. Add fresh-process benchmarking and efficient persisted CSR/array storage before millions of rows.
6. Measure pilot scaling, update the supervisor estimate, then launch the preregistered full matrix.
7. Generate the final manuscript artifacts and rewrite Overleaf from the verified outputs.
