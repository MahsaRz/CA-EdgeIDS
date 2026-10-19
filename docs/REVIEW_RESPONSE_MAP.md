# Reviewer concerns and revised evidence

This is an internal revision checklist, not a claim that the rejected paper is now acceptable or a response sent to the journal. The revised task is binary flow intrusion classification. Changing that task is essential: intrusion annotations cannot validate root-cause identification.

| Review item | Revised treatment | Evidence and remaining limits |
|---|---|---|
| Editor; R1.2; R3 major 1: causal labels and RCA | Replace RCA with one prediction per original labeled flow; remove initiating-node and propagation conclusions | Source schema and original label audits. No causal ground truth or physical-fault experiment is claimed |
| R1.1; R3 major 3: novelty and inherited samplers | State an empirical contribution; specify inherited message passing and METIS; compare exact and sampled training separately | Five graph settings plus full context, MLP/Forest and five training procedures at five seeds. Cluster/node sampling are explicit edge-task adaptations, not original-paper replications |
| R3 major 2: manuscript versus code | Specify PyTorch, original IPs, ten flow fields, two degree features, edge targets, training-only preprocessing, grouped splits and exact coverage | Resolved configs, frozen sources, manifests and equation/gradient tests. Historical notebooks remain clearly separated |
| R1.3; R3 major 4: misleading latency | Measure all original targets and all partitions; separate forward calls from complete inference and checked first-prediction pipeline cost | Original pilot and layer-wise confirmation complete. Larger workload and matched classifier-cost grids are frozen and awaiting execution after training |
| R3 major 5: complexity and distributed claims | Give arc-linear full-message complexity at fixed dimensions; describe halo repetition separately; remove distributed deployment claims | Actual execution is one CPU workstation; no multi-worker or communication experiment |
| R1.4; R3 major 6: topology/generalization | Describe pair-held-out development results accurately; add a separately frozen new-host incident-flow holdout | Original infrastructure may be shared. ToN/CSE test attacks: 12/22 flows involving 9/20 held-out novel hosts. Every CSE held-out attack is Infiltration. BoT has no held-out attacks; no broad topology claim follows |
| R3 major 7: minority metrics and baselines | Keep all five seeds; report per-class and macro/weighted results, FPR, AP and separate trapezoidal PR-AUC; compare model processing costs | Main quality and graph/sampler controls complete. All class supports remain visible. No threshold tuning or rollback based on test scores |
| R3 major 8: inconsistent results | Distinguish complete-source audit, uniform sample, split, original targets, message arcs and source/model cohorts | Tables identify the original systems checkpoint separately from later classifier refits; original row-ID/checkpoint digests anchor reports |
| R2.1: system caption | New pipeline figure explains data preparation, training decisions, inference context and evaluation labels | Labels never become inference features; offline test traffic supplies context |
| R2.2: unreferenced figure | Reference every current figure/table in the surrounding text | Final compiled cross-reference and visual checks remain before upload |
| R3 minor 1: overloaded K | Use L for GNN depth and P for partitions | Code depth and ownership are separate |
| R3 minor 2: unspecified windows | Explicit offline split snapshots | No event timestamp was identified; row order/duration cannot supply temporal windows |
| R3 minor 3: ambiguous node labels | Labels stay on original flow targets | No majority vote or single label for a host with mixed incident flows |
| R3 minor 4: graph direction | Original flows directed; reverse context arcs explicit | METIS alone receives simplified undirected adjacency; the message multiset preserves flow multiplicity |
| R3 minor 5: sigmoid versus argmax | Weighted cross entropy and two-logit softmax; fixed argmax | Ranking metrics use probabilities; no test-selected threshold |
| R3 minor 6: hyperparameters | Report feature allowlist, optimizer, architecture, seeds, limits, validation rule and actual histories | Fixed validation choices differ explicitly across classifier, sampler and original systems cohorts |
| R3 minor 7: heatmaps | Keep actual count confusion matrices separately from per-class report tables | Do not label classification-report heatmaps confusion matrices |
| R3 minor 8: placeholder citation | Replace the old reference list with checked primary sources relevant to the new task | No unresolved publication placeholder retained |
| R3 minor 9: technical/language revision | New title, structure, model description and restrained discussion | Private draft exists; final empirical completion, rendering and Overleaf update remain |

The complete-source multiclass gate fails under the declared pair split, and no genuine event-time protocol is available. These are retained feasibility results. The revision does not manufacture class support, choose another split seed from test outcomes, or expand the number of physical hosts by changing IP identities.
