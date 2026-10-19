# Rebuild status - 19 October 2026

## Latest completed milestone

The [layer-wise inference follow-up](CACHED_INFERENCE.md) now addresses repeated computation with a tested global node-state pass and chunked messages/decoding. The separate confirmation study completes 54 fresh-process conditions and 162 warm timings across three independently ordered blocks. All predictions remain unchanged, with no missing targets or failures. The 41-test Linux workflow passes. Sampled RSS falls on these graphs, while speed relative to Full depends on dataset/chunk settings. This is a global inference baseline, not evidence of a cluster-specific advantage.

Next: validation-only model development and matched baselines, then independent model seeds and larger controlled workloads. Minority quality and multiclass feasibility remain open. Earlier milestones below retain their original scope.

The [100,000-flow Linux binary pilot](LINUX_SCALING_100000.md) is complete for all three datasets. Real-data METIS ran successfully under local Ubuntu/WSL2. The verified bundle contains 79 completed conditions and 237 warm timing measurements, including 32 METIS conditions. Eight BoT eight-partition conditions are infeasible because that test graph has six IP nodes. Every completed condition preserves exactly-once target coverage and the saved Full class predictions. Full is fastest among completed conditions for every dataset; no consistent process-memory benefit is established.

Source snapshots match frozen experimental commit `a390799`. All 33 extended Linux tests passed at `b5026da`. Tables, validation histories, confusion matrices, timing components and requirements are in `artifacts/linux-scaling-100000/`. ToN's false-positive rate is 18.6%; BoT's macro F1 is 0.578 despite weighted F1 of 0.976. These limitations are part of the reported result. Earlier milestone counts below describe their original runs.

At that milestone the next gate was repeated exact-context computation; the layer-wise follow-up above now provides a tested alternative. Full-source repetitions, validation-only model development, matched baselines and independent seeds remain outstanding. Multiclass remains paused at its documented split-feasibility gate. Overleaf remains unchanged.

## Completed locally

- Read the supervisor's recommendation and the two-page tables/figures brief; adopted intrusion detection / flow-edge classification as the revised scope.
- Preserved the six historical notebooks and created local branch `revision/ca-edgeids`.
- Built a modular PyTorch project, six dataset/task configurations, official-archive audit and uniform-sampling scripts, correctness tests, reporting, and a Linux CI workflow.
- **21 correctness tests passed** on Windows/Python 3.11/PyTorch 2.9 CPU. The tests cover label exclusion, training-only scaling, label consistency and original-name preservation, CSE-family mapping, split isolation, target coverage, halo/full equivalence at depths 1-3 in both message directions, label-independent inference, metrics, safe serialization, missing-validation-class rejection, and the actual single-forward Full baseline.
- Downloaded all three official UQ v2 archives. Full CSV scans and archive-manifest checksum checks passed for 16,940,496 ToN flows; 37,763,497 BoT flows; and 18,893,708 CSE flows. Source audits are in `docs/provenance/`.
- Completed one **5,000-flow uniform binary pilot for each dataset**, with one model seed, initial untuned settings, validation checkpoint selection, Full/Random exact-context inference and fixed target IDs, CPU memory measured in fresh per-configuration processes, RF/MLP diagnostic baselines, and aggregate output generation.
- Generated Tables A-E, Figures 1-4, supplemental confusion matrices, CSV/LaTeX and vector SVG/300-dpi PNG artifacts. Every empirical chart is marked as a pilot; these are not final paper results.

## What the pilots establish

They establish that the complete data-to-output path runs on real data, label mappings are explicit, every retained test flow is scored once, and exact partitioned inference preserves the frozen model's predictions. They do **not** establish superior accuracy, speed, scalability, causal inference, or adaptation. The initial ToN GNN is worse than RF/MLP on this pilot, and its total inference time increases with extra exact-context work. Full results may differ; test observations must not be used to tune the model.

The MLP pilots reached their small iteration budget and emitted convergence warnings, which remain part of the diagnostic record. RF/MLP tuning budgets and benchmark process isolation are not matched to the GNN yet. These comparisons cannot be advertised as a final SOTA benchmark.

**BoT pilot protocol failure:** the 5,000-row grouped pilot has zero Benign validation examples (train: 13; test: 7). Its outputs are retained as a diagnostic failure, not valid evidence of all-class model selection. The pipeline now explicitly checks validation class coverage and stops by default when a class is absent. Do not quietly change seeds or relax isolation using test outcomes. A larger label-independent sample or a justified alternate protocol must be reviewed first.

The predefined larger 30,000-row BoT uniform sample was then run with unchanged split rules and model settings. It has 68 Benign training flows and 6 Benign validation flows, so it passes the class-coverage guard, although minority uncertainty remains large. Its test graph has only **four original IP nodes**, and the requested eight-partition condition is explicitly rejected. The larger diagnostic run is in `artifacts/pilots/bot-binary-30000/`; the small failed run remains clearly marked. Never expand graph size by randomizing source IP addresses to make a partition-count experiment possible.

## Open work

1. METIS remains unavailable in the inspected Windows environment, but both synthetic Linux verification and the larger real-data Linux pilot are now complete. Continue larger studies on the recorded Linux environment; do not reinterpret historical unavailable Windows conditions as measurements.
2. Resolve the multiclass feasibility gate before fitting multiclass models; then complete independently trained partition studies, controlled Cluster-GCN/GraphSAINT baselines, graph/edge-feature ablations, validation-only tuning and generalization studies.
3. Profile memory/halo expansion on larger pilots before loading millions of flows into the current in-memory graph builder. Full dataset repetitions, five seeds, equal tuning budgets and final uncertainty remain outstanding.
4. Rewrite the manuscript using verified new outputs. **Overleaf has not been modified.**
5. The repository is now `MahsaRz/CA-EdgeIDS`, with the working code on `revision/ca-edgeids`. Current draft review: https://github.com/MahsaRz/CA-EdgeIDS/pull/2 . It supersedes the closed fork-based draft #1. The repository history is preserved and main is unchanged.

## Next-phase findings

**31 correctness tests pass locally.** A new `preflight` command reports class support, class-specific host-pair counts, shared-host counts, and split feasibility without fitting a model. The run guard now also rejects missing training classes before allocating graphs. A regression fixture checks complete-source pair counts, reverse-direction grouping, and provenance count mismatches. The 29-test Linux run passed at commit `1e8c564`; the new batch-study tests extend that set.

All three 30,000-flow multiclass samples fail the all-class selection gate. Complete-source streaming checks of **73,597,701 flows** confirm that the frozen seed-42 pair split has no validation Ransomware in ToN, no validation Theft in BoT, and no validation BruteForce or DoS in CSE. CSE has no test BruteForce either; that family occupies only two pairs across the entire source. Increasing sample size cannot resolve these complete-source absences. No multiclass models were trained and no seeds were searched. See [the measured feasibility report](MULTICLASS_FEASIBILITY.md) and `artifacts/preflight/` for exact supports and a reproducible figure.

Binary scaling pilots can proceed independently where both training/validation classes exist. Full multiclass experiments require a documented protocol revision and appropriately limited claims. The new audit evidence must be reflected in the manuscript rather than hidden by changing seeds or labels.

The **30,000-flow ToN and CSE binary scaling pilots are complete**. All fixed targets are scored exactly once and Full/Random probabilities agree; partitioned total inference is slower in both pilots. See [binary scaling evidence](BINARY_SCALING_PILOTS.md). The aggregate bundles retain metrics and timing components, with METIS explicitly unavailable. These are one-seed initial-setting experiments, not final paper results; no claim of a performance advantage is supported.

The [graph-work profile](GRAPH_WORK.md) now includes all three 30,000-flow pilot test graphs. Small batches repeat message-edge work about 12–18 times. A one-partition batched control explains much of this repetition, so the cost must not be attributed entirely to partitioning. New runs use read-only memory-mapped graph arrays and explicit input-file/packed-tensor limits. Tests cover the new storage in fresh workers. These changes do not establish a total RAM bound or a measured memory benefit.

The [batch-size study](BATCH_SIZE_STUDY.md) is complete for all three 30,000-flow samples: 47 completed conditions, 141 timing records, and 40 explicit unavailable/infeasible conditions. Predictions match the saved references within tolerance, class predictions are identical, and all targets are scored once. Larger batches reduce batched inference time, but Full is fastest among the completed conditions in each dataset. The current benchmark code passed 31 local tests and the Linux workflow at `5e9325f`.

**Earlier Linux verification passed:** https://github.com/mahsaraeiszadeh/DC-RCA/actions/runs/39277093598 (core commit `8e44c9b`). All tests and the synthetic smoke workflow passed. The downloaded aggregate CI tables were independently inspected: actual METIS results exist for 2, 4 and 8 partitions, with zero unavailable configurations. `docs/LINUX_VERIFICATION.json` records the scope. This validates the METIS software path on Linux, **not real-data METIS accuracy or performance**. Real-data experiments were pending at that milestone; the completed larger Linux pilot is reported above. The workflow additionally requires all three methods to appear, so missing METIS cannot silently yield a green smoke check.

The final full-file node audit found **29,245 ToN IP nodes; 293 BoT IP nodes; 255,042 CSE IP nodes** before splits. This confirms that the original 1000-2000 partition sweep is infeasible for BoT with genuine IP identities. Full BoT configs now propose 2/4/8/16/32 partitions and continue to reject any value above the actual split's node count.

## Time estimate

The full revision plan is approximately **4-6 weeks** with stable data access and appropriate compute, revised after the scaling pilot. Code and first binary pilots are already available locally. Do not promise a full three-dataset multi-seed journal evaluation from these small pilot runtimes.
