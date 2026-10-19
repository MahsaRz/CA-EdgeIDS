# Rebuild status - 19 October 2026

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

1. METIS has no compatible PyMETIS wheel in the inspected Windows/Python environment. Those runs are explicitly unavailable. Validate on the provided Linux workflow or a suitable Linux machine; do not substitute another partitioner under its name.
2. Complete real multiclass experiments, independently trained partition studies, controlled Cluster-GCN/GraphSAINT baselines, graph/edge-feature ablations, validation-only tuning and generalization studies.
3. Profile memory/halo expansion on larger pilots before loading millions of flows into the current in-memory graph builder. Full dataset repetitions, five seeds, equal tuning budgets and final uncertainty remain outstanding.
4. Rewrite the manuscript using verified new outputs. **Overleaf has not been modified.**
5. Published on fork branch `mahsaraeiszadeh/DC-RCA:revision/ca-edgeids` and opened draft PR https://github.com/MahsaRz/DC-RCA/pull/1 . Direct write access to the original repository was unavailable, so the normal fork workflow was used. The original repository's main branch is unchanged.

**Linux verification passed:** https://github.com/mahsaraeiszadeh/DC-RCA/actions/runs/39277093598 (core commit `8e44c9b`). All tests and the synthetic smoke workflow passed. The downloaded aggregate CI tables were independently inspected: actual METIS results exist for 2, 4 and 8 partitions, with zero unavailable configurations. `docs/LINUX_VERIFICATION.json` records the scope. This validates the METIS software path on Linux, **not real-data METIS accuracy or performance**. Real-data experiments remain pending. The workflow additionally requires all three methods to appear, so missing METIS cannot silently yield a green smoke check.

The final full-file node audit found **29,245 ToN IP nodes; 293 BoT IP nodes; 255,042 CSE IP nodes** before splits. This confirms that the original 1000-2000 partition sweep is infeasible for BoT with genuine IP identities. Full BoT configs now propose 2/4/8/16/32 partitions and continue to reject any value above the actual split's node count.

## Time estimate

The full revision plan is approximately **4-6 weeks** with stable data access and appropriate compute, revised after the scaling pilot. Code and first binary pilots are already available locally. Do not promise a full three-dataset multi-seed journal evaluation from these small pilot runtimes.
