# Linux binary pilot: 100,000 flows per dataset

This pilot checks the larger workload with real METIS, rather than substituting random partitions. Each source sample contains 100,000 flows; the actual number of test flows is determined by the frozen host-pair split and is reported separately. This is one pilot size, not a full scaling curve or a five-seed journal evaluation.

## Frozen protocol

The three official archives were scanned again with the same uniform-priority sampling seed, 2026. Archive and CSV hashes, complete-source class counts, and BagIt checksums agree with the earlier audits. No label-based sampling or split-seed search is used. The plan and sample hashes are fixed in `configs/linux_scaling_100000_plan.json` before model fitting.

The binary models keep the earlier settings: seed 7, split seed 42, 10 selected flow fields, two layers, hidden width 16, dropout 0.2, training batch limit 512, a maximum of 10 epochs, and patience 3. Training uses random source ownership with exact context. Validation macro F1 selects the checkpoint. The models are retrained on the larger training samples, then frozen for all inference conditions. RF/MLP fitting is outside this systems pilot.

All conditions run on the same local Ubuntu environment under WSL2, using CPU PyTorch 2.9.0 and PyMETIS 2025.2.2. The main numerical library versions match the earlier Windows pilots. CPU model, available Linux memory, package versions and source hashes are recorded. Timings must not be compared directly with the earlier Windows measurements as though only workload size changed.

The inference matrix includes one Full pass plus one-partition, Random and METIS conditions at batch limits 256/512/1024/2048 and partition counts 2/4/8. Each condition runs in a fresh process, completes one full warm-up pass, then measures three complete graph predictions. All test targets must be scored once; probabilities must agree with the frozen Full reference within the stated tolerance and class predictions must be identical. Infeasible partitions, timeouts and memory failures remain outcomes.

Warm inference, packing, model forward time, partitioning, graph loading and checkpoint loading are recorded separately. First-prediction process time includes launch through the first complete prediction. It has one observation per condition, not three independent repeats, and does not include source CSV preparation or graph conversion. The operating system's file cache need not be cold. Process RSS is sampled at 5 ms; it excludes the parent process and the host/VM memory footprint.

## Execution

The code passed all 31 tests in the actual Linux environment with METIS installed. The fresh-worker integration check also passed on Windows after adding the first-prediction timings.

Run from the repository root in the prepared Linux environment:

```text
python scripts/run_linux_scaling.py
```

Private run files are written under `runs/linux-scaling-100000/`. Aggregate results will be published after the whole matrix is checked. The frozen plan, validation histories and complete failure records will accompany the figures; raw flows, model checkpoints and per-flow predictions stay local.
