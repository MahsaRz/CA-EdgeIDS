# Reproducing the studies

Use a clean Git clone of `revision/ca-edgeids` on Linux. The experiment drivers check committed source bytes; a source ZIP alone does not include the history needed for historical verification. Each run refuses to overwrite an earlier private run. Aggregate export paths below are new, ignored folders, so the checked-in evidence stays intact.

## Environment and exact samples

Install the pinned Linux packages, then install this project without changing those versions:

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -r artifacts/linux-scaling-100000/requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Use the official v2 archives linked in [data sources](DATA_SOURCES.md). Create a private JSON file with keys `ton`, `cse` and `bot`, each containing the path to its official archive. It contains local paths only and stays outside Git. The scripts verify the release archive/CSV checksums and full counts rather than accepting a similarly named third-party CSV.

```text
python scripts/prepare_reproduction_samples.py --archive-map /path/to/private-archives.json
```

This scans all three complete sources and writes the exact 100,000-flow uniform samples expected by the checked-in configs. If the samples already exist, `--check-existing` checks their content and row-ID digests against the frozen protocol; it does not repeat the full archive scan. Do not run `make_linux_scaling_configs.py` against an ordinary clone: the frozen configs are already included.

## Baseline run folders

```text
python scripts/run_linux_scaling.py
python scripts/verify_linux_scaling.py
```

This creates `runs/linux-scaling-100000/`, including each dataset's original pilot model, fitted preprocessor, training/validation histories, test graph and fixed-model benchmark anchors. Later drivers deliberately check these local manifests. They are regenerated here rather than assumed to be available from the original machine. The original three-model systems cohort uses learning rate 0.001 and remains separate from the classifier refits.

## Five seeds and graph controls

```text
python scripts/run_controlled_study.py --stage train
python scripts/report_controlled_study.py --stage train --public-root runs/reproduction-exports/controlled
python scripts/run_controlled_study.py --stage evaluate
python scripts/report_controlled_study.py --stage evaluate --public-root runs/reproduction-exports/controlled
```

Inspect/freeze the exported selection manifest before the evaluate command. It must contain all 120 models and checkpoint hashes. The settings are the already declared validation-selected settings; this stage performs no new hyperparameter search. The default private folder is `runs/controlled-binary-100000/`. `--run-root` can choose another folder when replaying in an existing checkout; pass the same value to both runner and reporter.

## Training procedures

```text
python scripts/run_sampler_study.py --stage select --public-root runs/reproduction-exports/sampler
python scripts/run_sampler_study.py --stage seeds --public-root runs/reproduction-exports/sampler
python scripts/run_sampler_study.py --stage evaluate --public-root runs/reproduction-exports/sampler
python scripts/report_sampler_study.py --public-root runs/reproduction-exports/sampler
```

Freeze the 30 validation candidates/15 choices before `seeds`, then all 75 selected checkpoint hashes before `evaluate`. Full-snapshot validation/test context and identical original targets are used by all five procedures. Exposure and empty-update counts remain part of the comparison. Use the same `--run-root`/`--public-root` on every stage if overriding them.

## Larger samples and new-host protocol

```text
python scripts/create_larger_samples.py --archive-map /path/to/private-archives.json
python scripts/novel_host_protocol.py --out runs/reproduction-exports/novel-host-preflight
python scripts/run_novel_host_study.py --stage train --public-root runs/reproduction-exports/novel-host
python scripts/run_novel_host_study.py --stage evaluate --public-root runs/reproduction-exports/novel-host
python scripts/report_novel_host_study.py --public-root runs/reproduction-exports/novel-host
```

The sample stage verifies all complete sources again and reproduces the original development row priorities. New-host inputs are checked against the declared preflight's CSV/target digests and supports. Freeze selected checkpoints between train and evaluate. ToN/CSE are feasible but have only 12/22 attack test flows; BoT fails and is not fitted. No seed search, class dropping or temporal labels are used to repair this. `--run-root` and export overrides work as above.

## Larger processing study

Finish training and full-source scans before timing:

```text
python scripts/run_larger_inference.py
python scripts/report_larger_inference.py --public-root runs/reproduction-exports/larger-inference
```

This uses the locally regenerated baseline's own checkpoint/preprocessor anchors and a fixed 135-condition grid. All three process blocks, every success and every resource failure must remain. The report checks numerical differences and class decisions against Full, or the independent layer-wise implementation where Full is rejected. It never relabels a mismatched condition equivalent. New model-file hashes and wall times can differ across hardware; the local run's own provenance is the anchor, not a requirement to possess the original private checkpoints.

## Figures and published numbers

```text
python scripts/build_paper_assets.py --complete
```

This rebuilds paper assets from the checked-in verified aggregate evidence, without requiring raw records or checkpoints. A replay's new exports are kept separately; compare their manifests, supports and metrics before explicitly replacing any publication bundle. Rebuilding plots alone is not a new model evaluation.

The frozen source commits and input/selection/target digests are recorded in each bundle. The current arithmetic-preserving neighbor-index improvement has exact closure/fitted-weight tests; historical experiments retain their own source snapshots. Graph-classification results use one reused development split, and system timing repeats do not become independent model seeds. See [claim checks](CLAIM_CHECKS.md) for the supported interpretation.
