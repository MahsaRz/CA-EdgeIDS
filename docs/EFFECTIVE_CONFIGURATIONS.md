# Reading the actual experiment settings

Dedicated study drivers start from a dataset configuration, then set their own split, model seeds, hyperparameters and evaluation schedule. The frozen study plan and individual checkpoint-selection records govern those runs. Some historical files named `resolved_config.json` still contain unused or overridden fields from the original pilot. They are preserved as the actual files written by that source version rather than silently rewritten after testing.

The new `effective_settings.json` sidecars bind explicit executed settings to each complete selected-model manifest. They record every model's actual seed, neural rate or Forest depth, checkpoint/source hash, epoch decision and optimization budget, along with the real split, input hash, feature width and class supports. Exporting this description does not change weights, predictions, selections or earlier files.

For example, the earlier new-host source obtains its split by calling the frozen novel-host assignment function directly. Its inherited base field `split_protocol: conversation` does not control that call. The actual new-host test IDs are independently checked against the preflight hash. Its inherited `seeds: [7]` also does not control the dedicated five-seed loop. Likewise, the sampler cohort's dataset base rate is not the separately selected rate passed to each method. Benchmark repetition fields in these base files do not schedule the quality study's predictions.

The current source will write the new-host split, source-audit path and complete model-seed list explicitly after the frozen historical experiment finishes. Historical experiments retain their source snapshots, original configurations and new explanatory sidecars. Table B and the manuscript describe the executed procedures, not every inherited base field.

Generate sidecars only after all selected models exist:

```text
python scripts/export_effective_settings.py --study controlled
python scripts/export_effective_settings.py --study sampler
python scripts/export_effective_settings.py --study novel-host
```

In a clone containing published sidecars, pass `--out` with a new ignored path, for example `runs/reproduction-exports/controlled-effective-settings.json`. The exporter refuses to overwrite an earlier description and checks the complete cohort size. This metadata clarification is separate from the fixed-weight numerical/source and original-target checks.
