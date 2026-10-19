# Execution notes

The initial million-flow new-host fit was stopped during the seventh epoch of its first model to remove redundant topology scanning. No model checkpoint or test prediction had been produced. The six completed validation macro F1 values, rounded as printed, were 0.51446, 0.50884, 0.53362, 0.55395, 0.52275 and 0.51772. Original private preparation files and the frozen plan are retained in `runs/novel-host-1000000-initial-context-attempt/`. This is an interrupted engineering attempt, not a completed model or a candidate to select.

The arithmetic backend now obtains each query's neighborhood closure from the summarized sparse neighbor-pair matrix rather than scanning every repeated flow arc. This changes only a set operation: the exact sorted node set is the same, including at multiple depths and with repeated flows. Multiplicities, means, tensor arithmetic, target batches, dropout ordering and optimizer settings are unchanged. Tests check both closure identity and bitwise-identical fitted weights/histories in a deterministic fixture, in addition to the existing output/gradient comparisons.

Restart the new-host study from the original initialization and unchanged settings after committing this change. Preserve the earlier printed validation history as an audit of the arithmetic-only restart; do not select settings or epochs from that interrupted run. All new models still select epochs using only their own full validation histories. The completed controlled and sampler studies retain their own earlier source snapshots and results.

Some training and data preparation ran concurrently. Reported fitting wall times are diagnostic, not isolated comparisons of training speed. The fresh-process inference grid begins only after those jobs finish.
