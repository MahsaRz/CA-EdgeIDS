# New-host incident-flow results

All 30 declared ToN/CSE models completed and were independently checked against saved probabilities, original row IDs, labels, class supports and validation-best epochs. Their checkpoint hashes were published at `fd7eda3` before any new-host test predictions. The five seeds and earlier hyperparameters were retained without another search. ToN fits use source `5e4fb3a`; CSE fits use `62d7f9f`. The two arithmetic-preserving engineering interruptions are recorded in [execution notes](EXECUTION_NOTES.md).

This protocol fits on one-million-flow uniform samples and evaluates flows incident to held-out novel IP identities absent from all original development rows. Known infrastructure may be shared. It is not an all-host-disjoint, temporal, episode-independent or physical-device deployment test.

| Dataset | Model | Macro F1, mean +/- sample SD | Attack recall | Attack FPR | Attack AP |
|---|---|---:|---:|---:|---:|
| ToN | GNN | 0.5378 +/- 0.0437 | 0.1333 | 0.0288 | 0.0790 |
| ToN | MLP | 0.4103 +/- 0.0145 | 0.6833 | 0.3821 | 0.0362 |
| ToN | Forest | 0.6897 +/- 0.0102 | 0.5667 | 0.0236 | 0.2298 |
| CSE | GNN | 0.5044 +/- 0.0107 | 0.0182 | 0.0056 | 0.0092 |
| CSE | MLP | 0.5152 +/- 0.0018 | 0.1364 | 0.0093 | 0.0086 |
| CSE | Forest | 0.5060 +/- 0.0144 | 0.0091 | 0.0004 | 0.0279 |

Recall, FPR and AP are means over the five model seeds, not population estimates from independent captures. ToN has 660 benign and 12 attack test flows; CSE has 15,539 benign and 22 attack test flows. The attacks involve only nine held-out novel ToN hosts and twenty held-out novel CSE hosts. Every CSE validation/test attack is Infiltration. ToN/CSE validation attack supports are only 11/35. BoT has no attacks in either new-host holdout and is not fitted.

The results do not establish reliable new-host adaptation. CSE GNN weighted F1 is 0.9951 despite mean attack recall of 1.8%; all three CSE models have weak attack recall. ToN MLP's 68.3% attack recall accompanies a 38.2% false-positive rate. Forest is stronger than the tested neural models on ToN, but its 12-attack test population is too small for a broad transfer claim. The original fixed argmax rule remains unchanged; no threshold is selected from these test scores.

These findings concern a different training/split protocol from the 100,000-flow host-pair development study. They must not be mixed into that table or interpreted as a controlled causal explanation of a performance difference. All individual metrics, confusion counts, summaries and target/label digests are in `artifacts/novel-host-1000000/test-evaluation/`. No raw flows, identities, model files or per-flow predictions are exported. Separate AP and trapezoidal PR-AUC are included in the final ranking-area bundle.
