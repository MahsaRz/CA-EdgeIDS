# Validation-selected classifier comparison

This is a single-seed development study on the existing 100,000-flow samples, whose test results have already been inspected during earlier pilots. It cannot be described as an untouched final evaluation. Keep the same original flow identities, features, train-fitted preprocessing and host-pair splits. This stage compares prediction quality; inference timers in its shared process are diagnostic and must not become a systems benchmark.

## Frozen selection budget

The plan is `configs/classifier_comparison_100000.json`. Each of GraphSAGE, a flow-only MLP and Random Forest receives two predefined candidates. Both neural candidates use learning rates 0.001/0.003; the forests use maximum depth 16/unrestricted with 100 trees each. There is one initialization seed, 7. Candidate selection and checkpoint selection use validation macro F1 only; exact ties choose the first declared candidate. Prediction uses fixed argmax with no threshold search.

The neural models share hidden width 16, dropout 0.2, Adam weight decay 0.0001, maximum 10 epochs, patience 3, gradient clipping 5 and the same training-class-balanced cross entropy. They use identical training target batches, formed by the original random source ownership, with at most 512 targets per batch. Both shuffle those batches using seed 7 each epoch. GraphSAGE has two message layers and an edge decoder; the MLP has two hidden layers and receives the same transformed flow features without graph context or degree features. Their parameter counts differ and are reported.

The original learning-rate-0.001 GraphSAGE fits are reused after checking their saved configuration, seed, source sample, split, checkpoint and validation score. Reuse avoids repeating an identical fit and is explicitly recorded alongside its original measured training time. The second GraphSAGE candidate and both MLP/forest candidates are fitted for this stage. The forest uses training-class balancing, one CPU thread, bootstrap sampling and square-root feature subsampling; it has no epoch-based stopping rule.

This matches the number of search candidates and the neural upper epoch/update budget. It does not equate CPU time, model capacity or forest trees with neural epochs. Actual epochs, updates, parameters/tree nodes, fit times and all validation scores must accompany the selected results. Ten-epoch selection is an initial development budget, not evidence that any family has converged or that two trials suffice for a final benchmark.

## Selection and evaluation gate

`python scripts/run_classifier_comparison.py --stage select` fits/checks all candidates using only training and validation inputs for model selection. It builds no test graph and computes no test predictions. Test label/schema/support checks in data preparation remain integrity checks, not a selection metric. The completed `selection.json` freezes one checkpoint per family/dataset before the evaluation stage. Its file hash and the selected model hashes are recorded before test predictions.

Only then run `python scripts/run_classifier_comparison.py --stage evaluate`. It evaluates the nine selected checkpoints on the original test row IDs and reports full per-class metrics, macro/weighted F1, attack false-positive rate, average precision and actual confusion matrices. Do not evaluate every candidate on test or retune after seeing these results. Private run directories refuse to overwrite earlier outputs. Keep raw data, checkpoints and per-flow predictions outside Git.

BoT still has only 11 Benign validation examples, so its model-selection uncertainty remains large. Neither model selection nor better test metrics establishes unseen-host, topology, causal or distributed generalization. Five independently trained seeds, stronger generalization protocols, matched graph sampler baselines and a viable multiclass protocol remain separate gates.
