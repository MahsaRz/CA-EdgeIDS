# Layer-wise inference comparison

## Purpose and scope

The earlier exact-halo pilot recomputes message passing for every target batch. This comparison computes each node layer once for a complete snapshot, then decodes target flows in chunks. Message tensors are also chunked. Each message arc is visited once per layer, regardless of the number of target batches. This is an implementation control for the existing model, with no claim of a new GNN algorithm or distributed execution.

All node states for a layer remain resident. The input graph is still present, output probabilities are retained, and message chunks have temporary activations. A working-tensor estimate screens oversized configurations; it is not a bound on total memory. This approach does not establish that a million-node graph fits this workstation.

Node states are recomputed on every prediction call, including every timing repeat. No representation is carried over from warm-up, another graph snapshot, or another model. Dropout is disabled in evaluation, just as in the Full reference. Training and saved checkpoints are unchanged.

## Frozen comparison

Use the three saved seed-7 binary models from the 100,000-flow Linux pilot and their original test graphs and row IDs. There is no retraining, hyperparameter selection or test-label input. Freeze the code and execution order before measuring:

- One Full control and one one-partition exact-halo control with target batch size 512.
- Four layer-wise conditions: message chunks of 4,096 or 16,384 arcs crossed with target chunks of 512 or 2,048 flows.
- Three datasets, giving 18 conditions; one fresh worker per condition, one complete warm-up prediction and three complete timed predictions.
- Shuffled execution order with seed 20261020; CPU, one PyTorch thread, the same local Ubuntu/WSL2 environment.
- Exactly-once test coverage, identical class predictions and probabilities within absolute/relative tolerance 1e-5 of the original saved Full reference.
- Warm time includes all encoding, decoding, transfers and output collection. Process launch through first prediction is separate and has one observation per condition. Neither timer includes original source preparation or graph construction. File caches need not be cold.
- Retain failures and avoid selecting a winning chunk size as a tuned final configuration. These are fixed-model systems measurements, not independent trained seeds or final journal evidence.

Run `python scripts/run_cached_study.py` in the prepared Linux environment. Private outputs go to `runs/cached-inference-100000/`; source, graph-array, checkpoint and reference hashes accompany the plan. The correctness tests compare against Full at depths 1-3 in both graph directions, test self-loops, parallel flows and isolated nodes, check actual message-module visits, and ensure graph/model changes recompute the states. Cached inference does not read graph labels.
