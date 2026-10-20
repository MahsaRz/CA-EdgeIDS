PILOT SUBSET - NOT FULL DATASET

Tables C and D use the same checkpoint per seed. Exact K-hop halos should keep F1 invariant. This is an inference-system comparison, not evidence of superior training accuracy.

95% intervals are Student-t intervals over training seeds after averaging timing repetitions within seed. Missing intervals indicate one seed. Memory is sampled process RSS, including resident input graphs; not an allocator-exact CPU peak.

Unavailable runs (never replaced with invented values):
[
  {
    "seed": 7,
    "method": "random",
    "partitions": 8,
    "error": "Requested 8 partitions for 4 nodes"
  },
  {
    "seed": 7,
    "method": "metis",
    "partitions": 2,
    "error": "METIS unavailable. Install the metis extra on a supported platform; no substitute is called METIS."
  },
  {
    "seed": 7,
    "method": "metis",
    "partitions": 4,
    "error": "METIS unavailable. Install the metis extra on a supported platform; no substitute is called METIS."
  },
  {
    "seed": 7,
    "method": "metis",
    "partitions": 8,
    "error": "Requested 8 partitions for 4 nodes"
  }
]