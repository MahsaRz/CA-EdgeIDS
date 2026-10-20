# Rebuild status - 20 October 2026

The declared experiments are complete: 225 selected-model evaluations (120 classifier/graph controls, 75 training-procedure controls and 30 new-host models), 36 matched classifier-cost conditions and 135 larger-workload requests. All 36 classifier-cost and 129 successful larger-workload conditions preserve class decisions within the declared probability tolerance; six CSE one-million-flow partition conditions time out. The current numerical package passes 71 Linux correctness tests. AP and trapezoidal PR-AUC are separately reported for all 225 models.

Forest leads the original ToN/BoT classifier comparison and the full GNN leads CSE. Removing neighbor messages helps ToN/BoT. Layer-wise inference is the fastest tested GNN schedule in all nine larger workloads; exact partition processing is substantially slower. New-host detection is weak: CSE GNN attack recall is 1.8% despite weighted F1 of 0.9951. ToN/CSE attack test supports are only 12/22; BoT is infeasible. These results support a conditional empirical comparison, not reliable adaptation, causal diagnosis or a new partitioning algorithm.


## Manuscript release

The private Overleaf project now uses `ca-edgeids-main.tex` and is named **CA-EdgeIDS: Graph Context and Processing Cost**. Its compiled paper is **Graph Context and Processing Cost in Flow-Level Intrusion Detection**. The original source files, authors, affiliations and funding acknowledgment are preserved. Live compilation reports zero errors; an unused inherited document-class option and underfull typesetting notices are non-blocking. The actual downloaded Overleaf PDF is reviewed separately from the local compiler output. No journal submission or merge to main has been made.

## Completed evidence and historical studies

| Study | Evidence |
|---|---|
| Source audit | 73,597,701 official flows; archive/file checksums and complete label/IP counts |
| Classifier and graph controls | 120 five-seed selected models; CONTROLLED_STUDY.md |
| Training procedures | 75 selected models; SAMPLER_COMPARISON.md |
| New-host protocol | 30 selected models; weak detection and limited support; NOVEL_HOST_RESULTS.md |
| Matched classifier cost | 36 successful fresh-process conditions; CLASSIFIER_COST.md |
| Larger inference | 129 successful conditions and six retained timeouts; LARGER_INFERENCE.md |
| Ranking reports | AP and trapezoidal PR-AUC for 225 models; artifacts/precision-recall-areas |
| Historical 100k systems pilot | 79 conditions, 237 warm measurements; eight infeasible BoT conditions; LINUX_SCALING_100000.md |
| Historical layer-wise confirmation | 54 process conditions, 162 warm measurements; CACHED_INFERENCE.md |
| Historical batch-size pilot | 47 successful and 40 unavailable/infeasible conditions; BATCH_SIZE_STUDY.md |

Earlier studies retain their original checkpoints, source hashes, test counts and measurement scopes. They are not substituted for the later five-seed quality study. The six original notebooks are historical material. Data, models and per-flow predictions remain outside Git.

## Scientific limits and author review

The contribution is an empirical evaluation, not a new graph algorithm. Multiclass split support and genuine event timestamps remain unavailable under the declared protocols. The new-host evidence does not establish reliable adaptation. No physical edge deployment, distributed execution, causal ground truth or online deadline was measured. Author/supervisor review and venue selection remain necessary; no acceptance claim is made. See CONTRIBUTION_ASSESSMENT.md and REVIEW_RESPONSE_MAP.md.
