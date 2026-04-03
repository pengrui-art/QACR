# Phase A.3 Same-Image / Different-Query Control Report

## Experiment Goal

This control experiment answers a specific question:

> On the same image, does the routing pattern change with the query in a way that better tracks query-relevant regions?

## Aggregate Summary

| Method | Flagged Errors | Key Recall | Deep Precision | Miss Rate | Separation | Shift Corr |
|---|---:|---:|---:|---:|---:|---:|
| QACR | 2 | 0.8383 | 0.6161 | 0.1617 | 0.4994 | 0.8852 |
| TokenPruning | 5 | 0.4970 | 0.3731 | 0.5030 | 0.0606 | 0.0000 |
| LVPruning-like | 3 | 0.8160 | 0.6155 | 0.1840 | 0.5563 | 0.9618 |
| CROP-like | 0 | 0.9307 | 1.0000 | 0.0693 | 0.9307 | 0.9971 |

## Main Takeaways

- `QACR` clearly outperforms `TokenPruning` on query-conditioned behavior: `key_token_recall 0.8383 vs 0.4970`, `miss_rate 0.1617 vs 0.5030`, `shift_corr 0.8852 vs 0.0000`.
- `QACR` also has far fewer flagged failures than `TokenPruning` (`2 vs 5`), which supports the claim that query-conditioned routing is doing something nontrivial.
- `LVPruning-like` remains strong on this synthetic control and even exceeds `QACR` on `shift_corr (0.9618)`, so A.3 should be used as mechanism evidence, not as a blanket claim that QACR dominates every heuristic.
- `CROP-like` is the upper-bound style oracle on this toy setting (`key_token_recall=0.9307`), so it is best interpreted as a reference point rather than a real competing method.

## Per-Query Cases

### QACR

| Query | Key Recall | Deep Precision | Miss Rate | Early Skip | Separation | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| left_focus | 0.9643 | 0.9310 | 0.0357 | 0.0357 | 0.8929 | 0 |
| right_focus | 1.0000 | 0.4286 | 0.0000 | 0.0000 | -0.0656 | 1 |
| bottom_text | 0.8286 | 0.3671 | 0.1714 | 0.2286 | 0.4624 | 0 |
| count_query | 0.9881 | 0.8737 | 0.0119 | 0.0238 | 0.9005 | 0 |
| relation_query | 0.9762 | 0.9318 | 0.0238 | 0.0238 | 0.9080 | 0 |
| center_focus | 0.2727 | 0.1644 | 0.7273 | 0.7273 | -0.1018 | 1 |

### TokenPruning

| Query | Key Recall | Deep Precision | Miss Rate | Early Skip | Separation | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| left_focus | 0.5000 | 0.4773 | 0.5000 | 0.5000 | 0.0893 | 1 |
| right_focus | 0.3571 | 0.3409 | 0.6429 | 0.6429 | -0.1607 | 1 |
| bottom_text | 0.9429 | 0.3750 | 0.0571 | 0.0571 | 0.6012 | 0 |
| count_query | 0.5000 | 0.4773 | 0.5000 | 0.5000 | 0.0893 | 1 |
| relation_query | 0.5000 | 0.4773 | 0.5000 | 0.5000 | 0.0893 | 1 |
| center_focus | 0.1818 | 0.0909 | 0.8182 | 0.8182 | -0.3445 | 1 |

### LVPruning-like

| Query | Key Recall | Deep Precision | Miss Rate | Early Skip | Separation | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| left_focus | 0.7143 | 0.6818 | 0.2857 | 0.2857 | 0.4643 | 1 |
| right_focus | 0.8214 | 0.7841 | 0.1786 | 0.1786 | 0.6518 | 0 |
| bottom_text | 1.0000 | 0.3977 | 0.0000 | 0.0000 | 0.6708 | 0 |
| count_query | 0.7143 | 0.6818 | 0.2857 | 0.2857 | 0.4643 | 1 |
| relation_query | 0.7143 | 0.6818 | 0.2857 | 0.2857 | 0.4643 | 1 |
| center_focus | 0.9318 | 0.4659 | 0.0682 | 0.0682 | 0.6226 | 0 |

### CROP-like

| Query | Key Recall | Deep Precision | Miss Rate | Early Skip | Separation | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| left_focus | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| right_focus | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| bottom_text | 0.8571 | 1.0000 | 0.1429 | 0.1429 | 0.8571 | 0 |
| count_query | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| relation_query | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| center_focus | 0.7273 | 1.0000 | 0.2727 | 0.0000 | 0.7273 | 0 |

## Usage Note

This A.3 report is best used as mechanism evidence for `same image, different query`, not as the final benchmark result. It complements A.1/A.2 by showing that the routing pattern actually shifts with the query.
