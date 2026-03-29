# Phase 3.9 Baseline Alignment Matrix

## Main Table Candidates

| Method | Family | Query-guided | Region Compression | Layer Skipping | Multi-path Depth | Budget | Local Alignment | Why Main Table |
|---|---|---|---|---|---|---|---|---|
| LowRes-9x9 | uniform_input_compression | no | yes | no | no | implicit | `scripts/train_low_resolution_baseline.py` | Strong compression baseline; mandatory for any honest QACR comparison. |
| TokenPruning-keep/drop | heuristic_token_pruning | no | no | no | no | fixed_keep_ratio | `scripts/train_token_pruning_baseline.py` | Nearest generic pruning baseline already available in repo. |
| QACR-DepthOnly | budgeted_multi_path_compute_allocation | yes | no | no | yes | yes | `scripts/train_query_adaptive_budget_sweep.py` | Core method; current novelty rests on compute allocation rather than keep/drop selection. |
| LVPruning | query_guided_token_pruning | yes | no | no | no | implicit | `text_guided_keep_drop_family` | Closest representative of language-guided keep/drop pruning; must be in main table. |
| CROP | query_relevant_region_compression | yes | yes | partial | no | implicit | `region_compression_family` | Most relevant region-aware compression baseline; main-table representative for spatially localized compression. |
| SPIDER | prune_plus_sub_layer_skip | semantic / weak query dependence | no | yes | partial | implicit | `prune_then_skip_family` | Best representative for compute-side skipping; main-table foil for the 'not just pruning' claim. |

## Appendix / Supplementary Candidates

| Method | Placement Reason | Source |
|---|---|---|
| ImageOnlyRouting | Needed to isolate the value of query conditioning. | internal |
| Script | Very relevant but highly correlated with LVPruning-family; better placed in appendix unless it is the strongest reproduced pruning result. | https://openreview.net/forum?id=F6xKzbgcHq |
| FlashVLM | Very relevant but still under submission; useful appendix stress test rather than mandatory main-table baseline. | https://arxiv.org/abs/2512.20561 |
| DyRate | Useful dynamic-rate reference, but less query-conditioned than QACR and less central than LVPruning/CROP/SPIDER. | https://arxiv.org/abs/2501.14204 |

## Related-Only References

| Method | Why Not Direct Baseline |
|---|---|
| TAMP | Important related work, but not the right head-to-head baseline for token-level query-adaptive compute routing. |

## Fair Comparison Protocol

- Backbone: use the same Qwen3.5-VL backbone and the same initial visual tokenization.
- Budgets: compare at matched budgets `0.35 / 0.45 / 0.60` whenever possible.
- Metrics: always report task score, compute ratio/FLOPs, latency, and peak memory.
- Hardware: identical GPU, batch size, and timing protocol.
- Interpretation: do not mix training-free and training-based baselines without explicit labeling.

## Immediate Recommendation

- Main text: `LowRes`, `TokenPruning`, `LVPruning`, `CROP`, `SPIDER`, `QACR`.
- Appendix: `ImageOnly`, `Script`, `FlashVLM`, `DyRate`.
- Related-only: `TAMP` and broader pruning-analysis papers.
