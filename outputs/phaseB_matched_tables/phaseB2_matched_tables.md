# Phase B.2 Matched Tables (textvqa)

## Notes

- Accuracy comes from the full benchmark summary for `textvqa`.
- Latency / throughput / peak memory come from Phase B.1 unified profiling.
- Per method, the profiling row is selected by: `min_sample_latency`.

## Matched Compute View

| Method | Accuracy | Mean Compute | Profile Batch | Sample Latency (ms) | Throughput (samples/s) | Peak GPU Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| ImageOnly@0.45 | 0.0595 | 0.4071 | 2 | 547.55 | 1.83 | 2024.08 |
| LowRes-9x9 | 0.0729 | 0.4133 | 2 | 267.64 | 3.74 | 1752.89 |
| QACR b0.45 | 0.2647 | 0.4482 | 2 | 294.28 | 3.40 | 2027.42 |
| TokenPruning@0.45 | 0.0714 | 0.4500 | 2 | 264.15 | 3.79 | 2023.51 |
| Original | 0.6689 | 1.0000 | 2 | 425.38 | 2.35 | 2003.90 |

## Matched Latency View

| Method | Accuracy | Sample Latency (ms) | Throughput (samples/s) | Mean Compute | Peak GPU Memory (MB) |
|---|---:|---:|---:|---:|---:|
| TokenPruning@0.45 | 0.0714 | 264.15 | 3.79 | 0.4500 | 2023.51 |
| LowRes-9x9 | 0.0729 | 267.64 | 3.74 | 0.4133 | 1752.89 |
| QACR b0.45 | 0.2647 | 294.28 | 3.40 | 0.4482 | 2027.42 |
| Original | 0.6689 | 425.38 | 2.35 | 1.0000 | 2003.90 |
| ImageOnly@0.45 | 0.0595 | 547.55 | 1.83 | 0.4071 | 2024.08 |

## Key Comparisons

- `QACR b0.45` vs `Original`: sample latency improves by `30.8%`, throughput improves by `44.5%`, while compute drops from `1.0000` to `0.4482`.
- `QACR b0.45` vs `TokenPruning@0.45`: QACR is slower (`294.28` vs `264.15 ms/sample`) but much more accurate (`0.2647` vs `0.0714`, gap `+0.1933`).
- `QACR b0.45` vs `LowRes-9x9`: QACR is slightly slower (`294.28` vs `267.64 ms/sample`) but far more accurate (`0.2647` vs `0.0729`, gap `+0.1917`).
