# Phase B.1 Unified Profiling

## Summary Table

| Method | Batch | Mean Compute | Batch Latency (ms) | Sample Latency (ms) | Throughput (samples/s) | Peak GPU Memory (MB) | Delta Peak (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 1 | 1.0000 | 682.59 | 682.59 | 1.47 | 1820.81 | 166.69 |
| Original | 2 | 1.0000 | 850.76 | 425.38 | 2.35 | 2003.90 | 330.13 |
| QACR b0.45 | 1 | 0.4498 | 561.92 | 561.92 | 1.78 | 1836.94 | 166.55 |
| QACR b0.45 | 2 | 0.4499 | 588.56 | 294.28 | 3.40 | 2027.42 | 335.19 |
| TokenPruning@0.45 | 1 | 0.4500 | 465.32 | 465.32 | 2.15 | 1834.64 | 164.99 |
| TokenPruning@0.45 | 2 | 0.4500 | 528.30 | 264.15 | 3.79 | 2023.51 | 334.83 |
| ImageOnly@0.45 | 1 | 0.3966 | 661.10 | 661.10 | 1.51 | 1834.98 | 163.76 |
| ImageOnly@0.45 | 2 | 0.3625 | 1095.11 | 547.55 | 1.83 | 2024.08 | 334.83 |
| LowRes-9x9 | 1 | 0.4133 | 450.06 | 450.06 | 2.22 | 1701.23 | 47.10 |
| LowRes-9x9 | 2 | 0.4133 | 535.28 | 267.64 | 3.74 | 1752.89 | 97.25 |

## Batch-Size Sensitivity

### Original

- `bs=1`: latency `682.59 ms/batch`, throughput `1.47 samples/s`, peak memory `1820.81 MB`.
- `bs=2`: latency `850.76 ms/batch`, throughput `2.35 samples/s`, peak memory `2003.90 MB`.

### QACR b0.45

- `bs=1`: latency `561.92 ms/batch`, throughput `1.78 samples/s`, peak memory `1836.94 MB`.
- `bs=2`: latency `588.56 ms/batch`, throughput `3.40 samples/s`, peak memory `2027.42 MB`.

### TokenPruning@0.45

- `bs=1`: latency `465.32 ms/batch`, throughput `2.15 samples/s`, peak memory `1834.64 MB`.
- `bs=2`: latency `528.30 ms/batch`, throughput `3.79 samples/s`, peak memory `2023.51 MB`.

### ImageOnly@0.45

- `bs=1`: latency `661.10 ms/batch`, throughput `1.51 samples/s`, peak memory `1834.98 MB`.
- `bs=2`: latency `1095.11 ms/batch`, throughput `1.83 samples/s`, peak memory `2024.08 MB`.

### LowRes-9x9

- `bs=1`: latency `450.06 ms/batch`, throughput `2.22 samples/s`, peak memory `1701.23 MB`.
- `bs=2`: latency `535.28 ms/batch`, throughput `3.74 samples/s`, peak memory `1752.89 MB`.
