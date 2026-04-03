# Phase 6.2/6.3 Final Summary

Datasets: textvqa, docvqa, mmmu

## Per-Dataset Results

| Method | Dataset | Accuracy | Raw Accuracy | Mean Compute | N |
|---|---:|---:|---:|---:|---:|
| FastV | docvqa | 0.67583 | 0.83511 | 0.45000 | 5349 |
| FastV | mmmu | 0.34222 | 0.34222 | 0.45000 | 900 |
| FastV | textvqa | 0.66887 | 0.69307 | 0.45000 | 5000 |
| ImageOnly@0.45 | docvqa | 0.01963 | 0.01963 | 0.39058 | 5349 |
| ImageOnly@0.45 | mmmu | 0.35111 | 0.35111 | 0.52889 | 900 |
| ImageOnly@0.45 | textvqa | 0.05953 | 0.05973 | 0.40709 | 5000 |
| LVPruning | docvqa | 0.67583 | 0.83511 | 0.45000 | 5349 |
| LVPruning | mmmu | 0.34222 | 0.34222 | 0.45000 | 900 |
| LVPruning | textvqa | 0.66887 | 0.69307 | 0.45000 | 5000 |
| LowRes-9x9 | docvqa | 0.01963 | 0.01944 | 0.41327 | 5349 |
| LowRes-9x9 | mmmu | 0.35444 | 0.35444 | 0.41327 | 900 |
| LowRes-9x9 | textvqa | 0.07293 | 0.07313 | 0.41327 | 5000 |
| Original | docvqa | 0.67583 | 0.83511 | 1.00000 | 5349 |
| Original | mmmu | 0.34222 | 0.34222 | 1.00000 | 900 |
| Original | textvqa | 0.66887 | 0.69307 | 1.00000 | 5000 |
| QACR b0.35 | docvqa | 0.02019 | 0.02019 | 0.34996 | 5349 |
| QACR b0.35 | mmmu | 0.32333 | 0.32333 | 0.34948 | 900 |
| QACR b0.35 | textvqa | 0.07633 | 0.07633 | 0.34981 | 5000 |
| QACR b0.45 | docvqa | 0.11984 | 0.12227 | 0.44102 | 5349 |
| QACR b0.45 | mmmu | 0.30444 | 0.30444 | 0.40418 | 900 |
| QACR b0.45 | textvqa | 0.26467 | 0.25640 | 0.44815 | 5000 |
| QACR b0.60 | docvqa | 0.02000 | 0.02000 | 0.59283 | 5349 |
| QACR b0.60 | mmmu | 0.34222 | 0.34222 | 0.54914 | 900 |
| QACR b0.60 | textvqa | 0.06927 | 0.06947 | 0.59904 | 5000 |
| TokenPruning@0.45 | docvqa | 0.01608 | 0.01608 | 0.45000 | 5349 |
| TokenPruning@0.45 | mmmu | 0.31778 | 0.31778 | 0.45000 | 900 |
| TokenPruning@0.45 | textvqa | 0.07140 | 0.07160 | 0.45000 | 5000 |

## Macro (Available Coverage)

| Method | Coverage | Macro Acc | Macro Raw Acc | Macro Compute |
|---|---:|---:|---:|---:|
| FastV | 3/3 | 0.56231 | 0.62347 | 0.45000 |
| ImageOnly@0.45 | 3/3 | 0.14342 | 0.14349 | 0.44218 |
| LVPruning | 3/3 | 0.56231 | 0.62347 | 0.45000 |
| LowRes-9x9 | 3/3 | 0.14900 | 0.14901 | 0.41327 |
| Original | 3/3 | 0.56231 | 0.62347 | 1.00000 |
| QACR b0.35 | 3/3 | 0.13995 | 0.13995 | 0.34975 |
| QACR b0.45 | 3/3 | 0.22965 | 0.22770 | 0.43112 |
| QACR b0.60 | 3/3 | 0.14383 | 0.14390 | 0.58034 |
| TokenPruning@0.45 | 3/3 | 0.13509 | 0.13515 | 0.45000 |
