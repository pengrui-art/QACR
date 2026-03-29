# Phase 3.11 Official Benchmark Subset Results

## Macro Summary

| Method | Budget | Samples | Avg Accuracy | Avg Compute | Avg Latency (ms) | Avg Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| LowRes | 0.35 | 7 | 0.2857 | 0.3265 | 852.5321 | 14.79 |
| LowRes | 0.45 | 7 | 0.2857 | 0.4133 | 745.1739 | 17.25 |
| LowRes | 0.60 | 7 | 0.2857 | 0.6173 | 747.8571 | 17.25 |
| QACR-DepthOnly | 0.35 | 7 | 0.2857 | 0.3493 | 842.0565 | 31.59 |
| QACR-DepthOnly | 0.45 | 7 | 0.2857 | 0.4492 | 845.9270 | 31.59 |
| QACR-DepthOnly | 0.60 | 7 | 0.2857 | 0.5994 | 840.8151 | 31.59 |
| TokenPruning | 0.35 | 7 | 0.2857 | 0.3520 | 795.7983 | 29.55 |
| TokenPruning | 0.45 | 7 | 0.4286 | 0.4490 | 762.5058 | 29.55 |
| TokenPruning | 0.60 | 7 | 0.2857 | 0.6020 | 780.6026 | 29.55 |

## Per-Dataset Summary

| Dataset | Method | Budget | Samples | Avg Accuracy | Avg Compute | Avg Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| DocVQA | LowRes | 0.35 | 1 | 0.0000 | 0.3265 | 950.8767 |
| DocVQA | LowRes | 0.45 | 1 | 0.0000 | 0.4133 | 950.6006 |
| DocVQA | LowRes | 0.60 | 1 | 0.0000 | 0.6173 | 949.6071 |
| DocVQA | QACR-DepthOnly | 0.35 | 1 | 0.0000 | 0.3485 | 1049.4659 |
| DocVQA | QACR-DepthOnly | 0.45 | 1 | 0.0000 | 0.4497 | 1078.9401 |
| DocVQA | QACR-DepthOnly | 0.60 | 1 | 0.0000 | 0.5995 | 1050.3605 |
| DocVQA | TokenPruning | 0.35 | 1 | 0.0000 | 0.3520 | 978.5182 |
| DocVQA | TokenPruning | 0.45 | 1 | 0.0000 | 0.4490 | 990.8907 |
| DocVQA | TokenPruning | 0.60 | 1 | 0.0000 | 0.6020 | 997.7858 |
| GQA | LowRes | 0.35 | 1 | 1.0000 | 0.3265 | 941.2669 |
| GQA | LowRes | 0.45 | 1 | 1.0000 | 0.4133 | 947.7835 |
| GQA | LowRes | 0.60 | 1 | 1.0000 | 0.6173 | 945.7173 |
| GQA | QACR-DepthOnly | 0.35 | 1 | 1.0000 | 0.3495 | 1004.5825 |
| GQA | QACR-DepthOnly | 0.45 | 1 | 1.0000 | 0.4495 | 1043.4786 |
| GQA | QACR-DepthOnly | 0.60 | 1 | 1.0000 | 0.5990 | 1035.7509 |
| GQA | TokenPruning | 0.35 | 1 | 1.0000 | 0.3520 | 955.9409 |
| GQA | TokenPruning | 0.45 | 1 | 1.0000 | 0.4490 | 967.7774 |
| GQA | TokenPruning | 0.60 | 1 | 1.0000 | 0.6020 | 966.9405 |
| MMBench | LowRes | 0.35 | 1 | 1.0000 | 0.3265 | 437.2054 |
| MMBench | LowRes | 0.45 | 1 | 1.0000 | 0.4133 | 431.9856 |
| MMBench | LowRes | 0.60 | 1 | 1.0000 | 0.6173 | 427.0804 |
| MMBench | QACR-DepthOnly | 0.35 | 1 | 1.0000 | 0.3495 | 528.0205 |
| MMBench | QACR-DepthOnly | 0.45 | 1 | 1.0000 | 0.4492 | 517.7934 |
| MMBench | QACR-DepthOnly | 0.60 | 1 | 1.0000 | 0.5987 | 527.0842 |
| MMBench | TokenPruning | 0.35 | 1 | 1.0000 | 0.3520 | 461.8002 |
| MMBench | TokenPruning | 0.45 | 1 | 1.0000 | 0.4490 | 455.2329 |
| MMBench | TokenPruning | 0.60 | 1 | 1.0000 | 0.6020 | 453.1946 |
| MMMU | LowRes | 0.35 | 1 | 0.0000 | 0.3265 | 529.8249 |
| MMMU | LowRes | 0.45 | 1 | 0.0000 | 0.4133 | 529.8417 |
| MMMU | LowRes | 0.60 | 1 | 0.0000 | 0.6173 | 530.2143 |
| MMMU | QACR-DepthOnly | 0.35 | 1 | 0.0000 | 0.3485 | 633.0205 |
| MMMU | QACR-DepthOnly | 0.45 | 1 | 0.0000 | 0.4485 | 634.8136 |
| MMMU | QACR-DepthOnly | 0.60 | 1 | 0.0000 | 0.5992 | 650.1389 |
| MMMU | TokenPruning | 0.35 | 1 | 0.0000 | 0.3520 | 581.4662 |
| MMMU | TokenPruning | 0.45 | 1 | 1.0000 | 0.4490 | 458.4206 |
| MMMU | TokenPruning | 0.60 | 1 | 0.0000 | 0.6020 | 579.1899 |
| POPE | LowRes | 0.35 | 1 | 0.0000 | 0.3265 | 394.1372 |
| POPE | LowRes | 0.45 | 1 | 0.0000 | 0.4133 | 388.7103 |
| POPE | LowRes | 0.60 | 1 | 0.0000 | 0.6173 | 396.6077 |
| POPE | QACR-DepthOnly | 0.35 | 1 | 0.0000 | 0.3500 | 504.2537 |
| POPE | QACR-DepthOnly | 0.45 | 1 | 0.0000 | 0.4485 | 526.1089 |
| POPE | QACR-DepthOnly | 0.60 | 1 | 0.0000 | 0.5995 | 514.3296 |
| POPE | TokenPruning | 0.35 | 1 | 0.0000 | 0.3520 | 429.5388 |
| POPE | TokenPruning | 0.45 | 1 | 0.0000 | 0.4490 | 436.8775 |
| POPE | TokenPruning | 0.60 | 1 | 0.0000 | 0.6020 | 447.9849 |
| TextVQA | LowRes | 0.35 | 1 | 0.0000 | 0.3265 | 991.0836 |
| TextVQA | LowRes | 0.45 | 1 | 0.0000 | 0.4133 | 988.2516 |
| TextVQA | LowRes | 0.60 | 1 | 0.0000 | 0.6173 | 995.4898 |
| TextVQA | QACR-DepthOnly | 0.35 | 1 | 0.0000 | 0.3490 | 1053.5491 |
| TextVQA | QACR-DepthOnly | 0.45 | 1 | 0.0000 | 0.4500 | 1072.2054 |
| TextVQA | QACR-DepthOnly | 0.60 | 1 | 0.0000 | 0.5997 | 1046.7865 |
| TextVQA | TokenPruning | 0.35 | 1 | 0.0000 | 0.3520 | 987.5743 |
| TextVQA | TokenPruning | 0.45 | 1 | 0.0000 | 0.4490 | 1021.9708 |
| TextVQA | TokenPruning | 0.60 | 1 | 0.0000 | 0.6020 | 1003.6663 |
| VQAv2 | LowRes | 0.35 | 1 | 0.0000 | 0.3265 | 1723.3300 |
| VQAv2 | LowRes | 0.45 | 1 | 0.0000 | 0.4133 | 979.0438 |
| VQAv2 | LowRes | 0.60 | 1 | 0.0000 | 0.6173 | 990.2830 |
| VQAv2 | QACR-DepthOnly | 0.35 | 1 | 0.0000 | 0.3500 | 1121.5033 |
| VQAv2 | QACR-DepthOnly | 0.45 | 1 | 0.0000 | 0.4492 | 1048.1490 |
| VQAv2 | QACR-DepthOnly | 0.60 | 1 | 0.0000 | 0.6000 | 1061.2553 |
| VQAv2 | TokenPruning | 0.35 | 1 | 0.0000 | 0.3520 | 1175.7494 |
| VQAv2 | TokenPruning | 0.45 | 1 | 0.0000 | 0.4490 | 1006.3708 |
| VQAv2 | TokenPruning | 0.60 | 1 | 0.0000 | 0.6020 | 1015.4563 |