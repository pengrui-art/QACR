# QACR Phase 3.9 外部强基线矩阵与取舍说明

## 1. 文档目的

本文档用于完成 `任务 3.9：最接近相关工作的统一复现与公平对齐` 中最关键的三项交付：

1. 明确哪些外部工作与 QACR 最接近。
2. 明确哪些 baseline 必须进主表，哪些更适合放附录。
3. 把后续实验的公平比较协议固定下来，避免边做边改口径。

当前时间基准：`2026-03-29`。

---

## 2. 为什么 3.9 要先做

QACR 当前最容易被 reviewer 质疑的点，不是“有没有 idea”，而是：

> 这是不是只是另一种 query-guided pruning / region compression / layer skipping？

因此，`3.9` 的任务不是立刻跑出所有结果，而是先把最接近的对手定义清楚，并把主表候选锁死。否则后面 `3.10 ~ 3.12` 即使跑了很多实验，也可能因为比较对象选得不对而失去说服力。

---

## 3. 最接近 QACR 的外部工作分组

### 3.1 Query-guided token pruning 家族

这条线和 QACR 的重合度最高，因为它们都已经不满足于 image-only 的静态压缩，而开始显式利用 query / text 信息。

优先代表工作：

- `LVPruning`：使用语言与视觉 token 的交互来决定哪些视觉 token 被剪掉。
  - 来源：NAACL Findings 2025
  - 链接：<https://aclanthology.org/2025.findings-naacl.242/>
- `Script`：graph-structured pruning + query-conditioned semantic pruning，强调 training-free。
  - 来源：TMLR 2025
  - 链接：<https://openreview.net/forum?id=F6xKzbgcHq>
- `FlashVLM`：text-guided visual token selection，并加入 diversity-preserving background selection。
  - 来源：arXiv 2025-12-23
  - 链接：<https://arxiv.org/abs/2512.20561>

这类方法对 QACR 的压力最大，因为 reviewer 很容易认为：

> “你们也是 query-guided，只不过不是 keep/drop，而是多分支深度分配。”

所以这组 baseline 至少要保留一个强代表放主表。

### 3.2 Query-relevant region compression 家族

这类方法比 token-level pruning 更接近“按区域分配资源”的叙事，因此也必须正面比较。

优先代表工作：

- `CROP`：先定位 query 相关区域，再做区域级压缩与 early-layer pruning。
  - 来源：EMNLP 2025
  - 链接：<https://aclanthology.org/2025.emnlp-main.492/>

这类方法虽然不一定像 QACR 一样走多路径深度 routing，但它已经在回答一个非常接近的问题：

> query 相关区域是否应该获得不同于背景区域的计算保留策略？

因此 `CROP` 必须进主表。

### 3.3 Compute-side skipping / prune+skip 家族

这类方法是 QACR 在“算力分配”叙事上的最危险对手，因为它们已经不只做 token 删除，而开始把计算冗余也纳入建模。

优先代表工作：

- `SPIDER`：multi-layer semantic token pruning + adaptive sub-layer skipping。
  - 来源：ICLR 2026 submission
  - 链接：<https://openreview.net/forum?id=aGpSK6QH3w>

这类方法最接近 reviewer 对 QACR 的反问：

> “既然别人已经在做 pruning + skipping，为什么你们的 multi-path compute allocation 还算新？”

所以 `SPIDER` 也应该进主表。

---

## 4. 主表、附录、仅 Related Work 的取舍

### 4.1 建议进入主表的 baseline

建议主表固定为：

- `LowRes-9x9`
- `TokenPruning-keep/drop`
- `LVPruning`
- `CROP`
- `SPIDER`
- `QACR-DepthOnly`

原因如下：

- `LowRes`：现实中最强、最难绕开的工程 baseline。
- `TokenPruning`：最基础的 keep/drop 对照。
- `LVPruning`：query-guided pruning 家族代表。
- `CROP`：region compression 家族代表。
- `SPIDER`：compute-side skipping 家族代表。
- `QACR`：核心方法。

这样一张主表已经足够回答 reviewer 最关键的问题：

> QACR 到底是优于传统 pruning、优于 query-guided pruning，还是只是在 region compression / skip 方法旁边换了个名字？

### 4.2 建议放附录的 baseline

建议附录保留：

- `ImageOnlyRouting`
- `Script`
- `FlashVLM`
- `DyRate`

原因如下：

- `ImageOnlyRouting`：解释 query conditioning 的必要性，但不是最强外部对手。
- `Script`：很相关，但和 `LVPruning` 家族重合度较高，更适合作为强化附录。
- `FlashVLM`：很新，且仍是 arXiv under submission，适合放附录做趋势对比。
- `DyRate`：动态压缩率很重要，但 query 相关性不如前几类直接。

### 4.3 只放 Related Work、不要做主对比的工作

以下工作重要，但不是 `3.9` 主对比对象：

- `TAMP`
  - 链接：<https://aclanthology.org/2025.findings-acl.359/>
  - 更偏 layerwise / weight-side pruning，不是最贴合 token-level query-adaptive compute routing 的正面对手。
- `Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?`
  - 链接：<https://aclanthology.org/2025.findings-acl.802/>
  - 这篇更像元分析与评测批判，对论文 framing 很重要，但不属于主实验 baseline。

---

## 5. 公平比较协议（后续必须固定）

后面所有主表实验统一遵守以下协议：

### 5.1 模型与输入

- 同一 backbone：优先 `Qwen3.5-VL`
- 同一输入分辨率
- 同一初始视觉 token 数
- 同一 batch size

### 5.2 预算与指标

- 统一 budget 点：`0.35 / 0.45 / 0.60`
- 必报指标：
  - `Accuracy / task score`
  - `Compute ratio or FLOPs`
  - `Latency`
  - `Peak memory`

### 5.3 报告规则

- 主文至少有两张表：
  - `matched compute`
  - `matched latency`
- 主文至少有两张图：
  - `Accuracy-Compute Pareto`
  - `Accuracy-Latency Pareto`
- training-free 与 training-based baseline 必须显式标注，不能混写成同类方法。

---

## 6. 当前对 QACR 最危险的三类重合

如果站在 reviewer 视角，QACR 当前最危险的“撞车”点是：

1. `LVPruning / Script / FlashVLM`
   - 说明 query-guided token selection 已经不新。
2. `CROP`
   - 说明 query-relevant region compression 已经不新。
3. `SPIDER`
   - 说明把“pruning + compute skipping”结合起来也开始不新。

因此 QACR 还剩下的真正创新表述应该收缩为：

> **budget-constrained, query-conditioned, multi-path compute allocation**

而不是更宽泛的：

> query-aware token selection / efficient visual token processing

---

## 7. 3.9 完成后的直接结论

### 7.1 当前主表 baseline 固定

主表：

- `LowRes`
- `TokenPruning`
- `LVPruning`
- `CROP`
- `SPIDER`
- `QACR`

附录：

- `ImageOnly`
- `Script`
- `FlashVLM`
- `DyRate`

### 7.2 当前最该立即启动的后续任务

`3.9` 完成后，最合理的顺序就是：

1. `3.10` 真实条件执行执行器
2. `3.11` 官方 benchmark 子集主表
3. `3.12` same-image-different-query + key-token 指标

换句话说，`3.9` 已经把“跟谁比、怎么比、哪些进主表”锁清楚了；接下来真正决定 CCF-A 水位的，就是跑出能压住这些 baseline 的证据。

---

## 8. 对 CCF-A 创新点的现实判断

截至当前，QACR 作为 CCF-A 级别论文还缺的，不是“再发明一个更复杂模块”，而是以下四类证据：

1. **证明不是换皮 pruning**
   - 必须用 `LVPruning / CROP / SPIDER` 级别 baseline 正面比较。
2. **证明 multi-path compute allocation 真的更强**
   - 不是只讲热图，而是在 matched budget 下拿出稳定更好的主表。
3. **证明 key-token 保护问题被修复**
   - 否则 reviewer 会认为只是“会变，但不够准”。
4. **证明真实条件执行不是伪命题**
   - 否则 reviewer 会认为这是机制稿，而不是可部署方法。

---

## 9. 对应脚本与产物

为避免后续每次手工整理，已新增统一导出脚本：

- `scripts/run_phase39_baseline_alignment.py`

建议运行：

```bash
cd /data1/pengrui/CCFA/QACR
conda run -n qacr python scripts/run_phase39_baseline_alignment.py
```

输出：

- `outputs/phase39_baseline_alignment.json`
- `outputs/phase39_baseline_alignment.md`

