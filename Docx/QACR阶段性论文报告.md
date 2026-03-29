# Query-Adaptive Compute Routing (QACR) 阶段性论文报告

## 0. 执行摘要

本报告基于当前已经完成的 `Phase 0 ~ Phase 3` 代码与实验结果，评估 QACR 的核心创新点是否已经被验证，并判断该工作距离 CCF-A 级别会议投稿还有多远。

当前结论可以概括为一句话：

> **QACR 的核心想法已经在原型层面得到验证，但证据强度还不足以直接支撑 CCF-A 投稿；更准确地说，它已经具备“有潜力的研究雏形”，但还不是“顶会可投的成熟论文”。**

支持这一判断的正面证据是：

- QACR 已完成端到端可训练原型，且 Router 额外计算开销仅为 `0.008070%`；
- 不同 Query 会触发不同的 token 深度分配热图，说明模型确实学到了 query-sensitive 的计算重心迁移；
- 在近预算条件下，QACR 相比传统 token pruning 基线显著降低了 proxy task loss；
- Budget 变化能够系统性影响 skip/shallow/deep 路径比例，说明“预算约束下的算力分配”这一主张具备实证基础。

限制当前投稿潜力的关键问题是：

- 评测仍以 proxy 为主，尚未进入官方 benchmark 的大规模主结果表；
- corner case 分析中 `6/6` 错误被触发，关键 token 保护问题非常突出；
- soft-to-hard gap 虽然可控，但 hard routing 偏置和 collapse 风险仍未真正消除；
- 当前硬件收益没有闭环，近预算下的实际延迟并未优于低分辨率强基线。

因此，本工作的当前定位更适合：

- 作为阶段性研究报告；
- 作为论文初稿/内部汇报稿；
- 作为后续冲击 CCF-A 的“主线已成立、证据仍需补强”的候选项目。

而不适合直接定位为：

- 已达到 CCF-A 稳定录用水位的成熟投稿版本。

---

## 1. 研究问题与论文主张

### 1.1 研究问题

多模态大模型在高分辨率图像输入下会产生大量视觉 token，导致计算量、推理延迟和显存占用快速增长。传统高效化方法通常聚焦于：

- token pruning；
- token merging；
- 固定低分辨率输入；
- 启发式区域筛选。

这些方法的共性是把问题建模为“保留哪些 token，丢弃哪些 token”。QACR 提出的不同视角是：

> 在给定计算预算下，不同视觉 token 不应获得统一计算深度，而应根据 query 与图像内容动态分配计算路径。

### 1.2 论文的一句话创新点

> **QACR 将高效视觉建模从 token selection 转化为 budget-constrained compute allocation：在全局预算约束下，基于文本 query 为视觉 token 动态分配 `skip / shallow / deep` 三种计算路径。**

### 1.3 当前版本的四个核心贡献表述

1. 提出 `query-adaptive compute routing`，让视觉 token 的计算深度由 query 条件决定，而非固定统一执行。
2. 引入 `budget-constrained optimization`，通过显式计算正则来学习预算内的算力分配。
3. 使用 `soft-to-hard routing`，兼顾训练可导性与推理时的离散执行需求。
4. 采用 `single-axis depth routing` 的最小实现，先验证“算力分配”本身是否成立，再决定是否升级到多轴路由。

---

## 2. 当前方法概述

当前系统的主流程为：

```text
Image + Query
    -> Coarse Visual Tokens
    -> Query-Adaptive Router
    -> Multi-Path Depth Execution (skip / shallow / deep)
    -> Fusion
    -> Proxy Task / MLLM-side Evaluation
```

其中：

- `DepthOnlyRouter` 使用轻量 MLP，对每个 coarse visual token 输出三路 logits；
- `DepthMultiPathExecutor` 为三条路径分别执行恒等、浅层和深层处理；
- 训练期采用 Gumbel-Softmax 的 soft routing；
- 推理期切换为 argmax hard routing；
- 预算损失将三种路径映射为不同计算代价（`skip=0.0`, `shallow=0.35`, `deep=1.0`）。

这套设计的目标不是直接击败所有现有方法，而是先验证如下科学假设是否成立：

- H1：不同 query 下，token 的计算需求不同；
- H2：query-conditioned routing 优于 image-only routing；
- H3：在相近预算下，compute routing 优于传统 pruning；
- H4：soft-to-hard routing 可以兼顾训练稳定性与可部署性；
- H5：即便只做 depth-only routing，也能观察到可测量收益。

---

## 3. 当前实验设置与完成情况

### 3.1 已完成阶段

当前已完成：

- `Phase 0`：MVP 搭建与稳定性验证；
- `Phase 1`：核心假设验证；
- `Phase 2`：与传统 pruning / low-resolution 的公平对比；
- `Phase 3`：可视化、机制分析、消融、corner case 与硬件 profiling。

### 3.2 当前统一设置

- 基座模型：`Qwen3.5-VL` 本地权重 `Model/Qwen35-08B`
- Coarse grid：`14 x 14`
- 视觉 token 数：`196`
- 训练环境：`conda env qacr`
- 主要评估指标：
  - `ProxyTaskLoss`
  - `expected_compute / compute ratio`
  - `latency`
  - `hard-soft gap`
  - `corner case miss rate`
  - `peak memory`

需要强调的是：

> 当前“性能”指标主要还是 proxy 任务损失与代理 benchmark 准确率，而不是完整官方 benchmark 的标准主结果。

这点对论文定位非常关键。

---

## 4. 核心创新点是否已经被验证

### 4.1 总结表

| 假设/创新点 | 当前证据 | 关键数值 | 当前判断 |
|---|---|---:|---|
| H1：不同 Query 需要不同计算重心 | Query 热图与分配比例分析 | `pairwise_l1_mean=0.117299`，`pairwise_l1_max=0.193636` | 已初步验证 |
| H2：Query-conditioned 优于 Image-only | 2.3 统一对比表 | QACR `loss=0.071846` vs Image-only `0.150486` | 已初步验证 |
| H3：Compute routing 优于 pruning | 2.3 近预算对比 | QACR `0.071846` vs TokenPruning `0.205060` | 已较强验证 |
| H4：Soft-to-hard 可训练且可部署 | gap 分析 + 稳定性冒烟 | `mean_gap=0.007583`，`max_gap=0.078344`，但 `final_hard_ratio_shallow=1.0` | 部分验证 |
| H5：Depth-only routing 本身有价值 | Budget sweep + 对比评测 + 可视化 | `compute=0.431846` 下 loss 接近 low-res 且优于 pruning | 已初步验证 |

### 4.2 对 H1 的验证：不同 Query 是否真的改变了计算分配？

答案是：**是的，已经有直接数值证据。**

在同一张图像、不同 query 的热图分析中：

- `left_focus` 的 `mean_deep_prob = 0.405356`
- `right_focus` 的 `mean_deep_prob = 0.393757`
- `center_focus` 的 `mean_deep_prob = 0.270868`
- `bottom_text` 的 `mean_deep_prob = 0.211720`
- query 间热图差异 `pairwise_l1_mean = 0.117299`
- 最大差异 `pairwise_l1_max = 0.193636`

这说明模型并没有学成“固定模板式路由”，而是在 query 变化时改变计算重心。对于论文来说，这一点非常重要，因为它直接支撑了“query-adaptive”而不是“image-only heuristic”的主张。

结论：

> **创新点 1（query-conditioned compute routing）已经得到可视化与数值层面的初步证明。**

### 4.3 对 H2 的验证：Query conditioning 是否优于 Image-only routing？

答案也是：**有初步支持，而且方向明确。**

在统一对比实验中：

| Method | ComputeRatio | ProxyTaskLoss | Latency(ms) |
|---|---:|---:|---:|
| ImageOnlyRouting | 0.576133 | 0.150486 | 1.536895 |
| QACR-QueryAdaptive | 0.431846 | 0.071846 | 2.983424 |

这组结果说明，在更低计算比例下，QACR 的 proxy loss 反而更低，说明 query 信息的确在帮助模型更有效地分配算力。

但这里也要诚实指出：

- 当前 latency 并未优于 image-only baseline；
- loss 指标仍是 proxy task loss，而不是大规模真实 benchmark accuracy。

结论：

> **创新点 2（query conditioning 的必要性）已经得到方向性验证，但还需要官方 benchmark 主结果来形成强结论。**

### 4.4 对 H3 的验证：Compute routing 是否优于传统 pruning？

这是当前证据最强的一项。

在近预算对比中：

| Method | ComputeRatio | ProxyTaskLoss | Latency(ms) |
|---|---:|---:|---:|
| TokenPruning-keep0.45 | 0.448980 | 0.205060 | 1.383043 |
| QACR-QueryAdaptive | 0.431846 | 0.071846 | 2.983424 |

以及相对 QACR 的差值：

- `delta_compute = +0.017134`（Pruning 算得更多）
- `delta_task_loss = +0.133214`（Pruning 损失明显更高）

这说明：

- QACR 的优势不是简单地“保留更多 token”；
- 在更低或相近预算下，QACR 的任务表现显著好于启发式 keep/drop；
- 因而“compute allocation 优于简单 token elimination”的论文主线已经有了比较清晰的支撑。

结论：

> **创新点 3（routing 优于 pruning）已经得到较强的原型证据，是当前最有希望写进论文主结果的部分。**

### 4.5 对 H4 的验证：Soft-to-Hard routing 是否成立？

这里的答案是：**部分成立，但还不够强。**

soft-hard gap 分析结果如下：

- `final_soft_eval_loss = 0.019988`
- `final_hard_eval_loss = 0.015271`
- `final_hard_minus_soft_gap = -0.004717`
- `mean_hard_minus_soft_gap = 0.007583`
- `max_hard_minus_soft_gap = 0.078344`
- `finite_gradients_all_steps = True`
- `final_hard_ratio_shallow = 1.000000`

可以看到：

- 从训练稳定性角度，soft routing 确实可导且稳定；
- 从 gap 数值角度，平均 gap 不大；
- 但从硬路由行为角度，最终仍然出现了单一路径偏置，这会被 reviewer 视为一个明显风险。

再结合消融实验：

- `anneal` 的 `hard_collapse_ratio = 0.760204`
- `fixed_low` 的 `hard_collapse_ratio = 1.000000`

这说明温度退火确实有效，但“硬路由不塌缩”还没有真正被完全解决。

结论：

> **创新点 4（soft-to-hard training strategy）已经被证明“可训练”，但还没有被证明“完全可部署且稳定优雅”。**

### 4.6 对 H5 的验证：只做 Depth-only 是否已经有价值？

答案是：**有价值，但价值主要体现在“研究主线成立”，而不是“最终系统性能封顶”。**

Budget sweep 结果显示：

| Budget | expected_compute | total_loss | soft_deep | latency_ms |
|---:|---:|---:|---:|---:|
| 0.35 | 0.413488 | 0.184522 | 0.291211 | 2.924792 |
| 0.45 | 0.445030 | 0.072546 | 0.311260 | 2.923490 |
| 0.60 | 0.531974 | 0.083678 | 0.379657 | 2.867716 |

同时，Budget vs 路径比例统计显示：

- `deep_ratio` 从 `0.238689` 增加到 `0.421315`
- `skip_ratio` 总体下降

这说明当前单轴 depth routing 已经能让模型响应预算变化，并改变算力分配行为。也就是说，至少在“是否值得继续研究”这个层面，答案是肯定的。

结论：

> **创新点 5（single-axis depth routing first）是成立的，且这一策略帮助我们把研究问题清晰化了。**

---

## 5. 当前主结果可以如何讲故事

### 5.1 主结果表

当前最适合作为论文核心大表雏形的结果如下：

| Method | ComputeRatio | ProxyTaskLoss | Latency(ms) |
|---|---:|---:|---:|
| UpperBound-Deep | 1.000000 | 0.005068 | 1.354893 |
| LowRes-9x9 | 0.413265 | 0.074067 | 1.031226 |
| TokenPruning-keep0.45 | 0.448980 | 0.205060 | 1.383043 |
| ImageOnlyRouting | 0.576133 | 0.150486 | 1.536895 |
| QACR-QueryAdaptive | 0.431846 | 0.071846 | 2.983424 |

这张表说明了三件事：

1. QACR 明显优于传统 pruning；
2. QACR 明显优于 image-only routing；
3. QACR 当前与 low-resolution baseline 的性能非常接近，但 latency 还没有占优。

从论文写作角度看，这意味着：

- **“优于 pruning”可以讲得比较硬；**
- **“优于 low-resolution”现在还讲不硬；**
- **“带来实际端到端加速”现在不能直接讲。**

### 5.2 机制证据表

| 机制问题 | 关键数值 | 解释 |
|---|---:|---|
| Query 是否改变热图？ | `pairwise_l1_max=0.193636` | 是，说明路由分配与 query 强相关 |
| Budget 是否改变路径比例？ | `deep_ratio: 0.238689 -> 0.421315` | 是，说明预算约束生效 |
| Router 是否轻量？ | `overhead_ratio=0.008070%` | 是，远低于 5% 限制 |
| Soft-hard gap 是否可控？ | `mean_gap=0.007583` | 基本可控，但仍有硬路由偏置 |
| 关键 token 是否被保护？ | `num_flagged_errors=6/6` | 没有，当前最严重短板 |

这意味着：

> 方法的“机制正确性”比“最终系统效果”更先被验证出来了。

这对于一篇顶会论文来说是好事，但还不够，因为 reviewer 仍然会追问：机制成立之后，最终收益是否足够强、是否能跨数据集复现、是否真的能带来硬件收益。

---

## 6. 当前工作的主要不足与 reviewer 风险

### 6.1 风险一：评测规模不足

`Phase 2.4` 当前使用的是 `12` 条代理样本，虽然覆盖了 `VQAv2/GQA/POPE/TextVQA/DocVQA/MMBench/MMMU` 的口径，但依然不能替代官方 benchmark。

这会带来两个问题：

- 无法形成足够可信的主结果表；
- 很容易被 reviewer 质疑为“只在 toy setup 上成立”。

### 6.2 风险二：关键 token 错配问题严重

corner case 分析结果非常直接：

- `num_cases = 6`
- `num_flagged_errors = 6`
- 所有 case 的 `miss_rate_key_tokens = 1.0`

这说明当前路由器虽然学会了“动态变化”，但还没有学会“可靠地保住关键 token”。如果这个问题不修掉，论文的故事就会出现结构性弱点：

> 模型会分配算力，但分得还不够准。

### 6.3 风险三：硬件收益尚未闭环

硬件 profiling 显示：

- 当前 dense executor 的 `dense_latency_ms` 约为 `0.94 ~ 0.97 ms`
- 稀疏执行仿真未表现出稳定的单调加速
- 近预算主表中，QACR latency 为 `2.983424 ms`，明显慢于 low-res 的 `1.031226 ms`

因此当前不能写：

- “QACR 已经显著降低真实推理延迟”

最多只能写：

- “QACR 在机制上具备条件执行潜力，但真实硬件收益仍依赖更高效的稀疏执行实现与底层算子支持”

### 6.4 风险四：Soft-to-Hard 的最终闭环仍不完整

虽然训练稳定，但 hard routing 偏置仍然存在：

- `final_hard_ratio_shallow = 1.0`
- 多组 ablation 中 `hard_collapse_ratio` 仍偏高

这会让 reviewer 质疑：

- 训练时学到的是不是“软分配”；
- 推理时是否真的在执行一个合理的离散策略；
- 论文是否存在“train-test mismatch”。

---

## 7. 当前是否具有 CCF-A 会议投稿潜力

### 7.1 结论先行

答案是：

> **有潜力，但目前还不具备直接投稿 CCF-A 的完成度。**

这里的“有潜力”是指：

- 研究问题有价值；
- 创新主线清晰；
- 与 pruning 的区分度明确；
- 已经出现了支持性数值结果；
- 如果把后续关键缺口补齐，确实有机会成长为强论文。

这里的“还不具备直接投稿完成度”是指：

- 主结果仍缺少官方 benchmark 的大样本说服力；
- 与 low-resolution 强基线相比，优势尚不稳定；
- 硬件效率故事没有闭环；
- corner case 暴露出关键失败模式；
- Phase 4 尚未经过必要的 go/no-go 验证。

### 7.2 以顶会标准做一个主观评分

下面的评分不是客观标准，而是基于当前材料的内部评估：

| 维度 | 当前评分（10分） | 判断 |
|---|---:|---|
| 问题价值 | 8.5 | 高，MLLM 高效推理是明确热点 |
| 创新清晰度 | 7.5 | “compute allocation vs pruning” 叙事清楚 |
| 方法完整性 | 6.5 | 主链路完整，但多处仍是 prototype |
| 机制证据 | 7.0 | 热图、预算、消融都在支持主张 |
| 主结果说服力 | 4.5 | 缺少官方 benchmark 和更强主表 |
| 硬件/系统闭环 | 3.5 | 目前尚未成立 |
| 可复现性 | 7.0 | 代码与脚本比较完整 |
| 综合投稿成熟度 | 5.5 | 有潜力，但不到 CCF-A ready |

### 7.3 如果现在投稿，最可能的 reviewer 评价

较可能收到的正面评价：

- 研究问题有意义；
- “compute allocation” 与 pruning 的切分有新意；
- 机制分析做得比较完整；
- depth-only 的最小化设计是合理的。

较可能收到的负面评价：

- benchmark 不够标准或规模不足；
- 真正的 latency gain 没有建立；
- hard routing 与 corner cases 暴露了方法不稳定性；
- 与强 baseline（尤其 low-resolution）相比，收益还不够压倒性。

因此，当前版本更像是：

- 一篇“非常值得继续做”的稿子；
- 而不是“一篇已经可以放心投 CCF-A”的稿子。

---

## 8. 进入 CCF-A 水位前还缺什么

如果目标是把这项工作真正推进到 CCF-A 级别，至少还需要补三类证据。

### 8.1 一类证据：官方 benchmark 主结果

至少需要在真实 benchmark 上补齐统一主表，重点包括：

- 通用视觉问答；
- OCR / 文档理解；
- 综合评测。

目标不是简单“跑一下”，而是要形成如下结论：

> 在相近计算预算下，QACR 稳定优于 pruning，并在若干重要任务上优于 low-resolution baseline。

### 8.2 二类证据：关键错误模式被修复

建议把当前 Phase 3.3 的结果作为明确优化目标：

- 将 `num_flagged_errors` 从 `6/6` 降到 `<= 2/6`
- 将关键 token `miss_rate` 显著压低
- 在复杂 query 下保持更高的 key/non-key separation

只有这样，Reviewer 才会相信 routing 学到的不是“会变”，而是“会准”。

### 8.3 三类证据：真实硬件收益或至少不吃亏

最终至少要达到下面两类结果中的一类：

1. 在相近精度下，真实 latency 低于 depth-only / low-resolution baseline；
2. 即便 latency 不占优，也能在关键任务上提供显著更好的效果，从而形成“accuracy wins justify moderate overhead”的论文叙事。

目前连第二种都还不够强，所以 Phase 4 之前的前置验证很关键。

---

## 9. 建议的下一步研究路线

基于当前材料，最合理的推进顺序不是直接全面开启 Phase 4，而是：

1. 先完成 `Phase 3.5` 的 go/no-go 前置实验；
2. 优先修复 key token 错配与 hard routing collapse；
3. 在真实 benchmark 子集上建立新的主表；
4. 只有当前置实验明确显示“高分辨率重编码或 attention routing 确实带来额外收益”时，再正式进入完整 Phase 4。

这一路线的好处是：

- 不会过早把系统做重；
- 不会在尚未解决主线缺口时引入更多变量；
- 可以把论文故事维持为一个非常清晰的主线：

> 先证明 depth-only compute routing 成立，再证明额外轴是否值得加入。

---

## 10. 最终结论

综合当前全部实验，可以给出一个明确判断：

> **QACR 的论文创新点已经得到“阶段性验证”，尤其是 query-adaptive compute allocation、budget-constrained routing、以及相对 pruning 的优势已经有具体数值支持。**

但同时也必须明确：

> **这些证据还不足以说明该工作已经达到 CCF-A 稳定投稿水位。**

当前最准确的定位是：

- 不是一个失败的 idea；
- 不是一个仅停留在概念层面的空方案；
- 而是一个已经被 prototype 证明“值得继续打磨”的论文雏形。

如果后续能补齐以下三点：

- 官方 benchmark 主结果；
- key token 错配修复；
- 真正可信的硬件效率闭环；

那么这项工作是有希望进入 CCF-A 讨论区间的。

在此之前，更稳妥的判断应该是：

> **目前具有 CCF-A 潜力，但尚未达到 CCF-A ready。**

---

## 附：当前最值得在论文中直接引用的数值

| 类别 | 关键数值 |
|---|---|
| Router 开销 | `overhead_ratio = 0.008070%` |
| MVP 稳定性 | `finite_gradients_all_steps = True`, `collapse_detected = False` |
| Query 热图差异 | `pairwise_l1_mean = 0.117299`, `pairwise_l1_max = 0.193636` |
| 预算响应 | `deep_ratio: 0.238689 -> 0.421315` |
| QACR vs Pruning | `0.071846 vs 0.205060`（ProxyTaskLoss） |
| QACR vs Image-only | `0.071846 vs 0.150486`（ProxyTaskLoss） |
| Soft-hard gap | `mean_hard_minus_soft_gap = 0.007583` |
| Corner case 风险 | `num_flagged_errors = 6/6` |
| 硬件现状 | QACR `latency = 2.983424 ms`，LowRes `1.031226 ms` |

以上数值可以直接作为后续论文初稿中的“当前证据池”使用。
