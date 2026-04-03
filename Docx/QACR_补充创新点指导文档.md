# QACR 补充创新点指导文档

## 1. 文档定位

这份文档服务于当前阶段两个目标：

1. 诚实判断：以现在的结果，`QACR` 是否已经适合直接投稿 `NeurIPS` 主会。
2. 补线指导：如果还不够，下一步最值得补的创新点、实验和论文叙事是什么。

这份文档默认基于 `2026-04-03` 的最新主表结果，作为旧版偏乐观评估的补充与纠偏。

---

## 2. 当前结论

先给直接判断：

> **以目前结果来看，QACR 还不建议直接以 NeurIPS 主会完成态投稿。**

更准确地说：

- 已经具备了明确的研究主线；
- 已经有了一批真实有效的工程证据；
- 但离“NeurIPS 主会会觉得证据足够硬”还差几块关键拼图。

当前更像是：

> **有论文潜力，但还需要补强主结果、系统证据和机制证据。**

---

## 3. 为什么现在还不够

### 3.1 方法主线成立，但主结果还不够强

当前最新正式主表中，`QACR b0.45` 为：

- `TextVQA = 0.26467`
- `DocVQA = 0.11984`
- `MMMU = 0.30444`
- `Macro Acc = 0.22965`
- `Macro Compute = 0.43112`

它已经明显好于近预算的弱效率基线：

- `TokenPruning@0.45`
- `ImageOnly@0.45`
- `LowRes-9x9`

但和同基座的强基线相比，差距仍然非常大：

- `Original / FastV / LVPruning`

这意味着 reviewer 更容易得到这样的第一印象：

> 你们的方法方向有意思，但当前主表还不像“成熟到足以压住质疑”的程度。

### 3.2 当前增益有相当一部分来自评测与后处理

最近几轮 `TextVQA / DocVQA` 的提升，很多来自：

- prompt 分流
- answer extraction 修复
- OCR-aware correction
- 评测口径修正

这些工作都很重要，而且是必须做的；但它们更像：

> inference calibration / evaluation engineering

而不是 reviewer 最想看到的“方法本体变强”。

如果主增益主要来自后处理，reviewer 很容易继续追问：

- 路由器本身到底提升了多少？
- 关键 token 保护到底是不是模型内部机制带来的？
- 如果拿掉后处理，方法本体还剩多少优势？

### 3.3 系统证据还不够硬

对一个效率方向的方法来说，`NeurIPS` 很看重这类证据：

- wall-clock latency
- throughput
- memory
- matched-compute / matched-latency fairness

当前我们已经有 `compute ratio` 的优势，也有近预算下优于 pruning-style 基线的迹象，但真实系统收益还不够扎实。

如果没有这部分，论文会容易被问成：

> 这是一个有趣的 routing 机制，还是一个真正可部署的高效推理方案？

### 3.4 机制解释还需要再前进一步

我们现在已经能较稳地讲：

- `query-adaptive routing` 确实存在；
- `compute allocation` 与 `token pruning` 不同；
- OCR / 文档任务暴露出关键 token 保护问题。

但如果要冲 `NeurIPS`，还需要把这层机制更“量化”一些，而不是只靠故事和个案。

最关键的缺口是：

> **缺少“关键 token 是否被保住”的显式指标与统计证据。**

---

## 4. 现在可以硬讲的创新点

下面这些是当前仍然可以稳讲、而且建议坚持讲的：

### 创新点 1：Query-adaptive compute allocation

QACR 的本质不是删 token，而是在预算约束下，依据 query 为视觉 token 分配不同计算深度。

建议固定表述：

> We study query-conditioned compute allocation rather than query-agnostic token elimination.

### 创新点 2：Skip / Shallow / Deep 三路径执行

这让方法天然区别于纯 keep/drop 方案，也更容易连接到条件计算与 MoE 叙事。

### 创新点 3：Budget-faithful routing

QACR 的一个优点是它不是“效果优先、预算随缘”，而是明确围绕目标预算学习路由。

### 创新点 4：OCR / 文档任务暴露出的关键 token 保护问题

这点现在还不是“主创新”，但已经可以成为一个很有价值的论文观察：

> 对多模态高效推理来说，真正难的不是让路由随 query 改变，而是让它稳定保住 query-critical fine-grained tokens。

如果写得好，这会是一个很好的 discussion / analysis 贡献。

---

## 5. 现在不该当主创新讲的东西

以下内容不要再放在主贡献前排：

- 高分辨率重编码
- 多轴联合 routing
- `min_keep_ratio / min_deep_ratio` 硬保底
- prompt 工程本身
- 后处理规则本身

原因不是它们没用，而是：

1. 没有稳定到可以独立立论。
2. 会稀释主线。
3. 很容易被 reviewer 视为工程补丁堆叠。

更准确的定位应该是：

- prompt / postprocess：实验协议与任务校准
- OCR-aware correction：对失败模式的补救性 inference layer

---

## 6. 最值得补的“创新点”

如果后面还要补创新，不建议再补“更花哨的新模块”，而建议补下面三类更值钱的东西。

### 6.1 补创新 A：Query-conditioned key-token protection

这是当前最值得升格为“方法增强版创新”的方向。

目标不是再写更多后处理，而是把下面这件事做进模型或路由策略里：

> 对 OCR / 文档 / 细粒度实体相关的 query，显式提高 query-critical token 获得 deep 路径的概率。

这条线一旦做成，论文会比现在更完整，因为它能把“失败分析”自然收束回“方法增强”。

比较好的切入方式：

- OCR-aware deep-route prior
- key-token auxiliary loss
- question-type-conditioned protection
- route regularization for text-like regions

### 6.2 补创新 B：Key-token recall / miss-rate 指标

这属于“分析创新”，但很值钱。

建议把它做成正式指标：

- `key-token recall`
- `key-token miss rate`
- `deep-allocation on query-critical tokens`

这样论文就不只是说：

- “我们好像保不住关键 token”

而是能正式说：

- “Pruning-style baseline 在 key-token recall 上明显更差，而 QACR 在该指标上更优”

这会大幅增强论文的说服力。

### 6.3 补创新 C：Matched-latency 的真实系统叙事

如果最终拿不到“精度接近 Original”的结果，那就更需要系统证据来托住故事。

建议补齐：

- 单卡固定环境 wall-clock latency
- throughput
- peak memory
- batch-size sensitivity

只有这样，QACR 才能从“算法上省算”更进一步变成“部署上有意义”。

---

## 7. NeurIPS 之前最值得补的实验

下面这些实验按优先级排序。

### 第一层：必须补

1. `QACR vs TokenPruning / ImageOnly / LowRes` 的同预算主表与 Pareto 图
2. `same image, different query` 的 routing 热图与路径比例可视化
3. `TextVQA / DocVQA` 上的 key-token failure analysis
4. 统一硬件下的 wall-clock latency / throughput / memory 表

### 第二层：强烈建议补

1. 第二个基座模型上的复现
2. 不同 budget 下的趋势图
3. soft-to-hard gap 与 route collapse 的正式消融
4. OCR / 文档任务上的 question-type breakdown

### 第三层：有余力再补

1. 更复杂的 key-token protection 机制
2. 更成熟的 runtime / kernel 支持
3. 更多跨数据集泛化展示

---

## 8. 论文应该怎么讲，才更稳

### 8.1 当前最稳的论文主叙事

建议把论文主叙事固定成：

> We argue that efficient MLLM inference should not only decide which tokens to keep, but also decide how much computation each token deserves under a query-conditioned budget.

然后用实验去支持三层结论：

1. 机制层：QACR 确实会随 query 改变算力分配。
2. 对比层：在 matched compute 下，QACR 明显优于 pruning-style / image-only / low-resolution 这类信息丢弃式基线。
3. 分析层：当前主要瓶颈是 key-token protection，而不是 query adaptivity 本身不存在。

### 8.2 当前不要硬讲的东西

不要写成：

- QACR 已经全面优于所有高效基线
- QACR 已经实现稳定端到端加速
- OCR / 文档问题已经被彻底解决
- prompt / 后处理本身就是主要创新

这些说法都会把论文暴露在最危险的位置。

### 8.3 更安全的替代表述

建议写成：

- QACR is particularly advantageous over information-dropping baselines under matched compute.
- QACR validates the mechanism of query-conditioned compute allocation.
- Reliable key-token protection remains the main open bottleneck.
- Current post-processing improvements are used to ensure fair evaluation, rather than being claimed as the core method contribution.

---

## 9. 当前阶段的诚实判断

如果今天必须做一个判断，我会这样写：

### 现在已经具备的条件

- 有清晰且不俗套的方法问题定义
- 有同基座近预算对比优势
- 有全量 benchmark 主表
- 有错误分析与 failure story
- 有可以继续延展的主线，而不是纯工程死胡同

### 现在仍然不足的条件

- 主结果还不够强，尤其 `TextVQA / DocVQA`
- 方法本体收益与后处理收益还没完全拆清
- latency / throughput 证据不够硬
- 还缺一个能强力支撑论文的“关键指标”或“关键图”

### 结论

> **目前更像“NeurIPS 潜力项目”，而不是“已经可以放心投的 NeurIPS 完成态项目”。**

如果时间允许，建议先补完本文件第 6 节和第 7 节里的内容，再决定是否以主会强投。

---

## 10. 接下来两周最推荐的推进顺序

1. 锁定当前主结果与论文图，不再反复改小后处理。
2. 补 `QACR vs TokenPruning / ImageOnly / LowRes` 的论文化图表和结论。
3. 做 `key-token recall / miss-rate` 指标与案例页。
4. 跑一组真实 latency / throughput / memory profiling。
5. 如果还有余力，再做模型侧的 `query-conditioned key-token protection`。

---

## 11. 一句话总结

QACR 现在最值得继续做的，不是再堆更多 prompt 和后处理，而是把下面这条线补硬：

> **从“query-adaptive routing 是成立的”走到“QACR 能更稳定地保住 query-critical tokens，并把省下来的计算真正转化成可信的效率收益”。**
