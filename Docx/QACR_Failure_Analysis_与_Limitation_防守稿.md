# QACR Failure Analysis 与 Limitation 防守稿

## 1. 文档定位

本文档用于完成论文计划表中的 `Phase 3.5：Failure Analysis 与 Limitation 防守稿`，目标不是放大负面结果，而是把当前风险写成：

- 真实且可验证；
- 不回避；
- 但不会摧毁论文主线；
- 并且能自然导向下一阶段实验。

建议使用方式：

- 正文中保留一小节 `Failure Analysis / Limitations`；
- 附录中放更完整的错误样例与 profiling；
- rebuttal 或答辩时可直接复用本文中的“安全表述”。

---

## 2. 当前最需要主动承认的四个问题

### 2.1 Low-resolution 是强基线，但它没有否定 QACR 的研究问题

当前近预算主表中：

| Method | ComputeRatio | ProxyTaskLoss | Latency(ms) |
|---|---:|---:|---:|
| LowRes-9x9 | 0.413265 | 0.074067 | 1.031226 |
| QACR-QueryAdaptive | 0.431846 | 0.071846 | 2.983424 |

这说明：

- 在当前 proxy 设定下，low-resolution 的确是非常强的对手；
- QACR 在性能上只表现出轻微优势，而没有形成压倒性领先；
- 因此现阶段不能把论文写成“QACR 全面优于 low-resolution”。

但这并不意味着 QACR 的问题设定是错的。更准确的解释应当是：

> low-resolution 证明“统一压缩输入”在一部分样本上已经很有效；而 QACR 关注的是另一类更困难的情形，即当 query 只依赖局部细粒度区域时，统一压缩可能会过早损失关键信息。

**建议在正文中的安全表述：**

> Low-resolution remains a strong baseline under matched compute, indicating that uniform input compression is already effective for many easy or globally answerable examples. We therefore do not claim that QACR universally dominates low-resolution. Instead, QACR is designed for cases where the answer depends on query-specific fine-grained regions, for which binary or uniform compression may be insufficient.

### 2.2 当前 latency 没有占优，不能把论文写成“已经实现真实加速”

当前硬件 profiling 与主表共同说明：

- QACR 的近预算 latency 为 `2.983424 ms`；
- LowRes 为 `1.031226 ms`；
- 稀疏执行仿真也没有表现出稳定单调加速；
- 当前执行器仍更接近“机制验证原型”，而不是成熟的条件执行系统。

这意味着：

- 论文现阶段不能把核心贡献写成“显著降低端到端推理延迟”；
- 更稳妥的定位是“算法上具备条件执行潜力，但真实系统收益仍依赖更高效的执行实现”。

**建议在正文中的安全表述：**

> Our current implementation validates the routing mechanism, but does not yet consistently translate algorithmic compute savings into end-to-end wall-clock speedup. We therefore position QACR primarily as a query-conditioned compute allocation framework, rather than a finished systems optimization for latency.

### 2.3 Hard routing 仍存在 collapse 风险，train-test mismatch 尚未完全闭环

当前证据显示：

- `mean_hard_minus_soft_gap = 0.007583`，平均 gap 不大；
- 但 `final_hard_ratio_shallow = 1.0`；
- 多组消融中 `hard_collapse_ratio` 仍偏高。

这说明：

- soft training 本身是可行的；
- 但 hard inference 还没有完全学成一个稳定、均衡、可解释的离散策略；
- reviewer 很可能会追问：模型到底学到的是“软加权”，还是“真实可部署的离散路由”。

**建议在正文中的安全表述：**

> Although the average soft-to-hard gap is small, hard routing can still exhibit route collapse in some settings. This suggests that optimization stability has improved, but the train-test transition is not yet fully resolved.

### 2.4 当前最严重的问题不是“不会变”，而是“变得不够准”

从热图与 budget 分析看，QACR 已经证明：

- 不同 query 下热图会改变；
- 不同 budget 下路径比例会改变；
- 因而“query-adaptive compute allocation”在机制上是成立的。

但 corner case 分析暴露出更关键的问题：

- `num_flagged_errors = 6/6`
- `miss_rate_key_tokens = 1.0`

这说明当前短板不是“没有动态性”，而是：

> 路由器学会了随着 query 改变分配，却还没有学会稳定保住真正关键的 token。

这条 limitation 很关键，因为它决定了论文的语气应该是：

- 不是“我们已经彻底解决了动态视觉算力分配”；
- 而是“我们验证了这条主线成立，但关键 token 保护仍是当前最重要的开放问题”。

**建议在正文中的安全表述：**

> Our failure cases suggest that the main challenge is no longer whether the router can change its allocation with the query, but whether it can reliably preserve the truly critical regions. Improving key-token protection is therefore a central next step rather than a minor implementation detail.

---

## 3. 正文可直接使用的英文草稿

下面给出一版可直接放进论文的 `Failure Analysis / Limitations` 草稿，后续只需根据真实 benchmark 结果微调。

### 3.1 Failure Analysis

```text
Failure Analysis. Our analysis reveals that the main remaining issue is not the absence of query adaptivity, but insufficient precision in protecting query-critical regions. While QACR produces clearly different routing heatmaps under different queries and responds systematically to budget changes, corner-case analysis shows that key tokens can still be under-allocated, with all six inspected cases being flagged by the current key-token miss criterion. This indicates that the router has learned to shift computation with the query, yet has not reliably learned where the computation must be concentrated. In addition, although the average soft-to-hard gap is small, hard routing can still collapse to a dominant path in some settings, suggesting that the train-test transition is not fully resolved.
```

### 3.2 Limitations

```text
Limitations. First, low-resolution processing remains a strong baseline under matched compute. In our current proxy setting, QACR only shows a small performance advantage over low-resolution, while not yet improving end-to-end latency. We therefore do not claim that QACR universally dominates uniform input compression. Second, our current implementation primarily validates the routing mechanism rather than a fully optimized conditional-execution runtime. As a result, algorithmic compute reduction does not consistently translate into wall-clock speedup. Third, the present study focuses on single-axis depth routing. Although this design keeps the method interpretable and controllable, it may not yet be sufficient for cases requiring finer-grained spatial recovery or attention-level adaptation. These limitations motivate future work on key-token protection, more faithful hard-routing optimization, and more efficient runtime support.
```

---

## 4. 更稳妥的结论口径

如果主表结果在下一阶段没有明显改变，正文和答辩建议使用如下口径：

### 4.1 能硬讲的

- QACR 明显优于传统 pruning。
- QACR 明显优于 image-only routing。
- QACR 已经证明 query-conditioned compute allocation 在机制上成立。

### 4.2 暂时不能硬讲的

- QACR 已经全面优于 low-resolution。
- QACR 已经显著降低真实端到端 latency。
- QACR 已经完全解决 hard routing 的部署问题。

### 4.3 更稳妥的替代表述

- 不说：`QACR delivers consistent system speedup.`
- 建议说：`QACR exposes a favorable mechanism for conditional computation, while realizing its full runtime benefit still depends on more efficient execution support.`

- 不说：`QACR outperforms all efficient baselines.`
- 建议说：`QACR is particularly advantageous over pruning-style baselines and is most promising on query-specific fine-grained cases.`

- 不说：`hard routing is solved.`
- 建议说：`hard routing is feasible but remains an open optimization bottleneck.`

---

## 5. Reviewer 可能的攻击点与建议回应

### 攻击点 1：这不就是另一种 query-guided pruning 吗？

建议回应：

> 不是。已有方法大多决定“哪些 token 保留/删除”，而 QACR 建模的是“每个 token 获得多少计算深度”。当前 limitation 恰恰也证明了这两者并不等价：即便热图会随 query 变化，若关键区域没有获得足够深算力，系统仍然会失败。

### 攻击点 2：为什么你们 latency 还更慢？

建议回应：

> 当前工作首先验证了路由机制，而不是完成了底层系统优化。真实 runtime gain 依赖更高效的条件执行和 kernel 支持，这一点我们在 limitation 中已明确承认。

### 攻击点 3：既然 low-resolution 这么强，为什么还要做 QACR？

建议回应：

> low-resolution 在大量简单样本上确实强，但它是统一压缩。QACR 关注的是 query-specific、局部细粒度信息主导的场景，尤其是 OCR、小目标和多区域干扰问题。论文的价值不在于否定 low-resolution，而在于说明统一压缩并不是唯一合理的效率路径。

### 攻击点 4：你们 hard routing 不是塌缩了吗？

建议回应：

> 是的，hard routing 仍存在未完全解决的稳定性问题，因此我们没有把它包装成“完全成熟的部署方案”。当前结果支持“可训练且有潜力”，但还需要更强的一致性优化与 key-token 保护机制。

---

## 6. 当前版本最适合放在论文中的一句话总结

> QACR has validated the mechanism of query-conditioned compute allocation, but its current limitations show that reliable key-token protection and faithful conditional execution remain the two critical steps before the method can become a fully competitive deployment-ready solution.

---

## 7. 建议的后续衔接

本稿完成后，下一步最自然衔接的是：

1. 把 `same image, different query` 控制实验做强；
2. 把 `key-token recall / miss rate` 升级为正文指标；
3. 用真实条件执行执行器重跑 latency；
4. 再根据结果回头修这份 limitation 文案的语气强弱。

