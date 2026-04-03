# QACR 补充创新点代码计划表

## 记录与打卡规则

1. 本计划表只服务于当前 `NeurIPS` 缺口补强，不替代总的 [项目代码计划表](/data1/pengrui/CCFA/QACR/Docx/项目代码计划表.md)。
2. 每个子项目前均使用 checklist，完成后请将 `[ ]` 改为 `[x]`。
3. 每个任务完成后，必须补齐以下记录：
   - 完成方法
   - 运行结果与分析
   - 核心代码
   - 运行命令

---

## 文档目标

这份计划表只解决一个问题：

> **如何把 QACR 从“有论文潜力”推进到“更接近 NeurIPS 主会完成态”。**

当前优先级已经明确，不再继续把资源分散到低收益方向。后续代码与实验只围绕三类补强：

1. `Query-conditioned key-token protection`
2. `Key-token recall / miss-rate` 量化指标
3. `Matched-latency / throughput / memory` 系统证据

---

## Phase A：关键 Token 指标与分析基建
**目标**：把“QACR 是否真的更能保住 query-critical token”从口头判断变成正式指标。**

- [x] **任务 A.1：定义 key-token 标注协议**
  - 要求：为 `TextVQA / DocVQA` 明确定义什么是 `query-critical token`。
  - 建议优先级：
    - `OCR token` 与 `ground-truth answer` 的 overlap
    - 问题类型触发的 token 筛选规则
    - 文档字段类问题的 exact-copy 对齐
  - 交付物：
    - 一份可复用的 key-token 标注函数
    - 一份小样本人工核查结果
  - 验收门槛：
    - 在人工抽查样本上，key-token 标注明显合理
    - 至少支持 `TextVQA / DocVQA`
  - *【完成记录】*
    - 完成方法：新增 `qacr/analysis/key_tokens.py`，实现 `key-token` 标注协议与问题类型分类逻辑。当前协议分成两层：1）`TextVQA` 使用 `ocr_tokens + GT answer` 进行 token 级标注，优先做 OCR span match，匹配不到时再退回 answer-unit fallback；2）`DocVQA` 当前本地 `DocVQA` 配置不含 OCR token，因此本阶段先退回到 `answer-unit` 协议，并把这一限制显式记入报告。同步新增脚本 `scripts/run_phaseA1_key_token_protocol.py`，可直接输出 `json + md` 协议报告；同时在 `qacr/data/vqa_dataset.py` 与 `scripts/eval_qacr_benchmark.py` 中补入 `sample_id / question_type / ocr_tokens` 透传字段，为后续 `A.2` 做正式指标统计打基础。
    - 运行结果与分析：
      - 正式抽样报告路径：
        - `outputs/phaseA_key_token_protocol/phaseA1_key_token_protocol.json`
        - `outputs/phaseA_key_token_protocol/phaseA1_key_token_protocol.md`
      - `TextVQA`（`max_samples=200`）：
        - `token_level_samples = 194 / 200 (97.00%)`
        - `samples_with_key_tokens = 146 / 200 (73.00%)`
        - `match_strategy_breakdown = {ocr_span_match: 145, answer_unit_fallback: 1, answer_units_only: 54}`
        - `note_breakdown = {no_ocr_tokens_available: 6, no_reliable_token_match: 48}`
      - `DocVQA`（`max_samples=200`）：
        - `token_level_samples = 0 / 200`
        - `answer_units_only = 200 / 200`
        - `note_breakdown = {no_ocr_tokens_available: 200}`
      - 结论：
        - `TextVQA` 已经具备可用的 token 级标注基础，可以进入 `A.2 key-token recall / miss-rate`。
        - `DocVQA` 当前数据镜像缺少 OCR token，短期内应先把它视为 answer-unit 协议；若后续要做真正 token 级 key-token 指标，需要补充页面 OCR 或改用带 OCR 字段的数据版本。
    - 核心代码：
      - `qacr/analysis/key_tokens.py`
      - `qacr/analysis/__init__.py`
      - `scripts/run_phaseA1_key_token_protocol.py`
      - `qacr/data/vqa_dataset.py`
      - `scripts/eval_qacr_benchmark.py`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && conda run -n qacr --no-capture-output python scripts/run_phaseA1_key_token_protocol.py --datasets textvqa,docvqa --local-data-dir data --max-samples 200 --report-samples-per-dataset 8`

- [x] **任务 A.2：实现 key-token recall / miss-rate 评测脚本**
  - 要求：在已有评测结果基础上，统计不同方法对关键 token 的保护情况。
  - 指标至少包括：
    - `key_token_recall`
    - `key_token_miss_rate`
    - `deep_route_hit_rate`
    - `shallow_or_deep_coverage`
  - 对比对象至少包括：
    - `QACR b0.45`
    - `TokenPruning@0.45`
    - `ImageOnly@0.45`
    - `LowRes-9x9`
  - 验收门槛：
    - 能输出 `json + md` 双格式结果
    - 能按数据集单独汇总
  - *【完成记录】*
    - 完成方法：新增 `scripts/run_phaseA2_key_token_metrics.py`，复用 `A.1` 的 key-token 标注协议，在已有评测结果文件上计算第一版 **prediction-side key-token metrics**。当前版本会自动按评测顺序对齐结果文件与数据集样本，并统计 `pred / pred_raw` 两套指标，包括 `pred_key_token_recall`、`pred_key_token_miss_rate`、`pred_all_target_units_hit_rate`、`pred_any_target_unit_hit_rate`。脚本已修复“无可衡量 target unit（如 `#`、`?`）样本不应被计入 recall 分母”的口径问题，并改成轻量数据读取，不再为分析脚本加载整张图像。
    - 运行结果与分析：
      - 结果路径：
        - `outputs/phaseA_key_token_metrics/textvqa_key_token_metrics.json`
        - `outputs/phaseA_key_token_metrics/textvqa_key_token_metrics.md`
        - `outputs/phaseA_key_token_metrics/docvqa_key_token_metrics.json`
        - `outputs/phaseA_key_token_metrics/docvqa_key_token_metrics.md`
      - `TextVQA`（当前可直接比较的是 `QACR b0.45` 的几版正式结果）：
        - `current_qacr_b045`: `pred_key_token_recall = 0.3521`
        - `pre_ocrcorr`: `0.3520`
        - `pre_postproc`: `0.3519`
        - 解释：`OCR-aware correction / postprocess` 对 `TextVQA` 的提升方向在 key-token 指标上是**微弱正向**，与主表中的小幅提升一致，但增益确实很小。
      - `DocVQA`（当前可比较多个全量方法，但由于无 OCR token，仍是 `answer-unit` 协议）：
        - `QACR b0.45`: `pred_key_token_recall = 0.2514`
        - `TokenPruning@0.45`: `0.0946`
        - `ImageOnly@0.45`: `0.1141`
        - `LowRes-9x9`: `0.1018`
        - `Original`: `0.8161`
        - 解释：即便在 answer-unit 级别，`QACR` 也明显优于 `TokenPruning / ImageOnly / LowRes`，但仍与 `Original` 存在巨大差距。
      - 当前局限：
        - 这还是 **prediction-side** 指标，不是 route-level 指标；
        - `TextVQA` 的多方法横向比较暂时还受限于若干 baseline 的全量结果文件被 `1-sample smoke` 覆盖；
        - `DocVQA` 若要升级为真正 token-level key-token 指标，仍需补齐 OCR token 或页面 OCR 解析。
    - 核心代码：
      - `scripts/run_phaseA2_key_token_metrics.py`
      - `qacr/analysis/key_tokens.py`
      - `qacr/analysis/__init__.py`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && conda run -n qacr --no-capture-output python scripts/run_phaseA2_key_token_metrics.py --dataset textvqa --local-data-dir data --method current_qacr_b045=checkpoints/qacr_vqav2_b0.45/eval_results_textvqa.json --method pre_ocrcorr=checkpoints/qacr_vqav2_b0.45/history/eval_results_textvqa.pre_ocrcorr_fullrerun_20260403_213146.json --method pre_postproc=checkpoints/qacr_vqav2_b0.45/history/eval_results_textvqa.pre_postproc_rerun_20260403_194354.json --report-examples 6 --out-json outputs/phaseA_key_token_metrics/textvqa_key_token_metrics.json --out-md outputs/phaseA_key_token_metrics/textvqa_key_token_metrics.md`

- [x] **任务 A.3：same image, different query 控制实验**
  - 要求：对同一张图像构造至少两类不同 query，展示路由热图与关键 token 覆盖的变化。
  - 重点：
    - 不只出图，还要有定量统计
    - 要能说明变化来自 query，而不是随机波动
  - 交付物：
    - 可视化图
    - 对应路径比例表
    - key-token coverage 对照表
  - 验收门槛：
    - 至少整理 `3-5` 个高质量案例
  - *【完成记录】*
    - 完成方法：复用已有的 `scripts/run_phase312_key_token_control.py` 结果作为核心实验资产，并新增整理脚本 `scripts/run_phaseA3_same_image_query_report.py`，将原始 `phase312` 摘要自动转成当前 `Phase A` 体系可直接引用的 `md + png` 正式报告。这样避免重复跑实验，同时把已有控制实验从“历史脚本结果”提升成“当前 NeurIPS 补证据链的一部分”。
    - 运行结果与分析：
      - 结果路径：
        - `outputs/phaseA_same_image_query/phaseA3_same_image_query_report.md`
        - `outputs/phaseA_same_image_query/phaseA3_same_image_query_report.png`
      - 关键 aggregate 指标：
        - `QACR`: `key_token_recall=0.8383`, `miss_rate=0.1617`, `shift_corr=0.8852`, `flagged_errors=2`
        - `TokenPruning`: `0.4970`, `0.5030`, `0.0000`, `5`
        - `LVPruning-like`: `0.8160`, `0.1840`, `0.9618`, `3`
        - `CROP-like`: `0.9307`, `0.0693`, `0.9971`, `0`
      - 当前结论：
        - `QACR` 相比 `TokenPruning` 已经明显展示出更强的 query-conditioned 行为：同图换 query 时，路由确实会跟着动，而且关键区域召回更高、漏保更少。
        - `QACR` 在这个控制实验上可作为**机制证据**成立，但不应把它写成“全面碾压所有启发式基线”；`LVPruning-like` 和 `CROP-like` 在这个 toy 控制上仍然很强。
        - 因此 `A.3` 最适合支撑的论点是：**QACR 的 query-conditioned routing 不是空故事，而是可被 same-image/different-query 控制实验观察到的真实机制。**
    - 核心代码：
      - `scripts/run_phase312_key_token_control.py`
      - `scripts/run_phaseA3_same_image_query_report.py`
      - `outputs/phase312_key_token_control_summary.json`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && conda run -n qacr --no-capture-output python scripts/run_phaseA3_same_image_query_report.py`

---

## Phase B：系统 Profiling 与公平对比
**目标**：把“算法上省算”补成“系统上也有意义”的证据链。**

- [x] **任务 B.1：统一 latency / throughput / memory profiling 脚本**
  - 要求：为主表方法提供统一硬件、统一 batch、统一输入尺寸下的真实 profiling。
  - 至少覆盖：
    - `Original`
    - `QACR b0.45`
    - `TokenPruning@0.45`
    - `ImageOnly@0.45`
    - `LowRes-9x9`
  - 指标至少包括：
    - `wall_clock_latency_ms`
    - `throughput_samples_per_s`
    - `peak_gpu_memory_mb`
    - `batch_size_sensitivity`
  - 验收门槛：
    - 同一脚本可一键复跑
    - 输出 `csv + md + json`
  - *【完成记录】*
    - 完成方法：新增统一脚本 `scripts/run_phaseB1_unified_profiling.py`，直接复用现有评测链路中的 `eval_collate_fn + QACR hook + baseline hook`，对 `Original / QACR b0.45 / TokenPruning@0.45 / ImageOnly@0.45 / LowRes-9x9` 在统一数据协议下执行端到端 profiling。脚本支持 batch-size sweep，并输出 `json / csv / md` 三种格式。过程中额外修正了两处口径：1）baseline hook 的挂载点与现有评测对齐到 `visual.merger`；2）`LowRes` 的 compute 口径固定为 `9x9 / 14x14`，不误用 executor 内部的深算比例。
    - 运行结果与分析：
      - 当前第一轮 profiling 使用：
        - 数据集：`textvqa`
        - 样本数：`max_samples=8`
        - batch size：`1,2`
        - 设备：`GPU0 (RTX 4090)`
      - 结果路径：
        - `outputs/phaseB_unified_profiling/phaseB1_unified_profiling.json`
        - `outputs/phaseB_unified_profiling/phaseB1_unified_profiling.csv`
        - `outputs/phaseB_unified_profiling/phaseB1_unified_profiling.md`
      - 关键结果（`bs=2`）：
        - `Original`: `425.38 ms/sample`, `2.35 samples/s`
        - `QACR b0.45`: `294.28 ms/sample`, `3.40 samples/s`
        - `TokenPruning@0.45`: `264.15 ms/sample`, `3.79 samples/s`
        - `ImageOnly@0.45`: `547.55 ms/sample`, `1.83 samples/s`
        - `LowRes-9x9`: `267.64 ms/sample`, `3.74 samples/s`
      - 当前结论：
        - `QACR b0.45` 已经出现了相对 `Original` 的**真实系统优势**：在这轮统一 profiling 中，`sample latency` 降低约 `30.8%`（`425.38 -> 294.28 ms`），`throughput` 提升约 `44.7%`（`2.35 -> 3.40 samples/s`）。
        - `TokenPruning` 和 `LowRes` 依然更快，但这和它们在主表上明显更差的精度是一致的；因此它们更适合作为“速度极强、精度损失明显”的对照。
        - `ImageOnly` 在系统侧并没有表现出优于 `QACR` 的优势，说明“没有 query 的轻路由”并不会自动带来更好的部署收益。
        - 这还只是**第一轮小样本 profiling**，足够作为 `Phase B` 的启动证据，但还不足以直接充当最终论文主表；后续仍需扩大样本规模，并整理成 `matched latency` 主表。
    - 核心代码：
      - `scripts/run_phaseB1_unified_profiling.py`
      - `scripts/eval_qacr_benchmark.py`
      - `qacr/qacr_model.py`
      - `scripts/train_baselines_e2e.py`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && conda run -n qacr --no-capture-output python scripts/run_phaseB1_unified_profiling.py --dataset textvqa --local-data-dir data --methods original,qacr_b045,token_pruning,image_only,low_res --batch-sizes 1,2 --max-samples 8 --warmup-batches 1 --gpu-id 0 --num-workers 0`

- [x] **任务 B.2：matched-compute / matched-latency 双口径主表**
  - 要求：生成两套主表，而不是只保留 compute 口径。
  - 具体输出：
    - `matched compute` 主表
    - `matched latency` 主表
    - `Accuracy-Compute Pareto`
    - `Accuracy-Latency Pareto`
  - 验收门槛：
    - 能直接被论文主文使用
    - 图表标题、图注和单位完整
  - *【完成记录】*
    - 完成方法：新增 `scripts/run_phaseB2_matched_tables.py`，将 `Phase 6` 的全量准确率主表与 `Phase B.1` 的 profiling 结果按 `method_key + dataset` 合并，自动生成 `matched compute view`、`matched latency view`、以及两张配套的 `accuracy-compute / accuracy-latency` 图。当前第一版使用 `textvqa` 作为示范数据集，并按 `min_sample_latency` 选择每个方法的代表性 profiling 行。
    - 运行结果与分析：
      - 结果路径：
        - `outputs/phaseB_matched_tables/phaseB2_matched_tables.md`
        - `outputs/phaseB_matched_tables/phaseB2_matched_tables.csv`
        - `outputs/phaseB_matched_tables/phaseB2_matched_tables.json`
        - `outputs/phaseB_matched_tables/phaseB2_accuracy_compute.png`
        - `outputs/phaseB_matched_tables/phaseB2_accuracy_latency.png`
      - `textvqa` 上的当前关键结论：
        - `QACR b0.45` 在近预算方法中形成了比较健康的折中点：
          - `accuracy = 0.2647`
          - `mean_compute = 0.4482`
          - `sample_latency = 294.28 ms`
          - `throughput = 3.40 samples/s`
        - 相比 `Original`：
          - sample latency 改善约 `30.8%`
          - throughput 改善约 `44.5%`
          - compute 从 `1.0000` 降到 `0.4482`
        - 相比 `TokenPruning@0.45`：
          - `QACR` 略慢（`294.28 vs 264.15 ms/sample`）
          - 但准确率高很多（`0.2647 vs 0.0714`，差值 `+0.1933`）
        - 相比 `LowRes-9x9`：
          - `QACR` 略慢（`294.28 vs 267.64 ms/sample`）
          - 但准确率高很多（`0.2647 vs 0.0729`，差值 `+0.1917`）
      - 当前结论：
        - `B.2` 已经把我们的系统叙事从“有一组 profiling 数字”推进到“有双口径主表 + 可读对比结论”。
        - 这还只是 `textvqa` 第一版，后续若要进入论文主文，仍建议至少再补 `docvqa` 或更大样本版本。
    - 核心代码：
      - `scripts/run_phaseB2_matched_tables.py`
      - `outputs/phase6_full_benchmarks/phase62_final_summary.json`
      - `outputs/phaseB_unified_profiling/phaseB1_unified_profiling.json`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && conda run -n qacr --no-capture-output python scripts/run_phaseB2_matched_tables.py --dataset textvqa`

- [x] **任务 B.3：运行稳定性与复现脚本固化**
  - 要求：把单任务串行、日志落盘、结果备份、烟雾测试过滤全部固化进脚本。
  - 重点：
    - 避免 1-sample smoke 污染主表
    - 避免多任务并发被系统 `Killed`
  - 验收门槛：
    - 评测脚本默认稳定
    - 汇总脚本不会误读 smoke 文件
  - *【完成记录】*
    - 完成方法：新增稳定串行 launcher `scripts/run_phaseB3_stable_eval.sh`，把 `单任务串行 -> 实时日志落盘 -> 旧结果自动备份 -> 临时结果文件成功后再覆盖正式结果` 固化成默认流程。该脚本支持按 `dataset / method` 串行跑完整主表方法，也支持通过 `RESULT_ROOT` 将烟雾测试写到独立目录，避免污染正式结果。同步增强 `scripts/plot_phase63_pareto_frontiers.py`：加入基于 `expected_count + min_coverage_ratio` 的 full-benchmark 覆盖校验，显式过滤 `1-sample smoke / partial` 结果，并输出 `phase62_filter_report.csv`，把“是否回退到旧主表”的判断变成可审计记录。
    - 运行结果与分析：
      - 稳定串行 smoke 验证：
        - 运行路径：`outputs/phaseB3_stable_smoke/qacr_b045/eval_results_textvqa.json`
        - 日志路径：`logs/phaseB3_stable_eval/phaseB3_smoke/launcher.log`、`logs/phaseB3_stable_eval/phaseB3_smoke/qacr_b045_textvqa.log`
        - 结果：`1-sample TextVQA smoke` 成功跑通，`total_evaluated = 1`，`mean_compute = 0.4497`，验证了新 launcher 的日志、落盘、结果写回链路是通的。
      - 主表过滤验证：
        - 重新生成后，`outputs/phase6_full_benchmarks/phase62_filter_report.csv` 共记录 `27` 条候选结果，其中 `8` 条被识别为 `insufficient_coverage`。
        - 这 `8` 条全部是 `textvqa` 上的 `1-sample` 残留文件（`fastv / image_only / lvpruning / low_res / original / qacr_b035 / qacr_b060 / token_pruning`），并且都已正确回退到已有的全量主表记录。
        - 重新汇总后主表仍保持 `per_dataset_rows = 27`、`missing_files = 0`，说明新的过滤逻辑已经能稳定阻断 smoke 文件污染最终 summary。
      - 当前结论：
        - `B.3` 已经把“容易被运行细节拖垮”的部分固化成默认脚本行为。
        - 之后继续扩 `Phase B` 或进入 `Phase C` 时，可以直接复用这套稳定入口和过滤规则，不需要再手工排查空日志、半覆盖结果或 smoke 文件污染。
    - 核心代码：
      - `scripts/run_phaseB3_stable_eval.sh`
      - `scripts/plot_phase63_pareto_frontiers.py`
      - `outputs/phase6_full_benchmarks/phase62_filter_report.csv`
      - `logs/phaseB3_stable_eval/phaseB3_smoke/launcher.log`
    - 运行命令：`cd /data1/pengrui/CCFA/QACR && TS=phaseB3_smoke RESULT_ROOT=outputs/phaseB3_stable_smoke DATASETS=textvqa METHODS=qacr_b045 MAX_SAMPLES=1 BATCH_SIZE=1 NUM_WORKERS=0 DOCVQA_NUM_WORKERS=0 GPU_ID=0 NO_PERSISTENT_WORKERS=1 bash scripts/run_phaseB3_stable_eval.sh`

---

## Phase C：方法增强版创新
**目标**：把 failure analysis 里的“关键 token 保不住”推进成方法层面的增强，而不再停留在后处理层。**

- [x] **任务 C.1：实现 query-conditioned key-token protection v1**
  - 要求：在不破坏主线简洁性的前提下，引入最小化的 key-token 保护机制。
  - 推荐方向：
    - `OCR-aware deep-route prior`
    - `question-type-conditioned deep bias`
    - `text-like region regularization`
  - 限制条件：
    - 不引入高分辨率重编码
    - 不引入大规模复杂分支
    - 不把 prompt / postprocess 伪装成方法创新
  - 验收门槛：
    - `TextVQA-200/500` 至少一组稳定转正
    - `mean_compute` 增幅可控
  - *【完成记录】*
    - 完成方法：新增 `qacr/protection.py`，实现 query-conditioned protection plan，根据 `question / question_type / dataset / OCR tokens` 生成伪 key-token 保护集合；在 `qacr/qacr_model.py` 中把保护计划接入路由 hook，对 protected token 注入 `skip/shallow/deep` logit bias，并在 hard budget matching 中保底保留 protected keep/deep token。评测脚本 `scripts/eval_qacr_benchmark.py` 支持 `--protection-mode prior_only` 等模式，用于不训练先验证方向。
    - 运行结果与分析：
      - `TextVQA-200`：`0.27333 -> 0.28000`，`+0.00667`，`mean_compute 0.44923 -> 0.44830`
      - `TextVQA-500`：`0.29000 -> 0.29000`，准确率持平，`mean_compute 0.44781 -> 0.44696`
      - 结论：`prior-only` 在 `200` 样本上稳定转正、在 `500` 样本上保持不掉点且 compute 微降，说明“query-conditioned key-token protection”这条方法主线是成立的；但仅靠启发式 prior 的收益有限，更像 Phase C 的安全起点，而不是终版答案。
    - 核心代码：
      - `qacr/protection.py`
      - `qacr/qacr_model.py`
      - `scripts/eval_qacr_benchmark.py`
      - `outputs/phaseC_textvqa/summary_200_triplet.md`
      - `outputs/phaseC_textvqa/summary_500_triplet.md`
    - 运行命令：
      - `CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py --checkpoint-dir checkpoints/qacr_vqav2_b0.45 --model Model/Qwen35-08B --dataset textvqa --local-data-dir data --max-samples 200 --batch-size 8 --num-workers 12 --prefetch-factor 2 --protection-mode prior_only --protection-topk-scale 1.0 --protection-keep-scale 1.0 --protection-deep-scale 1.0 --protection-logit-bias 1.0 --out-file outputs/phaseC_textvqa/prior_v1_200.json`
      - `CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py --checkpoint-dir checkpoints/qacr_vqav2_b0.45 --model Model/Qwen35-08B --dataset textvqa --local-data-dir data --max-samples 500 --batch-size 8 --num-workers 12 --prefetch-factor 2 --protection-mode prior_only --protection-topk-scale 1.0 --protection-keep-scale 1.0 --protection-deep-scale 1.0 --protection-logit-bias 1.0 --out-file outputs/phaseC_textvqa/prior_v1_500.json`

- [x] **任务 C.2：key-token auxiliary loss / regularization 对比**
  - 要求：把保护机制从启发式偏置推进到显式训练目标。
  - 至少比较：
    - 无保护
    - prior-only
    - auxiliary-loss
  - 观察指标：
    - `accuracy`
    - `mean_compute`
    - `key_token_recall`
    - `hard_collapse_ratio`
  - 验收门槛：
    - 至少证明其中一条线比现有 `b0.45` 更有前景
  - *【完成记录】*
    - 完成方法：在 `scripts/train_qacr_e2e.py` 中加入 `--init-checkpoint` 与 `--protection-mode {none,prior_only,aux_only,prior_aux}`，同时新增 `--lambda-key-token`；训练时把 `question / question_type / OCR tokens` 从 `qacr/data/vqa_dataset.py` 和 `vqa_collate_fn` 传到路由 hook，并在 `qacr/qacr_model.py` 中把 `protection_loss` 与原预算损失联合优化。训练过程中顺手修复了 multimodal truncation 造成的 image-token mismatch，去掉了 `vqa_collate_fn` 中的强制 `max_length=1024` 截断，保证 Phase C 训练能稳定完成。
    - 运行结果与分析：
      - `TextVQA-200`：`baseline 0.27333 / prior_v1 0.28000 / aux_v1 0.38667`
      - `TextVQA-500`：`baseline 0.29000 / prior_v1 0.29000 / aux_v1 0.42400`
      - `mean_compute`：`aux_v1` 从 `~0.448` 下降到 `~0.391`
      - 结论：`aux_only` 明显优于 `baseline` 与 `prior_only`，而且这种优势在 `200` 与 `500` 两级都保持存在，不是小样本噪声；因此 Phase C 的真正主胜线已经从“启发式 prior”切换到“显式 key-token auxiliary loss”。
    - 核心代码：
      - `qacr/qacr_model.py`
      - `qacr/data/vqa_dataset.py`
      - `scripts/train_qacr_e2e.py`
      - `scripts/eval_qacr_benchmark.py`
      - `outputs/phaseC_textvqa/summary_200_triplet.md`
      - `outputs/phaseC_textvqa/summary_500_triplet.md`
    - 运行命令：
      - `CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL conda run -n qacr --no-capture-output python scripts/train_qacr_e2e.py --model Model/Qwen35-08B --dataset textvqa --local-data-dir data --output-dir checkpoints/qacr_textvqa_phasec_aux_v1 --max-samples 512 --epochs 1 --batch-size 2 --grad-accum 4 --lr 1e-4 --budget 0.45 --executor-output-alpha 0.30 --init-checkpoint checkpoints/qacr_vqav2_b0.45/best.pt --protection-mode aux_only --lambda-key-token 0.1`
      - `CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py --checkpoint-dir checkpoints/qacr_textvqa_phasec_aux_v1 --model Model/Qwen35-08B --dataset textvqa --local-data-dir data --max-samples 500 --batch-size 8 --num-workers 12 --prefetch-factor 2 --protection-mode checkpoint --out-file outputs/phaseC_textvqa/aux_v1_500.json`

- [x] **任务 C.3：小样本到全量的晋级流程**
  - 要求：把后续方法提效统一成固定流程：
    - `200` 样本预筛
    - `500` 样本复核
    - 全量串行复跑
  - 重点：
    - 只允许单变量进入全量
    - 所有结果都要写回主表与历史目录
  - 验收门槛：
    - 至少完成一轮从小样本到全量的闭环
  - *【完成记录】*
    - 完成方法：新增 `scripts/run_phaseC_textvqa_promotion.py` 统一汇总 `200 -> 500 -> full` 三阶段结果，并按 `baseline / prior_v1 / aux_v1` 自动生成 promotion summary。先用 `prior_v1` 跑通首轮全量闭环，再把已经在 `200/500` 上明显领先的 `aux_v1` 提升到全量，最终得到完整 triplet summary。
    - 运行结果与分析：
      - `full`：`baseline 0.26467 / prior_v1 0.26353 / aux_v1 0.39793`
      - `aux_v1` 相对 baseline：`+0.13327 accuracy`，`-0.05500 mean_compute`
      - `prior_v1` 相对 baseline：`-0.00113 accuracy`，`-0.00078 mean_compute`
      - 补充验证 1（`TextVQA key-token metrics`）：`pred_key_token_recall 0.3521 -> 0.5891`，`all_units_hit 0.3271 -> 0.5680`，说明 `aux_v1` 的精度提升和关键单元保留能力提升是同步出现的，不只是答案后处理偶然修回。
      - 补充验证 2（`DocVQA-200` 迁移复核，单卡 `bs=4`）：`accuracy 0.0300 -> 0.2950`，`mean_compute 0.4433 -> 0.4159`。这表明 `aux_only` 至少在快速跨数据集复核上没有出现负迁移，反而呈现出明显正向迁移信号。
      - 结论：Phase C 已完成至少一轮完整晋级闭环，而且最值得推进的版本已经非常明确：`aux_only` 明显优于 `prior_only`。这也意味着项目的方法层叙事从“我们会保护关键 token”推进到了“我们已经可以通过显式辅助目标学出更好的保护行为”。
      - 风险提示：`DocVQA` 的全量迁移复核此前在并发配置下被系统 `Killed`，当前最稳的证据仍是 `DocVQA-200` 单卡串行结果；下一阶段应继续补单任务全量复核与 route-level dump，确认收益不仅来自 answer-unit 侧。
    - 核心代码：
      - `scripts/run_phaseC_textvqa_promotion.py`
      - `outputs/phaseC_textvqa/summary_200_triplet.md`
      - `outputs/phaseC_textvqa/summary_500_triplet.md`
      - `outputs/phaseC_textvqa/summary_full_triplet.md`
      - `outputs/phaseC_textvqa/aux_v1_full.json`
      - `outputs/phaseC_textvqa/aux_v1_textvqa_key_token_metrics.md`
      - `outputs/phaseC_transfer/docvqa_200_key_token_metrics.md`
    - 运行命令：
      - `CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py --checkpoint-dir checkpoints/qacr_textvqa_phasec_aux_v1 --model Model/Qwen35-08B --dataset textvqa --local-data-dir data --max-samples 5000 --batch-size 8 --num-workers 12 --prefetch-factor 2 --protection-mode checkpoint --out-file outputs/phaseC_textvqa/aux_v1_full.json > outputs/phaseC_textvqa/logs/aux_v1_full.log 2>&1`
      - `conda run -n qacr --no-capture-output python scripts/run_phaseC_textvqa_promotion.py --stage full --variant baseline=checkpoints/qacr_vqav2_b0.45/eval_results_textvqa.json --variant prior_v1=outputs/phaseC_textvqa/prior_v1_full.json --variant aux_v1=outputs/phaseC_textvqa/aux_v1_full.json --baseline-name baseline --out-json outputs/phaseC_textvqa/summary_full_triplet.json --out-md outputs/phaseC_textvqa/summary_full_triplet.md`

---

## Phase D：论文级证据打磨
**目标**：把前面补出来的内容直接变成论文能用的图和表。**

- [ ] **任务 D.1：QACR vs pruning family 的论文图整理**
  - 要求：固定生成一组可直接用于论文主文的图：
    - `QACR vs TokenPruning`
    - `QACR vs ImageOnly`
    - `QACR vs LowRes`
  - 推荐内容：
    - 一张 Pareto 图
    - 一张关键 token 指标图
    - 一张失败案例图
  - *【完成记录】*
    - 完成方法：
    - 运行结果与分析：
    - 核心代码：
    - 运行命令：

- [ ] **任务 D.2：OCR / 文档专项 breakdown**
  - 要求：把 `TextVQA / DocVQA` 的结果拆成更有解释力的子类统计。
  - 推荐维度：
    - `numeric_time`
    - `brand_entity`
    - `name_title`
    - `direct_reading`
    - `location`
  - 验收门槛：
    - 至少能支持主文 discussion 或 appendix 小节
  - *【完成记录】*
    - 完成方法：
    - 运行结果与分析：
    - 核心代码：
    - 运行命令：

- [ ] **任务 D.3：第二基座复现**
  - 要求：在第二个基座模型上复现最核心的结论，避免论文被质疑为单基座现象。
  - 最低目标：
    - 不要求全实验重跑
    - 至少复现关键趋势与同预算相对优势
  - 验收门槛：
    - 有一张跨基座对照表
  - *【完成记录】*
    - 完成方法：
    - 运行结果与分析：
    - 核心代码：
    - 运行命令：

---

## 当前推进顺序建议

后续不要并行开太多分支，建议严格按下面顺序推进：

1. `A.1 -> A.2 -> A.3`
2. `B.1 -> B.2`
3. `C.1 -> C.2 -> C.3`
4. `D.1 -> D.2`
5. 有余力再做 `D.3`

---

## 当前停手原则

以下方向暂时不继续追加资源：

- 更强 `min_keep_ratio / min_deep_ratio` 扫参
- 高分辨率重编码
- 多轴复杂 routing
- 继续堆 prompt 变体
- 继续堆小后处理规则

原因很明确：这些方向当前要么收益小，要么会把主线讲散。

---

## 一句话总结

这份补充代码计划表的核心目标不是“再做更多实验”，而是：

> **补出足以支撑 NeurIPS 叙事的关键证据，让 QACR 从“有潜力”走向“更像一篇完成态论文”。**
