# Query-Adaptive Compute Routing (QACR)

<div align="center">
  <img src="outputs/phase312_key_token_control.png" width="80%" alt="QACR Approach">
</div>

**QACR** is a novel computational routing framework for Multi-Modal Large Language Models (MLLMs). Instead of allocating uniform computation uniformly across all image patches (tokens), QACR dynamically directs high-capacity "deep" computational routes strictly to regions specified by the textual user query. By introducing **Conditional Execution**, QACR achieves strong Pareto alignments for both Absolute Accuracy and Latency vs. Dense baselines.

## 🚀 Key Features

- **Query-Adaptive Routing**: Tokens required to answer the query undergo dense computation, while background tokens are aggressively pruned or compressed.
- **Hardware-Efficient Conditional Execution**: Realistic latency speedups and memory footprint reduction.
- **Attention-Level Refinements**: Maintains strong expressiveness using an explicit attention-level routing policy alongside traditional depth routing.
- **Compatible with Qwen-VL / LLaVA**: Generalizes across popular Multi-Modal frameworks out-of-the-box.

## 📦 Installation

To reproduce the study, please clone the directory and install dependencies:

```bash
git clone https://github.com/pengrui-art/QACR.git
cd QACR

# Create conda environment
conda create -n qacr python=3.10 -y
conda activate qacr

# Install requirements
pip install -r requirements.txt
```

## 🛠️ Usage

### Running Evaluator
Check out the baseline alignment and performance evaluation scripts:
```bash
python scripts/run_phase39_baseline_alignment.py --out-json outputs/alignment.json
```

### Pareto Generation
```bash
python scripts/run_phase311_official_benchmark_subset.py \
  --datasets VQAv2,TextVQA \
  --budgets 0.35,0.45,0.60
```

## ⬇️ Model Zoo (Hugging Face)

We provide pre-trained router checkpoints across standard compute budgets natively on [Hugging Face](https://huggingface.co/TezBaby).

| Budget Limit | Router Type | Download |
| --- | --- | --- |
| ~0.35 Compute | Depth-Only | [TezBaby/QACR-Qwen35-08B-B0.35](https://huggingface.co/TezBaby/QACR-Qwen35-08B-B0.35) |
| ~0.45 Compute | Depth-Only | [TezBaby/QACR-Qwen35-08B-B0.45](https://huggingface.co/TezBaby/QACR-Qwen35-08B-B0.45) |
| ~0.55 Compute | Attention-Enhanced | [TezBaby/QACR-Qwen35-08B-Attn-B0.55](https://huggingface.co/TezBaby/QACR-Qwen35-08B-Attn-B0.55) |

### Inference with HF Checkpoint

```python
from qacr.vision.multipath_depth import DepthOnlyRouter
from transformers import AutoModelForCausalLM

# Mock implementation usage
router = DepthOnlyRouter.from_pretrained("TezBaby/QACR-Qwen35-08B-B0.45")
# Inject into vision tower layer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat")
```

## 📝 Project Planner
Progress and experimental details are extensively logged in `/Docx`, predominantly `项目代码计划表.md`.

## 📜 Citation
If you find this code helpful, please cite our upcoming NeurIPS submission.
