# D-CoT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.21786-b31b1b.svg)](https://arxiv.org/abs/2602.21786)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) often induces "overthinking" in Small Language Models (SLMs), leading to performance degradation and excessive token consumption. **D-CoT** enforces a structured reasoning process using control tags — such as `<TEMP_LOW>` for fact-checking and `<TEMP_HIGH>` for multi-perspective exploration — as auxiliary scaffolding during training. By optimizing the CoT trajectory, D-CoT suppresses reasoning drift and simultaneously achieves token reduction and performance improvement.

> **Key Insight:** Control tags act as "training wheels" — they guide the model during learning, but become unnecessary at inference time because the disciplined reasoning pattern is **internalized** by the model.

## Results

Evaluated on **Qwen3-8B** with a 5K synthetic ORPO dataset, single GPU training:

| Benchmark | Base Model | D-CoT (LoRA) | Improvement |
|---|---|---|---|
| **MMLU-Pro** (0-shot) | 55.66% | **64.73%** | +9.1% |
| **GPQA-Diamond** (0-shot, 5-seed avg) | 43.03% | **52.93%** | +9.9% |

D-CoT also reduces average output tokens by **14–63%**, demonstrating more efficient reasoning.

## Overview

Traditional distilled CoT models often fall into **overthinking** — infinite verification loops, unnecessary manual arithmetic, and self-doubt that causes correct answers to be overwritten. D-CoT addresses this by:

1. **Structured thinking via control tags** — `<TEMP_LOW>` (fact anchoring), `<TEMP_HIGH>` (multi-perspective exploration), `<TEMP_MID>` (logical convergence)
2. **ORPO preference learning** with 31 rejection categories targeting specific overthinking patterns
3. **Internalization** — the model learns the *rhythm* of disciplined reasoning, not just the tags themselves

## Repository Structure

```
├── Train-ORPO.py              # ORPO training script
├── MMLU-pro-bench.py          # MMLU-Pro evaluation
├── MMLU-pro-bench-n.py        # MMLU-Pro evaluation (chunked, for large runs)
├── MMLU-pro-bench-alt.py      # MMLU-Pro evaluation (alternative config)
├── GPQA-diamond-bench.py      # GPQA-Diamond evaluation
├── merge-results.py           # Merge chunked benchmark results
├── gpqa-result.py             # GPQA multi-seed result aggregation
├── experimental_data/
│   ├── hikaku-gpqa.py         # Contamination check (DPO vs GPQA)
│   └── mmlu-eliminate.py      # Contamination check (DPO vs MMLU-Pro)
├── Figure_1.html              # Figure 1 (two-panel comparison)
├── figure1_arxiv_v2.html      # Figure 1 (three-panel, arXiv version)
├── gen_scatter_figure.py      # Generate scatter plot (Figure 2)
└── setup.sh                   # Environment setup
```

## Quick Start

### Setup
```bash
bash setup.sh
```

### Training
```bash
bash run_train.sh
```

### Evaluation
```bash
# MMLU-Pro (0-shot, full)
python MMLU-pro-bench.py -m unsloth/Qwen3-8B -lp <lora_path> -up -ui

# GPQA-Diamond
python GPQA-diamond-bench.py -m unsloth/Qwen3-8B -lp <lora_path> -up -ui
```

## Japanese Version

The Japanese version of this paper was submitted to **JSAI 2025** (The 39th Annual Conference of the Japanese Society for Artificial Intelligence). It will be publicly available on J-STAGE around May 2025. For early access, please refer to `人工知能学会全国大会_原稿_生方俊輔.pdf` included in this repository.

## Links

- **Paper (arXiv):** [https://arxiv.org/abs/2602.21786](https://arxiv.org/abs/2602.21786)
- **Code:** [gitpullpull/DisciplinedChainOfThought](https://github.com/gitpullpull/DisciplinedChainOfThought)
- **Benchmark Results:** [gitpullpull/D-CoT-Benchmarks](https://huggingface.co/datasets/gitpullpull/D-CoT-Benchmarks)
- **Dataset (5,181 Chosen/Rejected pairs):** [gitpullpull/D-CoT-datasets](https://huggingface.co/datasets/gitpullpull/D-CoT-datasets)

## Contact

- **X (Twitter):** [@gitpullpull](https://x.com/gitpullpull)
- **Email:** gitpullpull@gmail.com

## Citation

```bibtex
@article{ubukata2025dcot,
  title={D-CoT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models},
  author={Ubukata, Shunsuke},
  journal={arXiv preprint arXiv:2602.21786},
  year={2025}
}
```
