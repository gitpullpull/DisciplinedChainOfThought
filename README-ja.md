# D-CoT: 小規模言語モデルにおける効率的推論のための規律ある思考連鎖学習

[![arXiv](https://img.shields.io/badge/arXiv-2602.21786-b31b1b.svg)](https://arxiv.org/abs/2602.21786)
[![License: arXiv](https://img.shields.io/badge/License-arXiv-green.svg)](https://arxiv.org/abs/2602.21786)

大規模言語モデル（LLM）からのChain-of-Thought（CoT）蒸留は、小規模言語モデル（SLM）において「過剰思考（Overthinking）」を誘発し、性能低下とトークン消費の増大を招きます。**D-CoT** は、`<TEMP_LOW>`（事実確認）や`<TEMP_HIGH>`（多角的探索）などの制御タグを学習時の補助的足場として用い、構造化された推論プロセスを強制するフレームワークです。CoT軌道を最適化することで、推論のドリフトを抑制し、トークン削減と性能向上を同時に実現します。

## 実験結果

**Qwen3-8B** を対象に、5K件の合成ORPOデータセット・単一GPU環境で学習：

| ベンチマーク | ベースモデル | D-CoT (LoRA) | 改善幅 |
|---|---|---|---|
| **MMLU-Pro** (0-shot) | 55.66% | **64.73%** | +9.1% |
| **GPQA-Diamond** (0-shot, 5-seed平均) | 43.03% | **52.93%** | +9.9% |

出力トークン数も**14〜63%削減**され、推論効率が大幅に向上しています。

## 概要

従来の蒸留CoTモデルは、無限検証ループ・不要な筆算模倣・正解後の自己疑念による答え書き換えといった**過剰思考**に陥りがちです。D-CoTは以下のアプローチでこれを解決します：

1. **制御タグによる思考の構造化** — `<TEMP_LOW>`（事実整理）、`<TEMP_HIGH>`（多角的探索）、`<TEMP_MID>`（論理的収束）
2. **ORPOによる選好学習** — 31種のRejectionカテゴリで過剰思考パターンを体系的に排除
3. **内在化** — モデルはタグそのものではなく、規律ある推論の「リズム」を学習

## リポジトリ構成

```
ごちゃごちゃしているので整理予定
```

## クイックスタート

### セットアップ
```bash
bash setup.sh
```

### 学習
```bash
bash run_train.sh
```

### 評価
```bash
# MMLU-Pro (0-shot, 全問)
python MMLU-pro-bench.py -m unsloth/Qwen3-8B -lp <lora_path> -up -ui

# GPQA-Diamond
python GPQA-diamond-bench.py -m unsloth/Qwen3-8B -lp <lora_path> -up -ui
```

## 日本語版論文

日本語版は **JSAI 2026（第40回人工知能学会全国大会）** に投稿済みです。J-STAGEでの公開は2026年5月頃になると思われます。

## リンク

- **論文 (arXiv):** [https://arxiv.org/abs/2602.21786](https://arxiv.org/abs/2602.21786)
- **コード:** [gitpullpull/DisciplinedChainOfThought](https://github.com/gitpullpull/DisciplinedChainOfThought)
- **ベンチマーク結果:** [gitpullpull/D-CoT-Benchmarks](https://huggingface.co/datasets/gitpullpull/D-CoT-Benchmarks)
- **データセット（5,181件のChosen/Rejectedペア）:** [gitpullpull/D-CoT-datasets](https://huggingface.co/datasets/gitpullpull/D-CoT-datasets)

## 連絡先

- **X (Twitter):** [@gitpullpull](https://x.com/gitpullpull)
- **Email:** gitpullpull@gmail.com

## 引用

```bibtex
@article{ubukata2026dcot,
  title={D-CoT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models},
  author={Ubukata, Shunsuke},
  journal={arXiv preprint arXiv:2602.21786},
  year={2026}
}
```
