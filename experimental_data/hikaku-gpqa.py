"""
hikaku-gpqa.py
GPQA版の汚染チェックパイプライン (MMLU版と同様の構造):
  - DPO JSONLとGPQA (diamond subset) をHF datasetsからロード
  - Embedding類似度チェック (RTX4090用に最適化)
  - 5-gram完全一致フィルター (転置インデックス)
  - CSV/プロットと手動チェックサンプル(100件)を生成
Usage:
  python3 hikaku-gpqa.py
"""

import os
import json
import time
import math
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

# -------------------------
# Config (edit as needed)
# -------------------------
DPO_PATH = "/home/owner/dynamic-temperature/re-gen/merged_dpo.jsonl"
GPQA_HF_NAME = "Idavidrein/gpqa"
GPQA_SUBSET = "gpqa_diamond"  # gpqa_diamond, gpqa_main, gpqa_extended
DEVICE = "cuda"  # RTX4090
# Embedding model preference
EMBED_MODELS_TRY = [
    "BAAI/bge-m3",
    "sentence-transformers/all-mpnet-base-v2"
]
EMBED_BATCH = 256  # RTX4090なら大きめでOK
SIM_THRESHOLD = 0.88   # 類似度閾値
TOP_K = 5              # 各DPOに対するトップマッチ数
N_MANUAL_CHECK = 100   # 手動チェック用サンプル数
RESULT_DIR = "./results_gpqa"

# 5-gram設定
NGRAM_N = 5

# Random seed
RND = 1234
random.seed(RND)
np.random.seed(RND)
torch.manual_seed(RND)

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------
# Load Embedding Model
# -------------------------
def load_embedder(prefer_list, device="cuda"):
    last_exc = None
    for mid in prefer_list:
        try:
            print(f"Trying to load embedding model: {mid}")
            model = SentenceTransformer(mid, device=device)
            print(f"Loaded model: {mid}")
            return model, mid
        except Exception as e:
            print(f"Failed to load {mid}: {e}")
            last_exc = e
    raise RuntimeError(f"Could not load any embedding model. Last error: {last_exc}")

# -------------------------
# Data loaders
# -------------------------
def load_dpo_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(obj)
            except Exception as e:
                print("Skipping invalid json line:", e)
    return out

def build_dpo_text(record: dict) -> str:
    d = record.get("data", {})
    parts = [
        d.get("user_prompt", ""),
        d.get("thought_chosen", ""),
        d.get("chosen_response", ""),
        d.get("thought_rejected", ""),
        d.get("rejected_response", ""),
    ]
    parts = [p.strip() for p in parts if p and p.strip()]
    return "\n".join(parts)

def build_gpqa_text(rec: dict) -> str:
    """
    GPQA dataset structure:
    - Question: 質問テキスト
    - Correct Answer: 正解
    - Incorrect Answer 1, 2, 3: 不正解
    - Subdomain: 分野 (e.g., "Organic Chemistry")
    """
    q = rec.get("Question", "")
    correct = rec.get("Correct Answer", "")
    inc1 = rec.get("Incorrect Answer 1", "")
    inc2 = rec.get("Incorrect Answer 2", "")
    inc3 = rec.get("Incorrect Answer 3", "")
    subdomain = rec.get("Subdomain", "")
    
    choices = [correct, inc1, inc2, inc3]
    choices = [c for c in choices if c]
    ch_text = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    
    return f"[{subdomain}]\n{q}\n{ch_text}\nAnswer: {correct}"

# -------------------------
# 5-gram inverted index
# -------------------------
def generate_ngrams_tokens(text: str, n: int) -> Set[str]:
    toks = text.split()
    if len(toks) < n:
        return set()
    s = set()
    for i in range(len(toks) - n + 1):
        s.add(" ".join(toks[i:i+n]))
    return s

def build_inverted_index(gpqa_texts: List[str], ngram_n: int):
    inv = defaultdict(set)
    print("Building inverted index for 5-grams...")
    for idx, t in enumerate(tqdm(gpqa_texts, desc="gpqa 5-grams")):
        grams = generate_ngrams_tokens(t, ngram_n)
        for g in grams:
            inv[g].add(idx)
    print(f"Inverted index size (unique {ngram_n}-grams): {len(inv)}")
    return inv

# -------------------------
# Main pipeline
# -------------------------
def main():
    start_time = time.time()
    out_dir = os.path.join(RESULT_DIR, timestamp())
    ensure_dir(out_dir)
    print("Results directory:", out_dir)

    # 1) Load datasets
    print(f"Loading GPQA ({GPQA_SUBSET}) from HF datasets...")
    ds_gpqa = load_dataset(GPQA_HF_NAME, GPQA_SUBSET, split="train")
    print(f"Loaded GPQA {GPQA_SUBSET}: {len(ds_gpqa)} samples")

    print("Loading DPO JSONL ...")
    dpo_raw = load_dpo_jsonl(DPO_PATH)
    print("Loaded DPO records:", len(dpo_raw))

    # 2) Build texts
    print("Building DPO texts ...")
    dpo_texts = []
    dpo_meta = []
    for rec in dpo_raw:
        t = build_dpo_text(rec)
        if t and t.strip():
            dpo_texts.append(t)
            dpo_meta.append(rec)
    print("Usable DPO texts:", len(dpo_texts))

    print("Building GPQA texts ...")
    gpqa_texts = [build_gpqa_text(x) for x in ds_gpqa]
    print("Total GPQA texts:", len(gpqa_texts))

    # 3) Load embedder
    embed_model, used_model_id = load_embedder(EMBED_MODELS_TRY, device=DEVICE)
    print("Using embedding model:", used_model_id)
    emb_dim = embed_model.get_sentence_embedding_dimension()
    print("Embedding dimension:", emb_dim)

    # 4) Embedding compute (batched, RTX4090最適化)
    def batch_encode(texts: List[str], batch_size: int):
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
            batch = texts[i:i+batch_size]
            emb = embed_model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            out.append(emb)
        return torch.cat(out, dim=0)

    print("Encoding DPO texts ...")
    emb_dpo = batch_encode(dpo_texts, EMBED_BATCH).to(DEVICE)
    print("Encoding GPQA texts ...")
    emb_gpqa = batch_encode(gpqa_texts, EMBED_BATCH).to(DEVICE)

    # 5) Similarity block computation (メモリ効率化)
    print("Computing similarity (blockwise) ...")
    N_d = emb_dpo.size(0)
    N_g = emb_gpqa.size(0)
    topk_vals = []
    topk_idx = []
    max_sims = np.zeros(N_d, dtype=float)

    block_m = 1024  # RTX4090なら大きめでOK
    with torch.no_grad():
        for start_m in range(0, N_g, block_m):
            end_m = min(N_g, start_m + block_m)
            block = emb_gpqa[start_m:end_m]
            sim_block = torch.matmul(emb_dpo, block.T)
            sim_block = sim_block.cpu()
            if start_m == 0:
                vals, idxs = torch.topk(sim_block, k=min(TOP_K, sim_block.size(1)), dim=1)
                idxs = idxs + start_m
                topk_vals = vals.numpy()
                topk_idx = idxs.numpy()
            else:
                vals_new, idxs_new = torch.topk(sim_block, k=min(TOP_K, sim_block.size(1)), dim=1)
                idxs_new = (idxs_new + start_m).numpy()
                vals_new = vals_new.numpy()
                merged_vals = np.hstack([topk_vals, vals_new])
                merged_idx = np.hstack([topk_idx, idxs_new])
                topk_order = np.argsort(-merged_vals, axis=1)[:, :TOP_K]
                rows = np.arange(N_d)[:, None]
                topk_vals = merged_vals[rows, topk_order]
                topk_idx = merged_idx[rows, topk_order]
            block_max = sim_block.max(dim=1).values.numpy()
            max_sims = np.maximum(max_sims, block_max)

    print("Top-k computed. Calculating summary stats...")
    topk_vals_list = topk_vals.tolist()
    topk_idx_list = topk_idx.tolist()

    # 6) 5-gram inverted index
    inv_idx = build_inverted_index(gpqa_texts, NGRAM_N)

    # 7) 5-gram matches
    print("Checking 5-gram exact overlaps...")
    dpo_5gram_matches = [set() for _ in range(len(dpo_texts))]
    for i, t in enumerate(tqdm(dpo_texts, desc="dpo 5-grams")):
        grams = generate_ngrams_tokens(t, NGRAM_N)
        if not grams:
            continue
        candidates = set()
        for g in grams:
            if g in inv_idx:
                candidates.update(inv_idx[g])
        dpo_5gram_matches[i] = candidates

    # 8) Contamination判定
    print("Applying contamination rules...")
    rows = []
    for i in range(len(dpo_texts)):
        row = {
            "dpo_idx": i,
            "dpo_text_snippet": dpo_texts[i][:400].replace("\n", " "),
            "best_gpqa_idx": int(topk_idx_list[i][0]),
            "best_sim": float(topk_vals_list[i][0]),
            "topk_gpqa_idx": [int(x) for x in topk_idx_list[i]],
            "topk_sim": [float(x) for x in topk_vals_list[i]],
            "max_sim_all": float(max_sims[i]),
            "5gram_matched_count": len(dpo_5gram_matches[i]),
            "5gram_matched_examples": list(sorted(list(dpo_5gram_matches[i]))[:5])
        }
        row["A_flag"] = row["best_sim"] >= SIM_THRESHOLD
        row["B_flag"] = row["5gram_matched_count"] > 0
        row["contamination_score"] = (
            0.7 * min(1.0, row["best_sim"]) +
            0.3 * (1.0 if row["B_flag"] else 0.0)
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "contamination_report.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    # 9) Plots
    plt.figure(figsize=(8,5))
    plt.hist(df["max_sim_all"], bins=50)
    plt.title("Histogram of max similarity per DPO -> GPQA")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    hist_path = os.path.join(out_dir, "hist_max_similarity.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print("Saved histogram:", hist_path)

    df_sorted = df.sort_values("contamination_score", ascending=False).reset_index(drop=True)
    topN = min(50, len(df_sorted))
    plt.figure(figsize=(10,6))
    plt.bar(range(topN), df_sorted["contamination_score"].values[:topN])
    plt.title(f"Top contamination_score (top {topN})")
    plt.xlabel("DPO rank")
    plt.ylabel("contamination_score")
    bar_path = os.path.join(out_dir, "top_contamination_scores.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print("Saved bar chart:", bar_path)

    # 10) Manual-check sample
    flagged = df[(df["A_flag"]) | (df["B_flag"])]
    if len(flagged) >= N_MANUAL_CHECK:
        sample_manual = flagged.sample(N_MANUAL_CHECK, random_state=RND)
    else:
        sample_manual = flagged.copy()
        remaining = df.loc[~df.index.isin(sample_manual.index)].sort_values("contamination_score", ascending=False)
        need = N_MANUAL_CHECK - len(sample_manual)
        if need > 0:
            sample_manual = pd.concat([sample_manual, remaining.head(need)])

    manual_path = os.path.join(out_dir, "manual_check_sample.jsonl")
    with open(manual_path, "w", encoding="utf-8") as fw:
        for _, r in sample_manual.iterrows():
            obj = {
                "dpo_idx": int(r["dpo_idx"]),
                "dpo_text": dpo_texts[int(r["dpo_idx"])],
                "topk_gpqa_idx": r["topk_gpqa_idx"],
                "topk_sim": r["topk_sim"],
                "5gram_matches": r["5gram_matched_examples"],
                "contamination_score": float(r["contamination_score"]),
                "A_flag": bool(r["A_flag"]),
                "B_flag": bool(r["B_flag"]),
            }
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("Saved manual-check sample:", manual_path)

    # 11) Top matches full CSV
    rows_full = []
    for _, r in df_sorted.head(500).iterrows():
        i = int(r["dpo_idx"])
        bests = r["topk_gpqa_idx"]
        sims = r["topk_sim"]
        for m_idx, simscore in zip(bests, sims):
            rows_full.append({
                "dpo_idx": i,
                "dpo_text_snippet": dpo_texts[i][:400].replace("\n", " "),
                "gpqa_idx": int(m_idx),
                "gpqa_text_snippet": gpqa_texts[int(m_idx)][:400].replace("\n", " "),
                "similarity": float(simscore),
                "A_flag": bool(r["A_flag"]),
                "B_flag": bool(r["B_flag"]),
                "contamination_score": float(r["contamination_score"])
            })
    df_full = pd.DataFrame(rows_full)
    full_csv_path = os.path.join(out_dir, "top_matches_full.csv")
    df_full.to_csv(full_csv_path, index=False)
    print("Saved top matches CSV:", full_csv_path)

    # 12) Summary
    total = len(df)
    flagged_count = df[(df["A_flag"]) | (df["B_flag"])].shape[0]
    perc_flagged = flagged_count / total * 100
    summary = {
        "timestamp": timestamp(),
        "dataset": f"GPQA ({GPQA_SUBSET})",
        "total_gpqa_samples": len(gpqa_texts),
        "total_dpo": total,
        "flagged_count": int(flagged_count),
        "percent_flagged": perc_flagged,
        "avg_max_similarity": float(df["max_sim_all"].mean()),
        "median_max_similarity": float(df["max_sim_all"].median()),
        "top1_max_similarity": float(df["max_sim_all"].max()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)
    print("Saved summary:", summary)

    elapsed = time.time() - start_time
    print(f"All done in {elapsed/60:.2f} minutes. Results in {out_dir}")

if __name__ == "__main__":
    main()