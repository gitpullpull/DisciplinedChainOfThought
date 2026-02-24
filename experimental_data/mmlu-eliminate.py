#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_clean_dataset_v2.py
修正版: 誤検知を防ぐため閾値を0.6に緩和し、変数名バグを修正。
"""

import os
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
# ★ パスは環境に合わせてください
DPO_PATH = "/home/owner/dynamic-temperature/re-gen/merged_orpo5k.jsonl"
MMLU_HF_NAME = "TIGER-Lab/MMLU-Pro"
RESULT_DIR = "./clean_results"

# 閾値設定
SIM_THRESHOLD = 0.55
NGRAM_N = 13

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODELS_TRY = [
    "BAAI/bge-m3-small",
    "voyager/embedding-voyage-3-large", 
    "sentence-transformers/all-mpnet-base-v2"
]
EMBED_BATCH = 256
RND = 1234

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_embedder(prefer_list, device):
    for mid in prefer_list:
        try:
            print(f"Loading embedding model: {mid}")
            model = SentenceTransformer(mid, device=device)
            return model, mid
        except Exception as e:
            print(f"Failed {mid}: {e}")
    raise RuntimeError("Could not load any embedding model.")

def load_dpo_jsonl(path):
    out = []
    if not os.path.exists(path):
        print(f"Error: File not found -> {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except:
                    pass
    return out

def build_dpo_text(record):
    d = record.get("data", {})
    parts = [
        d.get("user_prompt", ""),
        d.get("thought_chosen", ""),
        d.get("chosen_response", ""),
        d.get("rejected_response", "") 
    ]
    return "\n".join([p.strip() for p in parts if p and p.strip()])

def build_mmlu_text(rec):
    q = rec.get("question", "")
    choices = rec.get("choices", [])
    ch_str = "\n".join(choices) if choices else ""
    return f"{q}\n{ch_str}\nAnswer: {rec.get('answer', '')}"

def generate_ngrams(text, n):
    toks = text.split()
    if len(toks) < n: return set()
    return set(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))

# -------------------------
# Main Logic
# -------------------------
def main():
    start_time = time.time()
    out_dir = os.path.join(RESULT_DIR, timestamp())
    ensure_dir(out_dir)
    print(f"Output directory: {out_dir}")

    # 1. Load Data
    print("Loading datasets...")
    try:
        ds_test = load_dataset(MMLU_HF_NAME, split="test")
        ds_valid = load_dataset(MMLU_HF_NAME, split="validation")
        mmlu_texts = [build_mmlu_text(x) for x in ds_test] + [build_mmlu_text(x) for x in ds_valid]
    except Exception as e:
        print(f"Failed to load MMLU: {e}")
        return

    dpo_raw = load_dpo_jsonl(DPO_PATH)
    if not dpo_raw:
        print("No DPO data loaded. Exiting.")
        return

    dpo_texts = [build_dpo_text(r) for r in dpo_raw]
    print(f"DPO count: {len(dpo_texts)}, MMLU count: {len(mmlu_texts)}")

    # 2. Embedding Similarity
    model, model_name = load_embedder(EMBED_MODELS_TRY, DEVICE)
    
    print("Encoding texts...")
    dpo_emb = model.encode(dpo_texts, batch_size=EMBED_BATCH, convert_to_tensor=True, normalize_embeddings=True)
    mmlu_emb = model.encode(mmlu_texts, batch_size=EMBED_BATCH, convert_to_tensor=True, normalize_embeddings=True)

    print("Computing Max Similarity...")
    max_sims = torch.zeros(len(dpo_texts), device="cpu")
    dpo_emb = dpo_emb.cpu()
    mmlu_emb = mmlu_emb.cpu()
    
    block_size = 1024
    for i in tqdm(range(0, len(mmlu_emb), block_size)):
        end = min(len(mmlu_emb), i + block_size)
        block = mmlu_emb[i:end]
        sims = torch.mm(dpo_emb, block.T)
        batch_max, _ = sims.max(dim=1)
        max_sims = torch.max(max_sims, batch_max)

    max_sims = max_sims.numpy()

    # 3. N-gram Exact Match
    print("Checking N-grams...")
    inv_index = defaultdict(set)
    for idx, txt in enumerate(mmlu_texts):
        for g in generate_ngrams(txt, NGRAM_N):
            inv_index[g].add(idx)
            
    ngram_flags = []
    for txt in tqdm(dpo_texts):
        grams = generate_ngrams(txt, NGRAM_N)
        match = any(g in inv_index for g in grams)
        ngram_flags.append(match)

    # 4. Filtering & Saving
    clean_path = os.path.join(out_dir, "clean_dpo.jsonl")
    removed_path = os.path.join(out_dir, "removed_contamination.jsonl")
    report_path = os.path.join(out_dir, "contamination_report.csv")

    clean_count = 0
    removed_count = 0
    report_rows = []

    print(f"Filtering (Threshold >= {SIM_THRESHOLD})...")
    with open(clean_path, "w", encoding="utf-8") as f_clean, \
         open(removed_path, "w", encoding="utf-8") as f_removed:
        
        for i, raw_rec in enumerate(dpo_raw):
            sim = float(max_sims[i])
            is_ngram_match = ngram_flags[i]
            
            # 判定: 類似度0.6以上 OR N-gram一致
            is_contaminated = (sim >= SIM_THRESHOLD) or is_ngram_match
            
            row_stat = {
                "dpo_idx": i,
                "max_similarity": sim,
                "ngram_match": is_ngram_match,
                "status": "REMOVED" if is_contaminated else "KEPT",
                "text_snippet": dpo_texts[i][:100].replace("\n", " ")
            }
            report_rows.append(row_stat)

            if is_contaminated:
                f_removed.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                removed_count += 1
            else:
                f_clean.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                clean_count += 1

    # Save CSV Report
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    # 5. Plot for Paper
    plt.figure(figsize=(8, 5))
    plt.hist(max_sims, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='All Data')
    plt.axvline(x=SIM_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Cutoff ({SIM_THRESHOLD})')
    plt.title(f"Distribution of Similarity (Removed {removed_count} items)")
    plt.xlabel("Cosine Similarity to MMLU-Pro")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "similarity_distribution.png"), dpi=150)

    print("="*40)
    print(f"DONE! Results saved in: {out_dir}")
    print(f" - Clean Data:   {clean_count} items -> {clean_path}")
    print(f" - Removed Data: {removed_count} items -> {removed_path}") # 変数名修正済み
    if (clean_count + removed_count) > 0:
        print(f" - Removal Rate: {removed_count / (clean_count + removed_count) * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()