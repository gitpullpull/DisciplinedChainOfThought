import torch
import logging
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import ORPOTrainer, ORPOConfig  # DPOからORPOへ変更
from datasets import load_dataset
import os
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. ロギング設定
# =================================================================
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# =================================================================
# 2. モデル読み込み
# =================================================================
model_name = "Unsloth/qwen3-8b"
max_seq_length = 4096
output_dir = "./output_orpo_lion_base4"
os.makedirs(output_dir, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    bias = "none",
    lora_dropout = 0.05,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# =================================================================
# 3. データセット
# =================================================================
dataset_file = "cleaned_dpo.jsonl" 
dataset = load_dataset("json", data_files=dataset_file, split="train")

def filter_bad_rows(example):
    try:
        row = example.get("data", example)
        if row is None: return False
        if not isinstance(row.get("user_prompt"), str): return False
        if row.get("chosen_response") == row.get("rejected_response"): return False
        return True
    except: return False

dataset = dataset.filter(filter_bad_rows)

def format_orpo_data(example):
    row = example.get("data", example)
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["user_prompt"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    # ORPOもDPOと同じ形式(prompt, chosen, rejected)を使用
    formatted_chosen = f"<think>\n{row['thought_chosen']}\n</think>\n{row['chosen_response']}"
    formatted_rejected = f"<think>\n{row['thought_rejected']}\n</think>\n{row['rejected_response']}"
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
    }

formatted_dataset = dataset.map(format_orpo_data)

# =================================================================
# 4. ORPOトレーニング設定
# =================================================================

orpo_config = ORPOConfig(
    output_dir = output_dir,
    
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,

    # --- Lion設定 ---
    optim = "paged_lion_8bit",
    learning_rate = 1e-6, 
    weight_decay = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    
    # --- ORPO固有設定 ---
    beta = 0.1,                # Odds Ratioの重み（重要）
    # --------------------
    
    num_train_epochs = 1,
    max_steps = -1,            # エポック数ベース
    warmup_ratio = 0.1,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    lr_scheduler_type = "cosine",
    seed = 3407,
    report_to = "tensorboard",
    max_length = max_seq_length,
    max_prompt_length = max_seq_length // 2,
)

trainer = ORPOTrainer(
    model = model,
    args = orpo_config,
    train_dataset = formatted_dataset,
    tokenizer = tokenizer,
)

# =================================================================
# 5. 実行 & 保存
# =================================================================
print(f"Lion 8bit ORPOトレーニングを開始します (LR: {orpo_config.learning_rate})...")
trainer.train()

print("保存中...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ログ保存
log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

# 可視化 (ORPOのログ名に対応)
try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    loss_data = log_df.dropna(subset=["loss"])
    plt.plot(loss_data["step"], loss_data["loss"], label="Total Loss", color="blue", alpha=0.7)
    plt.title("ORPO Total Loss")
    plt.grid(True, linestyle='--')
    
    plt.subplot(1, 2, 2)
    # ORPO特有のメトリクス（必要に応じて変更してください）
    if "log_odds_ratio" in log_df.columns:
        plt.plot(log_df["step"], log_df["log_odds_ratio"], label="Log Odds Ratio", color="purple")
        plt.title("Log Odds Ratio")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plots.png"))
except Exception:
    pass

print("完了。")