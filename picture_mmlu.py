import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans' # Fallback for compatibility

# Define file paths and readable labels based on filenames
files = {
    "Run_UP_Templock": "Qwen3-8B_run_20251224_170022_UP_Templock_0shot_analysis.csv",
    "Run_UP_IT": "Qwen3-8B_run_20251224_170022_UP_IT_0shot_analysis.csv",
    "Run_NP_Templock": "Qwen3-8B_run_20251224_170022_NP_Templock_0shot_analysis.csv",
    "Base_UP_Templock": "Qwen3-8B_base_UP_Templock_0shot_analysis.csv",
    "Base_UP_IT": "Qwen3-8B_base_UP_IT_0shot_analysis.csv",
    "Base_NP_Templock": "Qwen3-8B_base_NP_Templock_0shot_analysis.csv"
}

data_frames = []

for label, filename in files.items():
    df = pd.read_csv(filename)
    
    # Extract metadata from label
    model_type = "D-CoT (Run)" if "Run" in label else "Baseline"
    method = "Templock" if "Templock" in label else "IT (Fixed)"
    prompt = "UP" if "UP" in label else "NP"
    
    # Calculate weighted metrics
    total_count = df['count'].sum()
    weighted_acc = (df['accuracy'] * df['count']).sum() / total_count
    weighted_tokens = (df['avg_tokens'] * df['count']).sum() / total_count
    
    # Store summary
    data_frames.append({
        "Label": label,
        "Model": model_type,
        "Method": method,
        "Prompt": prompt,
        "Accuracy": weighted_acc,
        "Avg_Tokens": weighted_tokens,
        "Total_Samples": total_count,
        "Raw_DF": df
    })

summary_df = pd.DataFrame(data_frames)

# sort by accuracy descending
summary_df = summary_df.sort_values(by="Accuracy", ascending=False)

# Display Summary Table
print("Summary of Performance:")
print(summary_df[['Model', 'Method', 'Prompt', 'Accuracy', 'Avg_Tokens']].to_markdown(index=False, floatfmt=".4f"))

# --- Visualizations ---

# 1. Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Accuracy", y="Label", hue="Model", palette="viridis")
plt.title("Overall Accuracy Comparison (Weighted Average)")
plt.xlabel("Accuracy")
plt.xlim(0.3, 0.7) # Adjust based on data range
plt.savefig("accuracy_comparison.png")

# 2. Token Usage vs Accuracy (Efficiency)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=summary_df, x="Avg_Tokens", y="Accuracy", hue="Model", style="Method", s=200, palette="deep")
for i in range(summary_df.shape[0]):
    plt.text(
        summary_df.Avg_Tokens.iloc[i]+20, 
        summary_df.Accuracy.iloc[i], 
        summary_df.Label.iloc[i].replace("Qwen3-8B_", ""), 
        fontsize=9
    )
plt.title("Efficiency Frontier: Accuracy vs. Average Tokens")
plt.xlabel("Average Token Count")
plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("efficiency_scatter.png")

# 3. Category Breakdown (Comparison between Best D-CoT and Best Baseline)
# Identify best run and best base based on UP_IT or UP_Templock (User Prompt usually standard)
best_run = summary_df[summary_df['Model'] == 'D-CoT (Run)'].iloc[0]
best_base = summary_df[summary_df['Model'] == 'Baseline'].iloc[0]

df_run = best_run['Raw_DF'].copy()
df_run['Type'] = 'D-CoT (Best)'
df_base = best_base['Raw_DF'].copy()
df_base['Type'] = 'Baseline (Best)'

combined_cat = pd.concat([df_run, df_base])

plt.figure(figsize=(12, 6))
sns.barplot(data=combined_cat, x="category", y="accuracy", hue="Type", palette="muted")
plt.title(f"Category Performance: {best_run['Label']} vs {best_base['Label']}")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.savefig("category_breakdown.png")

# 4. Token Reduction by Category
combined_cat['token_diff'] = 0
categories = df_run['category'].unique()
diff_data = []

for cat in categories:
    run_tokens = df_run[df_run['category'] == cat]['avg_tokens'].values[0]
    base_tokens = df_base[df_base['category'] == cat]['avg_tokens'].values[0]
    run_acc = df_run[df_run['category'] == cat]['accuracy'].values[0]
    base_acc = df_base[df_base['category'] == cat]['accuracy'].values[0]
    
    diff_data.append({
        "category": cat,
        "Token_Reduction": base_tokens - run_tokens,
        "Accuracy_Gain": run_acc - base_acc
    })

df_diff = pd.DataFrame(diff_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_diff, x="Token_Reduction", y="Accuracy_Gain", hue="category", s=150)
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.title("Token Reduction vs Accuracy Gain per Category")
plt.xlabel("Token Reduction (Base - D-CoT) [Positive = D-CoT used fewer tokens]")
plt.ylabel("Accuracy Gain (D-CoT - Base)")
plt.savefig("token_vs_acc_gain.png")
