#!/bin/bash

# エラーがあれば即停止
set -e

# ==========================================
# 設定: ログファイル名の重複回避
# ==========================================
# 実行時のタイムスタンプを取得 (例: 20251218_0900)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_log_${TIMESTAMP}.txt"

echo "=== Job Started at ${TIMESTAMP} ==="
echo "Logs will be saved to: ${LOG_FILE}"

# ==========================================
# 1. 依存ライブラリのインストール
# ==========================================
echo "[1/2] Installing dependencies..."
# -q でログを静かにしていますが、エラー時は表示されます
pip install pandas matplotlib huggingface_hub

# ==========================================
# 2. Pythonスクリプトの実行
# ==========================================
echo "[2/2] Starting Train-ORPO.py..."

# 標準出力(画面)とログファイルの両方に出力(| tee)
# unbuffered(-u)にしてリアルタイムでログが出るようにする
python -u Train-ORPO.py 2>&1 | tee "$LOG_FILE"

echo "=== Job Finished ==="