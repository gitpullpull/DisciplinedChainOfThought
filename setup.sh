#!/bin/bash

# 1. ライブラリのインストール
echo "--- Installing python libraries ---"
pip install pandas matplotlib huggingface_hub

apt-get install git-lfs -y
git lfs install

# 2. リポジトリをクローン (カレントディレクトリ内にフォルダが作成されます)
echo "--- Cloning the repository ---"
git clone https://huggingface.co/gitpullpull/Introspective_Temperature_test

echo "--- Setup Complete ---"
ls -ld Introspective_Temperature_test