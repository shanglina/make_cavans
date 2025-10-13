#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 激活虚拟环境
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  PY=".venv/bin/python"
elif [[ -f ".venv/Scripts/activate" ]]; then
  source .venv/Scripts/activate
  PY=".venv/Scripts/python.exe"
else
  echo "[INFO] 创建虚拟环境..."
  python3 -m venv .venv
  source .venv/bin/activate
  PY=".venv/bin/python"
fi

# 无论是否已安装，始终确保依赖满足
echo "[INFO] 安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 打印调试信息
"$PY" -V
"$PY" -m pip list

# 执行脚本
"$PY" auto_sheet_layout_transparent_300dpi.py --config layout_config.json