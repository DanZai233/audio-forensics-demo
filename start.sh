#!/bin/sh
set -eu
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo '请先安装 ffmpeg'; exit 1; }
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
if [ ! -f models/detector/model.safetensors ] || [ ! -f models/comparison/model.safetensors ]; then .venv/bin/python download_models.py; fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
