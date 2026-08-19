#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/hf-cache

exec /workspace/muscriptor-venv/bin/muscriptor serve \
  --model large \
  --device cuda \
  --host 127.0.0.1 \
  --port 8222
