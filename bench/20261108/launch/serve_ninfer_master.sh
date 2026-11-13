#!/usr/bin/env bash
# serve_ninfer_master.sh (2026-11-06): ninfer-serve from today's master checkout (<ninfer_src_dir>/ninfer_master), the same fastest
# single-request setting as serve_ninfer.sh (official NVFP4 artifact, DFlash 2 x7 + LM-head draft, FP8 KV, prefill chunk 1024,
# one request at a time, no prefix reuse; master renamed --kv-capacity -> --context-capacity and --lm-head-draft -> --proposal-head optimized). Port 18080. Logs to <log_dir>/ninfer-master-serve.log.
set -euo pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd <ninfer_src_dir>/ninfer_master
exec ./build/apps/ninfer-serve <model_dir>/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 135168 --context-capacity auto --max-concurrency 1 \
  --kv-dtype fp8 --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 7 --proposal-head optimized \
  --no-prefix-reuse --request-log-jsonl <log_dir>/ninfer-master-requests.jsonl \
  "$@" 2>&1 | tee -a <log_dir>/ninfer-master-serve.log
