#!/usr/bin/env bash
# Start ninfer-serve with NInfer's own fastest single-request setting for Qwen3.8-27B on the RTX 5090 (2026-10-13): the official NVFP4
# artifact, DFlash 2 with 7 draft tokens and the LM-head draft (docs/performance/qwen3.8-27b.md "dflash2_7" corpus runs; model card flags
# --spec dflash2 --draft-tokens 7 --lm-head-draft), FP8 KV cache (their measured setting), prefill chunk 1024, one request at a time.
# --no-prefix-reuse: every request starts from a fresh prompt, as on our server (the three seeds of one prompt would otherwise reuse its
# prefix; decode speed is unaffected). Context 135,168 = the AIME budget 131,072 + prompt room. Port 18080 (as their eval scripts). Run as root in WSL2; logs to <log_dir>.
set -euo pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd <ninfer_src_dir>/ninfer
exec ./build/apps/ninfer-serve <model_dir>/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 135168 --kv-capacity auto --max-concurrency 1 \
  --kv-dtype fp8 --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --no-prefix-reuse --request-log-jsonl <log_dir>/ninfer-requests.jsonl \
  "$@" 2>&1 | tee -a <log_dir>/ninfer-serve.log
