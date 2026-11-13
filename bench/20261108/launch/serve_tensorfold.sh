#!/usr/bin/env bash
# 2026-10-15: TensorFold 0.6.6 + DFlash 2 for Qwen3.8-27B on the RTX 5090, the configuration of a public post about
# TensorFold 0.6.6 + DFlash2 (MLX 4-bit). CUDA backend, one request at a time
# (CUDA --parallel auto = 1), prompt cache off (--prompt-cache-gib 0: every request starts from a fresh prompt, as on our server and
# NInfer with --no-prefix-reuse; decode speed is unaffected), context 69,632 (the 65,536-token reasoning budget + prompt room).
# Port 18090. Run as root in WSL2; logs to <log_dir>.
set -euo pipefail
. <tensorfold_venv>/bin/activate
cd <tensorfold_dir>
exec tensorfold serve <model_dir>/TensorFold/Qwen3.8-27B-MLX-4bit \
  --drafter <model_dir>/z-lab/Qwen3.8-27B-DFlash2 \
  --backend cuda --host 127.0.0.1 --port 18090 --name qwen3.8-27b \
  --context 69632 --prompt-cache-gib 0 --no-update-check \
  "$@" 2>&1 | tee -a <log_dir>/tensorfold-serve.log
