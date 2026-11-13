#!/usr/bin/env bash
# 2026-11-08, for the quality table: SGLang 0.5.22 with the model of the speed comparison's SGLang "nv" rows (NVIDIA's
# NVFP4 checkpoint, KV fp8_e4m3, FlashInfer, mamba state bf16 as 20261101/launch/sg_dflash_nv.sh), for teacher-forced input log-probs only: no drafter, no CUDA
# graphs, no radix cache, context 4096.  Port 30000 (common.sh).
D="$(cd "$(dirname "$0")" && pwd)"; source "$D/../../../20261101/launch/common.sh"
exec $SG -m sglang.launch_server --model-path $M/nvidia_NVFP4 --served-model-name qwen3.8-27b --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --max-running-requests 1 --host 127.0.0.1 --port $PORT --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.85 --chunked-prefill-size 1024 --context-length 4096 --disable-cuda-graph --disable-radix-cache 2>&1 | tee -a $LOGD/tq_sg_nv.log
