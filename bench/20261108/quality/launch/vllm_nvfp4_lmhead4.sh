#!/usr/bin/env bash
# 2026-11-08, for the quality table: vLLM 0.31 with the model of the speed comparison's vLLM rows (gittensor NVFP4
# LMHead4, KV fp8 as there), for teacher-forced prompt log-probs only: no drafter, eager (no CUDA graph capture: faster start, same prefill
# math), no prefix caching (every prompt token gets its log-probs), max-logprobs 100, context 4096.  Port 30000 (common.sh).
D="$(cd "$(dirname "$0")" && pwd)"; source "$D/../../../20261101/launch/common.sh"
exec $VL/vllm serve $M/gittensor_NVFP4_LMHead4 --served-model-name qwen3.8-27b --max-model-len 4096 --kv-cache-dtype fp8 --language-model-only --max-num-seqs 1 --host 127.0.0.1 --port $PORT --max-logprobs 100 --enforce-eager --no-enable-prefix-caching 2>&1 | tee -a $LOGD/tq_vl_lm4.log
