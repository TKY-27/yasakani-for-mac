#!/usr/bin/env bash
# vl_dspark_lm4 (generated 2026-10-18)
source "$(dirname "$0")/common.sh"
exec $VL/vllm serve $M/gittensor_NVFP4_LMHead4 --served-model-name qwen3.8-27b --max-model-len 16384 --kv-cache-dtype fp8 --reasoning-parser qwen3 --language-model-only --max-num-seqs 1 --host 127.0.0.1 --port $PORT --speculative-config '{"method":"dspark","model":"'$M'/redhat_speculator_dspark","num_speculative_tokens":8}' "$@" 2>&1 | tee -a $LOGD/vl_dspark_lm4.log
