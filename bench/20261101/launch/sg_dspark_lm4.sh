#!/usr/bin/env bash
# sg_dspark_lm4 (generated 2026-10-18)
source "$(dirname "$0")/common.sh"
exec $SG -m sglang.launch_server --model-path $M/gittensor_NVFP4_LMHead4 --served-model-name qwen3.8-27b --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --max-running-requests 1 --cuda-graph-max-bs-decode 1 --reasoning-parser qwen3 --host 127.0.0.1 --port $PORT --mem-fraction-static 0.91 --chunked-prefill-size 1024 --mamba-full-memory-ratio 6.94 --max-total-tokens 16384 --speculative-algorithm DSPARK --speculative-draft-model-path $M/gittensor_DSpark_NVFP4 --speculative-draft-attention-backend flashinfer --disable-radix-cache "$@" 2>&1 | tee -a $LOGD/sg_dspark_lm4.log
