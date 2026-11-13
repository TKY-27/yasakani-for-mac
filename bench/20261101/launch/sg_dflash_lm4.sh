#!/usr/bin/env bash
# sg_dflash_lm4 (generated 2026-10-18)
source "$(dirname "$0")/common.sh"
exec $SG -m sglang.launch_server --model-path $M/gittensor_NVFP4_LMHead4 --served-model-name qwen3.8-27b --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --max-running-requests 1 --cuda-graph-max-bs-decode 1 --reasoning-parser qwen3 --host 127.0.0.1 --port $PORT --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.91 --chunked-prefill-size 1024 --speculative-algorithm DFLASH --speculative-draft-model-path $DF2 --speculative-num-draft-tokens 8 --disable-radix-cache "$@" 2>&1 | tee -a $LOGD/sg_dflash_lm4.log
