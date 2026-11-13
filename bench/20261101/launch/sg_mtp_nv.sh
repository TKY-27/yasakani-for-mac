#!/usr/bin/env bash
# sg_mtp_nv (generated 2026-10-18)
source "$(dirname "$0")/common.sh"
exec $SG -m sglang.launch_server --model-path $M/nvidia_NVFP4 --served-model-name qwen3.8-27b --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --max-running-requests 1 --cuda-graph-max-bs-decode 1 --reasoning-parser qwen3 --host 127.0.0.1 --port $PORT --mem-fraction-static 0.94 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --disable-radix-cache "$@" 2>&1 | tee -a $LOGD/sg_mtp_nv.log
