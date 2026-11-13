#!/bin/bash
# ninfer_master_build.sh (2026-11-06; WSL2 Ubuntu-24.04 as root): a SECOND NInfer checkout at
# today's master (<ninfer_src_dir>/ninfer_master; the October build in <ninfer_src_dir>/ninfer stays as it is), built as build_ninfer.sh did
# (CUDA 13.1, Release, apps on). master's README now lists hwloc and liburing as dependencies: their -dev packages are installed if missing.
set -u
export DEBIAN_FRONTEND=noninteractive PATH=/usr/local/cuda-13.1/bin:$PATH CUDACXX=/usr/local/cuda-13.1/bin/nvcc
echo "[$(date +%T)] deps"
for p in libhwloc-dev liburing-dev libjpeg-dev libpng-dev libturbojpeg0-dev; do dpkg -s $p >/dev/null 2>&1 || { apt-get update -qq >/dev/null; apt-get install -y -qq --no-install-recommends $p 2>&1 | tail -1; }; done
pkg-config --modversion hwloc liburing libavformat libcurl 2>&1 | tr '\n' ' '; echo
cd <ninfer_src_dir> || exit 1
[ -d ninfer_master/.git ] || git clone -q https://github.com/Neroued/ninfer.git ninfer_master
cd ninfer_master && git fetch -q origin && git checkout -q master && git pull -q --ff-only
git log -1 --format='[build] source %H %cd %s' --date=iso
echo "[$(date +%T)] configure"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DCMAKE_CUDA_COMPILER=$CUDACXX 2>&1 | grep -vE "^-- (Detecting|Check for|Performing|Looking)" | tail -25
echo "[$(date +%T)] build"
cmake --build build --parallel 14 --target ninfer ninfer-serve 2>&1 | tail -8
ls -la build/apps/ninfer-serve 2>&1
echo "[$(date +%T)] serve --help (spec / kv / prefix lines)"
./build/apps/ninfer-serve --help 2>&1 | grep -iE "spec|draft|kv-dtype|prefix|prefill-chunk|lm-head|max-context|concurrency|host|port" | head -30
echo "[$(date +%T)] ninfer_master_build done"
