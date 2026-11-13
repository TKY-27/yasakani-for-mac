// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include "pm_math.cuh"


namespace yskn {
__device__ __forceinline__ unsigned long long kh_mix(unsigned long long z) {   
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }


__device__ __forceinline__ double kh_gumbel(unsigned long long seed, long long pos, int v, int pm) {
  const unsigned long long h = kh_mix(kh_mix(kh_mix(seed) ^ (unsigned long long)pos) ^ ((unsigned long long)(unsigned)v * 0xD6E8FEB86659FD93ull));
  const double u = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  return pm ? -pm_log_d(-pm_log_d(u)) : -log(-log(u));
}
}  
