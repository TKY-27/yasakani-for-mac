// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once


namespace yskn {
constexpr int KZ_NONE = -32768, KZ_CUT = 100;
__device__ __forceinline__ float kz_p2(int k) {                                        
  if (k >= -126) return k > 127 ? __int_as_float(0x7f800000) : __int_as_float((k + 127) << 23);
  return k >= -149 ? __int_as_float(1 << (k + 149)) : 0.f;
}


__device__ __forceinline__ void kz_ref(int& R, int X, float& rescale, float& vscale, bool& use) {
  rescale = 1.f;
  if (R == KZ_NONE) R = X;
  else if (X > R) { rescale = X - R > KZ_CUT ? 0.f : kz_p2(R - X); R = X; }
  use = R - X <= KZ_CUT; vscale = use ? kz_p2(X - R) : 0.f;
}
}  
