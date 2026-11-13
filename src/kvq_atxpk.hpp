// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once







#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace yskn {
constexpr size_t KVQAXPK_REC = 2048;                               
constexpr int KVQAXPK_NGRP = 128;                                  
inline float kvqaxpk_h2f(unsigned h) {
  const unsigned s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u; float v;
  if (e == 0) v = std::ldexp((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = std::ldexp((float)(m | 1024u), (int)e - 25);
  return s ? -v : v;
}
inline uint16_t kvqaxpk_f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
inline bool kvqaxpk_build(const unsigned char* b, size_t n, std::vector<unsigned char>& out) {
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (n < 100) return false;
  const unsigned ngrp = u32(28), off_g = u32(40), off_t = u32(36);
  if (ngrp != (unsigned)KVQAXPK_NGRP) return false;
  out.assign((size_t)ngrp * KVQAXPK_REC + ngrp, 0);
  for (unsigned g = 0; g < ngrp; ++g) {
    const size_t go = u32(off_g + 4 * (size_t)g);
    if (go + 8 > n) return false;
    const int mode = b[go], tix = b[go + 2], has_mu = b[go + 5];
    if (mode != 0) continue;
    const size_t to = u32(off_t + 4 * (size_t)tix);
    if (to + 12 + 512 > n) return false;
    float gain; std::memcpy(&gain, b + to + 4, 4);
    unsigned char* rec = out.data() + (size_t)g * KVQAXPK_REC;
    float cf[256]; float cmax = 0.f;
    for (int i = 0; i < 256; ++i) {
      const int ci = (int)(short)u16(to + 12 + 2 * (size_t)i);
      const float c = ((float)ci * 0.00390625f) * gain;
      const uint16_t h = kvqaxpk_f2bf(c); std::memcpy(rec + 2 * i, &h, 2);
      cf[i] = c; if (std::fabs(c) > cmax) cmax = std::fabs(c);
    }
    const float st = cmax > 0.f ? cmax / 127.f : 1.f;                  
    for (int i = 0; i < 256; ++i) { int q = (int)std::lrint(cf[i] / st); if (q > 127) q = 127; if (q < -127) q = -127; rec[1536 + i] = (unsigned char)(signed char)q; }
    std::memcpy(rec + 1792, &st, 4);
    if (has_mu) {
      if (go + 8 + 512 > n) return false;
      for (int i = 0; i < 256; ++i) { const float m = kvqaxpk_h2f(u16(go + 8 + 2 * (size_t)i)); std::memcpy(rec + 512 + 4 * i, &m, 4); }
      out[(size_t)ngrp * KVQAXPK_REC + g] = 1;
    }
  }
  return true;
}

inline bool kvqaxpk_packable(const unsigned char* b, size_t n, int layer, int kind, int kb = 4) {
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (n < 100 || layer < 0 || layer > 15) return false;
  const unsigned ngrp = u32(28), off_g = u32(40);
  for (int h = 0; h < 4; ++h) {
    const unsigned g = (unsigned)((layer * 4 + h) * 2 + kind); if (g >= ngrp) return false;
    const size_t go = u32(off_g + 4 * (size_t)g); if (go + 8 > n) return false;
    if (b[go] != 0 || b[go + 1] != kb || b[go + 3] != 0 || b[go + 4] != 0) return false;
  }
  return true;
}

inline int kvqaxpk_kb(const unsigned char* b, size_t n) {
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (n < 100) return 4;
  const unsigned ngrp = u32(28), off_g = u32(40);
  int cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned g = 0; g < ngrp; ++g) {
    const size_t go = u32(off_g + 4 * (size_t)g); if (go + 8 > n) return 4;
    if (b[go] == 0 && b[go + 1] < 8) ++cnt[b[go + 1]];
  }
  return cnt[5] > cnt[4] ? 5 : 4;
}
} 
