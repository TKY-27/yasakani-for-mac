// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_fp16.h>
#include <stdexcept>
#include "a8_prologue.cuh"
#include "post_kernels.hpp"
#include "args.hpp"
namespace yskn {







template <int MINB, bool HOIST, int G, int HD, int RD>
__global__ void __launch_bounds__(128, MINB) attention_gqa_kernel(AttnGArgs A) { pdl_wait(); pdl_trigger();
  static_assert(G >= 1 && G <= 8 && HD % 32 == 0 && RD % 64 == 0 && RD <= HD, "attention_gqa: G <= 8, HD multiple of 32, RD multiple of 64");
  constexpr int D = HD / 32, DR = RD / 32;                                
  __shared__ float sm[4][G], sl[4][G], sacc[4][G][HD];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *A.d_pos + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
  const int cur_group = pos / 32;
  const float qs = A.qscale;                                               
  float q[G][D];
  #pragma unroll
  for (int i = 0; i < G; ++i)
    #pragma unroll
    for (int d = 0; d < D; ++d) q[i][d] = __fmul_rn(h2f(A.q16[((size_t)tok * A.nh + kvh * G + i) * HD + lane + 32 * d]), qs);
  float m[G], l[G], acc[G][D];
  #pragma unroll
  for (int i = 0; i < G; ++i) { m[i] = -1e30f; l[i] = 0.f;
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[i][d] = 0.f; }
  struct Raw { uint8_t kb[D]; uint16_t ks[D], kz[D], ksh[D]; float2 r[DR / 2 > 0 ? DR / 2 : 1]; uint8_t vb[D]; uint16_t vs, vz; };
  auto load_raw = [&](int t, Raw& R) {
    if (t / 32 < cur_group) { const size_t gb = (size_t)kvh * (A.max_ctx / 32) + t / 32; const int tt = t & 31;
      #pragma unroll
      for (int d = 0; d < D; ++d) { const int ch = lane + 32 * d; R.kb[d] = A.kv.kq[(gb * 32 + tt) * (HD / 2) + (ch >> 1)]; R.ks[d] = A.kv.kscale[gb * HD + ch]; R.kz[d] = A.kv.kzero[gb * HD + ch]; } }
    else {
      #pragma unroll
      for (int d = 0; d < D; ++d) R.ksh[d] = A.kv.kshadow[((size_t)kvh * A.max_ctx + t) * HD + lane + 32 * d]; }
    #pragma unroll
    for (int j = 0; j < DR / 2; ++j) R.r[j] = A.rope[(size_t)t * (RD / 2) + lane + 32 * j];
    { const size_t slot = (size_t)kvh * A.max_ctx + t; R.vs = A.kv.vscale[slot]; R.vz = A.kv.vzero[slot];
      #pragma unroll
      for (int d = 0; d < D; ++d) R.vb[d] = A.kv.vq[slot * (HD / 2) + ((lane + 32 * d) >> 1)]; }
  };
  Raw cur, nxt; int t = t0 + warp;
  if (HOIST && t < t1) load_raw(t, cur);
  for (; t < t1; t += 4) {
    if (HOIST) { if (t + 4 < t1) load_raw(t + 4, nxt); } else load_raw(t, cur);
    float kd[D];
    if (t / 32 < cur_group) {                                              
      #pragma unroll
      for (int d = 0; d < D; ++d) { const int ch = lane + 32 * d; const int nib = (ch & 1) ? (cur.kb[d] >> 4) : (cur.kb[d] & 15);
        kd[d] = __fadd_rn(__fmul_rn((float)nib, h2f(cur.ks[d])), h2f(cur.kz[d])); }
    } else {                                                               
      #pragma unroll
      for (int d = 0; d < D; ++d) kd[d] = h2f(cur.ksh[d]);
    }
    #pragma unroll
    for (int j = 0; j < DR / 2; ++j) {                                     
      const float c = cur.r[j].x, s = cur.r[j].y, a = kd[j], b = kd[j + DR / 2];
      kd[j] = a * c - b * s; kd[j + DR / 2] = a * s + b * c; }
    float vd[D];
    { const float vs = h2f(cur.vs), vz = h2f(cur.vz);
      #pragma unroll
      for (int d = 0; d < D; ++d) { const int ch = lane + 32 * d; const int nib = (ch & 1) ? (cur.vb[d] >> 4) : (cur.vb[d] & 15); vd[d] = (float)nib * vs + vz; } }
    
    float p[8];
    #pragma unroll
    for (int i = 0; i < G; ++i) { float s = q[i][0] * kd[0];
      #pragma unroll
      for (int d = 1; d < D; ++d) s += q[i][d] * kd[d];
      p[i] = s; }
    #pragma unroll
    for (int i = G; i < 8; ++i) p[i] = 0.f;
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float v4[4], v2[2], v1;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
    #pragma unroll
    for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
    { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    #pragma unroll
    for (int i = 0; i < G; ++i) {
      const float s = __shfl_sync(0xffffffffu, v1, 4 * i);
      const float mn = fmaxf(m[i], s), a = expf(m[i] - mn), b = expf(s - mn);
      l[i] = l[i] * a + b;
      #pragma unroll
      for (int d = 0; d < D; ++d) acc[i][d] = acc[i][d] * a + b * vd[d];
      m[i] = mn;
    }
    if (HOIST) cur = nxt;
  }
  #pragma unroll
  for (int i = 0; i < G; ++i) { if (lane == 0) { sm[warp][i] = m[i]; sl[warp][i] = l[i]; }
    #pragma unroll
    for (int d = 0; d < D; ++d) sacc[warp][i][lane + 32 * d] = acc[i][d]; }
  __syncthreads();
  for (int i = warp; i < G; i += 4) {                                      
    float M2 = sm[0][i]; for (int w = 1; w < 4; ++w) M2 = fmaxf(M2, sm[w][i]);
    float L2 = 0.f, o4[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) o4[d] = 0.f;
    for (int w = 0; w < 4; ++w) { const float sc = (sm[w][i] > -1e29f) ? expf(sm[w][i] - M2) : 0.f; L2 += sc * sl[w][i];
      #pragma unroll
      for (int d = 0; d < D; ++d) o4[d] += sc * sacc[w][i][lane + 32 * d]; }
    const size_t pidx = ((size_t)tok * A.splits + split) * A.nh + kvh * G + i;
    if (lane == 0) { A.pm[pidx] = M2; A.pl[pidx] = L2; }
    #pragma unroll
    for (int d = 0; d < D; ++d) A.pacc[pidx * HD + lane + 32 * d] = o4[d];
  }
}




template <int HD>
__global__ void __launch_bounds__(512) attn_combine_gqa_xform_kernel(AttnGCombineArgs A, const int8_t* __restrict__ signs, int8_t* qx, float* alpha) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, e = blk * 512 + tid, head = e / HD, d = e % HD, kvh = head / A.g;
  float M2 = -1e30f; for (int s = 0; s < A.splits; ++s) M2 = fmaxf(M2, A.pm[((size_t)tok * A.splits + s) * A.nh + head]);
  float L2 = 0.f, o = 0.f;
  for (int s = 0; s < A.splits; ++s) { const size_t p = ((size_t)tok * A.splits + s) * A.nh + head; const float sc = (A.pm[p] > -1e29f) ? expf(A.pm[p] - M2) : 0.f; L2 += sc * A.pl[p]; o += sc * A.pacc[p * HD + d]; }
  float y = round_f16(o / L2);
  if (A.bv) y = __fadd_rn(y, h2f(A.bv[kvh * HD + d]));
  if (A.gate) { const float gt = h2f(A.gate[((size_t)tok * A.nh + head) * HD + d]); y = __fmul_rn(y, __fdiv_rn(1.f, __fadd_rn(1.f, expf(-gt)))); }
  const uint16_t o16 = f2h(y);
  A.o16[((size_t)tok * A.nh + head) * HD + d] = o16;
  if (tid == 0) s_amax = 0u;
  buf[tid] = __fmul_rn(h2f(o16), (float)signs[blk * 512 + tid]);
  __syncthreads();
  const int ng = A.nh * HD / 512;
  a8_group512_tail(buf, &s_amax, qx + (size_t)tok * A.nh * HD + blk * 512, alpha + (size_t)tok * ng + blk);
}


template <int HD>
__global__ void __launch_bounds__(HD) kv_group_quant_g_kernel(KvQuantArgs A) { pdl_wait(); pdl_trigger();
  const int pos0 = *A.d_pos, last = pos0 + A.M - 1, gi = pos0 / 32 + blockIdx.x, head = blockIdx.y, ch = threadIdx.x;
  const int gend = gi * 32 + 31; if (gend < pos0 || gend > last) return;
  float v[32], lo = 1e30f, hi = -1e30f;
  for (int t = 0; t < 32; ++t) { v[t] = h2f(A.kv.kshadow[((size_t)head * A.max_ctx + gi * 32 + t) * HD + ch]); lo = fminf(lo, v[t]); hi = fmaxf(hi, v[t]); }
  const float sc = (hi - lo) / 15.f, inv = hi > lo ? 15.f / (hi - lo) : 0.f;
  const size_t gb = (size_t)head * (A.max_ctx / 32) + gi;
  A.kv.kscale[gb * HD + ch] = f2h(sc); A.kv.kzero[gb * HD + ch] = f2h(lo);
  for (int t = 0; t < 32; ++t) { int q = __float2int_rn((v[t] - lo) * inv); q = q < 0 ? 0 : (q > 15 ? 15 : q); const int nb = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((ch & 1) == 0) A.kv.kq[(gb * 32 + t) * (HD / 2) + (ch >> 1)] = (uint8_t)((q & 15) | ((nb & 15) << 4)); }
}



__global__ void rope_table_g_kernel(float2* out, int max_ctx, float theta, int rd) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x, half = rd / 2; if (idx >= max_ctx * half) return;
  const int t = idx / half, i = idx % half; float s, c;
  sincosf((float)t * powf(theta, -(float)(2 * i) / (float)rd), &s, &c); out[idx] = make_float2(c, s);
}
void launch_rope_table_g(float2* out, int max_ctx, float theta, int rd, cudaStream_t st) {
  rope_table_g_kernel<<<(max_ctx * (rd / 2) + 255) / 256, 256, 0, st>>>(out, max_ctx, theta, rd);
}

void launch_attention_gqa(const AttnGArgs& A, int g, int hd, int rd, cudaStream_t st) {
  const dim3 grid(A.kvh, A.splits, A.M), block(128);
#define ATT_G(G_, HD_, RD_, MB_) { if (A.M <= 2) launch_k(attention_gqa_kernel<MB_, true, G_, HD_, RD_>, grid, block, 0, st, A); \
                                   else launch_k(attention_gqa_kernel<MB_, false, G_, HD_, RD_>, grid, block, 0, st, A); return; }
  if (g == 7 && hd == 128 && rd == 128) ATT_G(7, 128, 128, 4)          
  if (g == 6 && hd == 256 && rd == 64) ATT_G(6, 256, 64, 2)            
#undef ATT_G
  throw std::runtime_error("launch_attention_gqa: unsupported (G, head_dim, rotary_dim)");
}
void launch_attn_combine_gqa_xform(const AttnGCombineArgs& A, int hd, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t st) {
  const dim3 grid(A.nh * hd / 512, A.M), block(512);
  if (hd == 128) { launch_k(attn_combine_gqa_xform_kernel<128>, grid, block, 0, st, A, signs, qx, alpha); return; }
  if (hd == 256) { launch_k(attn_combine_gqa_xform_kernel<256>, grid, block, 0, st, A, signs, qx, alpha); return; }
  throw std::runtime_error("launch_attn_combine_gqa_xform: unsupported head_dim");
}
void launch_kv_group_quant_g(const KvQuantArgs& A, int kvh, int hd, cudaStream_t st) {
  const dim3 grid((A.M + 31) / 32 + 1, kvh);
  if (hd == 128) { launch_k(kv_group_quant_g_kernel<128>, grid, dim3(128), 0, st, A); return; }
  if (hd == 256) { launch_k(kv_group_quant_g_kernel<256>, grid, dim3(256), 0, st, A); return; }
  throw std::runtime_error("launch_kv_group_quant_g: unsupported head_dim");
}
} 
