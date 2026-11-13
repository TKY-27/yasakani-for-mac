// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_fp16.h>
#include "a8_prologue.cuh"
#include "post_kernels.hpp"
#include "args.hpp"
#include <algorithm>
#include <cstdlib>
namespace yskn {








__device__ __forceinline__ void attn_softmax7(float s, float& mh, float& lh, float (&acc)[7][4], const float (&vd)[4]) {
  const float mn = fmaxf(mh, s);
  const bool gt = s > mh;
  const float e = expf((gt ? mh : s) - mn);
  const float a = gt ? e : 1.f, b = gt ? 1.f : e;
  lh = lh * a + b; mh = mn;
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    const float ai = __shfl_sync(0xffffffffu, a, 4 * i), bi = __shfl_sync(0xffffffffu, b, 4 * i);
    #pragma unroll
    for (int d = 0; d < 4; ++d) acc[i][d] = acc[i][d] * ai + bi * vd[d];
  }
}






__device__ __forceinline__ void attn_split_combine(float4* sacc, float* sm, float* sl, int bar_id, int warp, int lane, float mh, float lh,
                                                   const float (&acc)[7][4], float* pm, float* pl, float* pacc, size_t pidx0) {
  if ((lane & 3) == 0 && lane < 28) { sm[warp * 7 + (lane >> 2)] = mh; sl[warp * 7 + (lane >> 2)] = lh; }
  #pragma unroll
  for (int i = 0; i < 7; ++i) sacc[(warp * 7 + i) * 32 + lane] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
  if (bar_id) asm volatile("bar.sync %0, 128;" :: "r"(bar_id) : "memory"); else __syncthreads();
  const int hi = min(lane >> 2, 6), wi = lane & 3;                         
  const float mv = sm[wi * 7 + hi], lv = sl[wi * 7 + hi];
  float M2 = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 1));
  M2 = fmaxf(M2, __shfl_xor_sync(0xffffffffu, M2, 2));
  const float sc = (mv > -1e29f) ? expf(mv - M2) : 0.f, pr = sc * lv;
  const int b4 = lane & ~3;
  float L2 = 0.f;
  L2 += __shfl_sync(0xffffffffu, pr, b4); L2 += __shfl_sync(0xffffffffu, pr, b4 | 1);
  L2 += __shfl_sync(0xffffffffu, pr, b4 | 2); L2 += __shfl_sync(0xffffffffu, pr, b4 | 3);
  for (int i = warp; i < 7; i += 4) {
    float o4[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float sw = __shfl_sync(0xffffffffu, sc, 4 * i + w); const float4 a = sacc[(w * 7 + i) * 32 + lane];
      o4[0] += sw * a.x; o4[1] += sw * a.y; o4[2] += sw * a.z; o4[3] += sw * a.w;
    }
    const float Mi = __shfl_sync(0xffffffffu, M2, 4 * i), Li = __shfl_sync(0xffffffffu, L2, 4 * i);
    const size_t pidx = pidx0 + i;
    if (lane == 0) { pm[pidx] = Mi; pl[pidx] = Li; }
    #pragma unroll
    for (int d = 0; d < 4; ++d) pacc[pidx * 128 + lane + 32 * d] = o4[d];
  }
}




template <int MINB, bool HOIST>
__global__ void __launch_bounds__(128, MINB) attention_kernel(AttnArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float sm[4 * 7], sl[4 * 7]; __shared__ float4 sacc[4 * 7 * 32];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *A.d_pos + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
  const int cur_group = pos / 32;
  float q[7][4]; const float qs = 0.08838834764831845f;      
  #pragma unroll
  for (int i = 0; i < 7; ++i)
    #pragma unroll
    for (int d = 0; d < 4; ++d) q[i][d] = __fmul_rn(h2f(A.q16[((size_t)tok * 28 + kvh * 7 + i) * 128 + lane + 32 * d]), qs);
  float mh = -1e30f, lh = 0.f, acc[7][4];                          
  #pragma unroll
  for (int i = 0; i < 7; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  
  
  struct Raw { uint8_t kb[4]; uint16_t ks[4], kz[4], ksh[4]; float2 r0, r1; uint8_t vb[4]; uint16_t vs, vz; };
  auto load_raw = [&](int t, Raw& R) {
    if (t / 32 < cur_group) { const size_t gb = (size_t)kvh * (A.max_ctx / 32) + t / 32; const int tt = t & 31;
      #pragma unroll
      for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; R.kb[d] = A.kv.kq[(gb * 32 + tt) * 64 + (ch >> 1)]; R.ks[d] = A.kv.kscale[gb * 128 + ch]; R.kz[d] = A.kv.kzero[gb * 128 + ch]; } }
    else {
      #pragma unroll
      for (int d = 0; d < 4; ++d) R.ksh[d] = A.kv.kshadow[((size_t)kvh * A.max_ctx + t) * 128 + lane + 32 * d]; }
    if (A.rope) { R.r0 = A.rope[(size_t)t * 64 + lane]; R.r1 = A.rope[(size_t)t * 64 + lane + 32]; }
    { const size_t slot = (size_t)kvh * A.max_ctx + t; R.vs = A.kv.vscale[slot]; R.vz = A.kv.vzero[slot];
      #pragma unroll
      for (int d = 0; d < 4; ++d) R.vb[d] = A.kv.vq[slot * 64 + ((lane + 32 * d) >> 1)]; }
  };
  
  
  Raw cur, nxt; int t = t0 + warp;
  if (HOIST && t < t1) load_raw(t, cur);
  for (; t < t1; t += 4) {
    if (HOIST) { if (t + 4 < t1) load_raw(t + 4, nxt); } else load_raw(t, cur);
    float kd[4];
    if (t / 32 < cur_group) {                                              
      #pragma unroll
      for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const int nib = (ch & 1) ? (cur.kb[d] >> 4) : (cur.kb[d] & 15);
        kd[d] = __fadd_rn(__fmul_rn((float)nib, h2f(cur.ks[d])), h2f(cur.kz[d])); }
    } else {                                                               
      #pragma unroll
      for (int d = 0; d < 4; ++d) kd[d] = h2f(cur.ksh[d]);
    }
    { float s0, c0, s1, c1;                                                
      if (A.rope) { c0 = cur.r0.x; s0 = cur.r0.y; c1 = cur.r1.x; s1 = cur.r1.y; }
      else { sincosf((float)t * powf(A.rope_theta, -(float)(2 * lane) / 128.f), &s0, &c0);
             sincosf((float)t * powf(A.rope_theta, -(float)(2 * (lane + 32)) / 128.f), &s1, &c1); }
      const float a = kd[0], b = kd[2], c = kd[1], dd = kd[3];
      kd[0] = a * c0 - b * s0; kd[2] = a * s0 + b * c0; kd[1] = c * c1 - dd * s1; kd[3] = c * s1 + dd * c1; }
    float vd[4];
    { const float vs = h2f(cur.vs), vz = h2f(cur.vz);
      #pragma unroll
      for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const int nib = (ch & 1) ? (cur.vb[d] >> 4) : (cur.vb[d] & 15); vd[d] = (float)nib * vs + vz; } }
    
    
    
    
    float p[8];
    #pragma unroll
    for (int i = 0; i < 7; ++i) p[i] = q[i][0] * kd[0] + q[i][1] * kd[1] + q[i][2] * kd[2] + q[i][3] * kd[3];
    p[7] = 0.f;
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float v4[4], v2[2], v1;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
    #pragma unroll
    for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
    { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    attn_softmax7(v1, mh, lh, acc, vd);
    if (HOIST) cur = nxt;
  }
  attn_split_combine(sacc, sm, sl, 0, warp, lane, mh, lh, acc, A.pm, A.pl, A.pacc, ((size_t)tok * A.splits + split) * 28 + kvh * 7);
}




__device__ __forceinline__ void attn_final_combine7(const AttnArgs& A, int tok, int kvh, const uint16_t* __restrict__ bv, uint16_t* __restrict__ o16) {
  const int tid = threadIdx.x;
  if (A.splits == 32) {
    __shared__ float s_sc[7][32], s_L[7];
    const int w = tid >> 5, lane = tid & 31;
    if (w < 7) {
      const size_t p = ((size_t)tok * 32 + lane) * 28 + kvh * 7 + w;
      const float pmv = __ldcg(A.pm + p), plv = __ldcg(A.pl + p);
      float M2 = pmv;
      for (int o2 = 16; o2; o2 >>= 1) M2 = fmaxf(M2, __shfl_xor_sync(0xffffffffu, M2, o2));
      M2 = fmaxf(-1e30f, M2);
      const float sc = (pmv > -1e29f) ? expf(pmv - M2) : 0.f;
      s_sc[w][lane] = sc;
      float L2 = 0.f;
      for (int k = 0; k < 32; ++k) L2 += __shfl_sync(0xffffffffu, sc, k) * __shfl_sync(0xffffffffu, plv, k);
      if (lane == 0) s_L[w] = L2;
    }
    __syncthreads();
    for (int e = tid; e < 7 * 128; e += blockDim.x) {
      const int i = e / 128, d = e % 128, head = kvh * 7 + i;
      float o = 0.f;
      #pragma unroll
      for (int s = 0; s < 32; ++s) o += s_sc[i][s] * __ldcg(A.pacc + (((size_t)tok * 32 + s) * 28 + head) * 128 + d);
      const float y = round_f16(o / s_L[i]);
      o16[((size_t)tok * 28 + head) * 128 + d] = f2h(__fadd_rn(y, h2f(bv[kvh * 128 + d])));
    }
    return;
  }
  for (int e = tid; e < 7 * 128; e += blockDim.x) {
    const int head = kvh * 7 + e / 128, d = e % 128;
    float M2 = -1e30f; for (int s = 0; s < A.splits; ++s) M2 = fmaxf(M2, __ldcg(A.pm + ((size_t)tok * A.splits + s) * 28 + head));
    float L2 = 0.f, o = 0.f;
    for (int s = 0; s < A.splits; ++s) { const size_t p = ((size_t)tok * A.splits + s) * 28 + head; const float pmv = __ldcg(A.pm + p);
      const float sc = (pmv > -1e29f) ? expf(pmv - M2) : 0.f; L2 += sc * __ldcg(A.pl + p); o += sc * __ldcg(A.pacc + p * 128 + d); }
    const float y = round_f16(o / L2);
    o16[((size_t)tok * 28 + head) * 128 + d] = f2h(__fadd_rn(y, h2f(bv[kvh * 128 + d])));
  }
}









__device__ __forceinline__ void kv_dq_token(const AttnArgs& A, const KvF32& F, int kvh, int t, int lane, bool want_q, bool want_s, bool want_v) {
    float s0, c0, s1, c1;                                                  
    if (A.rope) { const float2 r0 = A.rope[(size_t)t * 64 + lane], r1 = A.rope[(size_t)t * 64 + lane + 32]; c0 = r0.x; s0 = r0.y; c1 = r1.x; s1 = r1.y; }
    else { sincosf((float)t * powf(A.rope_theta, -(float)(2 * lane) / 128.f), &s0, &c0);
           sincosf((float)t * powf(A.rope_theta, -(float)(2 * (lane + 32)) / 128.f), &s1, &c1); }
    const size_t o = ((size_t)kvh * A.max_ctx + t) * 32 + lane;
    if (want_q) {                                                          
      const size_t gb = (size_t)kvh * (A.max_ctx / 32) + t / 32; const int tt = t & 31; float kd[4];
      #pragma unroll
      for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const uint8_t kb = A.kv.kq[(gb * 32 + tt) * 64 + (ch >> 1)];
        const int nib = (ch & 1) ? (kb >> 4) : (kb & 15);
        kd[d] = __fadd_rn(__fmul_rn((float)nib, h2f(A.kv.kscale[gb * 128 + ch])), h2f(A.kv.kzero[gb * 128 + ch])); }
      const float a = kd[0], b = kd[2], c = kd[1], dd = kd[3];
      F.kq[o] = make_float4(a * c0 - b * s0, c * c1 - dd * s1, a * s0 + b * c0, c * s1 + dd * c1);
    }
    if (want_s) {                                                          
      float kd[4];
      #pragma unroll
      for (int d = 0; d < 4; ++d) kd[d] = h2f(A.kv.kshadow[((size_t)kvh * A.max_ctx + t) * 128 + lane + 32 * d]);
      const float a = kd[0], b = kd[2], c = kd[1], dd = kd[3];
      F.ks[o] = make_float4(a * c0 - b * s0, c * c1 - dd * s1, a * s0 + b * c0, c * s1 + dd * c1);
    }
    if (want_v) { const size_t slot = (size_t)kvh * A.max_ctx + t; const float vs = h2f(A.kv.vscale[slot]), vz = h2f(A.kv.vzero[slot]); float vd[4];
      #pragma unroll
      for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const uint8_t vb = A.kv.vq[slot * 64 + (ch >> 1)];
        const int nib = (ch & 1) ? (vb >> 4) : (vb & 15); vd[d] = (float)nib * vs + vz; }
      F.v[o] = make_float4(vd[0], vd[1], vd[2], vd[3]); }
}
__global__ void __launch_bounds__(256) kv_dequant_kernel(AttnArgs A, KvF32 F) {
  const int kvh = blockIdx.y, lane = threadIdx.x & 31, t00 = blockIdx.x * 8 + (threadIdx.x >> 5), ts = gridDim.x * 8;
  const int pos0 = *A.d_pos, T = pos0 + A.M, cmin = pos0 / 32, cmax = (T - 1) / 32;
  for (int t = t00; t < pos0; t += ts) kv_dq_token(A, F, kvh, t, lane, t / 32 < cmin, t / 32 >= cmin, true);
  pdl_wait(); pdl_trigger();
  for (int t = t00; t < T; t += ts) {
    const bool q = t / 32 < cmax && (t >= pos0 || t / 32 >= cmin);       
    if (t >= pos0 || q) kv_dq_token(A, F, kvh, t, lane, q, t >= pos0, t >= pos0);
  }
}
void launch_kv_dequant(const AttnArgs& A, const KvF32& F, cudaStream_t st) { launch_k(kv_dequant_kernel, dim3(256, 4), dim3(256), 0, st, A, F); }

__device__ __forceinline__ void attn_token(const float (&q)[7][4], const float (&kd)[4], const float (&vd)[4], int lane, float& mh, float& lh, float (&acc)[7][4]) {
  float p[8];
  #pragma unroll
  for (int i = 0; i < 7; ++i) p[i] = q[i][0] * kd[0] + q[i][1] * kd[1] + q[i][2] * kd[2] + q[i][3] * kd[3];
  p[7] = 0.f;
  const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
  float v4[4], v2[2], v1;
  #pragma unroll
  for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
  #pragma unroll
  for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
  { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
  v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
  v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
  attn_softmax7(v1, mh, lh, acc, vd);
}

template <int MINB, bool HOIST>
__global__ void __launch_bounds__(128, MINB) attention_f32_kernel(AttnArgs A, KvF32 F) { pdl_wait(); pdl_trigger();
  __shared__ float sm[4 * 7], sl[4 * 7]; __shared__ float4 sacc[4 * 7 * 32];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *A.d_pos + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);
  const int cur_group = pos / 32;
  float q[7][4]; const float qs = 0.08838834764831845f;
  #pragma unroll
  for (int i = 0; i < 7; ++i)
    #pragma unroll
    for (int d = 0; d < 4; ++d) q[i][d] = __fmul_rn(h2f(A.q16[((size_t)tok * 28 + kvh * 7 + i) * 128 + lane + 32 * d]), qs);
  float mh = -1e30f, lh = 0.f, acc[7][4];
  #pragma unroll
  for (int i = 0; i < 7; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  const size_t kb = (size_t)kvh * A.max_ctx * 32 + lane;
  auto ld = [&](int t, float4& k4, float4& v4) { k4 = ((t / 32 < cur_group) ? F.kq : F.ks)[kb + (size_t)t * 32]; v4 = F.v[kb + (size_t)t * 32]; };
  int t = t0 + warp; float4 k4, v4, kn, vn;
  if (HOIST && t < t1) ld(t, k4, v4);
  for (; t < t1; t += 4) {
    if (HOIST) { if (t + 4 < t1) ld(t + 4, kn, vn); } else ld(t, k4, v4);
    const float kd[4] = {k4.x, k4.y, k4.z, k4.w}, vd[4] = {v4.x, v4.y, v4.z, v4.w};
    attn_token(q, kd, vd, lane, mh, lh, acc);
    if (HOIST) { k4 = kn; v4 = vn; }
  }
  attn_split_combine(sacc, sm, sl, 0, warp, lane, mh, lh, acc, A.pm, A.pl, A.pacc, ((size_t)tok * A.splits + split) * 28 + kvh * 7);
}
void launch_attention_f32(const AttnArgs& A, const KvF32& F, cudaStream_t st) {
  static const int hoist = [] { const char* e = std::getenv("YSKN_ATTN_F32_HOIST"); return e && e[0] ? std::atoi(e) : 1; }();
  if (hoist) launch_k(attention_f32_kernel<4, true>, dim3(4, A.splits, A.M), dim3(128), 0, st, A, F);
  else launch_k(attention_f32_kernel<4, false>, dim3(4, A.splits, A.M), dim3(128), 0, st, A, F);
}

__global__ void __launch_bounds__(512) attention_prefill_f32_kernel(AttnArgs A, KvF32 F, const uint16_t* __restrict__ bv, uint16_t* __restrict__ o16) {
  pdl_wait(); pdl_trigger();
  extern __shared__ float dsm[];                                         
  const int kvh = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, wg = tid >> 5, grp = wg >> 2, warp = wg & 3, lane = tid & 31;
  float* const sacc = dsm + (size_t)grp * 4 * 7 * 128;
  float* const sm = dsm + (size_t)16 * 7 * 128 + grp * 4 * 7;
  float* const sl = sm + 4 * 4 * 7;
  const int pos = *A.d_pos + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits;
  const int cur_group = pos / 32;
  float q[7][4]; const float qs = 0.08838834764831845f;
  #pragma unroll
  for (int i = 0; i < 7; ++i)
    #pragma unroll
    for (int d = 0; d < 4; ++d) q[i][d] = __fmul_rn(h2f(A.q16[((size_t)tok * 28 + kvh * 7 + i) * 128 + lane + 32 * d]), qs);
  const size_t kb = (size_t)kvh * A.max_ctx * 32 + lane;
  for (int split = grp; split < A.splits; split += 4) {
    const int t0 = split * per, t1 = min(len, t0 + per);
    float mh = -1e30f, lh = 0.f, acc[7][4];
    #pragma unroll
    for (int i = 0; i < 7; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    for (int t = t0 + warp; t < t1; t += 4) {
      const float4 k4 = ((t / 32 < cur_group) ? F.kq : F.ks)[kb + (size_t)t * 32], v4 = F.v[kb + (size_t)t * 32];
      const float kd[4] = {k4.x, k4.y, k4.z, k4.w}, vd[4] = {v4.x, v4.y, v4.z, v4.w};
      attn_token(q, kd, vd, lane, mh, lh, acc);
    }
    attn_split_combine(reinterpret_cast<float4*>(sacc), sm, sl, grp + 1, warp, lane, mh, lh, acc, A.pm, A.pl, A.pacc, ((size_t)tok * A.splits + split) * 28 + kvh * 7);
    asm volatile("bar.sync %0, 128;" :: "r"(grp + 1) : "memory");
  }
  __syncthreads();
  attn_final_combine7(A, tok, kvh, bv, o16);                                
}
void launch_attention_prefill_f32(const AttnArgs& A, const KvF32& F, const uint16_t* bv, uint16_t* o16, cudaStream_t st) {
  const size_t smem = ((size_t)16 * 7 * 128 + 2 * 16 * 7) * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(attention_prefill_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  launch_k(attention_prefill_f32_kernel, dim3(4, A.M), dim3(512), smem, st, A, F, bv, o16);
}

__global__ void __launch_bounds__(128) attn_combine_kernel(AttnCombineArgs A) { pdl_wait(); pdl_trigger();
  const int head = blockIdx.x, tok = blockIdx.y, d = threadIdx.x, kvh = head / 7;
  float M2 = -1e30f; for (int s = 0; s < A.splits; ++s) M2 = fmaxf(M2, A.pm[((size_t)tok * A.splits + s) * 28 + head]);
  float L2 = 0.f, o = 0.f;
  for (int s = 0; s < A.splits; ++s) { const size_t p = ((size_t)tok * A.splits + s) * 28 + head; const float sc = (A.pm[p] > -1e29f) ? expf(A.pm[p] - M2) : 0.f; L2 += sc * A.pl[p]; o += sc * A.pacc[p * 128 + d]; }
  const float y = round_f16(o / L2);
  A.o16[((size_t)tok * 28 + head) * 128 + d] = f2h(__fadd_rn(y, h2f(A.bv[kvh * 128 + d])));
}


__global__ void __launch_bounds__(512) attn_combine_xform_kernel(AttnCombineArgs A, const int8_t* __restrict__ signs, int8_t* qx, float* alpha) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ unsigned s_amax;
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, head = blk * 4 + (tid >> 7), d = tid & 127, kvh = head / 7;
  float y;
  if (A.splits == 32) {                                                    
    __shared__ float s_sc[4][32], s_L[4];                                  
    float pa[32];                                                          
    #pragma unroll
    for (int s = 0; s < 32; ++s) pa[s] = A.pacc[(((size_t)tok * 32 + s) * 28 + head) * 128 + d];
    if (tid < 128) {
      const int w = tid >> 5, sp = tid & 31; const size_t p = ((size_t)tok * 32 + sp) * 28 + blk * 4 + w;
      const float pmv = A.pm[p], plv = A.pl[p];
      float M2 = pmv;
      for (int o2 = 16; o2; o2 >>= 1) M2 = fmaxf(M2, __shfl_xor_sync(0xffffffffu, M2, o2));
      M2 = fmaxf(-1e30f, M2);
      const float sc = (pmv > -1e29f) ? expf(pmv - M2) : 0.f;
      s_sc[w][sp] = sc;
      float L2 = 0.f;
      for (int k = 0; k < 32; ++k) L2 += __shfl_sync(0xffffffffu, sc, k) * __shfl_sync(0xffffffffu, plv, k);
      if (sp == 0) s_L[w] = L2;
    }
    __syncthreads();
    const int hh = tid >> 7; float o = 0.f;
    #pragma unroll
    for (int s = 0; s < 32; ++s) o += s_sc[hh][s] * pa[s];
    y = round_f16(o / s_L[hh]);
  } else {
    float M2 = -1e30f; for (int s = 0; s < A.splits; ++s) M2 = fmaxf(M2, A.pm[((size_t)tok * A.splits + s) * 28 + head]);
    float L2 = 0.f, o = 0.f;
    for (int s = 0; s < A.splits; ++s) { const size_t p = ((size_t)tok * A.splits + s) * 28 + head; const float sc = (A.pm[p] > -1e29f) ? expf(A.pm[p] - M2) : 0.f; L2 += sc * A.pl[p]; o += sc * A.pacc[p * 128 + d]; }
    y = round_f16(o / L2);
  }
  const uint16_t o16 = f2h(__fadd_rn(y, h2f(A.bv[kvh * 128 + d])));
  A.o16[((size_t)tok * 28 + head) * 128 + d] = o16;
  if (tid == 0) s_amax = 0u;
  buf[tid] = __fmul_rn(h2f(o16), (float)signs[blk * 512 + tid]);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, qx + (size_t)tok * 3584 + blk * 512, alpha + (size_t)tok * 7 + blk);
}







__global__ void __launch_bounds__(512) attention_prefill_kernel(AttnArgs A, const uint16_t* __restrict__ bv, uint16_t* __restrict__ o16) {
  pdl_wait(); pdl_trigger();
  extern __shared__ float dsm[];                                         
  const int kvh = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, wg = tid >> 5, grp = wg >> 2, warp = wg & 3, lane = tid & 31;
  float* const sacc = dsm + (size_t)grp * 4 * 7 * 128;                   
  float* const sm = dsm + (size_t)16 * 7 * 128 + grp * 4 * 7;            
  float* const sl = sm + 4 * 4 * 7;
  const int pos = *A.d_pos + tok, len = pos + 1, per = (len + A.splits - 1) / A.splits;
  const int cur_group = pos / 32;
  float q[7][4]; const float qs = 0.08838834764831845f;
  #pragma unroll
  for (int i = 0; i < 7; ++i)
    #pragma unroll
    for (int d = 0; d < 4; ++d) q[i][d] = __fmul_rn(h2f(A.q16[((size_t)tok * 28 + kvh * 7 + i) * 128 + lane + 32 * d]), qs);
  for (int split = grp; split < A.splits; split += 4) {
    const int t0 = split * per, t1 = min(len, t0 + per);
    float mh = -1e30f, lh = 0.f, acc[7][4];                          
    #pragma unroll
    for (int i = 0; i < 7; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    for (int t = t0 + warp; t < t1; t += 4) {
      float kd[4];
      if (t / 32 < cur_group) { const size_t gb = (size_t)kvh * (A.max_ctx / 32) + t / 32; const int tt = t & 31;
        #pragma unroll
        for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const uint8_t kb = A.kv.kq[(gb * 32 + tt) * 64 + (ch >> 1)];
          const int nib = (ch & 1) ? (kb >> 4) : (kb & 15);
          kd[d] = __fadd_rn(__fmul_rn((float)nib, h2f(A.kv.kscale[gb * 128 + ch])), h2f(A.kv.kzero[gb * 128 + ch])); }
      } else {
        #pragma unroll
        for (int d = 0; d < 4; ++d) kd[d] = h2f(A.kv.kshadow[((size_t)kvh * A.max_ctx + t) * 128 + lane + 32 * d]);
      }
      { float s0, c0, s1, c1;
        if (A.rope) { const float2 r0 = A.rope[(size_t)t * 64 + lane], r1 = A.rope[(size_t)t * 64 + lane + 32]; c0 = r0.x; s0 = r0.y; c1 = r1.x; s1 = r1.y; }
        else { sincosf((float)t * powf(A.rope_theta, -(float)(2 * lane) / 128.f), &s0, &c0);
               sincosf((float)t * powf(A.rope_theta, -(float)(2 * (lane + 32)) / 128.f), &s1, &c1); }
        const float a = kd[0], b = kd[2], c = kd[1], dd = kd[3];
        kd[0] = a * c0 - b * s0; kd[2] = a * s0 + b * c0; kd[1] = c * c1 - dd * s1; kd[3] = c * s1 + dd * c1; }
      float vd[4];
      { const size_t slot = (size_t)kvh * A.max_ctx + t; const float vs = h2f(A.kv.vscale[slot]), vz = h2f(A.kv.vzero[slot]);
        #pragma unroll
        for (int d = 0; d < 4; ++d) { const int ch = lane + 32 * d; const uint8_t vb = A.kv.vq[slot * 64 + (ch >> 1)];
          const int nib = (ch & 1) ? (vb >> 4) : (vb & 15); vd[d] = (float)nib * vs + vz; } }
      float p[8];
      #pragma unroll
      for (int i = 0; i < 7; ++i) p[i] = q[i][0] * kd[0] + q[i][1] * kd[1] + q[i][2] * kd[2] + q[i][3] * kd[3];
      p[7] = 0.f;
      const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
      float v4[4], v2[2], v1;
      #pragma unroll
      for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
      { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
      v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
      v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
      attn_softmax7(v1, mh, lh, acc, vd);
    }
    attn_split_combine(reinterpret_cast<float4*>(sacc), sm, sl, grp + 1, warp, lane, mh, lh, acc, A.pm, A.pl, A.pacc, ((size_t)tok * A.splits + split) * 28 + kvh * 7);
    asm volatile("bar.sync %0, 128;" :: "r"(grp + 1) : "memory");            
  }
  __syncthreads();                                                          
  attn_final_combine7(A, tok, kvh, bv, o16);                                
}
void launch_attention_prefill(const AttnArgs& A, const uint16_t* bv, uint16_t* o16, cudaStream_t st) {
  const size_t smem = ((size_t)16 * 7 * 128 + 2 * 16 * 7) * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(attention_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  launch_k(attention_prefill_kernel, dim3(4, A.M), dim3(512), smem, st, A, bv, o16);
}
void launch_attn_combine_xform(const AttnCombineArgs& A, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t st) {
  launch_k(attn_combine_xform_kernel, dim3(dim3(7, A.M)), dim3(512), 0, st, A, signs, qx, alpha);
}


__global__ void rope_table_kernel(float2* out, int max_ctx, float theta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= max_ctx * 64) return;
  const int t = idx >> 6, i = idx & 63; float s, c;
  sincosf((float)t * powf(theta, -(float)(2 * i) / 128.f), &s, &c); out[idx] = make_float2(c, s);
}
void launch_rope_table(float2* out, int max_ctx, float theta, cudaStream_t st) { rope_table_kernel<<<(max_ctx * 64 + 255) / 256, 256, 0, st>>>(out, max_ctx, theta); }
void launch_attention(const AttnArgs& A, cudaStream_t st) {
  
  
  
  
  
  static const int hoist_m = [] { const char* e = std::getenv("YSKN_ATTN_HOIST_M"); return e && e[0] ? std::atoi(e) : 0; }();
  if (A.M <= hoist_m) launch_k(attention_kernel<4, true>, dim3(dim3(4, A.splits, A.M)), dim3(128), 0, st, A);
  else launch_k(attention_kernel<4, false>, dim3(dim3(4, A.splits, A.M)), dim3(128), 0, st, A);
}
void launch_attn_combine(const AttnCombineArgs& A, cudaStream_t st) { launch_k(attn_combine_kernel, dim3(dim3(28, A.M)), dim3(128), 0, st, A); }
} 
