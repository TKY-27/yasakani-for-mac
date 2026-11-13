// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_fp16.h>
#include "a8_prologue.cuh"
#include "post_kernels.hpp"
namespace yskn {

__device__ __forceinline__ float combine(const float* part, int nw, int M, int N, int tok, int n) {
  float s = part[(size_t)tok * N + n];
  for (int w = 1; w < nw; ++w) s = __fadd_rn(s, part[((size_t)w * M + tok) * N + n]);     
  return s;
}

__device__ __forceinline__ void inv_h512(float* buf, const int8_t* signs) {   
  __shared__ float fw_tmp[512];
  const float v = fwht512_v(buf[threadIdx.x], buf, fw_tmp);           
  buf[threadIdx.x] = __fmul_rn(__fmul_rn(v, 0.044194173824159216f), (float)signs[threadIdx.x]);
  __syncthreads();
}
__device__ __forceinline__ float rope_cos_sin(int pos, int d, float theta, float* s) {     
  float c; sincosf((float)pos * powf(theta, -(float)(2 * d) / 128.f), s, &c); return c;
}


template <bool XSM>
__global__ void __launch_bounds__(512) qkv_post_kernel(QkvPostArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ float wmn[16], wmx[16], mn[4], mx[4]; __shared__ float s_xsm[2];
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, pos = *A.d_pos + tok;
  
  
  float nus = 0.f, nrr = 0.f;
  if (XSM && !A.fy) { if (tid == 0) { s_xsm[0] = A.us[tok]; s_xsm[1] = xsum_r(A.rr + (size_t)tok * 8, A.xsm_eps); } __syncthreads(); nus = s_xsm[0]; nrr = s_xsm[1]; }
  auto xsm_in = [&](const float* part, int N, int n, float tau) -> float {
    return A.fy ? __fmul_rn(__ldcg(part + (size_t)tok * N + n), tau) : __fmul_rn(__fmul_rn(xsum_y(xsum_combine(part, 0, A.M, N, tok, n), nus), nrr), tau); };
  if (blk < 7) {
    const int n = blk * 512 + tid;
    buf[tid] = XSM ? xsm_in(A.part_q, 3584, n, A.tau_q[n])
                    : __fmul_rn(A.tau_q[n], combine(A.part_q, A.nw_q, A.M, 3584, tok, n)); __syncthreads();
    inv_h512(buf, A.sg_q + blk * 512);
    const float y = round_f16(__fadd_rn(round_f16(buf[tid]), h2f(A.bq[n]))); buf[tid] = y; __syncthreads();
    const int head = n >> 7, d = n & 127; float out;
    const int dd = d < 64 ? d : d - 64; float s, c;
    if (A.rope) { const float2 r = A.rope[(size_t)pos * 64 + dd]; c = r.x; s = r.y; } else c = rope_cos_sin(pos, dd, A.rope_theta, &s);
    if (d < 64) out = __fsub_rn(__fmul_rn(buf[tid], c), __fmul_rn(buf[tid + 64], s));
    else        out = __fadd_rn(__fmul_rn(buf[tid], c), __fmul_rn(buf[tid - 64], s));
    A.q16[((size_t)tok * 28 + head) * 128 + d] = f2h(out);
  } else if (blk == 7) {                                                       
    buf[tid] = XSM ? xsm_in(A.part_k, 512, tid, A.tau_k[tid])
                    : __fmul_rn(A.tau_k[tid], combine(A.part_k, A.nw_k, A.M, 512, tok, tid)); __syncthreads();
    inv_h512(buf, A.sg_k);
    const float y = round_f16(__fadd_rn(round_f16(buf[tid]), h2f(A.bk[tid])));
    const int head = tid >> 7, d = tid & 127;
    A.kv.kshadow[((size_t)head * A.max_ctx + pos) * 128 + d] = f2h(y);
    if (A.kvfuse && (pos & 31) == 31) {                                       
      const int gi = pos >> 5; float v[32], lo = 1e30f, hi = -1e30f;           
      for (int t = 0; t < 32; ++t) { v[t] = h2f(A.kv.kshadow[((size_t)head * A.max_ctx + gi * 32 + t) * 128 + d]); lo = fminf(lo, v[t]); hi = fmaxf(hi, v[t]); }
      const float sc = (hi - lo) / 15.f, inv = hi > lo ? 15.f / (hi - lo) : 0.f;
      const size_t gb = (size_t)head * (A.max_ctx / 32) + gi;
      A.kv.kscale[gb * 128 + d] = f2h(sc); A.kv.kzero[gb * 128 + d] = f2h(lo);
      for (int t = 0; t < 32; ++t) { int q = __float2int_rn((v[t] - lo) * inv); q = q < 0 ? 0 : (q > 15 ? 15 : q); const int nb = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((d & 1) == 0) A.kv.kq[(gb * 32 + t) * 64 + (d >> 1)] = (uint8_t)((q & 15) | ((nb & 15) << 4)); }
    }
  } else {                                                                     
    buf[tid] = XSM ? xsm_in(A.part_v, 512, tid, A.tau_v[tid])
                    : __fmul_rn(A.tau_v[tid], combine(A.part_v, A.nw_v, A.M, 512, tok, tid)); __syncthreads();
    inv_h512(buf, A.sg_v);
    const float y = round_f16(buf[tid]); buf[tid] = y; __syncthreads();
    const int head = tid >> 7, d = tid & 127;
    float lmn = y, lmx = y;
    for (int o = 16; o; o >>= 1) { lmn = fminf(lmn, __shfl_xor_sync(0xffffffffu, lmn, o)); lmx = fmaxf(lmx, __shfl_xor_sync(0xffffffffu, lmx, o)); }
    if ((tid & 31) == 0) { wmn[tid >> 5] = lmn; wmx[tid >> 5] = lmx; }
    __syncthreads();
    if (tid < 4) { float a = wmn[tid * 4], b = wmx[tid * 4]; for (int w = 1; w < 4; ++w) { a = fminf(a, wmn[tid * 4 + w]); b = fmaxf(b, wmx[tid * 4 + w]); } mn[tid] = a; mx[tid] = b; }
    __syncthreads();
    const float lo = mn[head], hi = mx[head], sc = (hi - lo) / 15.f, inv = hi > lo ? 15.f / (hi - lo) : 0.f;
    int q = __float2int_rn((y - lo) * inv); q = q < 0 ? 0 : (q > 15 ? 15 : q);
    const int nb = __shfl_xor_sync(0xffffffffu, q, 1);
    const size_t slot = (size_t)head * A.max_ctx + pos;
    if ((d & 1) == 0) A.kv.vq[slot * 64 + (d >> 1)] = (uint8_t)((q & 15) | ((nb & 15) << 4));
    if (d == 0) { A.kv.vscale[slot] = f2h(sc); A.kv.vzero[slot] = f2h(lo); }
  }
}


__global__ void __launch_bounds__(128) kv_group_quant_kernel(KvQuantArgs A) { pdl_wait(); pdl_trigger();
  const int pos0 = *A.d_pos, last = pos0 + A.M - 1, gi = pos0 / 32 + blockIdx.x, head = blockIdx.y, ch = threadIdx.x;
  const int gend = gi * 32 + 31; if (gend < pos0 || gend > last) return;
  float v[32], lo = 1e30f, hi = -1e30f;
  for (int t = 0; t < 32; ++t) { v[t] = h2f(A.kv.kshadow[((size_t)head * A.max_ctx + gi * 32 + t) * 128 + ch]); lo = fminf(lo, v[t]); hi = fmaxf(hi, v[t]); }
  const float sc = (hi - lo) / 15.f, inv = hi > lo ? 15.f / (hi - lo) : 0.f;
  const size_t gb = (size_t)head * (A.max_ctx / 32) + gi;
  A.kv.kscale[gb * 128 + ch] = f2h(sc); A.kv.kzero[gb * 128 + ch] = f2h(lo);
  for (int t = 0; t < 32; ++t) { int q = __float2int_rn((v[t] - lo) * inv); q = q < 0 ? 0 : (q > 15 ? 15 : q); const int nb = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((ch & 1) == 0) A.kv.kq[(gb * 32 + t) * 64 + (ch >> 1)] = (uint8_t)((q & 15) | ((nb & 15) << 4)); }
}


template <bool XSM>
__global__ void __launch_bounds__(512) residual_post_kernel(ResidualPostArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  buf[tid] = XSM ? __fmul_rn(xsum_y(xsum_combine(A.part, A.nw, A.M, A.N, tok, n), A.us[tok]), A.tau[n])   
                  : __fmul_rn(A.tau[n], combine(A.part, A.nw, A.M, A.N, tok, n)); __syncthreads();
  inv_h512(buf, A.sg + blk * 512);
  float* h = A.h + (size_t)tok * A.N + n; *h = round_f16(__fadd_rn(*h, round_f16(buf[tid])));
}

__global__ void __launch_bounds__(512) mlp_post_kernel(MlpPostArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float bg[512], bu[512]; const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  bg[tid] = __fmul_rn(A.tau_g[n], combine(A.part_g, A.nw, A.M, A.F, tok, n));
  bu[tid] = __fmul_rn(A.tau_u[n], combine(A.part_u, A.nw, A.M, A.F, tok, n)); __syncthreads();
  inv_h512(bg, A.sg_g + blk * 512); inv_h512(bu, A.sg_u + blk * 512);
  const float g16 = round_f16(bg[tid]), u16 = round_f16(bu[tid]);
  const float s = round_f16(__fdiv_rn(g16, __fadd_rn(1.f, expf(-g16))));
  A.xff[(size_t)tok * A.F + n] = f2h(__fmul_rn(s, u16));
}

__global__ void __launch_bounds__(512) residual_norm_xform_kernel(ResNormXArgs X) { pdl_wait(); pdl_trigger();
  extern __shared__ float hrow[];                                       
  __shared__ float buf[512], s_red[4]; __shared__ unsigned s_amax;
  const ResidualPostArgs& A = X.R; const int g = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, N = A.N, nb = N / 512;
  
  for (int blk = 0; blk < nb; ++blk) { const int n = blk * 512 + tid; hrow[n] = __fmul_rn(A.tau[n], combine(A.part, A.nw, A.M, N, tok, n)); }
  __syncthreads();
  for (int h = 1; h < 512; h <<= 1) {                                   
    for (int i = tid; i < nb * 256; i += 512) {
      const int blk = i >> 8, k = i & 255, lo = ((k & ~(h - 1)) << 1) | (k & (h - 1)); float* x = hrow + blk * 512;
      const float a = x[lo], b = x[lo + h]; x[lo] = a + b; x[lo + h] = a - b;
    }
    __syncthreads();
  }
  for (int blk = 0; blk < nb; ++blk) {                                  
    const int n = blk * 512 + tid;
    const float y = __fmul_rn(__fmul_rn(hrow[n], 0.044194173824159216f), (float)A.sg[n]);
    hrow[n] = round_f16(__fadd_rn(X.h_in[(size_t)tok * N + n], round_f16(y)));
  }
  __syncthreads();
  X.h_out[(size_t)tok * N + g * 512 + tid] = hrow[g * 512 + tid];
  if (tid < 128) {                                                       
    float acc = 0.f; for (int i = tid; i < N; i += 128) acc = __fadd_rn(acc, __fmul_rn(hrow[i], hrow[i]));
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
  }
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)N), X.eps));
  const int i = g * 512 + tid;
  buf[tid] = __fmul_rn(round_f16(__fmul_rn(round_f16(__fmul_rn(hrow[i], r)), h2f(X.gamma[i]))), (float)X.signs[i]);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, X.qx + (size_t)tok * N + g * 512, X.alpha + (size_t)tok * nb + g);
}
void launch_residual_norm_xform(const ResNormXArgs& X, cudaStream_t st) {
  const size_t sm = (size_t)X.R.N * 4;
  cudaFuncSetAttribute(residual_norm_xform_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  launch_k(residual_norm_xform_kernel, dim3(dim3(X.R.N / 512, X.R.M)), dim3(512), sm, st, X);
}

template <bool XSM>
__global__ void __launch_bounds__(512) mlp_post_xform_kernel(MlpPostArgs A, const int8_t* __restrict__ signs, int8_t* qx, float* alpha) { pdl_wait(); pdl_trigger();
  __shared__ float bg[512], bu[512], tg[512], tu[512]; __shared__ unsigned s_amax; __shared__ float s_xsm[2];
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, n = blk * 512 + tid;
  float vg, vu;
  if (XSM && A.fy) { vg = __fmul_rn(__ldcg(A.part_g + (size_t)tok * A.F + n), A.tau_g[n]); vu = __fmul_rn(__ldcg(A.part_u + (size_t)tok * A.F + n), A.tau_u[n]); }
  else if (XSM) {                                                         
    const long long ag = xsum_combine(A.part_g, A.nw, A.M, A.F, tok, n), au = xsum_combine(A.part_u, A.nw, A.M, A.F, tok, n);
    if (tid == 0) { s_xsm[0] = A.us[tok]; s_xsm[1] = xsum_r(A.rr + (size_t)tok * 8, A.xsm_eps); }
    __syncthreads();
    vg = __fmul_rn(__fmul_rn(xsum_y(ag, s_xsm[0]), s_xsm[1]), A.tau_g[n]); vu = __fmul_rn(__fmul_rn(xsum_y(au, s_xsm[0]), s_xsm[1]), A.tau_u[n]); }
  else { vg = __fmul_rn(A.tau_g[n], combine(A.part_g, A.nw, A.M, A.F, tok, n)); vu = __fmul_rn(A.tau_u[n], combine(A.part_u, A.nw, A.M, A.F, tok, n)); }
  if (tid == 0) s_amax = 0u;
  fwht512_v2(vg, vu, bg, tg, bu, tu);                                      
  const float g16 = round_f16(__fmul_rn(__fmul_rn(vg, 0.044194173824159216f), (float)A.sg_g[blk * 512 + tid]));
  const float u16 = round_f16(__fmul_rn(__fmul_rn(vu, 0.044194173824159216f), (float)A.sg_u[blk * 512 + tid]));
  const float s = round_f16(__fdiv_rn(g16, __fadd_rn(1.f, expf(-g16))));
  const uint16_t x16 = f2h(__fmul_rn(s, u16));
  A.xff[(size_t)tok * A.F + n] = x16;
  bu[tid] = __fmul_rn(h2f(x16), (float)signs[n]);                          
  __syncthreads();
  a8_group512_tail(bu, &s_amax, qx + (size_t)tok * A.F + blk * 512, alpha + (size_t)tok * (A.F / 512) + blk);
}
void launch_mlp_post_xform(const MlpPostArgs& A, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t st) {
  if (A.us) launch_k(mlp_post_xform_kernel<true>, dim3(dim3(A.F / 512, A.M)), dim3(512), 0, st, A, signs, qx, alpha);
  else launch_k(mlp_post_xform_kernel<false>, dim3(dim3(A.F / 512, A.M)), dim3(512), 0, st, A, signs, qx, alpha);
}

__global__ void __launch_bounds__(512) xform_a8_kernel(XformA8Args A) { pdl_wait(); pdl_trigger();
  extern __shared__ float buf[]; __shared__ unsigned amax[64]; const int tok = blockIdx.x, n = A.n, m = A.m;
  for (int i = threadIdx.x; i < n; i += 512) buf[i] = __fmul_rn(h2f(A.x16[(size_t)tok * n + i]), (float)A.signs[i]);
  for (int i = threadIdx.x; i < 64; i += 512) amax[i] = 0u;
  __syncthreads();
  fwht512_blocks(buf, m);
  if (A.qm) qm_mix_inplace(buf, A.qm, m, false);
  for (int i = threadIdx.x; i < n; i += 512) { const float v = round_f16(buf[i]); buf[i] = v; atomicMax(&amax[i >> 9], __float_as_uint(fabsf(v))); }
  __syncthreads();
  for (int g = threadIdx.x; g < m; g += 512) A.alpha[(size_t)tok * m + g] = __fdiv_rn(__uint_as_float(amax[g]), 127.f);
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += 512) { const float a = A.alpha[(size_t)tok * m + (i >> 9)]; const float v = a > 0.f ? __fdiv_rn(buf[i], a) : 0.f;
    int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q); A.qx[(size_t)tok * n + i] = (int8_t)q; }
}




constexpr int NORM_XFORM_THREADS = 128;
__global__ void __launch_bounds__(NORM_XFORM_THREADS) norm_xform_a8_kernel(const float* __restrict__ h, const uint16_t* __restrict__ gamma, float eps, XformA8Args A) { pdl_wait(); pdl_trigger();
  extern __shared__ float buf[];
  const int tok = blockIdx.x, n = A.n;
  norm_apply(h + (size_t)tok * n, gamma, eps, n, buf);
  XformDesc X{A.signs, A.qm, n, A.m};
  a8_from_fp32(buf, X, A.qx + (size_t)tok * n, A.alpha + (size_t)tok * A.m);
}





template <bool NORM, bool XSM>
__global__ void __launch_bounds__(128) xform_a8_group_kernel(const float* __restrict__ h, const uint16_t* __restrict__ gamma, float eps, XformA8Args A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax, s_umax;
  const int grp = blockIdx.x, tok = blockIdx.y, n = A.n, tid = threadIdx.x;
  if (XSM && tid == 0) s_umax = 0u;
  float r = 0.f;
  if (NORM && XSM) { const float sb = xsum_block_ss(h + (size_t)tok * n + grp * 512, s_red); if (A.r_out && tid == 0) A.r_out[(size_t)tok * 8 + grp] = sb; }
  else if (NORM) {
    const float* hr = h + (size_t)tok * n; float acc = 0.f;
    for (int i = tid; i < n; i += 128) acc = __fadd_rn(acc, __fmul_rn(hr[i], hr[i]));
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
    __syncthreads();
    float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
    r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)n), eps));
  }
  for (int k = tid; k < 512; k += 128) {
    const int i = grp * 512 + k;
    const float v = NORM ? (XSM ? (A.xsum == 2 ? __fmul_rn(h[(size_t)tok * n + i], h2f(gamma[i]))   
                                                      : round_f16(__fmul_rn(h[(size_t)tok * n + i], h2f(gamma[i]))))
                                     : round_f16(__fmul_rn(round_f16(__fmul_rn(h[(size_t)tok * n + i], r)), h2f(gamma[i]))))
                         : h2f(A.x16[(size_t)tok * n + i]);
    buf[k] = __fmul_rn(v, (float)A.signs[i]);
    if (XSM && NORM && A.stats) atomicMax(&s_umax, __float_as_uint(fabsf(v)));       
  }
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  for (int hh = 1; hh < 512; hh <<= 1) {
    for (int k = tid; k < 256; k += 128) { const int lo = ((k & ~(hh - 1)) << 1) | (k & (hh - 1)); const float a = buf[lo], b = buf[lo + hh]; buf[lo] = a + b; buf[lo + hh] = a - b; }
    __syncthreads();
  }
  { unsigned mx = 0u;
    for (int k = tid; k < 512; k += 128) { const float v = round_f16(__fmul_rn(buf[k], 0.044194173824159216f)); buf[k] = v; mx = max(mx, __float_as_uint(fabsf(v)));
      if (XSM && A.nf && !isfinite(v)) atomicAdd(A.nf, 1u); }                                 
    for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));            
    if ((tid & 31) == 0) atomicMax(&s_amax, mx); }
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(s_amax), 127.f);
  if (tid == 0) A.alpha[(size_t)tok * A.m + grp] = al;
  if (XSM && A.stats && tid == 0) { atomicMax(A.nf + 1, s_umax); atomicMax(A.nf + 2, s_amax); }   
  for (int k = tid; k < 512; k += 128) {
    const float v = al > 0.f ? __fdiv_rn(buf[k], al) : 0.f; int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q);
    A.qx[(size_t)tok * n + grp * 512 + k] = (int8_t)q;
  }
}








__global__ void __launch_bounds__(512) res_norm_last_kernel(ResNormLastArgs X) { pdl_wait();
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax;
  const ResidualPostArgs& A = X.R; const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, N = A.N, nb = N / 512, n = blk * 512 + tid;
  buf[tid] = __fmul_rn(A.tau[n], combine(A.part, A.nw, A.M, N, tok, n)); __syncthreads();
  inv_h512(buf, A.sg + blk * 512);
  float* hp = A.h + (size_t)tok * N + n; *hp = round_f16(__fadd_rn(*hp, round_f16(buf[tid])));
  __threadfence(); __syncthreads();
  if (tid == 0) {                                                        
    const unsigned old = atomicAdd(X.cnt + tok, 1u), target = (old / (unsigned)nb + 1u) * (unsigned)nb;
    while ((int)(*reinterpret_cast<volatile unsigned*>(X.cnt + tok) - target) < 0) __nanosleep(40);
    __threadfence();
  }
  __syncthreads();
  pdl_trigger();
  const float* hr = A.h + (size_t)tok * N;
  if (tid < 128) {                                                       
    float acc = 0.f; for (int i = tid; i < N; i += 128) { const float x = __ldcg(hr + i); acc = __fadd_rn(acc, __fmul_rn(x, x)); }
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
  }
  if (tid == 0) s_amax = 0u;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)N), X.eps));
  buf[tid] = __fmul_rn(round_f16(__fmul_rn(round_f16(__fmul_rn(__ldcg(hr + n), r)), h2f(X.gamma[n]))), (float)X.signs[n]);
  __syncthreads();
  a8_group512_tail(buf, &s_amax, X.qx + (size_t)tok * N + blk * 512, X.alpha + (size_t)tok * nb + blk);
}
void launch_res_norm_last(const ResNormLastArgs& X, cudaStream_t st) { launch_k(res_norm_last_kernel, dim3(X.R.N / 512, X.R.M), dim3(512), 0, st, X); }

__global__ void __launch_bounds__(512) xsum_res_xform_kernel(XsmResXArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float buf[512]; __shared__ float s_red[4]; __shared__ unsigned s_amax; __shared__ float s_us, fw_tmp[512];
  const int b = blockIdx.x, tok = blockIdx.y, k = threadIdx.x, n = b * 512 + k;
  
  float y0;
  if (A.fy) y0 = __ldcg(A.part + (size_t)tok * A.N + n);
  else {
    const long long accv = xsum_combine(A.part, 0, A.M, A.N, tok, n);
    if (A.stats) {                                                  
      unsigned long long a = (unsigned long long)(accv < 0 ? -accv : accv);
      for (int o = 16; o; o >>= 1) { const unsigned long long b = __shfl_xor_sync(0xffffffffu, a, o); a = b > a ? b : a; }
      if ((k & 31) == 0) atomicMax(reinterpret_cast<unsigned long long*>(A.nf + 4), a); }
    if (k == 0) s_us = A.us[tok];
    __syncthreads();
    y0 = xsum_y(accv, s_us); }
  buf[k] = __fmul_rn(y0, A.tau[n]); __syncthreads();
  inv_h512(buf, A.sg_out + b * 512);
  float* hp = A.h + (size_t)tok * A.N + n; const float hn = round_f16(__fadd_rn(*hp, round_f16(buf[k]))); *hp = hn;
  __syncthreads();
  buf[k] = hn;
  if (k == 0) s_amax = 0u;
  __syncthreads();
  
  if (k < 128) { float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { const float x = buf[k + 128 * j]; acc = __fadd_rn(acc, __fmul_rn(x, x)); }
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((k & 31) == 0) s_red[k >> 5] = acc; }
  __syncthreads();
  if (k == 0) { float s = 0.f; for (int w = 0; w < 4; ++w) s = __fadd_rn(s, s_red[w]); A.S[(size_t)tok * 8 + b] = s; }
  const float u = A.u16 ? round_f16(__fmul_rn(hn, h2f(A.gamma[n]))) : __fmul_rn(hn, h2f(A.gamma[n]));
  const float v = round_f16(__fmul_rn(fwht512_v(__fmul_rn(u, (float)A.sg_in[n]), buf, fw_tmp), 0.044194173824159216f));   
  { unsigned mx = __float_as_uint(fabsf(v)); for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o)); if ((k & 31) == 0) atomicMax(&s_amax, mx); }
  if (A.nf && !isfinite(v)) atomicAdd(A.nf, 1u);
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(s_amax), 127.f);
  if (k == 0) A.alpha[(size_t)tok * (A.N / 512) + b] = al;
  const float qv = al > 0.f ? __fdiv_rn(v, al) : 0.f; int q = __float2int_rn(qv); q = q < -127 ? -127 : (q > 127 ? 127 : q);
  A.qx[(size_t)tok * A.N + n] = (int8_t)q;
}
void launch_xsum_res_xform(const XsmResXArgs& A, cudaStream_t st) { launch_k(xsum_res_xform_kernel, dim3(A.N / 512, A.M), dim3(512), 0, st, A); }
void launch_qkv_post(const QkvPostArgs& A, cudaStream_t st) { if (A.us) launch_k(qkv_post_kernel<true>, dim3(dim3(9, A.M)), dim3(512), 0, st, A); else launch_k(qkv_post_kernel<false>, dim3(dim3(9, A.M)), dim3(512), 0, st, A); }
void launch_kv_group_quant(const KvQuantArgs& A, cudaStream_t st) { launch_k(kv_group_quant_kernel, dim3(dim3((A.M + 31) / 32 + 1, 4)), dim3(128), 0, st, A); }
void launch_residual_post(const ResidualPostArgs& A, cudaStream_t st) { if (A.us) launch_k(residual_post_kernel<true>, dim3(dim3(A.N / 512, A.M)), dim3(512), 0, st, A); else launch_k(residual_post_kernel<false>, dim3(dim3(A.N / 512, A.M)), dim3(512), 0, st, A); }
void launch_mlp_post(const MlpPostArgs& A, cudaStream_t st) { launch_k(mlp_post_kernel, dim3(dim3(A.F / 512, A.M)), dim3(512), 0, st, A); }
void launch_xform_a8(const XformA8Args& A, cudaStream_t st) {
  if (!A.qm) { launch_k(xform_a8_group_kernel<false, false>, dim3(dim3(A.m, A.M)), dim3(128), 0, st, nullptr, nullptr, 0.f, A); return; }
  cudaFuncSetAttribute(xform_a8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, A.n * 4 + 1024);
  launch_k(xform_a8_kernel, dim3(A.M), dim3(512), A.n * 4, st, A);
}
void launch_norm_xform_a8(const float* h, const uint16_t* gamma, float eps, const XformA8Args& A, cudaStream_t st) {
  if (!A.qm) { if (A.xsum) launch_k(xform_a8_group_kernel<true, true>, dim3(dim3(A.m, A.M)), dim3(128), 0, st, h, gamma, eps, A);
               else launch_k(xform_a8_group_kernel<true, false>, dim3(dim3(A.m, A.M)), dim3(128), 0, st, h, gamma, eps, A); return; }
  cudaFuncSetAttribute(norm_xform_a8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, A.n * 4 + 1024);
  launch_k(norm_xform_a8_kernel, dim3(A.M), dim3(NORM_XFORM_THREADS), A.n * 4, st, h, gamma, eps, A);
}
} 
