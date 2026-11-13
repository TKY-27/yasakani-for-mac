// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <cuda_fp16.h>
#include <cfloat>
#include "a8_prologue.cuh"
#include "args.hpp"
namespace yskn {

__device__ __forceinline__ uint32_t read_window(const uint32_t* w, uint32_t start, uint32_t width) {   
  const uint32_t idx = start >> 5, sh = start & 31; uint32_t v = w[idx] >> sh;
  if (sh + width > 32) v |= w[idx + 1] << (32 - sh);
  return v & ((1u << width) - 1u);
}


__global__ void __launch_bounds__(512) embedding_kernel(EmbArgs A) { pdl_wait(); pdl_trigger();
  extern __shared__ float buf[]; const int id = A.d_tok[blockIdx.x]; const uint32_t* row = A.stream + (size_t)id * A.seqs * A.wps; const float sc = h2f(A.scale16[id]);
  float* const h = A.h + (size_t)blockIdx.x * A.m * 512;
  for (int i = threadIdx.x; i < A.seqs * 128; i += 512) {
    const int q = i >> 7, j = i & 127; const uint32_t* w = row + q * A.wps; const uint32_t e = (uint32_t)A.K * (j + 1);
    const uint32_t s = (e >= 12u) ? read_window(w, e - 12u, 12u) : (read_window(w, 0u, e) << (12u - e));
    buf[q * 256 + 2 * j] = __fmul_rn(h2f(A.lut16[2 * s]), sc); buf[q * 256 + 2 * j + 1] = __fmul_rn(h2f(A.lut16[2 * s + 1]), sc);
  }
  __syncthreads();
  if (A.qm) qm_mix_inplace(buf, A.qm, A.m, true);        
  fwht512_blocks(buf, A.m);
  for (int i = threadIdx.x; i < A.m * 512; i += 512) { const float v = round_f16(__fmul_rn(buf[i], (float)A.signs[i])); h[i] = v; buf[i] = v; }
  if (!A.gamma) return;
  
  
  
  __shared__ float s_red[4]; __shared__ unsigned s_amax[64]; const int n = A.m * 512, tid = threadIdx.x;
  __syncthreads();
  if (tid < 128) { float acc = 0.f; for (int i = tid; i < n; i += 128) acc = __fadd_rn(acc, __fmul_rn(buf[i], buf[i]));
    for (int o = 16; o; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    if ((tid & 31) == 0) s_red[tid >> 5] = acc; }
  for (int g = tid; g < 64; g += 512) s_amax[g] = 0u;
  __syncthreads();
  float tot = 0.f; for (int w = 0; w < 4; ++w) tot = __fadd_rn(tot, s_red[w]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)n), A.eps));
  for (int i = tid; i < n; i += 512) buf[i] = __fmul_rn(round_f16(__fmul_rn(round_f16(__fmul_rn(buf[i], r)), h2f(A.gamma[i]))), (float)A.signs_in[i]);
  __syncthreads();
  fwht512_blocks(buf, A.m);                                            
  for (int i = tid; i < n; i += 512) { const float v = round_f16(buf[i]); buf[i] = v;
    unsigned mx = __float_as_uint(fabsf(v)); for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));   
    if ((tid & 31) == 0) atomicMax(&s_amax[i >> 9], mx); }
  __syncthreads();
  for (int g = tid; g < A.m; g += 512) A.alpha[(size_t)blockIdx.x * A.m + g] = __fdiv_rn(__uint_as_float(s_amax[g]), 127.f);
  __syncthreads();
  for (int i = tid; i < n; i += 512) { const float al = A.alpha[(size_t)blockIdx.x * A.m + (i >> 9)]; const float v = al > 0.f ? __fdiv_rn(buf[i], al) : 0.f;
    int q = __float2int_rn(v); q = q < -127 ? -127 : (q > 127 ? 127 : q); A.qx[(size_t)blockIdx.x * n + i] = (int8_t)q; }
}



template <bool XSM>
__global__ void argmax_stage1(ArgmaxArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float sv[256]; __shared__ int si[256]; float best = -FLT_MAX; int bi = 0x7fffffff;
  const int row = A.row + blockIdx.y; float* const logits = A.logits + (size_t)blockIdx.y * A.V;
  float nrr = 0.f, nus = 0.f;                                            
  if (XSM) { __shared__ float s_xsm[2];                                  
    if (threadIdx.x == 0) { const float* S = A.rr + (size_t)row * A.xsm_stride; float s = 0.f;
      for (int b = 0; b < A.xsm_nb; ++b) s = __fadd_rn(s, S[b]);
      s_xsm[0] = rsqrtf(__fadd_rn(__fdiv_rn(s, A.xsm_n), A.xsm_eps)); s_xsm[1] = A.us[row]; }
    __syncthreads(); nrr = s_xsm[0]; nus = s_xsm[1]; }
  for (int v = blockIdx.x * 256 + threadIdx.x; v < A.V; v += gridDim.x * 256) {
    float x;
    if (XSM) x = __fmul_rn(__fmul_rn(xsum_y(xsum_combine(A.part, A.nw, A.M, A.V, row, v), nus), nrr), A.tau[v]);   
    else { float s = A.part[(size_t)row * A.V + v];                               
      for (int w = 1; w < A.nw; ++w) s = __fadd_rn(s, A.part[((size_t)w * A.M + row) * A.V + v]);
      x = __fmul_rn(A.tau[v], s); }
    logits[v] = x; if (x > best || (x == best && v < bi)) { best = x; bi = v; } }
  sv[threadIdx.x] = best; si[threadIdx.x] = bi; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) { const float o = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
    if (o > sv[threadIdx.x] || (o == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = o; si[threadIdx.x] = oi; } } __syncthreads(); }
  if (threadIdx.x == 0) { A.blk_val[blockIdx.y * 256 + blockIdx.x] = sv[0]; A.blk_idx[blockIdx.y * 256 + blockIdx.x] = si[0]; }
  if (!A.cnt) return;
  
  __shared__ int s_last; __threadfence(); __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + blockIdx.y, 1u) & 255u) == 255u;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  sv[threadIdx.x] = __ldcg(A.blk_val + blockIdx.y * 256 + threadIdx.x); si[threadIdx.x] = __ldcg(A.blk_idx + blockIdx.y * 256 + threadIdx.x); __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) { const float o = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
    if (o > sv[threadIdx.x] || (o == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = o; si[threadIdx.x] = oi; } } __syncthreads(); }
  if (threadIdx.x == 0) { A.d_tok_out[blockIdx.y] = si[0]; if (A.adv_pos) { *A.adv_pos += 1; if (A.adv_greedy) *A.adv_tok = si[0]; } }
}
__global__ void argmax_stage2(ArgmaxArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float sv[256]; __shared__ int si[256]; const int r = blockIdx.x;
  sv[threadIdx.x] = A.blk_val[r * 256 + threadIdx.x]; si[threadIdx.x] = A.blk_idx[r * 256 + threadIdx.x]; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) { const float o = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
    if (o > sv[threadIdx.x] || (o == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = o; si[threadIdx.x] = oi; } } __syncthreads(); }
  if (threadIdx.x == 0) A.d_tok_out[r] = si[0];
}



__device__ __forceinline__ bool tk_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__device__ __forceinline__ void tk_block_pop8(float (&tv)[8], int (&ti)[8], float* out_val, int* out_idx) {
  __shared__ float wv[8]; __shared__ int wi[8], wt[8]; __shared__ int s_win;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int k = 0; k < 8; ++k) {
    float v = tv[0]; int i = ti[0], t = tid;
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, v, o); const int oi = __shfl_xor_sync(0xffffffffu, i, o), ot = __shfl_xor_sync(0xffffffffu, t, o);
      if (tk_better(ov, oi, v, i)) { v = ov; i = oi; t = ot; } }
    if (lane == 0) { wv[warp] = v; wi[warp] = i; wt[warp] = t; }
    __syncthreads();
    if (tid == 0) { float bv = wv[0]; int bi = wi[0], bt = wt[0];
      for (int w = 1; w < 8; ++w) if (tk_better(wv[w], wi[w], bv, bi)) { bv = wv[w]; bi = wi[w]; bt = wt[w]; }
      out_val[k] = bv; out_idx[k] = bi; s_win = bt; }
    __syncthreads();
    if (tid == s_win) {
      #pragma unroll
      for (int q = 0; q < 7; ++q) { tv[q] = tv[q + 1]; ti[q] = ti[q + 1]; }
      tv[7] = -FLT_MAX; ti[7] = 0x7fffffff; }
    __syncthreads();
  }
}
__global__ void __launch_bounds__(256) topk8_stage1(TopkArgs A) { pdl_wait(); pdl_trigger();
  const int r = blockIdx.y; const float* lg = A.logits + (size_t)r * A.V; float tv[8]; int ti[8];
  #pragma unroll
  for (int q = 0; q < 8; ++q) { tv[q] = -FLT_MAX; ti[q] = 0x7fffffff; }
  for (int v = blockIdx.x * 256 + threadIdx.x; v < A.V; v += gridDim.x * 256) { float x = lg[v]; int ix = v;
    #pragma unroll
    for (int q = 0; q < 8; ++q) if (tk_better(x, ix, tv[q], ti[q])) { const float t0 = tv[q]; const int i0 = ti[q]; tv[q] = x; ti[q] = ix; x = t0; ix = i0; } }
  const size_t c = ((size_t)r * gridDim.x + blockIdx.x) * 8;
  tk_block_pop8(tv, ti, A.cand_val + c, A.cand_idx + c);
  if (!A.cnt) return;
  __shared__ int s_last; __threadfence(); __syncthreads();                  
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + r, 1u) & 255u) == 255u;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  { const size_t c2 = ((size_t)r * 256 + threadIdx.x) * 8;
    #pragma unroll
    for (int q = 0; q < 8; ++q) { tv[q] = __ldcg(A.cand_val + c2 + q); ti[q] = __ldcg(A.cand_idx + c2 + q); }
    __syncthreads();
    tk_block_pop8(tv, ti, A.top_val + (size_t)r * 8, A.top_idx + (size_t)r * 8); }
}
__global__ void __launch_bounds__(256) topk8_stage2(TopkArgs A) { pdl_wait(); pdl_trigger();
  const int r = blockIdx.x; float tv[8]; int ti[8]; const size_t c = ((size_t)r * 256 + threadIdx.x) * 8;
  #pragma unroll
  for (int q = 0; q < 8; ++q) { tv[q] = A.cand_val[c + q]; ti[q] = A.cand_idx[c + q]; }
  tk_block_pop8(tv, ti, A.top_val + (size_t)r * 8, A.top_idx + (size_t)r * 8);
}


__global__ void nll_stage1(NllArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float sm[256]; float mx = -FLT_MAX; const float* lg = A.logits + (size_t)blockIdx.y * A.V;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < A.V; v += gridDim.x * 256) mx = fmaxf(mx, lg[v]);
  sm[threadIdx.x] = mx; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]); __syncthreads(); }
  if (threadIdx.x == 0) A.blk_max[blockIdx.y * 256 + blockIdx.x] = sm[0];
}
__global__ void nll_stage2(NllArgs A) { pdl_wait(); pdl_trigger();
  __shared__ float sm[256]; __shared__ double sd[256]; const int r = blockIdx.x; const float* lg = A.logits + (size_t)r * A.V;
  sm[threadIdx.x] = A.blk_max[r * 256 + threadIdx.x]; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]); __syncthreads(); }
  const float M2 = sm[0]; double acc = 0.0;
  for (int v = threadIdx.x; v < A.V; v += 256) acc += exp((double)lg[v] - (double)M2);
  sd[threadIdx.x] = acc; __syncthreads();
  for (int s = 128; s; s >>= 1) { if (threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) A.out_nll[r] = (float)(log(sd[0]) + (double)M2 - (double)lg[A.d_target[r]]);
}
__global__ void advance_kernel(int* d_pos, int* d_tok, const int* d_next, int greedy) { pdl_wait(); pdl_trigger(); if (threadIdx.x == 0) { *d_pos += 1; if (greedy) *d_tok = *d_next; } }
__global__ void set_int_kernel(int* p, int v) { pdl_wait(); pdl_trigger(); if (threadIdx.x == 0) *p = v; }

void launch_embedding(const EmbArgs& A, cudaStream_t st) { cudaFuncSetAttribute(embedding_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, A.m * 512 * 4); launch_k(embedding_kernel, dim3(A.rows), dim3(512), A.m * 512 * 4, st, A); }
void launch_argmax(const ArgmaxArgs& A, cudaStream_t st) { if (A.us) launch_k(argmax_stage1<true>, dim3(dim3(256, A.rows)), dim3(256), 0, st, A); else launch_k(argmax_stage1<false>, dim3(dim3(256, A.rows)), dim3(256), 0, st, A);
  if (!A.cnt) launch_k(argmax_stage2, dim3(A.rows), dim3(256), 0, st, A); }
void launch_nll(const NllArgs& A, cudaStream_t st) { launch_k(nll_stage1, dim3(dim3(256, A.rows)), dim3(256), 0, st, A); launch_k(nll_stage2, dim3(A.rows), dim3(256), 0, st, A); }
void launch_topk8(const TopkArgs& A, cudaStream_t st) { launch_k(topk8_stage1, dim3(dim3(256, A.rows)), dim3(256), 0, st, A); if (!A.cnt) launch_k(topk8_stage2, dim3(A.rows), dim3(256), 0, st, A); }
void launch_advance(int* d_pos, int* d_tok, const int* d_next, int greedy, cudaStream_t st) { launch_k(advance_kernel, dim3(1), dim3(32), 0, st, d_pos, d_tok, d_next, greedy); }
void launch_set_int(int* p, int v, cudaStream_t st) { launch_k(set_int_kernel, dim3(1), dim3(32), 0, st, p, v); }
__global__ void set_ints_kernel(SetInts s) { pdl_wait(); pdl_trigger(); const int i = threadIdx.x; if (i < s.n) *s.p[i] = s.src[i] ? *s.src[i] : s.v[i]; }
void launch_set_ints(const SetInts& s, cudaStream_t st) { if (s.n > 0) launch_k(set_ints_kernel, dim3(1), dim3(32), 0, st, s); }


__global__ void verify_prologue_kernel(const int* vin, int* d_pos, int* d_toks, int n_host) { pdl_wait(); pdl_trigger();
  if (threadIdx.x == 0) *d_pos = vin[0];
  if (threadIdx.x < n_host) d_toks[threadIdx.x] = vin[1 + threadIdx.x];
}



__global__ void __launch_bounds__(256) prefetch_kernel(const PfEntry* plan, int n, const volatile int* prog) {
  for (int e = 0; e < n; ++e) {
    const PfEntry E = plan[e];
    if (threadIdx.x == 0) { long long spins = 0; while (*prog < E.gate && spins < (1ll << 24)) { __nanosleep(256); ++spins; } }
    __syncthreads();
    const char* base = reinterpret_cast<const char*>(E.p);
    const size_t lines = (E.bytes + 127) / 128;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < lines; i += (size_t)gridDim.x * blockDim.x)
      asm volatile("prefetch.global.L2 [%0];" :: "l"(base + i * 128));
  }
}
void launch_prefetch(const PfEntry* plan, int n, const int* prog, int ctas, cudaStream_t st) { prefetch_kernel<<<ctas, 256, 0, st>>>(plan, n, prog); }
void launch_verify_prologue(const int* vin, int* d_pos, int* d_toks, int n_host, cudaStream_t st) {
  launch_k(verify_prologue_kernel, dim3(1), dim3(32), 0, st, vin, d_pos, d_toks, n_host); }


__global__ void hist_spec_kernel(const float4* cap, float4* hist, const int* d_pos, int M, int H4, int max_ctx, size_t slot4, size_t hs4) {
  pdl_wait(); pdl_trigger();
  const int r = blockIdx.x, k = blockIdx.y, pos = *d_pos + r;
  if (r >= M || pos >= max_ctx) return;
  const float4* src = cap + k * slot4 + (size_t)r * H4; float4* dst = hist + k * hs4 + (size_t)pos * H4;
  for (int i = threadIdx.x; i < H4; i += blockDim.x) dst[i] = src[i];
}
void launch_hist_spec(const float* cap, float* hist, const int* d_pos, int M, int H, int max_ctx, size_t slot, size_t hs, cudaStream_t st) {
  launch_k(hist_spec_kernel, dim3(M, 3), dim3(256), 0, st, reinterpret_cast<const float4*>(cap), reinterpret_cast<float4*>(hist), d_pos, M, H / 4, max_ctx, slot / 4, hs / 4); }
} 
