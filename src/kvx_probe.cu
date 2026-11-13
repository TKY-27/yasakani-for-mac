// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "kvx_codec.cuh"
#include <cuda_runtime.h>
#include <vector>
using namespace yskn;
__global__ void codec_probe(const float* x,int8_t* q,float* scales,float* y,int rows) {
  __shared__ float buf[512];
  const int row=blockIdx.x*2+(threadIdx.x>>8),d=threadIdx.x&255;
  float v=x[row*256+d];
#if YSKN_KVX_ROT
  v=kvx_rotate(v,buf,false);
#endif
  kvx_store(v,q,scales,row,d);__syncthreads();
  v=__fmul_rn(kvx_scalar(q[row*256+d]),scales[row]);
#if YSKN_KVX_ROT
  v=kvx_rotate(v,buf,true);
#endif
  y[row*256+d]=v;
}
extern "C"  int yskn_kvx_codec() { return YSKN_KVX_MODE; }
extern "C"  int yskn_kvx_probe(const float* x,int8_t* q,float* scales,float* y,int rows) {
  if(rows<=0 || rows%2)return -1;
  const size_t n=(size_t)rows*256;float *dx=nullptr,*ds=nullptr,*dy=nullptr;int8_t* dq=nullptr;
  cudaError_t rc=cudaMalloc(&dx,n*4);
  if(rc==cudaSuccess)rc=cudaMalloc(&dy,n*4);
  if(rc==cudaSuccess)rc=cudaMalloc(&dq,n);
  if(rc==cudaSuccess)rc=cudaMalloc(&ds,rows*4);
  if(rc==cudaSuccess)rc=cudaMemcpy(dx,x,n*4,cudaMemcpyHostToDevice);
  if(rc==cudaSuccess){codec_probe<<<rows/2,512>>>(dx,dq,ds,dy,rows);rc=cudaGetLastError();}
  if(rc==cudaSuccess)rc=cudaDeviceSynchronize();
  if(rc==cudaSuccess)rc=cudaMemcpy(y,dy,n*4,cudaMemcpyDeviceToHost);
  if(rc==cudaSuccess)rc=cudaMemcpy(q,dq,n,cudaMemcpyDeviceToHost);
  if(rc==cudaSuccess)rc=cudaMemcpy(scales,ds,rows*4,cudaMemcpyDeviceToHost);
  cudaFree(dx);cudaFree(dy);cudaFree(dq);cudaFree(ds);return (int)rc;
}

#if YSKN_KVX_MODE == 3

__global__ void tb_probe_kernel(const unsigned char* blob, const float* x, const int* grp, const int* pos, float* y) {
  const int row = blockIdx.x * 2 + (threadIdx.x >> 8), d = threadIdx.x & 255;
  y[row * 256 + d] = tb_fakequant(x[row * 256 + d], blob, grp[row], pos[row]);
}
extern "C"  int yskn_tb_probe(const void* blob, size_t blob_len, const float* x, const int* grp, const int* pos, float* y, int rows) {
  if (rows <= 0 || rows % 2) return -1;
  const size_t n = (size_t)rows * 256; unsigned char* db = nullptr; float *dx = nullptr, *dy = nullptr; int *dg = nullptr, *dp = nullptr;
  cudaError_t rc = cudaMalloc(&db, blob_len);
  if (rc == cudaSuccess) rc = cudaMalloc(&dx, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dy, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dg, rows * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dp, rows * 4);
  if (rc == cudaSuccess) rc = cudaMemcpy(db, blob, blob_len, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dx, x, n * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dg, grp, rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dp, pos, rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) { tb_probe_kernel<<<rows / 2, 512>>>(db, dx, dg, dp, dy); rc = cudaGetLastError(); }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  if (rc == cudaSuccess) rc = cudaMemcpy(y, dy, n * 4, cudaMemcpyDeviceToHost);
  cudaFree(db); cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(dp);
  return (int)rc;
}
#endif

#if YSKN_KVX_MODE == 3
#include "kvq_fast.hpp"
#include <algorithm>


extern "C"  int yskn_tb_probe2(const void* blob, size_t blob_len, const float* x, const int* grp, const int* pos, float* y, int rows, int mode, int reps, float* ms_out) {
  if (rows <= 0 || (mode == 1 && rows % 2)) return -1;
  int nm = 0, has = 0;
  if (!tbf_parse_blob((const unsigned char*)blob, blob_len, nm, has) || nm < 1 || nm > TBF_MAXNM) return -2;
  const size_t n = (size_t)rows * 256; unsigned char *db = nullptr, *scr = nullptr; float *dx = nullptr, *dy = nullptr; int *dg = nullptr, *dp = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  cudaError_t rc = cudaMalloc(&db, blob_len);
  if (rc == cudaSuccess) rc = cudaMalloc(&dx, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dy, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dg, (size_t)rows * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dp, (size_t)rows * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&scr, tbf_scratch_bytes());
  if (rc == cudaSuccess) rc = cudaMemcpy(db, blob, blob_len, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dx, x, n * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dg, grp, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dp, pos, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e0);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e1);
  auto run = [&]() -> cudaError_t {
    if (mode == 1) { tb_probe_kernel<<<rows / 2, 512>>>(db, dx, dg, dp, dy); return cudaGetLastError(); }
    for (int r0 = 0; r0 < rows; r0 += TBF_TILE * 8) {
      TbFastArgs F{}; F.blob = db; F.nm = nm; F.has_cdq = has; F.yin = dx; F.grp_l = dg; F.pos_l = dp; F.yout = dy; F.row0 = r0; F.nrow = std::min(TBF_TILE * 8, rows - r0); F.scr = scr;
      F.vit2_off = (mode == 2 || mode == 4) ? 1 : 0;                         
      F.stage = (mode == 3 || mode == 4) ? 1 : (mode == 5 ? 2 : 3);
      launch_tb_fast(F, 0);
    }
    return cudaGetLastError();
  };
  if (rc == cudaSuccess) rc = run();
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  if (rc == cudaSuccess) rc = cudaMemcpy(y, dy, n * 4, cudaMemcpyDeviceToHost);
  for (int r = 0; rc == cudaSuccess && r < reps; ++r) {
    float ms = 0.f;
    cudaEventRecord(e0, 0); rc = run(); cudaEventRecord(e1, 0);
    if (rc == cudaSuccess) rc = cudaEventSynchronize(e1);
    if (rc == cudaSuccess) rc = cudaEventElapsedTime(&ms, e0, e1);
    if (ms_out) ms_out[r] = ms;
  }
  cudaFree(db); cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(dp); cudaFree(scr);
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  return (int)rc;
}
#endif

#if YSKN_KVX_MODE == 3

#include "kvq_fast.hpp"
#include "kvq_attn.cuh"
#include "q35.hpp"
#include <algorithm>
#include <vector>


extern "C"  int yskn_tbp_pack_unpack(const void* blob, size_t blob_len, const float* x, const int* grp, const int* pos, int rows, int stride, float* y_fq, unsigned char* pay, unsigned* side, float* y_dec, float* ms_out) {
  if (rows <= 0 || stride < 96 || stride % 4) return -1;
  int nm = 0, has = 0;
  if (!tbf_parse_blob((const unsigned char*)blob, blob_len, nm, has) || nm < 1 || nm > TBF_MAXNM) return -2;
  const size_t n = (size_t)rows * 256; unsigned char *db = nullptr, *scr = nullptr, *dpay = nullptr; unsigned* dside = nullptr; float *dx = nullptr, *dy = nullptr, *dd = nullptr; int *dg = nullptr, *dp = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
  cudaError_t rc = cudaMalloc(&db, blob_len);
  if (rc == cudaSuccess) rc = cudaMalloc(&dx, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dy, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dd, n * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dg, (size_t)rows * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&dp, (size_t)rows * 4);
  if (rc == cudaSuccess) rc = cudaMalloc(&scr, tbf_scratch_bytes());
  if (rc == cudaSuccess) rc = cudaMalloc(&dpay, (size_t)rows * stride);
  if (rc == cudaSuccess) rc = cudaMalloc(&dside, (size_t)rows * 8);
  if (rc == cudaSuccess) rc = cudaMemcpy(db, blob, blob_len, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dx, x, n * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dg, grp, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dp, pos, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemset(dpay, 0, (size_t)rows * stride);
  if (rc == cudaSuccess) rc = cudaMemset(dside, 0, (size_t)rows * 8);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e0);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e1);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e2);
  if (rc == cudaSuccess) {
    cudaEventRecord(e0, 0);
    for (int r0 = 0; r0 < rows; r0 += TBF_TILE * 8) {
      TbFastArgs F{}; F.blob = db; F.nm = nm; F.has_cdq = has; F.yin = dx; F.grp_l = dg; F.pos_l = dp; F.yout = dy; F.row0 = r0; F.nrow = std::min(TBF_TILE * 8, rows - r0); F.scr = scr;
      F.pk_pay = dpay; F.pk_side = dside; F.pk_stride = stride;
      launch_tb_fast(F, 0);
    }
    cudaEventRecord(e1, 0);
    tba_unpack_kernel<<<rows, 256>>>(db, dg, dp, dpay, dside, stride, dx, dd);
    cudaEventRecord(e2, 0);
    rc = cudaGetLastError();
  }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  if (rc == cudaSuccess) { float a = 0.f, b = 0.f; cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); if (ms_out) { ms_out[0] = a; ms_out[1] = b; } }
  if (rc == cudaSuccess) rc = cudaMemcpy(y_fq, dy, n * 4, cudaMemcpyDeviceToHost);
  if (rc == cudaSuccess) rc = cudaMemcpy(y_dec, dd, n * 4, cudaMemcpyDeviceToHost);
  if (rc == cudaSuccess) rc = cudaMemcpy(pay, dpay, (size_t)rows * stride, cudaMemcpyDeviceToHost);
  if (rc == cudaSuccess) rc = cudaMemcpy(side, dside, (size_t)rows * 8, cudaMemcpyDeviceToHost);
  cudaFree(db); cudaFree(dx); cudaFree(dy); cudaFree(dd); cudaFree(dg); cudaFree(dp); cudaFree(scr); cudaFree(dpay); cudaFree(dside);
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  if (e2) cudaEventDestroy(e2);
  return (int)rc;
}







template <int KB>
static bool tba_launch(int code, const TbAttnArgs& T, int sp, cudaStream_t st) {
  const dim3 g(4, sp, 1);
#define TBA_L(PFV, VARV) case ((PFV) * 256 + (VARV)): tba_attention_kernel<KB, PFV, VARV><<<g, 128, 0, st>>>(T); return true
  if constexpr (KB == 4) {
    switch (code) { TBA_L(4, 0); TBA_L(8, 0); TBA_L(8, 1); TBA_L(8, 35); TBA_L(8, 39); TBA_L(8, 47); TBA_L(8, 55); TBA_L(8, 63); TBA_L(4, 63); TBA_L(12, 63); default: return false; }
  } else {
    switch (code) { TBA_L(4, 0); TBA_L(8, 0); TBA_L(4, 35); TBA_L(4, 59); default: return false; }
  }
#undef TBA_L
  return false;
}
extern "C"  int yskn_tbp_bench(const void* blob, size_t blob_len, const float* xu, const int* posu, int U, int layer, const unsigned short* q16h, int N, int nlayers, int nsp, const int* splits, int reps,
                                                    int nvar, const int* codes, float* o_var, float* o_i8b, float* o_i8u, float* ms, float* xhat, float* meta) {
  if (U < 1 || N < 1 || nlayers < 1 || nsp < 1 || nsp > 16 || reps < 1 || nvar < 1 || nvar > 24 || !codes) return -1;
  const unsigned char* b = (const unsigned char*)blob;
  int nm = 0, has = 0;
  if (!tbf_parse_blob(b, blob_len, nm, has) || nm < 1 || nm > TBF_MAXNM) return -2;
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  int mode = -1, kk = -1;
  for (int kvh = 0; kvh < 4; ++kvh) for (int kind = 0; kind < 2; ++kind) {
    const int g = (layer * 4 + kvh) * 2 + kind; if ((unsigned)g >= u32(28)) return -3;
    const size_t go = u32(u32(40) + 4 * (size_t)g); const int m = b[go], k = b[go + 1];
    if (mode < 0) { mode = m; kk = k; } else if (m != mode || (m == 0 && k != kk)) return -5;       
  }
  if (mode != 0 && mode != 1) return -6;
  const int kb = mode == 1 ? 4 : kk; if (kb < 3 || kb > 5) return -6;
  const int stride = 32 * kb, rows = 8 * U, maxsp = *std::max_element(splits, splits + nsp);
  const size_t NR = (size_t)nlayers * 4 * N;                                
  unsigned char *db = nullptr, *scr = nullptr, *dpay_u = nullptr, *kpay = nullptr, *vpay = nullptr; unsigned *dside_u = nullptr, *kside = nullptr, *vside = nullptr;
  float *dxu = nullptr, *dxh = nullptr, *dqsu = nullptr, *pm = nullptr, *pl = nullptr, *pacc = nullptr, *dout = nullptr, *ksc = nullptr, *vsc = nullptr; int8_t *dq8u = nullptr, *k8 = nullptr, *v8 = nullptr;
  int *dgrp = nullptr, *dposu = nullptr, *dpos = nullptr; uint16_t* dq16 = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr; cudaStream_t st = nullptr;
  cudaError_t rc = cudaSuccess;
  auto A = [&](void** p, size_t n) { if (rc == cudaSuccess) rc = cudaMalloc(p, n); };
  A((void**)&db, blob_len); A((void**)&scr, tbf_scratch_bytes()); A((void**)&dxu, (size_t)rows * 1024); A((void**)&dxh, (size_t)rows * 1024); A((void**)&dgrp, (size_t)rows * 4); A((void**)&dposu, (size_t)rows * 4);
  A((void**)&dpay_u, (size_t)rows * stride); A((void**)&dside_u, (size_t)rows * 8); A((void**)&dq8u, (size_t)rows * 256); A((void**)&dqsu, (size_t)rows * 4);
  A((void**)&kpay, NR * stride); A((void**)&vpay, NR * stride); A((void**)&kside, NR * 8); A((void**)&vside, NR * 8);
  A((void**)&k8, NR * 256); A((void**)&v8, NR * 256); A((void**)&ksc, NR * 4); A((void**)&vsc, NR * 4);
  A((void**)&pm, (size_t)maxsp * 24 * 4); A((void**)&pl, (size_t)maxsp * 24 * 4); A((void**)&pacc, (size_t)maxsp * 24 * 256 * 4); A((void**)&dout, 24 * 256 * 4);
  A((void**)&dpos, 4); A((void**)&dq16, 24 * 256 * 2);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e0);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e1);
  if (rc == cudaSuccess) rc = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);       
  std::vector<int> grp(rows);
  for (int kvh = 0; kvh < 4; ++kvh) for (int kind = 0; kind < 2; ++kind) for (int u = 0; u < U; ++u) grp[(kvh * 2 + kind) * U + u] = (layer * 4 + kvh) * 2 + kind;
  const int posN = N - 1;
  if (rc == cudaSuccess) rc = cudaMemcpy(db, blob, blob_len, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dxu, xu, (size_t)rows * 1024, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dgrp, grp.data(), (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dposu, posu, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dpos, &posN, 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dq16, q16h, 24 * 256 * 2, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) {                                                  
    for (int r0 = 0; r0 < rows; r0 += TBF_TILE * 8) {
      TbFastArgs F{}; F.blob = db; F.nm = nm; F.has_cdq = has; F.yin = dxu; F.grp_l = dgrp; F.pos_l = dposu; F.yout = dxh; F.row0 = r0; F.nrow = std::min(TBF_TILE * 8, rows - r0); F.scr = scr;
      F.pk_pay = dpay_u; F.pk_side = dside_u; F.pk_stride = stride;
      launch_tb_fast(F, 0);
    }
    tba_i8_kernel<<<rows / 2, 512>>>(dxh, dq8u, dqsu);
    rc = cudaGetLastError();
  }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  if (rc == cudaSuccess && xhat) rc = cudaMemcpy(xhat, dxh, (size_t)rows * 1024, cudaMemcpyDeviceToHost);
  if (rc == cudaSuccess) {                                                  
    const int T = 128, gx = (N + T - 1) / T;
    for (int l = 0; l < nlayers; ++l) {
      const size_t lr = (size_t)l * 4 * N;
      tba_tile_kernel<<<dim3(gx, 4), T>>>(dpay_u, kpay, U, N, N, stride, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>(dpay_u + (size_t)U * stride, vpay, U, N, N, stride, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dside_u, (unsigned char*)kside, U, N, N, 8, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dside_u + (size_t)U * 8, (unsigned char*)vside, U, N, N, 8, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dq8u, (unsigned char*)k8, U, N, N, 256, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dq8u + (size_t)U * 256, (unsigned char*)v8, U, N, N, 256, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dqsu, (unsigned char*)ksc, U, N, N, 4, 2 * U, (int)lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dqsu + (size_t)U * 4, (unsigned char*)vsc, U, N, N, 4, 2 * U, (int)lr);
    }
    rc = cudaGetLastError();
  }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  bool bad_code = false;
  auto run_tb = [&](int l, int sp, int code) {
    TbAttnArgs T{}; T.blob = db; T.layer = layer; T.q16 = dq16; T.kpay = kpay + (size_t)l * 4 * N * stride; T.vpay = vpay + (size_t)l * 4 * N * stride;
    T.kside = kside + (size_t)l * 4 * N * 2; T.vside = vside + (size_t)l * 4 * N * 2; T.max_ctx = N; T.M = 1; T.splits = sp; T.d_pos = dpos; T.pm = pm; T.pl = pl; T.pacc = pacc;
    const bool ok = kb == 3 ? tba_launch<3>(code, T, sp, st) : (kb == 4 ? tba_launch<4>(code, T, sp, st) : tba_launch<5>(code, T, sp, st));
    if (!ok) bad_code = true;
  };
  auto run_i8 = [&](int l, int sp, int which) {
    Q35AttnArgs T{}; T.q16 = dq16; T.kc = nullptr; T.vc = nullptr; T.max_ctx = N; T.M = 1; T.d_pos = dpos; T.splits = sp; T.pm = pm; T.pl = pl; T.pacc = pacc; T.early = 0;
    T.k8 = k8 + (size_t)l * 4 * N * 256; T.v8 = v8 + (size_t)l * 4 * N * 256; T.ks = ksc + (size_t)l * 4 * N; T.vs = vsc + (size_t)l * 4 * N; T.kv_fp16 = 0; T.vsp = 3;
    if (which == 2) launch_q35_attention(T, st); else launch_q35_atx(T, nullptr, st);
  };
  int last_atx = -1; for (int i = 0; i < nsp; ++i) if (splits[i] <= 64) last_atx = i;               
  try {
    for (int si = 0; si < nsp && rc == cudaSuccess; ++si) {
      const int sp = splits[si];
      for (int which = 0; which < nvar + 2 && rc == cudaSuccess; ++which) {
        const int mcol = si * (nvar + 2) + which;
        if (which == nvar + 1 && sp > 64) { ms[mcol] = -1.f; continue; }                               
        auto sweep = [&]() { for (int l = 0; l < nlayers; ++l) { if (which < nvar) run_tb(l, sp, codes[which]); else run_i8(l, sp, which - nvar + 2); } };
        cudaMemsetAsync(pm, 0, (size_t)maxsp * 24 * 4, st); cudaMemsetAsync(pl, 0, (size_t)maxsp * 24 * 4, st); cudaMemsetAsync(pacc, 0, (size_t)maxsp * 24 * 256 * 4, st);
        sweep(); if (bad_code) { rc = (cudaError_t)1000; break; } rc = cudaGetLastError(); if (rc == cudaSuccess) rc = cudaStreamSynchronize(st);    
        std::vector<float> t;
        for (int r = 0; r < reps && rc == cudaSuccess; ++r) { float m = 0.f; cudaEventRecord(e0, st); sweep(); cudaEventRecord(e1, st); rc = cudaEventSynchronize(e1); if (rc == cudaSuccess) cudaEventElapsedTime(&m, e0, e1); t.push_back(m); }
        if (rc != cudaSuccess) break;
        std::sort(t.begin(), t.end()); ms[mcol] = t[t.size() / 2];
        if (si == (which == nvar + 1 ? last_atx : nsp - 1)) {                 
          cudaMemsetAsync(pm, 0, (size_t)maxsp * 24 * 4, st); cudaMemsetAsync(pl, 0, (size_t)maxsp * 24 * 4, st); cudaMemsetAsync(pacc, 0, (size_t)maxsp * 24 * 256 * 4, st);
          if (which < nvar) run_tb(0, sp, codes[which]); else run_i8(0, sp, which - nvar + 2);
          tba_combine_kernel<<<24, 256, 0, st>>>(pm, pl, pacc, sp, dout);
          rc = cudaGetLastError(); if (rc == cudaSuccess) rc = cudaStreamSynchronize(st);
          float* dst = which < nvar ? (o_var ? o_var + (size_t)which * 24 * 256 : nullptr) : (which == nvar ? o_i8b : o_i8u);
          if (rc == cudaSuccess && dst) rc = cudaMemcpy(dst, dout, 24 * 256 * 4, cudaMemcpyDeviceToHost);
        }
      }
    }
  } catch (...) { rc = cudaErrorUnknown; }
  if (meta) { meta[0] = (float)stride; meta[1] = 8.f; meta[2] = (float)kb; meta[3] = (float)mode; }
  cudaFree(db); cudaFree(scr); cudaFree(dxu); cudaFree(dxh); cudaFree(dgrp); cudaFree(dposu); cudaFree(dpay_u); cudaFree(dside_u); cudaFree(dq8u); cudaFree(dqsu);
  cudaFree(kpay); cudaFree(vpay); cudaFree(kside); cudaFree(vside); cudaFree(k8); cudaFree(v8); cudaFree(ksc); cudaFree(vsc); cudaFree(pm); cudaFree(pl); cudaFree(pacc); cudaFree(dout); cudaFree(dpos); cudaFree(dq16);
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  if (st) cudaStreamDestroy(st);
  return (int)rc;
}
#endif

#if YSKN_KVX_MODE == 3

#include "kvq_atxpk.hpp"
#include "q35_atxpk.hpp"
__global__ void atxpk_side_scl_kernel(const unsigned* side, float* scl, int rows) { const int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < rows) scl[r] = __uint_as_float(side[2 * (size_t)r]); }











extern "C"  int yskn_atxpk_bench(const void* blob, size_t blob_len, const float* xu, const int* posu, int U, int layer, const unsigned short* q16h, int M, int N, int nlayers, int nsp, const int* splits,
                                                     int reps, int kmode, float* o_pk, float* o_i8, float* ms, float* xhat, float* meta, int* inv) {
  if (U < 1 || N < 2 * M || M < 1 || M > 64 || nlayers < 1 || nsp < 1 || nsp > 16 || reps < 1 || (kmode & 3) == 0) return -1;
  const unsigned char* b = (const unsigned char*)blob;
  int nm = 0, has = 0;
  if (!tbf_parse_blob(b, blob_len, nm, has) || nm < 1 || nm > TBF_MAXNM) return -2;
  const int kb = kvqaxpk_kb(b, blob_len), PB = 32 * kb;
  for (int kind = 0; kind < 2; ++kind) if ((kmode >> kind) & 1) if (!kvqaxpk_packable(b, blob_len, layer, kind, kb)) return -8;
  if (kb != 4 && !(kmode & 1)) return -10;                                                                  
  std::vector<unsigned char> rec; if (!kvqaxpk_build(b, blob_len, rec)) return -9;
  const int rows = 8 * U, maxsp = *std::max_element(splits, splits + nsp);
  const size_t NR = (size_t)nlayers * 4 * N;
  unsigned char *db = nullptr, *scr = nullptr, *dpay_u = nullptr, *kpay = nullptr, *vpay = nullptr, *drec = nullptr; unsigned* dside_u = nullptr;
  float *dxu = nullptr, *dxh = nullptr, *dqsu = nullptr, *dscl_u = nullptr, *pm = nullptr, *pl = nullptr, *pacc = nullptr, *dout = nullptr, *ksc = nullptr, *vsc = nullptr, *kscp = nullptr, *vscp = nullptr;
  int8_t *dq8u = nullptr, *k8 = nullptr, *v8 = nullptr;
  int *dgrp = nullptr, *dposu = nullptr, *dpos = nullptr, *dpos1 = nullptr; uint16_t* dq16 = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr; cudaStream_t st = nullptr;
  cudaError_t rc = cudaSuccess;
  auto A = [&](void** p, size_t n) { if (rc == cudaSuccess) rc = cudaMalloc(p, n); };
  A((void**)&db, blob_len); A((void**)&drec, rec.size()); A((void**)&scr, tbf_scratch_bytes()); A((void**)&dxu, (size_t)rows * 1024); A((void**)&dxh, (size_t)rows * 1024); A((void**)&dgrp, (size_t)rows * 4);
  A((void**)&dposu, (size_t)rows * 4); A((void**)&dpay_u, (size_t)rows * PB); A((void**)&dside_u, (size_t)rows * 8); A((void**)&dq8u, (size_t)rows * 256); A((void**)&dqsu, (size_t)rows * 4); A((void**)&dscl_u, (size_t)rows * 4);
  A((void**)&kpay, NR * PB); A((void**)&vpay, NR * PB); A((void**)&kscp, NR * 4); A((void**)&vscp, NR * 4);
  A((void**)&k8, NR * 256); A((void**)&v8, NR * 256); A((void**)&ksc, NR * 4); A((void**)&vsc, NR * 4);
  A((void**)&pm, (size_t)M * maxsp * 24 * 4); A((void**)&pl, (size_t)M * maxsp * 24 * 4); A((void**)&pacc, (size_t)M * maxsp * 24 * 256 * 4); A((void**)&dout, (size_t)M * 24 * 256 * 4);
  A((void**)&dpos, 4); A((void**)&dpos1, 4); A((void**)&dq16, (size_t)M * 24 * 256 * 2);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e0);
  if (rc == cudaSuccess) rc = cudaEventCreate(&e1);
  if (rc == cudaSuccess) rc = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  std::vector<int> grp(rows);
  for (int kvh = 0; kvh < 4; ++kvh) for (int kind = 0; kind < 2; ++kind) for (int u = 0; u < U; ++u) grp[(kvh * 2 + kind) * U + u] = (layer * 4 + kvh) * 2 + kind;
  const int base = N - M;
  if (rc == cudaSuccess) rc = cudaMemcpy(db, blob, blob_len, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(drec, rec.data(), rec.size(), cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dxu, xu, (size_t)rows * 1024, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dgrp, grp.data(), (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dposu, posu, (size_t)rows * 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dpos, &base, 4, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemcpy(dq16, q16h, (size_t)M * 24 * 256 * 2, cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) rc = cudaMemset(dpay_u, 0, (size_t)rows * PB);
  if (rc == cudaSuccess) {
    for (int r0 = 0; r0 < rows; r0 += TBF_TILE * 8) {
      TbFastArgs F{}; F.blob = db; F.nm = nm; F.has_cdq = has; F.yin = dxu; F.grp_l = dgrp; F.pos_l = dposu; F.yout = dxh; F.row0 = r0; F.nrow = std::min(TBF_TILE * 8, rows - r0); F.scr = scr;
      F.pk_pay = dpay_u; F.pk_side = dside_u; F.pk_stride = PB;
      launch_tb_fast(F, 0);
    }
    tba_i8_kernel<<<rows / 2, 512>>>(dxh, dq8u, dqsu);
    atxpk_side_scl_kernel<<<(rows + 255) / 256, 256>>>(dside_u, dscl_u, rows);
    rc = cudaGetLastError();
  }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  if (rc == cudaSuccess && xhat) rc = cudaMemcpy(xhat, dxh, (size_t)rows * 1024, cudaMemcpyDeviceToHost);
  if (rc == cudaSuccess) {
    const int T = 128, gx = (N + T - 1) / T;
    for (int l = 0; l < nlayers; ++l) {
      const int lr = l * 4 * N;
      tba_tile_kernel<<<dim3(gx, 4), T>>>(dpay_u, kpay, U, N, N, PB, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>(dpay_u + (size_t)U * PB, vpay, U, N, N, PB, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dscl_u, (unsigned char*)kscp, U, N, N, 4, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dscl_u + (size_t)U * 4, (unsigned char*)vscp, U, N, N, 4, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dq8u, (unsigned char*)k8, U, N, N, 256, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dq8u + (size_t)U * 256, (unsigned char*)v8, U, N, N, 256, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dqsu, (unsigned char*)ksc, U, N, N, 4, 2 * U, lr);
      tba_tile_kernel<<<dim3(gx, 4), T>>>((const unsigned char*)dqsu + (size_t)U * 4, (unsigned char*)vsc, U, N, N, 4, 2 * U, lr);
    }
    rc = cudaGetLastError();
  }
  if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
  const bool kpk = (kmode & 1) != 0, vpk = (kmode & 2) != 0;
  auto run = [&](int l, int sp, int which, int Mq, const int* dp, const uint16_t* q16) {
    Q35AttnArgs T{}; T.q16 = q16; T.kc = nullptr; T.vc = nullptr; T.max_ctx = N; T.M = Mq; T.d_pos = dp; T.splits = sp; T.pm = pm; T.pl = pl; T.pacc = pacc; T.early = (kmode >> 2) & 1;      
    const size_t lr = (size_t)l * 4 * N;
    T.k8 = k8 + lr * 256; T.v8 = v8 + lr * 256; T.ks = ksc + lr; T.vs = vsc + lr; T.kv_fp16 = 0; T.vsp = 3;
    if (which == 0 || which == 3) {                                                                         
      Q35PkArgs P{}; P.kp = kpk ? kpay + lr * PB : nullptr; P.vp = vpk ? vpay + lr * PB : nullptr; P.kps = kscp + lr; P.vps = vscp + lr; P.rec = drec; P.layer = layer; P.kb = kb; P.variant = which == 3 ? 1 : 0;
      if (!kpk) { T.k8 = k8 + lr * 256; T.ks = ksc + lr; } if (!vpk) { T.v8 = v8 + lr * 256; T.vs = vsc + lr; }
      launch_q35_atx_pk(T, nullptr, P, st);
    } else { T.vsp = which == 1 ? 3 : 0; launch_q35_atx(T, nullptr, st); }
  };
  auto combine = [&](int sp, int Mq) { for (int tk = 0; tk < Mq; ++tk) tba_combine_kernel<<<24, 256, 0, st>>>(pm + (size_t)tk * sp * 24, pl + (size_t)tk * sp * 24, pacc + (size_t)tk * sp * 24 * 256, sp, dout + (size_t)tk * 24 * 256); };
  int inv_bad = 0, inv_n = 0; long long eq_bad = 0, eq_n = 0;
  try {
    for (int si = 0; si < nsp && rc == cudaSuccess; ++si) {
      const int sp = splits[si];
      for (int which = 0; which < 4 && rc == cudaSuccess; ++which) {
        if (which == 3 && kb != 4) { ms[si * 4 + which] = -1.f; continue; }
        auto sweep = [&]() { for (int l = 0; l < nlayers; ++l) run(l, sp, which, M, dpos, dq16); };
        cudaMemsetAsync(pm, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pl, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pacc, 0, (size_t)M * maxsp * 24 * 256 * 4, st);
        sweep(); rc = cudaGetLastError(); if (rc == cudaSuccess) rc = cudaStreamSynchronize(st);              
        std::vector<float> t;
        for (int r = 0; r < reps && rc == cudaSuccess; ++r) { float m = 0.f; cudaEventRecord(e0, st); sweep(); cudaEventRecord(e1, st); rc = cudaEventSynchronize(e1); if (rc == cudaSuccess) cudaEventElapsedTime(&m, e0, e1); t.push_back(m); }
        if (rc != cudaSuccess) break;
        std::sort(t.begin(), t.end()); ms[si * 4 + which] = t[t.size() / 2];
        if (which <= 1 || which == 3) {                                                                     
          cudaMemsetAsync(pm, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pl, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pacc, 0, (size_t)M * maxsp * 24 * 256 * 4, st);
          run(0, sp, which, M, dpos, dq16); combine(sp, M);
          rc = cudaGetLastError(); if (rc == cudaSuccess) rc = cudaStreamSynchronize(st);
          std::vector<float> v1o; float* dst = (which == 0 ? o_pk : o_i8) + (size_t)si * M * 24 * 256;
          if (which == 3) { v1o.resize((size_t)M * 24 * 256); dst = v1o.data(); }
          if (rc == cudaSuccess && dst) rc = cudaMemcpy(dst, dout, (size_t)M * 24 * 256 * 4, cudaMemcpyDeviceToHost);
          if (which == 3 && rc == cudaSuccess && o_pk) {                                                     
            const float* a = o_pk + (size_t)si * M * 24 * 256;
            for (size_t i = 0; i < v1o.size(); ++i) { ++eq_n; unsigned atx, ub; memcpy(&atx, a + i, 4); memcpy(&ub, v1o.data() + i, 4); if (atx != ub) ++eq_bad; }
          }
          if (which == 0 && rc == cudaSuccess && si == 0) {                                                  
            std::vector<float> full((size_t)M * 24 * 256), one(24 * 256);
            if (dst) memcpy(full.data(), dst, full.size() * 4);
            for (int tk = 0; tk < M && rc == cudaSuccess; ++tk) {
              const int bp = base + tk; cudaMemcpy(dpos1, &bp, 4, cudaMemcpyHostToDevice);
              cudaMemsetAsync(pm, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pl, 0, (size_t)M * maxsp * 24 * 4, st); cudaMemsetAsync(pacc, 0, (size_t)M * maxsp * 24 * 256 * 4, st);
              run(0, sp, 0, 1, dpos1, dq16 + (size_t)tk * 24 * 256); combine(sp, 1);
              rc = cudaGetLastError(); if (rc == cudaSuccess) rc = cudaStreamSynchronize(st);
              if (rc == cudaSuccess) rc = cudaMemcpy(one.data(), dout, 24 * 256 * 4, cudaMemcpyDeviceToHost);
              if (rc == cudaSuccess) { ++inv_n; if (memcmp(one.data(), full.data() + (size_t)tk * 24 * 256, 24 * 256 * 4) != 0) ++inv_bad; }
            }
          }
        }
      }
    }
  } catch (...) { rc = cudaErrorUnknown; }
  if (inv) { inv[0] = inv_bad; inv[1] = inv_n; inv[2] = (int)std::min<long long>(eq_bad, 2147483647LL); inv[3] = (int)std::min<long long>(eq_n, 2147483647LL); }
  if (meta) { meta[0] = (float)PB; meta[1] = 4.f; meta[2] = (float)((kpk ? PB + 4 : 260) + (vpk ? PB + 4 : 260)); meta[3] = (float)kb; }
  cudaFree(db); cudaFree(drec); cudaFree(scr); cudaFree(dxu); cudaFree(dxh); cudaFree(dgrp); cudaFree(dposu); cudaFree(dpay_u); cudaFree(dside_u); cudaFree(dq8u); cudaFree(dqsu); cudaFree(dscl_u);
  cudaFree(kpay); cudaFree(vpay); cudaFree(kscp); cudaFree(vscp); cudaFree(k8); cudaFree(v8); cudaFree(ksc); cudaFree(vsc); cudaFree(pm); cudaFree(pl); cudaFree(pacc); cudaFree(dout); cudaFree(dpos); cudaFree(dpos1); cudaFree(dq16);
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  if (st) cudaStreamDestroy(st);
  return (int)rc;
}
#endif
