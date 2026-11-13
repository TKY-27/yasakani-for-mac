// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef YSKN_KVX_MODE
#define YSKN_KVX_MODE 0
#endif
#if YSKN_KVX_MODE == 2 || YSKN_KVX_MODE == 3
#define YSKN_KVX_ROT 1      
#else
#define YSKN_KVX_ROT 0
#endif
#if YSKN_KVX_MODE == 3
#include "kvq_gpu.cuh"   
#endif
namespace yskn {

__device__ __forceinline__ float kvx_scalar(int8_t q) {
#if YSKN_KVX_MODE == 1
  __nv_fp8_e4m3 v; v.__x = (unsigned char)q; return (float)v;
#else
  return (float)q;
#endif
}
__device__ __forceinline__ float kvx_sign(int d) {
  unsigned x=(unsigned)d+1u;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return (x & 1u) ? -1.f : 1.f;
}


__device__ __forceinline__ float kvx_rotate(float x, float* buf, bool inverse) {
  const int tid=threadIdx.x, d=tid & 255;
  buf[tid]=inverse ? x : __fmul_rn(x,kvx_sign(d));
  __syncthreads();
  #pragma unroll
  for (int h=1;h<256;h*=2) {
    const float a=buf[tid], b=buf[tid ^ h];
    __syncthreads();
    buf[tid]=(tid & h) ? __fsub_rn(b,a) : __fadd_rn(a,b);
    __syncthreads();
  }
  float y=__fmul_rn(buf[tid],0.0625f);
  if (inverse) y=__fmul_rn(y,kvx_sign(d));
  return y;
}
__device__ __forceinline__ void kvx_store(float x,int8_t* q8,float* qs,size_t row,int d) {
  __shared__ unsigned mx[16];
  const int tid=threadIdx.x;
  unsigned m=__float_as_uint(fabsf(x));
  for(int o=16;o;o>>=1)m=max(m,__shfl_xor_sync(0xffffffffu,m,o));
  if((tid&31)==0)mx[tid>>5]=m;
  __syncthreads();
  const int w0=(tid>>8)*8;unsigned mm=0;
  #pragma unroll
  for(int w=0;w<8;++w)mm=max(mm,mx[w0+w]);
#if YSKN_KVX_MODE == 1
  const float sc=__fdiv_rn(__uint_as_float(mm),448.f);
  const unsigned char q=__nv_cvt_float_to_fp8(sc>0.f?__fdiv_rn(x,sc):0.f,__NV_SATFINITE,__NV_E4M3);
  q8[row*256+d]=(int8_t)q;
#else
  const float sc=__fdiv_rn(__uint_as_float(mm),127.f);
  int q=sc>0.f?__float2int_rn(__fdiv_rn(x,sc)):0;
  q8[row*256+d]=(int8_t)max(-127,min(127,q));
#endif
  if(d==0)qs[row]=sc;
}
} 

namespace yskn {
__device__ __forceinline__ void kvx_rotate_warp16(float (&x)[16]) {
  const int lane=threadIdx.x&31;
  #pragma unroll
  for(int i=0;i<16;++i)x[i]=__fmul_rn(x[i],kvx_sign(lane+32*(i&7)));
  #pragma unroll
  for(int h=1;h<32;h*=2) {
    #pragma unroll
    for(int i=0;i<16;++i){const float a=x[i],b=__shfl_xor_sync(0xffffffffu,a,h);x[i]=(lane&h)?__fsub_rn(b,a):__fadd_rn(a,b);}
  }
  #pragma unroll
  for(int h=1;h<8;h*=2) {
    #pragma unroll
    for(int i=0;i<16;++i)if(!(i&h)){const float a=x[i],b=x[i+h];x[i]=__fadd_rn(a,b);x[i+h]=__fsub_rn(a,b);}
  }
  #pragma unroll
  for(int i=0;i<16;++i)x[i]=__fmul_rn(x[i],0.0625f);
}
}
