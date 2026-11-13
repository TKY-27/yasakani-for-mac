// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include "a8_prologue.cuh"
#include "q35.hpp"
namespace yskn {


__global__ void __launch_bounds__(512) capture_a8_kernel(const float* x,int slot,int row0,int count,const int8_t* signs,int8_t* qx,float* alpha){
  pdl_wait();pdl_trigger();
  __shared__ float buf[512];__shared__ unsigned mx;
  const int g=blockIdx.x,m=blockIdx.y,t=threadIdx.x,i=g*512+t;
  if(t==0)mx=0;
  buf[t]=x[(size_t)m*Q35_H+i]*(float)signs[slot*Q35_H+i];
  __syncthreads();
  a8_group512_tail(buf,&mx,qx+((size_t)(row0+m)*count+slot)*Q35_H+g*512,
                  alpha+((size_t)(row0+m)*count+slot)*(Q35_H/512)+g);
}
void launch_q35_capture_a8(const float* x,int slot,int row0,int count,const int8_t* signs,int8_t* qx,float* alpha,int rows,cudaStream_t st){
  launch_k(capture_a8_kernel,dim3(Q35_H/512,rows),dim3(512),0,st,x,slot,row0,count,signs,qx,alpha);
}
}
