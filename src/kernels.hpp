// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
namespace yskn {
struct XformDesc;   
constexpr int GEMV_WARPS = 4;      
constexpr int GEMV_NT_PASS = 4;    
constexpr int GEMV_STAGES = 2;     
struct TensorDev {                        
  const uint4*    frag;                   
  const uint64_t* tile_u4off;             
  const uint8_t*  tile_k;                 
  int             k_default;
  const uint16_t* lut8;                   
  const float*    tau;                    
  int N, Kc;
  const int8_t*   group_worker;           
  int n_workers;
  const uint4*    lno = nullptr;          
};
struct GemvSeg { TensorDev t; float* part; int cta_begin; };     
struct GemvLaunch {
  GemvSeg seg[3]; int n_seg; int n_ctas; int max_workers;
  const int8_t* qx; const float* alpha;                          
  const float* h32; const uint16_t* h16; const uint16_t* gamma; float eps; int use_norm;   
  const int8_t* signs; const uint16_t* qm; int m;                
  int fuse; int M; int stage_bytes;
  const uint16_t* r8 = nullptr; int draft = 0;                 
  int pdl_prew_m = 1 << 30; int prew = 1;                 
  int pre = 3;                                                 
  int xsum = 0; float* us_out = nullptr;                     
  int xsm_part = 0;                                            
  int lno_cfg = -1;                                            
  int lo = 0, lo_t0 = 0;                                       
  const float* xsm_S = nullptr; int xsm_sst = 16, xsm_nb = 10; float xsm_n = 5120.f, xsm_eps = 1e-6f;   
  
  int e_fuse = 0, e_u16 = 0; float* e_h = nullptr; const float* e_tau = nullptr; const int8_t* e_sg_out = nullptr;
  const uint16_t* e_gamma = nullptr; const int8_t* e_sg_in = nullptr; int8_t* e_qx = nullptr; float* e_alpha = nullptr;
  float* e_S = nullptr; unsigned* e_cnt = nullptr; unsigned* e_dbg = nullptr;
  
  
  
  int f_mode = 0; const float* f_tau = nullptr; const int8_t* f_sg_out = nullptr; float* f_h = nullptr; const uint16_t* f_w = nullptr;
  const int8_t* f_sg_in = nullptr; int8_t* f_qx = nullptr; float* f_alpha = nullptr; float* f_S = nullptr; uint16_t* f_xu = nullptr;
  unsigned* f_cnt = nullptr;                                     
};
struct GemmLaunch { TensorDev t; float* part; const int8_t* qx; const float* alpha; int M; int stage_bytes;
                    int xsum = 0; float* us_out = nullptr; };   
size_t gemv_smem_bytes(const GemvLaunch& P);
bool   lnord_eligible(const GemvLaunch& P);                      
void   launch_lnord(const GemvLaunch& P, cudaStream_t st);
size_t lnord_bytes(int N, int Kc, int K);
void   lnord_build(const uint4* frag, int N, int Kc, int K, uint4* out, cudaStream_t st);
void   launch_gemv(const GemvLaunch& P, cudaStream_t st);


bool   pfa_eligible(const GemvLaunch& P);
void   launch_pfa(const GemvLaunch& P, int mode, int wpc, cudaStream_t st);
size_t gemm_smem_bytes(const GemmLaunch& P);
void   launch_gemm(const GemmLaunch& P, cudaStream_t st);
}
