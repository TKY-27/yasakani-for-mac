// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace yskn {
struct KvLayer {                 
  uint8_t*  kq;      
  uint16_t* kscale;  
  uint16_t* kzero;   
  uint16_t* kshadow; 
  uint8_t*  vq;      
  uint16_t* vscale;  
  uint16_t* vzero;   
};
struct QkvPostArgs {
  const float *part_q, *part_k, *part_v; const float *tau_q, *tau_k, *tau_v; int nw_q, nw_k, nw_v;
  const int8_t *sg_q, *sg_k, *sg_v; const uint16_t *bq, *bk;
  int M; const int* d_pos; uint16_t* q16; KvLayer kv; int max_ctx; float rope_theta;
  const float2* rope = nullptr;    
  const float* us = nullptr; const float* rr = nullptr; float xsm_eps = 1e-6f;   
                                                                                   
  int kvfuse = 0;                  
  int fy = 0;                      
};
struct KvQuantArgs { KvLayer kv; const int* d_pos; int M; int max_ctx; };
struct ResidualPostArgs { const float* part; const float* tau; int nw; const int8_t* sg; int M; int N; float* h; const float* us = nullptr; };          
struct MlpPostArgs { const float *part_g, *part_u; const float *tau_g, *tau_u; int nw; const int8_t *sg_g, *sg_u; int M; uint16_t* xff;
                     const float* us = nullptr; const float* rr = nullptr; float xsm_eps = 1e-6f; int F = 18944;    
                     int fy = 0; };   
struct XformA8Args { const uint16_t* x16; const int8_t* signs; const uint16_t* qm; int n, m; int8_t* qx; float* alpha; int M;
                     int xsum = 0; float* r_out = nullptr; unsigned* nf = nullptr; int stats = 0; };   


struct XsmResXArgs { const float* part; const float* us; const float* tau; const int8_t* sg_out; float* h; const uint16_t* gamma; const int8_t* sg_in;
                     int8_t* qx; float* alpha; float* S; int N; int M; int u16; unsigned* nf; int stats = 0;
                     int fy = 0; };   
void launch_xsum_res_xform(const XsmResXArgs&, cudaStream_t);
void launch_qkv_post(const QkvPostArgs&, cudaStream_t);
void launch_kv_group_quant(const KvQuantArgs&, cudaStream_t);
void launch_kv_group_quant_g(const KvQuantArgs&, int kvh, int hd, cudaStream_t);   
void launch_residual_post(const ResidualPostArgs&, cudaStream_t);
void launch_mlp_post(const MlpPostArgs&, cudaStream_t);


struct ResNormXArgs { ResidualPostArgs R; const float* h_in; float* h_out; const uint16_t* gamma; float eps; const int8_t* signs; int8_t* qx; float* alpha; };
void launch_residual_norm_xform(const ResNormXArgs&, cudaStream_t);


struct ResNormLastArgs { ResidualPostArgs R; const uint16_t* gamma; float eps; const int8_t* signs; int8_t* qx; float* alpha; unsigned* cnt; };
void launch_res_norm_last(const ResNormLastArgs&, cudaStream_t);
void launch_mlp_post_xform(const MlpPostArgs&, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t);   
void launch_xform_a8(const XformA8Args&, cudaStream_t);


void launch_norm_xform_a8(const float* h, const uint16_t* gamma, float eps, const XformA8Args&, cudaStream_t);
}
