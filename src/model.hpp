// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "format.hpp"
#include "kernels.hpp"
#include "post_kernels.hpp"
namespace yskn {
struct DevTensor {                       
  TensorDev dev{}; int bits = 6; int kmax = 6;            
  uint4* frag = nullptr; uint16_t* lut8 = nullptr; float* tau = nullptr; uint64_t* tile_off = nullptr; uint8_t* tile_k = nullptr; int8_t* plan = nullptr;
  int8_t *signs_in = nullptr, *signs_out = nullptr; uint16_t* qm_in = nullptr; int m_in = 1;
  uint32_t* row_stream = nullptr; uint16_t* lut16 = nullptr; uint16_t* scale16 = nullptr;   
  float* part = nullptr;                                                                     
  uint4* lno = nullptr;                                                                      
  int nw_big = 0;                                                                            
  std::vector<std::vector<int>> tiles_by_k;                                                 
};
struct DevLayer { DevTensor q, k, v, o, gate, up, down; uint16_t *ln1 = nullptr, *ln2 = nullptr, *bq = nullptr, *bk = nullptr, *bv = nullptr; KvLayer kv{}; };
struct Arena {                           
  int Mmax = 32; float* h = nullptr; float* h2 = nullptr;   uint16_t *q16 = nullptr, *o16 = nullptr, *xff = nullptr; int8_t* qx_ff = nullptr; float* alpha_ff = nullptr;
  float *pm = nullptr, *pl = nullptr, *pacc = nullptr; float* logits = nullptr; float* blk_val = nullptr; int* blk_idx = nullptr; float* blk_max = nullptr; float* blk_sum = nullptr;
  int *d_pos = nullptr, *d_tok = nullptr, *d_next = nullptr, *d_target = nullptr; float* d_nll = nullptr;
  
  
  int *d_toks = nullptr, *d_next8 = nullptr, *d_target8 = nullptr; float* d_nll8 = nullptr;
  float* tk_val = nullptr; int* tk_idx = nullptr; float* top_val = nullptr; int* top_idx = nullptr;
};
struct Model {
  Container c; ModelDesc cfg; std::vector<DevLayer> layers; DevTensor emb, head; uint16_t* final_norm = nullptr;
  int max_ctx = 2048, splits = 32, sm_count = 0; size_t stage_bytes_body = 0, stage_bytes_head = 0;
  void load(const std::string& path, int max_ctx_, int Mmax);
  DevTensor upload_linear(const TensorDesc& d, int Mmax, int want_workers, int force_workers = 0, const Container* src = nullptr);   
  void upload_embedding(const TensorDesc& d);
  uint16_t* upload_vector(const std::string& name, int n);
};
}
