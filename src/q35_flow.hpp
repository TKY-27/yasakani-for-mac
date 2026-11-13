// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include "kernels.hpp"
namespace yskn {



struct Q35FlowMlpArgs {
  GemvLaunch gu, dn; int M = 0, F = 0, nwe_gu = 0, nwe_dn = 0;
  const float *tau_g = nullptr, *tau_u = nullptr; const int8_t *sg_g = nullptr, *sg_u = nullptr, *sg_dn_in = nullptr;
  const float* S = nullptr; float eps = 0.f; int8_t* qxd = nullptr; float* ald = nullptr;          
  const float* tau_dn = nullptr; const int8_t* sg_dn_out = nullptr; float* h = nullptr; const uint16_t* w = nullptr;
  const int8_t* sg_next = nullptr; int8_t* qx = nullptr; float* alpha = nullptr; float* S_out = nullptr; uint16_t* xu = nullptr;   
  unsigned long long* trace = nullptr;   
  unsigned* cnt = nullptr; unsigned* cnt_next = nullptr;   
};
void launch_q35_flow_mlp(const Q35FlowMlpArgs&, cudaStream_t);
void prepare_q35_flow_mlp(const Q35FlowMlpArgs&);   
}  
