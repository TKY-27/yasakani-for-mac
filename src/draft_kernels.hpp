// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace yskn {

struct D3PrepArgs { const float* src[3]; int stride[3]; const uint16_t* gain[3]; int nparts; float eps; const int8_t* signs; int8_t* qx; float* alpha; int M; };
struct D3LinOutArgs { const float* part; const float* tau; int nw; const int8_t* sg; int M; int N; float* out; int out_stride; };
void launch_d3_prep(const D3PrepArgs&, cudaStream_t);
void launch_d3_linear_out(const D3LinOutArgs&, cudaStream_t);
void launch_d3_gather(int* toks, int* verify_toks, const int* next8, int a, cudaStream_t);
void launch_d3_set_from(int* dst, const int* src, int add, cudaStream_t);
void launch_d3_map(const int* idx, const int* ids, int* verify_slot, int* cur_tok, cudaStream_t);
}
