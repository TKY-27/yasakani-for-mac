// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once


#include <cuda_runtime.h>
#include <cstdint>
#include "post_kernels.hpp"
namespace yskn {
struct AttnArgs { const uint16_t* q16; KvLayer kv; int max_ctx; int M; const int* d_pos; int splits; float* pm; float* pl; float* pacc; float rope_theta;
                  const float2* rope = nullptr; };   
void launch_rope_table(float2* out, int max_ctx, float theta, cudaStream_t);
struct AttnCombineArgs { const float* pm; const float* pl; const float* pacc; int splits; int M; const uint16_t* bv; uint16_t* o16; };

struct AttnGArgs { const uint16_t* q16; KvLayer kv; int max_ctx; int M; const int* d_pos; int splits; float* pm; float* pl; float* pacc;
                   const float2* rope; int kvh; int nh; float qscale; };   
struct AttnGCombineArgs { const float* pm; const float* pl; const float* pacc; int splits; int M; const uint16_t* bv; uint16_t* o16;
                          const uint16_t* gate; int nh; int g; };            
void launch_attention_gqa(const AttnGArgs&, int g, int hd, int rd, cudaStream_t);
void launch_attn_combine_gqa_xform(const AttnGCombineArgs&, int hd, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t);
void launch_rope_table_g(float2* out, int max_ctx, float theta, int rd, cudaStream_t);
struct EmbArgs { const uint32_t* stream; int wps; int seqs; int K; const uint16_t* lut16; const uint16_t* scale16; const int8_t* signs; const uint16_t* qm; int m; const int* d_tok; float* h; int rows = 1;
                 const uint16_t* gamma = nullptr; float eps = 0.f; const int8_t* signs_in = nullptr; int8_t* qx = nullptr; float* alpha = nullptr; };   
struct ArgmaxArgs { const float* part; const float* tau; int V; float* logits; int* d_tok_out; float* blk_val; int* blk_idx; int nw = 1; int M = 1; int row = 0; int rows = 1;
                    const float* us = nullptr; const float* rr = nullptr; float xsm_eps = 1e-6f;    
                    unsigned* cnt = nullptr; int* adv_pos = nullptr; int* adv_tok = nullptr; int adv_greedy = 0;
                    int xsm_nb = 7, xsm_stride = 8; float xsm_n = 3584.f; };   
struct NllArgs { const float* logits; int V; const int* d_target; float* out_nll; float* blk_max; float* blk_sum; int rows = 1; };   

struct TopkArgs { const float* logits; int V; float* cand_val; int* cand_idx; float* top_val; int* top_idx; int rows; unsigned* cnt = nullptr; };   
void launch_attention(const AttnArgs&, cudaStream_t);
void launch_attn_combine(const AttnCombineArgs&, cudaStream_t);
void launch_attn_combine_xform(const AttnCombineArgs&, const int8_t* signs, int8_t* qx, float* alpha, cudaStream_t);   
void launch_attention_prefill(const AttnArgs&, const uint16_t* bv, uint16_t* o16, cudaStream_t);   



struct KvF32 { float4* kq; float4* ks; float4* v; };
void launch_kv_dequant(const AttnArgs&, const KvF32&, cudaStream_t);
void launch_attention_f32(const AttnArgs&, const KvF32&, cudaStream_t);
void launch_attention_prefill_f32(const AttnArgs&, const KvF32&, const uint16_t* bv, uint16_t* o16, cudaStream_t);
void launch_embedding(const EmbArgs&, cudaStream_t);
void launch_argmax(const ArgmaxArgs&, cudaStream_t);
void launch_nll(const NllArgs&, cudaStream_t);
void launch_topk8(const TopkArgs&, cudaStream_t);
void launch_advance(int* d_pos, int* d_tok, const int* d_next, int greedy, cudaStream_t);
void launch_set_int(int* p, int v, cudaStream_t);


struct SetInts { int* p[6] = {}; const int* src[6] = {}; int v[6] = {}; int n = 0;
                 void set(int* q, int x) { p[n] = q; src[n] = nullptr; v[n] = x; ++n; } void copy(int* q, const int* s) { p[n] = q; src[n] = s; v[n] = 0; ++n; } };
void launch_set_ints(const SetInts& s, cudaStream_t);
void launch_verify_prologue(const int* vin, int* d_pos, int* d_toks, int n_host, cudaStream_t);   
struct PfEntry { const void* p; size_t bytes; int gate; };      
void launch_prefetch(const PfEntry* plan, int n, const int* prog, int ctas, cudaStream_t);
void launch_hist_spec(const float* cap, float* hist, const int* d_pos, int M, int H, int max_ctx, size_t slot, size_t hs, cudaStream_t);
}
