// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <map>
#include <initializer_list>
#include <utility>
#include "model.hpp"
#include "args.hpp"
namespace yskn {


struct DraftModel {
  bool on = false; Container c; DevTensor fuse, fc_in, head; DevLayer L; int stage_bytes = 0, head_rows = 0;
  uint16_t *nf[3] = {nullptr, nullptr, nullptr}, *n_feat = nullptr, *n_emb = nullptr, *norm = nullptr; int* head_ids = nullptr;
  float *feat = nullptr, *last = nullptr, *g = nullptr, *emb = nullptr, *hd = nullptr, *aprev = nullptr, *logits = nullptr;   
  int *toks = nullptr, *d_pos = nullptr, *cur_tok = nullptr, *idx = nullptr;
  float* hist = nullptr;   
};
class Engine {
public:
  Engine(const std::string& path, int max_ctx, int flags);
  ~Engine();
  void reset();
  int  prefill(const int32_t* ids, int n);                 
  void decode(int32_t* out, int steps);                    
  void nll(const int32_t* ids, int n, float* out);         
  void logits(float* out);
  int  position() const { return pos_; }
  int  vocab() const { return m_.cfg.vocab; }
  int  max_ctx() const { return m_.max_ctx; }
  
  
  void debug_hidden(const int32_t* ids, int n, int upto, int mode, float* out);
  
  void debug_buffer(int which, int M, uint16_t* out);
  
  uint64_t debug_kv_hash();
  
  
  
  
  
  void verify(const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals,
              const int32_t* targets = nullptr, float* out_nll = nullptr);
  void commit(int n_accept);
  
  int  prefill_rows(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
  void hidden_rows(int n, float* out);                     
  void logits_rows(int row0, int n, float* out);           
  
  
  float time_verify(int M, int mask, int iters);
  unsigned xsum_nonfinite() const;
  void xsmg(GemvLaunch& P, bool main_model) const;
  void xsmm(GemmLaunch& G, bool on) const;
  void xsm_head_gemm(int M, bool topk);
  XformA8Args xsmx(XformA8Args A, bool on) const;
  
  void set_capture(const int* layers, int n);
  void load_draft(const std::string& path, const std::string& head_ids_path);   
  bool has_draft() const { return dm_.on; }
  float time_draft(int mode, int a, int steps, int iters);   
  void draft_debug(int which, int n, float* out);           
  
  
  void draft_round(int mode, int a, int steps, int use_graph);
  void set_verify_tokens(const int32_t* ids, int first_row, int n);              
  int32_t get_verify_token(int row);                                            
  
  
  
  void hist_store(int pos0, int n);
  void draft_catchup(int p0, int n, const int32_t* toks, int use_graph);
  void draft_steps(int steps, int use_graph);
  void verify_vals(int M, float* top_vals);                                     
  void verify_device(int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals = nullptr);   
  
  
  
  void verify_rows(int M, const int32_t* rows, int n_host, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals, bool hist);
  static constexpr bool kVerifyRows = true;
  void set_specx_scope(bool) {}                              
  void capture_rows(int k, int n, float* out);
private:
  void layer_forward(int l, int M, bool fused);            
  void head_forward(int M, int* adv_pos = nullptr, int* adv_tok = nullptr, int adv_greedy = 0);   
  void head_rows(int M);                                   
  void verify_body(int M);
  void verify_body_part(int M, int l0, int l1);
  void step_body(bool greedy);                             
  void ensure_graph();
  
  
  int pfa_ = 1, tpm_ = 512, pfa_wpc_ = 0;
  float *tp_h_ = nullptr, *tp_alpha_ = nullptr, *tp_fold_ = nullptr; uint16_t *tp_q16_ = nullptr, *tp_xff_ = nullptr; int8_t* tp_qx_ = nullptr;
  int *tp_toks_ = nullptr, *tp_pos_ = nullptr;
  void pfa_alloc();
  void pfa_gemm(std::initializer_list<std::pair<DevTensor*, float*>> segs, int M, bool normed = false);   
  bool pfa_xsm() const { return xsum_ && xsm_mode_ == 4; }   
  void layer_forward_pfa(int l, int M);
  int  prefill_pfa(const int32_t* ids, int n);
  int  prefill_rows_pfa(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals);   
  float* tp_cap_ = nullptr;                                  
  float* tp_head_ = nullptr;                                 
  int tpa_ = 512, tpc_ = 1;                                  
  float *tp_pm_ = nullptr, *tp_pl_ = nullptr, *tp_pacc_ = nullptr; uint16_t* tp_o16_ = nullptr;
  Model m_; Arena a_; cudaStream_t st_{}; cudaGraphExec_t gexec_ = nullptr; bool graph_ready_ = false; int pos_ = 0; int flags_ = 0;
  bool a4b_ = true;      
  bool res_fuse_ = true; 
  float* hcur_ = nullptr; float* hnext_ = nullptr;
  void input_norm(int M);                                  
  int8_t* xsm_qx2_ = nullptr; float* xsm_alpha2_ = nullptr; unsigned* xsm_cnt_ = nullptr; unsigned* xsm_dbg_ = nullptr;   
  bool fuse2_ = true; bool last_f2_ = false; bool emb_f2_ = false; unsigned* fuse_cnt_ = nullptr; unsigned* am_cnt_ = nullptr; unsigned* tk_cnt_ = nullptr;   
  int lno_ = 0;                                       
  size_t lno_build(const std::vector<DevTensor*>& ts); 
  int xsm_stats_ = 0;                                 
  bool xsum_ = false; int xsm_mode_ = 0; float* xsm_rr_ = nullptr; float* xsm_us_ = nullptr; unsigned* xsm_nf_ = nullptr;   
  float* part_scratch_ = nullptr;                           
  float2* rope_ = nullptr;                                 
  KvF32 kvf_{};                                            
  int f32_m_ = 8, tpf_ = 1;                                
                                                           
  int kmask_ = 15;                                         
  std::map<int, cudaGraphExec_t> vgraph_;                  
  std::map<int, cudaGraphExec_t> vrgraph_;                 
  int32_t* h_vin_ = nullptr; int* d_vin_ = nullptr;        
  
  int pf_ = 0, pf_ctas_ = 16, pf_n_ = 0; bool pf_active_ = false; PfEntry* pf_plan_ = nullptr; int* prog_ = nullptr;
  cudaStream_t pst_ = nullptr; cudaEvent_t pev0_ = nullptr, pev1_ = nullptr;
  void pf_mark(int l, int j);
  int last_M_ = 0;                                         
  int32_t* h_dec_ = nullptr;                               
  int32_t* h_pin_ = nullptr;                               
  float* h_pinf_ = nullptr;                                
  std::vector<int> cap_layers_; float* cap_ = nullptr;      
  int* pr_am_ = nullptr; int* pr_ti_ = nullptr; float* pr_tv_ = nullptr;   
  DraftModel dm_; std::map<int, cudaGraphExec_t> dgraph_;
  void draft_body(int mode, int a, int steps);
  void draft_chunk_body(int m, bool head);
  void draft_steps_body(int steps);
  std::map<int, cudaGraphExec_t> cgraph_, sgraph_;          
  int32_t* h_toks_ = nullptr;                              
  void draft_rows(int M, bool fused, const float* feat_src);   
  void draft_fused_rows(int M, bool fused);                
  void draft_stage(int M, int last_row);                   
  void draft_head(int row, int k);                         
  void draft_prefill_chunk(const int32_t* ids, int M);     
};
}
