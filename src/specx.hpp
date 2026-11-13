// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once










#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "engine.hpp"
#include "engine35.hpp"
namespace yskn {
struct SpecxConfig {
  std::atomic<int>* cancel = nullptr;
  int gamma = 7;            
  int use_d0 = 1, use_d1 = 1;
  int d0_nmin = 2, d0_nmax = 8;
  int d0_priority_n = 4;    
  int extend = 1;           
  int d1_rejected = 1;      
  int auto_gamma = 0;       
  int use_graph = 1;
  int use_d3 = 0;           
  int d3_steps = 0;         
  int d1_bigram = 0;        
  int d1_static = 0;        
  int d3_extend = 0;        
  int d3_lazy = 0;          
  int d3_gate = 0;          
  int gamma_long = 0;       
  int long_n = 6;
  int d1_trigram = 0;       
  int eos1 = -1, eos2 = -1; 
  int long_d0 = 0;          
  int sel = 0;              
  int cost_tok_us = 0;      
  int cost_row_us = 0;      
  int cost_mtp_us = 0;      
  int mtp_auto = 0;         
  int prmode = 0;          
                            
  int cost_row2_us = 0;     
  int lfrow = 0;           
  int wk_q_pm = 300;        
  int cost_tok_k = 0, cost_row2_k = 0, cost_mtp_k = 0;   
  int mark_tok = -1;        
  
  
  
  int bdraft = 0;           
  int bdr_long_n = 4;       
  int bdr_adapt = 1;        
  int bdr_probe = 4;        
  int bdr_d0_len = 15;      
  
  int prefill_head = 1;     
                            
  int snap = 0;           
  int snap_tok = -1;      
};
struct SpecxStats {
  int64_t passes = 0, tokens = 0, rows_hist[17] = {};
  int64_t drafted[3] = {}, accepted[3] = {};   
  int64_t chain_none = 0;                      
  double ms_verify = 0, ms_draft = 0, ms_total = 0;   
  double ms_commit = 0, ms_record = 0;   
  int64_t produced = 0; int eos = 0;     
  int64_t leaves = 0, leaf_breaks = 0, leaf_hits = 0;   
  double ms_mark = -1; int64_t mark_at = -1;   
  int64_t bdr_rounds = 0, bdr_d0 = 0, bdr_probes = 0;   
  double ms_ctx = 0;                                     
  int sh_mode = 0; int64_t sh_start = 0;                 
};

struct SpecxStatic { bool loaded = false; std::vector<int32_t> uni; std::unordered_map<uint64_t, int32_t> bi; };
const SpecxStatic& specx_static();
void specx_load_static(const std::string& path);

template <class Eng>
class SpecxT {
public:
  SpecxT(Eng& e, const SpecxConfig& c, int vocab);
  int  prefill(const int32_t* ids, int n);     
  void generate(int n_tokens, int32_t* out);   
  SpecxStats stats;
  const std::vector<int32_t>& seq() const { return S_; }   
private:
  Eng& e_; SpecxConfig c_; int V_;
  std::vector<int32_t> S_;                     
  std::vector<std::array<int32_t, 8>> T_;      
  std::unordered_map<uint64_t, int32_t> T2_;   
  std::unordered_map<uint64_t, float> T2m_;    
  std::unordered_map<uint64_t, int32_t> T3_;   
  int draft_done_ = 0;                         
  std::vector<std::unordered_map<uint64_t, int>> ng_;   
  int indexed_ = 0;                            
  std::vector<int> recent_;                    
  
  double reach_[3][16] = {}, acc_[3][16] = {}; int skipped_[3] = {};
  double p_acc(int s, int j) const;            
  int best_len(int s, int lmax, double fixed_us, double per_tok_us, int rows0 = 1) const;   
  double tok_us_ = 0, ema_us_ = 0, ema_tok_ = 0; int ema_n_ = 0;   
  double wq_n_[16] = {}, wq_h_[16] = {};          
  double dema_[2] = {3.0, 3.0};                 
  struct BdrPend { int F; int which; std::vector<int32_t> ch; };
  std::vector<BdrPend> dpend_;                    
  int dprobe_ = 0;
  double row_us(int rows) const { return rows > 8 ? (c_.cost_row2_us > 0 ? c_.cost_row2_us : c_.cost_row_us) + c_.cost_row2_k * dk_ : c_.cost_row_us; }   
  double dk_ = 0;            
  double tok_val() const { return (c_.prmode && tok_us_ > 0) ? tok_us_ : c_.cost_tok_us + c_.cost_tok_k * dk_; }   
  double mtp_us() const { return c_.cost_mtp_us + c_.cost_mtp_k * dk_; }
  uint64_t gram_hash(int end, int n) const;
  void index_upto(int end);                    
  int  d0_lookup(int F, std::vector<int32_t>& chain, int gmax);   
  int32_t next_tok(int32_t p2, int32_t prev, int32_t cur) const;  
  void d1_chain(std::vector<int32_t>& chain, int32_t p2, int32_t prev, int32_t cur, int gmax);
  void record(int32_t p2, int32_t prev, int32_t cur, const int32_t* top8, const float* val8 = nullptr);   
  uint64_t tkey(int32_t a, int32_t b, int32_t c) const { return ((uint64_t)(uint32_t)a * (uint64_t)V_ + (uint64_t)(uint32_t)b) * (uint64_t)V_ + (uint64_t)(uint32_t)c; }
  uint64_t bkey(int32_t a, int32_t b) const { return (uint64_t)(uint32_t)a * (uint64_t)V_ + (uint64_t)(uint32_t)b; }
};
using Specx = SpecxT<Engine>;
using Specx35 = SpecxT<Engine35>;
}
