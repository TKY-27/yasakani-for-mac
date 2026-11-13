// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once






#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "model.hpp"
#include "q35.hpp"
#include "args.hpp"
#include "q35_g6.hpp"
namespace yskn {
struct Shared35Weights;
struct Q35Layer {
  bool full = false;
  DevTensor qkv, z, out, q, k, v, o, gate, up, down;
  uint16_t *ln1 = nullptr, *ln2 = nullptr;
  uint16_t *wa = nullptr, *wb = nullptr, *conv = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;   
  uint16_t *qn = nullptr, *kn = nullptr;                                                                         
  float* S = nullptr; uint16_t* hist = nullptr; uint16_t* pmix = nullptr; float* pab = nullptr;   
  float *tq = nullptr, *tk = nullptr, *tv = nullptr, *tbeta = nullptr, *tg = nullptr;           
  uint16_t *kc = nullptr, *vc = nullptr;                                                           
  int8_t *k8 = nullptr, *v8 = nullptr; float *ks = nullptr, *vs = nullptr;                        
  uint8_t *kp = nullptr, *vp = nullptr; float *kps = nullptr, *vps = nullptr;                      
};





struct Q35Mtp {
  bool on = false; Container c; DevTensor fc, head; Q35Layer L; int nsub = 0;
  uint16_t *en = nullptr, *hn = nullptr, *norm = nullptr; int* ids = nullptr;   
  uint16_t *hin = nullptr, *hout = nullptr, *pend = nullptr;                    
  int *toks = nullptr, *d_pos = nullptr, *idx = nullptr, *out = nullptr, *vtoks = nullptr, *iota = nullptr;
  uint16_t* hist = nullptr; int* ctoks = nullptr; int32_t* h_ctoks = nullptr; int last_rows = 1;   
  std::map<int, cudaGraphExec_t> graphs;
  float *tkv = nullptr, *top_v = nullptr, *lg_val = nullptr; int *tki = nullptr, *top_i = nullptr, *lg_idx = nullptr, *lg_cnt = nullptr; unsigned* tkc = nullptr;   
};
class Engine35 {
public:
  Engine35(const std::string& path, int max_ctx, int flags);
  ~Engine35();
  void shared_info(uint64_t* out) const;
  void reset();
  int  prefill(const int32_t* ids, int n);                 
  
  
  
  
  
  
  
  
  void set_visual(const uint16_t* features, const int32_t* slots, int n_image,
                  const int32_t* positions, int n_prompt, int delta);
  void decode(int32_t* out, int steps);                    
  void nll(const int32_t* ids, int n, float* out);         
  void logits(float* out);                                 
  int  position() const { return pos_; }
  int  vocab() const { return V_; }
  int  max_ctx() const { return max_ctx_; }
  void verify(const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals,
              const int32_t* targets = nullptr, float* out_nll = nullptr);
  void commit(int n_accept);
  void hidden_rows(int n, float* out);
  void logits_rows(int row0, int n, float* out);
  
  
  float time_verify(int M, int mask, int iters);
  int  prefill_rows(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals);   
  
  void load_draft(const std::string& path, const std::string& head_ids_path);   
  bool has_draft() const { return mtp_.on; }
  void set_verify_tokens(const int32_t* ids, int first_row, int n);
  int32_t get_verify_token(int row);
  void verify_device(int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals = nullptr);
  static constexpr bool kVerifyRows = false;                 
  
  
  
  
  
  void set_opt(int key, int value);
  void set_specx_scope(bool on) { specx_scope_ = on; }
  
  void mtp_log_reset();
  
  
  
  void lfrow_table(const uint8_t* tab48);
  void lfrow_fill(int k, int M);
  void lfrow_promote(int src, int dst);
  int  mtp_log_read(int32_t* ids, float* vals, int max_steps);   
  
  
  void draft_round(int mode, int a, int steps, int use_graph);
  
  
  void hist_store(int pos0, int n);
  void draft_catchup(int p0, int n, const int32_t* toks, int use_graph);
  void draft_steps(int steps, int use_graph);
  
  
  void debug_hidden(const int32_t* ids, int n, int upto, int mode, float* out);
  
  
  
  
  int state_hash(uint64_t* out, int max_out);
  
  
  
  
  
  
  
  void snap_save(const int32_t* ids, int n);        
  void snap_load();                                  
  int  snap_pos() const { return sh_pos_; }
  const std::vector<int32_t>& snap_ids() const { return sh_ids_; }
  void snap_note(const int32_t* ids, int n) { sh_cur_.assign(ids, ids + n); sh_note_ = sh_epoch_; }   
  bool snap_live_ok() const { return sh_note_ == sh_epoch_ && (int)sh_cur_.size() == pos_ && pos_ > 0; }
  const std::vector<int32_t>& snap_live() const { return sh_cur_; }
  void snap_drop() { sh_pos_ = -1; sh_ids_.clear(); }
  
  
  
  
  
  
  int chk_set(const std::string& name, const long long* c, int Kc, unsigned long long key, int w_max);
  void chk_stats(unsigned long long* out, int reset);
  void chk_fault(int kind, const std::string& name, int a, int b);
  void chk_enable(int on);
  
  
  void set_capture(const int* layers, int n);
  void capture_rows(int k, int n, float* out);
  
  void set_kz_debug(int layer);
  void kz_debug_read(int n, float* o, uint16_t* q);
  int kv_format() const { return kv8_ ? 8 : (kv_fp16_ ? 16 : 116); }
  void kv16_read(int layer, int pos0, int n, uint16_t* k, uint16_t* v);
  void kv8_read(int layer, int pos0, int n, int8_t* k8, float* ks, int8_t* v8, float* vs);
  
  
  
  
  
  
  void load_bdraft(const std::string& path);
  bool has_bdraft() const { return bdr_.on; }
  void bdraft_ctx(int row0, int n, int pos0);
  void bdraft_draft(int use_graph);
  
  
  void bdraft_step(int n_ctx, int pos_ctx, int draft, int use_graph);
  
  int prefill_rows_k20(const int32_t* ids, int n, int32_t* out_argmax, int32_t* k20_ids, float* k20_vals);
  void bdraft_debug(const float* feats, int n, int pos0, int32_t anchor, float* hidden, int32_t* top_ids, float* top_vals, int32_t* path);
  int  bdraft_out(int32_t* drafts);            
  void bdraft_ring(int layer, int slot0, int n, uint16_t* k_out, uint16_t* v_out);   
private:
  std::shared_ptr<Shared35Weights> shared_weights_;
  struct Bdr {
    bool on = false; Container c; DevTensor fc; DevTensor L[5][9];        
    uint16_t *hnorm = nullptr, *fnorm = nullptr, *ln1[5] = {}, *ln2[5] = {}, *qn[5] = {}, *kn[5] = {}, *abase[5] = {}, *mbase[5] = {};
    uint16_t *hp = nullptr, *pc = nullptr, *sc = nullptr, *rk[5] = {}, *rv[5] = {};
    float *x = nullptr, *x1 = nullptr, *x2 = nullptr, *r = nullptr, *alpha = nullptr, *us = nullptr, *h = nullptr, *dyn = nullptr, *att = nullptr;
    float *yq = nullptr, *yk = nullptr, *yv = nullptr, *yo = nullptr, *yg = nullptr, *yu = nullptr, *ys = nullptr, *yd = nullptr, *yfc = nullptr;
    float *hf = nullptr, *lg = nullptr, *tval = nullptr, *hpv = nullptr; int8_t* qx = nullptr; long long* hacc = nullptr;
    int *tidx = nullptr, *ids = nullptr, *pos = nullptr, *out = nullptr; const int8_t* signs = nullptr;
    float *apart = nullptr, *tcv = nullptr; int *tci = nullptr, *cpos = nullptr;   
    int* cpl = nullptr; int cpl_pm = 850;                                 
    cudaGraphExec_t sgraph[3][17] = {};   
    
    
    Container csub; DevTensor hsub; int* sub_ids = nullptr; int nsub = 0; int sub_on = 0;
    
    int* alt = nullptr; int* alt3 = nullptr; float* marg = nullptr; float* hb_tab = nullptr; int* hb_cfg = nullptr; int* hb_go = nullptr; int hb_on = 0; int hb_pass = 0;
    
    
    
    float *sk_bt = nullptr, *sk_as = nullptr, *sk_ws = nullptr, *sk_q = nullptr, *sk_s = nullptr, *sk_cv = nullptr; int8_t *sk_a = nullptr, *sk_w = nullptr;
    unsigned* sk_hist = nullptr; int *sk_thr = nullptr, *sk_ci = nullptr; int sk_m = 0, sk_grid = 0;
  } bdr_;
  void load_shead(const char* path);                                      
  void free_shead();                                                      
  void bdr_gemv(std::initializer_list<DevTensor*> ts, int M);          
  void bdr_ctx_rows(const float* x_dev, int row0, int m);                
  void bdr_block(int rows);                                                       
  void bdr_block_impl(int* toks, bool hcap, int rows);   
  void chk_after(std::initializer_list<DevTensor*> ts, int M, const int8_t* qx, const float* alpha);
  float* cap_ = nullptr; std::vector<int> cap_layers_;
  float* d_k20v_ = nullptr; int* d_k20i_ = nullptr; bool k20_on_ = false;   
  struct ChkEntry { long long* c = nullptr; unsigned long long key = 0; int w_max = 127; int idx = 0; };
  std::map<const DevTensor*, ChkEntry> chk_; Q35ChkStats* d_mstats_ = nullptr; int chk_on_ = 1;
  int mm_kind_ = 0, mm_row_ = 0, mm_bit_ = 0; const DevTensor* mm_t_ = nullptr;
  DevTensor* tensor_by_name(const std::string& name);
  void drop_pass_graphs();
public:
  bool hcap_on() const { return bdr_.on && bdr_.hb_on; }            
  void hcap_pass(int on) { bdr_.hb_pass = on; }                     
  void hcap_cfg(int* out8) const;                                     
  void lfrow_device_table() { wk_set_ = true; }                       
private:
  void apply_kz();   
  [[noreturn]] static void no_d3() { throw std::runtime_error("qwen3_5 engine: lazy D3 mode is not supported (use the eager MTP draft)"); }
  enum { HEAD_NONE = 0, HEAD_LAST = 1, HEAD_ALL = 2 };
  DevTensor up(const std::string& name, int force_workers = 0, const Container* src = nullptr);
  uint16_t* vec(const std::string& name, size_t n, const Container* src = nullptr, bool host = false);
  void gemv(std::initializer_list<DevTensor*> ts, int M);
  void res_norm(const DevTensor* src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out);
  void emb_src(const int* ids, int M, const uint16_t*& emb, const int*& gids);   
  void attn_block(Q35Layer& Y, int M, const int* dpos, int* slices);
  void mlp_block(Q35Layer& Y, int M, bool down = true);   
  void layer(int l, int M, int n_commit);
  void mtp_pass(int M, bool head);                          
  void run_mtp(int M, bool head, bool use_graph);
  void mtp_catchup(int M);                                  
  void ensure_part(size_t floats);
  void head_slice(int row0, int rows, bool advance, bool topk, bool nll);
  void pass_body(int M, bool commit, int upto, int head_mode, bool topk, bool nll, bool advance);
  void replay_commit(int rows);
  void run_pass(int M, bool commit, int head_mode, bool topk, bool nll, bool use_graph);
  void upload_rows(const int32_t* ids, int M, const int32_t* targets);
  Model m_; std::vector<Q35Layer> L_; DevTensor head_; uint16_t *fnorm_ = nullptr, *emb_ = nullptr; Q35Mtp mtp_; size_t part_floats_ = 0; int vt_slot_ = 0;
  int V_ = 0, H_ = 0, F_ = 0, nlayers_ = 0, max_ctx_ = 0, splits_ = 32, want_ = 0, pos_ = 0, last_M_ = 0, flags_ = 0;
  int Mmax_ = 64;        
  int Pmax_ = 64;        
  uint16_t* pmix_big_ = nullptr; float* pab_big_ = nullptr;   
  int *pos_slices_ = nullptr, *mtp_pos_slices_ = nullptr;     
  int kskip_ = 0;        
                         
                         
  bool ks(int bit) const { return (kskip_ & bit) != 0; }
  int pfa_ = 1, pfa_wpc_ = 0;   
  int nwk(const DevTensor& t, int M) const { return (pfa_ && M > Mmax_) ? 1 : t.dev.n_workers; }   
  int xsm_ = 1;          
  Q35G6Workspace g6_; 
  bool xsmc_ = false;    
  float *us_ = nullptr, *S_ = nullptr, *mtp_part_ = nullptr;   
  float* ybuf_ = nullptr;  
  float* pt(const DevTensor& t, int M) const { return (xsmc_ && M > 16) ? ybuf_ + (t.part - part_) : t.part; }
  int xsm_part_ = 0;     
  int nmode(int M, bool normed = false) const { return M > 16 ? (normed ? 3 : 1) : (xsm_part_ ? 2 : 0); }   
  void gemv(std::initializer_list<DevTensor*> ts, int M, bool normed, const GemvLaunch* fin = nullptr);   
  
  
  void gemv_res(DevTensor& src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out);
  
  
  
  
  
  
  
  int kv8_ = 1;
  int tbcap_ = 0;        
  int tbcap_kb_ = 4;     
  int stg_rows_ = 0; int8_t *stg_k8_ = nullptr, *stg_v8_ = nullptr; float *stg_ks_ = nullptr, *stg_vs_ = nullptr; unsigned char* tb_rec_ = nullptr;   
  int kv_fp16_ = 0; 
  int emb_host_ = 0;     
  
  
  
  bool emb_coded_ = false; int emb_K_ = 0; uint32_t* emb_codes_ = nullptr; uint16_t *emb_lut_ = nullptr, *emb_sc_ = nullptr, *emb_rows_ = nullptr;
  int8_t* emb_sg_ = nullptr; int* emb_iota_ = nullptr;
  uint32_t* emb2_codes_ = nullptr; uint16_t *emb2_lut_ = nullptr, *emb2_sc_ = nullptr; int emb2_K_ = 0;   
  uint32_t* emb3_codes_ = nullptr; uint16_t *emb3_lut_ = nullptr, *emb3_sc_ = nullptr; int emb3_K_ = 0;   
  int fa_ = 1, fa_splits_ = 31;
  int atx_ = 1, atx_splits_ = 42;   
  
  
  int atx_short_ = 31, atx_long_ = 4096;
  int atx_vsp_ = 3;   
                      
  
  
  int kz_ = 0, kz_dbg_layer_ = -1, kz_dbg_rows_ = 0; float* kz_dbg_ = nullptr; uint16_t* kz_dbgq_ = nullptr;
  
  unsigned long long* flow_trace_ = nullptr;
  int opt_flow_ = 0; int8_t* qxd_ = nullptr; float* ald_ = nullptr; float* usd_ = nullptr; unsigned* flow_cnt_ = nullptr;
  bool flow_ok(int M) const { return opt_flow_ && xsmc_ && !xsm_part_ && M <= 8 && !kskip_ && !opt_fin_ && !in2_; }
  void flow_mlp(Q35Layer& Y, int M, int l, bool prepare_only = false);
  void flow_trace_read(float* out, int n);
  void flow_prepare() { if (opt_flow_ && xsm_ && !xsm_part_) for (int M = 1; M <= 8; ++M) flow_mlp(L_[0], M, 0, true); }
  int opt_fin_ = 0; bool in2_ = false; int8_t* qx2_ = nullptr; float* alpha2_ = nullptr; unsigned* fin_cnt_ = nullptr;
  
  
  
  int l2pf_ = 0; double l2pf_mb_[3] = {4.0, 6.0, 6.0}; Q35Pf next_pf_;
  
  
  double l2pf2_mb_ = 0.0; cudaStream_t st3_{}; cudaEvent_t ev_g_{}, ev_pf_{}; bool pf2_live_ = false;
  
  
  int opt_lo_ = 1; bool lo_now_ = false;                                   
  Q35Pf pf_of(std::initializer_list<const DevTensor*> ts, double mb) const;
  bool pf_on(int bit, int M) const { return (l2pf_ & bit) && xsmc_ && M <= 16; }
  
  
  int smp_ = 0; Q35SmpParams smp_p_{0.6, 0.95, 20, 0, 0ull}; Q35SmpParams* d_smp_ = nullptr; float* kj_val_ = nullptr; int* kj_idx_ = nullptr; unsigned* kj_cnt_ = nullptr;
  int HM_ = 16;          
  int PM_ = 64;          
  float eps_ = 1e-6f; size_t lno_bytes_ = 0;
  cudaStream_t st_{};
  
  
  
  cudaStream_t st2_{}; cudaEvent_t ev_v_{}, ev_rp_{}; bool rp_pending_ = false; int opt_rp_async_ = 1;   
  
  
  int opt_rp_defer_ = 0; cudaGraphExec_t rp_graph_ = nullptr; bool rp_deferred_ = false;   
  void rp_launch() { if (rp_deferred_) { cudaStreamWaitEvent(st2_, ev_v_, 0); cudaGraphLaunch(rp_graph_, st2_); cudaEventRecord(ev_rp_, st2_); rp_deferred_ = false; } }
  void replay_fence() { if (rp_pending_) { rp_launch(); cudaStreamWaitEvent(st_, ev_rp_, 0); rp_pending_ = false; } }
  float *h_ = nullptr, *alpha_ = nullptr, *z_ = nullptr, *gq_ = nullptr, *gk_ = nullptr, *gv_ = nullptr, *gbeta_ = nullptr, *gg_ = nullptr, *core_ = nullptr;
  uint16_t *xn_ = nullptr, *q16_ = nullptr, *gate16_ = nullptr; int8_t* qx_ = nullptr;
  float *pm_ = nullptr, *pl_ = nullptr, *pacc_ = nullptr; float2* rope_ = nullptr; float* part_ = nullptr;
  float *tc_h_ = nullptr, *tc_al_ = nullptr, *tc_S_ = nullptr; uint16_t* tc_xn_ = nullptr; int8_t* tc_qx_ = nullptr; int* tc_ids_ = nullptr; int tc_rows_ = 0;
  int8_t* tq_qx_ = nullptr; float* tq_al_ = nullptr; int8_t* scr_qx_ = nullptr; float* scr_al_ = nullptr;   
  
  
  int uat2_ = 0;    
  int ovl_ = 0; cudaStream_t nb_st_ = nullptr; cudaEvent_t nb_e1_[2] = {}, nb_e2_[2] = {}; int nb_rows_ = 0;
  uint16_t* nb_pmx_[2] = {}; float *nb_z_[2] = {}, *nb_cq_[2] = {}, *nb_ck_[2] = {}, *nb_cv_[2] = {}, *nb_cb_[2] = {}, *nb_cg_[2] = {}, *nb_core_[2] = {};
  void gdn_a(int l, int M, int b, const uint16_t* prev); void gdn_s(int l, int M, int b); void gdn_b(int l, int M, int b);
  float* tc_cap_ = nullptr; const float* dctx_cap_ = nullptr; int dctx_rows_ = 0;   
  int8_t* tc_dq_=nullptr; float* tc_da_=nullptr; const int8_t* dctx_q_=nullptr; const float* dctx_a_=nullptr;
  int layer_rows_=0; bool layer_ready_=false, layer_unavailable_=false;
  uint4* layer_weights_=nullptr; size_t layer_weight_bytes_=0;
  bool layer_prepare(); void layer_decode(int l);
  int prefill_vsp(const int32_t* ids,int n);
  int prefill_chunks(const int32_t* ids,int n);
public:
  void nll_layer(const int32_t* ids,int n,float* out);
  void layer_info(uint64_t* out) const { out[0]=layer_ready_;out[1]=layer_unavailable_;out[2]=tc_rows_;out[3]=ovl_;out[4]=layer_weight_bytes_;out[5]=layer_rows_; }
private:
  bool visual_on_ = false;
  int visual_cap_ = 0, visual_prompt_ = 0, visual_delta_ = 0;
  uint16_t* visual_features_ = nullptr;
  int *visual_map_ = nullptr, *visual_positions_ = nullptr;
  float *logits_ = nullptr, *blk_val_ = nullptr, *blk_max_ = nullptr, *blk_sum_ = nullptr, *d_nll8_ = nullptr, *tk_val_ = nullptr, *top_val_ = nullptr;
  int *blk_idx_ = nullptr, *d_pos_ = nullptr, *d_toks_ = nullptr, *d_next8_ = nullptr, *d_target8_ = nullptr, *tk_idx_ = nullptr, *top_idx_ = nullptr;
  unsigned *cnt_ = nullptr, *am_cnt_ = nullptr, *tk_cnt_ = nullptr;
  
  int32_t* h_pin_ = nullptr; float* h_pinf_ = nullptr;
  int32_t* h_dec_ = nullptr;   
  cudaGraphExec_t gexec_ = nullptr; std::map<int, cudaGraphExec_t> pgraph_;
  cudaGraphExec_t gexec_alt_ = nullptr;   
  std::map<int, cudaGraphExec_t> rgraph_;                     
  int opt_replay_graph_ = 1, opt_single_commit_ = 1, opt_catchup_nosync_ = 1, opt_mtp_log_ = 0;
  int opt_lfrow_ = 0; bool wk_now_ = false, wk_set_ = false; uint8_t* d_wk_ = nullptr; uint8_t* h_wk_ = nullptr; int wk_slot_ = 0;
  WkCopy* d_wkcopy_ = nullptr; int n_wkcopy_ = 0;
  bool wk_pass(int M, bool commit, int head_mode) const { return opt_lfrow_ && !commit && M <= HM_ && head_mode == HEAD_ALL; }
  void wk_plain_table();
  int opt_kconv_ = 1;      
  bool specx_scope_ = false;  
  bool pre_committed_ = false;   
  
  float* sh_S_ = nullptr; uint16_t* sh_hist_ = nullptr; uint16_t* sh_ring_ = nullptr; uint16_t* sh_pend_ = nullptr;
  int sh_pos_ = -1; std::vector<int32_t> sh_ids_, sh_cur_; unsigned long long sh_epoch_ = 1, sh_note_ = 0;
  
  
  
  int vis_n_ = 0, vis_delta_ = 0; std::vector<int32_t> vis_pos_; std::vector<int> vis_map_; std::vector<unsigned long long> vis_hash_;
  bool visual_prefix_agrees(const std::vector<int>& map, const int32_t* positions, int n_prompt, int delta,
                            const std::vector<unsigned long long>& hash) const;
};
} 
