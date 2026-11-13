// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "engine35.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "q35_flow.hpp"
#include "signs.hpp"
#include "devmem.hpp"   

#if YSKN_KVX_MODE == 3
#include <vector>
#include <string>
#include "kvq_fast.hpp"
#include "kvq_atxpk.hpp"
#include "q35_atxpk.hpp"



static std::vector<unsigned char> g_tb_host; static unsigned char* g_tb_rec = nullptr; static int g_tb_kb = 4;
static bool tb_cap_packable(int ord, int kind) { return yskn::kvqaxpk_packable(g_tb_host.data(), g_tb_host.size(), ord, kind, g_tb_kb); }
static const unsigned char* tb_dev_ptr() {
  static unsigned char* dev = nullptr; static bool tried = false;
  if (tried) return dev;
  tried = true;
  const char* e = std::getenv("YSKN_TB_PROFILE");
  if (!e || !e[0]) throw std::runtime_error("YSKN_KVX_MODE=3 needs YSKN_TB_PROFILE=<KVQP profile blob>");
  FILE* f = std::fopen(e, "rb");
  if (!f) throw std::runtime_error(std::string("YSKN_TB_PROFILE: cannot open ") + e);
  std::vector<unsigned char> b; unsigned char tmp[65536]; size_t n;
  while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) b.insert(b.end(), tmp, tmp + n);
  std::fclose(f);
  auto u16 = [&](size_t o) { return (unsigned)b[o] | ((unsigned)b[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (b.size() < 100 || std::memcmp(b.data(), "KVQP", 4) != 0 || u16(4) != 1 || u16(6) != 96 || u32(8) != b.size()) throw std::runtime_error("YSKN_TB_PROFILE: not a KVQP v1 blob");
  if (u16(16) != 256 || u16(18) != 16 || u16(20) != 4 || u32(28) != 128u) throw std::runtime_error("YSKN_TB_PROFILE: needs D 256, 16 layers, 4 kv heads");
  const unsigned nt = u16(22), off_t = u32(36);
  for (unsigned t = 0; t < nt; ++t) { const size_t o = u32(off_t + 4 * t); if (o + 12 > b.size() || b[o] != 8 || b[o + 2] != 64 || b[o + 3] != 8) throw std::runtime_error("YSKN_TB_PROFILE: GPU codec needs CDQ tables with L 8, chain 64, F 8"); }
  if ((cudaMalloc)((void**)&dev, b.size()) != cudaSuccess || cudaMemcpy(dev, b.data(), b.size(), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("YSKN_TB_PROFILE: device upload failed");
  g_tb_host = b; g_tb_kb = yskn::kvqaxpk_kb(b.data(), b.size());
  { std::vector<unsigned char> rec;
    if (yskn::kvqaxpk_build(b.data(), b.size(), rec) && ((cudaMalloc)((void**)&g_tb_rec, rec.size()) != cudaSuccess || cudaMemcpy(g_tb_rec, rec.data(), rec.size(), cudaMemcpyHostToDevice) != cudaSuccess)) throw std::runtime_error("YSKN_TB_PROFILE: decode record upload failed"); }
  {                                                                          
    yskn::TbHost& H = yskn::tbf_host(); int nm = 0, has = 0;
    if (!yskn::tbf_parse_blob(b.data(), b.size(), nm, has) || nm < 1 || nm > yskn::TBF_MAXNM) throw std::runtime_error("YSKN_TB_PROFILE: bad scale-multiplier count");
    const char* f = std::getenv("YSKN_TB_FAST"); const bool fast = !f || !f[0] || std::atoi(f) != 0;
    H.nm = nm; H.has_cdq = has; H.scr = nullptr;
    if (fast && (cudaMalloc)((void**)&H.scr, yskn::tbf_scratch_bytes()) != cudaSuccess) throw std::runtime_error("YSKN_TB_PROFILE: scratch allocation failed");
  }
  return dev;
}
#endif
namespace yskn {


struct Shared35Weights {
  std::mutex mutex;
  std::map<std::string, DevTensor> tensors;
  std::map<std::string, void*> blobs;
  std::string draft_path;
  ~Shared35Weights() { devmem_release(this); }
};
static std::shared_ptr<Shared35Weights> get_shared35(const std::string& path, int want, bool xsm) {
  const char* enabled = std::getenv("YSKN_Q35_SHARED_WEIGHTS");
  if (!enabled || enabled[0] != '1') return {};
  static std::mutex mutex;
  static std::map<std::string, std::weak_ptr<Shared35Weights>> registry;
  const auto file = std::filesystem::canonical(path);
  const std::string key = file.string() + ":" + std::to_string(std::filesystem::file_size(file)) +
    ":" + std::to_string(std::filesystem::last_write_time(file).time_since_epoch().count()) +
    ":" + std::to_string(want) + ":" + std::to_string(xsm);
  std::lock_guard<std::mutex> guard(mutex);
  auto p = registry[key].lock();
  if (!p) { p = std::make_shared<Shared35Weights>(); registry[key] = p; }
  return p;
}
void Engine35::shared_info(uint64_t* out) const {
  out[0] = shared_weights_ ? shared_weights_->tensors.size() : 0;
  out[1] = shared_weights_ ? shared_weights_.use_count() : 0;
  out[2] = reinterpret_cast<uint64_t>(head_.lno);
  out[3] = reinterpret_cast<uint64_t>(part_);
  out[4] = reinterpret_cast<uint64_t>(L_[0].S);
  out[5] = 0;
  for (const auto& l : L_) if (l.full) { out[5] = reinterpret_cast<uint64_t>(l.kc); break; }
  out[6] = bdr_.sk_m;
  out[7] = reinterpret_cast<uint64_t>(bdr_.sk_w);
}
template<class Upload> static void* shared35_buffer(const std::shared_ptr<Shared35Weights>& pool,
    const std::string& key, size_t bytes, Upload upload) {
  std::unique_lock<std::mutex> lock;
  if (pool) {
    lock = std::unique_lock<std::mutex>(pool->mutex);
    auto it = pool->blobs.find(key);
    if (it != pool->blobs.end()) return it->second;
  }
  DevMemScope owner(pool ? static_cast<void*>(pool.get()) : devmem_owner());
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) throw std::runtime_error("shared35: out of memory");
  try { upload(p); } catch (...) { cudaFree(p); throw; }
  if (pool) pool->blobs.emplace(key, p);
  return p;
}

extern thread_local int g_pdl_rows;                                   



template <class T2> static T2* dz35(size_t bytes) { T2* p = nullptr; if (cudaMalloc(&p, bytes) != cudaSuccess) throw std::runtime_error("Engine35: cudaMalloc failed");
                                                    cudaMemset(p, 0, bytes); cudaDeviceSynchronize(); return p; }

static constexpr int PIN_HB = 22900;
static constexpr int PIN_IDS = 0, PIN_AM = 2048, PIN_TI = 4096, PIN_TG = 20480, PIN_VT = 22528, PIN_GT = 22800, PIN_N = 23040, PINF_TV = 0, PINF_NLL = 16384, PINF_N = 18432;



static int q35_nw_override(const std::string& name) {
  static std::map<std::string, int> tab = [] {                       
    std::map<std::string, int> t{{"down", 9}, {"out", 4}, {"o", 4}, {"gate", 5}, {"up", 5}, {"qkv", 8}, {"z", 8}, {"q", 4}, {"k", 4}, {"v", 4}};
    const char* e = std::getenv("YSKN_Q35_NW"); if (!e) return t;
    std::string s(e); size_t i = 0; while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
      const std::string kv = s.substr(i, j - i); const size_t q = kv.find('='); if (q != std::string::npos) t[kv.substr(0, q)] = std::atoi(kv.c_str() + q + 1); i = j + 1; }
    return t; }();
  static const std::pair<const char*, const char*> kinds[] = {{"mlp.down_proj", "down"}, {"mlp.gate_proj", "gate"}, {"mlp.up_proj", "up"},
    {"linear_attn.out_proj", "out"}, {"linear_attn.in_proj_qkv", "qkv"}, {"linear_attn.in_proj_z", "z"}, {"self_attn.q_proj", "q"},
    {"self_attn.k_proj", "k"}, {"self_attn.v_proj", "v"}, {"self_attn.o_proj", "o"}};
  if (name.rfind("layers.", 0) != 0) return 0;
  for (const auto& kv : kinds) if (name.size() >= std::strlen(kv.first) && name.compare(name.size() - std::strlen(kv.first), std::string::npos, kv.first) == 0) {
    const auto it = tab.find(kv.second); return it == tab.end() ? 0 : it->second; }
  return 0;
}
DevTensor Engine35::up(const std::string& name, int force_workers, const Container* src) {
  const Container& C = src ? *src : m_.c;
  int nw_big = 0;
  if (xsm_ && !force_workers && !src) {                                
    force_workers = q35_nw_override(name);
    const TensorDesc& d = C.tensor(name); const int G = d.cols / 512, nt = d.physical_rows / 64;
    nw_big = std::max(1, std::min(G, (want_ + nt - 1) / nt)); }
  const bool pooled = shared_weights_ && (!src || src == &bdr_.c);
  std::unique_lock<std::mutex> lock;
  const std::string cache_key = (src ? "draft/" : "target/") + name + ":" + std::to_string(force_workers);
  if (pooled) {
    lock = std::unique_lock<std::mutex>(shared_weights_->mutex);
    auto it = shared_weights_->tensors.find(cache_key);
    if (it != shared_weights_->tensors.end()) {
      DevTensor t = it->second;
      t.part = nullptr; 
      lno_bytes_ += lnord_bytes(t.dev.N, t.dev.Kc, t.bits);
      return t;
    }
  }
  DevMemScope payload_owner(pooled ? static_cast<void*>(shared_weights_.get()) : devmem_owner());
  DevTensor t = m_.upload_linear(C.tensor(name), 1, want_, force_workers, src);
  if (t.tile_k || t.dev.tile_u4off) throw std::runtime_error("qwen3_5 engine: RAR tensors are not supported (" + name + ")");
  if (t.part) { cudaFree(t.part); t.part = nullptr; }
  const size_t sz = lnord_bytes(t.dev.N, t.dev.Kc, t.bits);                 
  if (cudaMalloc(&t.lno, sz) != cudaSuccess) throw std::runtime_error("Engine35: out of memory (LNORD " + name + ")");
  lnord_build(t.frag, t.dev.N, t.dev.Kc, t.bits, t.lno, st_); cudaStreamSynchronize(st_);
  cudaFree(t.frag); t.frag = nullptr; t.dev.frag = nullptr; t.dev.lno = t.lno; lno_bytes_ += sz;
  t.nw_big = nw_big;
  if (pooled) shared_weights_->tensors.emplace(cache_key, t);
  return t;
}
uint16_t* Engine35::vec(const std::string& name, size_t n, const Container* src, bool host) {
  const Container& C = src ? *src : m_.c; const VectorDesc& v = C.vector(name);
  if ((size_t)v.n != n || v.dtype != "bf16") throw std::runtime_error("qwen3_5 vector " + name + ": expected bf16 [" + std::to_string(n) + "]");
  if (host) {                                                          
    uint16_t* h = nullptr; if (cudaMallocHost(&h, n * 2) != cudaSuccess) throw std::runtime_error("Engine35: out of pinned memory (" + name + ")");
    std::memcpy(h, C.at<uint16_t>(v.off), n * 2);
    void* d = nullptr; if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) throw std::runtime_error("Engine35: no device pointer for " + name);
    return static_cast<uint16_t*>(d); }
  uint16_t* d = nullptr; if (cudaMalloc(&d, n * 2) != cudaSuccess) throw std::runtime_error("Engine35: out of memory (" + name + ")");
  cudaMemcpy(d, C.at<uint16_t>(v.off), n * 2, cudaMemcpyHostToDevice); return d;
}
void Engine35::ensure_part(size_t floats) {                          
  if (floats <= part_floats_) return;
  throw std::runtime_error("Engine35: partial-sum scratch too small for " + std::to_string(floats) + " floats");
}

Engine35::Engine35(const std::string& path, int max_ctx, int flags) : flags_(flags) {
  DevMemScope dm_scope(this);
  cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&st2_, cudaStreamNonBlocking); cudaEventCreateWithFlags(&ev_v_, cudaEventDisableTiming); cudaEventCreateWithFlags(&ev_rp_, cudaEventDisableTiming);
  { const char* e = std::getenv("YSKN_Q35_RP_ASYNC"); if (e && e[0]) opt_rp_async_ = std::atoi(e); }
  { const char* e = std::getenv("YSKN_RP_DEFER"); if (e && e[0]) opt_rp_defer_ = std::atoi(e); }
  cudaStreamCreateWithFlags(&st3_, cudaStreamNonBlocking); cudaEventCreateWithFlags(&ev_g_, cudaEventDisableTiming); cudaEventCreateWithFlags(&ev_pf_, cudaEventDisableTiming);
  { const char* e = std::getenv("YSKN_L2PF2"); if (e && e[0]) l2pf2_mb_ = std::atof(e); }
  { const char* e = std::getenv("YSKN_Q35_LO"); if (e && e[0]) opt_lo_ = std::atoi(e); }
  { const char* e = std::getenv("YSKN_PFA"); if (e && e[0]) pfa_ = std::atoi(e); const char* w = std::getenv("YSKN_PFA_WPC"); if (w && w[0]) pfa_wpc_ = std::atoi(w); }
  Pmax_ = pfa_ ? 512 : Mmax_;
  { const char* e = std::getenv("YSKN_Q35_PMAX"); if (e && pfa_) { const int n=std::atoi(e);
      if (n >= 512 && n <= 2048 && n % 64 == 0) Pmax_=n; } }
  PM_ = Pmax_;
  {const char* e=std::getenv("YSKN_Q35_LAYER_ROWS");if(e)layer_rows_=std::max(0,std::min(16384,std::atoi(e)));
   const char* o=std::getenv("YSKN_Q35_LAYER_OVERLAP");if(o)ovl_=std::atoi(o)!=0;}                              
  { const char* e = std::getenv("YSKN_Q35_KV8"); if (e && e[0]) kv8_ = std::atoi(e); }
  { const char* e = std::getenv("YSKN_Q35_KV_FP16"); if (e && e[0]) kv_fp16_ = std::atoi(e) != 0; }
#if YSKN_KVX_MODE == 2 || YSKN_KVX_MODE == 3
  if (!kv8_ || kv_fp16_) throw std::runtime_error("Rotated INT8 runtime requires INT8 storage");
#endif
#if YSKN_KVX_MODE == 3
  (void)tb_dev_ptr();
  { const char* e = std::getenv("YSKN_TB_CAP"); tbcap_ = (e && e[0]) ? std::atoi(e) : 0; tbcap_kb_ = g_tb_kb; }
  if (tbcap_) { tb_rec_ = g_tb_rec; if (!tb_rec_) throw std::runtime_error("YSKN_TB_CAP=1: the KVQ profile has no decode records"); if (!tbf_host().scr) throw std::runtime_error("YSKN_TB_CAP=1 needs the fast encoder (YSKN_TB_FAST=1)"); }
#endif
  if (kv_fp16_ && kv8_) throw std::runtime_error("FP16 KV requires YSKN_Q35_KV8=0");
  { const char* e = std::getenv("YSKN_Q35_XSUM"); if (e && e[0]) xsm_ = std::atoi(e) != 0;
    const char* p = std::getenv("YSKN_Q35_XSM_PART"); if (p && p[0]) xsm_part_ = std::atoi(p) != 0; }
  { const char* e = std::getenv("YSKN_Q35_PM"); if (e && std::atoi(e) > 0) PM_ = std::min(Pmax_, std::atoi(e)); }
  m_.c.open(path); m_.cfg = m_.c.model; const ModelDesc& c = m_.cfg;
  if (c.arch != "qwen3_5") throw std::runtime_error("Engine35: arch " + c.arch);
  if (c.hidden != Q35_H || c.heads != Q35_NH || c.kv_heads != Q35_KVH || c.head_dim != Q35_HD || c.ffn % 512)
    throw std::runtime_error("Engine35: kernels are built for the 27B shapes (hidden 5120, 24 / 4 heads of 256)");
  V_ = c.vocab; H_ = c.hidden; F_ = c.ffn; nlayers_ = c.layers; eps_ = c.eps;
  max_ctx_ = ((max_ctx + 63) / 64) * 64; m_.max_ctx = max_ctx_;
  cudaDeviceGetAttribute(&m_.sm_count, cudaDevAttrMultiProcessorCount, 0);
  want_ = m_.sm_count * 8;
  { const char* e = std::getenv("YSKN_Q35_WANT"); if (e && std::atoi(e) > 0) want_ = std::atoi(e); }   
  shared_weights_ = get_shared35(path, want_, xsm_ != 0);
  const json& J = m_.c.raw.at("model");
  const json& types = J.at("layer_types");
  if ((int)types.size() != nlayers_) throw std::runtime_error("Engine35: layer_types size");
  L_.resize(nlayers_);
  int cap_ord = 0, cap_pk_k = 0, cap_pk_v = 0, cap_nfull = 0; size_t cap_bytes = 0, cap_bytes_i8ref = 0;
#if YSKN_KVX_MODE == 3
  if (tbcap_) {                                                       
    stg_rows_ = std::max({Pmax_, layer_rows_, 512});
    stg_k8_ = dz35<int8_t>((size_t)Q35_KVH * stg_rows_ * Q35_HD); stg_v8_ = dz35<int8_t>((size_t)Q35_KVH * stg_rows_ * Q35_HD);
    stg_ks_ = dz35<float>((size_t)Q35_KVH * stg_rows_ * 4); stg_vs_ = dz35<float>((size_t)Q35_KVH * stg_rows_ * 4);
  }
#endif
  for (int l = 0; l < nlayers_; ++l) {
    Q35Layer& Y = L_[l]; const std::string p = "layers." + std::to_string(l) + ".";
    Y.full = types[l].get<std::string>() == "full_attention";
    if (Y.full) {
      Y.q = up(p + "self_attn.q_proj"); Y.k = up(p + "self_attn.k_proj"); Y.v = up(p + "self_attn.v_proj"); Y.o = up(p + "self_attn.o_proj");
      Y.qn = vec(p + "self_attn.q_norm", Q35_HD); Y.kn = vec(p + "self_attn.k_norm", Q35_HD);
      if (kv8_) {
        bool pk_k = false, pk_v = false;
#if YSKN_KVX_MODE == 3
        if (tbcap_) { pk_k = tb_cap_packable(cap_ord, 0); pk_v = tb_cap_packable(cap_ord, 1); }
#endif
        ++cap_ord; ++cap_nfull; cap_pk_k += pk_k; cap_pk_v += pk_v;
        const size_t rows = (size_t)Q35_KVH * max_ctx_;
        const size_t pkb = (size_t)32 * tbcap_kb_;
        if (pk_k) { Y.kp = dz35<uint8_t>(rows * pkb); Y.kps = dz35<float>(rows * 4); cap_bytes += rows * (pkb + 4); } else { Y.k8 = dz35<int8_t>(rows * Q35_HD); Y.ks = dz35<float>(rows * 4); cap_bytes += rows * 260; }
        if (pk_v) { Y.vp = dz35<uint8_t>(rows * pkb); Y.vps = dz35<float>(rows * 4); cap_bytes += rows * (pkb + 4); } else { Y.v8 = dz35<int8_t>(rows * Q35_HD); Y.vs = dz35<float>(rows * 4); cap_bytes += rows * 260; }
        cap_bytes_i8ref += rows * 520;
      }
      else { Y.kc = dz35<uint16_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD * 2); Y.vc = dz35<uint16_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD * 2); }
    } else {
      Y.qkv = up(p + "linear_attn.in_proj_qkv"); Y.z = up(p + "linear_attn.in_proj_z"); Y.out = up(p + "linear_attn.out_proj");
      Y.wa = vec(p + "linear_attn.in_proj_a", (size_t)Q35_GV * H_); Y.wb = vec(p + "linear_attn.in_proj_b", (size_t)Q35_GV * H_);
      Y.conv = vec(p + "linear_attn.conv1d", (size_t)Q35_CONV * 4); Y.A_log = vec(p + "linear_attn.A_log", Q35_GV);
      Y.dt_bias = vec(p + "linear_attn.dt_bias", Q35_GV); Y.gnorm = vec(p + "linear_attn.norm", Q35_GD);
      Y.S = dz35<float>((size_t)Q35_GV * Q35_GD * Q35_GD * 4); Y.hist = dz35<uint16_t>((size_t)3 * Q35_CONV * 2);
      Y.pmix = dz35<uint16_t>((size_t)Mmax_ * Q35_CONV * 2); Y.pab = dz35<float>((size_t)Mmax_ * 96 * 4);
      Y.tq = dz35<float>((size_t)HM_ * Q35_GK * Q35_GD * 4); Y.tk = dz35<float>((size_t)HM_ * Q35_GK * Q35_GD * 4);   
      Y.tv = dz35<float>((size_t)HM_ * Q35_Z * 4); Y.tbeta = dz35<float>((size_t)HM_ * Q35_GV * 4); Y.tg = dz35<float>((size_t)HM_ * Q35_GV * 4);
    }
    Y.gate = up(p + "mlp.gate_proj"); Y.up = up(p + "mlp.up_proj"); Y.down = up(p + "mlp.down_proj");
    Y.ln1 = vec(p + "input_layernorm", H_); Y.ln2 = vec(p + "post_attention_layernorm", H_);
  }
  { int hw = m_.c.tensor("lm_head").cols / 512; const char* e = std::getenv("YSKN_Q35_HEAD_W"); if (e && std::atoi(e) > 0) hw = std::atoi(e);   
    head_ = up("lm_head", hw); }
  { const char* e = std::getenv("YSKN_Q35_EMB_HOST"); if (e && e[0]) emb_host_ = std::atoi(e); }
  fnorm_ = vec("norm", H_);
  { bool coded = false; for (const TensorDesc& t : m_.c.tensors) if (t.name == "embed_tokens") coded = true;
    if (coded) {                                                         
      const TensorDesc& d = m_.c.tensor("embed_tokens"); const size_t nwords = (size_t)V_ * (H_ / 256) * 4 * d.bits;
      if (d.role != "embedding" || d.layout != "row" || d.rows != V_ || d.cols != H_ || d.physical_rows != V_ || d.L != 12 || d.start != "zero" ||
          d.has_rot_out || d.rot_in.kind != "H512" || d.has_rar || d.stream_bytes != nwords * 4)
        throw std::runtime_error("qwen3_5 engine: unsupported coded embedding (role embedding, row layout, L 12, zero start, H512 input rotation)");
      emb_K_ = d.bits;
      emb_codes_ = static_cast<uint32_t*>(shared35_buffer(shared_weights_, "embedding/codes", nwords * 4, [&](void* p) {
        if (cudaMemcpy(p, m_.c.at<uint32_t>(d.stream_off), nwords * 4, cudaMemcpyHostToDevice) != cudaSuccess)
          throw std::runtime_error("shared embedding upload failed"); }));
      emb_lut_ = dz35<uint16_t>((size_t)4096 * 2 * 2); cudaMemcpy(emb_lut_, m_.c.at<uint16_t>(d.lut_off), (size_t)4096 * 2 * 2, cudaMemcpyHostToDevice);
      emb_sc_ = dz35<uint16_t>((size_t)V_ * 2); cudaMemcpy(emb_sc_, m_.c.at<uint16_t>(d.scale_off), (size_t)V_ * 2, cudaMemcpyHostToDevice);
      const std::vector<int8_t> sg = sign_stream_rule(d.rot_in.rule, d.rot_in.sign_seed, d.rot_in.sign_offset, (size_t)H_);
      emb_sg_ = dz35<int8_t>((size_t)H_); cudaMemcpy(emb_sg_, sg.data(), (size_t)H_, cudaMemcpyHostToDevice);
      emb_rows_ = dz35<uint16_t>((size_t)Pmax_ * H_ * 2); emb_iota_ = dz35<int>((size_t)Pmax_ * 4); launch_q35_iota(emb_iota_, Pmax_, st_);
      for (const TensorDesc& t : m_.c.tensors) if (t.name == "embed_tokens.res" || t.name == "embed_tokens.res2") {   
        const size_t nw2 = (size_t)V_ * (H_ / 256) * 4 * t.bits;                                               
        if (t.role != "embedding" || t.layout != "row" || t.rows != V_ || t.cols != H_ || t.physical_rows != V_ || t.L != 12 || t.start != "zero" ||
            t.has_rot_out || t.rot_in.kind != d.rot_in.kind || t.rot_in.sign_offset != d.rot_in.sign_offset || t.has_rar || t.stream_bytes != nw2 * 4)
          throw std::runtime_error("qwen3_5 engine: unsupported residual embedding table");
        const bool third = t.name == "embed_tokens.res2";
        uint32_t*& codes = third ? emb3_codes_ : emb2_codes_; uint16_t*& lut = third ? emb3_lut_ : emb2_lut_; uint16_t*& sc = third ? emb3_sc_ : emb2_sc_;
        (third ? emb3_K_ : emb2_K_) = t.bits; codes = dz35<uint32_t>(nw2 * 4); cudaMemcpy(codes, m_.c.at<uint32_t>(t.stream_off), nw2 * 4, cudaMemcpyHostToDevice);
        lut = dz35<uint16_t>((size_t)4096 * 2 * 2); cudaMemcpy(lut, m_.c.at<uint16_t>(t.lut_off), (size_t)4096 * 2 * 2, cudaMemcpyHostToDevice);
        sc = dz35<uint16_t>((size_t)V_ * 2); cudaMemcpy(sc, m_.c.at<uint16_t>(t.scale_off), (size_t)V_ * 2, cudaMemcpyHostToDevice); }
      if (emb3_codes_ && !emb2_codes_) throw std::runtime_error("qwen3_5 engine: embed_tokens.res2 needs embed_tokens.res");
      emb_coded_ = true; }
    else emb_ = vec("embed_tokens", (size_t)V_ * H_, nullptr, emb_host_ != 0); }   
  
  
  { auto sz = [&](const DevTensor& t, int rows) { return std::max((size_t)t.dev.n_workers * rows, (size_t)Pmax_) * t.dev.N; };   
    size_t need = 0;
    for (const Q35Layer& Y : L_) need = std::max({need, Y.full ? sz(Y.q, Mmax_) + sz(Y.k, Mmax_) + sz(Y.v, Mmax_) : sz(Y.qkv, Mmax_) + sz(Y.z, Mmax_),
                                                   Y.full ? sz(Y.o, Mmax_) : sz(Y.out, Mmax_), sz(Y.gate, Mmax_) + sz(Y.up, Mmax_), sz(Y.down, Mmax_)});
    part_ = dz35<float>(need * 4); part_floats_ = need;
    for (Q35Layer& Y : L_) {
      if (Y.full) { Y.q.part = part_; Y.k.part = part_ + sz(Y.q, Mmax_); Y.v.part = part_ + sz(Y.q, Mmax_) + sz(Y.k, Mmax_); Y.o.part = part_; }
      else { Y.qkv.part = part_; Y.z.part = part_ + sz(Y.qkv, Mmax_); Y.out.part = part_; }
      Y.gate.part = part_; Y.up.part = part_ + sz(Y.gate, Mmax_); Y.down.part = part_;
    }
    head_.part = dz35<float>((size_t)head_.dev.n_workers * HM_ * head_.dev.N * 4); }
  
  { const char* e = std::getenv("YSKN_Q35_G6"); if (e && std::atoi(e) != 0 && xsm_ && !xsm_part_) {
      auto n = [](const DevTensor& t) { return (size_t)t.dev.N * t.dev.Kc; };
      for (const Q35Layer& Y : L_) g6_.weight_bytes = std::max({g6_.weight_bytes,
        n(Y.q), n(Y.k), n(Y.v), n(Y.qkv), n(Y.z), n(Y.o), n(Y.out), n(Y.gate), n(Y.up), n(Y.down)});
      q35_g6_prepare(g6_, m_.sm_count);
      g6_.weights = dz35<uint4>(g6_.weight_bytes); g6_.partial = dz35<long long>(g6_.partial_bytes); g6_.flags = dz35<int>(g6_.flags_bytes);
      std::fprintf(stderr, "[q35_g6] prefill only, temporary workspace %zu bytes\n", g6_.weight_bytes + g6_.partial_bytes + g6_.flags_bytes);
    } }
  const int M = Pmax_;
  h_ = dz35<float>((size_t)M * H_ * 4); xn_ = dz35<uint16_t>((size_t)M * H_ * 2);
  qx_ = dz35<int8_t>((size_t)M * std::max(F_, Q35_Z)); alpha_ = dz35<float>((size_t)M * (std::max(F_, Q35_Z) / 512) * 4);
  z_ = dz35<float>((size_t)M * Q35_Z * 4); core_ = dz35<float>((size_t)M * Q35_Z * 4);
  gq_ = dz35<float>((size_t)M * Q35_GK * Q35_GD * 4); gk_ = dz35<float>((size_t)M * Q35_GK * Q35_GD * 4); gv_ = dz35<float>((size_t)M * Q35_Z * 4);
  gbeta_ = dz35<float>((size_t)M * Q35_GV * 4); gg_ = dz35<float>((size_t)M * Q35_GV * 4);
  q16_ = dz35<uint16_t>((size_t)M * Q35_NH * Q35_HD * 2); gate16_ = dz35<uint16_t>((size_t)M * Q35_NH * Q35_HD * 2);
  { const char* u = std::getenv("YSKN_Q35_ATX_SPLITS"); if (u && std::atoi(u) > 0) atx_splits_ = std::min(64, std::atoi(u));   
    const char* l = std::getenv("YSKN_Q35_ATX_LONG"); if (l) atx_long_ = std::atoi(l); }
  { const int ns = std::max(splits_, atx_splits_);
    pm_ = dz35<float>((size_t)Mmax_ * ns * Q35_NH * 4); pl_ = dz35<float>((size_t)Mmax_ * ns * Q35_NH * 4); pacc_ = dz35<float>((size_t)Mmax_ * ns * Q35_NH * Q35_HD * 4); }
  pos_slices_ = dz35<int>(64 * 4); mtp_pos_slices_ = dz35<int>(64 * 4);
  if (Pmax_ > Mmax_) { pmix_big_ = dz35<uint16_t>((size_t)Pmax_ * Q35_CONV * 2); pab_big_ = dz35<float>((size_t)Pmax_ * 96 * 4); }
  rope_ = dz35<float2>((size_t)max_ctx_ * (Q35_RD / 2) * sizeof(float2)); launch_q35_rope_table(rope_, max_ctx_, st_);
  logits_ = dz35<float>((size_t)HM_ * V_ * 4); blk_val_ = dz35<float>((size_t)HM_ * 256 * 4); blk_idx_ = dz35<int>((size_t)HM_ * 256 * 4);
  blk_max_ = dz35<float>((size_t)HM_ * 256 * 4); blk_sum_ = dz35<float>((size_t)HM_ * 256 * 4);
  tk_val_ = dz35<float>((size_t)HM_ * 256 * 8 * 4); tk_idx_ = dz35<int>((size_t)HM_ * 256 * 8 * 4);
  d_pos_ = dz35<int>(4); d_toks_ = dz35<int>((size_t)M * 4); d_next8_ = dz35<int>((size_t)M * 4); d_target8_ = dz35<int>((size_t)M * 4); d_nll8_ = dz35<float>((size_t)M * 4);
  top_val_ = dz35<float>((size_t)M * 8 * 4); top_idx_ = dz35<int>((size_t)M * 8 * 4);
  cnt_ = dz35<unsigned>((size_t)M * 4); am_cnt_ = dz35<unsigned>((size_t)HM_ * 4); tk_cnt_ = dz35<unsigned>((size_t)HM_ * 4);
  qx2_ = dz35<int8_t>((size_t)Mmax_ * H_); alpha2_ = dz35<float>((size_t)Mmax_ * (H_ / 512) * 4); fin_cnt_ = dz35<unsigned>(64 * 4);   
  d_smp_ = dz35<Q35SmpParams>(sizeof(Q35SmpParams)); kj_val_ = dz35<float>((size_t)HM_ * 256 * Q35_SMP_K * 4);   
  kj_idx_ = dz35<int>((size_t)HM_ * 256 * Q35_SMP_K * 4); kj_cnt_ = dz35<unsigned>((size_t)HM_ * 4);
  { const char* v = std::getenv("YSKN_SMP_VERSION"); smp_p_.version = (v && v[0] == '2') ? 2 : 3; }   
                                                                       
  cudaMemcpy(d_smp_, &smp_p_, sizeof(Q35SmpParams), cudaMemcpyHostToDevice);
  { const char* f = std::getenv("YSKN_Q35_FIN"); if (f && f[0]) opt_fin_ = std::atoi(f); }
  { const char* f = std::getenv("YSKN_Q35_FLOW"); if (f && f[0]) opt_flow_ = std::atoi(f); }
  qxd_ = dz35<int8_t>((size_t)8 * F_); ald_ = dz35<float>((size_t)8 * (F_ / 512) * 4); usd_ = dz35<float>(64 * 4);   
  flow_cnt_ = dz35<unsigned>((size_t)nlayers_ * 64 * 4);
  { const char* f = std::getenv("YSKN_Q35_FLOW_TRACE"); if (f && f[0] == '1') flow_trace_ = dz35<unsigned long long>((size_t)4 * 16384 * 8); }
  flow_prepare();
  { const char* u = std::getenv("YSKN_Q35_ATX"); if (u && u[0]) atx_ = std::atoi(u); }
  if (kv_fp16_ && !atx_) throw std::runtime_error("FP16 KV requires ATX attention");
  std::fprintf(stderr, "[KV-PROTOCOL] target_kv=%s\n", kv8_ ? "int8-row-f32-scale" : (kv_fp16_ ? "fp16" : "bf16"));
  { const char* u = std::getenv("YSKN_ATX_VSP"); if (u && u[0]) atx_vsp_ = std::atoi(u); }
  if (tbcap_) std::fprintf(stderr, "[TB-CAP] capacity mode (k=%d): packed K streams %d/%d, packed V streams %d/%d; KV of the attention layers %.1f MiB (INT8 containers: %.1f MiB); staging %d rows\n",
                           tbcap_kb_, cap_pk_k, cap_nfull, cap_pk_v, cap_nfull, cap_bytes / 1048576.0, cap_bytes_i8ref / 1048576.0, stg_rows_);
  { const char* u = std::getenv("YSKN_L2PF"); if (u && u[0]) l2pf_ = std::atoi(u);
    const char* nm[3] = {"YSKN_L2PF_GU", "YSKN_L2PF_DN", "YSKN_L2PF_IN"};
    for (int i = 0; i < 3; ++i) { const char* v = std::getenv(nm[i]); if (v && v[0]) l2pf_mb_[i] = std::atof(v); } }
  { const char* u = std::getenv("YSKN_Q35_PMATH"); if (u && u[0]) kz_ = std::atoi(u); }
  apply_kz();
  { const char* f = std::getenv("YSKN_Q35_FA"); if (f && f[0]) fa_ = std::atoi(f); const char* s = std::getenv("YSKN_Q35_FA_SPLITS"); if (s && std::atoi(s) > 0) fa_splits_ = std::min(32, std::atoi(s)); }
  if (xsm_) { us_ = dz35<float>((size_t)M * 4); S_ = dz35<float>((size_t)M * Q35_SST * 4); ybuf_ = dz35<float>(part_floats_ * 4); }
  cudaMallocHost(&h_pin_, (size_t)PIN_N * 4); cudaMallocHost(&h_pinf_, (size_t)PINF_N * 4); cudaMallocHost(&h_dec_, (size_t)max_ctx_ * 4);
  d_wk_ = dz35<uint8_t>(64); cudaMallocHost(&h_wk_, 4 * 64);           
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("Engine35 load: ") + cudaGetErrorString(e));
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  std::fprintf(stderr, "[qwen3_5] %d layers loaded, LNORD codes %.2f GB, free %.1f / %.1f GB\n", nlayers_, lno_bytes_ / 1e9, fr / 1e9, tot / 1e9);
}
Engine35::~Engine35() {
  cudaStreamSynchronize(st2_); cudaStreamSynchronize(st_);
  if (kz_) { q35_set_pm(0, st_); q35_flow_set_pm(0, st_); pfa_set_pm(0, st_); cudaStreamSynchronize(st_); }   
                                                                       
  if (gexec_) cudaGraphExecDestroy(gexec_);
  if (gexec_alt_) cudaGraphExecDestroy(gexec_alt_);
  for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);
  for (auto& kv : mtp_.graphs) cudaGraphExecDestroy(kv.second);
  for (auto& kv : rgraph_) cudaGraphExecDestroy(kv.second);
  if (h_pin_) cudaFreeHost(h_pin_);
  if (h_pinf_) cudaFreeHost(h_pinf_);
  if(nb_st_){cudaStreamSynchronize(nb_st_);cudaStreamDestroy(nb_st_);}
  for(int i=0;i<2;++i){if(nb_e1_[i])cudaEventDestroy(nb_e1_[i]);if(nb_e2_[i])cudaEventDestroy(nb_e2_[i]);}
  devmem_release(this);                                                 
  cudaStreamDestroy(st_);
  cudaEventDestroy(ev_v_); cudaEventDestroy(ev_rp_); cudaStreamDestroy(st2_);
  cudaStreamSynchronize(st3_); cudaEventDestroy(ev_g_); cudaEventDestroy(ev_pf_); cudaStreamDestroy(st3_);
}
void Engine35::reset() {
  replay_fence();
  snap_drop(); ++sh_epoch_;                                           
  pos_ = 0; last_M_ = 0; pre_committed_ = false; launch_set_int(d_pos_, 0, st_);
  if (xsm_) { cudaMemsetAsync(part_, 0, part_floats_ * 4, st_);          
              cudaMemsetAsync(head_.part, 0, (size_t)head_.dev.n_workers * HM_ * head_.dev.N * 4, st_); }
  for (Q35Layer& Y : L_) if (!Y.full) { cudaMemsetAsync(Y.S, 0, (size_t)Q35_GV * Q35_GD * Q35_GD * 4, st_); cudaMemsetAsync(Y.hist, 0, (size_t)3 * Q35_CONV * 2, st_); }
  if (mtp_.on) cudaMemsetAsync(mtp_.pend, 0, (size_t)H_ * 2, st_);   
  cudaStreamSynchronize(st_);
}
void Engine35::gemv(std::initializer_list<DevTensor*> ts, int M) { gemv(ts, M, false); }
void Engine35::gemv(std::initializer_list<DevTensor*> ts, int M, bool normed, const GemvLaunch* fin) {
  GemvLaunch P{}; P.M = M; P.stage_bytes = 0;
  for (DevTensor* t : ts) { const int ctas = (t->dev.N / 64 + GEMV_WARPS - 1) / GEMV_WARPS;
    P.seg[P.n_seg] = GemvSeg{t->dev, pt(*t, M), P.n_ctas};
    if (xsmc_ && !xsm_part_ && M >= 4 && t->nw_big > 0) P.seg[P.n_seg].t.n_workers = t->nw_big;   
    P.max_workers = std::max(P.max_workers, P.seg[P.n_seg].t.n_workers); P.n_seg++; P.n_ctas += ctas; }
  P.qx = in2_ ? qx2_ : qx_; P.alpha = in2_ ? alpha2_ : alpha_; in2_ = false;   
  P.lo = lo_now_ ? 1 : 0;                                                
  if (fin) { P.f_mode = fin->f_mode; P.f_tau = fin->f_tau; P.f_sg_out = fin->f_sg_out; P.f_h = fin->f_h; P.f_w = fin->f_w; P.f_sg_in = fin->f_sg_in;
             P.f_qx = fin->f_qx; P.f_alpha = fin->f_alpha; P.f_S = fin->f_S; P.f_xu = fin->f_xu; P.f_cnt = fin->f_cnt; }
  if (xsmc_) { P.xsum = 1; P.us_out = us_; P.xsm_part = xsm_part_;    
    if (normed && M > 16) { P.xsm_S = S_; P.xsm_sst = Q35_SST; P.xsm_nb = H_ / 512; P.xsm_n = (float)H_; P.xsm_eps = eps_; }
    if (xsm_part_ && M <= 16) for (DevTensor* t : ts) if (t->dev.n_workers > 24) throw std::runtime_error("Engine35: XSUM partials need <= 24 workers"); }
  if ((pfa_ && M > Mmax_) || (xsmc_ && M > 16)) {                       
    if (!pfa_eligible(P)) throw std::runtime_error("Engine35: GEMM launch not PFA-eligible");
    if (launch_q35_g6(P, g6_, smp_p_.pm, st_)) return;
    launch_pfa(P, pfa_, pfa_wpc_, st_); return; }
  if (!lnord_eligible(P)) throw std::runtime_error("Engine35: GEMV launch not LNORD-eligible");
  if (pf2_live_) {                                                       
    Q35Pf f; int s = 0; const size_t cap = (size_t)(l2pf2_mb_ * 1048576.0 / (double)std::max<size_t>(1, std::min<size_t>(2, ts.size())));
    for (DevTensor* t : ts) { if (s >= 2 || !t->dev.lno) continue;
      const size_t n = (size_t)(t->dev.N / 16) * (size_t)(t->dev.Kc / 512) * (size_t)t->dev.k_default * 512;
      f.p[s] = t->dev.lno; f.n[s] = std::min(n, cap) & ~(size_t)15; ++s; }
    cudaStreamWaitEvent(st3_, ev_g_, 0); launch_l2pf(f, st3_); }
  launch_lnord(P, st_);
  if (pf2_live_) cudaEventRecord(ev_g_, st_);
  if (xsmc_ && !xsm_part_ && M <= 16 && !P.f_mode) chk_after(ts, M, P.qx, P.alpha);   
}
void Engine35::chk_after(std::initializer_list<DevTensor*> ts, int M, const int8_t* qx, const float* alpha) {
  if (!chk_on_ || chk_.empty()) return;
  for (DevTensor* t : ts) {
    const auto it = chk_.find(t); if (it == chk_.end()) continue;
    unsigned long long* acc = reinterpret_cast<unsigned long long*>(t->part);
    if (mm_kind_ && mm_t_ == t) { launch_q35_chk_fault(acc, mm_kind_, mm_row_, mm_bit_, st_); mm_kind_ = 0; }
    launch_q35_chk(Q35ChkArgs{acc, qx, alpha, it->second.c, t->dev.N, t->dev.Kc, it->second.key, it->second.w_max, it->second.idx, d_mstats_}, M, st_);
  }
}
DevTensor* Engine35::tensor_by_name(const std::string& name) {
  if (name == "D.fc") { if (!bdr_.on) throw std::runtime_error("chk: no drafter loaded"); return &bdr_.fc; }
  if (name.size() > 1 && name[0] == 'D') {                               
    int i = -1; char fld[16] = {0}; static const char* F[9] = {"q", "k", "v", "o", "gate", "up", "down", "akp", "mkp"};
    if (!bdr_.on || std::sscanf(name.c_str(), "D%d.%15s", &i, fld) != 2 || i < 0 || i > 4) throw std::runtime_error("chk: bad drafter tensor " + name);
    for (int k = 0; k < 9; ++k) if (std::string(fld) == F[k]) return &bdr_.L[i][k];
    throw std::runtime_error("chk: bad drafter tensor " + name);
  }
  int l = -1; char field[16] = {0};
  if (std::sscanf(name.c_str(), "L%d.%15s", &l, field) != 2 || l < 0 || l >= (int)L_.size()) throw std::runtime_error("chk: bad tensor name " + name);
  Q35Layer& Y = L_[l]; const std::string f = field;
  DevTensor* t = f == "qkv" ? &Y.qkv : f == "z" ? &Y.z : f == "out" ? &Y.out : f == "q" ? &Y.q : f == "k" ? &Y.k : f == "v" ? &Y.v : f == "o" ? &Y.o :
                 f == "gate" ? &Y.gate : f == "up" ? &Y.up : f == "down" ? &Y.down : nullptr;
  if (!t || !t->lno) throw std::runtime_error("chk: no such tensor in this layer: " + name);
  return t;
}


void Engine35::apply_kz() {
  int on = kz_ ? 1 : 0;
  { const char* p = std::getenv("YSKN_Q35_PM"); if (p && p[0]) on = std::atoi(p) ? 1 : 0; }   
  cudaStreamSynchronize(st_);
  q35_set_pm(on, st_); q35_flow_set_pm(on, st_); pfa_set_pm(on, st_);
  if (visual_on_) launch_q35_visual_rope(rope_, max_ctx_, visual_positions_, visual_prompt_, visual_delta_, st_, on);
  else launch_q35_rope_table(rope_, max_ctx_, st_, on);
  smp_p_.pm = on; cudaStreamSynchronize(st_); cudaMemcpy(d_smp_, &smp_p_, sizeof(Q35SmpParams), cudaMemcpyHostToDevice);
}
void Engine35::drop_pass_graphs() {
  rp_launch(); cudaStreamSynchronize(st2_); replay_fence(); cudaStreamSynchronize(st_);
  for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);
  pgraph_.clear();
  if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; }
  if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; }
  for (auto& kv : mtp_.graphs) cudaGraphExecDestroy(kv.second);
  mtp_.graphs.clear();
  for (auto& row : bdr_.sgraph) for (cudaGraphExec_t& ge : row) if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
}
int Engine35::chk_set(const std::string& name, const long long* c, int Kc, unsigned long long key, int w_max) {
  DevTensor* t = tensor_by_name(name);
  if (Kc != t->dev.Kc) throw std::runtime_error("chk: Kc mismatch for " + name);
  if (!d_mstats_) d_mstats_ = dz35<Q35ChkStats>(sizeof(Q35ChkStats));
  ChkEntry& E = chk_[t];
  if (!E.c) { E.c = dz35<long long>((size_t)Kc * 8); E.idx = (int)chk_.size() - 1; }
  cudaMemcpyAsync(E.c, c, (size_t)Kc * 8, cudaMemcpyHostToDevice, st_); cudaStreamSynchronize(st_); E.key = key; E.w_max = w_max;
  drop_pass_graphs();
  return E.idx;
}
void Engine35::chk_stats(unsigned long long* out, int reset) {
  replay_fence(); cudaStreamSynchronize(st_);
  Q35ChkStats s{};
  if (d_mstats_) cudaMemcpy(&s, d_mstats_, sizeof s, cudaMemcpyDeviceToHost);
  out[0] = s.checks; out[1] = s.eq_fail; out[2] = s.range_fail; out[3] = s.first_set; out[4] = (unsigned long long)(long long)s.first_tensor;
  out[5] = (unsigned long long)(long long)s.first_row; out[6] = s.first_lhs; out[7] = s.first_rhs; out[8] = s.first_viol;
  if (reset && d_mstats_) { cudaMemsetAsync(d_mstats_, 0, sizeof(Q35ChkStats), st_); cudaStreamSynchronize(st_); }   
}
void Engine35::chk_fault(int kind, const std::string& name, int a, int b) {
  DevTensor* t = tensor_by_name(name);
  if (kind == 1 || kind == 2) { mm_kind_ = kind; mm_row_ = a; mm_bit_ = b; mm_t_ = t; return; }
  if (kind == 3) {
    const size_t nw = lnord_bytes(t->dev.N, t->dev.Kc, t->bits) / 4;
    if (a < 0 || (size_t)a >= nw || b < 0 || b > 31) throw std::runtime_error("chk: code word / bit out of range");
    cudaStreamSynchronize(st_); uint32_t w = 0; uint32_t* p = reinterpret_cast<uint32_t*>(t->lno) + a;
    cudaMemcpy(&w, p, 4, cudaMemcpyDeviceToHost); w ^= 1u << b; cudaMemcpyAsync(p, &w, 4, cudaMemcpyHostToDevice, st_); cudaStreamSynchronize(st_); return;
  }
  throw std::runtime_error("chk: unknown fault kind");
}
void Engine35::chk_enable(int on) { chk_on_ = on; drop_pass_graphs(); }
void Engine35::set_capture(const int* layers, int n) {
  drop_pass_graphs();                                                    
  if (cap_) { cudaFree(cap_); cap_ = nullptr; }
  if (d_wkcopy_) { cudaFree(d_wkcopy_); d_wkcopy_ = nullptr; n_wkcopy_ = 0; }   
  cap_layers_.assign(layers, layers + n);
  if (n > 0) cap_ = dz35<float>((size_t)n * Pmax_ * H_ * 4);
}
void Engine35::set_kz_debug(int layer) {
  drop_pass_graphs();                                                    
  kz_dbg_layer_ = layer; kz_dbg_rows_ = 0;
  if (layer >= 0 && !kz_dbg_) { kz_dbg_ = dz35<float>((size_t)Pmax_ * Q35_NH * Q35_HD * 4); kz_dbgq_ = dz35<uint16_t>((size_t)Pmax_ * Q35_NH * Q35_HD * 2); }
}
void Engine35::kz_debug_read(int n, float* o, uint16_t* q) {
  if (!kz_dbg_ || n < 1 || n > kz_dbg_rows_) throw std::runtime_error("kz_debug_read: no debug rows (set_kz_debug + a PMATH pass first)");
  replay_fence(); cudaStreamSynchronize(st_);
  cudaMemcpy(o, kz_dbg_, (size_t)n * Q35_NH * Q35_HD * 4, cudaMemcpyDeviceToHost); cudaMemcpy(q, kz_dbgq_, (size_t)n * Q35_NH * Q35_HD * 2, cudaMemcpyDeviceToHost);
}
void Engine35::kv8_read(int layer, int pos0, int n, int8_t* k8, float* ks, int8_t* v8, float* vs) {
  if (layer < 0 || layer >= (int)L_.size() || !L_[layer].full || !L_[layer].k8) throw std::runtime_error("kv8_read: not a KV8 attention layer");
  if (pos0 < 0 || n < 1 || pos0 + n > max_ctx_) throw std::runtime_error("kv8_read: bad rows");
  const Q35Layer& Y = L_[layer]; replay_fence(); cudaStreamSynchronize(st_);
  for (int h = 0; h < Q35_KVH; ++h) {                                     
    const size_t row = (size_t)h * max_ctx_ + pos0;
    cudaMemcpy(k8 + (size_t)h * n * Q35_HD, Y.k8 + row * Q35_HD, (size_t)n * Q35_HD, cudaMemcpyDeviceToHost);
    cudaMemcpy(v8 + (size_t)h * n * Q35_HD, Y.v8 + row * Q35_HD, (size_t)n * Q35_HD, cudaMemcpyDeviceToHost);
    cudaMemcpy(ks + (size_t)h * n, Y.ks + row, (size_t)n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(vs + (size_t)h * n, Y.vs + row, (size_t)n * 4, cudaMemcpyDeviceToHost);
  }
}
void Engine35::kv16_read(int layer, int pos0, int n, uint16_t* k, uint16_t* v) {
  if (layer < 0 || layer >= (int)L_.size() || !L_[layer].full || !L_[layer].kc)
    throw std::runtime_error("kv16_read: not a 16-bit attention layer");
  if (pos0 < 0 || n < 1 || pos0 + n > pos_) throw std::runtime_error("kv16_read: bad rows");
  const Q35Layer& Y = L_[layer]; replay_fence(); cudaStreamSynchronize(st_);
  for (int h = 0; h < Q35_KVH; ++h) {
    const size_t row = (size_t)h * max_ctx_ + pos0;
    cudaMemcpy(k + (size_t)h * n * Q35_HD, Y.kc + row * Q35_HD, (size_t)n * Q35_HD * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(v + (size_t)h * n * Q35_HD, Y.vc + row * Q35_HD, (size_t)n * Q35_HD * 2, cudaMemcpyDeviceToHost);
  }
}
void Engine35::capture_rows(int k, int n, float* out) {
  if (k < 0 || k >= (int)cap_layers_.size() || n < 1 || n > Pmax_) throw std::runtime_error("capture_rows: no such capture slot / rows");
  replay_fence(); cudaStreamSynchronize(st_); cudaMemcpy(out, cap_ + (size_t)k * Pmax_ * H_, (size_t)n * H_ * 4, cudaMemcpyDeviceToHost);
}



Q35Pf Engine35::pf_of(std::initializer_list<const DevTensor*> ts, double mb) const {
  Q35Pf f; int s = 0; const size_t cap = mb < 0 ? ~(size_t)0 : (size_t)(mb * 1048576.0 / (double)std::max<size_t>(1, std::min<size_t>(2, ts.size())));
  for (const DevTensor* t : ts) { if (s >= 2 || !t->dev.lno) continue;
    const size_t n = (size_t)(t->dev.N / 16) * (size_t)(t->dev.Kc / 512) * (size_t)t->dev.k_default * 512;
    f.p[s] = t->dev.lno; f.n[s] = std::min(n, cap) & ~(size_t)15; ++s; }
  return f; }
void Engine35::res_norm(const DevTensor* src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out) {
  const Q35Pf pf = next_pf_; next_pf_ = Q35Pf{};                         
  if (ks(64)) return;
  if (xsmc_) {                                                           
    const uint16_t* E = nullptr; const int* I = d_toks_; if (!src) emb_src(d_toks_, M, E, I);
    launch_q35_res_norm_xsm(Q35XsmResArgs{src ? pt(*src, M) : nullptr, us_, src ? src->tau : nullptr, src ? src->signs_out : nullptr,
                                          E, I, M, h_, w, sg_in, qx_, alpha_, S_, nmode(M), src ? src->dev.n_workers : 0, xn_out, pf, lo_now_ ? 1 : 0}, st_);
    return; }
  Q35ResNormArgs A{}; A.M = M; A.N = H_; A.h = h_; A.w = w; A.eps = eps_; A.sg_in = sg_in; A.qx = qx_; A.alpha = alpha_; A.xn = xn_out; A.cnt = cnt_;
  if (src) { A.part = src->part; A.tau = src->tau; A.nw = nwk(*src, M); A.sg_out = src->signs_out; }
  else emb_src(d_toks_, M, A.emb, A.ids);
  if (M > Mmax_) { launch_q35_res_norm_row(A, st_); return; }            
  if (src && M > 16) { launch_q35_residual(A, st_); A.part = nullptr; }
  launch_q35_res_norm(A, st_);
}


void Engine35::emb_src(const int* ids, int M, const uint16_t*& emb, const int*& gids) {
  if (visual_on_ && ids == d_toks_) {
    if (M > Pmax_) throw std::runtime_error("visual: embedding pass exceeds scratch");
    if (emb_coded_) launch_q35_embed(Q35EmbArgs{emb_codes_, emb_K_, emb_lut_, emb_sc_, emb_sg_, ids, emb_rows_, emb2_codes_, emb2_K_, emb2_lut_, emb2_sc_,
                                               emb3_codes_, emb3_K_, emb3_lut_, emb3_sc_}, M, st_);
    launch_q35_visual_embed(emb_rows_, emb_coded_ ? nullptr : emb_, ids, visual_features_, visual_map_, d_pos_, M, max_ctx_, st_);
    emb = emb_rows_; gids = emb_iota_; return;
  }
  if (!emb_coded_) { emb = emb_; gids = ids; return; }
  if (M > Pmax_) throw std::runtime_error("Engine35: embedding pass larger than the decoded-row scratch");
  if (M > 0) launch_q35_embed(Q35EmbArgs{emb_codes_, emb_K_, emb_lut_, emb_sc_, emb_sg_, ids, emb_rows_, emb2_codes_, emb2_K_, emb2_lut_, emb2_sc_,
                                         emb3_codes_, emb3_K_, emb3_lut_, emb3_sc_}, M, st_);
  emb = emb_rows_; gids = emb_iota_;
}

static unsigned long long vis_row_hash(const uint16_t* row, int H) {
  unsigned long long h = 1469598103934665603ULL; const unsigned char* b = reinterpret_cast<const unsigned char*>(row);
  const size_t n = (size_t)H * 2;
  for (size_t i = 0; i + 8 <= n; i += 8) { unsigned long long w; std::memcpy(&w, b + i, 8); h ^= w; h *= 1099511628211ULL; }
  for (size_t i = n & ~(size_t)7; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
  return h;
}



bool Engine35::visual_prefix_agrees(const std::vector<int>& map, const int32_t* positions, int n_prompt, int delta,
                                    const std::vector<unsigned long long>& hash) const {
  if (pos_ <= 0 || pos_ > max_ctx_) return false;
  if (visual_on_ && ((int)vis_map_.size() != max_ctx_ || (int)vis_pos_.size() != 3 * vis_n_)) return false;
  for (int p = 0; p < pos_; ++p) {
    const int om = visual_on_ ? vis_map_[p] : -1, nm = map[p];
    if ((om < 0) != (nm < 0)) return false;
    if (nm >= 0 && (om >= (int)vis_hash_.size() || nm >= (int)hash.size() || vis_hash_[om] != hash[nm])) return false;
    for (int a = 0; a < 3; ++a) {
      const int op = !visual_on_ ? p : (p < vis_n_ ? vis_pos_[(size_t)a * vis_n_ + p] : p + vis_delta_);
      const int np = p < n_prompt ? positions[(size_t)a * n_prompt + p] : p + delta;
      if (op != np) return false;
    }
  }
  return true;
}
void Engine35::set_visual(const uint16_t* features, const int32_t* slots, int n_image,
                          const int32_t* positions, int n_prompt, int delta) {
  DevMemScope mem_scope(this);
  if (n_image < 0 || n_image > n_prompt || n_prompt < 0 || n_prompt >= max_ctx_)
    throw std::runtime_error("visual: invalid image token count or prompt length");
  if (n_image && (!features || !slots || !positions || !n_prompt))
    throw std::runtime_error("visual: missing input");
  if (n_image && mtp_.on) throw std::runtime_error("visual: MTP drafter is not supported; use Bdraft or plain decode");
  std::vector<int> map(max_ctx_, -1);
  for (int i = 0; i < n_image; ++i) {
    if (slots[i] < 0 || slots[i] >= n_prompt || map[slots[i]] != -1) throw std::runtime_error("visual: invalid slot");
    map[slots[i]] = i;
  }
  if (n_image) {
    for (int i = 0; i < 3 * n_prompt; ++i) if (positions[i] < 0 || positions[i] >= max_ctx_)
      throw std::runtime_error("visual: invalid position");
    if (n_prompt + delta < 0 || max_ctx_ - 1LL + delta < 0) throw std::runtime_error("visual: invalid position delta");
  }
  
  std::vector<unsigned long long> hash((size_t)n_image, 0ULL);
  for (int i = 0; i < n_image; ++i) hash[i] = vis_row_hash(features + (size_t)i * H_, H_);
  const bool keep = n_image > 0 && visual_prefix_agrees(map, positions, n_prompt, delta, hash);
  drop_pass_graphs();
  if (!keep) reset();
  visual_on_ = n_image > 0; visual_prompt_ = n_prompt; visual_delta_ = delta;
  vis_n_ = n_image ? n_prompt : 0; vis_delta_ = n_image ? delta : 0; vis_map_ = map; vis_hash_.swap(hash);
  if (n_image) vis_pos_.assign(positions, positions + (size_t)3 * n_prompt); else vis_pos_.clear();
  if (visual_on_) {
    if (!visual_map_) visual_map_ = dz35<int>((size_t)max_ctx_ * 4);
    if (!visual_positions_) visual_positions_ = dz35<int>((size_t)max_ctx_ * 3 * 4);
    if (!emb_rows_) emb_rows_ = dz35<uint16_t>((size_t)Pmax_ * H_ * 2);
    if (!emb_iota_) { emb_iota_ = dz35<int>((size_t)Pmax_ * 4); launch_q35_iota(emb_iota_, Pmax_, st_); }
    if (n_image > visual_cap_) {
      if (visual_features_) cudaFree(visual_features_);
      visual_features_ = nullptr; visual_cap_ = 0;
      visual_features_ = dz35<uint16_t>((size_t)n_image * H_ * 2); visual_cap_ = n_image;
    }
    cudaMemcpyAsync(visual_map_, map.data(), (size_t)max_ctx_ * 4, cudaMemcpyHostToDevice, st_);
    cudaMemcpyAsync(visual_positions_, positions, (size_t)3 * n_prompt * 4, cudaMemcpyHostToDevice, st_);
    cudaMemcpyAsync(visual_features_, features, (size_t)n_image * H_ * 2, cudaMemcpyHostToDevice, st_);
  }
  int pm = kz_ ? 1 : 0; const char* e = std::getenv("YSKN_Q35_PM"); if (e && e[0]) pm = std::atoi(e) ? 1 : 0;
  if (visual_on_) launch_q35_visual_rope(rope_, max_ctx_, visual_positions_, n_prompt, delta, st_, pm);
  else launch_q35_rope_table(rope_, max_ctx_, st_, pm);
  cudaStreamSynchronize(st_);
  const auto err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(std::string("visual: ") + cudaGetErrorString(err));
}
void Engine35::gemv_res(DevTensor& src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out) {
  const bool G = !(kskip_ & 16), S = !(kskip_ & 32);
  if (opt_fin_ && !tq_qx_ && M == 1 && xsmc_ && nmode(M) == 0 && !in2_ && G && S && !ks(64) && src.dev.N == H_) {
    GemvLaunch F{}; F.f_mode = opt_fin_; F.f_tau = src.tau; F.f_sg_out = src.signs_out; F.f_h = h_; F.f_w = w; F.f_sg_in = sg_in;
    F.f_qx = qx2_; F.f_alpha = alpha2_; F.f_S = S_; F.f_xu = xn_out; F.f_cnt = fin_cnt_;
    next_pf_ = Q35Pf{}; gemv({&src}, M, false, &F); in2_ = true; return; }
  if (G) gemv({&src}, M);
  if (tq_qx_ && src.dev.N==H_ && src.dev.Kc==F_) { qx_=tq_qx_;alpha_=tq_al_; }
  if (S) res_norm(&src, M, w, sg_in, xn_out);
}
void Engine35::attn_block(Q35Layer& Y, int M, const int* dpos, int* slices) {
  int cap_ord = -1;                                                    
  if (tbcap_ && (Y.kp || Y.vp)) { cap_ord = 0; for (const Q35Layer& Z : L_) { if (&Z == &Y) break; if (Z.full) ++cap_ord; } if (kz_) throw std::runtime_error("YSKN_TB_CAP=1 does not support PMATH"); }
#if YSKN_KVX_MODE == 2 || YSKN_KVX_MODE == 3
  if (!atx_ || kz_) throw std::runtime_error("Rotated INT8 requires ATX attention");
#endif
  const bool G = !(kskip_ & 16), S = !(kskip_ & 32);
  if (G) gemv({&Y.q, &Y.k, &Y.v}, M, true);
  if(tq_qx_) {qx_=scr_qx_;alpha_=scr_al_;}
  if (S && !ks(8192)) { Q35AttnPostArgs X{pt(Y.q, M), pt(Y.k, M), pt(Y.v, M), Y.q.tau, Y.k.tau, Y.v.tau, nwk(Y.q, M), nwk(Y.k, M), nwk(Y.v, M),
                                        Y.q.signs_out, Y.k.signs_out, Y.v.signs_out, Y.qn, Y.kn, eps_, M, dpos, rope_, q16_, gate16_, Y.kc, Y.vc, max_ctx_};
    if (xsmc_) X.xsm = Q35Xsm{us_, S_, eps_, nmode(M, true)};
    X.k8 = Y.k8; X.v8 = Y.v8; X.ks = Y.ks; X.vs = Y.vs; X.kv_fp16 = kv_fp16_;   
#if YSKN_KVX_MODE == 3
    { const long li = (long)(&Y - L_.data()); int ord = -1;                
      if (li >= 0 && li < (long)L_.size() && &L_[(size_t)li] == &Y && L_[(size_t)li].full) { ord = 0; for (long i = 0; i < li; ++i) if (L_[(size_t)i].full) ++ord; }
      X.tb = ord >= 0 ? tb_dev_ptr() : nullptr; X.tb_layer = ord < 0 ? 0 : ord; }
#endif
#if YSKN_KVX_MODE == 3
    if (cap_ord >= 0) {                                                
      if (M > stg_rows_) throw std::runtime_error("YSKN_TB_CAP=1: a pass has more rows than the staging container");
      X.pks = 32 * tbcap_kb_;
      if (Y.kp) { X.k8 = stg_k8_; X.ks = stg_ks_; X.kstg = stg_rows_; X.kp = Y.kp; X.kps = Y.kps; }
      if (Y.vp) { X.v8 = stg_v8_; X.vs = stg_vs_; X.vstg = stg_rows_; X.vp = Y.vp; X.vps = Y.vps; }
    }
#endif
    if (pf_on(2, M)) X.pf = pf_of({&Y.o}, -1.0);                          
    if (wk_now_) launch_wk_attn_post(X, d_wk_, st_); else launch_q35_attn_post(X, st_);
#if YSKN_KVX_MODE == 3
    if (X.tb && X.k8 && X.v8) launch_tb_requant(X, st_);                  
#endif
  }
  const bool fa = fa_ && M > HM_; const int sp = atx_ ? atx_splits_ : (fa ? fa_splits_ : splits_);   
  const bool kz = kz_ && atx_ && Y.k8;                                    
  float* const kdbg = (kz && kz_dbg_ && (int)(&Y - L_.data()) == kz_dbg_layer_) ? kz_dbg_ : nullptr;
  if (kdbg && S) { cudaMemcpyAsync(kz_dbgq_, q16_, (size_t)M * Q35_NH * Q35_HD * 2, cudaMemcpyDeviceToDevice, st_); kz_dbg_rows_ = M; }
  if (S && M <= Mmax_) {
    if (!ks(16384)) { Q35AttnArgs AT{q16_, Y.kc, Y.vc, max_ctx_, M, dpos, sp, pm_, pl_, pacc_, 1}; AT.k8 = Y.k8; AT.v8 = Y.v8; AT.ks = Y.ks; AT.vs = Y.vs; AT.kv_fp16 = kv_fp16_;
                      if (atx_ && atx_long_ > 0 && sp != atx_short_) { AT.splits_short = atx_short_; AT.long_len = atx_long_; }
                      AT.vsp = atx_vsp_;
                      if (kz) launch_q35_kz(AT, wk_now_ ? d_wk_ : nullptr, st_);
                      else if (atx_) { if (cap_ord >= 0) launch_q35_atx_pk(AT, wk_now_ ? d_wk_ : nullptr, Q35PkArgs{Y.kp, Y.vp, Y.kps, Y.vps, tb_rec_, cap_ord, tbcap_kb_}, st_); else launch_q35_atx(AT, wk_now_ ? d_wk_ : nullptr, st_); }
                      else if (wk_now_) launch_wk_attention(AT, d_wk_, st_);
                      else if (fa) launch_q35_fa(AT, st_); else launch_q35_attention(AT, st_); }
    if (!ks(16384)) { Q35CombineArgs CB{pm_, pl_, pacc_, sp, M, gate16_, Y.o.signs_in, qx_, alpha_}; CB.lo = lo_now_ ? 1 : 0;
                      if (atx_ && atx_long_ > 0 && sp != atx_short_) { CB.d_pos = dpos; CB.wk = wk_now_ ? d_wk_ : nullptr; CB.splits_short = atx_short_; CB.long_len = atx_long_; }
                      if (kz) launch_q35_kz_combine(CB, kdbg, st_); else launch_q35_combine(CB, st_); }
  } else if (S) {                                                       
    launch_q35_pos_slices(dpos, slices, (M + Mmax_ - 1) / Mmax_, Mmax_, st_);
    for (int s = 0; s < M; s += Mmax_) {
      const int Ms = std::min(Mmax_, M - s); const size_t qo = (size_t)s * Q35_NH * Q35_HD;
      Q35AttnArgs AS{q16_ + qo, Y.kc, Y.vc, max_ctx_, Ms, slices + s / Mmax_, sp, pm_, pl_, pacc_}; AS.k8 = Y.k8; AS.v8 = Y.v8; AS.ks = Y.ks; AS.vs = Y.vs; AS.kv_fp16 = kv_fp16_;
      if (atx_ && atx_long_ > 0 && sp != atx_short_) { AS.splits_short = atx_short_; AS.long_len = atx_long_; }
      AS.vsp = atx_vsp_;
      if (kz) launch_q35_kz(AS, nullptr, st_); else if (atx_) { if (cap_ord >= 0) launch_q35_atx_pk(AS, nullptr, Q35PkArgs{Y.kp, Y.vp, Y.kps, Y.vps, tb_rec_, cap_ord, tbcap_kb_}, st_); else launch_q35_atx(AS, nullptr, st_); } else if (fa) launch_q35_fa(AS, st_); else launch_q35_attention(AS, st_);
      Q35CombineArgs CB{pm_, pl_, pacc_, sp, Ms, gate16_ + qo, Y.o.signs_in, qx_ + qo, alpha_ + (size_t)s * (Q35_NH * Q35_HD / 512)};
      if (atx_ && atx_long_ > 0 && sp != atx_short_) { CB.d_pos = slices + s / Mmax_; CB.splits_short = atx_short_; CB.long_len = atx_long_; }
      if (kz) launch_q35_kz_combine(CB, kdbg ? kdbg + qo : nullptr, st_); else launch_q35_combine(CB, st_);
    }
  }
  if (pf_on(4, M)) next_pf_ = pf_of({&Y.gate, &Y.up}, l2pf_mb_[0]);
  gemv_res(Y.o, M, Y.ln2, Y.gate.signs_in, nullptr);
}
void Engine35::flow_mlp(Q35Layer& Y, int M, int l, bool prepare_only) {
  Q35FlowMlpArgs A;
  auto mk = [&](std::initializer_list<DevTensor*> ts, const int8_t* qx, const float* alpha, float* us_out) {   
    GemvLaunch P{}; P.M = M; P.stage_bytes = 0;
    for (DevTensor* t : ts) { P.seg[P.n_seg] = GemvSeg{t->dev, pt(*t, M), 0};
      if (M >= 4 && t->nw_big > 0) P.seg[P.n_seg].t.n_workers = t->nw_big;
      P.max_workers = std::max(P.max_workers, P.seg[P.n_seg].t.n_workers); P.n_seg++; }
    P.qx = qx; P.alpha = alpha; P.xsum = 1; P.us_out = us_out; P.xsm_part = 0; return P; };
  A.gu = mk({&Y.gate, &Y.up}, qx_, alpha_, us_); A.dn = mk({&Y.down}, qxd_, ald_, usd_); A.M = M; A.F = F_;
  A.tau_g = Y.gate.tau; A.tau_u = Y.up.tau; A.sg_g = Y.gate.signs_out; A.sg_u = Y.up.signs_out; A.sg_dn_in = Y.down.signs_in;
  A.S = S_; A.eps = eps_; A.qxd = qxd_; A.ald = ald_; A.tau_dn = Y.down.tau; A.sg_dn_out = Y.down.signs_out; A.h = h_;
  if (l + 1 < nlayers_) { const Q35Layer& Nx = L_[l + 1]; A.w = Nx.ln1; A.sg_next = Nx.full ? Nx.q.signs_in : Nx.qkv.signs_in; A.xu = Nx.full ? nullptr : xn_; }
  else { A.w = fnorm_; A.sg_next = head_.signs_in; A.xu = mtp_.on ? xn_ : nullptr; }
  A.qx = qx_; A.alpha = alpha_; A.S_out = S_;
  A.cnt = flow_cnt_ + (size_t)l * 64; A.cnt_next = flow_cnt_ + (size_t)((l + 1) % nlayers_) * 64;
  if (prepare_only) { prepare_q35_flow_mlp(A); return; }
  if (l == 0) A.trace = flow_trace_;
  launch_q35_flow_mlp(A, st_);
}
void Engine35::mlp_block(Q35Layer& Y, int M, bool down) {
  const bool G = !(kskip_ & 16), S = !(kskip_ & 32);
  if (G) gemv({&Y.gate, &Y.up}, M, true);
  if (S && !ks(128)) { Q35MlpArgs X{pt(Y.gate, M), pt(Y.up, M), Y.gate.tau, Y.up.tau, nwk(Y.gate, M), nwk(Y.up, M), Y.gate.signs_out, Y.up.signs_out, M, F_,
                            Y.down.signs_in, qx_, alpha_};
    if (xsmc_) X.xsm = Q35Xsm{us_, S_, eps_, nmode(M, true)};
    if (pf_on(8, M)) X.pf = pf_of({&Y.down}, l2pf_mb_[1]);
    X.lo = lo_now_ ? 1 : 0;
    launch_q35_mlp(X, st_); }
  if (G && down) gemv({&Y.down}, M);
}
void Engine35::layer(int l, int M, int n_commit) {
  Q35Layer& Y = L_[l]; const bool G = !(kskip_ & 16), S = !(kskip_ & 32);
  if (!Y.full) {
    uint16_t* const pmx = M > Mmax_ ? pmix_big_ : Y.pmix; float* const pab = M > Mmax_ ? pab_big_ : Y.pab;
    const bool fuse_ab = M <= 32;                                          
    Q35AbArgs AB{xn_, Y.wb, Y.wa, M, pab};
    if (xsmc_) { AB.S = S_; AB.eps = eps_; }                               
    if (S && !ks(256) && !fuse_ab) launch_q35_ab(AB, st_);
    if (G) gemv({&Y.qkv, &Y.z}, M, true);
    if(tq_qx_) {qx_=scr_qx_;alpha_=scr_al_;}
    if (S && !ks(512)) {
      Q35MixArgs X{pt(Y.qkv, M), pt(Y.z, M), Y.qkv.tau, Y.z.tau, nwk(Y.qkv, M), nwk(Y.z, M), Y.qkv.signs_out, Y.z.signs_out, M, pmx, z_};
      if (fuse_ab && !ks(256)) { X.ab = AB; X.with_ab = 1; }
      if (xsmc_) X.xsm = Q35Xsm{us_, S_, eps_, nmode(M, true)};
      if (pf_on(1, M)) X.pf = pf_of({&Y.out}, -1.0);                       
      launch_q35_mix(X, st_);
    } else if (S && fuse_ab && !ks(256)) launch_q35_ab(AB, st_);
    const bool keep = opt_kconv_ && M <= HM_;                         
    float *cq = keep ? Y.tq : gq_, *ck = keep ? Y.tk : gk_, *cv = keep ? Y.tv : gv_, *cb = keep ? Y.tbeta : gbeta_, *cg = keep ? Y.tg : gg_;
    if (S && !ks(1024)) { const Q35ConvArgs CA{pmx, Y.conv, Y.hist, pab, Y.A_log, Y.dt_bias, M, n_commit, cq, ck, cv, cb, cg};
                          if (wk_now_) launch_wk_conv(CA, d_wk_, st_); else launch_q35_conv(CA, st_); }
    if (S && !ks(2048)) { const Q35StateArgs SA{cq, ck, cv, cb, cg, Y.S, M, n_commit, core_, pmx, Y.hist};
                          if (wk_now_) launch_wk_state(SA, d_wk_, st_); else launch_q35_state(SA, st_); }
    if (S && !ks(4096)) launch_q35_gdn_out(Q35GdnOutArgs{core_, z_, Y.gnorm, eps_, M, Y.out.signs_in, qx_, alpha_, lo_now_ ? 1 : 0}, st_);
    if (pf_on(4, M)) next_pf_ = pf_of({&Y.gate, &Y.up}, l2pf_mb_[0]);
    gemv_res(Y.out, M, Y.ln2, Y.gate.signs_in, nullptr);
  } else attn_block(Y, M, d_pos_, pos_slices_);
  if (!tq_qx_ && flow_ok(M)) { flow_mlp(Y, M, l); return; }                       
  mlp_block(Y, M, false);
  if (l + 1 < nlayers_ && pf_on(16, M)) { const Q35Layer& Nx = L_[l + 1]; next_pf_ = Nx.full ? pf_of({&Nx.q, &Nx.k}, l2pf_mb_[2]) : pf_of({&Nx.qkv, &Nx.z}, l2pf_mb_[2]); }
  if (l + 1 < nlayers_) { const Q35Layer& Nx = L_[l + 1]; gemv_res(Y.down, M, Nx.ln1, Nx.full ? Nx.q.signs_in : Nx.qkv.signs_in, Nx.full ? nullptr : xn_); }
  else gemv_res(Y.down, M, fnorm_, head_.signs_in, mtp_.on ? xn_ : nullptr);   
}


void Engine35::head_slice(int row0, int rows, bool advance, bool topk, bool nll) {
  GemvLaunch P{}; P.M = rows; P.stage_bytes = 0; P.seg[0] = GemvSeg{head_.dev, head_.part, 0}; P.n_seg = 1;
  P.n_ctas = (head_.dev.N / 64 + GEMV_WARPS - 1) / GEMV_WARPS; P.max_workers = head_.dev.n_workers;
  P.qx = (in2_ ? qx2_ : qx_) + (size_t)row0 * H_; P.alpha = (in2_ ? alpha2_ : alpha_) + (size_t)row0 * (H_ / 512);   
  if (lo_now_) { P.qx = qx_; P.lo = 1; P.lo_t0 = row0; }                
  if (xsmc_) { P.xsum = 1; P.us_out = us_; }
  launch_lnord(P, st_);
  ArgmaxArgs AA{head_.part, head_.tau, V_, logits_, d_next8_ + row0, blk_val_, blk_idx_, head_.dev.n_workers, rows, 0, rows}; AA.cnt = am_cnt_;
  if (xsmc_) { AA.us = us_; AA.rr = S_ + (size_t)row0 * Q35_SST; AA.xsm_eps = eps_; AA.xsm_nb = H_ / 512; AA.xsm_stride = Q35_SST; AA.xsm_n = (float)H_; }
  if (advance && !smp_) { AA.adv_pos = d_pos_; AA.adv_tok = d_toks_; AA.adv_greedy = 1; }
  launch_argmax(AA, st_);
  if (smp_) {                                                         
    Q35SmpArgs KA{logits_, V_, rows, row0, d_next8_ + row0, d_pos_, wk_now_ ? d_wk_ : nullptr, d_smp_, kj_val_, kj_idx_, kj_cnt_};
    if (advance) { KA.adv_pos = d_pos_; KA.adv_tok = d_toks_; }
    if (topk && q35_smp2_on()) { KA.top_val = top_val_ + (size_t)row0 * 8; KA.top_idx = top_idx_ + (size_t)row0 * 8; }   
    if (k20_on_ && q35_smp2_on()) { KA.topk_val = d_k20v_ + (size_t)row0 * Q35_SMP_K; KA.topk_idx = d_k20i_ + (size_t)row0 * Q35_SMP_K; }   
    launch_q35_smp(KA, st_);
  }
  if (topk && !(smp_ && q35_smp2_on())) { TopkArgs TA{logits_, V_, tk_val_, tk_idx_, top_val_ + (size_t)row0 * 8, top_idx_ + (size_t)row0 * 8, rows}; TA.cnt = tk_cnt_; launch_topk8(TA, st_); }
  if (nll) launch_nll(NllArgs{logits_, V_, d_target8_ + row0, d_nll8_ + row0, blk_max_, blk_sum_, rows}, st_);
}
void Engine35::pass_body(int M, bool commit, int upto, int head_mode, bool topk, bool nll, bool advance) {
  g_pdl_rows = M; xsmc_ = xsm_ != 0; wk_now_ = wk_pass(M, commit, head_mode);
  pf2_live_ = l2pf2_mb_ > 0 && xsmc_ && M <= 16; if (pf2_live_) cudaEventRecord(ev_g_, st_);   
  lo_now_ = opt_lo_ && xsmc_ && M <= 8 && !opt_flow_ && !(opt_fin_ && M == 1) && !(chk_on_ && !chk_.empty());   
  const Q35Layer& L0 = L_[0];
  res_norm(nullptr, M, L0.ln1, L0.full ? L0.q.signs_in : L0.qkv.signs_in, L0.full ? nullptr : xn_);   
  for (int l = 0; l < upto; ++l) {
    layer(l, M, commit ? M : 0);
    if (cap_) for (size_t k = 0; k < cap_layers_.size(); ++k) if (cap_layers_[k] == l)   
      cudaMemcpyAsync(cap_ + k * (size_t)Pmax_ * H_, h_, (size_t)M * H_ * 4, cudaMemcpyDeviceToDevice, st_);
  }
  g_pdl_rows = std::min(M, HM_);
  if (head_mode == HEAD_LAST) head_slice(M - 1, 1, advance, topk, nll);
  else if (head_mode == HEAD_ALL) for (int s = 0; s < M; s += HM_) head_slice(s, std::min(HM_, M - s), false, topk, nll);
  if (xsmc_ && mtp_.on && upto == nlayers_ && !ks(32)) { g_pdl_rows = M; launch_q35_xsm_xn(h_, S_, fnorm_, eps_, M, xn_, st_); }   
  if (pf2_live_) { cudaEventRecord(ev_pf_, st3_); cudaStreamWaitEvent(st_, ev_pf_, 0); pf2_live_ = false; }   
  lo_now_ = false;
  xsmc_ = false; wk_now_ = false; in2_ = false;
}
void Engine35::replay_commit(int rows) {
  auto body = [&] {
    g_pdl_rows = rows;
    for (Q35Layer& Y : L_) {
      if (Y.full) continue;
      if (opt_kconv_) {                                               
        launch_q35_state(Q35StateArgs{Y.tq, Y.tk, Y.tv, Y.tbeta, Y.tg, Y.S, rows, rows, core_, Y.pmix, Y.hist}, st_); continue; }
      launch_q35_conv(Q35ConvArgs{Y.pmix, Y.conv, Y.hist, Y.pab, Y.A_log, Y.dt_bias, rows, rows, gq_, gk_, gv_, gbeta_, gg_}, st_);
      launch_q35_state(Q35StateArgs{gq_, gk_, gv_, gbeta_, gg_, Y.S, rows, rows, core_, Y.pmix, Y.hist}, st_);
    }
  };
  if (!opt_replay_graph_) { replay_fence(); body(); return; }
  const int rkey = rows + (opt_kconv_ ? 1000 : 0);
  auto it = rgraph_.find(rkey);                                         
  if (it == rgraph_.end()) {
    cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); body(); cudaStreamEndCapture(st_, &g);
    if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("Engine35: replay graph instantiate failed");
    cudaGraphDestroy(g); it = rgraph_.emplace(rkey, ge).first;
  }
  if (opt_rp_async_) {                                                 
    if (opt_rp_defer_) { rp_launch(); cudaEventRecord(ev_v_, st_); rp_graph_ = it->second; rp_deferred_ = true; rp_pending_ = true; return; }   
    cudaEventRecord(ev_v_, st_); cudaStreamWaitEvent(st2_, ev_v_, 0); cudaGraphLaunch(it->second, st2_); cudaEventRecord(ev_rp_, st2_); rp_pending_ = true; return; }
  replay_fence(); cudaGraphLaunch(it->second, st_);
}
void Engine35::lfrow_table(const uint8_t* tab48) {
  uint8_t* stg = h_wk_ + 64 * (wk_slot_++ & 3); std::memcpy(stg, tab48, 48);
  cudaMemcpyAsync(d_wk_, stg, 48, cudaMemcpyHostToDevice, st_); wk_set_ = true;
}
void Engine35::wk_plain_table() {
  uint8_t t[48] = {}; for (int i = 0; i < 16; ++i) { t[i] = (uint8_t)i; t[16 + i] = (uint8_t)i; }
  lfrow_table(t);
}
void Engine35::lfrow_fill(int k, int M) { if (mtp_.on) launch_wk_fill(d_toks_, d_wk_, mtp_.lg_idx, k, M, st_); }
void Engine35::hcap_cfg(int* out8) const { std::memcpy(out8, h_pin_ + PIN_HB, 8 * 4); }
void Engine35::lfrow_promote(int src, int dst) {
  if (!d_wkcopy_) {                                                   
    std::vector<WkCopy> v;
    for (Q35Layer& Y : L_) {
      if (!Y.full) {
        v.push_back({(unsigned long long)Y.pmix, Q35_CONV * 2, 0}); v.push_back({(unsigned long long)Y.pab, 96 * 4, 0});
        v.push_back({(unsigned long long)Y.tq, Q35_GK * Q35_GD * 4, 0}); v.push_back({(unsigned long long)Y.tk, Q35_GK * Q35_GD * 4, 0});
        v.push_back({(unsigned long long)Y.tv, Q35_Z * 4, 0}); v.push_back({(unsigned long long)Y.tbeta, Q35_GV * 4, 0});
        v.push_back({(unsigned long long)Y.tg, Q35_GV * 4, 0});
      } else if (Y.kp || Y.vp) {                                          
        if (Y.kp) { v.push_back({(unsigned long long)Y.kp, 32 * tbcap_kb_, 4}); v.push_back({(unsigned long long)Y.kps, 0, 3}); } else { v.push_back({(unsigned long long)Y.k8, 0, 2}); v.push_back({(unsigned long long)Y.ks, 0, 3}); }
        if (Y.vp) { v.push_back({(unsigned long long)Y.vp, 32 * tbcap_kb_, 4}); v.push_back({(unsigned long long)Y.vps, 0, 3}); } else { v.push_back({(unsigned long long)Y.v8, 0, 2}); v.push_back({(unsigned long long)Y.vs, 0, 3}); }
      } else if (Y.k8) { v.push_back({(unsigned long long)Y.k8, 0, 2}); v.push_back({(unsigned long long)Y.v8, 0, 2});
                         v.push_back({(unsigned long long)Y.ks, 0, 3}); v.push_back({(unsigned long long)Y.vs, 0, 3}); }
      else { v.push_back({(unsigned long long)Y.kc, 0, 1}); v.push_back({(unsigned long long)Y.vc, 0, 1}); }
    }
    v.push_back({(unsigned long long)xn_, H_ * 2, 0}); v.push_back({(unsigned long long)d_next8_, 4, 0}); v.push_back({(unsigned long long)d_toks_, 4, 0});
    for (size_t k = 0; cap_ && k < cap_layers_.size(); ++k)            
      v.push_back({(unsigned long long)(cap_ + k * (size_t)Pmax_ * H_), H_ * 4, 0});
    { DevMemScope dm_scope(this); d_wkcopy_ = dz35<WkCopy>(v.size() * sizeof(WkCopy)); }
    cudaMemcpy(d_wkcopy_, v.data(), v.size() * sizeof(WkCopy), cudaMemcpyHostToDevice); n_wkcopy_ = (int)v.size();
  }
  launch_wk_promote(d_wkcopy_, n_wkcopy_, src, dst, pos_, max_ctx_, st_);
}
void Engine35::mtp_log_reset() { if (mtp_.on) cudaMemsetAsync(mtp_.lg_cnt, 0, 4, st_); }
int Engine35::mtp_log_read(int32_t* ids, float* vals, int max_steps) {
  if (!mtp_.on) return 0;
  cudaStreamSynchronize(st_); int n = 0; cudaMemcpy(&n, mtp_.lg_cnt, 4, cudaMemcpyDeviceToHost); n = std::min(std::min(n, 16), max_steps);
  if (n > 0) { cudaMemcpy(ids, mtp_.lg_idx, (size_t)n * 32, cudaMemcpyDeviceToHost); cudaMemcpy(vals, mtp_.lg_val, (size_t)n * 32, cudaMemcpyDeviceToHost); }
  return n;
}
void Engine35::set_opt(int key, int value) {
  if (kv_fp16_ && key == 9 && value == 0) throw std::runtime_error("FP16 KV requires ATX attention");
  rp_launch(); cudaStreamSynchronize(st2_); replay_fence();           
  if (key == 15) { opt_rp_async_ = value; return; }
  if (key == 26) { opt_rp_defer_ = value; return; }                   
  if (key == 27) { if (value != (int)l2pf2_mb_) drop_pass_graphs(); l2pf2_mb_ = value; return; }   
  if (key == 28) { if (value != opt_lo_) drop_pass_graphs(); opt_lo_ = value; return; }   
  if (key == 1) opt_replay_graph_ = value; else if (key == 2) opt_single_commit_ = value; else if (key == 3) opt_catchup_nosync_ = value;
  else if (key == 4) opt_mtp_log_ = value;
  else if (key == 6) opt_lfrow_ = value;
  else if (key == 8) { if (value != fa_) {                         
                         for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);
                         pgraph_.clear(); if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; } if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; }
                         for (auto& kv : mtp_.graphs) cudaGraphExecDestroy(kv.second); mtp_.graphs.clear(); }
                       fa_ = value; }
  else if (key == 9) { if (value != atx_) std::swap(gexec_, gexec_alt_);   
                       atx_ = value; }
  else if (key == 14) { if (value != opt_flow_) {                  
                         for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);
                         pgraph_.clear(); if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; } if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; } }
                       opt_flow_ = value; flow_prepare(); }
  else if (key == 7) { if (value != opt_fin_) {                     
                         for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);
                         pgraph_.clear(); if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; } if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; } }
                       opt_fin_ = value; }
  else if (key == 5) { if (value != opt_kconv_ && gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; }
                       if (value != opt_kconv_ && gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; }   
                       opt_kconv_ = value; }
  else if (key == 25) { if (value != l2pf_) drop_pass_graphs(); l2pf_ = value; }   
  else if (key == 24) { if (value && !bdr_.on) throw std::runtime_error("set_opt 24: HCAP needs the Bdraft 2 drafter");
                       if (value != bdr_.hb_on) drop_pass_graphs(); bdr_.hb_on = value; }   
  else if (key == 23) { if (value && !bdr_.nsub) throw std::runtime_error("set_opt 23: no drafter sub-head (YSKN_BDR_SUB)");
                       if (value != bdr_.sub_on) drop_pass_graphs(); bdr_.sub_on = value; }   
  else if (key == 30) { if (value && !bdr_.sk_a) throw std::runtime_error("set_opt 30: no SHEAD head (YSKN_BDR_SHEAD)");   
                       if (value && (value < 16 || value > BDR_SHEAD_CAP || value % 8)) throw std::runtime_error("set_opt 30: M must be 0 or 16..2048, a multiple of 8");
                       if (value != bdr_.sk_m) drop_pass_graphs(); bdr_.sk_m = value; }
  else if (key == 22) { if (value != atx_vsp_) drop_pass_graphs(); atx_vsp_ = value; }   
  else if (key == 21) { if (value != kz_) drop_pass_graphs(); kz_ = value; apply_kz(); }   
  else if (key == 29) { if (value < 0 || value > 1000) throw std::runtime_error("set_opt 29: min-p x 1000 must be 0..1000");   
                       smp_p_.min_p = value / 1000.0; cudaStreamSynchronize(st_); cudaMemcpy(d_smp_, &smp_p_, sizeof(Q35SmpParams), cudaMemcpyHostToDevice); }
  else if (key == 20) bdr_.cpl_pm = value;                            
  else if (key == 16) {                                                
    if (value != 2 && value != 3) throw std::runtime_error("set_opt: SMP version must be 2 or 3");
    smp_p_.version = value; cudaStreamSynchronize(st_); cudaMemcpy(d_smp_, &smp_p_, sizeof(Q35SmpParams), cudaMemcpyHostToDevice); }
  else if (key >= 10 && key <= 13) {                                  
    if (key == 10) { const int on = value > 0 ? 1 : 0;
                     if (on != smp_) { if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; } if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; } }
                     smp_ = on; if (on) smp_p_.temp = value / 1000.0; }
    else if (key == 11) { if (value <= 0 || value > 1000) throw std::runtime_error("set_opt: top-p x 1000 must be 1..1000"); smp_p_.top_p = value / 1000.0; }
    else if (key == 12) { if (value < 1 || value > Q35_SMP_K) throw std::runtime_error("set_opt: top-k must be 1..20"); smp_p_.top_k = value; }
    else smp_p_.seed = (unsigned long long)(unsigned)value;
    cudaStreamSynchronize(st_); cudaMemcpy(d_smp_, &smp_p_, sizeof(Q35SmpParams), cudaMemcpyHostToDevice);
  }
  else throw std::runtime_error("set_opt: unknown key " + std::to_string(key));
}

void Engine35::run_pass(int M, bool commit, int head_mode, bool topk, bool nll, bool use_graph) {
  replay_fence(); ++sh_epoch_;
  if (!use_graph) { pass_body(M, commit, nlayers_, head_mode, topk, nll, false); return; }
  const int key = M * 64 + (commit ? 32 : 0) + head_mode * 4 + (topk ? 2 : 0) + (nll ? 1 : 0) + (opt_kconv_ ? (1 << 30) : 0)
                  + (wk_pass(M, commit, head_mode) ? (1 << 29) : 0) + (atx_ ? (1 << 28) : 0) + (smp_ ? (1 << 27) : 0); auto it = pgraph_.find(key);
  if (it == pgraph_.end()) {
    cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); pass_body(M, commit, nlayers_, head_mode, topk, nll, false);
    cudaStreamEndCapture(st_, &g);
    if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("Engine35: graph instantiate failed");
    cudaGraphDestroy(g); it = pgraph_.emplace(key, ge).first;
  }
  cudaGraphLaunch(it->second, st_);
}
void Engine35::upload_rows(const int32_t* ids, int M, const int32_t* targets) {
  std::memcpy(h_pin_ + PIN_IDS, ids, (size_t)M * 4); cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, (size_t)M * 4, cudaMemcpyHostToDevice, st_);
  if (targets) { std::memcpy(h_pin_ + PIN_TG, targets, (size_t)M * 4); cudaMemcpyAsync(d_target8_, h_pin_ + PIN_TG, (size_t)M * 4, cudaMemcpyHostToDevice, st_); }
  launch_set_int(d_pos_, pos_, st_);
}
int Engine35::prefill(const int32_t* ids,int n) {
  if (n<1 || pos_+n>max_ctx_) throw std::runtime_error("prefill: context overflow");
  if(layer_rows_>0 && n>=1024 && !mtp_.on && !visual_on_ && xsm_ && !xsm_part_ && !kz_ && atx_ && !kskip_ && layer_prepare())
    return prefill_vsp(ids,n);
  return prefill_chunks(ids,n);
}





bool Engine35::layer_prepare() {
  
  
  if (layer_ready_) return !(bdr_.on && cap_ && !tc_dq_);
  if (layer_unavailable_ || !g6_.weights) return false;
  size_t extra=0;
  for (Q35Layer& Y:L_) {
    std::vector<DevTensor*> ts=Y.full ? std::vector<DevTensor*>{&Y.q,&Y.k,&Y.v,&Y.o,&Y.gate,&Y.up,&Y.down}
                                      : std::vector<DevTensor*>{&Y.qkv,&Y.z,&Y.out,&Y.gate,&Y.up,&Y.down};
    size_t sum=0,big=0;
    for (auto* t:ts) { const auto& d=t->dev;
      if (!d.lno || !d.lut8 || d.tile_k || d.group_worker || d.N%16 || d.Kc%512 || (d.k_default!=4 && d.k_default!=6 && d.k_default!=8)) {
        layer_unavailable_=true; return false;
      }
      size_t z=(size_t)d.N*d.Kc;sum+=z;big=std::max(big,z);
    }
    if (big>g6_.weight_bytes) {layer_unavailable_=true;return false;}
    extra=std::max(extra,sum-big);
  }
  const int ncap=bdr_.on && cap_ ? (int)cap_layers_.size() : 0;
  if(ncap && ncap!=5){layer_unavailable_=true;return false;}
  const size_t capbytes=(size_t)ncap*2048*(H_+(H_/512)*4);
  const size_t rowbytes=(size_t)H_*7+(H_/512)*4+Q35_SST*4+4;
  const size_t nbbytes=(size_t)PM_*(Q35_CONV*2+Q35_Z*8+2*Q35_GK*Q35_GD*4+Q35_GV*Q35_GD*4+2*Q35_GV*4);
  size_t free=0,total=0; if(cudaMemGetInfo(&free,&total)!=cudaSuccess) throw std::runtime_error("layer memory query failed");
  const size_t margin=(size_t)1<<30;
  int sc=std::max(PM_*2,layer_rows_);sc=((sc+PM_-1)/PM_)*PM_;
  bool overlap=ovl_!=0;size_t need=0;
  while (true) {
    need=extra+capbytes+rowbytes*sc+(overlap?nbbytes:0);
    if(free>margin && need<=free-margin) break;
    if(sc>PM_*2) {sc=std::max(PM_*2,(sc/2/PM_)*PM_);continue;}
    if(overlap) {overlap=false;continue;}
    layer_unavailable_=true;
    std::fprintf(stderr,"[q35_layer] capacity fallback; no context reduction\n");return false;
  }
  DevMemScope dm_scope(this);
  layer_weights_=dz35<uint4>(extra);layer_weight_bytes_=extra;
  tc_h_=dz35<float>((size_t)sc*H_*4);tc_xn_=dz35<uint16_t>((size_t)sc*H_*2);
  tc_qx_=dz35<int8_t>((size_t)sc*H_);tc_al_=dz35<float>((size_t)sc*(H_/512)*4);
  tc_S_=dz35<float>((size_t)sc*Q35_SST*4);tc_ids_=dz35<int>((size_t)sc*4);tc_rows_=sc;
  if(ncap){tc_dq_=dz35<int8_t>((size_t)ncap*2048*H_);tc_da_=dz35<float>((size_t)ncap*2048*(H_/512)*4);}
  ovl_=overlap?1:0;
  if(overlap) {
    int lo=0,hi=0;cudaDeviceGetStreamPriorityRange(&lo,&hi);cudaStreamCreateWithPriority(&nb_st_,cudaStreamNonBlocking,hi);
    for(int i=0;i<2;++i){cudaEventCreateWithFlags(&nb_e1_[i],cudaEventDisableTiming);cudaEventCreateWithFlags(&nb_e2_[i],cudaEventDisableTiming);}
    const size_t R=(size_t)PM_;
    nb_pmx_[0]=pmix_big_;nb_z_[0]=z_;nb_cq_[0]=gq_;nb_ck_[0]=gk_;nb_cv_[0]=gv_;nb_cb_[0]=gbeta_;nb_cg_[0]=gg_;nb_core_[0]=core_;
    nb_pmx_[1]=dz35<uint16_t>(R*Q35_CONV*2);nb_z_[1]=dz35<float>(R*Q35_Z*4);
    nb_cq_[1]=dz35<float>(R*Q35_GK*Q35_GD*4);nb_ck_[1]=dz35<float>(R*Q35_GK*Q35_GD*4);
    nb_cv_[1]=dz35<float>(R*Q35_Z*4);nb_cb_[1]=dz35<float>(R*Q35_GV*4);nb_cg_[1]=dz35<float>(R*Q35_GV*4);
    nb_core_[1]=dz35<float>(R*Q35_Z*4);nb_rows_=PM_;
  }
  layer_ready_=true;
  std::fprintf(stderr,"[q35_layer] rows=%d overlap=%d extra=%zu bytes\n",sc,ovl_,need);
  return true;
}
void Engine35::layer_decode(int l) {
  auto& Y=L_[l];
  std::vector<DevTensor*> ts=Y.full ? std::vector<DevTensor*>{&Y.q,&Y.k,&Y.v,&Y.o,&Y.gate,&Y.up,&Y.down}
                                    : std::vector<DevTensor*>{&Y.qkv,&Y.z,&Y.out,&Y.gate,&Y.up,&Y.down};
  auto largest=*std::max_element(ts.begin(),ts.end(),[](auto* a,auto* b){return (size_t)a->dev.N*a->dev.Kc<(size_t)b->dev.N*b->dev.Kc;});
  size_t off=0;g6_.cached.clear();
  for(auto* t:ts) {
    uint4* dst=t==largest?g6_.weights:reinterpret_cast<uint4*>(reinterpret_cast<char*>(layer_weights_)+off);
    if(t!=largest)off+=(size_t)t->dev.N*t->dev.Kc;
    if(off>layer_weight_bytes_)throw std::runtime_error("layer decoded storage overflow");
    q35_g6_decode(t->dev,dst,st_);g6_.cached[t->dev.lno]=dst;
  }
}
void Engine35::gdn_a(int l, int M, int b, const uint16_t* prev) {
  Q35Layer& Y = L_[l];
  Q35AbArgs AB{xn_, Y.wb, Y.wa, M, pab_big_};
  if (xsmc_) { AB.S = S_; AB.eps = eps_; }
  launch_q35_ab(AB, st_);
  gemv({&Y.qkv, &Y.z}, M, true);
  if (tq_qx_) { qx_ = scr_qx_; alpha_ = scr_al_; }                      
  { Q35MixArgs X{pt(Y.qkv, M), pt(Y.z, M), Y.qkv.tau, Y.z.tau, nwk(Y.qkv, M), nwk(Y.z, M), Y.qkv.signs_out, Y.z.signs_out, M, nb_pmx_[b], nb_z_[b]};
    if (xsmc_) X.xsm = Q35Xsm{us_, S_, eps_, nmode(M, true)};
    if (pf_on(1, M)) X.pf = pf_of({&Y.out}, -1.0);
    launch_q35_mix(X, st_); }
  const Q35ConvArgs CA{nb_pmx_[b], Y.conv, prev ? const_cast<uint16_t*>(prev) : Y.hist, pab_big_, Y.A_log, Y.dt_bias, M, M,
                       nb_cq_[b], nb_ck_[b], nb_cv_[b], nb_cb_[b], nb_cg_[b]};
  launch_q35_conv(CA, st_);
}

void Engine35::gdn_s(int l, int M, int b) {
  Q35Layer& Y = L_[l];
  const Q35StateArgs SA{nb_cq_[b], nb_ck_[b], nb_cv_[b], nb_cb_[b], nb_cg_[b], Y.S, M, M, nb_core_[b], nb_pmx_[b], Y.hist};
  launch_q35_state_nb(SA, nb_st_);
}

void Engine35::gdn_b(int l, int M, int b) {
  Q35Layer& Y = L_[l];
  if (tq_qx_) { qx_ = scr_qx_; alpha_ = scr_al_; }
  launch_q35_gdn_out(Q35GdnOutArgs{nb_core_[b], nb_z_[b], Y.gnorm, eps_, M, Y.out.signs_in, qx_, alpha_, lo_now_ ? 1 : 0}, st_);
  if (pf_on(4, M)) next_pf_ = pf_of({&Y.gate, &Y.up}, l2pf_mb_[0]);
  gemv_res(Y.out, M, Y.ln2, Y.gate.signs_in, nullptr);
  if (flow_ok(M)) { flow_mlp(Y, M, l); return; }
  mlp_block(Y, M, false);
  if (l + 1 < nlayers_ && pf_on(16, M)) { const Q35Layer& Nx = L_[l + 1]; next_pf_ = Nx.full ? pf_of({&Nx.q, &Nx.k}, l2pf_mb_[2]) : pf_of({&Nx.qkv, &Nx.z}, l2pf_mb_[2]); }
  if (l + 1 < nlayers_) { const Q35Layer& Nx = L_[l + 1]; gemv_res(Y.down, M, Nx.ln1, Nx.full ? Nx.q.signs_in : Nx.qkv.signs_in, Nx.full ? nullptr : xn_); }
  else gemv_res(Y.down, M, fnorm_, head_.signs_in, mtp_.on ? xn_ : nullptr);
}
int Engine35::prefill_vsp(const int32_t* ids, int n) {
  if (n < 1 || pos_ + n > max_ctx_) throw std::runtime_error("prefill_vsp: context overflow");
  replay_fence(); ++sh_epoch_;
  const int SC=tc_rows_, G0=H_/512;
  const int ncap=(bdr_.on && cap_) ? (int)cap_layers_.size() : 0;
  const int TW=ncap ? std::min(2048,n) : 0;
  float* const h_scr = h_; uint16_t* const xn_scr = xn_; float* const S_scr = S_; scr_qx_ = qx_; scr_al_ = alpha_;
  struct Restore { Engine35* e; float* h; uint16_t* xn; float* S; ~Restore() { e->h_ = h; e->xn_ = xn; e->S_ = S; e->qx_ = e->scr_qx_; e->alpha_ = e->scr_al_;
                                                                            e->tq_qx_ = nullptr; e->tq_al_ = nullptr; e->g6_.cached.clear(); } } restore{this, h_scr, xn_scr, S_scr};
  const int pos0 = pos_; int last_m = 0;
  for (int r0 = 0; r0 < n; r0 += SC) {
    const int nsc = std::min(SC, n - r0), nsub = (nsc + PM_ - 1) / PM_;
    const int per = std::min(PM_, ((nsc + nsub - 1) / nsub + 63) / 64 * 64);   
    cudaStreamSynchronize(st_);                                          
    if (cudaMemcpy(tc_ids_, ids + r0, (size_t)nsc * 4, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("prefill_vsp: id upload");
    const bool nb_go = ovl_ && nb_rows_ >= per && kskip_ == 0 && per > 64 && nsub >= 2 && pmix_big_;
    auto cap_rows = [&](int l, int s0, int m) {
      for (int k = 0; k < ncap; ++k) if (cap_layers_[k] == l) {        
        const int g0 = r0 + s0, a = std::max(g0, n - TW), b = std::min(g0 + m, n);
        if(b>a)launch_q35_capture_a8(h_+(size_t)(a-g0)*H_,k,a-(n-TW),ncap,bdr_.signs,tc_dq_,tc_da_,b-a,st_);
      } };
    for (int l = 0; l < nlayers_; ++l) {
      layer_decode(l);
      if (nb_go && !L_[l].full) {                                      
        const int ns = (nsc + per - 1) / per;
        auto sub_m = [&](int i) { return std::min(per, nsc - i * per); };
        auto views = [&](int i) {
          const int s0 = i * per, m = sub_m(i);
          launch_set_int(d_pos_, pos_ + s0, st_);
          g_pdl_rows = m; xsmc_ = true; wk_now_ = false; lo_now_ = false; in2_ = false; pf2_live_ = false;
          h_ = tc_h_ + (size_t)s0 * H_; xn_ = tc_xn_ + (size_t)s0 * H_; S_ = tc_S_ + (size_t)s0 * Q35_SST;
          tq_qx_ = tc_qx_ + (size_t)s0 * H_; tq_al_ = tc_al_ + (size_t)s0 * G0; qx_ = tq_qx_; alpha_ = tq_al_; };
        auto part_a = [&](int i) {
          const int s0 = i * per, m = sub_m(i);
          views(i);
          if (l == 0) { cudaMemcpyAsync(d_toks_, tc_ids_ + s0, (size_t)m * 4, cudaMemcpyDeviceToDevice, st_);
                        const Q35Layer& L0 = L_[0]; res_norm(nullptr, m, L0.ln1, L0.full ? L0.q.signs_in : L0.qkv.signs_in, L0.full ? nullptr : xn_); }
          gdn_a(l, m, i & 1, i > 0 ? nb_pmx_[(i - 1) & 1] + (size_t)(sub_m(i - 1) - 3) * Q35_CONV : nullptr);
          cudaEventRecord(nb_e1_[i & 1], st_); };
        part_a(0);
        for (int i = 0; i < ns; ++i) {
          const int s0 = i * per, m = sub_m(i);
          cudaStreamWaitEvent(nb_st_, nb_e1_[i & 1], 0); gdn_s(l, m, i & 1); cudaEventRecord(nb_e2_[i & 1], nb_st_);
          if (i + 1 < ns) part_a(i + 1);
          cudaStreamWaitEvent(st_, nb_e2_[i & 1], 0);
          views(i); gdn_b(l, m, i & 1);
          cap_rows(l, s0, m);
          last_m = m;
        }
        continue;
      }
      for (int s0 = 0; s0 < nsc; s0 += per) {
        const int m = std::min(per, nsc - s0);
        launch_set_int(d_pos_, pos_ + s0, st_);
        g_pdl_rows = m; xsmc_ = true; wk_now_ = false; lo_now_ = false; in2_ = false; pf2_live_ = false;
        h_ = tc_h_ + (size_t)s0 * H_; xn_ = tc_xn_ + (size_t)s0 * H_; S_ = tc_S_ + (size_t)s0 * Q35_SST;   
        tq_qx_ = tc_qx_ + (size_t)s0 * H_; tq_al_ = tc_al_ + (size_t)s0 * G0; qx_ = tq_qx_; alpha_ = tq_al_;
        if (l == 0) {                                                    
          cudaMemcpyAsync(d_toks_, tc_ids_ + s0, (size_t)m * 4, cudaMemcpyDeviceToDevice, st_);
          const Q35Layer& L0 = L_[0]; res_norm(nullptr, m, L0.ln1, L0.full ? L0.q.signs_in : L0.qkv.signs_in, L0.full ? nullptr : xn_);
        }
        layer(l, m, m);
        cap_rows(l, s0, m);
        last_m = m;
      }
    }
    g6_.cached.clear();
    if (r0 + nsc == n) { g_pdl_rows = 1; head_slice(last_m - 1, 1, false, false, false); }   
    xsmc_ = false;
    pos_ += nsc;
  }
  if(TW){dctx_q_=tc_dq_;dctx_a_=tc_da_;bdraft_ctx(0,TW,pos0+n-TW);dctx_q_=nullptr;dctx_a_=nullptr;}
  cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_ + last_m - 1, 4, cudaMemcpyDeviceToHost, st_);
  cudaStreamSynchronize(st_);
  const int next = h_pin_[PIN_AM];
  h_pin_[PIN_IDS] = next; cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, pos_, st_); cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_vsp: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
  return next;
}
void Engine35::nll_layer(const int32_t* ids, int n, float* out) {
  if (n < 2) return;
  const int r = n - 1;
  if (smp_ || !layer_prepare() || r>tc_rows_) throw std::runtime_error("nll_layer: needs one supported super-chunk and greedy heads");
  prefill_vsp(ids, r);
  const int G0 = H_ / 512;
  int8_t* const qx_s = qx_; float* const al_s = alpha_; float* const S_s = S_;
  xsmc_ = true; in2_ = false; lo_now_ = false; wk_now_ = false;
  for (int c = 0; c < r; c += HM_) {
    const int rows = std::min(HM_, r - c);
    g_pdl_rows = rows;
    qx_ = tc_qx_ + (size_t)c * H_; alpha_ = tc_al_ + (size_t)c * G0; S_ = tc_S_ + (size_t)c * Q35_SST;
    std::memcpy(h_pin_ + PIN_TG, ids + c + 1, (size_t)rows * 4); cudaMemcpyAsync(d_target8_, h_pin_ + PIN_TG, (size_t)rows * 4, cudaMemcpyHostToDevice, st_);
    head_slice(0, rows, false, false, true);
    cudaMemcpyAsync(h_pinf_ + PINF_NLL, d_nll8_, (size_t)rows * 4, cudaMemcpyDeviceToHost, st_);
    cudaStreamSynchronize(st_);
    std::memcpy(out + c, h_pinf_ + PINF_NLL, (size_t)rows * 4);
  }
  qx_ = qx_s; alpha_ = al_s; S_ = S_s; xsmc_ = false; g_pdl_rows = 1;
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("nll_layer: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
}

int Engine35::prefill_chunks(const int32_t* ids, int n) {
  if (n < 1 || pos_ + n > max_ctx_) throw std::runtime_error("prefill: context overflow");
  for (int done = 0; done < n;) {
    const int M = std::min(PM_, n - done); const bool last = done + M == n;
    upload_rows(ids + done, M, nullptr);
    run_pass(M, true, last ? HEAD_LAST : HEAD_NONE, false, false, true);
    if (mtp_.on) mtp_catchup(M);
    if (bdr_.on) bdraft_ctx(0, M, pos_);                               
    if (last) cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_ + M - 1, 4, cudaMemcpyDeviceToHost, st_);
    cudaStreamSynchronize(st_);                                          
    pos_ += M; done += M;
  }
  const int next = h_pin_[PIN_AM];
  h_pin_[PIN_IDS] = next; cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, pos_, st_); cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
  return next;
}
int Engine35::prefill_rows_k20(const int32_t* ids, int n, int32_t* out_argmax, int32_t* k20_ids, float* k20_vals) {
  if (!smp_ || !q35_smp2_on()) throw std::runtime_error("prefill_rows_k20: needs SMP on (set_sampling T > 0) and SMP 2");
  if (!d_k20v_) { d_k20v_ = dz35<float>((size_t)Pmax_ * Q35_SMP_K * 4); d_k20i_ = dz35<int>((size_t)Pmax_ * Q35_SMP_K * 4); }
  drop_pass_graphs(); k20_on_ = true;                                    
  struct Off { Engine35* e; ~Off() { e->drop_pass_graphs(); e->k20_on_ = false; } } off{this};
  replay_fence();
  if (n < 1 || pos_ + n > max_ctx_) throw std::runtime_error("prefill_rows_k20: context overflow");
  int next = 0;
  for (int done = 0; done < n;) {
    const int M = std::min(PM_, n - done);
    upload_rows(ids + done, M, nullptr);
    run_pass(M, true, HEAD_ALL, true, false, true);
    if (mtp_.on) mtp_catchup(M);
    if (bdr_.on) bdraft_ctx(0, M, pos_);
    cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
    cudaStreamSynchronize(st_);
    if (out_argmax) std::memcpy(out_argmax + done, h_pin_ + PIN_AM, (size_t)M * 4);
    cudaMemcpy(k20_ids + (size_t)done * Q35_SMP_K, d_k20i_, (size_t)M * Q35_SMP_K * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(k20_vals + (size_t)done * Q35_SMP_K, d_k20v_, (size_t)M * Q35_SMP_K * 4, cudaMemcpyDeviceToHost);
    next = h_pin_[PIN_AM + M - 1]; pos_ += M; done += M;
  }
  h_pin_[PIN_IDS] = next; cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, pos_, st_); cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_rows_k20: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
  return next;
}
int Engine35::prefill_rows(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals) {
  replay_fence();
  if (n < 1 || pos_ + n > max_ctx_) throw std::runtime_error("prefill_rows: context overflow");
  int next = 0;
  for (int done = 0; done < n;) {
    const int M = std::min(PM_, n - done);
    upload_rows(ids + done, M, nullptr);
    run_pass(M, true, HEAD_ALL, true, false, true);
    if (mtp_.on) mtp_catchup(M);
    if (bdr_.on) bdraft_ctx(0, M, pos_);
    cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
    cudaMemcpyAsync(h_pin_ + PIN_TI, top_idx_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
    cudaMemcpyAsync(h_pinf_ + PINF_TV, top_val_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
    cudaStreamSynchronize(st_);
    if (out_argmax) std::memcpy(out_argmax + done, h_pin_ + PIN_AM, (size_t)M * 4);
    if (top_ids) std::memcpy(top_ids + (size_t)done * 8, h_pin_ + PIN_TI, (size_t)M * 32);
    if (top_vals) std::memcpy(top_vals + (size_t)done * 8, h_pinf_ + PINF_TV, (size_t)M * 32);
    next = h_pin_[PIN_AM + M - 1]; pos_ += M; done += M;
  }
  h_pin_[PIN_IDS] = next; cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, pos_, st_); cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_rows: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
  return next;
}
void Engine35::decode(int32_t* out, int steps) {
  replay_fence(); ++sh_epoch_;
  if (!gexec_) {
    cudaGraph_t g; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); pass_body(1, true, nlayers_, HEAD_LAST, false, false, true); cudaStreamEndCapture(st_, &g);
    if (cudaGraphInstantiate(&gexec_, g, 0) != cudaSuccess) throw std::runtime_error("decode: graph instantiate failed");
    cudaGraphDestroy(g);
  }
  launch_set_int(d_pos_, pos_, st_);
  for (int i = 0; i < steps; ++i) {
    if (pos_ + 1 > max_ctx_) throw std::runtime_error("decode: context overflow");
    cudaGraphLaunch(gexec_, st_); pos_++;
    cudaMemcpyAsync(h_dec_ + i, d_next8_, 4, cudaMemcpyDeviceToHost, st_);   
  }
  cudaStreamSynchronize(st_); std::memcpy(out, h_dec_, (size_t)steps * 4);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("decode: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
}
void Engine35::nll(const int32_t* ids, int n, float* out) {
  replay_fence();
  for (int i = 0; i + 1 < n;) {
    const int M = std::min(PM_, n - 1 - i);
    upload_rows(ids + i, M, ids + i + 1);
    run_pass(M, true, HEAD_ALL, false, true, true);
    cudaMemcpyAsync(h_pinf_ + PINF_NLL, d_nll8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
    cudaStreamSynchronize(st_);
    std::memcpy(out + i, h_pinf_ + PINF_NLL, (size_t)M * 4);
    pos_ += M; i += M;
  }
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("nll: ") + cudaGetErrorString(e));
  last_M_ = 0; pre_committed_ = false;
}
void Engine35::verify(const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals, const int32_t* targets, float* out_nll) {
  if (M < 1 || M > HM_) throw std::runtime_error("verify: M must be 1..16");
  if (pos_ + M > max_ctx_) throw std::runtime_error("verify: context overflow");
  upload_rows(ids, M, targets);
  const bool c1 = M == 1 && (opt_single_commit_ == 2 || (opt_single_commit_ == 1 && specx_scope_));   
  if (wk_pass(M, c1, HEAD_ALL) && !wk_set_) wk_plain_table();          
  run_pass(M, c1, HEAD_ALL, true, targets != nullptr, use_graph != 0); pre_committed_ = c1; wk_set_ = false;
  if (mtp_.on) cudaMemcpyAsync(mtp_.vtoks, d_toks_, (size_t)M * 4, cudaMemcpyDeviceToDevice, st_);
  if (targets) cudaMemcpyAsync(h_pinf_ + PINF_NLL, d_nll8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(h_pin_ + PIN_TI, top_idx_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(h_pinf_ + PINF_TV, top_val_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("verify: ") + cudaGetErrorString(e));
  if (out_argmax) std::memcpy(out_argmax, h_pin_ + PIN_AM, (size_t)M * 4);
  if (top_ids) std::memcpy(top_ids, h_pin_ + PIN_TI, (size_t)M * 32);
  if (top_vals) std::memcpy(top_vals, h_pinf_ + PINF_TV, (size_t)M * 32);
  if (out_nll) std::memcpy(out_nll, h_pinf_ + PINF_NLL, (size_t)M * 4);
  last_M_ = M;
}
void Engine35::commit(int n_accept) {
  ++sh_epoch_;
  if (n_accept < 0 || n_accept >= last_M_) throw std::runtime_error("commit: n_accept out of range of the last verify pass");
  if (pre_committed_) { if (n_accept != 0) throw std::runtime_error("commit: the 1-row verify pass committed already"); }
  else replay_commit(n_accept + 1);                                     
  pre_committed_ = false;
  pos_ += n_accept + 1;
  if (opt_rp_defer_) { SetInts S; S.set(d_pos_, pos_); S.copy(d_toks_, d_next8_ + n_accept); launch_set_ints(S, st_); }   
  else { launch_set_int(d_pos_, pos_, st_);
  cudaMemcpyAsync(d_toks_, d_next8_ + n_accept, 4, cudaMemcpyDeviceToDevice, st_); }  
  last_M_ = 0;
}
void Engine35::flow_trace_read(float* out, int n) { cudaStreamSynchronize(st_); if (flow_trace_) cudaMemcpy(out, flow_trace_, (size_t)n * 8, cudaMemcpyDeviceToHost); }
void Engine35::hidden_rows(int n, float* out) { cudaStreamSynchronize(st_); cudaMemcpy(out, h_, (size_t)n * H_ * 4, cudaMemcpyDeviceToHost); }
void Engine35::logits_rows(int row0, int n, float* out) {
  if (row0 < 0 || row0 + n > HM_) throw std::runtime_error("logits_rows: the logits buffer holds the last head slice (16 rows)");
  cudaStreamSynchronize(st_); cudaMemcpy(out, logits_ + (size_t)row0 * V_, (size_t)n * V_ * 4, cudaMemcpyDeviceToHost); }
void Engine35::logits(float* out) { cudaStreamSynchronize(st_); cudaMemcpy(out, logits_, (size_t)V_ * 4, cudaMemcpyDeviceToHost); }
float Engine35::time_verify(int M, int mask, int iters) {
  replay_fence();
  if (M < 1 || M > Pmax_ || pos_ + M > max_ctx_) throw std::runtime_error("time_verify: bad M");
  for (int i = 0; i < M; ++i) h_pin_[PIN_IDS + i] = 0;
  cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, (size_t)M * 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, pos_, st_);
  const int hm = M <= HM_ ? HEAD_ALL : HEAD_LAST; const bool topk = (mask & 1) != 0 && M <= HM_;
  kskip_ = mask & ~15; const int key = M * 64 + hm * 4 + (topk ? 2 : 0) + 100000 * (kskip_ >> 4) + (opt_kconv_ ? (1 << 30) : 0)
                                       + (wk_pass(M, false, hm) ? (1 << 29) : 0) + (atx_ ? (1 << 28) : 0);
  if (kskip_ && !pgraph_.count(key)) {                                  
    cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); pass_body(M, false, nlayers_, hm, topk, false, false);
    cudaStreamEndCapture(st_, &g); cudaGraphInstantiate(&ge, g, 0); cudaGraphDestroy(g); pgraph_.emplace(key, ge); }
  kskip_ = 0;
  auto run = [&]() { if (mask & ~15) cudaGraphLaunch(pgraph_.at(key), st_); else run_pass(M, false, hm, topk, false, true); };
  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st_); for (int i = 0; i < iters; ++i) run(); cudaEventRecord(b, st_); cudaEventSynchronize(b);
  float ms = 0.f; cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("time_verify: ") + cudaGetErrorString(e));
  return ms / (float)iters;
}

void Engine35::load_draft(const std::string& path, const std::string&) {
  DevMemScope dm_scope(this);
  Q35Mtp& T = mtp_; if (T.on) throw std::runtime_error("load_draft: an MTP draft is already loaded");
  T.c.open(path); const Container* C = &T.c;
  if (T.c.model.arch != "qwen3_5_mtp" || T.c.model.hidden != H_) throw std::runtime_error("load_draft: not a qwen3_5 MTP container");
  T.fc = up("mtp.fc", 0, C);
  T.L.full = true;
  T.L.q = up("mtp.self_attn.q_proj", 0, C); T.L.k = up("mtp.self_attn.k_proj", 0, C); T.L.v = up("mtp.self_attn.v_proj", 0, C); T.L.o = up("mtp.self_attn.o_proj", 0, C);
  T.L.gate = up("mtp.mlp.gate_proj", 0, C); T.L.up = up("mtp.mlp.up_proj", 0, C); T.L.down = up("mtp.mlp.down_proj", 0, C);
  T.head = up("mtp.head", C->tensor("mtp.head").cols / 512, C); T.nsub = T.head.dev.N;
  T.L.qn = vec("mtp.self_attn.q_norm", Q35_HD, C); T.L.kn = vec("mtp.self_attn.k_norm", Q35_HD, C);
  T.L.ln1 = vec("mtp.input_layernorm", H_, C); T.L.ln2 = vec("mtp.post_attention_layernorm", H_, C);
  T.en = vec("mtp.pre_fc_norm_embedding", H_, C); T.hn = vec("mtp.pre_fc_norm_hidden", H_, C); T.norm = vec("mtp.norm", H_, C);
  { const VectorDesc& v = C->vector("mtp.head_ids"); if (v.dtype != "i32" || v.n != T.nsub) throw std::runtime_error("load_draft: mtp.head_ids");
    cudaMalloc(&T.ids, (size_t)T.nsub * 4); cudaMemcpy(T.ids, C->at<int32_t>(v.off), (size_t)T.nsub * 4, cudaMemcpyHostToDevice); }
  if (kv8_) { T.L.k8 = dz35<int8_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD); T.L.v8 = dz35<int8_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD);
              T.L.ks = dz35<float>((size_t)Q35_KVH * max_ctx_ * 4); T.L.vs = dz35<float>((size_t)Q35_KVH * max_ctx_ * 4); }
  else { T.L.kc = dz35<uint16_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD * 2); T.L.vc = dz35<uint16_t>((size_t)Q35_KVH * max_ctx_ * Q35_HD * 2); }
  auto sz = [&](const DevTensor& t, int rows) { return std::max((size_t)t.dev.n_workers * rows, (size_t)Pmax_) * t.dev.N; };
  ensure_part(std::max({sz(T.fc, Mmax_), sz(T.L.q, Mmax_) + sz(T.L.k, Mmax_) + sz(T.L.v, Mmax_), sz(T.L.o, Mmax_), sz(T.L.gate, Mmax_) + sz(T.L.up, Mmax_), sz(T.L.down, Mmax_)}));
  float* tp = part_;                                                    
  if (xsm_) { const size_t nd = std::max({sz(T.fc, Mmax_), sz(T.L.q, Mmax_) + sz(T.L.k, Mmax_) + sz(T.L.v, Mmax_), sz(T.L.o, Mmax_), sz(T.L.gate, Mmax_) + sz(T.L.up, Mmax_), sz(T.L.down, Mmax_)});
              mtp_part_ = dz35<float>(nd * 4); tp = mtp_part_; }
  T.fc.part = tp; T.L.q.part = tp; T.L.k.part = tp + sz(T.L.q, Mmax_); T.L.v.part = tp + sz(T.L.q, Mmax_) + sz(T.L.k, Mmax_); T.L.o.part = tp;
  T.L.gate.part = tp; T.L.up.part = tp + sz(T.L.gate, Mmax_); T.L.down.part = tp;
  T.head.part = dz35<float>((size_t)T.head.dev.n_workers * T.head.dev.N * 4);
  T.hin = dz35<uint16_t>((size_t)Pmax_ * H_ * 2); T.hout = dz35<uint16_t>((size_t)Pmax_ * H_ * 2); T.pend = dz35<uint16_t>((size_t)H_ * 2);
  T.toks = dz35<int>((size_t)Pmax_ * 4); T.d_pos = dz35<int>(4); T.idx = dz35<int>(4); T.out = dz35<int>(4); T.vtoks = dz35<int>((size_t)HM_ * 4);
  T.iota = dz35<int>((size_t)Pmax_ * 4); launch_q35_iota(T.iota, Pmax_, st_);
  T.tkv = dz35<float>((size_t)256 * 8 * 4); T.tki = dz35<int>((size_t)256 * 8 * 4); T.top_v = dz35<float>(8 * 4); T.top_i = dz35<int>(8 * 4);
  T.tkc = dz35<unsigned>(4); T.lg_idx = dz35<int>(16 * 8 * 4); T.lg_val = dz35<float>(16 * 8 * 4); T.lg_cnt = dz35<int>(4);
  T.hist = dz35<uint16_t>((size_t)max_ctx_ * H_ * 2); T.ctoks = dz35<int>((size_t)max_ctx_ * 4); cudaMallocHost(&T.h_ctoks, (size_t)max_ctx_ * 4);
  for (auto& kv : pgraph_) cudaGraphExecDestroy(kv.second);            
  pgraph_.clear(); if (gexec_) { cudaGraphExecDestroy(gexec_); gexec_ = nullptr; } if (gexec_alt_) { cudaGraphExecDestroy(gexec_alt_); gexec_alt_ = nullptr; }
  T.on = true; cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("load_draft: ") + cudaGetErrorString(e));
  std::fprintf(stderr, "[qwen3_5] MTP draft loaded: sub-vocabulary %d, LNORD codes now %.2f GB\n", T.nsub, lno_bytes_ / 1e9);
}
void Engine35::mtp_pass(int M, bool head) {
  Q35Mtp& T = mtp_; g_pdl_rows = M;
  
  Q35ResNormArgs A{}; A.M = M; A.N = H_; A.h = h_; A.eps = eps_; A.qx = qx_; A.alpha = alpha_; A.cnt = cnt_; A.keep_h = 1;
  A.qx_stride = 2 * H_; A.al_stride = 2 * H_ / 512;
  emb_src(T.toks, M, A.emb, A.ids); A.w = T.en; A.sg_in = T.fc.signs_in; launch_q35_res_norm(A, st_);
  A.emb = T.hin; A.ids = T.iota; A.w = T.hn; A.sg_in = T.fc.signs_in + H_; A.qx = qx_ + H_; A.alpha = alpha_ + H_ / 512; launch_q35_res_norm(A, st_);
  cudaMemsetAsync(h_, 0, (size_t)M * H_ * 4, st_);
  gemv({&T.fc}, M);
  res_norm(&T.fc, M, T.L.ln1, T.L.q.signs_in, nullptr);                 
  attn_block(T.L, M, T.d_pos, mtp_pos_slices_);
  mlp_block(T.L, M);
  res_norm(&T.L.down, M, T.norm, T.head.signs_in, T.hout);             
  if (head) {
    g_pdl_rows = 1;
    GemvLaunch P{}; P.M = 1; P.seg[0] = GemvSeg{T.head.dev, T.head.part, 0}; P.n_seg = 1; P.n_ctas = (T.nsub / 64 + GEMV_WARPS - 1) / GEMV_WARPS;
    P.max_workers = T.head.dev.n_workers; P.qx = qx_ + (size_t)(M - 1) * H_; P.alpha = alpha_ + (size_t)(M - 1) * (H_ / 512);
    launch_lnord(P, st_);
    ArgmaxArgs AA{T.head.part, T.head.tau, T.nsub, logits_, T.idx, blk_val_, blk_idx_, T.head.dev.n_workers, 1, 0, 1}; AA.cnt = am_cnt_;
    launch_argmax(AA, st_);
    launch_q35_pick(T.idx, T.ids, T.out, T.toks, st_);                 
    if (opt_mtp_log_) {                                               
      TopkArgs TA{logits_, T.nsub, T.tkv, T.tki, T.top_v, T.top_i, 1}; TA.cnt = T.tkc; launch_topk8(TA, st_);
      launch_q35_mtp_log(T.top_i, T.top_v, T.ids, T.lg_idx, T.lg_val, T.lg_cnt, st_);
    }
  }
}
void Engine35::run_mtp(int M, bool head, bool use_graph) {
  if (!use_graph) { mtp_pass(M, head); return; }
  const int key = M * 2 + (head ? 1 : 0) + (opt_mtp_log_ ? 1000000 : 0) + (atx_ ? 2000000 : 0); auto it = mtp_.graphs.find(key);
  if (it == mtp_.graphs.end()) {
    cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); mtp_pass(M, head); cudaStreamEndCapture(st_, &g);
    if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("Engine35: MTP graph instantiate failed");
    cudaGraphDestroy(g); it = mtp_.graphs.emplace(key, ge).first;
  }
  cudaGraphLaunch(it->second, st_);
}


void Engine35::mtp_catchup(int M) {
  Q35Mtp& T = mtp_; const size_t row = (size_t)H_ * 2;
  cudaMemcpyAsync(T.toks, d_toks_, (size_t)M * 4, cudaMemcpyDeviceToDevice, st_);
  cudaMemcpyAsync(T.hin, T.pend, row, cudaMemcpyDeviceToDevice, st_);
  if (M > 1) cudaMemcpyAsync(T.hin + H_, xn_, (M - 1) * row, cudaMemcpyDeviceToDevice, st_);
  cudaMemcpyAsync(T.pend, xn_ + (size_t)(M - 1) * H_, row, cudaMemcpyDeviceToDevice, st_);
  cudaMemcpyAsync(T.hist + (size_t)pos_ * H_, xn_, M * row, cudaMemcpyDeviceToDevice, st_);   
  launch_set_int(T.d_pos, pos_, st_);
  run_mtp(M, false, true);
}
void Engine35::draft_round(int mode, int a, int steps, int use_graph) {
  Q35Mtp& T = mtp_; if (!T.on) throw std::runtime_error("draft_round: no MTP draft loaded");
  const size_t row = (size_t)H_ * 2; int rows, p0;
  if (mode == 0) {                                                      
    rows = 1; p0 = pos_;
    cudaMemcpyAsync(T.toks, d_toks_, 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(T.hin, T.pend, row, cudaMemcpyDeviceToDevice, st_);
  } else {                                                              
    if (a < 0 || a + 2 > Mmax_ || pos_ - a - 1 < 0) throw std::runtime_error("draft_round: bad accepted count");
    rows = a + 2; p0 = pos_ - a - 1;
    cudaMemcpyAsync(T.toks, T.vtoks, (size_t)(a + 1) * 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(T.toks + a + 1, d_toks_, 4, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(T.hin, T.pend, row, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(T.hin + H_, xn_, (size_t)(a + 1) * row, cudaMemcpyDeviceToDevice, st_);
    cudaMemcpyAsync(T.pend, xn_ + (size_t)a * H_, row, cudaMemcpyDeviceToDevice, st_);
  }
  if (pos_ + steps > max_ctx_) steps = std::max(0, max_ctx_ - pos_);
  launch_set_int(T.d_pos, p0, st_);
  run_mtp(rows, steps > 0, use_graph != 0);                             
  if (steps > 0) cudaMemcpyAsync(d_toks_ + 1, T.out, 4, cudaMemcpyDeviceToDevice, st_);
  int last = rows;
  for (int k = 2; k <= steps; ++k) {                                    
    cudaMemcpyAsync(T.hin, T.hout + (size_t)(last - 1) * H_, row, cudaMemcpyDeviceToDevice, st_);
    launch_set_int(T.d_pos, pos_ + k - 1, st_);
    run_mtp(1, true, use_graph != 0); last = 1;
    cudaMemcpyAsync(d_toks_ + k, T.out, 4, cudaMemcpyDeviceToDevice, st_);
  }
}
void Engine35::hist_store(int pos0, int n) {
  Q35Mtp& T = mtp_; if (!T.on || n <= 0) return;
  cudaMemcpyAsync(T.hist + (size_t)pos0 * H_, xn_, (size_t)n * H_ * 2, cudaMemcpyDeviceToDevice, st_);
}
void Engine35::draft_catchup(int p0, int n, const int32_t* toks, int use_graph) {
  Q35Mtp& T = mtp_; if (!T.on) throw std::runtime_error("draft_catchup: no MTP draft loaded");
  if (n <= 0) return;
  if (p0 < 0 || p0 + n > max_ctx_) throw std::runtime_error("draft_catchup: positions out of range");
  int base = 0;
  if (opt_catchup_nosync_) {                                           
    base = p0;                                                         
    std::memcpy(T.h_ctoks + p0, toks, (size_t)n * 4); cudaMemcpyAsync(T.ctoks + p0, T.h_ctoks + p0, (size_t)n * 4, cudaMemcpyHostToDevice, st_);
  } else {
    cudaStreamSynchronize(st_);                                        
    std::memcpy(T.h_ctoks, toks, (size_t)n * 4); cudaMemcpyAsync(T.ctoks, T.h_ctoks, (size_t)n * 4, cudaMemcpyHostToDevice, st_);
  }
  const size_t row = (size_t)H_ * 2;
  for (int done = 0; done < n;) {
    const int m = std::min(Pmax_, n - done), q0 = p0 + done; const bool last = done + m == n;
    cudaMemcpyAsync(T.toks, T.ctoks + base + done, (size_t)m * 4, cudaMemcpyDeviceToDevice, st_);
    if (q0 == 0) { cudaMemsetAsync(T.hin, 0, row, st_); if (m > 1) cudaMemcpyAsync(T.hin + H_, T.hist, (m - 1) * row, cudaMemcpyDeviceToDevice, st_); }
    else cudaMemcpyAsync(T.hin, T.hist + (size_t)(q0 - 1) * H_, m * row, cudaMemcpyDeviceToDevice, st_);
    launch_set_int(T.d_pos, q0, st_);
    run_mtp(m, last, use_graph != 0);
    T.last_rows = m; done += m;
  }
  cudaMemcpyAsync(d_toks_ + 1, T.out, 4, cudaMemcpyDeviceToDevice, st_);   
}
void Engine35::draft_steps(int steps, int use_graph) {
  Q35Mtp& T = mtp_; const size_t row = (size_t)H_ * 2;
  if (pos_ + steps > max_ctx_) steps = std::max(0, max_ctx_ - pos_);
  for (int k = 2; k <= steps; ++k) {
    cudaMemcpyAsync(T.hin, T.hout + (size_t)(T.last_rows - 1) * H_, row, cudaMemcpyDeviceToDevice, st_);
    launch_set_int(T.d_pos, pos_ + k - 1, st_);
    run_mtp(1, true, use_graph != 0); T.last_rows = 1;
    cudaMemcpyAsync(d_toks_ + k, T.out, 4, cudaMemcpyDeviceToDevice, st_);
  }
}
void Engine35::set_verify_tokens(const int32_t* ids, int first_row, int n) {
  if (first_row < 0 || n < 0 || first_row + n > HM_) throw std::runtime_error("set_verify_tokens: rows out of range");
  int32_t* stg = h_pin_ + PIN_VT + 64 * (vt_slot_++ & 3);                  
  std::memcpy(stg, ids, (size_t)n * 4); cudaMemcpyAsync(d_toks_ + first_row, stg, (size_t)n * 4, cudaMemcpyHostToDevice, st_);
}
int32_t Engine35::get_verify_token(int row) {
  cudaMemcpyAsync(h_pin_ + PIN_GT, d_toks_ + row, 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_); return h_pin_[PIN_GT];
}
void Engine35::verify_device(int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals) {
  if (M < 1 || M > HM_) throw std::runtime_error("verify_device: M must be 1..16");
  if (pos_ + M > max_ctx_) throw std::runtime_error("verify_device: context overflow");
  launch_set_int(d_pos_, pos_, st_);
  const bool c1 = M == 1 && (opt_single_commit_ == 2 || (opt_single_commit_ == 1 && specx_scope_));   
  if (wk_pass(M, c1, HEAD_ALL) && !wk_set_) wk_plain_table();          
  run_pass(M, c1, HEAD_ALL, true, false, use_graph != 0); pre_committed_ = c1; wk_set_ = false;
  if (mtp_.on) cudaMemcpyAsync(mtp_.vtoks, d_toks_, (size_t)M * 4, cudaMemcpyDeviceToDevice, st_);
  cudaMemcpyAsync(h_pin_ + PIN_AM, d_next8_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  cudaMemcpyAsync(h_pin_ + PIN_IDS, d_toks_, (size_t)M * 4, cudaMemcpyDeviceToHost, st_);
  if (top_ids) cudaMemcpyAsync(h_pin_ + PIN_TI, top_idx_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
  if (top_vals) cudaMemcpyAsync(h_pinf_ + PINF_TV, top_val_, (size_t)M * 32, cudaMemcpyDeviceToHost, st_);
  if (bdr_.on && bdr_.hb_on) cudaMemcpyAsync(h_pin_ + PIN_HB, bdr_.hb_cfg, 8 * 4, cudaMemcpyDeviceToHost, st_);   
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("verify_device: ") + cudaGetErrorString(e));
  if (out_argmax) std::memcpy(out_argmax, h_pin_ + PIN_AM, (size_t)M * 4);
  if (out_toks) std::memcpy(out_toks, h_pin_ + PIN_IDS, (size_t)M * 4);
  if (top_ids) std::memcpy(top_ids, h_pin_ + PIN_TI, (size_t)M * 32);
  if (top_vals) std::memcpy(top_vals, h_pinf_ + PINF_TV, (size_t)M * 32);
  last_M_ = M;
}
static uint64_t fnv_words(uint64_t h, const std::vector<uint8_t>& b) {
  const size_t n8 = b.size() / 8; const uint64_t* w = reinterpret_cast<const uint64_t*>(b.data());
  for (size_t i = 0; i < n8; ++i) { h ^= w[i]; h *= 0x100000001b3ull; }
  for (size_t i = n8 * 8; i < b.size(); ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
  return h;
}




void Engine35::snap_save(const int32_t* ids, int n) {
  if (n != pos_ || n < 1) throw std::runtime_error("snap_save: the ids must cover positions 0..pos-1");
  replay_fence();
  constexpr size_t SB = (size_t)Q35_GV * Q35_GD * Q35_GD, HB = (size_t)3 * Q35_CONV, RB = (size_t)8 * 2048 * 128;
  int ng = 0; for (const Q35Layer& Y : L_) if (!Y.full) ++ng;
  if (!sh_S_) { DevMemScope dm_scope(this);                          
    auto pin = [](auto** p, size_t bytes) { if (cudaMallocHost(p, bytes) != cudaSuccess) throw std::runtime_error("snap_save: out of pinned memory"); };
    pin(&sh_S_, ng * SB * 4); pin(&sh_hist_, ng * HB * 2);
    if (bdr_.on) pin(&sh_ring_, 10 * RB * 2);
    if (mtp_.on) pin(&sh_pend_, (size_t)H_ * 2); }
  if ((bdr_.on && !sh_ring_) || (mtp_.on && !sh_pend_)) throw std::runtime_error("snap_save: drafter loaded after the first snapshot");
  int g = 0;
  for (const Q35Layer& Y : L_) if (!Y.full) {
    cudaMemcpyAsync(sh_S_ + g * SB, Y.S, SB * 4, cudaMemcpyDeviceToHost, st_); cudaMemcpyAsync(sh_hist_ + g * HB, Y.hist, HB * 2, cudaMemcpyDeviceToHost, st_); ++g; }
  if (bdr_.on) for (int i = 0; i < 5; ++i) { cudaMemcpyAsync(sh_ring_ + (2 * i) * RB, bdr_.rk[i], RB * 2, cudaMemcpyDeviceToHost, st_);
                                            cudaMemcpyAsync(sh_ring_ + (2 * i + 1) * RB, bdr_.rv[i], RB * 2, cudaMemcpyDeviceToHost, st_); }
  if (mtp_.on) cudaMemcpyAsync(sh_pend_, mtp_.pend, (size_t)H_ * 2, cudaMemcpyDeviceToHost, st_);
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("snap_save: ") + cudaGetErrorString(e));
  sh_pos_ = n; sh_ids_.assign(ids, ids + n);
}
void Engine35::snap_load() {
  if (sh_pos_ < 1 || !sh_S_) throw std::runtime_error("snap_load: no snapshot");
  replay_fence(); ++sh_epoch_;
  constexpr size_t SB = (size_t)Q35_GV * Q35_GD * Q35_GD, HB = (size_t)3 * Q35_CONV, RB = (size_t)8 * 2048 * 128;
  int g = 0;
  for (Q35Layer& Y : L_) if (!Y.full) {
    cudaMemcpyAsync(Y.S, sh_S_ + g * SB, SB * 4, cudaMemcpyHostToDevice, st_); cudaMemcpyAsync(Y.hist, sh_hist_ + g * HB, HB * 2, cudaMemcpyHostToDevice, st_); ++g; }
  if (bdr_.on && sh_ring_) for (int i = 0; i < 5; ++i) { cudaMemcpyAsync(bdr_.rk[i], sh_ring_ + (2 * i) * RB, RB * 2, cudaMemcpyHostToDevice, st_);
                                                       cudaMemcpyAsync(bdr_.rv[i], sh_ring_ + (2 * i + 1) * RB, RB * 2, cudaMemcpyHostToDevice, st_); }
  if (mtp_.on && sh_pend_) cudaMemcpyAsync(mtp_.pend, sh_pend_, (size_t)H_ * 2, cudaMemcpyHostToDevice, st_);
  pos_ = sh_pos_; last_M_ = 0; pre_committed_ = false; launch_set_int(d_pos_, pos_, st_);
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("snap_load: ") + cudaGetErrorString(e));
}
int Engine35::state_hash(uint64_t* out, int max_out) {
  replay_fence(); cudaStreamSynchronize(st_);
  std::vector<uint8_t> b; int n = 0; const uint64_t H0 = 0xcbf29ce484222325ull;
  auto put = [&](uint64_t v) { if (n >= max_out) throw std::runtime_error("state_hash: output array too small"); out[n++] = v; };
  auto dev = [&](uint64_t h, const void* p, size_t bytes) { b.resize(bytes); if (bytes) cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost); return fnv_words(h, b); };
  put((uint64_t)pos_);
  for (Q35Layer& Y : L_) {
    if (!Y.full) { put(dev(H0, Y.S, (size_t)Q35_GV * Q35_GD * Q35_GD * 4)); put(dev(H0, Y.hist, (size_t)3 * Q35_CONV * 2)); continue; }
    uint64_t hk = H0, hv = H0, hks = H0, hvs = H0;
    for (int kh = 0; kh < Q35_KVH; ++kh) {
      const size_t r = (size_t)kh * max_ctx_;
      if (kv8_) {
        if (Y.kp) { hk = dev(hk, Y.kp + r * (32 * tbcap_kb_), (size_t)pos_ * (32 * tbcap_kb_)); hks = dev(hks, Y.kps + r, (size_t)pos_ * 4); }
        else { hk = dev(hk, Y.k8 + r * Q35_HD, (size_t)pos_ * Q35_HD); hks = dev(hks, Y.ks + r, (size_t)pos_ * 4); }
        if (Y.vp) { hv = dev(hv, Y.vp + r * (32 * tbcap_kb_), (size_t)pos_ * (32 * tbcap_kb_)); hvs = dev(hvs, Y.vps + r, (size_t)pos_ * 4); }
        else { hv = dev(hv, Y.v8 + r * Q35_HD, (size_t)pos_ * Q35_HD); hvs = dev(hvs, Y.vs + r, (size_t)pos_ * 4); } }
      else { hk = dev(hk, Y.kc + r * Q35_HD, (size_t)pos_ * Q35_HD * 2); hv = dev(hv, Y.vc + r * Q35_HD, (size_t)pos_ * Q35_HD * 2); }
    }
    put(hk); put(hv); if (kv8_) { put(hks); put(hvs); }
  }
  if (mtp_.on && mtp_.hist) put(dev(H0, mtp_.hist, (size_t)pos_ * H_ * 2));
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("state_hash: ") + cudaGetErrorString(e));
  return n;
}

void Engine35::load_bdraft(const std::string& path) {
  DevMemScope instance_owner(this);
  if (shared_weights_) {
    const auto f = std::filesystem::canonical(path);
    const std::string id = f.string() + ":" + std::to_string(std::filesystem::file_size(f)) + ":" +
      std::to_string(std::filesystem::last_write_time(f).time_since_epoch().count());
    std::lock_guard<std::mutex> guard(shared_weights_->mutex);
    if (!shared_weights_->draft_path.empty() && shared_weights_->draft_path != id)
      throw std::runtime_error("shared instances require the same drafter file");
    shared_weights_->draft_path = id;
  }
  bdr_.c.open(path);
  if (bdr_.c.model.arch != "bdraft2") throw std::runtime_error("load_bdraft: not a bdraft2 container: " + path);
  static const char* FN[9] = {"q", "k", "v", "o", "gate", "up", "down", "akp", "mkp"};
  auto mk = [&](const std::string& name) { DevTensor t = up(name, 0, &bdr_.c); t.part = reinterpret_cast<float*>(dz35<long long>((size_t)16 * t.dev.N * 8)); return t; };
  bdr_.fc = mk("bdraft2.fc");
  for (int i = 0; i < 5; ++i) for (int k = 0; k < 9; ++k) bdr_.L[i][k] = mk("bdraft2.layers." + std::to_string(i) + "." + FN[k]);
  bdr_.hnorm = vec("bdraft2.hidden_norm", 5120, &bdr_.c); bdr_.fnorm = vec("bdraft2.norm", 5120, &bdr_.c);
  for (int i = 0; i < 5; ++i) {
    const std::string pf = "bdraft2.layers." + std::to_string(i) + ".";
    bdr_.ln1[i] = vec(pf + "ln1", 5120, &bdr_.c); bdr_.ln2[i] = vec(pf + "ln2", 5120, &bdr_.c);
    bdr_.qn[i] = vec(pf + "q_norm", 128, &bdr_.c); bdr_.kn[i] = vec(pf + "k_norm", 128, &bdr_.c);
    bdr_.abase[i] = vec(pf + "abase", 4 * 5120, &bdr_.c); bdr_.mbase[i] = vec(pf + "mbase", 4 * 5120, &bdr_.c);
    bdr_.rk[i] = dz35<uint16_t>((size_t)8 * 2048 * 128 * 2); bdr_.rv[i] = dz35<uint16_t>((size_t)8 * 2048 * 128 * 2);
  }
  bdr_.hp = vec("bdraft2.sel.hp", (size_t)256 * 5120, &bdr_.c);
  bdr_.pc = vec("bdraft2.sel.pred", (size_t)V_ * 256, &bdr_.c); bdr_.sc = vec("bdraft2.sel.succ", (size_t)V_ * 256, &bdr_.c);
  bdr_.signs = bdr_.fc.signs_in;                                       
  bdr_.x = dz35<float>((size_t)16 * 25600 * 4); bdr_.x1 = dz35<float>((size_t)BDR_B * 5120 * 4); bdr_.x2 = dz35<float>((size_t)BDR_B * 5120 * 4);
  bdr_.r = dz35<float>(16 * 4); bdr_.alpha = dz35<float>((size_t)16 * 50 * 4); bdr_.us = dz35<float>(16 * 4); bdr_.qx = dz35<int8_t>((size_t)16 * 25600);
  bdr_.h = dz35<float>((size_t)BDR_B * 5120 * 4); bdr_.dyn = dz35<float>((size_t)BDR_B * 1280 * 4); bdr_.att = dz35<float>((size_t)BDR_B * 4096 * 4);
  bdr_.yq = dz35<float>((size_t)BDR_B * 4096 * 4); bdr_.yk = dz35<float>((size_t)16 * 1024 * 4); bdr_.yv = dz35<float>((size_t)16 * 1024 * 4);
  bdr_.yo = dz35<float>((size_t)BDR_B * 5120 * 4); bdr_.yd = dz35<float>((size_t)BDR_B * 5120 * 4); bdr_.yfc = dz35<float>((size_t)16 * 5120 * 4);
  bdr_.yg = dz35<float>((size_t)BDR_B * 17408 * 4); bdr_.yu = dz35<float>((size_t)BDR_B * 17408 * 4); bdr_.ys = dz35<float>((size_t)BDR_B * 17408 * 4);
  bdr_.hf = dz35<float>((size_t)BDR_J * 5120 * 4); bdr_.lg = dz35<float>((size_t)BDR_J * V_ * 4); bdr_.hacc = dz35<long long>((size_t)BDR_B * V_ * 8);
  bdr_.tval = dz35<float>(BDR_J * 16 * 4); bdr_.tidx = dz35<int>(BDR_J * 16 * 4); bdr_.hpv = dz35<float>(BDR_J * 256 * 4);
  bdr_.ids = dz35<int>(BDR_B * 4); bdr_.pos = dz35<int>(4); bdr_.out = dz35<int>(BDR_B * 4); bdr_.cpos = dz35<int>(4);
  bdr_.apart = dz35<float>(bdr_attn_part_bytes()); bdr_.tcv = dz35<float>(BDR_J * bdr_topk_cand() * 4); bdr_.tci = dz35<int>(BDR_J * bdr_topk_cand() * 4);
  bdr_.cpl = dz35<int>(4); launch_set_int(bdr_.cpl, -1, st_);
  const int caps[5] = {5, 19, 33, 47, 61}; set_capture(caps, 5);
  { bool own = false; for (const auto& t : bdr_.c.tensors) own |= t.name == "bdraft2.head_sub";          
    const char* s = std::getenv("YSKN_BDR_SUB");                      
    if (own || (s && s[0])) {
      const Container* C = &bdr_.c; std::string hn = "bdraft2.head_sub", in = "bdraft2.head_sub_ids";
      if (!own) { bdr_.csub.open(s); C = &bdr_.csub; hn = "mtp.head"; in = "mtp.head_ids"; }
      bdr_.hsub = up(hn, C->tensor(hn).cols / 512, C); bdr_.nsub = bdr_.hsub.dev.N;
      const VectorDesc& v = C->vector(in); if (v.dtype != "i32" || v.n != bdr_.nsub) throw std::runtime_error("load_bdraft: " + in);
      bdr_.sub_ids = dz35<int>((size_t)bdr_.nsub * 4); cudaMemcpy(bdr_.sub_ids, C->at<int32_t>(v.off), (size_t)bdr_.nsub * 4, cudaMemcpyHostToDevice);
      const char* o = std::getenv("YSKN_BDR_SUB_ON"); bdr_.sub_on = (o && o[0]) ? std::atoi(o) : 1; } }
  { 
    
    
    
    const char* s = std::getenv("YSKN_BDR_SHEAD"); const char* m = std::getenv("YSKN_BDR_SHEAD_M");
    const bool off = s && std::strcmp(s, "0") == 0, expl = s && s[0] && !off;
    auto bad_m = [](int M) { return M && (M < 16 || M > BDR_SHEAD_CAP || M % 8); };
    if (expl) {
      load_shead(s);
      const int M = (m && m[0]) ? std::atoi(m) : 0;
      if (bad_m(M)) throw std::runtime_error("load_bdraft: YSKN_BDR_SHEAD_M must be 0 or 16..2048, a multiple of 8");
      bdr_.sk_m = M;
    } else if (!off) {
      std::string fn = path; const size_t sl = fn.find_last_of("/\\"), dot = fn.find_last_of('.');
      fn = (dot != std::string::npos && (sl == std::string::npos || dot > sl) ? fn.substr(0, dot) : fn) + ".shead";
      const int M = (m && m[0]) ? std::atoi(m) : 512;
      if (bad_m(M)) throw std::runtime_error("load_bdraft: YSKN_BDR_SHEAD_M must be 0 or 16..2048, a multiple of 8");
      std::FILE* t = M > 0 ? std::fopen(fn.c_str(), "rb") : nullptr;
      if (t) {
        std::fclose(t);
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        const size_t need = (size_t)BDR_SHEAD_R * 5120 * 4 + (size_t)V_ * (BDR_SHEAD_R + 5120 + 8) + ((size_t)2 << 30);
        if (fr < need) std::fprintf(stderr, "[qwen3_5] SHEAD skipped: %.1f GB free on the GPU, %.1f GB needed\n", fr / 1e9, need / 1e9);
        else {
          try { load_shead(fn.c_str()); bdr_.sk_m = M; }
          catch (const std::exception& ex) { free_shead(); cudaGetLastError(); std::fprintf(stderr, "[qwen3_5] SHEAD skipped: %s\n", ex.what()); }
        }
      }
    } }
  { bdr_.alt = dz35<int>(BDR_B * 4); bdr_.alt3 = dz35<int>(BDR_B * 4); bdr_.marg = dz35<float>(BDR_B * 4); bdr_.hb_cfg = dz35<int>(8 * 4); bdr_.hb_go = dz35<int>(4);   
    #include "hcap_tables.inc"
    std::vector<float> tb; tb.insert(tb.end(), HB_EDGES, HB_EDGES + 11); tb.insert(tb.end(), HB_PI, HB_PI + 84); tb.insert(tb.end(), HB_RHO, HB_RHO + 84); tb.insert(tb.end(), HB_RHO3, HB_RHO3 + 84);
    bdr_.hb_tab = dz35<float>(tb.size() * 4); cudaMemcpy(bdr_.hb_tab, tb.data(), tb.size() * 4, cudaMemcpyHostToDevice);
    const char* h = std::getenv("YSKN_HCAP"); bdr_.hb_on = (h && h[0]) ? std::atoi(h) : 1; }   
  bdr_.on = true;
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("load_bdraft: ") + cudaGetErrorString(e));
}


void Engine35::load_shead(const char* path) {
  DevMemScope dm_scope(this);
  std::FILE* f = std::fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("SHEAD: cannot open ") + path);
  struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
  char magic[8]; int32_t hd[4];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "SHEAD001", 8) != 0 || std::fread(hd, 4, 4, f) != 4) throw std::runtime_error("SHEAD: bad header");
  if (hd[0] != V_ || hd[1] != 5120 || hd[2] != BDR_SHEAD_R) throw std::runtime_error("SHEAD: (V, H, r) of the file do not match the engine");
  const size_t V = (size_t)V_, R = BDR_SHEAD_R, H = 5120, BUF = (size_t)16 << 20;
  char* buf = nullptr;                                                   
  if (cudaMallocHost(&buf, BUF) != cudaSuccess || !buf) throw std::runtime_error("SHEAD: no pinned staging buffer");
  struct Pin { char* p; ~Pin() { cudaFreeHost(p); } } pin{buf};
  auto rd = [&](void* dst, size_t bytes) {
    for (size_t done = 0; done < bytes;) {
      const size_t n = std::min(BUF, bytes - done);
      if (std::fread(buf, 1, n, f) != n) throw std::runtime_error("SHEAD: file too short");
      const cudaError_t ce = cudaMemcpy(static_cast<char*>(dst) + done, buf, n, cudaMemcpyHostToDevice);
      if (ce != cudaSuccess) throw std::runtime_error(std::string("SHEAD: upload failed: ") + cudaGetErrorString(ce));
      done += n;
    } };
  const auto sk_file = std::filesystem::canonical(path);
  const std::string sk_key = sk_file.string() + ":" + std::to_string(std::filesystem::file_size(sk_file)) + ":" +
    std::to_string(std::filesystem::last_write_time(sk_file).time_since_epoch().count());
  auto payload = [&](const char* name, size_t bytes) {
    const auto offset = _ftelli64(f);
    void* p = shared35_buffer(shared_weights_, sk_key + name, bytes, [&](void* dst) { rd(dst, bytes); });
    if (_fseeki64(f, offset + bytes, SEEK_SET)) throw std::runtime_error("shared head seek failed");
    return p;
  };
  bdr_.sk_bt = static_cast<float*>(payload("/bt", R * H * 4));
  bdr_.sk_a = static_cast<int8_t*>(payload("/a", V * R));
  bdr_.sk_as = static_cast<float*>(payload("/as", V * 4));
  bdr_.sk_w = static_cast<int8_t*>(payload("/w", V * H));
  bdr_.sk_ws = static_cast<float*>(payload("/ws", V * 4));
  if (std::fgetc(f) != EOF) throw std::runtime_error("SHEAD: trailing bytes after w_s");
  bdr_.sk_q = dz35<float>((BDR_J + 1) * R * 4); bdr_.sk_s = dz35<float>((BDR_J + 1) * V * 4); bdr_.sk_hist = dz35<unsigned>((size_t)BDR_SHEAD_HIST * 4);
  bdr_.sk_thr = dz35<int>(4 * BDR_J * 4); bdr_.sk_ci = dz35<int>((size_t)BDR_J * BDR_SHEAD_CAP * 4); bdr_.sk_cv = dz35<float>((size_t)BDR_J * BDR_SHEAD_CAP * 4);
  bdr_.sk_grid = bdr_shead_grid();
  cudaDeviceSynchronize();                                               
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("SHEAD: ") + cudaGetErrorString(e));
  size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
  std::fprintf(stderr, "[qwen3_5] SHEAD drafter head loaded (%s): rank %d, %.2f GB, score grid %d blocks; GPU free %.1f / %.1f GB\n", path, (int)R,
               (double)(R * H * 4 + V * (R + H + 8)) / 1e9, bdr_.sk_grid, fr / 1e9, tot / 1e9);
}
void Engine35::free_shead() {                                            
  for (void* p : {(void*)bdr_.sk_bt, (void*)bdr_.sk_a, (void*)bdr_.sk_as, (void*)bdr_.sk_w, (void*)bdr_.sk_ws, (void*)bdr_.sk_q, (void*)bdr_.sk_s,
                  (void*)bdr_.sk_hist, (void*)bdr_.sk_thr, (void*)bdr_.sk_ci, (void*)bdr_.sk_cv}) if (p) {
    bool shared = false;
    if (shared_weights_) for (const auto& kv : shared_weights_->blobs) if (kv.second == p) shared = true;
    if (!shared) cudaFree(p);
  }
  bdr_.sk_bt = bdr_.sk_as = bdr_.sk_ws = bdr_.sk_q = bdr_.sk_s = bdr_.sk_cv = nullptr; bdr_.sk_a = bdr_.sk_w = nullptr; bdr_.sk_hist = nullptr;
  bdr_.sk_thr = bdr_.sk_ci = nullptr; bdr_.sk_m = 0; bdr_.sk_grid = 0;
}
static int bdr_lno_cfg() { static const int v = [] { const char* e = std::getenv("YSKN_BDR_CFG"); return e && e[0] ? std::atoi(e) : -1; }(); return v; }   
void Engine35::bdr_gemv(std::initializer_list<DevTensor*> ts, int M) {
  GemvLaunch P{}; P.M = M; P.stage_bytes = 0;
  for (DevTensor* t : ts) {
    const int ctas = (t->dev.N / 64 + GEMV_WARPS - 1) / GEMV_WARPS;
    P.seg[P.n_seg] = GemvSeg{t->dev, t->part, P.n_ctas}; P.max_workers = std::max(P.max_workers, t->dev.n_workers); P.n_seg++; P.n_ctas += ctas;
  }
  P.qx = bdr_.qx; P.alpha = bdr_.alpha; P.xsum = 1; P.us_out = bdr_.us;
  P.lno_cfg = bdr_lno_cfg();   
  if (!lnord_eligible(P)) throw std::runtime_error("bdraft: GEMV launch not LNORD-eligible");
  launch_lnord(P, st_);
  chk_after(ts, M, bdr_.qx, bdr_.alpha);
}
void Engine35::bdr_ctx_rows(const float* x_dev, int row0, int m) {   
  constexpr int D = 5120;
  auto acc = [](DevTensor& t) { return reinterpret_cast<long long*>(t.part); };
  if (x_dev) launch_bdr_a8(x_dev, 25600, 25600, nullptr, nullptr, bdr_.signs, nullptr, bdr_.qx, bdr_.alpha, m, st_);
  else if(dctx_q_){
    cudaMemcpyAsync(bdr_.qx,dctx_q_+(size_t)row0*25600,(size_t)m*25600,cudaMemcpyDeviceToDevice,st_);
    cudaMemcpyAsync(bdr_.alpha,dctx_a_+(size_t)row0*50,(size_t)m*50*4,cudaMemcpyDeviceToDevice,st_);
  }
  else launch_bdr_a8_cap(dctx_cap_ ? dctx_cap_ : cap_, dctx_cap_ ? dctx_rows_ : Pmax_, row0, bdr_.signs, bdr_.qx, bdr_.alpha, m, st_);
  bdr_gemv({&bdr_.fc}, m);
  launch_bdr_post(acc(bdr_.fc), bdr_.us, bdr_.fc.tau, D, bdr_.yfc, D, m, st_);
  launch_bdr_a8n(bdr_.yfc, D, D, eps_, bdr_.hnorm, bdr_.signs, nullptr, bdr_.qx, bdr_.alpha, m, st_);
  for (int i = 0; i < 5; ++i) {
    DevTensor* L = bdr_.L[i];
    bdr_gemv({&L[1], &L[2]}, m);
    launch_bdr_kv_ring(acc(L[1]), acc(L[2]), bdr_.us, L[1].tau, L[2].tau, bdr_.kn[i], bdr_.cpos, eps_, bdr_.rk[i], bdr_.rv[i], m, st_);
  }
}
void Engine35::bdraft_ctx(int row0, int n, int pos0) {
  if (!bdr_.on) return;
  ++sh_epoch_;
  for (int s = 0; s < n; s += 16) { launch_set_int(bdr_.cpos, pos0 + s, st_); bdr_ctx_rows(nullptr, row0 + s, std::min(16, n - s)); }
}
void Engine35::bdr_block(int rows) { bdr_block_impl(d_toks_, true, rows); }
void Engine35::bdr_block_impl(int* toks, bool hcap, int rows) {   
  constexpr int D = 5120, FF = 17408;
  if (rows < 2 || rows > BDR_B) throw std::runtime_error("bdraft: block rows must be 2..BDR_B");   
  const int R = rows, J = rows - 1;
  auto acc = [](DevTensor& t) { return reinterpret_cast<long long*>(t.part); };
  launch_bdr_block_ids(toks, 248070, bdr_.ids, R, st_);
  const uint16_t* emb = nullptr; const int* gids = nullptr; emb_src(bdr_.ids, R, emb, gids);
  launch_bdr_embed(emb, gids, bdr_.h, R, st_);
  for (int i = 0; i < 5; ++i) {
    DevTensor* L = bdr_.L[i];
    launch_bdr_a8n(bdr_.h, D, D, eps_, bdr_.ln1[i], bdr_.signs, bdr_.x1, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[7]}, R);
    launch_bdr_dynconv_a8(acc(L[7]), bdr_.us, L[7].tau, bdr_.dyn, bdr_.x1, bdr_.abase[i], bdr_.signs, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[0], &L[1], &L[2]}, R);
    launch_bdr_qkv_post(acc(L[0]), acc(L[1]), acc(L[2]), bdr_.us, L[0].tau, L[1].tau, L[2].tau, bdr_.qn[i], bdr_.kn[i], bdr_.pos, eps_, bdr_.yq, bdr_.yk, bdr_.yv, R, st_);
    launch_bdr_attn_part(bdr_.yq, bdr_.yk, bdr_.yv, bdr_.rk[i], bdr_.rv[i], bdr_.pos, R, bdr_.apart, st_);
    launch_bdr_attn_comb_a8(bdr_.apart, bdr_.signs, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[3]}, R);
    launch_bdr_post_conv_res(acc(L[3]), bdr_.us, L[3].tau, bdr_.dyn, bdr_.abase[i] + 2 * D, R, bdr_.h, st_);
    launch_bdr_a8n(bdr_.h, D, D, eps_, bdr_.ln2[i], bdr_.signs, bdr_.x1, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[8]}, R);
    launch_bdr_dynconv_a8(acc(L[8]), bdr_.us, L[8].tau, bdr_.dyn, bdr_.x1, bdr_.mbase[i], bdr_.signs, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[4], &L[5]}, R);
    launch_bdr_glu_a8(acc(L[4]), acc(L[5]), bdr_.us, L[4].tau, L[5].tau, FF, bdr_.signs, bdr_.qx, bdr_.alpha, R, st_);
    bdr_gemv({&L[6]}, R);
    launch_bdr_post_conv_res(acc(L[6]), bdr_.us, L[6].tau, bdr_.dyn, bdr_.mbase[i] + 2 * D, R, bdr_.h, st_);
  }
  launch_bdr_a8n(bdr_.h + D, D, D, eps_, bdr_.fnorm, bdr_.signs, bdr_.hf, bdr_.qx, bdr_.alpha, J, st_);   
  if (bdr_.sk_m > 0) {                                                  
    launch_bdr_shead(bdr_.hf, bdr_.sk_bt, bdr_.sk_a, bdr_.sk_as, bdr_.sk_w, bdr_.sk_ws, V_, bdr_.sk_m, bdr_.sk_grid, bdr_.sk_q, bdr_.sk_s, bdr_.sk_hist,
                     bdr_.sk_thr, bdr_.sk_ci, bdr_.sk_cv, bdr_.tval, bdr_.tidx, J, st_);
  } else {
  const bool sub = bdr_.sub_on && bdr_.nsub > 0; const DevTensor& HT = sub ? bdr_.hsub : head_; const int NV = sub ? bdr_.nsub : V_;   
  GemvLaunch P{}; P.M = J; P.stage_bytes = 0; P.seg[0] = GemvSeg{HT.dev, reinterpret_cast<float*>(bdr_.hacc), 0}; P.n_seg = 1;   
  P.n_ctas = (HT.dev.N / 64 + GEMV_WARPS - 1) / GEMV_WARPS; P.max_workers = HT.dev.n_workers;
  P.qx = bdr_.qx; P.alpha = bdr_.alpha; P.xsum = 1; P.us_out = bdr_.us; P.lno_cfg = bdr_lno_cfg();
  if (!lnord_eligible(P)) throw std::runtime_error("bdraft: head GEMV not LNORD-eligible");
  launch_lnord(P, st_);
  launch_bdr_post(bdr_.hacc, bdr_.us, HT.tau, NV, bdr_.lg, NV, J, st_);
  launch_bdr_topk16(bdr_.lg, NV, bdr_.tval, bdr_.tidx, J, bdr_.tcv, bdr_.tci, st_);
  if (sub) launch_bdr_map_ids(bdr_.tidx, bdr_.sub_ids, J * 16, st_);   
  }
  launch_bdr_hproj(bdr_.hf, bdr_.hp, bdr_.hpv, J, st_);
  launch_bdr_select(bdr_.hpv, bdr_.tval, bdr_.tidx, bdr_.pc, bdr_.sc, toks, bdr_.out, d_smp_, bdr_.cpl, bdr_.pos, J, st_, bdr_.alt, bdr_.marg, bdr_.alt3);
  if (hcap && bdr_.hb_on) launch_bdr_hcap(toks, bdr_.alt, bdr_.alt3, bdr_.marg, bdr_.hb_tab, d_wk_, bdr_.hb_cfg, bdr_.cpl, bdr_.hb_go, st_);   
}
void Engine35::bdraft_draft(int use_graph) { bdraft_step(0, 0, 1, use_graph); }
void Engine35::bdraft_step(int n_ctx, int pos_ctx, int draft, int use_graph) {
  if (!bdr_.on) throw std::runtime_error("bdraft_step: no drafter loaded");
  if (n_ctx < 0 || n_ctx > 16 || (n_ctx && (pos_ctx < 0 || pos_ctx + n_ctx > max_ctx_))) throw std::runtime_error("bdraft_step: bad context rows");
  if (!n_ctx && !draft) return;
  ++sh_epoch_;
  if (opt_rp_defer_) {                                                  
    SetInts S; if (n_ctx) S.set(bdr_.cpos, pos_ctx);
    if (draft) { S.set(bdr_.pos, pos_); if (bdr_.hb_on) S.set(bdr_.hb_go, bdr_.hb_pass); S.set(bdr_.cpl, (smp_ && smp_p_.version == 3) ? bdr_.cpl_pm : -1); }
    launch_set_ints(S, st_); } else {
  if (n_ctx) launch_set_int(bdr_.cpos, pos_ctx, st_);
  if (draft) { launch_set_int(bdr_.pos, pos_, st_);                    
               if (bdr_.hb_on) launch_set_int(bdr_.hb_go, bdr_.hb_pass, st_);   
               launch_set_int(bdr_.cpl, (smp_ && smp_p_.version == 3) ? bdr_.cpl_pm : -1, st_); } }   
  auto body = [&] { if (n_ctx) bdr_ctx_rows(nullptr, 0, n_ctx); if (draft) bdr_block(draft >= 2 ? BDR_B : 8); };   
  if (!use_graph) { body(); rp_launch(); return; }
  cudaGraphExec_t& ge = bdr_.sgraph[draft >= 2 ? 2 : (draft ? 1 : 0)][n_ctx];
  if (!ge) {
    cudaGraph_t g = nullptr; cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal); body(); cudaStreamEndCapture(st_, &g);
    if (!g || cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) throw std::runtime_error("bdraft_step: graph capture failed");
    cudaGraphDestroy(g);
  }
  cudaGraphLaunch(ge, st_); rp_launch();                               
}
void Engine35::bdraft_ring(int layer, int slot0, int n, uint16_t* k_out, uint16_t* v_out) {
  cudaStreamSynchronize(st_);
  for (int hh = 0; hh < 8; ++hh) {
    cudaMemcpy(k_out + (size_t)hh * n * 128, bdr_.rk[layer] + ((size_t)hh * 2048 + slot0) * 128, (size_t)n * 128 * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_out + (size_t)hh * n * 128, bdr_.rv[layer] + ((size_t)hh * 2048 + slot0) * 128, (size_t)n * 128 * 2, cudaMemcpyDeviceToHost);
  }
}
int Engine35::bdraft_out(int32_t* drafts) {
  cudaMemcpyAsync(h_pin_ + PIN_GT, bdr_.out, BDR_J * 4, cudaMemcpyDeviceToHost, st_); cudaStreamSynchronize(st_);
  std::memcpy(drafts, h_pin_ + PIN_GT, BDR_J * 4); return BDR_J;
}
void Engine35::bdraft_debug(const float* feats, int n, int pos0, int32_t anchor, float* hidden, int32_t* top_ids, float* top_vals, int32_t* path) {
  if (!bdr_.on) throw std::runtime_error("bdraft_debug: no drafter loaded");
  replay_fence(); cudaStreamSynchronize(st_); ++sh_epoch_; snap_drop();
  static const int chunk = [] { const char* v = std::getenv("YSKN_BDR_CHUNK"); const int c = v ? std::atoi(v) : 16; return c >= 1 && c <= 16 ? c : 16; }();   
  for (int s = 0; s < n; s += chunk) {
    const int m = std::min(chunk, n - s);
    
    
    cudaMemcpyAsync(bdr_.x, feats + (size_t)s * 25600, (size_t)m * 25600 * 4, cudaMemcpyHostToDevice, st_);
    launch_set_int(bdr_.cpos, pos0 + s, st_); bdr_ctx_rows(bdr_.x, 0, m); cudaStreamSynchronize(st_);
  }
  cudaMemcpyAsync(d_toks_, &anchor, 4, cudaMemcpyHostToDevice, st_); cudaStreamSynchronize(st_);
  launch_set_int(bdr_.pos, pos0 + n, st_); launch_set_int(bdr_.cpl, -1, st_); bdr_block(8); cudaStreamSynchronize(st_);
  cudaMemcpy(hidden, bdr_.hf, (size_t)7 * 5120 * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(top_ids, bdr_.tidx, 7 * 16 * 4, cudaMemcpyDeviceToHost); cudaMemcpy(top_vals, bdr_.tval, 7 * 16 * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(path, bdr_.out, 7 * 4, cudaMemcpyDeviceToHost);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("bdraft_debug: ") + cudaGetErrorString(e));
}
void Engine35::debug_hidden(const int32_t* ids, int n, int upto, int mode, float* out) {
  reset();
  if (upto < 0 || upto > nlayers_) throw std::runtime_error("debug_hidden: upto out of range");
  if (mode == 99) { flow_trace_read(out, n); return; }                   
  if (mode == 0) {
    if (n < 1 || n > Pmax_) throw std::runtime_error("debug_hidden: mode 0 needs 1..Pmax tokens");
    upload_rows(ids, n, nullptr);
    pass_body(n, true, upto, HEAD_NONE, false, false, false);
    cudaMemcpyAsync(out, h_, (size_t)n * H_ * 4, cudaMemcpyDeviceToHost, st_);
  } else {
    for (int i = 0; i < n; ++i) {
      h_pin_[PIN_IDS] = ids[i]; cudaMemcpyAsync(d_toks_, h_pin_ + PIN_IDS, 4, cudaMemcpyHostToDevice, st_); launch_set_int(d_pos_, i, st_);
      pass_body(1, true, upto, HEAD_NONE, false, false, false);
      cudaStreamSynchronize(st_);
    }
    cudaMemcpyAsync(out, h_, (size_t)H_ * 4, cudaMemcpyDeviceToHost, st_);
  }
  cudaStreamSynchronize(st_);
  const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("debug_hidden: ") + cudaGetErrorString(e));
}
} 
