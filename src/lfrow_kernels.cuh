// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors




















namespace yskn {

struct WkConvArgs     { Q35ConvArgs a;     const uint8_t* wk; };
struct WkStateArgs    { Q35StateArgs a;    const uint8_t* wk; };
struct WkAttnPostArgs { Q35AttnPostArgs a; const uint8_t* wk; };
struct WkAttnArgs     { Q35AttnArgs a;     const uint8_t* wk; };


__global__ void __launch_bounds__(128) wk_conv_kernel(WkConvArgs W) {
  const Q35ConvArgs& A = W.a;
  __shared__ float s_red[4];
  const int slot = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kind = slot < Q35_GK ? 0 : (slot < 2 * Q35_GK ? 1 : 2);        
  const int head = kind == 0 ? slot : (kind == 1 ? slot - Q35_GK : slot - 2 * Q35_GK);
  const int c = (kind == 0 ? 0 : (kind == 1 ? Q35_GK * Q35_GD : 2 * Q35_GK * Q35_GD)) + head * Q35_GD + tid;
  const uint16_t cw0 = A.conv_w[c * 4], cw1 = A.conv_w[c * 4 + 1], cw2 = A.conv_w[c * 4 + 2], cw3 = A.conv_w[c * 4 + 3];
  const int dp = W.wk ? (int)W.wk[t] : t;                                  
  uint16_t hv[3] = {0, 0, 0};
  #pragma unroll
  for (int k = 0; k < 3; ++k) if (dp - 3 + k < 0) hv[k] = __ldcg(A.hist + (size_t)(dp + k) * Q35_CONV + c);                
  uint16_t c_dt = 0, c_al = 0;
  if (kind == 2 && tid == 0) { c_dt = A.dt_bias[head]; c_al = A.A_log[head]; }
  pdl_wait(); pdl_trigger();
  const float w0 = bf2f(cw0), w1 = bf2f(cw1), w2 = bf2f(cw2), w3 = bf2f(cw3);
  auto xin = [&](int r) { return r >= 0 ? bf2f(A.mixed[(size_t)r * Q35_CONV + c]) : bf2f(hv[r - dp + 3]); };              
  const float s1 = xin(dp - 3), s2 = xin(dp - 2), s3 = xin(dp - 1), x = bf2f(A.mixed[(size_t)t * Q35_CONV + c]);          
  const float acc = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(w0, s1), __fmul_rn(w1, s2)), __fmul_rn(w2, s3)), __fmul_rn(w3, x));
  const float act = rbf(q35_silu(rbf(acc)));
  if (kind == 2) {
    A.v[((size_t)t * Q35_GV + head) * Q35_GD + tid] = act;
    if (tid == 0) {
      const float b = A.ab[(size_t)t * 96 + head], a = A.ab[(size_t)t * 96 + 48 + head];
      A.beta[(size_t)t * Q35_GV + head] = rbf(q35_sigmoid(b));
      const float sp_in = __fadd_rn(a, bf2f(c_dt));
      const float sp = q35_softplus(sp_in);
      A.g[(size_t)t * Q35_GV + head] = __fmul_rn(-q35_exp(bf2f(c_al)), sp);
    }
  } else {
    float ss = __fmul_rn(act, act);
    for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(__fadd_rn(s_red[0], s_red[1]), s_red[2]), s_red[3]);
    const float y = __fmul_rn(act, q35_rsqrt(__fadd_rn(tot, 1e-6f)));
    if (kind == 0) A.q[((size_t)t * Q35_GK + head) * Q35_GD + tid] = __fmul_rn(y, 0.08838834764831845f);
    else A.k[((size_t)t * Q35_GK + head) * Q35_GD + tid] = y;
  }
}
void launch_wk_conv(const Q35ConvArgs& A, const uint8_t* wk, cudaStream_t st) {
  if (wk && A.M > 16) throw std::runtime_error("launch_wk_conv: a row table needs M <= 16");
  launch_k(wk_conv_kernel, dim3(2 * Q35_GK + Q35_GV, A.M), dim3(128), 0, st, WkConvArgs{A, wk}); }





__global__ void __launch_bounds__(128) wk_state_kernel(WkStateArgs W) {
  const Q35StateArgs& A = W.a;
  __shared__ float sk[2][Q35_GD], sq[2][Q35_GD], skv[2][4][32], so[2][4][32];
  const int h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, j = blockIdx.y * 32 + lane, kh = h / (Q35_GV / Q35_GK);
  float* const Sp = A.S + (size_t)h * Q35_GD * Q35_GD;
  float s[32];
  #pragma unroll
  for (int r = 0; r < 32; ++r) s[r] = __ldcg(Sp + (size_t)(warp * 32 + r) * Q35_GD + j);
  const bool tab = W.wk != nullptr;                                        
  unsigned long long sw0 = 0ull, sw1 = 0ull;
  if (tab) { const uint4 u = __ldcg(reinterpret_cast<const uint4*>(W.wk + 16));
    sw0 = (unsigned long long)u.x | ((unsigned long long)u.y << 32); sw1 = (unsigned long long)u.z | ((unsigned long long)u.w << 32); }
  auto step = [&](int i) { return tab ? (int)(((i < 8 ? sw0 : sw1) >> (8 * (i & 7))) & 0xffull) : i; };   
  pdl_wait(); pdl_trigger();
  if (A.n_commit > 0) {                                                    
    const int gt = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid;
    if (gt < Q35_CONV) { const int n = A.n_commit; uint16_t o[3], nw[3];
      #pragma unroll
      for (int k = 0; k < 3; ++k) o[k] = A.hist[(size_t)k * Q35_CONV + gt];
      #pragma unroll
      for (int k = 0; k < 3; ++k) { const int r = n - 3 + k; nw[k] = r >= 0 ? A.mixed[(size_t)r * Q35_CONV + gt] : o[3 + r]; }
      #pragma unroll
      for (int k = 0; k < 3; ++k) A.hist[(size_t)k * Q35_CONV + gt] = nw[k]; }
  }
  auto ld = [&](int t, float& k, float& q, float& g, float& b, float& v) {
    k = A.k[((size_t)t * Q35_GK + kh) * Q35_GD + tid]; q = A.q[((size_t)t * Q35_GK + kh) * Q35_GD + tid];
    g = A.g[(size_t)t * Q35_GV + h]; b = A.beta[(size_t)t * Q35_GV + h]; v = A.v[((size_t)t * Q35_GV + h) * Q35_GD + j]; };
  float nk = 0.f, nq = 0.f, ng = 0.f, nb = 0.f, nv = 0.f;
  int e = A.M > 0 ? step(0) : 0, prow = 0;                                 
  if (A.M > 0) ld(e & 0x7f, nk, nq, ng, nb, nv);
  for (int i = 0; i < A.M; ++i) {
    const int pb = i & 1, row = e & 0x7f; const bool leaf = (e & 0x80) != 0;                  
    sk[pb][tid] = nk; sq[pb][tid] = nq;
    const float eg = q35_exp(ng), bt = nb, vj = nv;
    const int en = i + 1 < A.M ? step(i + 1) : 0;                                             
    if (i + 1 < A.M) ld(en & 0x7f, nk, nq, ng, nb, nv);
    __syncthreads();
    if (i > 0 && warp == 0) { const int qb = pb ^ 1;
      A.core[((size_t)prow * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(so[qb][0][lane], so[qb][1][lane]), so[qb][2][lane]), so[qb][3][lane])); }
    float kv = 0.f;
    if (!leaf) {
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fmul_rn(s[r], eg); kv = __fadd_rn(kv, __fmul_rn(s[r], sk[pb][warp * 32 + r])); }
    } else {                                                                                  
      #pragma unroll
      for (int r = 0; r < 32; ++r) { const float sr = __fmul_rn(s[r], eg); kv = __fadd_rn(kv, __fmul_rn(sr, sk[pb][warp * 32 + r])); }
    }
    skv[pb][warp][lane] = kv;
    __syncthreads();
    const float d = __fmul_rn(__fsub_rn(vj, __fadd_rn(__fadd_rn(__fadd_rn(skv[pb][0][lane], skv[pb][1][lane]), skv[pb][2][lane]), skv[pb][3][lane])), bt);
    float o = 0.f;
    if (!leaf) {
      #pragma unroll
      for (int r = 0; r < 32; ++r) { s[r] = __fadd_rn(s[r], __fmul_rn(sk[pb][warp * 32 + r], d)); o = __fadd_rn(o, __fmul_rn(s[r], sq[pb][warp * 32 + r])); }
    } else {                                                                                  
      #pragma unroll
      for (int r = 0; r < 32; ++r) { const float sr = __fadd_rn(__fmul_rn(s[r], eg), __fmul_rn(sk[pb][warp * 32 + r], d)); o = __fadd_rn(o, __fmul_rn(sr, sq[pb][warp * 32 + r])); }
    }
    so[pb][warp][lane] = o;
    if (!leaf && row + 1 == A.n_commit) {
      #pragma unroll
      for (int r = 0; r < 32; ++r) Sp[(size_t)(warp * 32 + r) * Q35_GD + j] = s[r];
    }
    prow = row; e = en;                                                                       
  }
  __syncthreads();
  if (A.M > 0 && warp == 0) { const int qb = (A.M - 1) & 1;
    A.core[((size_t)prow * Q35_GV + h) * Q35_GD + j] = rbf(__fadd_rn(__fadd_rn(__fadd_rn(so[qb][0][lane], so[qb][1][lane]), so[qb][2][lane]), so[qb][3][lane])); }
}
void launch_wk_state(const Q35StateArgs& A, const uint8_t* wk, cudaStream_t st) {
  if (wk && (A.M > 16 || A.n_commit > 0)) throw std::runtime_error("launch_wk_state: a row table needs M <= 16 and a pass that does not commit");
  launch_k(wk_state_kernel, dim3(Q35_GV, Q35_GD / 32), dim3(128), 0, st, WkStateArgs{A, wk}); }



__global__ void __launch_bounds__(512) wk_attn_post_kernel(WkAttnPostArgs W) {
  const Q35AttnPostArgs& A = W.a;
  __shared__ float buf[512], s_red[16];
  const int blk = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* part; const float* tau; const int8_t* sg; int nw, N, n;
  if (blk < Q35_NH) { part = A.part_q; tau = A.tau_q; nw = A.nw_q; N = 2 * Q35_NH * Q35_HD; n = blk * 512 + tid; sg = A.sg_q + blk * 512; }
  else if (blk < Q35_NH + 2) { part = A.part_k; tau = A.tau_k; nw = A.nw_k; N = Q35_KVH * Q35_HD; n = (blk - Q35_NH) * 512 + tid; sg = A.sg_k + (blk - Q35_NH) * 512; }
  else { part = A.part_v; tau = A.tau_v; nw = A.nw_v; N = Q35_KVH * Q35_HD; n = (blk - Q35_NH - 2) * 512 + tid; sg = A.sg_v + (blk - Q35_NH - 2) * 512; }
  const int d = tid & (Q35_HD - 1);
  const float c_tau = tau[n]; const int8_t c_sg = sg[tid]; const uint16_t c_nw = blk < Q35_NH + 2 ? (blk < Q35_NH ? A.qn_w : A.kn_w)[d] : 0;
  const int dp = W.wk ? (int)W.wk[tok] : tok;                              
  pdl_wait(); pdl_trigger();
  if (tid == 0 && A.pf.p[0]) q35_l2pf(A.pf, blk + tok * gridDim.x, gridDim.x * gridDim.y);   
  const int base = *A.d_pos, pos = base + tok, rpos = base + dp;           
  long long acc = 0; float2 nrs = make_float2(0.f, 0.f);
  if (A.xsm.us) { acc = q35_xsm_ld(part, A.xsm.mode, nw, A.M, N, tok, n); if (A.xsm.mode != 3) nrs = q35_xsm_row(A.xsm.us, A.xsm.S, A.xsm.eps, tok); }
  buf[tid] = A.xsm.us ? (A.xsm.mode == 3 ? __fmul_rn(__uint_as_float((unsigned)acc), c_tau) : __fmul_rn(__fmul_rn(q35_xsm_fin(acc, A.xsm.mode, nrs.x), nrs.y), c_tau))
                      : __fmul_rn(c_tau, q35_combine(part, nw, A.M, N, tok, n)); __syncthreads();
  q35_inv_h512s(buf, (float)c_sg);
  const float y = rbf(buf[tid]);
  if (blk >= Q35_NH + 2) {
    const int kvh = (blk - Q35_NH - 2) * 2 + (tid >> 8);
    if (A.v8) {
      float vy=y;
#if YSKN_KVX_ROT
      vy=kvx_rotate(vy,buf,false);
#endif
      q35_kv8_store(vy,A.v8,A.vs,q35_kvrow(A.vstg,A.max_ctx,kvh,pos,tok),d); return; }   
    A.vc[((size_t)kvh * A.max_ctx + pos) * Q35_HD + d] = q35_f2kv16(y, A.kv_fp16);
    return;
  }
  if (blk < Q35_NH && tid >= Q35_HD) A.gate16[((size_t)tok * Q35_NH + blk) * Q35_HD + d] = f2bf(y);
  float ss = __fmul_rn(y, y);
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  if (lane == 0) s_red[warp] = ss;
  __syncthreads();
  const int w0 = (tid >> 8) * 8; float tot = 0.f;
  for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, s_red[w0 + w]);
  const float r = q35_rsqrt(__fadd_rn(__fdiv_rn(tot, (float)Q35_HD), A.eps));
  const float xn = rbf(__fmul_rn(__fmul_rn(y, r), __fadd_rn(1.f, bf2f(c_nw))));
  buf[tid] = xn;
  __syncthreads();
  float out = xn;
  if (d < Q35_RD / 2) { const float2 cs = A.rope[(size_t)rpos * (Q35_RD / 2) + d];                                        
    out = rbf(__fadd_rn(rbf(__fmul_rn(xn, cs.x)), rbf(__fmul_rn(-buf[tid + Q35_RD / 2], cs.y)))); }
  else if (d < Q35_RD) { const float2 cs = A.rope[(size_t)rpos * (Q35_RD / 2) + d - Q35_RD / 2];                          
    out = rbf(__fadd_rn(rbf(__fmul_rn(xn, cs.x)), rbf(__fmul_rn(buf[tid - Q35_RD / 2], cs.y)))); }
#if YSKN_KVX_ROT
  out=kvx_rotate(out,buf,false);
#endif
  if (blk < Q35_NH) { if (tid < Q35_HD) A.q16[((size_t)tok * Q35_NH + blk) * Q35_HD + d] = f2bf(out); }
  else { const int kvh = (blk - Q35_NH) * 2 + (tid >> 8);
         if (A.k8) q35_kv8_store(out, A.k8, A.ks, q35_kvrow(A.kstg, A.max_ctx, kvh, pos, tok), d); else A.kc[((size_t)kvh * A.max_ctx + pos) * Q35_HD + d] = q35_f2kv16(out, A.kv_fp16); }
}
void launch_wk_attn_post(const Q35AttnPostArgs& A, const uint8_t* wk, cudaStream_t st) {
  if (wk && A.M > 16) throw std::runtime_error("launch_wk_attn_post: a row table needs M <= 16");
  launch_k(wk_attn_post_kernel, dim3(Q35_NH + 4, A.M), dim3(512), 0, st, WkAttnPostArgs{A, wk}); }





template <bool KV8>
__global__ void __launch_bounds__(128) wk_attention_kernel(WkAttnArgs W) {
  const Q35AttnArgs& A = W.a;
  constexpr int G = Q35_NH / Q35_KVH, PF = 4;
  __shared__ float sm[4][G], sl[4][G], sacc[4][G][Q35_HD];
  const int kvh = blockIdx.x, split = blockIdx.y, tok = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* kb = A.kc + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane;
  const uint16_t* vb = A.vc + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane;
  const int8_t* const kb8 = KV8 ? A.k8 + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane : nullptr;   
  const int8_t* const vb8 = KV8 ? A.v8 + (size_t)kvh * A.max_ctx * Q35_HD + 8 * lane : nullptr;
  const float* const ksb = KV8 ? A.ks + (size_t)kvh * A.max_ctx : nullptr; const float* const vsb = KV8 ? A.vs + (size_t)kvh * A.max_ctx : nullptr;
  auto ldkv = [&](uint4& kk, uint4& vv, float& ss, float& tt, size_t t) {
    if constexpr (KV8) { const uint2 k = __ldcg(reinterpret_cast<const uint2*>(kb8 + t * Q35_HD)), v = __ldcg(reinterpret_cast<const uint2*>(vb8 + t * Q35_HD));
                         kk = make_uint4(k.x, k.y, 0u, 0u); vv = make_uint4(v.x, v.y, 0u, 0u); ss = __ldcg(ksb + t); tt = __ldcg(vsb + t); }
    else { kk = __ldcg(reinterpret_cast<const uint4*>(kb + t * Q35_HD)); vv = __ldcg(reinterpret_cast<const uint4*>(vb + t * Q35_HD)); (void)ss; (void)tt; }
  };
  const int dp = W.wk ? (int)W.wk[tok] : tok;                              
  uint4 kq[PF], vq[PF]; float ksq[PF], vsq[PF];
  #pragma unroll
  for (int i = 0; i < PF; ++i) { kq[i] = vq[i] = make_uint4(0u, 0u, 0u, 0u); ksq[i] = vsq[i] = 0.f; }
  int base = 0;
  if (A.early) {
    base = *A.d_pos; const int len = base + dp + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);   
    #pragma unroll
    for (int i = 0; i < PF; ++i) { const int t = t0 + warp + 4 * i;
      if (t < t1 && t < base) { ldkv(kq[i], vq[i], ksq[i], vsq[i], (size_t)t); } }
  }
  pdl_wait(); pdl_trigger();
  if (!A.early) base = *A.d_pos;
  const int len = base + dp + 1, per = (len + A.splits - 1) / A.splits, t0 = split * per, t1 = min(len, t0 + per);                     
  const int own = base + tok;                                              
  auto sl_of = [&](int t) { return (size_t)(t == len - 1 ? own : t); };    
  #pragma unroll
  for (int i = 0; i < PF; ++i) { const int t = t0 + warp + 4 * i;
    if (t < t1 && (t >= base || !A.early)) { ldkv(kq[i], vq[i], ksq[i], vsq[i], sl_of(t)); } }   
  float q[G][8];
  #pragma unroll
  for (int i = 0; i < G; ++i) { const uint4 u = *reinterpret_cast<const uint4*>(A.q16 + ((size_t)tok * Q35_NH + kvh * G + i) * Q35_HD + 8 * lane);
    const unsigned uu[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int e = 0; e < 4; ++e) { q[i][2 * e] = __fmul_rn(__uint_as_float(uu[e] << 16), 0.0625f); q[i][2 * e + 1] = __fmul_rn(__uint_as_float(uu[e] & 0xffff0000u), 0.0625f); } }
  float mh = -1e30f, lh = 0.f, acc[G][8];
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f; }
  for (int tb = t0 + warp; tb < t1; tb += 4 * PF) {
    uint4 kn[PF], vn[PF]; float ksn[PF], vsn[PF];
    #pragma unroll
    for (int i = 0; i < PF; ++i) { const int t = tb + 4 * PF + 4 * i; kn[i] = vn[i] = make_uint4(0u, 0u, 0u, 0u); ksn[i] = vsn[i] = 0.f;
      if (t < t1) { ldkv(kn[i], vn[i], ksn[i], vsn[i], sl_of(t)); } }   
    #pragma unroll
    for (int i = 0; i < PF; ++i) {
    if (tb + 4 * i >= t1) break;
    const uint4 kc = kq[i], vc = vq[i];
    const float ksc = KV8 ? ksq[i] : 0.f, vsc = KV8 ? vsq[i] : 0.f; (void)ksc; (void)vsc;
    float kd[8], vd[8];
    if constexpr (KV8) {                                                   
      #pragma unroll
      for (int e = 0; e < 8; ++e) { kd[e] = (float)(signed char)((e < 4 ? kc.x : kc.y) >> (8 * (e & 3))); vd[e] = (float)(signed char)((e < 4 ? vc.x : vc.y) >> (8 * (e & 3))); }
    } else
    { const unsigned ku[4] = {kc.x, kc.y, kc.z, kc.w}, vu[4] = {vc.x, vc.y, vc.z, vc.w};
      #pragma unroll
      for (int e = 0; e < 4; ++e) { kd[2 * e] = __uint_as_float(ku[e] << 16); kd[2 * e + 1] = __uint_as_float(ku[e] & 0xffff0000u);
                                    vd[2 * e] = __uint_as_float(vu[e] << 16); vd[2 * e + 1] = __uint_as_float(vu[e] & 0xffff0000u); } }
    float p[8];
    #pragma unroll
    for (int i = 0; i < G; ++i) { float s = __fmul_rn(q[i][0], kd[0]);
      #pragma unroll
      for (int e = 1; e < 8; ++e) s = __fadd_rn(s, __fmul_rn(q[i][e], kd[e]));
      p[i] = s; }
    #pragma unroll
    for (int i = G; i < 8; ++i) p[i] = 0.f;
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float v4[4], v2[2], v1;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float send = b4 ? p[k] : p[k + 4], keep = b4 ? p[k + 4] : p[k]; v4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16); }
    #pragma unroll
    for (int k = 0; k < 2; ++k) { const float send = b3 ? v4[k] : v4[k + 2], keep = b3 ? v4[k + 2] : v4[k]; v2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8); }
    { const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0]; v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4); }
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    if constexpr (KV8) q35_softmax8<G>(__fmul_rn(v1, ksc), mh, lh, acc, vd, vsc); else q35_softmax<G>(v1, mh, lh, acc, vd);
    }
    #pragma unroll
    for (int i = 0; i < PF; ++i) { kq[i] = kn[i]; vq[i] = vn[i]; ksq[i] = ksn[i]; vsq[i] = vsn[i]; }
  }
  if ((lane & 3) == 0 && lane < 4 * G) { sm[warp][lane >> 2] = mh; sl[warp][lane >> 2] = lh; }
  #pragma unroll
  for (int i = 0; i < G; ++i) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) sacc[warp][i][8 * lane + e] = acc[i][e]; }
  __syncthreads();
  for (int i = warp; i < G; i += 4) {
    float M2 = sm[0][i]; for (int w = 1; w < 4; ++w) M2 = fmaxf(M2, sm[w][i]);
    float L2 = 0.f, o8[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) o8[e] = 0.f;
    for (int w = 0; w < 4; ++w) { const float sc = (sm[w][i] > -1e29f) ? expf(sm[w][i] - M2) : 0.f; L2 += sc * sl[w][i];
      #pragma unroll
      for (int e = 0; e < 8; ++e) o8[e] += sc * sacc[w][i][8 * lane + e]; }
    const size_t pidx = ((size_t)tok * A.splits + split) * Q35_NH + kvh * G + i;
    if (lane == 0) { A.pm[pidx] = M2; A.pl[pidx] = L2; }
    #pragma unroll
    for (int e = 0; e < 8; ++e) A.pacc[pidx * Q35_HD + 8 * lane + e] = o8[e];
  }
}
void launch_wk_attention(const Q35AttnArgs& A, const uint8_t* wk, cudaStream_t st) {
  if (wk && A.M > 16) throw std::runtime_error("launch_wk_attention: a row table needs M <= 16");
  if (A.k8) launch_k(wk_attention_kernel<true>, dim3(Q35_KVH, A.splits, A.M), dim3(128), 0, st, WkAttnArgs{A, wk});
  else launch_k(wk_attention_kernel<false>, dim3(Q35_KVH, A.splits, A.M), dim3(128), 0, st, WkAttnArgs{A, wk}); }





struct WkPromoteArgs2 { const WkCopy* tab; int src; int dst; int base; int max_ctx; };
__global__ void __launch_bounds__(256) wk_promote2_kernel(WkPromoteArgs2 A) {
  pdl_wait(); pdl_trigger();
  const WkCopy c = A.tab[blockIdx.x]; const int tid = threadIdx.x, nt = blockDim.x;
  if (c.kind == 0) {
    unsigned* const p = reinterpret_cast<unsigned*>(c.ptr); const int w = c.row_bytes / 4;
    for (int i = tid; i < w; i += nt) p[(size_t)A.dst * w + i] = p[(size_t)A.src * w + i];
  } else if (c.kind == 2) {                                               
    int8_t* const k8 = reinterpret_cast<int8_t*>(c.ptr);
    for (int i = tid; i < Q35_KVH * Q35_HD; i += nt) { const int kvh = i / Q35_HD, d = i % Q35_HD;
      k8[((size_t)kvh * A.max_ctx + A.base + A.dst) * Q35_HD + d] = k8[((size_t)kvh * A.max_ctx + A.base + A.src) * Q35_HD + d]; }
  } else if (c.kind == 3) {                                               
    float* const sc = reinterpret_cast<float*>(c.ptr);
    if (tid < Q35_KVH) sc[(size_t)tid * A.max_ctx + A.base + A.dst] = sc[(size_t)tid * A.max_ctx + A.base + A.src];
  } else if (c.kind == 4) {                                               
    unsigned* const p = reinterpret_cast<unsigned*>(c.ptr); const int w = c.row_bytes / 4;
    for (int i = tid; i < Q35_KVH * w; i += nt) { const int kvh = i / w, j = i - kvh * w;
      p[((size_t)kvh * A.max_ctx + A.base + A.dst) * w + j] = p[((size_t)kvh * A.max_ctx + A.base + A.src) * w + j]; }
  } else {
    uint16_t* const kc = reinterpret_cast<uint16_t*>(c.ptr);
    for (int i = tid; i < Q35_KVH * Q35_HD; i += nt) { const int kvh = i / Q35_HD, d = i % Q35_HD;
      kc[((size_t)kvh * A.max_ctx + A.base + A.dst) * Q35_HD + d] = kc[((size_t)kvh * A.max_ctx + A.base + A.src) * Q35_HD + d]; }
  }
}
void launch_wk_promote(const WkCopy* tab, int n, int src, int dst, int base, int max_ctx, cudaStream_t st) {
  if (src == dst || n <= 0) return;
  launch_k(wk_promote2_kernel, dim3(n), dim3(256), 0, st, WkPromoteArgs2{tab, src, dst, base, max_ctx}); }



__global__ void wk_fill_kernel(int* toks, const uint8_t* wk, const int* lg_idx, int k, int M) {
  pdl_wait(); pdl_trigger();
  const int t = threadIdx.x;
  if (t > k && t < M) { const int d = wk[t], rank = wk[32 + t]; toks[t] = lg_idx[(d - 1) * 8 + rank]; }
}
void launch_wk_fill(int* toks, const uint8_t* wk, const int* lg_idx, int k, int M, cudaStream_t st) {
  launch_k(wk_fill_kernel, dim3(1), dim3(32), 0, st, toks, wk, lg_idx, k, M); }

} 
