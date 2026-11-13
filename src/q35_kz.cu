// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include "q35.hpp"
#include "kz_common.cuh"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
namespace yskn {














namespace {
constexpr int KZ_KT = 32, KZ_RS = 272, KZ_VS = 36;
constexpr size_t KZ_QM = (size_t)6 * 2 * 8 * 16 * 32;                               
constexpr size_t kz_smem(int ns) {
  return KZ_QM + (size_t)ns * 2 * KZ_KT * KZ_RS + (size_t)ns * 2 * KZ_KT * 4 + (size_t)Q35_HD * KZ_VS + (size_t)3 * KZ_KT * 4 + 256 * 4; }
__constant__ unsigned kz_exp2t[256] = {                                                
  2147483648u, 2141676973u, 2135885998u, 2130110682u, 2124350982u, 2118606857u, 2112878262u, 2107165158u,
  2101467502u, 2095785251u, 2090118366u, 2084466803u, 2078830522u, 2073209480u, 2067603638u, 2062012954u,
  2056437387u, 2050876895u, 2045331439u, 2039800978u, 2034285470u, 2028784876u, 2023299156u, 2017828268u,
  2012372174u, 2006930832u, 2001504204u, 1996092249u, 1990694927u, 1985312200u, 1979944027u, 1974590370u,
  1969251188u, 1963926443u, 1958616096u, 1953320108u, 1948038440u, 1942771053u, 1937517909u, 1932278970u,
  1927054196u, 1921843549u, 1916646992u, 1911464486u, 1906295993u, 1901141476u, 1896000896u, 1890874216u,
  1885761398u, 1880662405u, 1875577199u, 1870505744u, 1865448001u, 1860403934u, 1855373507u, 1850356681u,
  1845353420u, 1840363688u, 1835387448u, 1830424663u, 1825475297u, 1820539314u, 1815616678u, 1810707353u,
  1805811301u, 1800928489u, 1796058879u, 1791202437u, 1786359126u, 1781528911u, 1776711757u, 1771907628u,
  1767116489u, 1762338305u, 1757573041u, 1752820662u, 1748081133u, 1743354420u, 1738640488u, 1733939301u,
  1729250827u, 1724575029u, 1719911875u, 1715261330u, 1710623359u, 1705997930u, 1701385007u, 1696784557u,
  1692196547u, 1687620943u, 1683057710u, 1678506817u, 1673968228u, 1669441912u, 1664927835u, 1660425963u,
  1655936265u, 1651458706u, 1646993254u, 1642539877u, 1638098541u, 1633669214u, 1629251865u, 1624846459u,
  1620452965u, 1616071351u, 1611701585u, 1607343634u, 1602997467u, 1598663052u, 1594340357u, 1590029350u,
  1585730000u, 1581442275u, 1577166143u, 1572901575u, 1568648537u, 1564406999u, 1560176931u, 1555958300u,
  1551751076u, 1547555228u, 1543370725u, 1539197537u, 1535035634u, 1530884983u, 1526745556u, 1522617322u,
  1518500250u, 1514394310u, 1510299473u, 1506215708u, 1502142985u, 1498081275u, 1494030547u, 1489990772u,
  1485961921u, 1481943963u, 1477936870u, 1473940611u, 1469955159u, 1465980482u, 1462016553u, 1458063343u,
  1454120821u, 1450188960u, 1446267730u, 1442357104u, 1438457051u, 1434567544u, 1430688553u, 1426820052u,
  1422962010u, 1419114401u, 1415277195u, 1411450365u, 1407633882u, 1403827719u, 1400031848u, 1396246240u,
  1392470869u, 1388705706u, 1384950723u, 1381205894u, 1377471191u, 1373746586u, 1370032052u, 1366327563u,
  1362633090u, 1358948606u, 1355274085u, 1351609500u, 1347954824u, 1344310030u, 1340675091u, 1337049980u,
  1333434672u, 1329829140u, 1326233356u, 1322647296u, 1319070932u, 1315504238u, 1311947188u, 1308399756u,
  1304861917u, 1301333643u, 1297814910u, 1294305692u, 1290805962u, 1287315695u, 1283834865u, 1280363448u,
  1276901417u, 1273448747u, 1270005413u, 1266571390u, 1263146652u, 1259731174u, 1256324931u, 1252927899u,
  1249540052u, 1246161366u, 1242791816u, 1239431376u, 1236080024u, 1232737732u, 1229404479u, 1226080238u,
  1222764986u, 1219458698u, 1216161350u, 1212872918u, 1209593378u, 1206322705u, 1203060876u, 1199807867u,
  1196563654u, 1193328213u, 1190101520u, 1186883552u, 1183674286u, 1180473697u, 1177281762u, 1174098458u,
  1170923762u, 1167757650u, 1164600099u, 1161451085u, 1158310587u, 1155178580u, 1152055042u, 1148939949u,
  1145833280u, 1142735011u, 1139645120u, 1136563583u, 1133490379u, 1130425485u, 1127368878u, 1124320536u,
  1121280436u, 1118248556u, 1115224875u, 1112209370u, 1109202018u, 1106202798u, 1103211687u, 1100228665u,
  1097253708u, 1094286796u, 1091327906u, 1088377016u, 1085434106u, 1082499153u, 1079572136u, 1076653033u};
constexpr unsigned long long KZ_LN2 = 2977044472ull;                                  
constexpr unsigned KZ_K_LOG2E16 = 0x3db8aa3bu;                                         

__device__ __forceinline__ unsigned long long kz_rne(unsigned long long v, int s) {   
  if (s >= 62) return 0ull;
  if (s <= 0) return v;
  const unsigned long long q = v >> s, rem = v & ((1ull << s) - 1ull), half = 1ull << (s - 1);
  return q + ((rem > half || (rem == half && (q & 1ull))) ? 1ull : 0ull);
}
__device__ __forceinline__ unsigned kz_exp2i(float x, const unsigned* tab) {           
  const unsigned b = __float_as_uint(x) ^ 0x80000000u; const int e8 = (b >> 23) & 255;   
  unsigned long long xf = 0ull;
  if (e8 > 0) {
    const int sh = e8 - 118; if (sh > 20) return 0u;
    const unsigned long long mu = (b & 0x7FFFFFu) | 0x800000u;
    xf = sh >= 0 ? (mu << sh) : (-sh >= 32 ? 0ull : (mu >> -sh));
    if (xf >= (40ull << 32)) return 0u;
  }
  const unsigned n = (unsigned)(xf >> 32), fr = (unsigned)xf, i = fr >> 24, r = fr & 0xFFFFFFu;
  const unsigned long long q1 = ((unsigned long long)r * KZ_LN2) >> 33;
  const unsigned long long corr = (1ull << 31) - q1 + ((q1 * q1) >> 32);
  const unsigned long long ep = ((unsigned long long)tab[i] * corr) >> 31;
  const unsigned long long p = kz_rne(ep, (int)n + 7);
  return p > 0xFFFFFFull ? 0xFFFFFFu : (unsigned)p;
}



__device__ __forceinline__ unsigned kz_exp2b(float x, const unsigned* tab) {
  const float u = -x; const bool dead = !(u < 40.f); const float uc = dead ? 0.f : u;
  const unsigned n = (unsigned)__float2int_rz(uc);
  const float f = __fsub_rn(uc, (float)n);
  const unsigned fr = __float2uint_rz(__fmul_rn(f, 4294967296.f));
  const unsigned i = fr >> 24, r = fr & 0xFFFFFFu;
  const unsigned q1 = __umulhi(r, (unsigned)KZ_LN2) >> 1;
  const unsigned corr = 0x80000000u - q1 + __umulhi(q1, q1);
  const unsigned T = tab[i], lo = T * corr, hi = __umulhi(T, corr);
  const unsigned ep = (hi << 1) | (lo >> 31);
  const int sh = (int)n + 7; const int sc = sh > 31 ? 31 : sh;
  const unsigned q = ep >> sc, rem = ep & ((1u << sc) - 1u), half = 1u << (sc - 1);
  unsigned P = q + (unsigned)((rem > half) | ((rem == half) & (q & 1u)));
  P = sh > 31 ? 0u : P; P = P > 0xFFFFFFu ? 0xFFFFFFu : P;
  return dead ? 0u : P;
}

__device__ __forceinline__ unsigned kz_wround(unsigned P, unsigned M, int s) {
  const unsigned long long v = (unsigned long long)P * M;
  const int sc = s < 1 ? 1 : (s > 63 ? 63 : s);
  const unsigned long long q = v >> sc, rem = v & ((1ull << sc) - 1ull), half = 1ull << (sc - 1);
  const unsigned long long r = q + (unsigned long long)((rem > half) | ((rem == half) & (q & 1ull)));
  return s >= 62 ? 0u : (unsigned)r;
}
__device__ __forceinline__ void kz_vsplit(float s, unsigned& M, int& e) {             
  const unsigned b = __float_as_uint(s); const int e8 = (b >> 23) & 255;
  M = e8 ? ((b & 0x7FFFFFu) | 0x800000u) : (b & 0x7FFFFFu); e = e8 ? e8 - 126 : -125;
}
__device__ __forceinline__ void kz_mma_s8(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void kz_mma_u8(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void kz_cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void kz_cp4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void kz_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void kz_waitn() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ unsigned kz_ld16(const uint8_t* p) { return *reinterpret_cast<const unsigned short*>(p); }
__device__ __forceinline__ unsigned kz_ld32(const uint8_t* p) { return *reinterpret_cast<const unsigned*>(p); }

__device__ __forceinline__ int kz_qi(unsigned short bfbits, float sc) { return __float2int_rn(__fmul_rn(__uint_as_float((unsigned)bfbits << 16), sc)); }
}  

struct Q35KzArgs { Q35AttnArgs a; const uint8_t* wk; };

template <int NS>
__global__ void __launch_bounds__(192, 1) q35_kz_kernel(Q35KzArgs U) {
  constexpr int TPC = 16, NST = NS, G = Q35_NH / Q35_KVH, D = Q35_HD;
  const Q35AttnArgs& A = U.a;
  extern __shared__ __align__(16) uint8_t ksm[];
  __shared__ int s_lo, s_hi;
  const int t0 = blockIdx.x * TPC, split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gi = lane >> 2, tig = lane & 3;
  const bool early = A.early != 0;
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  int tk[2], hd[2], lo[2], ph[2], own[2], len[2]; bool self_[2], vrow[2];
  bool need = false;
  #pragma unroll
  for (int h = 0; h < 2; ++h) {                                                          
    const int i = 16 * warp + gi + 8 * h; tk[h] = t0 + i / G; hd[h] = kvh * G + i % G; vrow[h] = i < G * TPC && tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    len[h] = vrow[h] ? base + dp + 1 : 0; own[h] = base + tk[h];
    const int S = (A.splits_short > 0 && len[h] < A.long_len) ? A.splits_short : A.splits;
    need |= vrow[h] && split < S;
    const int per = (len[h] + S - 1) / S;
    lo[h] = split * per; const int hi = min(len[h], lo[h] + per);
    ph[h] = max(lo[h], min(hi, len[h] - 1));
    self_[h] = vrow[h] && lo[h] <= len[h] - 1 && len[h] - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  __syncthreads();
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / KZ_KT) * KZ_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + KZ_KT - 1) / KZ_KT : 0;
  uint8_t* const qsm = ksm;                                                              
  uint8_t* const st8 = ksm + KZ_QM;                                                      
  float* const ssc = reinterpret_cast<float*>(st8 + (size_t)NST * 2 * KZ_KT * KZ_RS);    
  uint8_t* const vts = reinterpret_cast<uint8_t*>(ssc + NST * 2 * KZ_KT);                
  float* const kfk = reinterpret_cast<float*>(vts + (size_t)D * KZ_VS);                  
  unsigned* const vMk = reinterpret_cast<unsigned*>(kfk + KZ_KT); int* const vek = reinterpret_cast<int*>(vMk + KZ_KT);
  unsigned* const e2t = reinterpret_cast<unsigned*>(vek + KZ_KT);                       
  for (int i = threadIdx.x; i < 256; i += blockDim.x) e2t[i] = kz_exp2t[i];
  __syncthreads();
  const int8_t* const kg8 = A.k8 + (size_t)kvh * A.max_ctx * D; const int8_t* const vg8 = A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = A.ks + (size_t)kvh * A.max_ctx; const float* const vsg = A.vs + (size_t)kvh * A.max_ctx;
  auto load_tile = [&](int t, int stage, int part = 0) {
    const int k0 = tbeg + t * KZ_KT;
    auto mine = [&](int kidx) { return part == 0 || (part == 1 ? kidx < base : kidx >= base); };
    uint8_t* const sk = st8 + (size_t)stage * 2 * KZ_KT * KZ_RS;
    for (int c = threadIdx.x; c < 2 * KZ_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (KZ_KT * (D / 16)), rem = c - kv * (KZ_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      uint8_t* const dst = sk + (size_t)kv * KZ_KT * KZ_RS + key * KZ_RS + ch * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) kz_cp16(dst, (kv ? vg8 : kg8) + (size_t)kidx * D + ch * 16); }
      else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * KZ_KT; c += blockDim.x) {
      const int kv = c / KZ_KT, key = c % KZ_KT, kidx = k0 + key;
      float* const dst = ssc + (stage * 2 + kv) * KZ_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) kz_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (part != 2) *dst = 0.f;
    }
  };
  if constexpr (NST >= 2) {
    #pragma unroll
    for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, early ? 1 : 0); kz_commit(); }
    if (early) { pdl_wait(); pdl_trigger();
      #pragma unroll
      for (int p = 0; p < NST - 1; ++p) { if (p < ntiles) load_tile(p, p, 2); kz_commit(); } }
  }
  
  
  unsigned qh[8][4]; float cq[2]; int eq[2];
  {
    const unsigned short* q0 = A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + hd[0]) * D;
    const unsigned short* q1 = A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + hd[1]) * D;
    unsigned mx[2] = {0u, 0u};
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks)
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int d = 16 * ks + 8 * j + 2 * tig + x;
          mx[0] = max(mx[0], vrow[0] ? ((unsigned)q0[d] & 0x7fffu) : 0u); mx[1] = max(mx[1], vrow[1] ? ((unsigned)q1[d] & 0x7fffu) : 0u);
        }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = max(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1)); mx[h] = max(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      const int e8 = (int)(mx[h] >> 7);                                                  
      eq[h] = mx[h] == 0u ? 0 : (e8 > 0 ? e8 - 126 : -125); cq[h] = kz_p2(eq[h] - 22);
    }
    const float sc0 = kz_p2(22 - eq[0]), sc1 = kz_p2(22 - eq[1]);
    uint8_t* const qm = qsm + (size_t)(warp * 2 + 0) * 8 * 16 * 32; uint8_t* const ql = qsm + (size_t)(warp * 2 + 1) * 8 * 16 * 32;
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)
      #pragma unroll
      for (int r = 0; r < 4; ++r) {                                                      
        const int h = r & 1, half = r >> 1; const unsigned short* q = h ? q1 : q0; const bool v = vrow[h]; const float sc = h ? sc1 : sc0;
        unsigned hi = 0u, mid = 0u, lw = 0u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = 32 * kk + 16 * half + 8 * (i >> 1) + 2 * tig + (i & 1);
          const int qi = v ? kz_qi(q[d], sc) : 0;
          hi |= ((unsigned)(qi >> 16) & 0xffu) << (8 * i); mid |= ((unsigned)(qi >> 8) & 0xffu) << (8 * i); lw |= ((unsigned)qi & 0xffu) << (8 * i);
        }
        qh[kk][r] = hi;
        const int row = gi + 8 * h, col = 16 * half + 4 * tig;
        *reinterpret_cast<unsigned*>(qm + ((size_t)kk * 16 + row) * 32 + col) = mid; *reinterpret_cast<unsigned*>(ql + ((size_t)kk * 16 + row) * 32 + col) = lw;
      }
  }
  __syncwarp();
  const bool busy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);
  float o[32][4];
  #pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  int RN[2] = {KZ_NONE, KZ_NONE}, RD[2] = {KZ_NONE, KZ_NONE}; float aD[2] = {0.f, 0.f};
  const uint8_t* const qmw = qsm + (size_t)(warp * 2 + 0) * 8 * 16 * 32; const uint8_t* const qlw = qsm + (size_t)(warp * 2 + 1) * 8 * 16 * 32;
  for (int t = 0; t < ntiles; ++t) {
    if constexpr (NST >= 2) { if (t + NST - 1 < ntiles) load_tile(t + NST - 1, (t + NST - 1) % NST); kz_commit(); kz_waitn<NST - 1>(); }
    else { load_tile(t, 0); kz_commit(); kz_waitn<0>(); }
    __syncthreads();
    const int stg = NST >= 2 ? t % NST : 0;
    const uint8_t* const skk = st8 + (size_t)stg * 2 * KZ_KT * KZ_RS; const uint8_t* const svv = skk + (size_t)KZ_KT * KZ_RS;
    for (int u = threadIdx.x; u < (D / 4) * (KZ_KT / 4); u += blockDim.x) {               
      const int db = u >> 3, pb = u & 7, half = pb >> 2, tt = pb & 3;
      unsigned r[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) r[i] = kz_ld32(svv + (size_t)(16 * half + 8 * (i >> 1) + 2 * tt + (i & 1)) * KZ_RS + 4 * db);
      const unsigned a01 = __byte_perm(r[0], r[1], 0x5140), b01 = __byte_perm(r[0], r[1], 0x7362);
      const unsigned a23 = __byte_perm(r[2], r[3], 0x5140), b23 = __byte_perm(r[2], r[3], 0x7362);
      uint8_t* const dst = vts + (size_t)(4 * db) * KZ_VS + 4 * pb;
      *reinterpret_cast<unsigned*>(dst) = __byte_perm(a01, a23, 0x5410); *reinterpret_cast<unsigned*>(dst + KZ_VS) = __byte_perm(a01, a23, 0x7632);
      *reinterpret_cast<unsigned*>(dst + 2 * KZ_VS) = __byte_perm(b01, b23, 0x5410); *reinterpret_cast<unsigned*>(dst + 3 * KZ_VS) = __byte_perm(b01, b23, 0x7632);
    }
    if (threadIdx.x < KZ_KT) {
      const float* const sc = ssc + stg * 2 * KZ_KT;
      kfk[threadIdx.x] = __fmul_rn(sc[threadIdx.x], __uint_as_float(KZ_K_LOG2E16));
      unsigned Mv; int ev; kz_vsplit(sc[KZ_KT + threadIdx.x], Mv, ev); vMk[threadIdx.x] = Mv; vek[threadIdx.x] = ev;
    }
    __syncthreads();
    if (busy) {
      const int kt0 = tbeg + t * KZ_KT;
      float tv[4][4]; bool vk[4][4];
      float mxv[2] = {-INFINITY, -INFINITY};
      #pragma unroll
      for (int hf = 0; hf < 2; ++hf) {                                                  
        int ac[3][2][4];
        #pragma unroll
        for (int s3 = 0; s3 < 3; ++s3)
          #pragma unroll
          for (int j = 0; j < 2; ++j) ac[s3][j][0] = ac[s3][j][1] = ac[s3][j][2] = ac[s3][j][3] = 0;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const uint8_t* const qm = qmw + (size_t)kk * 16 * 32; const uint8_t* const ql = qlw + (size_t)kk * 16 * 32;
          const unsigned am[4] = {kz_ld32(qm + gi * 32 + 4 * tig), kz_ld32(qm + (gi + 8) * 32 + 4 * tig), kz_ld32(qm + gi * 32 + 16 + 4 * tig), kz_ld32(qm + (gi + 8) * 32 + 16 + 4 * tig)};
          const unsigned al[4] = {kz_ld32(ql + gi * 32 + 4 * tig), kz_ld32(ql + (gi + 8) * 32 + 4 * tig), kz_ld32(ql + gi * 32 + 16 + 4 * tig), kz_ld32(ql + (gi + 8) * 32 + 16 + 4 * tig)};
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint8_t* const kr = skk + (size_t)(8 * (2 * hf + j) + gi) * KZ_RS + 32 * kk;
            const unsigned b0 = kz_ld16(kr + 2 * tig) | (kz_ld16(kr + 8 + 2 * tig) << 16), b1 = kz_ld16(kr + 16 + 2 * tig) | (kz_ld16(kr + 24 + 2 * tig) << 16);
            kz_mma_s8(ac[0][j], qh[kk], b0, b1); kz_mma_u8(ac[1][j], am, b0, b1); kz_mma_u8(ac[2][j], al, b0, b1);
          }
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j)
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int nb = 2 * hf + j, h = c >> 1, kl = 8 * nb + 2 * tig + (c & 1), key = kt0 + kl;
            const long long S = ((long long)ac[0][j][c] << 16) + ((long long)ac[1][j][c] << 8) + (long long)ac[2][j][c];
            vk[nb][c] = vrow[h] && key >= lo[h] && key < ph[h];
            tv[nb][c] = vk[nb][c] ? __fmul_rn(__ll2float_rn(S), __fmul_rn(kfk[kl], cq[h])) : -INFINITY;
            mxv[h] = fmaxf(mxv[h], tv[nb][c]);
          }
      }
      int Eh[2], Fh[2]; unsigned Bh[2];
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        mxv[h] = fmaxf(mxv[h], __shfl_xor_sync(0xffffffffu, mxv[h], 1)); mxv[h] = fmaxf(mxv[h], __shfl_xor_sync(0xffffffffu, mxv[h], 2));
        Eh[h] = mxv[h] > -INFINITY ? (int)ceilf(mxv[h]) : 0;
      }
      unsigned pw[4][4]; int fl[2] = {-100000, -100000};
      #pragma unroll
      for (int nb = 0; nb < 4; ++nb)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int h = c >> 1, kl = 8 * nb + 2 * tig + (c & 1);
          pw[nb][c] = vk[nb][c] ? kz_exp2i(__fsub_rn(tv[nb][c], (float)Eh[h]), e2t) : 0u;
          if (vk[nb][c]) fl[h] = max(fl[h], vek[kl]);
        }
      #pragma unroll
      for (int h = 0; h < 2; ++h) { fl[h] = max(fl[h], __shfl_xor_sync(0xffffffffu, fl[h], 1)); fl[h] = max(fl[h], __shfl_xor_sync(0xffffffffu, fl[h], 2)); Fh[h] = fl[h]; Bh[h] = 0u; }
      #pragma unroll
      for (int nb = 0; nb < 4; ++nb)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int h = c >> 1, kl = 8 * nb + 2 * tig + (c & 1);
          Bh[h] += pw[nb][c];
          pw[nb][c] = vk[nb][c] ? (unsigned)kz_rne((unsigned long long)pw[nb][c] * vMk[kl], 24 + Fh[h] - vek[kl]) : 0u;   
        }
      #pragma unroll
      for (int h = 0; h < 2; ++h) { Bh[h] += __shfl_xor_sync(0xffffffffu, Bh[h], 1); Bh[h] += __shfl_xor_sync(0xffffffffu, Bh[h], 2); }
      unsigned wa[3][4];                                                                 
      #pragma unroll
      for (int s = 0; s < 3; ++s) {
        const int shf = 8 * s;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int h = r & 1, nb0 = 2 * (r >> 1);
          const unsigned x0 = (pw[nb0][2 * h] >> shf) & 0xffu, x1 = (pw[nb0][2 * h + 1] >> shf) & 0xffu;
          const unsigned x2 = (pw[nb0 + 1][2 * h] >> shf) & 0xffu, x3 = (pw[nb0 + 1][2 * h + 1] >> shf) & 0xffu;
          wa[s][r] = x0 | (x1 << 8) | (x2 << 16) | (x3 << 24);
        }
      }
      float sN[2]; bool onN[2];
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        onN[h] = false; sN[h] = 0.f;
        if (mxv[h] > -INFINITY) {                                                        
          float rs, vsN; bool use; kz_ref(RN[h], Eh[h] + Fh[h] - 24, rs, vsN, use);
          if (rs != 1.f) {
            #pragma unroll
            for (int n = 0; n < 32; ++n) { o[n][2 * h] = __fmul_rn(o[n][2 * h], rs); o[n][2 * h + 1] = __fmul_rn(o[n][2 * h + 1], rs); }
          }
          onN[h] = use; sN[h] = vsN;
          float rd, vsD; bool useD; kz_ref(RD[h], Eh[h] - 24, rd, vsD, useD);
          if (rd != 1.f) aD[h] = __fmul_rn(aD[h], rd);
          if (useD) aD[h] = __fadd_rn(aD[h], __fmul_rn(__uint2float_rn(Bh[h]), vsD));
        }
      }
      #pragma unroll
      for (int nb2 = 0; nb2 < 32; ++nb2) {
        const uint8_t* const vr = vts + (size_t)(8 * nb2 + gi) * KZ_VS;
        const unsigned b0 = kz_ld32(vr + 4 * tig), b1 = kz_ld32(vr + 16 + 4 * tig);
        int c2[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0}, c0[4] = {0, 0, 0, 0};
        kz_mma_u8(c2, wa[2], b0, b1); kz_mma_u8(c1, wa[1], b0, b1); kz_mma_u8(c0, wa[0], b0, b1);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int h = k >> 1;
          if (onN[h]) {
            const long long Av = ((long long)c2[k] << 16) + ((long long)c1[k] << 8) + (long long)c0[k];
            o[nb2][k] = __fadd_rn(o[nb2][k], __fmul_rn(__ll2float_rn(Av), sN[h]));
          }
        }
      }
    }
    __syncthreads();                                                                     
  }
  
  if (busy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool sf = self_[h];                                                          
      long long part = 0;
      const int row = gi + 8 * h;
      if (sf) {
        const int8_t* const kr = kg8 + (size_t)own[h] * D;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            const unsigned hi = qh[kk][2 * half + h], mid = kz_ld32(qmw + ((size_t)kk * 16 + row) * 32 + 16 * half + 4 * tig);
            const unsigned lw = kz_ld32(qlw + ((size_t)kk * 16 + row) * 32 + 16 * half + 4 * tig);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int d = 32 * kk + 16 * half + 8 * (i >> 1) + 2 * tig + (i & 1);
              const int qi = (int)(signed char)((hi >> (8 * i)) & 0xffu) * 65536 + (int)((mid >> (8 * i)) & 0xffu) * 256 + (int)((lw >> (8 * i)) & 0xffu);
              part += (long long)qi * (long long)kr[d];
            }
          }
      }
      unsigned plo = (unsigned)(unsigned long long)part, phi = (unsigned)((unsigned long long)part >> 32);
      #pragma unroll
      for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        const unsigned xlo = __shfl_xor_sync(0xffffffffu, plo, o2), xhi = __shfl_xor_sync(0xffffffffu, phi, o2);
        const unsigned long long sum = (((unsigned long long)phi << 32) | plo) + (((unsigned long long)xhi << 32) | xlo);
        plo = (unsigned)sum; phi = (unsigned)(sum >> 32);
      }
      if (sf) {
        const long long So = (long long)(((unsigned long long)phi << 32) | plo);
        const float kfo = __fmul_rn(ksg[own[h]], __uint_as_float(KZ_K_LOG2E16));
        const float to = __fmul_rn(__ll2float_rn(So), __fmul_rn(kfo, cq[h]));
        const int Eo = (int)ceilf(to); const unsigned Po = kz_exp2i(__fsub_rn(to, (float)Eo), e2t);
        unsigned Mo; int eo; kz_vsplit(vsg[own[h]], Mo, eo);
        const unsigned Wo = (unsigned)kz_rne((unsigned long long)Po * Mo, 24);
        float rs, vsN; bool use; kz_ref(RN[h], Eo + eo - 24, rs, vsN, use);
        const int8_t* const vr = vg8 + (size_t)own[h] * D;
        #pragma unroll
        for (int n = 0; n < 32; ++n) {
          if (rs != 1.f) { o[n][2 * h] = __fmul_rn(o[n][2 * h], rs); o[n][2 * h + 1] = __fmul_rn(o[n][2 * h + 1], rs); }
          if (use) {
            o[n][2 * h] = __fadd_rn(o[n][2 * h], __fmul_rn(__ll2float_rn((long long)Wo * (long long)vr[8 * n + 2 * tig]), vsN));
            o[n][2 * h + 1] = __fadd_rn(o[n][2 * h + 1], __fmul_rn(__ll2float_rn((long long)Wo * (long long)vr[8 * n + 2 * tig + 1]), vsN));
          }
        }
        float rd, vsD; bool useD; kz_ref(RD[h], Eo - 24, rd, vsD, useD);
        if (rd != 1.f) aD[h] = __fmul_rn(aD[h], rd);
        if (useD) aD[h] = __fadd_rn(aD[h], __fmul_rn(__uint2float_rn(Po), vsD));
      }
    }
  }
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h]) continue;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd[h];
    if (tig == 0) { A.pm[pidx] = __int_as_float(RN[h] == KZ_NONE ? (int)0x80008000u : (int)(((unsigned)RN[h] << 16) | ((unsigned)RD[h] & 0xffffu))); A.pl[pidx] = aD[h]; }
    float* const pa = A.pacc + pidx * D + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 32; ++n) *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(o[n][2 * h], o[n][2 * h + 1]);
  }
}









namespace {
constexpr int KR_VS = 32;
constexpr size_t KR_ST = (size_t)2 * 2 * KZ_KT * KZ_RS, KR_SC = (size_t)2 * 2 * KZ_KT * 4, KR_VT = (size_t)Q35_HD * KR_VS;
__host__ __device__ constexpr size_t kr_q(int rt) { return (size_t)rt * 3 * 8 * 16 * 32; }       
__host__ __device__ constexpr size_t kr_wf(int rt) { return (size_t)rt * 3 * 16 * 32; }          
__host__ __device__ constexpr size_t kr_x(int rt) { return (size_t)rt * 4 * 16 * 4; }            
constexpr size_t kr_smem(int rt) { return KR_ST + KR_SC + KR_VT + kr_q(rt) + kr_wf(rt) + 2 * kr_x(rt) + 2 * kr_x(rt) + 2 * kr_x(rt) + 256 * 4; }
__device__ __forceinline__ void kz_ldm4(unsigned* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ int kz_swz(int row) { return ((row >> 2) & 1) << 4; }       
__device__ __forceinline__ unsigned long long kz_shfl64(unsigned long long v, int o) {
  const unsigned lo = __shfl_xor_sync(0xffffffffu, (unsigned)v, o), hi = __shfl_xor_sync(0xffffffffu, (unsigned)(v >> 32), o);
  return ((unsigned long long)hi << 32) | lo;
}
}  

template <int RT>
__global__ void __launch_bounds__(128 * RT, 1) q35_kzs_kernel(Q35KzArgs U) {
  constexpr int G = Q35_NH / Q35_KVH, D = Q35_HD;
  const Q35AttnArgs& A = U.a;
  extern __shared__ __align__(16) uint8_t ksm[];
  __shared__ int s_lo, s_hi;
  const int split = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rt = warp >> 2, cg = warp & 3;
  const int gi = lane >> 2, tig = lane & 3, lm = lane >> 3, lr = lane & 7;
  const bool early = A.early != 0;
  if (!early) { pdl_wait(); pdl_trigger(); }
  const int base = *A.d_pos, M = A.M;
  int tk[2], lo[2], ph[2]; bool self_[2], vrow[2];
  bool need = false;
  #pragma unroll
  for (int h = 0; h < 2; ++h) {                                                          
    const int i = 16 * RT * blockIdx.x + 16 * rt + gi + 8 * h; tk[h] = i / G; vrow[h] = tk[h] < M;
    const int dp = vrow[h] ? (U.wk ? (int)U.wk[tk[h]] : tk[h]) : -1;
    const int len = vrow[h] ? base + dp + 1 : 0;
    const int S = (A.splits_short > 0 && len < A.long_len) ? A.splits_short : A.splits;
    need |= vrow[h] && split < S;
    const int per = (len + S - 1) / S;
    lo[h] = split * per; const int hi = min(len, lo[h] + per);
    ph[h] = max(lo[h], min(hi, len - 1));
    self_[h] = vrow[h] && lo[h] <= len - 1 && len - 1 < lo[h] + per;
  }
  if (!__syncthreads_or(need)) { if (early) { pdl_wait(); pdl_trigger(); } return; }
  uint8_t* const st8 = ksm;                                                              
  float* const ssc = reinterpret_cast<float*>(ksm + KR_ST);                              
  uint8_t* const vts = ksm + KR_ST + KR_SC;                                              
  uint8_t* const qsm = vts + KR_VT;                                                      
  uint8_t* const wf = qsm + kr_q(RT);                                                    
  float* const mxs = reinterpret_cast<float*>(wf + kr_wf(RT));                           
  int* const fxs = reinterpret_cast<int*>(mxs + RT * 4 * 16);                            
  long long* const sow = reinterpret_cast<long long*>(fxs + RT * 4 * 16);                
  unsigned long long* const bsx = reinterpret_cast<unsigned long long*>(sow + RT * 4 * 16);   
  unsigned* const e2t = reinterpret_cast<unsigned*>(bsx + RT * 4 * 16);
  if (threadIdx.x == 0) { s_lo = 0x7fffffff; s_hi = 0; }
  for (int i = threadIdx.x; i < 256; i += blockDim.x) e2t[i] = kz_exp2t[i];
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) if (vrow[h] && ph[h] > lo[h]) { atomicMin(&s_lo, lo[h]); atomicMax(&s_hi, ph[h]); }
  const int8_t* const kg8 = A.k8 + (size_t)kvh * A.max_ctx * D; const int8_t* const vg8 = A.v8 + (size_t)kvh * A.max_ctx * D;
  const float* const ksg = A.ks + (size_t)kvh * A.max_ctx; const float* const vsg = A.vs + (size_t)kvh * A.max_ctx;
  __syncthreads();                                                                       
  const int klo = s_lo, khi = s_hi;
  const int tbeg = khi > klo ? (klo / KZ_KT) * KZ_KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + KZ_KT - 1) / KZ_KT : 0;
  const bool rbusy = __any_sync(0xffffffffu, vrow[0] || vrow[1]);                      
  const uint8_t* const qrt = qsm + (size_t)rt * 3 * 8 * 16 * 32;
  
  auto load_step = [&](int st, int stage, int part = 0) {
    const bool withv = st >= ntiles; const int k0 = tbeg + (withv ? st - ntiles : st) * KZ_KT;
    auto mine = [&](int kidx) { return part == 0 || (part == 1 ? kidx < base : kidx >= base); };
    uint8_t* const sk = st8 + (size_t)stage * 2 * KZ_KT * KZ_RS;
    const int nkv = withv ? 2 : 1;
    for (int c = threadIdx.x; c < nkv * KZ_KT * (D / 16); c += blockDim.x) {
      const int kv = c / (KZ_KT * (D / 16)), rem = c - kv * (KZ_KT * (D / 16)), key = rem / (D / 16), ch = rem - key * (D / 16);
      uint8_t* const dst = sk + (size_t)kv * KZ_KT * KZ_RS + key * KZ_RS + ch * 16;
      const int kidx = k0 + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) kz_cp16(dst, (kv ? vg8 : kg8) + (size_t)kidx * D + ch * 16); }
      else if (part != 2) *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int c = threadIdx.x; c < 2 * KZ_KT; c += blockDim.x) {
      const int kv = c / KZ_KT, key = c % KZ_KT, kidx = k0 + key;
      float* const dst = ssc + (stage * 2 + kv) * KZ_KT + key;
      if (kidx >= klo && kidx < khi) { if (mine(kidx)) kz_cp4(dst, (kv ? vsg : ksg) + kidx); } else if (part != 2) *dst = 0.f;
    }
  };
  const int nsteps = 2 * ntiles;
  if (nsteps > 0) load_step(0, 0, early ? 1 : 0);                                        
  kz_commit();
  if (early) { pdl_wait(); pdl_trigger(); }                                              
  float cq[2];
  {                                                                                      
    const unsigned short* q0 = A.q16 + ((size_t)(vrow[0] ? tk[0] : 0) * Q35_NH + kvh * G + (16 * RT * blockIdx.x + 16 * rt + gi) % G) * D;
    const unsigned short* q1 = A.q16 + ((size_t)(vrow[1] ? tk[1] : 0) * Q35_NH + kvh * G + (16 * RT * blockIdx.x + 16 * rt + gi + 8) % G) * D;
    unsigned mx[2] = {0u, 0u};
    #pragma unroll
    for (int ks = 0; ks < 16; ++ks)
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int d = 16 * ks + 8 * j + 2 * tig + x;
          mx[0] = max(mx[0], vrow[0] ? ((unsigned)q0[d] & 0x7fffu) : 0u); mx[1] = max(mx[1], vrow[1] ? ((unsigned)q1[d] & 0x7fffu) : 0u);
        }
    float sc[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = max(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1)); mx[h] = max(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      const int e8 = (int)(mx[h] >> 7); const int eq = mx[h] == 0u ? 0 : (e8 > 0 ? e8 - 126 : -125);
      cq[h] = kz_p2(eq - 22); sc[h] = kz_p2(22 - eq);
    }
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int h = r & 1, half = r >> 1, kk = 2 * cg + j, row = gi + 8 * h; const unsigned short* q = h ? q1 : q0;
        unsigned b2 = 0u, b1 = 0u, b0 = 0u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = 32 * kk + 16 * half + 4 * tig + i;
          const int qi = vrow[h] ? kz_qi(q[d], sc[h]) : 0;
          b2 |= ((unsigned)(qi >> 16) & 0xffu) << (8 * i); b1 |= ((unsigned)(qi >> 8) & 0xffu) << (8 * i); b0 |= ((unsigned)qi & 0xffu) << (8 * i);
        }
        const size_t off = ((size_t)kk * 16 + row) * 32 + ((16 * half + 4 * tig) ^ kz_swz(row));
        *reinterpret_cast<unsigned*>(qsm + (size_t)(rt * 3 + 0) * 8 * 16 * 32 + off) = b2;
        *reinterpret_cast<unsigned*>(qsm + (size_t)(rt * 3 + 1) * 8 * 16 * 32 + off) = b1;
        *reinterpret_cast<unsigned*>(qsm + (size_t)(rt * 3 + 2) * 8 * 16 * 32 + off) = b0;
      }
  }
  __syncthreads();                                                                       
  
  if (rbusy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      long long part = 0;
      if (self_[h]) {
        const int8_t* const kr = kg8 + (size_t)(base + tk[h]) * D;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int kk = 2 * cg + j, row = gi + 8 * h; const size_t off = ((size_t)kk * 16 + row) * 32 + ((16 * half + 4 * tig) ^ kz_swz(row));
            const unsigned s0 = kz_ld32(qrt + off), s1 = kz_ld32(qrt + (size_t)8 * 16 * 32 + off), s2 = kz_ld32(qrt + (size_t)2 * 8 * 16 * 32 + off);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int d = 32 * kk + 16 * half + 4 * tig + i;
              const int qi = (int)(signed char)((s0 >> (8 * i)) & 0xffu) * 65536 + (int)((s1 >> (8 * i)) & 0xffu) * 256 + (int)((s2 >> (8 * i)) & 0xffu);
              part += (long long)qi * (long long)kr[d];
            }
          }
      }
      unsigned long long pu = (unsigned long long)part;
      pu += kz_shfl64(pu, 1); pu += kz_shfl64(pu, 2);
      if (tig == 0) sow[(rt * 4 + cg) * 16 + gi + 8 * h] = (long long)pu;
    }
  }
  __syncthreads();
  float tow[2] = {-INFINITY, -INFINITY}; int eow[2] = {-100000, -100000}; unsigned Mow[2] = {0u, 0u};
  if (rbusy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (!self_[h]) continue;
      long long So = 0;
      #pragma unroll
      for (int c2 = 0; c2 < 4; ++c2) So += sow[(rt * 4 + c2) * 16 + gi + 8 * h];
      const int own = base + tk[h];
      const float kfo = __fmul_rn(ksg[own], __uint_as_float(KZ_K_LOG2E16));
      tow[h] = __fmul_rn(__ll2float_rn(So), __fmul_rn(kfo, cq[h]));
      kz_vsplit(vsg[own], Mow[h], eow[h]);
    }
  }
  if (early) { if (nsteps > 0) load_step(0, 0, 2); kz_commit(); }
  int acc[3][8][4];
  #pragma unroll
  for (int s3 = 0; s3 < 3; ++s3)
    #pragma unroll
    for (int n = 0; n < 8; ++n) acc[s3][n][0] = acc[s3][n][1] = acc[s3][n][2] = acc[s3][n][3] = 0;
  float pm1[2] = {-INFINITY, -INFINITY}; int pf1[2] = {-100000, -100000};
  int Er[2] = {KZ_NONE, KZ_NONE}, Fr[2] = {KZ_NONE, KZ_NONE}; unsigned long long Bl[2] = {0ull, 0ull};
  auto refs = [&](bool fromx) {                                                          
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      float cm = tow[h]; int cf = eow[h];
      if (fromx) {
        #pragma unroll
        for (int c2 = 0; c2 < 4; ++c2) { cm = fmaxf(cm, mxs[(rt * 4 + c2) * 16 + gi + 8 * h]); cf = max(cf, fxs[(rt * 4 + c2) * 16 + gi + 8 * h]); }
      }
      if (cm > -INFINITY) { Er[h] = (int)ceilf(cm); Fr[h] = cf; }
    }
  };
  for (int st = 0; st < nsteps; ++st) {
    kz_waitn<0>();
    __syncthreads();                                                                     
    if (st + 1 < nsteps) load_step(st + 1, (st + 1) & 1);                                
    kz_commit();
    const bool p2 = st >= ntiles; const int t = p2 ? st - ntiles : st, stg = st & 1, kt0 = tbeg + t * KZ_KT;
    const uint8_t* const skk = st8 + (size_t)stg * 2 * KZ_KT * KZ_RS; const uint8_t* const svv = skk + (size_t)KZ_KT * KZ_RS;
    const float* const sks = ssc + stg * 2 * KZ_KT;
    if (p2) {
      if (st == ntiles && rbusy) refs(true);
      for (int u = threadIdx.x; u < (D / 4) * (KZ_KT / 4); u += blockDim.x) {             
        const int db = u >> 3, kb = u & 7;
        unsigned r[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) r[i] = kz_ld32(svv + (size_t)(4 * kb + i) * KZ_RS + 4 * db);
        const unsigned a01 = __byte_perm(r[0], r[1], 0x5140), b01 = __byte_perm(r[0], r[1], 0x7362);
        const unsigned a23 = __byte_perm(r[2], r[3], 0x5140), b23 = __byte_perm(r[2], r[3], 0x7362);
        const unsigned o4[4] = {__byte_perm(a01, a23, 0x5410), __byte_perm(a01, a23, 0x7632), __byte_perm(b01, b23, 0x5410), __byte_perm(b01, b23, 0x7632)};
        #pragma unroll
        for (int j = 0; j < 4; ++j) { const int dr = 4 * db + j; *reinterpret_cast<unsigned*>(vts + (size_t)dr * KR_VS + ((4 * kb) ^ kz_swz(dr))) = o4[j]; }
      }
    }
    if (rbusy) {                                                                         
      int ac[2][3][4];
      #pragma unroll
      for (int x = 0; x < 2; ++x)
        #pragma unroll
        for (int s3 = 0; s3 < 3; ++s3) ac[x][s3][0] = ac[x][s3][1] = ac[x][s3][2] = ac[x][s3][3] = 0;
      #pragma unroll
      for (int k2 = 0; k2 < 4; ++k2) {
        unsigned kb[4];
        kz_ldm4(kb, skk + (size_t)(8 * cg + lr) * KZ_RS + 32 * (2 * k2 + (lm >> 1)) + 16 * (lm & 1));
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int kk = 2 * k2 + x, row = 8 * (lm & 1) + lr;
          unsigned af[3][4];
          #pragma unroll
          for (int s3 = 0; s3 < 3; ++s3) kz_ldm4(af[s3], qrt + ((size_t)(s3 * 8 + kk) * 16 + row) * 32 + ((16 * (lm >> 1)) ^ kz_swz(row)));
          kz_mma_s8(ac[x][0], af[0], kb[2 * x], kb[2 * x + 1]); kz_mma_u8(ac[x][1], af[1], kb[2 * x], kb[2 * x + 1]); kz_mma_u8(ac[x][2], af[2], kb[2 * x], kb[2 * x + 1]);
        }
      }
      unsigned vM[2]; int ve[2];
      #pragma unroll
      for (int x = 0; x < 2; ++x) kz_vsplit(sks[KZ_KT + 8 * cg + 2 * tig + x], vM[x], ve[x]);
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        float tv[2]; bool vk[2];
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int c = 2 * h + x, kl = 8 * cg + 2 * tig + x, key = kt0 + kl;
          const long long S = (((long long)ac[0][0][c] + ac[1][0][c]) << 16) + (((long long)ac[0][1][c] + ac[1][1][c]) << 8) + ((long long)ac[0][2][c] + ac[1][2][c]);
          vk[x] = vrow[h] && key >= lo[h] && key < ph[h];
          tv[x] = vk[x] ? __fmul_rn(__ll2float_rn(S), __fmul_rn(__fmul_rn(sks[kl], __uint_as_float(KZ_K_LOG2E16)), cq[h])) : -INFINITY;
        }
        if (!p2) {
          #pragma unroll
          for (int x = 0; x < 2; ++x) { pm1[h] = fmaxf(pm1[h], tv[x]); pf1[h] = max(pf1[h], vk[x] ? ve[x] : -100000); }
        } else {
          unsigned Wx[2];
          #pragma unroll
          for (int x = 0; x < 2; ++x) {                                                
            const unsigned P = kz_exp2b(__fsub_rn(tv[x], (float)Er[h]), e2t);
            Wx[x] = kz_wround(P, vM[x], 24 + Fr[h] - ve[x]);
            Bl[h] += P;
          }
          const int row = gi + 8 * h;
          #pragma unroll
          for (int s3 = 0; s3 < 3; ++s3) {
            const int sh = 16 - 8 * s3;
            *reinterpret_cast<unsigned short*>(wf + ((size_t)(rt * 3 + s3) * 16 + row) * 32 + ((8 * cg + 2 * tig) ^ kz_swz(row))) =
                (unsigned short)(((Wx[0] >> sh) & 0xffu) | (((Wx[1] >> sh) & 0xffu) << 8));
          }
        }
      }
    }
    if (!p2 && st == ntiles - 1 && rbusy) {                                              
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        float m = pm1[h]; int f = pf1[h];
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        f = max(f, __shfl_xor_sync(0xffffffffu, f, 1)); f = max(f, __shfl_xor_sync(0xffffffffu, f, 2));
        if (tig == 0) { mxs[(rt * 4 + cg) * 16 + gi + 8 * h] = m; fxs[(rt * 4 + cg) * 16 + gi + 8 * h] = f; }
      }
    }
    if (p2) {
      __syncthreads();                                                                   
      if (rbusy) {
        unsigned wa[3][4];
        #pragma unroll
        for (int s3 = 0; s3 < 3; ++s3) { const int row = 8 * (lm & 1) + lr; kz_ldm4(wa[s3], wf + ((size_t)(rt * 3 + s3) * 16 + row) * 32 + ((16 * (lm >> 1)) ^ kz_swz(row))); }
        #pragma unroll
        for (int n2 = 0; n2 < 4; ++n2) {
          unsigned vb[4];
          const int dr = 64 * cg + 8 * (2 * n2 + (lm >> 1)) + lr;
          kz_ldm4(vb, vts + (size_t)dr * KR_VS + ((16 * (lm & 1)) ^ kz_swz(dr)));
          #pragma unroll
          for (int x = 0; x < 2; ++x) {
            const int n = 2 * n2 + x;
            kz_mma_u8(acc[0][n], wa[0], vb[2 * x], vb[2 * x + 1]); kz_mma_u8(acc[1][n], wa[1], vb[2 * x], vb[2 * x + 1]); kz_mma_u8(acc[2][n], wa[2], vb[2 * x], vb[2 * x + 1]);
          }
        }
      }
    }
  }
  if (ntiles == 0 && rbusy) refs(false);                                                 
  
  if (rbusy) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      unsigned long long b = Bl[h]; b += kz_shfl64(b, 1); b += kz_shfl64(b, 2);
      if (tig == 0) bsx[(rt * 4 + cg) * 16 + gi + 8 * h] = b;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!vrow[h] || !rbusy) continue;
    const int i = 16 * RT * blockIdx.x + 16 * rt + gi + 8 * h, hd = kvh * G + i % G;
    const size_t pidx = ((size_t)tk[h] * A.splits + split) * Q35_NH + hd;
    unsigned Wo = 0u; unsigned long long Bt = 0ull;
    #pragma unroll
    for (int c2 = 0; c2 < 4; ++c2) Bt += bsx[(rt * 4 + c2) * 16 + gi + 8 * h];
    if (self_[h]) {
      const unsigned Po = kz_exp2b(__fsub_rn(tow[h], (float)Er[h]), e2t);
      Wo = kz_wround(Po, Mow[h], 24 + Fr[h] - eow[h]); Bt += Po;
    }
    if (cg == 0 && tig == 0) {
      A.pm[pidx] = __int_as_float(Er[h] == KZ_NONE ? (int)0x80008000u : (int)(((unsigned)(Er[h] + Fr[h] - 24) << 16) | ((unsigned)(Er[h] - 24) & 0xffffu)));
      A.pl[pidx] = __ull2float_rn(Bt);
    }
    float* const pa = A.pacc + pidx * D + 64 * cg + 2 * tig;
    const int8_t* const vr = vg8 + (size_t)(base + tk[h]) * D + 64 * cg + 2 * tig;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      float vv[2];
      #pragma unroll
      for (int x = 0; x < 2; ++x) {
        const int k = 2 * h + x;
        long long a = ((long long)acc[0][n][k] << 16) + ((long long)acc[1][n][k] << 8) + (long long)acc[2][n][k];
        if (self_[h]) a += (long long)Wo * (long long)vr[8 * n + x];
        vv[x] = __ll2float_rn(a);
      }
      *reinterpret_cast<float2*>(pa + 8 * n) = make_float2(vv[0], vv[1]);
    }
  }
}

void launch_q35_kz(const Q35AttnArgs& A, const uint8_t* wk, cudaStream_t st) {
  static bool attr = false;
  static const bool v1 = [] { const char* e = std::getenv("YSKN_Q35_KZV1"); return e && e[0] == '1'; }();   
  static const int rt = [] { const char* e = std::getenv("YSKN_Q35_KZRT"); return e && e[0] == '2' ? 2 : 3; }();   
  if (!attr) { cudaFuncSetAttribute(q35_kz_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kz_smem(2));
               cudaFuncSetAttribute(q35_kzs_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kr_smem(2));
               cudaFuncSetAttribute(q35_kzs_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kr_smem(3)); attr = true; }
  if (!A.k8) throw std::runtime_error("PMATH needs the KV8 cache");
  if (v1) { const dim3 grid((A.M + 15) / 16, A.splits, Q35_KVH); launch_k(q35_kz_kernel<2>, grid, dim3(192), kz_smem(2), st, Q35KzArgs{A, wk}); return; }
  const int rows = 16 * rt; const dim3 grid((A.M * (Q35_NH / Q35_KVH) + rows - 1) / rows, A.splits, Q35_KVH);
  if (rt == 2) launch_k(q35_kzs_kernel<2>, grid, dim3(256), kr_smem(2), st, Q35KzArgs{A, wk});
  else launch_k(q35_kzs_kernel<3>, grid, dim3(384), kr_smem(3), st, Q35KzArgs{A, wk});
}
}  
