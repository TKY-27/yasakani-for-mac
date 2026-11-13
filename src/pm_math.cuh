// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once







#include <cstdint>
namespace yskn {

__device__ __forceinline__ float pm_f(unsigned b) { return __uint_as_float(b); }

__device__ __forceinline__ float pm_rsqrt(float x) { return __fdiv_rn(1.f, __fsqrt_rn(x)); }

__device__ __forceinline__ float pm_expf(float x) {
  const unsigned ux = __float_as_uint(x), ax = ux & 0x7fffffffu; const int neg = (int)(ux >> 31);
  if (ax > 0x7f800000u) return __fadd_rn(x, x);                                  
  if (!neg && ax >= 0x42b17218u) return pm_f(0x7f800000u);                       
  if (ux > 0xc2cff1b5u) return 0.f;                                              
  const float LN2HI = pm_f(0x3f317200u), LN2LO = pm_f(0x35bfbe8eu);
  float hi = 0.f, lo = 0.f, r = x; int k = 0;
  if (ax > 0x3eb17218u) {                                                        
    if (ax < 0x3f851592u) { hi = __fsub_rn(x, neg ? -LN2HI : LN2HI); lo = neg ? -LN2LO : LN2LO; k = neg ? -1 : 1; }
    else { k = __float2int_rz(__fadd_rn(__fmul_rn(pm_f(0x3fb8aa3bu), x), neg ? -0.5f : 0.5f)); const float t = (float)k;
           hi = __fsub_rn(x, __fmul_rn(t, LN2HI)); lo = __fmul_rn(t, LN2LO); }
    r = __fsub_rn(hi, lo);
  } else if (ax < 0x39000000u) return __fadd_rn(1.f, x);                         
  const float t = __fmul_rn(r, r);
  const float c = __fsub_rn(r, __fmul_rn(t, __fadd_rn(pm_f(0x3e2aaa8fu), __fmul_rn(t, pm_f(0xbb355215u)))));
  if (k == 0) return __fsub_rn(1.f, __fsub_rn(__fdiv_rn(__fmul_rn(r, c), __fsub_rn(c, 2.f)), r));
  const float y = __fsub_rn(1.f, __fsub_rn(__fsub_rn(lo, __fdiv_rn(__fmul_rn(r, c), __fsub_rn(2.f, c))), hi));
  if (k >= -125) { if (k == 128) return __fmul_rn(__fmul_rn(y, 2.f), pm_f(0x7f000000u));
                   return __fmul_rn(y, pm_f((unsigned)(0x7f + k) << 23)); }
  return __fmul_rn(__fmul_rn(y, pm_f((unsigned)(0x7f + k + 100) << 23)), pm_f(0x0d800000u));   
}

__device__ __forceinline__ float pm_logf(float x) {
  unsigned ix = __float_as_uint(x); int k = 0;
  if (ix < 0x00800000u) { x = __fmul_rn(x, 33554432.f); k = -25; ix = __float_as_uint(x); }   
  k += (int)(ix >> 23) - 127; ix &= 0x007fffffu;
  const unsigned i = (ix + 0x4afb20u) & 0x800000u;                               
  const float m = pm_f(ix | (i ^ 0x3f800000u)); k += (int)(i >> 23);
  const float f = __fsub_rn(m, 1.f), s = __fdiv_rn(f, __fadd_rn(2.f, f)), z = __fmul_rn(s, s), w = __fmul_rn(z, z);
  const float t1 = __fmul_rn(w, __fadd_rn(pm_f(0x3ecccE13u), __fmul_rn(w, pm_f(0x3e789e26u))));
  const float t2 = __fmul_rn(z, __fadd_rn(pm_f(0x3f2aaaaau), __fmul_rn(w, pm_f(0x3e91e9eeu))));
  const float R = __fadd_rn(t2, t1), hfsq = __fmul_rn(__fmul_rn(0.5f, f), f), dk = (float)k;
  return __fsub_rn(__fmul_rn(dk, pm_f(0x3f317180u)),
                   __fsub_rn(__fsub_rn(hfsq, __fadd_rn(__fmul_rn(s, __fadd_rn(hfsq, R)), __fmul_rn(dk, pm_f(0x3717f7d1u)))), f));
}

__device__ __forceinline__ float pm_log1p_pos(float y) {
  const float u = __fadd_rn(1.f, y);
  const float c = y <= 1.f ? __fsub_rn(y, __fsub_rn(u, 1.f)) : __fsub_rn(1.f, __fsub_rn(u, y));
  return __fadd_rn(pm_logf(u), __fdiv_rn(c, u));
}
__device__ __forceinline__ float pm_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.f, pm_expf(-x))); }        
__device__ __forceinline__ float pm_sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, pm_expf(-x))); }   
__device__ __forceinline__ float pm_softplus(float x) { return x > 20.f ? x : pm_log1p_pos(pm_expf(x)); }    

__device__ __forceinline__ double pm_d(unsigned hi, unsigned lo) { return __hiloint2double((int)hi, (int)lo); }

__device__ __forceinline__ double pm_exp_d(double x) {
  const unsigned hxs = (unsigned)__double2hiint(x), lx = (unsigned)__double2loint(x); const int xsb = (int)(hxs >> 31);
  const unsigned hx = hxs & 0x7fffffffu;
  if (hx >= 0x40862E42u) {                                                        
    if (hx >= 0x7ff00000u) { if (((hx & 0xfffffu) | lx) != 0u) return __dadd_rn(x, x); return xsb == 0 ? x : 0.0; }
    if (x > pm_d(0x40862E42u, 0xFEFA39EFu)) return __longlong_as_double(0x7ff0000000000000ll);
    if (x < pm_d(0xc0874910u, 0xD52D3051u)) return 0.0;
  }
  const double LN2HI = pm_d(0x3fe62e42u, 0xfee00000u), LN2LO = pm_d(0x3dea39efu, 0x35793c76u);
  double hi = 0.0, lo = 0.0, r = x; int k = 0;
  if (hx > 0x3fd62e42u) {                                                         
    if (hx < 0x3FF0A2B2u) { hi = __dsub_rn(x, xsb ? -LN2HI : LN2HI); lo = xsb ? -LN2LO : LN2LO; k = 1 - xsb - xsb; }
    else { k = __double2int_rz(__dadd_rn(__dmul_rn(pm_d(0x3ff71547u, 0x652b82feu), x), xsb ? -0.5 : 0.5)); const double t = (double)k;
           hi = __dsub_rn(x, __dmul_rn(t, LN2HI)); lo = __dmul_rn(t, LN2LO); }
    r = __dsub_rn(hi, lo);
  } else if (hx < 0x3e300000u) return __dadd_rn(1.0, x);                          
  const double t = __dmul_rn(r, r);
  const double p = __dadd_rn(pm_d(0x3FC55555u, 0x5555553Eu), __dmul_rn(t, __dadd_rn(pm_d(0xBF66C16Cu, 0x16BEBD93u), __dmul_rn(t,
                   __dadd_rn(pm_d(0x3F11566Au, 0xAF25DE2Cu), __dmul_rn(t, __dadd_rn(pm_d(0xBEBBBD41u, 0xC5D26BF1u), __dmul_rn(t, pm_d(0x3E663769u, 0x72BEA4D0u)))))))));
  const double c = __dsub_rn(r, __dmul_rn(t, p));
  if (k == 0) return __dsub_rn(1.0, __dsub_rn(__ddiv_rn(__dmul_rn(r, c), __dsub_rn(c, 2.0)), r));
  const double y = __dsub_rn(1.0, __dsub_rn(__dsub_rn(lo, __ddiv_rn(__dmul_rn(r, c), __dsub_rn(2.0, c))), hi));
  if (k >= -1021) return __hiloint2double(__double2hiint(y) + k * 1048576, __double2loint(y));             
  return __dmul_rn(__hiloint2double(__double2hiint(y) + (k + 1000) * 1048576, __double2loint(y)), pm_d(0x01700000u, 0u));   
}

__device__ __forceinline__ double pm_log_d(double x) {
  int hx = __double2hiint(x), k = 0;
  if (hx < 0x00100000) { x = __dmul_rn(x, pm_d(0x43500000u, 0u)); k = -54; hx = __double2hiint(x); }   
  k += (hx >> 20) - 1023; hx &= 0x000fffff;
  const int i0 = (hx + 0x95f64) & 0x100000;
  x = __hiloint2double(hx | (i0 ^ 0x3ff00000), __double2loint(x)); k += i0 >> 20;  
  const double LN2HI = pm_d(0x3fe62e42u, 0xfee00000u), LN2LO = pm_d(0x3dea39efu, 0x35793c76u);
  const double f = __dsub_rn(x, 1.0);
  if ((0x000fffff & (2 + hx)) < 3) {                                              
    if (f == 0.0) { if (k == 0) return 0.0; const double dk = (double)k; return __dadd_rn(__dmul_rn(dk, LN2HI), __dmul_rn(dk, LN2LO)); }
    const double R = __dmul_rn(__dmul_rn(f, f), __dsub_rn(0.5, __dmul_rn(0.33333333333333333, f)));
    if (k == 0) return __dsub_rn(f, R);
    const double dk = (double)k; return __dsub_rn(__dmul_rn(dk, LN2HI), __dsub_rn(__dsub_rn(R, __dmul_rn(dk, LN2LO)), f));
  }
  const double s = __ddiv_rn(f, __dadd_rn(2.0, f)), dk = (double)k, z = __dmul_rn(s, s), w = __dmul_rn(z, z);
  int i = hx - 0x6147a; const int j = 0x6b851 - hx;
  const double t1 = __dmul_rn(w, __dadd_rn(pm_d(0x3FD99999u, 0x9997FA04u), __dmul_rn(w, __dadd_rn(pm_d(0x3FCC71C5u, 0x1D8E78AFu), __dmul_rn(w, pm_d(0x3FC39A09u, 0xD078C69Fu))))));
  const double t2 = __dmul_rn(z, __dadd_rn(pm_d(0x3FE55555u, 0x55555593u), __dmul_rn(w, __dadd_rn(pm_d(0x3FD24924u, 0x94229359u),
                    __dmul_rn(w, __dadd_rn(pm_d(0x3FC74664u, 0x96CB03DEu), __dmul_rn(w, pm_d(0x3FC2F112u, 0xDF3E5244u))))))));
  i |= j;
  const double R = __dadd_rn(t2, t1);
  if (i > 0) {
    const double hfsq = __dmul_rn(__dmul_rn(0.5, f), f);
    if (k == 0) return __dsub_rn(f, __dsub_rn(hfsq, __dmul_rn(s, __dadd_rn(hfsq, R))));
    return __dsub_rn(__dmul_rn(dk, LN2HI), __dsub_rn(__dsub_rn(hfsq, __dadd_rn(__dmul_rn(s, __dadd_rn(hfsq, R)), __dmul_rn(dk, LN2LO))), f));
  }
  if (k == 0) return __dsub_rn(f, __dmul_rn(s, __dsub_rn(f, R)));
  return __dsub_rn(__dmul_rn(dk, LN2HI), __dsub_rn(__dsub_rn(__dmul_rn(s, __dsub_rn(f, R)), __dmul_rn(dk, LN2LO)), f));
}



__device__ __forceinline__ double pm_ksin(double r) {
  const double z = __dmul_rn(r, r), v = __dmul_rn(z, r);
  const double R = __dadd_rn(pm_d(0x3F811111u, 0x1110F8A6u), __dmul_rn(z, __dadd_rn(pm_d(0xBF2A01A0u, 0x19C161D5u), __dmul_rn(z,
                   __dadd_rn(pm_d(0x3EC71DE3u, 0x57B1FE7Du), __dmul_rn(z, __dadd_rn(pm_d(0xBE5AE5E6u, 0x8A2B9CEBu), __dmul_rn(z, pm_d(0x3DE5D93Au, 0x5ACFD57Cu)))))))));
  return __dadd_rn(r, __dmul_rn(v, __dadd_rn(pm_d(0xBFC55555u, 0x55555549u), __dmul_rn(z, R))));
}
__device__ __forceinline__ double pm_kcos(double r) {
  const double z = __dmul_rn(r, r);
  const double R = __dmul_rn(z, __dadd_rn(pm_d(0x3FA55555u, 0x5555554Cu), __dmul_rn(z, __dadd_rn(pm_d(0xBF56C16Cu, 0x16C15177u), __dmul_rn(z,
                   __dadd_rn(pm_d(0x3EFA01A0u, 0x19CB1590u), __dmul_rn(z, __dadd_rn(pm_d(0xBE927E4Fu, 0x809C52ADu), __dmul_rn(z,
                   __dadd_rn(pm_d(0x3E21EE9Eu, 0xBDB4B1C4u), __dmul_rn(z, pm_d(0xBDA8FAE9u, 0xBE8838D4u))))))))))));
  const unsigned ix = (unsigned)__double2hiint(r) & 0x7fffffffu;
  const double zr = __dmul_rn(z, R);
  if (ix < 0x3FD33333u) return __dsub_rn(1.0, __dsub_rn(__dmul_rn(0.5, z), zr));        
  const double qx = ix > 0x3fe90000u ? 0.28125 : pm_d(ix - 0x00200000u, 0u);           
  return __dsub_rn(__dsub_rn(1.0, qx), __dsub_rn(__dsub_rn(__dmul_rn(0.5, z), qx), zr));
}
__device__ __forceinline__ void pm_sincos_f(float a, float& c, float& s) {
  const double x = (double)a;
  const int n = __double2int_rz(__dadd_rn(__dmul_rn(x, pm_d(0x3FE45F30u, 0x6DC9C883u)), 0.5));
  const double fn = (double)n;
  const double r = __dsub_rn(__dsub_rn(__dsub_rn(x, __dmul_rn(fn, pm_d(0x3FF921FBu, 0x54400000u))), __dmul_rn(fn, pm_d(0x3DD0B461u, 0x1A600000u))),
                             __dmul_rn(fn, pm_d(0x3BA3198Au, 0x2E037073u)));
  const double ks = pm_ksin(r), kc = pm_kcos(r);
  double cs, sn;
  switch (n & 3) { case 0: cs = kc; sn = ks; break; case 1: cs = -ks; sn = kc; break; case 2: cs = -kc; sn = -ks; break; default: cs = ks; sn = -kc; }
  c = __double2float_rn(cs); s = __double2float_rn(sn);
}
}  
