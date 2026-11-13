// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
namespace yskn {
inline uint64_t splitmix64(uint64_t& st) { st += 0x9E3779B97F4A7C15ull; uint64_t z = st; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
inline std::vector<int8_t> sign_stream(uint64_t seed, uint64_t offset, size_t n) {
  std::vector<int8_t> out; out.reserve(n); uint64_t st = seed;
  for (uint64_t k = 0; k < offset + n; ++k) { const uint64_t v = splitmix64(st); if (k >= offset) out.push_back((v >> 63) ? 1 : -1); }
  return out;
}



inline std::vector<int8_t> sign_stream_mulberry32(uint64_t seed, uint64_t offset, size_t n) {
  uint32_t s = uint32_t(seed) * 7919u + 13u; if (!s) s = 0x9e3779b9u;
  std::vector<int8_t> out; out.reserve(n);
  for (uint64_t k = 0; k < offset + n; ++k) {
    uint32_t t = (s += 0x6d2b79f5u); t = (t ^ (t >> 15)) * (t | 1u); t ^= t + (t ^ (t >> 7)) * (t | 61u); t ^= t >> 14;
    if (k >= offset) out.push_back(t < 0x80000000u ? -1 : 1);
  }
  return out;
}
inline std::vector<int8_t> sign_stream_rule(const std::string& rule, uint64_t seed, uint64_t offset, size_t n) {
  if (rule == "splitmix64") return sign_stream(seed, offset, n);
  if (rule == "mulberry32_7919_13") return sign_stream_mulberry32(seed, offset, n);
  throw std::runtime_error("unknown sign rule: " + rule);                 
}
}
