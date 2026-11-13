// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <nlohmann/json.hpp>
#include "mapped_file.hpp"
namespace yskn {
using json = nlohmann::json;
struct RotDesc { std::string kind = "H512"; int m = 1; uint64_t sign_seed = 0, sign_offset = 0, qm_off = 0; std::string rule = "splitmix64"; };   
struct RarDesc { int tile_rows = 64, tiles = 0; uint64_t k_off = 0, off_off = 0; };
struct TensorDesc {
  std::string name, role, start = "zero", layout = "warp"; int rows = 0, cols = 0, physical_rows = 0, L = 12, d = 2, T = 256, bits = 6;
  RotDesc rot_in, rot_out; bool has_rot_out = false; uint64_t lut_off = 0, scale_off = 0, stream_off = 0, stream_bytes = 0, bias_off = 0; bool has_rar = false; RarDesc rar; float proxy = 0.f;
};
struct VectorDesc { std::string name; uint64_t off = 0; int n = 0; std::string dtype = "f16"; };
struct ModelDesc { std::string arch; int layers = 0, hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0; float eps = 0.f, rope_theta = 0.f; bool tie_embeddings = false; };
struct Container {
  ysknio::MappedFile mf; const uint8_t* base = nullptr; size_t len = 0; ModelDesc model; std::vector<TensorDesc> tensors; std::vector<VectorDesc> vectors;
  json raw;                                                       
  void open(const std::string& path) {
    mf.open(path); base = mf.data(); len = mf.size();
    if (len < 0x20 || std::memcmp(base, "YSKN", 4) != 0) throw std::runtime_error("not a YSKN container");
    uint32_t version, json_len; std::memcpy(&version, base + 4, 4); std::memcpy(&json_len, base + 8, 4);
    if (version != 1) throw std::runtime_error("unsupported YSKN version");
    if ((size_t)json_len + 0x20 > len) throw std::runtime_error("YSKN header truncated");
    const json j = json::parse(std::string((const char*)base + 0x20, json_len)); raw = j;
    const json& m = j.at("model");
    model.arch = m.at("arch").get<std::string>(); model.layers = m.at("layers").get<int>(); model.hidden = m.at("hidden").get<int>();
    model.ffn = m.at("ffn").get<int>(); model.heads = m.at("heads").get<int>(); model.kv_heads = m.at("kv_heads").get<int>();
    model.head_dim = m.at("head_dim").get<int>(); model.vocab = m.at("vocab").get<int>(); model.eps = m.at("eps").get<float>();
    model.rope_theta = m.at("rope_theta").get<float>(); model.tie_embeddings = m.value("tie_embeddings", false);
    auto rot = [](const json& r) { RotDesc d; d.kind = r.at("kind").get<std::string>(); d.m = r.at("m").get<int>(); d.sign_seed = r.at("sign_seed").get<uint64_t>();
                                   d.sign_offset = r.at("sign_offset").get<uint64_t>(); d.qm_off = r.value("qm_off", (uint64_t)0);
                                   d.rule = r.value("rule", std::string("splitmix64")); return d; };
    for (const json& t : j.at("tensors")) {
      TensorDesc d; d.name = t.at("name").get<std::string>(); d.role = t.at("role").get<std::string>(); d.rows = t.at("rows").get<int>(); d.cols = t.at("cols").get<int>();
      
      
      
      d.physical_rows = t.value("physical_rows", d.rows); d.L = t.value("l", 12); d.d = t.value("d", 2); d.T = t.value("t", 256); d.bits = t.at("bits").get<int>();
      d.start = t.value("start", std::string("zero")); d.layout = t.value("layout", std::string("warp"));
      auto def_rot = [](int m, uint64_t off) { RotDesc r; r.kind = "H512"; r.m = m; r.sign_seed = 7; r.sign_offset = off; r.qm_off = 0; r.rule = "mulberry32_7919_13"; return r; };
      if (t.contains("rot_in")) d.rot_in = rot(t.at("rot_in")); else d.rot_in = def_rot(d.cols / 512, 0);
      if (t.contains("rot_out")) { if (!t.at("rot_out").is_null()) { d.rot_out = rot(t.at("rot_out")); d.has_rot_out = d.rot_out.kind != "none"; } }
      else if (d.role != "head" && d.role != "embedding") { d.rot_out = def_rot(d.rows / 512, (uint64_t)d.cols); d.has_rot_out = true; }   
      d.lut_off = t.at("lut_off").get<uint64_t>(); d.scale_off = t.at("scale_off").get<uint64_t>(); d.stream_off = t.at("stream_off").get<uint64_t>();
      d.stream_bytes = t.at("stream_bytes").get<uint64_t>(); d.bias_off = t.value("bias_off", (uint64_t)0); d.proxy = t.value("proxy", 0.f);
      if (t.contains("rar") && !t.at("rar").is_null()) { const json& r = t.at("rar"); d.has_rar = true;
        d.rar = RarDesc{r.at("tile_rows").get<int>(), r.at("tiles").get<int>(), r.at("k_off").get<uint64_t>(), r.at("off_off").get<uint64_t>()}; }
      if (d.d != 2 || d.T != 256 || d.L < 8 || d.L > 14 || d.bits < 4 || d.bits > 8 || (d.start != "zero" && d.start != "carry") || d.cols % 256)
        throw std::runtime_error("YSKN reader: unsupported/invalid tensor description: " + d.name);   
      tensors.push_back(d);
    }
    for (const json& v : j.at("vectors")) vectors.push_back(VectorDesc{v.at("name").get<std::string>(), v.at("off").get<uint64_t>(), v.at("n").get<int>(), v.value("dtype", std::string("f16"))});
  }
  const TensorDesc& tensor(const std::string& n) const { for (auto& t : tensors) if (t.name == n) return t; throw std::runtime_error("missing tensor " + n); }
  const VectorDesc& vector(const std::string& n) const { for (auto& v : vectors) if (v.name == n) return v; throw std::runtime_error("missing vector " + n); }
  template <class T2> const T2* at(uint64_t off) const { return reinterpret_cast<const T2*>(base + off); }
};
}
