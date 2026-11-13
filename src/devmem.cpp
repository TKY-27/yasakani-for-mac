// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
namespace yskn {
namespace {
struct Rec { void* owner; size_t bytes; bool host; };
std::mutex g_mu;
std::unordered_map<void*, Rec>& reg() { static std::unordered_map<void*, Rec> r; return r; }
thread_local void* t_owner = nullptr;


int fill_byte() { static const int f = [] { const char* e = std::getenv("YSKN_DEVMEM_FILL"); return e && e[0] ? std::atoi(e) : 0; }(); return f; }
}
void* devmem_owner() { return t_owner; }
void devmem_set_owner(void* owner) { t_owner = owner; }
cudaError_t tracked_malloc(void** p, size_t n) {
  const cudaError_t e = ::cudaMalloc(p, n);
  if (e == cudaSuccess && *p && t_owner) {
    { std::lock_guard<std::mutex> l(g_mu); reg()[*p] = Rec{t_owner, n, false}; }
    ::cudaMemset(*p, fill_byte(), n); ::cudaDeviceSynchronize();      
  }
  return e;
}
cudaError_t tracked_malloc_host(void** p, size_t n) {
  const cudaError_t e = ::cudaMallocHost(p, n);
  if (e == cudaSuccess && *p && t_owner) { { std::lock_guard<std::mutex> l(g_mu); reg()[*p] = Rec{t_owner, n, true}; } std::memset(*p, fill_byte(), n); }
  return e;
}
cudaError_t tracked_free(void* p) {
  if (p) { std::lock_guard<std::mutex> l(g_mu); reg().erase(p); }
  return ::cudaFree(p);
}
cudaError_t tracked_free_host(void* p) {
  if (p) { std::lock_guard<std::mutex> l(g_mu); reg().erase(p); }
  return ::cudaFreeHost(p);
}
size_t devmem_release(void* owner) {
  std::vector<std::pair<void*, Rec>> mine;
  { std::lock_guard<std::mutex> l(g_mu);
    for (auto it = reg().begin(); it != reg().end();) { if (it->second.owner == owner) { mine.emplace_back(*it); it = reg().erase(it); } else ++it; } }
  size_t bytes = 0;
  for (auto& [p, r] : mine) { if (r.host) ::cudaFreeHost(p); else { ::cudaFree(p); bytes += r.bytes; } }
  return bytes;
}
}
