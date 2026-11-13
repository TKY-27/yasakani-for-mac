// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once



#include <cuda_runtime.h>
#include <cstddef>
namespace yskn {
cudaError_t tracked_malloc(void** p, size_t n);
cudaError_t tracked_free(void* p);
cudaError_t tracked_malloc_host(void** p, size_t n);
cudaError_t tracked_free_host(void* p);
void* devmem_owner();
void devmem_set_owner(void* owner);
size_t devmem_release(void* owner);           
struct DevMemScope {                           
  void* prev;
  explicit DevMemScope(void* owner) : prev(devmem_owner()) { devmem_set_owner(owner); }
  ~DevMemScope() { devmem_set_owner(prev); }
  DevMemScope(const DevMemScope&) = delete; DevMemScope& operator=(const DevMemScope&) = delete;
};
}
#define cudaMalloc(pp, n) ::yskn::tracked_malloc(reinterpret_cast<void**>(pp), (n))
#define cudaFree(p) ::yskn::tracked_free((void*)(p))
#define cudaMallocHost(pp, n) ::yskn::tracked_malloc_host(reinterpret_cast<void**>(pp), (n))
#define cudaFreeHost(p) ::yskn::tracked_free_host((void*)(p))
