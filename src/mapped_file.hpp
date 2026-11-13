// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once


#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <utility>
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace ysknio {

class MappedFile {
public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path) { open(path); }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept { swap(o); }
  MappedFile& operator=(MappedFile&& o) noexcept { if (this != &o) { close(); swap(o); } return *this; }
  ~MappedFile() { close(); }

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

  void open(const std::string& path) {
    close();
#if defined(_WIN32)
    const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wn <= 0) throw std::runtime_error("bad UTF-8 path: " + path);
    std::wstring w((size_t)wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], wn);
    file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) { file_ = nullptr; throw std::runtime_error("open failed: " + path); }
    LARGE_INTEGER sz; if (!GetFileSizeEx(file_, &sz)) { close(); throw std::runtime_error("size failed: " + path); }
    size_ = (size_t)sz.QuadPart;
    if (size_ == 0) return;
    map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) { close(); throw std::runtime_error("CreateFileMapping failed: " + path); }
    data_ = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    if (!data_) { close(); throw std::runtime_error("MapViewOfFile failed: " + path); }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);
    struct stat st; if (fstat(fd_, &st) != 0) { close(); throw std::runtime_error("fstat failed: " + path); }
    size_ = (size_t)st.st_size;
    if (size_ == 0) return;
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { close(); throw std::runtime_error("mmap failed: " + path); }
    data_ = (const uint8_t*)p;
#endif
  }

  void close() {
#if defined(_WIN32)
    if (data_) UnmapViewOfFile(data_);
    if (map_) CloseHandle(map_);
    if (file_) CloseHandle(file_);
    map_ = nullptr; file_ = nullptr;
#else
    if (data_) munmap((void*)data_, size_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    data_ = nullptr; size_ = 0;
  }

private:
  void swap(MappedFile& o) noexcept {
    std::swap(data_, o.data_); std::swap(size_, o.size_);
#if defined(_WIN32)
    std::swap(file_, o.file_); std::swap(map_, o.map_);
#else
    std::swap(fd_, o.fd_);
#endif
  }
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
#if defined(_WIN32)
  HANDLE file_ = nullptr;
  HANDLE map_ = nullptr;
#else
  int fd_ = -1;
#endif
};

} 
