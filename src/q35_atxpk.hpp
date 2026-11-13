// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once







#include "q35.hpp"
namespace yskn {
struct Q35PkArgs { const uint8_t* kp = nullptr; const uint8_t* vp = nullptr; const float* kps = nullptr; const float* vps = nullptr; const unsigned char* rec = nullptr; int layer = 0; int kb = 4; int variant = 0; };
void launch_q35_atx_pk(const Q35AttnArgs&, const uint8_t* wk, const Q35PkArgs&, cudaStream_t);
}  
