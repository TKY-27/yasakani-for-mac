// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#  if defined(YSKN_BUILD)
#    define YSKN_API __declspec(dllexport)
#  else
#    define YSKN_API __declspec(dllimport)
#  endif
#else
#  define YSKN_API __attribute__((visibility("default")))
#endif
#define YSKNCUDA_API
#ifdef __cplusplus
extern "C" {
#endif
YSKNCUDA_API int yskn_repack_warp_to_frag(const uint32_t* d_words, uint32_t* d_frag, int rows, int seqs, int wps, void* cuda_stream);
YSKNCUDA_API int yskn_repack_warp_to_frag_host(const uint32_t* h_words, uint32_t* h_frag, int rows, int seqs, int wps);
typedef struct yskn_engine yskn_engine;
 
YSKN_API yskn_engine* yskn_open(const char* yskn_path, int max_ctx, int device, int flags);
YSKN_API void          yskn_close(yskn_engine* e);
YSKN_API int           yskn_reset(yskn_engine* e);                                        
YSKN_API int           yskn_prefill(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred);  


YSKN_API int           yskn_set_visual(yskn_engine* e, const uint16_t* features, const int32_t* slots, int n_image,
                                       const int32_t* positions, int n_prompt, int delta);
YSKN_API int           yskn_decode(yskn_engine* e, int32_t* out_ids, int steps);           
YSKN_API int           yskn_nll(yskn_engine* e, const int32_t* ids, int n, float* out_nll);  
YSKN_API int           yskn_logits(yskn_engine* e, float* out_vocab);                      
YSKN_API int           yskn_position(yskn_engine* e);
YSKN_API const char*   yskn_last_error(void);



YSKN_API int           yskn_verify(yskn_engine* e, const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_verify_nll(yskn_engine* e, const int32_t* ids, const int32_t* targets, int M, int use_graph, int32_t* out_argmax, float* out_nll);
YSKN_API int           yskn_commit(yskn_engine* e, int n_accept);
YSKN_API int           yskn_prefill_rows(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_hidden_rows(yskn_engine* e, int n, float* out);               
YSKN_API int           yskn_logits_rows(yskn_engine* e, int row0, int n, float* out);     
 
YSKN_API int           yskn_set_capture(yskn_engine* e, const int32_t* layers, int n);
YSKN_API int           yskn_capture_rows(yskn_engine* e, int k, int n, float* out);




 
YSKN_API int           yskn_load_draft(yskn_engine* e, const char* yskn_path, const char* head_ids_path);
YSKN_API int           yskn_draft_round(yskn_engine* e, int mode, int a, int steps, int use_graph);    
YSKN_API int           yskn_verify_device(yskn_engine* e, int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids);
YSKN_API int           yskn_set_verify_tokens(yskn_engine* e, const int32_t* ids, int first_row, int n);
YSKN_API int           yskn_draft_catchup(yskn_engine* e, int p0, int n, const int32_t* toks, int use_graph);
YSKN_API int           yskn_spec_generate(yskn_engine* e, const int32_t* prompt, int n_prompt, int n_gen, const int32_t* cfg, int n_cfg, int32_t* out_ids, double* stats, double* times_ms);




YSKN_API int           yskn_specx_load_static(const char* path);


YSKN_API int           yskn_set_opt(yskn_engine* e, int key, int value);


YSKN_API int           yskn_lfrow_table(yskn_engine* e, const uint8_t* tab48);
YSKN_API int           yskn_lfrow_promote(yskn_engine* e, int src, int dst);



#ifdef __cplusplus
}
#endif
