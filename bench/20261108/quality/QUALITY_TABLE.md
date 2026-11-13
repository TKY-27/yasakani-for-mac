# Quality on thinking text: all measured configurations

Twenty windows of 2,048 tokens; the last 1,023 positions of every window (20,460 positions) against the BF16 reference (perplexity of the reference on these positions: 2.0652). KL = mean KL divergence from the reference (nat, smaller is closer); top-1 agreement = how often the most likely next token is the same as the reference's. Sizes are in GB (10^9 bytes).

| # | Configuration | Size, GB | KV cache | KL | Top-1 agreement | Perplexity | In the table of docs/MEASUREMENTS.md | Score file |
|---|---|---|---|---|---|---|---|---|
| 1 | YASAKANI quality model | 12.37 | FP16 | 0.0122 | 95.24% | 2.0678 | yes | `scores/yasakani_quality.json` |
| 2 | NInfer, official NVFP4 | 23.72 | FP8 | 0.0178 | 94.67% | 2.0844 | yes | `scores/ninfer_official_nvfp4.json` |
| 3 | llama.cpp, Unsloth UD-Q3_K_XL | 13.15 (main model 12.80) | FP16 | 0.0204 | 94.32% | 2.1013 | yes | `scores/llamacpp_unsloth_ud_q3_k_xl.json` |
| 4 | llama.cpp, GSQ-RCO IQ3_S | 12.12 (main model 11.77) | FP16 | 0.0229 | 93.64% | 2.0894 | yes | `scores/llamacpp_gsq_rco_iq3_s.json` |
| 5 | SGLang, NVIDIA NVFP4 | 21.92 | FP8 (e4m3) | >= 0.0283 (top 100) | 92.70% | 2.1039 | yes | `scores/sglang_nvidia_nvfp4.json` |
| 6 | YASAKANI fastest model | 10.31 | FP16 | 0.0304 | 93.01% | 2.0749 | yes | `scores/yasakani_fastest.json` |
| 7 | llama.cpp, Unsloth UD-IQ3_S | 12.04 (main model 11.69) | FP16 | 0.0311 | 93.22% | 2.1142 | no | `scores/llamacpp_unsloth_ud_iq3_s.json` |
| 8 | MegaCapybara, Tiny | 16.59 | 6/4 | 0.0321 | 92.80% | 2.0954 | yes | `scores/megacapybara_tiny.json` |
| 9 | TensorFold, MLX 4-bit | 16.05 | BF16 | 0.0353 | 92.05% | 2.1143 | yes | `scores/tensorfold_mlx_4bit.json` |
| 10 | mkyt, gdnfp4 | 19.50 | NVFP4 | >= 0.0362 (top 64) | 92.17% | - | yes | `scores/mkyt_gdnfp4_top64.json` |
| 11 | llama.cpp, Unsloth UD-IQ3_XXS | 10.93 (main model 10.58) | FP16 | 0.0458 | 91.32% | 2.1324 | yes | `scores/llamacpp_unsloth_ud_iq3_xxs.json` |
| 12 | ignis, NVFP4full-v2 | 19.41 | hq-e8-2b | 0.0473 | 91.09% | 2.1360 | yes | `scores/ignis_nvfp4full_v2.json` |
| 13 | vLLM, NVFP4 LMHead4 | 18.77 | FP8 | >= 0.0509 (top 100) | 90.22% | 2.1486 | yes | `scores/vllm_nvfp4_lmhead4.json` |
| 14 | llama.cpp, GSQ-RCO IQ3_XXS | 10.44 (main model 10.09) | FP16 | 0.0518 | 91.22% | 2.1210 | yes | `scores/llamacpp_gsq_rco_iq3_xxs.json` |
| 15 | llama.cpp, Unsloth UD-Q2_K_XL | 9.83 (main model 9.48) | FP16 | 0.0759 | 89.01% | 2.1823 | no | `scores/llamacpp_unsloth_ud_q2_k_xl.json` |

## The same size bands

KL of the GGUF files next to the two YASAKANI models in size, measured with llama.cpp, divided by the KL of the YASAKANI model of the band. "Main model" is the GGUF without its auxiliary layers (the tensors `blk.64.*`); YASAKANI's size is its model file.

| Band | Configuration | File, GB | Main model, GB | KL | KL / KL of the YASAKANI model |
|---|---|---|---|---|---|
| 12 GB | **YASAKANI quality model** | 12.37 | 12.37 | **0.0122** | 1 |
| | llama.cpp, Unsloth UD-Q3_K_XL | 13.15 | 12.80 | 0.0204 | 1.67 |
| | llama.cpp, GSQ-RCO IQ3_S | 12.12 | 11.77 | 0.0229 | 1.87 |
| | llama.cpp, Unsloth UD-IQ3_S | 12.04 | 11.69 | 0.0311 | 2.55 |
| 10 GB | **YASAKANI fastest model** | 10.31 | 10.31 | **0.0304** | 1 |
| | llama.cpp, Unsloth UD-IQ3_XXS | 10.93 | 10.58 | 0.0458 | 1.50 |
| | llama.cpp, GSQ-RCO IQ3_XXS | 10.44 | 10.09 | 0.0518 | 1.70 |
| | llama.cpp, Unsloth UD-Q2_K_XL | 9.83 | 9.48 | 0.0759 | 2.49 |

## Further records (not rows of the table)

| Record | KL | Top-1 agreement | Score file |
|---|---|---|---|
| MegaCapybara, Tiny (the earlier run through the API, top-1 only) | - | 92.69% | `scores/megacapybara_tiny_top1_by_api.json` |
| mkyt, gdnfp4 (the earlier run with the top 20) | >= 0.0345 (top 20) | 92.17% | `scores/mkyt_gdnfp4_top20.json` |
