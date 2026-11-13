# Speed tables made from the records

Output of `scripts/tables.py`. The rules are in the docstring of the script; the conditions and limits are in the README of `bench/`.

## Thinking off: average without repetition loops and unrequested tool calls (the table of the README), tok/s

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 529 | 285 | 494 | 909 | 1,222 | 1,156 | 1,260 |
| YASAKANI quality model | 486 | 179 | 483 | 930 | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 139 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |
| (largest value of the other engines) | 465 | 204 | 374 | 599 | 441 | 658 | 666 |

## Thinking on: fastest run of each task (the table of the README), tok/s

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 337 | 287 | 497 | 536 | 667 | 1,108 | 704 |
| YASAKANI quality model | 299 | 242 | 428 | 480 | 743 | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |
| (largest value of the other engines) | 290 | 206 | 348 | 429 | 550 | 663 | 516 |

## Thinking off: average of all runs, tok/s

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 900 | 285 | 494 | 909 | 1,222 | 1,156 | 1,260 |
| YASAKANI quality model | 486 | 263 | 483 | 930 | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 438 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 282 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 312 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 270 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 143 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |
| (largest value of the other engines) | 465 | 204 | 374 | 599 | 441 | 658 | 666 |

## Thinking on: average of all runs, tok/s

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 324 | 273 | 467 | 471 | 627 | 1,073 | 657 |
| YASAKANI quality model | 282 | 234 | 407 | 432 | 626 | 1,013 | 568 |
| MegaCapybara 1.39 Tiny | 285 | 192 | 298 | 418 | 508 | 662 | 508 |
| mkyt v0.3.2 | 242 | 193 | 340 | 361 | 357 | 597 | 448 |
| ignis v0.9.6 | 208 | 128 | 268 | 273 | 270 | 472 | 316 |
| NInfer (November 2 build) | 167 | 136 | 245 | 252 | 254 | 396 | 280 |
| vLLM 0.31.0 | 128 | 109 | 192 | 189 | 168 | 365 | 246 |
| SGLang 0.5.22 | 153 | 122 | 202 | 217 | 250 | 400 | 274 |
| llama.cpp b11700 | 136 | 122 | 180 | 184 | 197 | 346 | 219 |
| TensorFold 0.6.6 | 127 | 77 | 152 | 175 | 320 | 356 | 232 |
| (largest value of the other engines) | 285 | 193 | 340 | 418 | 508 | 662 | 508 |

## Settings of each engine (thinking off: average of all runs / without the runs left out; thinking on: average / fastest run)

* MegaCapybara 1.39 Tiny, `nov3:megacapybara_kv64`, thinking off, 29 runs: Code 465 / 465; Story 164 / 164; Translation 268 / 268; Structured 599 / 599; Code edit 441 / 441; JSONL edit 658 / 658; Long JSONL 666 / 666
* MegaCapybara 1.39 Tiny, `megacapybara_tiny_think`, thinking on, 30 runs: Code 285 / 290; Story 192 / 206; Translation 298 / 308; Structured 418 / 429; Code edit 508 / 550; JSONL edit 662 / 663; Long JSONL 508 / 516
* mkyt v0.3.2, `mkyt_v032`, thinking off, 30 runs: Code 438 / 395; Story 204 / 204; Translation 374 / 374; Structured 555 / 555; Code edit 423 / 423; JSONL edit 634 / 634; Long JSONL 639 / 639
* mkyt v0.3.2, `mkyt_v032_think`, thinking on, 30 runs: Code 242 / 244; Story 193 / 196; Translation 340 / 348; Structured 361 / 384; Code edit 357 / 378; JSONL edit 597 / 603; Long JSONL 448 / 454
* ignis v0.9.6, `ignis_v096`, thinking off, 28 runs: Code 330 / 330; Story 167 / 167; Translation 298 / 298; Structured 439 / 439; Code edit 331 / 331; JSONL edit 510 / 510; Long JSONL 514 / 514
* ignis v0.9.6, `ignis_v096_think`, thinking on, 30 runs: Code 208 / 211; Story 128 / 130; Translation 268 / 276; Structured 273 / 277; Code edit 270 / 278; JSONL edit 472 / 485; Long JSONL 316 / 327
* NInfer (November 2 build), `n12_ninfer_master`, thinking off, 30 runs: Code 282 / 264; Story 123 / 123; Translation 257 / 257; Structured 386 / 386; Code edit 295 / 295; JSONL edit 443 / 443; Long JSONL 438 / 438
* NInfer (November 2 build), `n12t_ninfer_master`, thinking on, 30 runs: Code 167 / 172; Story 136 / 140; Translation 245 / 258; Structured 252 / 263; Code edit 254 / 263; JSONL edit 396 / 404; Long JSONL 280 / 322
* vLLM 0.31.0, `n12_vl_dflash_lm4`, thinking off, 30 runs: Code 312 / 290; Story 113 / 113; Translation 261 / 261; Structured 335 / 335; Code edit 237 / 237; JSONL edit 374 / 374; Long JSONL 397 / 397
* vLLM 0.31.0, `n12_vl_dspark_lm4`, thinking off, 30 runs: Code 268 / 268; Story 104 / 104; Translation 198 / 198; Structured 354 / 354; Code edit 241 / 241; JSONL edit 426 / 426; Long JSONL 431 / 431
* vLLM 0.31.0, `n12_vl_mtp_lm4`, thinking off, 30 runs: Code 202 / 192; Story 150 / 150; Translation 185 / 185; Structured 208 / 208; Code edit 218 / 218; JSONL edit 211 / 211; Long JSONL 208 / 208
* vLLM 0.31.0, `n12t_vl_dflash_lm4`, thinking on, 30 runs: Code 128 / 131; Story 109 / 112; Translation 192 / 197; Structured 189 / 195; Code edit 164 / 187; JSONL edit 365 / 369; Long JSONL 246 / 253
* vLLM 0.31.0, `n12t_vl_mtp_lm4`, thinking on, 30 runs: Code 119 / 124; Story 108 / 109; Translation 150 / 153; Structured 146 / 152; Code edit 168 / 171; JSONL edit 208 / 210; Long JSONL 163 / 177
* SGLang 0.5.22, `n12_sg_dflash_lm4_norc`, thinking off, 30 runs: Code 249 / 252; Story 116 / 116; Translation 202 / 202; Structured 321 / 321; Code edit 255 / 255; JSONL edit 413 / 413; Long JSONL 374 / 374
* SGLang 0.5.22, `n12_sg_dflash_nv`, thinking off, 30 runs: Code 270 / 242; Story 124 / 124; Translation 246 / 246; Structured 371 / 371; Code edit 269 / 269; JSONL edit 385 / 385; Long JSONL 411 / 411
* SGLang 0.5.22, `n12_sg_dspark_lm4`, thinking off, 30 runs: Code 222 / 213; Story 120 / 120; Translation 202 / 202; Structured 357 / 357; Code edit 430 / 430; JSONL edit 466 / 466; Long JSONL 451 / 451
* SGLang 0.5.22, `n12_sg_mtp_nv`, thinking off, 30 runs: Code 189 / 179; Story 129 / 129; Translation 176 / 176; Structured 199 / 199; Code edit 202 / 202; JSONL edit 205 / 205; Long JSONL 210 / 210
* SGLang 0.5.22, `n12t_sg_dflash_nv`, thinking on, 30 runs: Code 153 / 157; Story 122 / 124; Translation 202 / 206; Structured 217 / 230; Code edit 250 / 257; JSONL edit 400 / 402; Long JSONL 274 / 282
* SGLang 0.5.22, `n12t_sg_mtp_nv`, thinking on, 30 runs: Code 119 / 120; Story 105 / 107; Translation 149 / 151; Structured 147 / 154; Code edit 168 / 181; JSONL edit 206 / 207; Long JSONL 164 / 169
* llama.cpp b11700, `n12_llama_dflq4`, thinking off, 30 runs: Code 242 / 242; Story 125 / 125; Translation 232 / 232; Structured 337 / 337; Code edit 244 / 244; JSONL edit 360 / 360; Long JSONL 362 / 362
* llama.cpp b11700, `n12_llama_mtp2`, thinking off, 30 runs: Code 180 / 180; Story 143 / 139; Translation 175 / 175; Structured 187 / 187; Code edit 190 / 190; JSONL edit 191 / 191; Long JSONL 193 / 193
* llama.cpp b11700, `n12t_llama_dflq4`, thinking on, 30 runs: Code 125 / 133; Story 102 / 103; Translation 180 / 184; Structured 184 / 197; Code edit 197 / 206; JSONL edit 346 / 349; Long JSONL 219 / 221
* llama.cpp b11700, `n12t_llama_mtp2`, thinking on, 30 runs: Code 136 / 140; Story 122 / 122; Translation 156 / 156; Structured 151 / 155; Code edit 175 / 178; JSONL edit 188 / 188; Long JSONL 166 / 170
* TensorFold 0.6.6, `n12_tensorfold`, thinking off, 30 runs: Code 181 / 181; Story 104 / 104; Translation 172 / 172; Structured 405 / 405; Code edit 383 / 383; JSONL edit 408 / 408; Long JSONL 261 / 261
* TensorFold 0.6.6, `n12t_tensorfold`, thinking on, 30 runs: Code 127 / 134; Story 77 / 79; Translation 152 / 167; Structured 175 / 183; Code edit 320 / 439; JSONL edit 356 / 360; Long JSONL 232 / 268

## Runs left out of the first table (thinking off)

* YASAKANI fastest model: scenario_code_cuda r0 L, scenario_code_cuda r1 L, scenario_code_typescript r0 T, scenario_code_typescript r1 T
* YASAKANI quality model: scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L
* mkyt v0.3.2 `mkyt_v032`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* NInfer (November 2 build) `n12_ninfer_master`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* vLLM 0.31.0 `n12_vl_dflash_lm4`: scenario_code_cuda r0 T, scenario_code_cuda r1 T, scenario_code_typescript r0 T, scenario_code_typescript r1 T
* vLLM 0.31.0 `n12_vl_dspark_lm4`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* vLLM 0.31.0 `n12_vl_mtp_lm4`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* SGLang 0.5.22 `n12_sg_dflash_lm4_norc`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* SGLang 0.5.22 `n12_sg_dflash_nv`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* SGLang 0.5.22 `n12_sg_dspark_lm4`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* SGLang 0.5.22 `n12_sg_mtp_nv`: scenario_code_typescript r0 T, scenario_code_typescript r1 T
* llama.cpp b11700 `n12_llama_mtp2`: scenario_story_en_mystery r1 L

T = tool call written although the request has no tools, L = repetition loop. Thinking on: 0 runs flagged.

## Measured the same day, not used in the tables

* NInfer, October build, `n12_ninfer`, thinking off, 30 runs: Code 290 / 277; Story 122 / 122; Translation 249 / 249; Structured 386 / 386; Code edit 294 / 294; JSONL edit 442 / 442; Long JSONL 435 / 435
