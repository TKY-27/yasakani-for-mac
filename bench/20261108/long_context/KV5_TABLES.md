# KV5 and an FP16 key-value cache with the fastest model

Four fixed sequences of 131,072 tokens; at each context length the last 1,023 positions of every sequence (4,092 in all) are compared with the BF16 reference. kl = mean KL divergence (nat), ci95 = 95% interval, same_top_pct = top-1 agreement with BF16. paired = KL(KV5) - KL(FP16) at the same positions. memory_gib = GPU memory of the engine when the context is reserved at start (CUDA context included, the desktop's share excluded).

| Context | KL with FP16 KV (95% interval) | KL with KV5 (95% interval) | KV5 - FP16 at the same positions (95% interval) | Top-1 agreement with BF16: FP16 KV / KV5 | KV5 has the same top-1 as FP16 KV |
|---|---|---|---|---|---|
| 2,048 | 0.02010 (0.01893 to 0.02135) | 0.02020 (0.01908 to 0.02139) | +0.000097 (-0.000114 to +0.000288) | 94.75% / 94.79% | 99.17% |
| 4,096 | 0.02102 (0.01931 to 0.02337) | 0.02154 (0.01980 to 0.02387) | +0.000516 (+0.000153 to +0.001017) | 94.87% / 94.75% | 99.10% |
| 8,192 | 0.03034 (0.02868 to 0.03217) | 0.03071 (0.02901 to 0.03258) | +0.000371 (+0.000111 to +0.000627) | 93.30% / 93.43% | 98.63% |
| 16,384 | 0.02202 (0.01999 to 0.02415) | 0.02249 (0.02037 to 0.02474) | +0.000465 (+0.000131 to +0.000853) | 96.07% / 96.07% | 99.58% |
| 32,768 | 0.02892 (0.02712 to 0.03081) | 0.02922 (0.02745 to 0.03113) | +0.000298 (+0.000077 to +0.000508) | 94.48% / 94.35% | 99.00% |
| 65,536 | 0.02907 (0.02752 to 0.03074) | 0.02911 (0.02756 to 0.03077) | +0.000034 (-0.000215 to +0.000272) | 93.55% / 93.72% | 98.75% |
| 131,072 | 0.02646 (0.02510 to 0.02788) | 0.02687 (0.02545 to 0.02844) | +0.000412 (+0.000177 to +0.000674) | 93.08% / 92.94% | 98.92% |

| Reserved context | FP16 KV, GiB | INT8 KV, GiB | KV5, GiB |
|---|---|---|---|
| 2,048 | 11.18848 | 11.35645 | 11.34180 |
| 4,096 | 11.31348 | 11.42090 | 11.42773 |
| 8,192 | 11.56543 | 11.54785 | 11.49805 |
| 16,384 | 12.06738 | 11.80371 | 11.65625 |
| 32,768 | 13.07129 | 12.31543 | 11.99805 |
| 65,536 | 15.07910 | 13.33887 | 12.68164 |
| 131,072 | 19.09473 | 15.38574 | 14.04883 |
| 262,144 | 27.12598 | 19.47949 | 16.78320 |
| KV cache per token, KiB | 64.250 | 32.750 | 21.875 |
