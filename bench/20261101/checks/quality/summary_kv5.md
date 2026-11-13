# Quality of the public build (RC1), kv5

Date: 2026-11-01 05:47:16.  Setting: public default.

| corpus | positions | PPL | KLD mean | same top % | PPL (engine NLL) | argmax mismatches | max NLL diff | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | 20460 | 6.2325 | 0.02376 | 93.964 | 6.2325 | 0 | 9.04e-07 | 145 |
| c4 | 20460 | 10.8741 | 0.01891 | 92.928 | 10.8741 | 0 | 8.75e-07 | 149 |
| fineweb_edu | 20460 | 7.9742 | 0.01920 | 93.397 | 7.9742 | 0 | 8.03e-07 | 150 |

All corpora together: 61380 positions, PPL 8.1454, KLD mean 0.02062, same top 93.429 %.

Reference (research build, FP16 KV, 2026-10-29): 61,380 positions, PPL 8.135, KLD 0.0205, same top 93.41 %.
