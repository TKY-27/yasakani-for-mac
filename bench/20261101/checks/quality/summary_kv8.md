# Quality of the public build (RC1), kv8

Date: 2026-11-01 05:54:55.  Setting: INT8 KV (no 5-bit codec settings).

| corpus | positions | PPL | KLD mean | same top % | PPL (engine NLL) | argmax mismatches | max NLL diff | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | 20460 | 6.2335 | 0.02311 | 93.842 | 6.2335 | 0 | 9.56e-07 | 134 |
| c4 | 20460 | 10.8712 | 0.01895 | 92.884 | 10.8712 | 0 | 9.08e-07 | 134 |
| fineweb_edu | 20460 | 7.9709 | 0.01897 | 93.368 | 7.9709 | 0 | 5.94e-07 | 134 |

All corpora together: 61380 positions, PPL 8.1440, KLD mean 0.02035, same top 93.364 %.

Reference (research build, FP16 KV, 2026-10-29): 61,380 positions, PPL 8.135, KLD 0.0205, same top 93.41 %.
