# Long-context check of the public build

Date: 2026-11-01 05:42:29.  Context: 262144 tokens.  Server ready after 8.6 s.

| memory (nvidia-smi, MiB, desktop included) | value |
|---|---:|
| baseline_used_mib | 2516 |
| after_load_used_mib | 24619 |
| after_read_used_mib | 24635 |
| peak_used_mib | 24641 |
| final_used_mib | 24641 |
| peak_minus_baseline_mib | 22125 |

Code base: 33 files, 249629 tokens (sha256 bd7ceb1c1d49...); pagoda conversation 249744 prompt tokens.

| step | prompt tokens | cached tokens | first token after (s) | prompt tokens per s | written | tokens per s | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| R_read | 249658 | 0 | 133.02 | 1876.9 | 2 |  | stop |
| F_followup | 249689 | 249659 | 0.419 | 596070.7 | 5 | 31.7 | stop |
| S_resend | 249658 | 0 | 133.247 | 1873.6 | 2 |  | stop |
| P_pagoda | 249744 | 249659 | 0.477 | 523297.0 | 108 | 242.6 | stop |

Pagoda: complete HTML page = False (0 characters).
