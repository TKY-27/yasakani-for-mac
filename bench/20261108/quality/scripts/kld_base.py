"""kld_base.py -- the reference file and the per-position formula of the quality table.

The reference (the original BF16 model's next-token distribution at every scored position) is kept in the file format of llama.cpp's
perplexity tool (tools/perplexity/perplexity.cpp, --kl-divergence-base), so that llama.cpp and every other engine are scored on the same
token ids, the same positions and the same reference:
  "_logits_" | uint32 n_ctx | int32 n_vocab | int32 n_chunk | int32 tokens[n_chunk * n_ctx] |
  per chunk, rows i = 0 .. n_ctx - 2 - first (first = n_ctx / 2; row i = the logits at position first + i, target tokens[first + i + 1]):
  nv = 2 * ((n_vocab + 1) / 2) + 4 uint16: float scale, float min_log_prob, then n_vocab uint16 q_i with
  log p_i ~= scale * q_i + min_log_prob (log-probs below max - 16 clamp to the minimum).
`llama-perplexity -m X.gguf -c n_ctx -b n_ctx --kl-divergence-base FILE --kl-divergence` reads the tokens from FILE and prints PPL(Q),
PPL(base), KLD, the change of the token probability and "Same top p" on exactly those positions.  `metrics` below repeats its per-row
math (kl_divergence's log_softmax) in numpy for engines that give the logits of the whole vocabulary, and `summary` its final
statistics, so every row of the table uses one formula.  The reference file of the table itself is not published.
Needs numpy.  usage (self-test, CPU only): python kld_base.py --selftest"""
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"_logits_"


def nv_of(n_vocab):
    return 2 * ((n_vocab + 1) // 2) + 4


def quantize_rows(logits):
    """llama.cpp log_softmax(n_vocab, logits, uint16 *log_prob, tok) for every row of logits [n, V] float32 -> uint16 [n, nv]"""
    lg = np.ascontiguousarray(logits, np.float32); n, V = lg.shape; nv = nv_of(V)
    out = np.zeros((n, nv), np.uint16)
    for r in range(n):
        x = lg[r]; mx = np.float32(x.max()); mn = np.float32(max(np.float32(x.min()), np.float32(mx - np.float32(16))))
        sum_exp = float(np.exp(x - mx, dtype=np.float32).astype(np.float64).sum())
        lse = np.float32(np.log(sum_exp))
        min_lp = np.float32(np.float32(mn - mx) - lse)
        scale = np.float32(np.float32(mx - mn) / np.float32(65535.0))
        out[r, 0:2] = np.frombuffer(np.float32(scale).tobytes(), np.uint16); out[r, 2:4] = np.frombuffer(np.float32(min_lp).tobytes(), np.uint16)
        if scale:
            inv = np.float32(np.float32(1) / scale)
            q = np.rint((inv * (x - mn)).astype(np.float32))                # nearest_int: round half to even in float32
            out[r, 4:4 + V] = np.where(x > mn, q, 0).astype(np.uint16)
    return out


class KldWriter:
    """tokens: int32 [n_chunk, n_ctx]; then add_chunk(logits [n_ctx, V] float32) once per chunk, in order"""
    def __init__(self, path, tokens, n_vocab):
        tokens = np.ascontiguousarray(tokens, np.int32); self.n_chunk, self.n_ctx = tokens.shape; self.V = n_vocab; self.done = 0
        self.f = open(path, "wb")
        self.f.write(MAGIC + struct.pack("<Iii", self.n_ctx, n_vocab, self.n_chunk)); self.f.write(tokens.tobytes())

    def add_chunk(self, logits):
        first = self.n_ctx // 2
        assert logits.shape == (self.n_ctx, self.V), logits.shape
        for r0 in range(first, self.n_ctx - 1, 128):                     # 128 rows at a time (memory)
            self.f.write(quantize_rows(logits[r0:min(r0 + 128, self.n_ctx - 1)]).tobytes())
        self.done += 1

    def close(self):
        self.f.close()
        if self.done != self.n_chunk: raise RuntimeError(f"KldWriter: {self.done} of {self.n_chunk} chunks written")


class KldReader:
    def __init__(self, path):
        with open(path, "rb") as f:
            if f.read(8) != MAGIC: raise ValueError(f"{path}: not a llama.cpp logits file")
            self.n_ctx, self.V, self.n_chunk = struct.unpack("<Iii", f.read(12))
        self.first = self.n_ctx // 2; self.rows = self.n_ctx - 1 - self.first; self.nv = nv_of(self.V)
        off = 20 + 4 * self.n_ctx * self.n_chunk
        self.tokens = np.memmap(path, np.int32, "r", 20, (self.n_chunk, self.n_ctx))
        self.lp = np.memmap(path, np.uint16, "r", off, (self.n_chunk, self.rows, self.nv))

    def chunk(self, c):
        """uint16 [rows, nv] of chunk c (row i = position first + i)"""
        return self.lp[c]

    def close(self):
        for m in (self.tokens, self.lp): m._mmap.close()
        del self.tokens, self.lp


def base_logprobs(rows_u16, V):
    """dequantised base log-probs float32 [n, V] (llama.cpp: scale * q + min_log_prob)"""
    sc = rows_u16[:, 0:2].copy().view(np.float32)[:, 0]; mn = rows_u16[:, 2:4].copy().view(np.float32)[:, 0]
    return (sc[:, None] * rows_u16[:, 4:4 + V].astype(np.float32) + mn[:, None]).astype(np.float32)


def metrics(logits, rows_u16, targets):
    """llama.cpp kl_divergence's per-row math for our logits [n, V] float32 vs base rows [n, nv] -> dict of per-row arrays:
    nll, nll_base, kld, same_top (bool), p_diff"""
    lg = np.ascontiguousarray(logits, np.float32); n, V = lg.shape; tg = np.asarray(targets, np.int64)
    imax = lg.argmax(1); mx = lg[np.arange(n), imax]
    sum_exp = np.exp(lg - mx[:, None], dtype=np.float32).astype(np.float64).sum(1)
    lse = np.log(sum_exp).astype(np.float32)
    nll = (mx + lse - lg[np.arange(n), tg]).astype(np.float32)
    lpb = base_logprobs(rows_u16, V)
    nll_base = (-lpb[np.arange(n), tg]).astype(np.float32)
    mx2 = (mx + lse).astype(np.float32)
    keep = lpb > np.float32(-16)
    pb = np.where(keep, np.exp(lpb, dtype=np.float32), np.float32(0))
    kld = (pb.astype(np.float64) * (lpb - lg + mx2[:, None]).astype(np.float64)).sum(1)
    imax_base = lpb.argmax(1)
    p_diff = (np.exp(-nll) - np.exp(-nll_base)).astype(np.float32)
    return dict(nll=nll, nll_base=nll_base, kld=kld, same_top=imax == imax_base, p_diff=p_diff)


def summary(m):
    """llama.cpp's final statistics from concatenated per-row metrics"""
    nll = m["nll"].astype(np.float64); nb = m["nll_base"].astype(np.float64); k = np.sort(m["kld"]); pd = m["p_diff"].astype(np.float64)
    n = len(nll)
    pct = lambda v, f: float(np.interp(f * (len(v) - 1), np.arange(len(v)), v))
    return dict(count=n, ppl=float(np.exp(nll.mean())), ppl_base=float(np.exp(nb.mean())), ln_ppl_ratio=float(nll.mean() - nb.mean()),
                kld_mean=float(k.mean()), kld_median=float(np.median(k)), kld_p99=pct(k, 0.99), kld_p999=pct(k, 0.999), kld_max=float(k[-1]),
                rms_dp_pct=float(100 * np.sqrt((pd * pd).mean())), same_top_pct=float(100 * np.mean(m["same_top"])))


def _selftest():
    """write a small base file from random logits, read it back, and check: a model equal to the base has KLD ~ 0 and 100 % same top;
    a perturbed one has KLD > 0; PPL(base) from the file matches the exact log-softmax to the quantisation step"""
    import tempfile
    rng = np.random.default_rng(0); n_ctx, V, n_chunk = 16, 1001, 2
    tok = rng.integers(0, V, (n_chunk, n_ctx)).astype(np.int32)
    lg = (rng.standard_normal((n_chunk, n_ctx, V)) * 3).astype(np.float32)
    p = Path(tempfile.gettempdir()) / "kld_base_selftest.bin"
    w = KldWriter(p, tok, V)
    for c in range(n_chunk): w.add_chunk(lg[c])
    w.close()
    r = KldReader(p); assert (r.n_ctx, r.V, r.n_chunk, r.rows, r.nv) == (n_ctx, V, n_chunk, n_ctx - 1 - n_ctx // 2, nv_of(V))
    assert np.array_equal(np.asarray(r.tokens), tok) and p.stat().st_size == 20 + 4 * n_ctx * n_chunk + 2 * n_chunk * r.rows * r.nv
    for c in range(n_chunk):
        rows = np.asarray(r.chunk(c)); x = lg[c, r.first:n_ctx - 1]; tg = tok[c, r.first + 1:]
        m = metrics(x, rows, tg)
        exact = np.log(np.exp(x.astype(np.float64)).sum(1)) - x.astype(np.float64)[np.arange(len(tg)), tg]
        step = 16.0 / 65535
        assert np.abs(m["nll"] - exact).max() < 1e-4 and np.abs(m["nll_base"] - exact).max() < step + 1e-4, np.abs(m["nll_base"] - exact).max()
        assert m["same_top"].all() and np.abs(m["kld"]).max() < 1e-3, m["kld"]
        m2 = metrics(x + rng.standard_normal(x.shape).astype(np.float32), rows, tg); assert m2["kld"].min() > 0.01
    del rows; r.close(); p.unlink()
    print("[kld_base] selftest ok: header, tokens, rows, size, base NLL within one step, KLD ~ 0 for the base itself, > 0 when perturbed")


if __name__ == "__main__":
    if "--selftest" in sys.argv: _selftest()
    else: print(__doc__)
