#!/usr/bin/env python3
"""Independent GPU consistency check (2026-11-10): does this GPU give the same bits when the same deterministic work is repeated,
in a program that has nothing to do with YASAKANI?  Uses the PyTorch that is already installed (nothing is downloaded).

  A  float32 matrix products (cuBLAS):   the same product chain repeated, compared bit for bit with the first result
  B  elementwise float32 (tanh / multiply / add) repeated, bit for bit
  C  64-bit integer arithmetic (multiply, xor, shift, sum) repeated, exact
  D  memory: fixed pseudo-random data kept in VRAM (several GiB) and re-read; checksums compared with the first reading

  python -X utf8 gpu_consistency.py --out result.json [--seconds 60] [--vram-gib 12]
Exit 0: every repeat equal.  Exit 1: at least one mismatch.  Exit 2: PyTorch has no CUDA here."""
import argparse
import datetime as dt
import json
import os
import sys
import time


def log(*a):
    print("[%s]" % dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=60, help="seconds per part A, B, C")
    ap.add_argument("--vram-gib", type=float, default=12.0)
    ap.add_argument("--vram-rounds", type=int, default=12)
    ap.add_argument("--bursty-seconds", type=int, default=0, help="part E: small kernels with a host round trip each (a partial, bursty load like one-row decoding)")
    ap.add_argument("--only-bursty", action="store_true")
    ap.add_argument("--power-seconds", type=int, default=0, help="part P: large half-precision matrix products without pause (drives the GPU to its power limit)")
    ap.add_argument("--only-power", action="store_true")
    a = ap.parse_args()
    import torch
    res = dict(kind="gpu_consistency", session=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__,
               cuda=getattr(torch.version, "cuda", None), parts={})
    if not torch.cuda.is_available():
        log("PyTorch %s has no CUDA here" % torch.__version__)
        res["error"] = "no CUDA"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 2
    dev = torch.device("cuda")
    res["gpu"] = torch.cuda.get_device_name(0)
    log("PyTorch %s, CUDA %s, %s" % (torch.__version__, res["cuda"], res["gpu"]))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                                             # noqa: BLE001
        pass
    g = torch.Generator(device="cpu").manual_seed(20261110)

    def repeat(name, fn, seconds):
        ref = fn()
        ref_bits = ref.view(torch.int32) if ref.dtype == torch.float32 else ref
        n = bad = 0
        first_bad = None
        t0 = time.time()
        while time.time() - t0 < seconds:
            y = fn()
            yb = y.view(torch.int32) if y.dtype == torch.float32 else y
            n += 1
            if not torch.equal(yb, ref_bits):
                bad += 1
                if first_bad is None:
                    d = (yb != ref_bits)
                    first_bad = dict(repeat=n, elements_different=int(d.sum().item()), of=int(d.numel()))
        torch.cuda.synchronize()
        res["parts"][name] = dict(repeats=n, mismatches=bad, first_mismatch=first_bad, seconds=round(time.time() - t0, 1))
        log("%s: %d repeats, %d mismatches %s" % (name, n, bad, first_bad or ""))
        return bad

    total_bad = 0
    if a.bursty_seconds > 0:
        # E: many small kernels, each followed by a host round trip (low, bursty utilisation; clocks and power keep changing)
        W = torch.randn(4096, 4096, generator=g).to(dev)             # 64 MiB "weights"
        v0 = torch.randn(4096, 1, generator=g).to(dev)
        Wi = torch.randint(-(1 << 30), 1 << 30, (1 << 22,), generator=g, dtype=torch.int64).to(dev)   # 32 MiB

        def step():
            v = v0
            for _ in range(6):                                       # a short chain, like a few layers
                v = torch.tanh((W @ v) * 0.02)
            z = (Wi * 6364136223846793005 + 1442695040888963407)
            z = (z ^ (z >> 31)).sum()
            return v, z
        rv, rz = step()
        rvb = rv.view(torch.int32).clone()
        rzi = int(rz.item())
        n = bad_f = bad_i = 0
        samples = []
        t0 = time.time()
        while time.time() - t0 < a.bursty_seconds:
            v, z = step()
            zi = int(z.item())                                       # host round trip (synchronises)
            n += 1
            if zi != rzi:
                bad_i += 1
            if not torch.equal(v.view(torch.int32), rvb):
                bad_f += 1
            if n % 400 == 0:
                time.sleep(0.02)                                     # a short pause now and then, like the host side of a decode loop
        res["parts"]["E_bursty_small_kernels"] = dict(repeats=n, float_mismatches=bad_f, integer_mismatches=bad_i,
                                                      seconds=round(time.time() - t0, 1))
        log("E_bursty_small_kernels: %d repeats, float mismatches %d, integer mismatches %d" % (n, bad_f, bad_i))
        total_bad += bad_f + bad_i
        del W, Wi
        if a.only_bursty:
            res["total_mismatches"] = total_bad
            json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
            log("total mismatches: %d" % total_bad)
            return 1 if total_bad else 0
    if a.power_seconds > 0:
        # P: large half-precision products (tensor cores), no pause: the highest power this check can draw
        Ah = torch.randn(16384, 16384, generator=g).half().to(dev)      # 512 MiB
        Bh = torch.randn(16384, 4096, generator=g).half().to(dev)       # 128 MiB
        Ii = torch.randint(-(1 << 30), 1 << 30, (1 << 24,), generator=g, dtype=torch.int64).to(dev)   # 128 MiB

        def step_p():
            y = Ah @ Bh
            y = torch.tanh(y.float() * 0.001).half()
            y = Ah.t() @ y
            z = Ii * 6364136223846793005 + 1442695040888963407
            return y, (z ^ (z >> 31)).sum()
        ry, rz = step_p()
        ryb = ry.view(torch.int16).clone()
        rzi = int(rz.item())
        n = bad_f = bad_i = 0
        t0 = time.time()
        while time.time() - t0 < a.power_seconds:
            y, z = step_p()
            n += 1
            if int(z.item()) != rzi:
                bad_i += 1
            if not torch.equal(y.view(torch.int16), ryb):
                bad_f += 1
        res["parts"]["P_full_power_half_matmul"] = dict(repeats=n, float_mismatches=bad_f, integer_mismatches=bad_i, seconds=round(time.time() - t0, 1))
        log("P_full_power_half_matmul: %d repeats, float mismatches %d, integer mismatches %d" % (n, bad_f, bad_i))
        total_bad += bad_f + bad_i
        del Ah, Bh, Ii
        if a.only_power:
            res["total_mismatches"] = total_bad
            json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
            log("total mismatches: %d" % total_bad)
            return 1 if total_bad else 0
    # A: float32 matrix products
    A = torch.randn(8192, 8192, generator=g).to(dev)
    X = torch.randn(8192, 512, generator=g).to(dev)

    def part_a():
        y = A @ X
        y = torch.tanh(y * 0.01)
        return A.t() @ y
    total_bad += repeat("A_matmul_float32", part_a, a.seconds)
    # B: elementwise float32
    E = torch.randn(1 << 26, generator=g).to(dev)                 # 256 MiB

    def part_b():
        return torch.tanh(E * 1.0009765625) * E + 0.5
    total_bad += repeat("B_elementwise_float32", part_b, a.seconds)
    del E
    # C: 64-bit integers
    I = torch.randint(-(1 << 62), 1 << 62, (64, 1 << 20), generator=g, dtype=torch.int64).to(dev)   # 512 MiB

    def part_c():
        z = I * 6364136223846793005 + 1442695040888963407
        z = z ^ (z >> 33)
        z = z * -49064778989728563
        return (z ^ (z >> 29)).sum(dim=1)
    total_bad += repeat("C_integer_int64", part_c, a.seconds)
    del I, A, X
    torch.cuda.empty_cache()
    # D: data kept in VRAM
    chunks = []
    n_chunks = max(1, int(a.vram_gib))
    try:
        for i in range(n_chunks):
            c = torch.randint(-(1 << 62), 1 << 62, (1 << 27,), generator=g, dtype=torch.int64).to(dev)      # 1 GiB each
            chunks.append(c)
    except RuntimeError as ex:
        log("stopped allocating at %d GiB: %s" % (len(chunks), str(ex)[:80]))
    ref = [int((c ^ (c >> 17)).sum().item()) for c in chunks]
    bad = 0
    bad_chunks = set()
    t0 = time.time()
    for r in range(a.vram_rounds):
        for i, c in enumerate(chunks):
            if int((c ^ (c >> 17)).sum().item()) != ref[i]:
                bad += 1
                bad_chunks.add(i)
        time.sleep(1.0)
    res["parts"]["D_vram_reread"] = dict(gib=len(chunks), rounds=a.vram_rounds, checks=a.vram_rounds * len(chunks), mismatches=bad,
                                         chunks_with_mismatch=sorted(bad_chunks), seconds=round(time.time() - t0, 1))
    log("D_vram_reread: %d GiB x %d rounds, %d mismatches" % (len(chunks), a.vram_rounds, bad))
    total_bad += bad
    res["total_mismatches"] = total_bad
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    log("total mismatches: %d" % total_bad)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
