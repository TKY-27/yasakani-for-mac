#!/usr/bin/env python3
"""Background GPU load for a test (2026-11-10): another process that uses the same GPU in short bursts, to see whether sharing
the GPU changes how often YASAKANI's repeated runs differ.  Uses the PyTorch that is already installed; about 0.3 GiB of VRAM.
  python -X utf8 gpu_bg_load.py --seconds 240 --work-ms 3 --sleep-ms 20     (light: short bursts)
  python -X utf8 gpu_bg_load.py --seconds 240 --work-ms 20 --sleep-ms 0     (heavy: no pause)
Stops after --seconds, or as soon as the file given with --stop exists.  Prints one line at the end."""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument("--work-ms", type=float, default=3.0)
    ap.add_argument("--sleep-ms", type=float, default=20.0)
    ap.add_argument("--stop", default="")
    a = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    dev = torch.device("cuda")
    e = torch.randn(1 << 24, device=dev)                          # 64 MiB
    t0 = time.time()
    bursts = 0
    while time.time() - t0 < a.seconds:
        if a.stop and os.path.exists(a.stop):
            break
        t1 = time.perf_counter()
        while (time.perf_counter() - t1) * 1000.0 < a.work_ms:
            y = torch.tanh(e * 1.0009765625) * e + 0.5
            torch.cuda.synchronize()
        bursts += 1
        if a.sleep_ms > 0:
            time.sleep(a.sleep_ms / 1000.0)
    print("background load: %d bursts in %.0f s (work %.1f ms, sleep %.1f ms)" % (bursts, time.time() - t0, a.work_ms, a.sleep_ms), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
