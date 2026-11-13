#!/usr/bin/env python3
"""Wait until nothing is using the GPU (2026-11-10).  Samples nvidia-smi every 2 s and returns when the utilisation stayed at or
below --max-util for --hold seconds in a row, or when --timeout seconds have passed.
Exit 0: quiet.  Exit 4: the time limit passed without a quiet period.  Prints one summary line; writes nothing.
usage: python -X utf8 quiet_gate.py [--max-util 1] [--hold 30] [--timeout 1500]"""
import argparse
import datetime as dt
import subprocess
import sys
import time


def util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        u, m = [int(x) for x in out.split(",")]
        return u, m
    except Exception:                                             # noqa: BLE001
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-util", type=int, default=1)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()
    t0 = time.time()
    streak_start = None
    samples = []
    while True:
        u, m = util()
        now = time.time()
        if u is not None:
            samples.append(u)
            if u <= a.max_util:
                streak_start = streak_start or now
                if now - streak_start >= a.hold:
                    break
            else:
                streak_start = None
        if now - t0 > a.timeout:
            print("[%s] quiet gate: NOT quiet after %d s (%d samples, mean %.1f %%, max %d %%, last %s %%)" % (
                dt.datetime.now().strftime("%H:%M:%S"), now - t0, len(samples), sum(samples) / max(1, len(samples)), max(samples or [0]), u))
            return 4
        time.sleep(2)
    tail = samples[-(a.hold // 2):]
    print("[%s] quiet gate: quiet after %d s of waiting (%d samples; before the quiet period mean %.1f %%; the last %d samples: max %d %%; memory used %s MiB)" % (
        dt.datetime.now().strftime("%H:%M:%S"), time.time() - t0, len(samples),
        sum(samples[:-len(tail)] or [0]) / max(1, len(samples) - len(tail)), len(tail), max(tail or [0]), m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
