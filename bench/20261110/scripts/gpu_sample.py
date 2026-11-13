#!/usr/bin/env python3
"""Sample the GPU state once a second (2026-11-10): time, SM clock, memory clock, power, utilisation, temperature, P-state.
  python -X utf8 gpu_sample.py --seconds 300 --out samples.csv [--stop FILE]
Stops after --seconds or when the stop file exists.  Writes only --out."""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop", default="")
    a = ap.parse_args()
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("t_s,clocks_sm_mhz,clocks_mem_mhz,power_w,util_pct,temp_c,pstate\n")
        while time.time() - t0 < a.seconds:
            if a.stop and os.path.exists(a.stop):
                break
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,pstate",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                f.write("%.1f,%s\n" % (time.time() - t0, out.replace(" ", "")))
                f.flush()
            except Exception:                                     # noqa: BLE001
                pass
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
