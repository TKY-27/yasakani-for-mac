@echo off
rem The checks of 2026-11-10 on the files of this release: yskn.dll (SHA-256 0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb),
rem the Python package 0.1.0.dev0 and the two model folders. These are the command lines that ran, with the folders replaced.
rem "speed" in the file names = the fastest model, "quality" = the quality model. Edit the folders below first.
set "MODEL_DIR_FASTEST=<model_dir_fastest>"
set "MODEL_DIR_QUALITY=<model_dir_quality>"
set "DLL=<yasakani_dir>\bin\yskn.dll"
set "PYTHONPATH=<yasakani_py>"
set "S=<scripts_dir>"
rem <scripts_dir> holds repeat_probe.py, quiet_gate.py, gpu_consistency.py, gpu_bg_load.py, gpu_sample.py (20261110\scripts) together with
rem onoff_eq.py and a_speed.py (20261101\scripts); the prompt folders are set near the top of main() in onoff_eq.py and repeat_probe.py.
set "M=%MODEL_DIR_FASTEST%"

rem 1. info, verify, selfcheck, bench (checks\selfcheck; the same for the quality model)
python -X utf8 -m yasakani info --model-dir "%M%" --dll "%DLL%" > speed_info.log 2>&1
python -X utf8 -m yasakani verify --model-dir "%M%" > speed_verify.log 2>&1
python -X utf8 -u -m yasakani selfcheck --model-dir "%M%" --dll "%DLL%" --write speed_selfcheck_ref.json > speed_sc_write.log 2>&1
python -X utf8 -u -m yasakani selfcheck --model-dir "%M%" --dll "%DLL%" --reference speed_selfcheck_ref.json > speed_sc_again.log 2>&1
python -X utf8 -u -m yasakani selfcheck --model-dir "%M%" --dll "%DLL%" --max-ctx 16384 --reference speed_selfcheck_ref.json > speed_sc_ctx16k.log 2>&1
python -X utf8 -u -m yasakani bench --model-dir "%M%" --dll "%DLL%" --max-ctx 16384 --n 256 --runs 2 --json > speed_bench.json 2> speed_bench.err
rem    speed_sc_fresh.log: after selfcheck.json was put into the model folder, six times in new processes:
rem      python -X utf8 -u -m yasakani selfcheck --model-dir "%M%" --dll "%DLL%"

rem 2. speculative decoding on against off (checks\on_off; the same for the quality model)
python -X utf8 -u "%S%\onoff_eq.py" --model-dir "%M%" --dll "%DLL%" --pypath "%PYTHONPATH%" --out speed_H_onoff.json > speed_H_onoff.log 2>&1

rem 3. the same request repeated (checks\repeat)
rem    the full plan, both models (speed_repeat_probe, quality_repeat_probe):
python -X utf8 -u "%S%\repeat_probe.py" --model-dir "%M%" --dll "%DLL%" --pypath "%PYTHONPATH%" --tools "%S%" --out speed_repeat_probe.json > speed_repeat_probe.log 2>&1
rem    the short plan (four requests without drafts and one with the default settings, 12 repeats each), fastest model:
set "PR=python -X utf8 -u "%S%\repeat_probe.py" --model-dir "%M%" --dll "%DLL%" --pypath "%PYTHONPATH%" --tools "%S%" --plan off4 --reps 12"
%PR% --out env_A_base.json > env_A_base.log 2>&1
%PR% --out env_F_base2.json > env_F_base2.log 2>&1
%PR% --max-ctx 262144 --out ctl_G_ctx262144.json > ctl_G_ctx262144.log 2>&1
%PR% --max-ctx 16384 --out ctl_H_ctx16384.json > ctl_H_ctx16384.log 2>&1
%PR% --warmup --out ctl_I_warmup.json > ctl_I_warmup.log 2>&1
%PR% --no-state-hash --out ctl_J_nohash.json > ctl_J_nohash.log 2>&1
rem    with the GPU state sampled once a second (clk_engine_off4.csv):
rem      start "" /b python -X utf8 "%S%\gpu_sample.py" --seconds 400 --out clk_engine_off4.csv --stop clk_engine.stop
%PR% --out clk_engine_off4.json > clk_engine_off4.log 2>&1
rem    after the GPU had been unused for 30 s (quiet_1 ... quiet_4):
python -X utf8 "%S%\quiet_gate.py" --max-util 1 --hold 30 --timeout 1500
%PR% --out quiet_1.json > quiet_1.log 2>&1
rem    while another process used the GPU (share_S1_bursts: 3 ms of work every 23 ms; share_S2_nopause: without pause):
rem      start "" /b python -X utf8 "%S%\gpu_bg_load.py" --seconds 600 --work-ms 3 --sleep-ms 20 --stop bg_S1.stop > bg_S1.log 2>&1
rem      %PR% --max-gpu-used-mib 9000 --max-gpu-util 100 --out share_S1_bursts.json > share_S1_bursts.log 2>&1
rem      start "" /b python -X utf8 "%S%\gpu_bg_load.py" --seconds 900 --work-ms 20 --sleep-ms 0 --stop bg_S2.stop > bg_S2.log 2>&1
rem      %PR% --max-gpu-used-mib 9000 --max-gpu-util 100 --out share_S2_nopause.json > share_S2_nopause.log 2>&1

rem 4. the GPU itself, with PyTorch 2.10.0+cu128 and no YASAKANI code (checks\gpu)
python -X utf8 -u "%S%\gpu_consistency.py" --out gpu_consistency.json --seconds 70 --vram-gib 12 > gpu_consistency.log 2>&1
python -X utf8 -u "%S%\gpu_consistency.py" --out gpu_consistency_bursty.json --bursty-seconds 240 --only-bursty > gpu_consistency_bursty.log 2>&1
rem    two processes at once at the power limit (gpu_consistency_power_a, _b), GPU state in clk_torch_power.csv:
rem      start "" /b python -X utf8 -u "%S%\gpu_consistency.py" --out gpu_consistency_power_b.json --power-seconds 235 --only-power
python -X utf8 -u "%S%\gpu_consistency.py" --out gpu_consistency_power_a.json --power-seconds 240 --only-power > gpu_consistency_power_a.log 2>&1
