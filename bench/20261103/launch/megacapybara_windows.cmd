@echo off
rem MegaCapybara 1.39 (Windows x64, github.com/perkel666/MegaCapybara) as started on 2026-11-03 on the same PC, stock GPU clocks.
rem megacapybara.exe SHA-256 a36600f036e827f881b933a6260dd70977e36ea3f8bbfac463796aacc718d327
rem Tiny model  qwen3.8-27b-tiny.mcapy    16,594,763,776 bytes, SHA-256 5bec1521fd7f1c99a5150ae37dca06122fbf5745f1b7baa167cf20cfd7c41118
rem Drafter     qwen3.8-27b-dflash2.mcapy  1,293,942,784 bytes, SHA-256 e6cb73f63d7c85d8a3f8e15ff222b3d99b375c3437c2d61f031115fb04013854
rem Settings: the official one-conversation preset's defaults with the loop guard switched off. KV 6/4 is the preset default;
rem a second set of runs used --kv-mode 8/8 (results\megacapybara_kv88*). Edit the two folders below first.
set "APP=<megacapybara_dir>\app\MegaCapybara"
set "MDL=<megacapybara_dir>\model"
cd /d "%APP%"
start "mc" /b "%APP%\megacapybara.exe" --model "%MDL%\qwen3.8-27b-tiny.mcapy" --max-concurrency 1 --max-context 32768 --kv-mode 6/4 --rope-scaling off --activations fp8 --speculator dflash2 --draft-len 0 --prefill-policy balanced --loop-guard off --no-vision --ram-tier-gb 0 --vram-reserve-gb 1 --host 127.0.0.1 --port 8097

rem When http://127.0.0.1:8097/v1/models answers, run the clients (from the 20261103\scripts folder):
rem   NInfer's 12 prompts, greedy, 2 runs each:
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8097/v1 --label mc_tiny_kv64 --engine "MegaCapybara 1.39 Tiny kv6/4" --out-dir megacapybara_kv64 --repeats 2 --temperature 0
rem   the three code prompts with answers up to 2,000 tokens:
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8097/v1 --label mc_tiny_kv64_code2000 --engine "MegaCapybara 1.39 Tiny kv6/4" --out-dir megacapybara_kv64_code2000 --cats code --max-new 2000 --repeats 2 --temperature 0
rem   the single recorded runs (video_runs):
rem     python -X utf8 -u stream_capture.py --base http://127.0.0.1:8097/v1 --engine "MegaCapybara_Tiny" --out-dir video_runs --fixtures scenario_structured_jsonl,scenario_story_en_mystery,scenario_code_python,scenario_structured_sql,scenario_translation_zh_en
rem   the own tests:
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8097/v1 --label MegaCapybara_Tiny --out-dir own_tests --cases ..\prompts\cases.json --repeats 2
rem The scripts read NInfer's fixture files from the folder set in FIX at the top of each script (not included here).
