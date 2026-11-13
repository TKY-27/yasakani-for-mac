@echo off
rem MegaCapybara 1.39 (Windows x64, github.com/perkel666/MegaCapybara), Tiny model, as started on 2026-11-06 for the runs with thinking on.
rem The same executable, model files and settings as on 2026-11-03 (..\..\20261103\launch\megacapybara_windows.cmd has the SHA-256 values):
rem the official one-conversation preset's defaults (KV 6/4) with the loop guard switched off. The runs with thinking off are the ones of
rem 2026-11-03 (..\..\20261103\results\megacapybara_kv64 and own_tests). Edit the two folders below first.
set "APP=<megacapybara_dir>\app\MegaCapybara"
set "MDL=<megacapybara_dir>\model"
cd /d "%APP%"
start "mc" /b "%APP%\megacapybara.exe" --model "%MDL%\qwen3.8-27b-tiny.mcapy" --max-concurrency 1 --max-context 32768 --kv-mode 6/4 --rope-scaling off --activations fp8 --speculator dflash2 --draft-len 0 --prefill-policy balanced --loop-guard off --no-vision --ram-tier-gb 0 --vram-reserve-gb 1 --host 127.0.0.1 --port 8097

rem When http://127.0.0.1:8097/v1/models answers, run the clients (from the 20261108\scripts folder), thinking on, 2 runs
rem (results\megacapybara_tiny_think):
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8097/v1 --label megacapybara_tiny_think --engine "MegaCapybara 1.39 Tiny kv6/4 thinking" --out-dir megacapybara_tiny_think\n12 --repeats 2 --temperature 1.0 --no-thinking-kw --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8097/v1 --label megacapybara_tiny_think --out-dir megacapybara_tiny_think\own --cases ..\..\20261103\prompts\cases.json --repeats 2 --temperature 1.0 --thinking on --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
rem The server was stopped with: taskkill /im megacapybara.exe /f
