@echo off
rem mkyt/qwen3-27b-5090 v0.3.2, the native Windows package (no Docker), as started on 2026-11-06 on the same PC, stock GPU clocks.
rem qwen3-27b-5090-v0.3.2-windows-x64.zip  SHA-256 9e4b8868b95567185b87b9f77da9def0f5e6c9f16b20fac31838f3bd65ee5295
rem Weights as downloaded by the package: juzbox/Qwen3.8-27B-NVFP4-GDN (revision 3e68640) and incoai/Qwen3.8-27B-DFlash2 (revision 015e795).
rem Default settings (gdnfp4 weights, NVFP4 KV, DFlash 2 draft, prefix cache on). The power limit was left at 600 W; mkyt's own published
rem numbers are at 400 W. Edit the folder below first.
set "ROOT=<mkyt_dir>"
cd /d "%ROOT%"
start "mkyt" /b "%ROOT%\python\python.exe" app\scripts\start_package.py

rem When http://127.0.0.1:8000/health answers ok, run the clients (from the 20261108\scripts folder):
rem   thinking off, greedy, 2 runs (results\mkyt_v032):
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8000/v1 --label mkyt_v032 --engine "mkyt v0.3.2 gdnfp4 nvfp4-kv dflash2" --out-dir mkyt_v032\n12 --repeats 2 --temperature 0
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8000/v1 --label mkyt_v032 --out-dir mkyt_v032\own --cases ..\..\20261103\prompts\cases.json --repeats 2
rem   thinking on, 2 runs (results\mkyt_v032_think), after a new start of the server:
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8000/v1 --label mkyt_v032_think --engine "mkyt v0.3.2 gdnfp4 nvfp4-kv dflash2 thinking" --out-dir mkyt_v032_think\n12 --repeats 2 --temperature 1.0 --no-thinking-kw --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8000/v1 --label mkyt_v032_think --out-dir mkyt_v032_think\own --cases ..\..\20261103\prompts\cases.json --repeats 2 --temperature 1.0 --thinking on --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
