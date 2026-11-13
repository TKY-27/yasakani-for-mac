@echo off
rem ignis v0.9.6 (Windows release, github.com/gpillon/ignis) as started on 2026-11-06 on the same PC, stock GPU clocks, power limit 600 W.
rem ignis-v0.9.6-x86_64-pc-windows-msvc.zip  SHA-256 3436aaf28e5a6951354d1a4f07527381b35b042811927a16d18a2a25c176b01b
rem Model qwen3_8_27b_nvfp4full-v2.ninfer (its catalog model), 19,406,942,468 bytes, SHA-256 abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af
rem Settings: the example of its README without vision: 512K context, DFlash 2 with 7 draft tokens. Edit the folder below first.
set "T=<ignis_dir>"
set "APP=%T%\ignis-v0.9.6-x86_64-pc-windows-msvc"
cd /d "%APP%"
start "ignis" /b "%APP%\ignis-server.exe" --download-path "%T%" --model-max-context 524288 --model-rope-scaling yarn:2 --spec-backend dflash2 --spec-draft-tokens 7

rem When http://127.0.0.1:8000/v1/models lists the model, run the clients (from the 20261108\scripts folder):
rem   thinking off, greedy, 2 runs (results\ignis_v096):
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8000/v1 --label ignis_v096 --engine "ignis v0.9.6 nvfp4full dflash2-7" --out-dir ignis_v096\n12 --repeats 2 --temperature 0
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8000/v1 --label ignis_v096 --out-dir ignis_v096\own --cases ..\..\20261103\prompts\cases.json --repeats 2
rem   thinking on, 2 runs (results\ignis_v096_think), after a new start of the server:
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8000/v1 --label ignis_v096_think --engine "ignis v0.9.6 nvfp4full dflash2-7 thinking" --out-dir ignis_v096_think\n12 --repeats 2 --temperature 1.0 --no-thinking-kw --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8000/v1 --label ignis_v096_think --out-dir ignis_v096_think\own --cases ..\..\20261103\prompts\cases.json --repeats 2 --temperature 1.0 --thinking on --extra "{\"chat_template_kwargs\":{\"enable_thinking\":true},\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --seed-per-run
rem The server was stopped with: taskkill /im ignis-server.exe /f
