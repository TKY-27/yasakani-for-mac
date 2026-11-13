@echo off
rem YASAKANI as measured for the speed tables on 2026-11-08: the two model folders of this release with the yskn.dll built on 2026-11-07
rem (SHA-256 0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb, the DLL of this release), yasakani 0.1.0.dev0, native on
rem Windows 11, stock GPU clocks, power limit 600 W.
rem   fastest model  qwen38_27b_speed.yskn    10,311,344,128 bytes, SHA-256 begins 3ba17a76871a2b5e   (records: ysk_speed_v8)
rem   quality model  qwen38_27b_quality.yskn  12,365,242,368 bytes, SHA-256 begins 6493983cf9539182   (records: ysk_quality_v8)
rem Edit the folders below first. Run the clients from the 20261108\scripts folder.
set "MODEL_DIR_FASTEST=<model_dir_fastest>"
set "MODEL_DIR_QUALITY=<model_dir_quality>"
set "YASAKANI_DIR=<yasakani_dir>"
set "PYTHONPATH=<yasakani_py>"
set "DLL=%YASAKANI_DIR%\bin\yskn.dll"

rem One server at a time, nothing else on the GPU. Wait until http://127.0.0.1:8099/health says ok:
start "ysk" /b python -X utf8 -u -m yasakani serve --model-dir "%MODEL_DIR_FASTEST%" --dll "%DLL%" --host 127.0.0.1 --port 8099 --max-tokens-cap 16384

rem run_wf_n12.py (plans speedv8, speedv8t, qualv8, qualv8t) started the server, waited 5 s, sent two warm-up requests
rem ("Write two sentences about the sea.", 128 tokens, with capture_stream.py of 20261101\scripts) and then ran the two clients:
rem   thinking off, greedy, 2 runs (results\n12_ysk_speed_v8):
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8099/v1 --label ysk_speed_v8 --engine ysk_speed_v8 --out-dir n12_ysk_speed_v8\n12 --repeats 2 --temperature 0
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8099/v1 --label ysk_speed_v8 --out-dir n12_ysk_speed_v8\own --cases ..\..\20261103\prompts\cases.json --repeats 2
rem   thinking on, runs 0 and 1 (results\n12t_ysk_speed_v8), after a new start of the server:
rem     python -X utf8 -u oai_bench_client.py --base http://127.0.0.1:8099/v1 --label ysk_speed_v8 --engine ysk_speed_v8 --out-dir n12t_ysk_speed_v8\n12 --repeats 2 --temperature 1.0 --no-thinking-kw --extra "{\"chat_template_kwargs\": {\"enable_thinking\": true}, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0}" --seed-per-run
rem     python -X utf8 -u custom_bench.py --base http://127.0.0.1:8099/v1 --label ysk_speed_v8 --out-dir n12t_ysk_speed_v8\own --cases ..\..\20261103\prompts\cases.json --repeats 2 --temperature 1.0 --thinking on --extra "{\"chat_template_kwargs\": {\"enable_thinking\": true}, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0}" --seed-per-run
rem   thinking on, runs 2, 3 and 4 (results_seeds_2_4\n12t_ysk_speed_v8), think_more.py, after a new start of the server:
rem     python -X utf8 -u oai_bench_client_seedoff.py (the same arguments) --repeats 3 --seed-offset 2
rem     python -X utf8 -u custom_bench_seedoff.py (the same arguments) --repeats 3 --seed-offset 2
rem The quality model was run the same way (--model-dir "%MODEL_DIR_QUALITY%", label ysk_quality_v8).
rem The scripts read NInfer's fixture files from the folder set in FIX at the top of oai_bench_client.py (not included here).
