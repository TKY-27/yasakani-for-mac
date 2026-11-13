@echo off
rem YASAKANI public candidate build of 2026-11-03 (yasakani 0.1.0.dev0; yskn.dll SHA-256 begins d86fe2d0b2e0f27c, see logs\client\yasakani_build_info.log),
rem native on Windows 11, stock GPU clocks. "std" = the standard model folder (12.4 GB, the same file names as on 2026-11-01);
rem "fast" = the fast model folder (pure 3-bit weights, 11.7 GB). Edit the folders below first. Run from the 20261103\scripts folder.
set "MODEL_DIR=<model_dir>"
set "MODEL_DIR_FAST=<model_dir_fast>"
set "YASAKANI_DIR=<yasakani_dir>"
set "PYTHONPATH=<yasakani_py>"
set "DLL=%YASAKANI_DIR%\bin\yskn.dll"

rem In-process runs on NInfer's 12 prompts (Python API, no server):
python -X utf8 -u pub_corpus_greedy.py --suite scenario --model-dir "%MODEL_DIR%" --dll "%DLL%" --out-dir yasakani_std_greedy --seeds 0,1 --temperature 0 --label greedy
python -X utf8 -u pub_corpus_cond.py --suite scenario --model-dir "%MODEL_DIR%" --dll "%DLL%" --out-dir yasakani_std_greedy_kv8 --seeds 0,1 --temperature 0 --env YSKN_TB_CAP=0 --label kv8_greedy
python -X utf8 -u pub_corpus_cond2.py --suite scenario --cats code --max-new 2000 --model-dir "%MODEL_DIR%" --dll "%DLL%" --out-dir yasakani_std_code2000 --seeds 0,1 --temperature 0 --label mc_code2000_shipped
python -X utf8 -u pub_corpus_cond2.py --suite scenario --cats code --max-new 2000 --model-dir "%MODEL_DIR%" --dll "%DLL%" --out-dir yasakani_std_code2000_kv8 --seeds 0,1 --temperature 0 --env YSKN_TB_CAP=0 --label mc_code2000_kv8
python -X utf8 -u pub_corpus_direct.py --suite scenario --model-dir "%MODEL_DIR%" --dll "%DLL%" --out-dir yasakani_std_t06
python -X utf8 -u pub_corpus_cond2.py --suite scenario --model-dir "%MODEL_DIR_FAST%" --dll "%DLL%" --out-dir yasakani_fast_greedy --seeds 0,1 --temperature 0 --label fast_greedy
python -X utf8 -u pub_corpus_cond2.py --suite scenario --cats code --max-new 2000 --model-dir "%MODEL_DIR_FAST%" --dll "%DLL%" --out-dir yasakani_fast_code2000 --seeds 0,1 --temperature 0 --label fast_code2000
python -X utf8 -u pub_corpus_cond2.py --suite scenario --model-dir "%MODEL_DIR_FAST%" --dll "%DLL%" --out-dir yasakani_fast_t06 --seeds 0,1,2,3,4 --temperature 0.6 --label fast_t06

rem Through the OpenAI-compatible server (single recorded runs and own tests). Start one server, wait until /health says ok:
rem   start "ysk" /b python -X utf8 -u -m yasakani serve --model-dir "%MODEL_DIR_FAST%" --dll "%DLL%" --host 127.0.0.1 --port 8098 --max-tokens-cap 8192
rem   python -X utf8 -u stream_capture.py --base http://127.0.0.1:8098/v1 --engine "YASAKANI_fast" --out-dir video_runs --fixtures scenario_structured_jsonl,scenario_story_en_mystery,scenario_code_python,scenario_structured_sql,scenario_translation_zh_en
rem   (own tests: the same server with --max-tokens-cap 16384, then)
rem   python -X utf8 -u custom_bench.py --base http://127.0.0.1:8098/v1 --label YASAKANI_fast --out-dir own_tests --cases ..\prompts\cases.json --repeats 2
rem The standard model was run the same way on port 8099 (--model-dir "%MODEL_DIR%", labels YASAKANI_std).
