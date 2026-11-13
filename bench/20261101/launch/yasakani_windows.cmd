@echo off
rem YASAKANI public build 0.1.0.dev0 as measured on 2026-11-01, native on Windows 11.
rem yskn.dll: 26,184,192 bytes, SHA-256 1fa3a25362f150c885ab7eb23fa32e672fbeebf7820c5b19a56ee452227a3718 (also in results\yasakani\A_pub.json).
rem Model folder: qwen38_27b_e3x.yskn (12,365,242,368 bytes), bdraft2_k8.yskn (1,164,969,472), bdraft2_k8.shead (1,411,010,584), kvq_u5.bin (17,432).
rem Edit the three folders below first. One server at a time, nothing else on the GPU. Run from the 20261101\launch folder.
set "MODEL_DIR=<model_dir>"
set "YASAKANI_DIR=<yasakani_dir>"
set "NINFER_CORPUS=<ninfer_corpus_dir>"

rem The server alone (default settings, port 8765). a_speed.py below starts and stops it by itself.
rem   python -X utf8 -u -m yasakani serve --model-dir "%MODEL_DIR%" --dll "%YASAKANI_DIR%\bin\yskn.dll" --port 8765

rem 1. Speed, two back-to-back rounds of the ten tests, loop guard off (sent per request) -> A_pub.json (results\yasakani\)
python -X utf8 -u ..\scripts\a_speed.py run --serve --model-dir "%MODEL_DIR%" --dll "%YASAKANI_DIR%\bin\yskn.dll" --rounds 2 --loop-guard off --tag pub_default --prompts ..\prompts --ninfer-dir "%NINFER_CORPUS%" --out A_pub.json
python -X utf8 ..\scripts\a_speed.py summary A_pub.json --csv A_pub_runs.csv

rem 2. The six thinking tests once more with the loop guard on (the shipped default), one round -> A_pub_guard.json
python -X utf8 -u ..\scripts\a_speed.py run --serve --model-dir "%MODEL_DIR%" --dll "%YASAKANI_DIR%\bin\yskn.dll" --rounds 1 --tests chat,explain,code,math,json,long --loop-guard on --tag pub_guard_on --prompts ..\prompts --ninfer-dir "%NINFER_CORPUS%" --out A_pub_guard.json

rem 3. The published table from the records (faster round of YASAKANI, fastest setting of each other engine):
python ..\scripts\headline.py
