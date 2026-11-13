@echo off
setlocal
title YASAKANI quickstart
cd /d "%~dp0"
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto run
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto run
echo.
echo  Python 3.10 or newer was not found on this computer.
echo.
echo  1. Install Python from https://www.python.org/downloads/windows/
echo     (tick "Add python.exe to PATH" in the installer).
echo  2. Double-click START.cmd again.
echo.
pause
exit /b 1
:run
%PY% "%~dp0quickstart.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
