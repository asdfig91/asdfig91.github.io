@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
rem If the folder was moved, the Python environment breaks: rebuild it.
".venv\Scripts\python.exe" -c "pass" >nul 2>&1
if errorlevel 1 goto setup
goto run

:setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  pause
  exit /b 1
)

:run
".venv\Scripts\python.exe" -X utf8 make_subtitles.py %*
pause
