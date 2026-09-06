@echo off
REM GIG 24/7 paper terminal — double-click or point Task Scheduler / NSSM here.
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\run_24_7.py --trade-yes %*
) else (
  python scripts\run_24_7.py --trade-yes %*
)
