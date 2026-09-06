@echo off
REM Weekday lake refresh — schedule ~07:30 America/New_York.
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\daily_ingest.py
) else (
  python scripts\daily_ingest.py
)
