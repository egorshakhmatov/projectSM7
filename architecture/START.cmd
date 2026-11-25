@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py"
    goto finished
)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    py -3.12 "%~dp0run.py"
    goto finished
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    python "%~dp0run.py"
    goto finished
)
echo Python 3.12 was not found. Install Python 3.12 or open docs\overview.html.
pause
exit /b 1
:finished
if errorlevel 1 (
    echo The launcher stopped with an error. See the message above.
    pause
    exit /b 1
)
exit /b 0
