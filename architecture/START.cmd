@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
    echo SM7 could not start. Details: .runtime\setup.log
    pause
    exit /b 1
)
exit /b 0
