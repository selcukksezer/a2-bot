@echo off
REM Gatherion Pico flash helper wrapper (no UAC).
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flash_pico.ps1" %*
exit /b %ERRORLEVEL%
