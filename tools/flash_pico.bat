@echo off
REM Gatherion Pico flash helper wrapper (no UAC).
cd /d "%~dp0\.."
if not exist "%~dp0flash_pico.ps1" (
  echo FAIL: missing flash_pico.ps1: %~dp0flash_pico.ps1
  exit /b 2
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flash_pico.ps1" %*
exit /b %ERRORLEVEL%
