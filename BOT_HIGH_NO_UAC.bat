@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

echo [Gatherion] Bot High (no per-start UAC) launcher
echo   - Task: Gatherion_BotHigh  (Task Scheduler RunLevel=Highest)
echo   - Cift tik / --start : UAC YOK (gorev kayitliyken)
echo   - Ilk --register     : bir kez UAC (Evet) gerekebilir
echo   - Zaten High bot varsa dokunulmaz ( --restart ile yenile )
echo.

rem Default: smart start. Pass-through: --status --register --start --restart --force-register
"%PYTHON_EXE%" "%~dp0tools\bot_elevated_launch.py" %*
set "ERR=%ERRORLEVEL%"
echo.
if "%ERR%"=="0" (
  echo [OK] Cift tik BOT_HIGH_NO_UAC.bat — UAC yok.
  echo      Mobil/remote restart da bu gorevi tercih eder.
) else (
  echo [HATA] Yukaridaki JSON/ERROR satirina bakin.
  echo        Gorev yoksa: BOT_HIGH_NO_UAC.bat --register
  echo        ^(yalnizca kayitta bir kez UAC^)
)
echo.
if /I not "%~1"=="--status" if /I not "%~1"=="--json" pause
exit /b %ERR%
