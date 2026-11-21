@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

echo [Gatherion] Steam Medium (unelevated) launcher
echo   - No UAC prompt
echo   - Uses Task Scheduler RunLevel=Limited so Steam stays Medium
echo   - Even if this window is elevated, Steam will NOT inherit High
echo.

"%PYTHON_EXE%" "%~dp0tools\steam_unelevated_launch.py" --restart %*
set "ERR=%ERRORLEVEL%"
echo.
if "%ERR%"=="0" (
  echo [OK] Steam Medium. AION2 icin: STEAM_UNELEVATED.bat --aion2
  echo      Bot icin unelevated: baslat-mcp.bat  ^(baslat.bat yonetici ister^)
) else (
  echo [HATA] Basarisiz. Yukaridaki JSON/ERROR satirina bakin.
  echo        Steam High ise once elevated Task Manager ile steam.exe oldurun.
)
echo.
pause
exit /b %ERR%
