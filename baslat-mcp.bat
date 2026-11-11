@echo off
rem MCP / computer-use smoke: NO UAC elevation so Cursor can click the bot UI.
rem Game must also be non-elevated or bot inputs (Tab/skills) are dropped by UIPI.
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_DIR=%~dp0"
set "PYTHONPATH=%PROJECT_DIR%"
set "PYTHONUTF8=1"
set "GATHERION_HOME=F:\a2"
if not exist "%GATHERION_HOME%\main.py" set "GATHERION_HOME=F:\Gatherion"
if not exist "%GATHERION_HOME%" mkdir "%GATHERION_HOME%" >nul 2>&1
if not exist "%GATHERION_HOME%" (
    echo [UYARI] "%GATHERION_HOME%" olusturulamadi; GATHERION_HOME proje dizinine aliniyor.
    set "GATHERION_HOME=%~dp0"
)
set "TEMP=%GATHERION_HOME%\temp"
set "TMP=%GATHERION_HOME%\temp"
set "YOLO_CONFIG_DIR=%GATHERION_HOME%\appdata"
set "ULTRALYTICS_CONFIG_DIR=%GATHERION_HOME%\appdata"
set "TORCH_HOME=%GATHERION_HOME%\torch"
set "PIP_CACHE_DIR=%GATHERION_HOME%\pip-cache"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%GATHERION_HOME%\logs" mkdir "%GATHERION_HOME%\logs"

set "PYTHONW_EXE=C:\Users\selçuk\AppData\Local\Programs\Python\Python310\pythonw.exe"
if not exist "%PYTHONW_EXE%" set "PYTHONW_EXE=pythonw"
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYTHONW_EXE=%~dp0.venv\Scripts\pythonw.exe"
if exist "%~dp0venv\Scripts\pythonw.exe" set "PYTHONW_EXE=%~dp0venv\Scripts\pythonw.exe"

echo [MCP] Gatherion unelevated (no UAC). Cursor computer-use can click this window.
echo [MCP] If AION2 is elevated, Tab/skills will NOT reach the game — run game without admin.
where "%PYTHONW_EXE%" >nul 2>&1
if errorlevel 1 (
    echo [HATA] pythonw.exe bulunamadi: "%PYTHONW_EXE%"
    exit /b 1
)
start "" /D "%~dp0" "%PYTHONW_EXE%" "%~dp0main.py" %*
timeout /t 3 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root='%~dp0'; $root=$root.TrimEnd('\'); $p=Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*main.py*') -and ($_.CommandLine -like ('*'+$root+'*')) }; if ($p) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [HATA] pythonw baslatilamadi veya hemen kapandi.
    exit /b 1
)
exit /b 0
