@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_DIR=%~dp0"
set "PYTHONPATH=%PROJECT_DIR%"
set "PYTHONUTF8=1"
set "GATHERION_HOME=F:\a2"
if not exist "%GATHERION_HOME%\main.py" set "GATHERION_HOME=F:\Gatherion"
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
set "GATHERION_LOG=%GATHERION_HOME%\logs\gatherion.log"

set "PYTHON_EXE=C:\Users\selçuk\AppData\Local\Programs\Python\Python310\python.exe"
set "PYTHONW_EXE=C:\Users\selçuk\AppData\Local\Programs\Python\Python310\pythonw.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if not exist "%PYTHONW_EXE%" set "PYTHONW_EXE=pythonw"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYTHONW_EXE=%~dp0.venv\Scripts\pythonw.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\pythonw.exe" set "PYTHONW_EXE=%~dp0venv\Scripts\pythonw.exe"

net session >nul 2>&1
if %errorLevel% == 0 goto :ADMIN_OK

echo [UYARI] Bot yonetici olarak calistirilmiyor!
echo         Yonetici (Administrator) yetkisi isteniyor...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs" >nul 2>&1
if %errorLevel% == 0 exit /b 0
echo [UYARI] Yonetici yetkisi alinamadi, mevcut yetkilerle devam ediliyor...

:ADMIN_OK
cd /d "%~dp0"
if not exist "%PYTHONW_EXE%" (
    echo [HATA] pythonw.exe bulunamadi: "%PYTHONW_EXE%"
    echo        Konsol ile baslatmak icin baslat-konsol.bat kullanin.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import PyQt6, cv2, mss, numpy, pydirectinput, keyboard, PIL" >nul 2>&1
if %errorLevel% == 0 goto :RUN_BOT

echo [BILGI] Eksik Python paketleri tespit edildi, requirements.txt yukleniyor...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"

:RUN_BOT
set "WORKDIR=%~dp0"
set "WORKDIR=%WORKDIR:~0,-1%"
>>"%GATHERION_LOG%" echo [%DATE% %TIME%] baslat.bat pythonw "%PYTHONW_EXE%" cwd "%WORKDIR%"

rem Elevated relaunch: Medium shell cannot kill High pythonw (UIPI). Admin baslat.bat can.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root='%~dp0'; $root=$root.TrimEnd('\'); Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" -ErrorAction SilentlyContinue | ForEach-Object { $cl=$_.CommandLine; if ($cl -and ($cl -like '*main.py*') -and ($cl -like ('*'+$root+'*') -or $cl -like '*Gatherion*' -or $cl -like '*\a2\*')) { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Add-Content -Path '%GATHERION_LOG%' -Value ('[baslat] killed stale pythonw PID='+$_.ProcessId) } catch {} } }" >nul 2>&1
timeout /t 2 /nobreak >nul

start "" /D "%WORKDIR%" "%PYTHONW_EXE%" "%~dp0main.py" %*
if errorlevel 1 (
    echo [HATA] pythonw baslatilamadi. Ayrinti: "%GATHERION_LOG%"
    echo        Konsol icin baslat-konsol.bat
    pause
    exit /b 1
)
exit /b 0
