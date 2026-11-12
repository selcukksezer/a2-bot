@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [YOLO UI] Tasarim dali: F:\a2 (fix/v20-birlesik) — onayla-r4 filmstrip.
echo          freeze-cumulative eski fabrika UI gosterir; yeni tasarim icin BU bat.
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
where "%PYTHONW_EXE%" >nul 2>&1
if errorlevel 1 (
    echo [HATA] pythonw.exe bulunamadi: "%PYTHONW_EXE%"
    echo        Konsol ile baslatmak icin baslat-konsol.bat kullanin.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import PyQt6, cv2, mss, numpy, pydirectinput, keyboard, yaml, onnxruntime, psutil, serial" >nul 2>&1
if %errorLevel% == 0 goto :RUN_BOT

echo [BILGI] Eksik Python paketleri tespit edildi, requirements.txt yukleniyor...
rem sweep-26 F11: kurulum yalnizca import kontrolu basarisizsa calisir;
rem requirements.txt alt sinir (>=) icerir, surum pin'i yok — taze
rem kurulumda en yeni uygun surum gelir. Basarisizlik loga da yazilir.
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [HATA] requirements.txt kurulumu basarisiz oldu; bot baslatilmiyor.
    >>"%GATHERION_LOG%" echo [%DATE% %TIME%] [HATA] requirements.txt kurulumu basarisiz oldu; bot baslatilmadi.
    pause
    exit /b 1
)

:RUN_BOT
set "WORKDIR=%~dp0"
set "WORKDIR=%WORKDIR:~0,-1%"
>>"%GATHERION_LOG%" echo [%DATE% %TIME%] baslat.bat pythonw "%PYTHONW_EXE%" cwd "%WORKDIR%"

rem sweep-26 F4: eski "stale kill" blogu kaldirildi — calisan saglikli botu
rem olduruyor, iki baslat.bat birbirini oldurebiliyordu. Tek instance
rem artik main.py'deki CreateMutex kilidiyle korunuyor; ikinci ornek
rem kendisi cikip "zaten calisiyor" diyor, eskisi oldurulmuyor.

start "" /D "%WORKDIR%" "%PYTHONW_EXE%" "%~dp0main.py" %*
rem sweep-26 F4/F9: 3 sn'lik tek kontrol yetersizdi — yavas acilista
rem yanlis alarm, 3 sn sonrasi cokuste sessiz "basarili" veriyordu.
rem Iki asama: 20 sn icinde surec gorunsun, 10 sn daha yasasin.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root='%~dp0'; $root=$root.TrimEnd('\'); $found=$false; for ($i=0; $i -lt 20; $i++) { $p=Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*main.py*') -and ($_.CommandLine -like ('*'+$root+'*')) }; if ($p) { $found=$true; break }; Start-Sleep -Seconds 1 }; if (-not $found) { exit 1 }; Start-Sleep -Seconds 10; $p2=Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*main.py*') -and ($_.CommandLine -like ('*'+$root+'*')) }; if ($p2) { exit 0 } else { exit 2 }" >nul 2>&1
if errorlevel 2 (
    echo [HATA] pythonw basladi ama 10 sn icinde kapandi. Ayrinti: "%GATHERION_LOG%"
    echo        Ikinci ornek olabilir (Gatherion zaten calisiyor olabilir) ya da baslangic hatasi.
    pause
    exit /b 1
)
if errorlevel 1 (
    echo [HATA] pythonw baslatilamadi veya 20 sn icinde gorunmedi. Ayrinti: "%GATHERION_LOG%"
    echo        Konsol icin baslat-konsol.bat
    pause
    exit /b 1
)
exit /b 0
