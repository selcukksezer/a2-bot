@echo off
rem Konsol isteyerek acilir. Normal baslatma: baslat.bat (pythonw, pencere kapanir)
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
set "GATHERION_LOG=%GATHERION_HOME%\logs\gatherion.log"
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
title Gatherion v2.0 Konsol - Aion 2 Otomatik Toplayici ve Rota Botu
color 0B

echo ============================================================================
echo               GATHERION v2.0 - AION 2 AKILLI TOPLAYICI BOT
echo ============================================================================
echo   [+] 3x Materyal Toplama Dongusu (1/3 -^> 2/3 -^> 3/3)
echo   [+] Hibrit Oto-Haritalama (Sag Ust Minimap + 3B Dunya Aria/Mela/[E]/[F])
echo   [+] Mobil Web Kumanda Paneli + Kendi Kendini Egiten YOLO Veri Fabrikasi
echo   [+] %%100 Turkce Arayuz
echo ============================================================================
echo.

set "PYTHON_EXE=C:\Users\selçuk\AppData\Local\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

net session >nul 2>&1
if %errorLevel% == 0 goto :ADMIN_OK

echo [UYARI] Bot yonetici olarak calistirilmiyor!
echo         Yonetici (Administrator) yetkisi isteniyor...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs" >nul 2>&1
if %errorLevel% == 0 exit /b 0
echo [UYARI] Yonetici yetkisi alinamadi, mevcut yetkilerle devam ediliyor...

:ADMIN_OK
cd /d "%~dp0"
echo [TAMAM] Calisma Dizini : %CD%
echo [BILGI] Python Yorumlayicisi : "%PYTHON_EXE%"

echo [STARTUP] on-kontrol basladi %TIME%
rem Hizli on-kontrol: importlib.util.find_spec modul kodunu CALISTIRMAZ, DLL yuklemez;
rem yalnizca modulun bulunup bulunmadigina bakar (tam import yerine).
rem Eksik modul varsa pip paket adi "%TEMP%\missing_pkgs.txt" dosyasina yazilir.
"%PYTHON_EXE%" -c "import importlib.util,sys; mods={'PyQt6':'PyQt6','cv2':'opencv-python','mss':'mss','numpy':'numpy','pydirectinput':'pydirectinput','keyboard':'keyboard','yaml':'pyyaml','onnxruntime':'onnxruntime-gpu','psutil':'psutil','serial':'pyserial'}; missing=[pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None]; print(' '.join(missing)); sys.exit(1 if missing else 0)" > "%TEMP%\missing_pkgs.txt" 2>nul
if %errorLevel% == 0 goto :PRECHECK_DONE
set "MISSING_PKGS="
set /p MISSING_PKGS=<"%TEMP%\missing_pkgs.txt"
if "%MISSING_PKGS%" == "" (
    echo [HATA] On-kontrol calistirilamadi; eksik paket listesi alinamadi.
    >>"%GATHERION_LOG%" echo [%DATE% %TIME%] [HATA] On-kontrol calistirilamadi; bot baslatilmadi.
    pause
    exit /b 1
)
echo [BILGI] Eksik Python paketleri tespit edildi: %MISSING_PKGS%
echo [BILGI] Yalnizca eksik paketler kuruluyor (requirements.txt tam kurulum yapilmiyor)...
"%PYTHON_EXE%" -m pip install %MISSING_PKGS%
if errorlevel 1 (
    echo [HATA] Eksik paket kurulumu basarisiz oldu; bot baslatilmiyor.
    >>"%GATHERION_LOG%" echo [%DATE% %TIME%] [HATA] Eksik paket kurulumu basarisiz oldu; bot baslatilmadi.
    pause
    exit /b 1
)

:PRECHECK_DONE
del "%TEMP%\missing_pkgs.txt" >nul 2>&1
echo [STARTUP] on-kontrol bitti %TIME%

:RUN_BOT
echo [BILGI] Gatherion v2.0 konsol modunda baslatiliyor...
echo         - Oyun icinde [F10] tusuna basarak Botu Aninda Durdurabilir / Baslatabilirsiniz.
echo         - Oyun icinde [F9] tusuna basarak Hizli Kontrol Menusunu (Savas / Toplama) acabilirsiniz.
echo         - Oyun icinde [F8] tusuna basarak kaynak noktasini kaydedebilirsiniz.
echo ============================================================================
echo.

echo [STARTUP] main.py baslatiliyor %TIME%
"%PYTHON_EXE%" "%~dp0main.py" %*
set "APP_EXIT_CODE=%errorLevel%"

if "%APP_EXIT_CODE%" == "0" goto :END_SCRIPT
color 0C
echo.
echo [HATA] Uygulama beklenmeyen bir hata kodu ile kapandi (Kod: %APP_EXIT_CODE%).
echo        Detaylar icin %GATHERION_HOME%\logs\bot.log dosyasini inceleyebilirsiniz.
pause

:END_SCRIPT
exit /b %APP_EXIT_CODE%
