@echo off
cd /d "%~dp0"
set "PROJECT_DIR=%~dp0"
set "PYTHONPATH=%PROJECT_DIR%"
set "PYTHONUTF8=1"
title Gatherion v2.0 - Aion 2 Otomatik Toplayici ve Rota Botu
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

set "PYTHON_EXE=python"
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

"%PYTHON_EXE%" -c "import PyQt6, cv2, mss, numpy, pydirectinput, keyboard, PIL" >nul 2>&1
if %errorLevel% == 0 goto :RUN_BOT

echo [BILGI] Eksik Python paketleri tespit edildi, requirements.txt yukleniyor...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"

:RUN_BOT
echo [BILGI] Gatherion v2.0 baslatiliyor...
echo         - Oyun icinde [F9] tusuna basarak Hizli Kontrol Menusunu acabilirsiniz.
echo         - Oyun icinde [F8] tusuna basarak kaynak noktasini kaydedebilirsiniz.
echo ============================================================================
echo.

"%PYTHON_EXE%" "%~dp0main.py" %*
set "APP_EXIT_CODE=%errorLevel%"

if "%APP_EXIT_CODE%" == "0" goto :END_SCRIPT
color 0C
echo.
echo [HATA] Uygulama beklenmeyen bir hata kodu ile kapandi (Kod: %APP_EXIT_CODE%).
echo        Detaylar icin logs\bot.log dosyasini inceleyebilirsiniz.
pause

:END_SCRIPT
exit /b %APP_EXIT_CODE%
