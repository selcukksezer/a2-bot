@echo off
chcp 65001 >nul
cd /d "%~dp0"
net session >nul 2>&1
if %errorLevel% == 0 goto :GO
echo [Gatherion] Yönetici izni gerekli — UAC'de Evet deyin.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -WorkingDirectory '%~dp0' -Verb RunAs"
exit /b 0

:GO
echo [Gatherion] Eski process temizleniyor + güncel kodla başlatılıyor...
call "%~dp0baslat.bat"
exit /b %errorLevel%
