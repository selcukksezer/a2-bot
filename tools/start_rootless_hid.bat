@echo off
REM 1B rootless KEY bridge → Interception (stock phone is NOT USB HID).
REM Goal stays OPEN until 1A Magisk/hidg0 proves. Do not run Magisk from here.
setlocal EnableExtensions
cd /d "%~dp0.."
echo === Gatherion 1B rootless HID ===
set PY=python
where py >nul 2>&1 && set PY=py -3
%PY% tools\interception_status.py
if errorlevel 1 (
  echo WARN: Interception not ready — bridge starts with --no-soft, so keys will NOT reach AION2.
  echo       Install Interception first: tools\install_interception_admin.bat
)
%PY% tools\set_hid_tcp.py --rootless
echo.
echo Starting UDP listener 127.0.0.1:18766 ^(window stays open^)...
echo Bot same session: set GATHERION_HID_TCP=udp://127.0.0.1:18766
echo Smoke: %PY% tools\hid_tcp_smoke.py --host udp://127.0.0.1:18766 --keys tab,1,2,3
echo Self-smoke: %PY% tools\rootless_hid_bridge.py --self-smoke --no-soft
echo.
start "Gatherion 1B rootless_hid_bridge" cmd /k %PY% tools\rootless_hid_bridge.py --no-soft
echo OK: bridge window launched. Keep it open while bot runs.
pause
