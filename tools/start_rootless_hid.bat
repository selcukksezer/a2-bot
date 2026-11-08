@echo off
REM 1B rootless KEY bridge → Interception (stock phone is NOT USB HID).
REM Goal stays OPEN until 1A Magisk/hidg0 proves. Do not run Magisk from here.
cd /d "%~dp0.."
echo === Gatherion 1B rootless HID ===
python tools\interception_status.py
if errorlevel 1 (
  echo WARN: Interception not ready — bridge may fall back to soft ^(AION2 often drops soft^).
)
python tools\set_hid_tcp.py --rootless
echo.
echo Starting UDP listener 127.0.0.1:18766 ^(window stays open^)...
echo Bot same session: set GATHERION_HID_TCP=udp://127.0.0.1:18766
echo Smoke: python tools\hid_tcp_smoke.py --host udp://127.0.0.1:18766 --keys tab,1,2,3
echo Self-smoke: python tools\rootless_hid_bridge.py --self-smoke --no-soft
echo.
start "Gatherion 1B rootless_hid_bridge" cmd /k python tools\rootless_hid_bridge.py --no-soft
echo OK: bridge window launched. Keep it open while bot runs.
pause
