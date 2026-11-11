@echo off
REM One-shot AFTER Magisk root: push HID server, enable gadget, forward, settings, smoke.
REM Does NOT flash Magisk/TWRP. Requires: adb device + su + (best-effort) hidg0.
setlocal EnableExtensions
cd /d "%~dp0.."

REM Resolve adb with absolute fallbacks (PATH often lacks platform-tools).
set ADB=
if exist "%~dp0..\vendor\android_sdk\platform-tools\adb.exe" set "ADB=%~dp0..\vendor\android_sdk\platform-tools\adb.exe"
if exist "F:\a2\vendor\android_sdk\platform-tools\adb.exe" set "ADB=F:\a2\vendor\android_sdk\platform-tools\adb.exe"
if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" set "ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
if exist "%ANDROID_HOME%\platform-tools\adb.exe" set "ADB=%ANDROID_HOME%\platform-tools\adb.exe"
if exist "%ANDROID_SDK_ROOT%\platform-tools\adb.exe" set "ADB=%ANDROID_SDK_ROOT%\platform-tools\adb.exe"
if exist "C:\platform-tools\adb.exe" set "ADB=C:\platform-tools\adb.exe"
if exist "C:\Android\Sdk\platform-tools\adb.exe" set "ADB=C:\Android\Sdk\platform-tools\adb.exe"
if "%ADB%"=="" set ADB=adb
set "PATH=%~dp0..\vendor\android_sdk\platform-tools;F:\a2\vendor\android_sdk\platform-tools;%PATH%"

set PY=python
where py >nul 2>&1 && set PY=py -3

echo === after_magisk_hid: adb=%ADB% ===
"%ADB%" version
echo === after_magisk_hid: root probe ===
%PY% tools\setup_android_usb_hid.py --check
if errorlevel 3 (
  echo WARN: hidg0 missing, root OK - the --enable-gadget step below will try to enable it.
  goto :CHECK_DONE
)
if errorlevel 2 (
  echo FAIL: su missing — finish Magisk first. See docs\S8_MAGISK_SEQUENCE_SAFE.md
  exit /b 2
)
if errorlevel 1 (
  echo FAIL: no adb device - enable USB debugging and authorize this PC, then retry.
  exit /b 1
)
:CHECK_DONE

echo.
echo === push + enable-gadget + forward + settings ===
%PY% tools\setup_android_usb_hid.py --push --enable-gadget --forward --write-settings --start-server
if errorlevel 1 (
  echo FAIL: setup_android_usb_hid.py exited %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo.
echo === smoke (TCP via adb forward) ===
set GATHERION_HID_TCP=127.0.0.1:18765
%PY% tools\hid_tcp_smoke.py --host 127.0.0.1:18765
set SMOKE_RC=%ERRORLEVEL%

echo.
if %SMOKE_RC% neq 0 (
  echo smoke RC=%SMOKE_RC% — if hidg0 missing: NetHunter / USB Gadget Tool / enable_hid_gadget.sh
  echo Docs: docs\S8_USB_HID_GADGET_TR.md  docs\S8_POST_ROOT_ONELINER.md
  exit /b %SMOKE_RC%
)
echo OK: after_magisk_hid complete. GATHERION_HID_TCP=127.0.0.1:18765
endlocal
exit /b 0
