@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
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

echo Using %ADB%
"%ADB%" devices
set "SRC1=%~dp0..\vendor\android_hid\usb_gadget\android_usb_hid_server.py"
set "SRC2=%~dp0..\vendor\android_hid\usb_gadget\enable_hid_gadget.sh"
set "SRC3=%~dp0..\vendor\android_hid\s8_gadget\s8_hid_tcp_server.py"
if not exist "%SRC1%" (
  echo FAIL: missing server source: %SRC1%
  exit /b 2
)
if not exist "%SRC2%" (
  echo FAIL: missing gadget script source: %SRC2%
  exit /b 2
)
if not exist "%SRC3%" (
  echo FAIL: missing TCP server source: %SRC3%
  exit /b 2
)
"%ADB%" push "%SRC1%" /data/local/tmp/android_usb_hid_server.py
if errorlevel 1 (
  echo FAIL: adb push android_usb_hid_server.py failed
  exit /b 3
)
"%ADB%" push "%SRC2%" /data/local/tmp/enable_hid_gadget.sh
if errorlevel 1 (
  echo FAIL: adb push enable_hid_gadget.sh failed
  exit /b 3
)
"%ADB%" push "%SRC3%" /data/local/tmp/s8_hid_tcp_server.py
if errorlevel 1 (
  echo FAIL: adb push s8_hid_tcp_server.py failed
  exit /b 3
)
"%ADB%" shell "su -c 'chmod 755 /data/local/tmp/android_usb_hid_server.py /data/local/tmp/enable_hid_gadget.sh /data/local/tmp/s8_hid_tcp_server.py; ls -l /dev/hidg0 /data/local/tmp/android_usb_hid_server.py'"
echo.
echo Prefer: python tools\setup_android_usb_hid.py --push --forward --write-settings
echo Start on phone (root):
echo   su -c "python /data/local/tmp/android_usb_hid_server.py"
echo Then PC: set GATHERION_HID_TCP=127.0.0.1:18765  (after adb forward)
echo Docs: docs\ANDROID_USB_NO_BT_TR.md
echo OK: push_s8_hid_server complete.
endlocal
exit /b 0
