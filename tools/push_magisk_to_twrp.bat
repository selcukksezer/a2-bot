@echo off
setlocal EnableExtensions
rem Wait for TWRP recovery ADB, push Magisk-v31.1.zip to /sdcard/, print Install tip.
set ADB=
if exist "F:\a2\vendor\android_sdk\platform-tools\adb.exe" set ADB=F:\a2\vendor\android_sdk\platform-tools\adb.exe
if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" set ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe
if exist "C:\platform-tools\adb.exe" set ADB=C:\platform-tools\adb.exe
if "%ADB%"=="" set ADB=adb

set ZIP=%~dp0..\vendor\android_hid\s8_root\Magisk-v31.1.zip
if not exist "%ZIP%" (
  echo FAIL: missing Magisk zip: %ZIP%
  exit /b 2
)

echo Using %ADB%
echo Waiting for recovery/sideload device...
set /a N=0
:wait
set /a N+=1
"%ADB%" devices > "%TEMP%\adb_magisk_devices.txt" 2>&1
findstr /i /c:"recovery" /c:"sideload" "%TEMP%\adb_magisk_devices.txt" >nul
if not errorlevel 1 goto ready
if %N% GEQ 60 (
  echo FAIL: no recovery/sideload after ~60s. Current:
  type "%TEMP%\adb_magisk_devices.txt"
  echo Fallback: TWRP Mount -^> Enable MTP, copy Magisk-v31.1.zip, or Advanced -^> ADB Sideload.
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto wait

:ready
type "%TEMP%\adb_magisk_devices.txt"
echo.
echo Pushing Magisk-v31.1.zip ...
"%ADB%" push "%ZIP%" /sdcard/Magisk-v31.1.zip
if errorlevel 1 (
  echo FAIL: adb push /sdcard/ failed
  exit /b 3
)
"%ADB%" shell "mkdir -p /sdcard/Download" >nul 2>&1
"%ADB%" push "%ZIP%" /sdcard/Download/Magisk-v31.1.zip
if errorlevel 1 (
  echo FAIL: adb push /sdcard/Download/ failed
  exit /b 3
)
"%ADB%" shell "ls -la /sdcard/Magisk-v31.1.zip /sdcard/Download/Magisk-v31.1.zip"
echo.
echo === TWRP Install ===
echo Select Storage -^> Internal Storage -^> OK
echo Find Magisk-v31.1.zip -^> Install -^> swipe to flash -^> Reboot System
echo Zip on phone: /sdcard/Magisk-v31.1.zip  and  /sdcard/Download/Magisk-v31.1.zip
endlocal
exit /b 0
