@echo off
REM One-click UAC prompt for Interception driver install (operator still clicks Yes).
REM Does NOT silent-install. Gatherion bots never call this.
REM This build accepts /install | /uninstall only — /command is INVALID.
setlocal EnableExtensions
cd /d "%~dp0.."
set "ROOT=%CD%"
set "INSTALLER=%ROOT%\vendor\interception\install-interception.exe"

echo ========================================
echo  Interception Admin Install Helper
echo  ROOT=%ROOT%
echo ========================================
echo.

if not exist "%INSTALLER%" (
  echo FAIL: missing installer
  echo Expected: %INSTALLER%
  echo Unzip vendor\interception\Interception-v1.0.1.zip first.
  echo.
  echo Press any key to close...
  pause >nul
  exit /b 1
)

echo Staged installer: %INSTALLER%
echo Launching elevated: install-interception.exe /install
echo You must click Yes on the UAC prompt. This is not silent.
echo.
echo If UAC is cancelled, exit code is usually non-zero.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $p = Start-Process -FilePath '%INSTALLER%' -ArgumentList '/install' -Verb RunAs -Wait -PassThru; if ($null -eq $p) { exit 2 }; exit ([int]$p.ExitCode) } catch { Write-Host $_.Exception.Message; exit 3 }"
set "EC=%ERRORLEVEL%"

echo.
echo ----------------------------------------
echo Installer / helper exit code: %EC%
if "%EC%"=="0" (
  echo STATUS: elevated process finished with 0
) else (
  echo STATUS: FAIL or UAC cancelled / installer error
  echo Tip: open Admin CMD and run:
  echo   cd /d %ROOT%\vendor\interception
  echo   install-interception.exe /install
  echo   pause
)
echo ----------------------------------------
echo Next: reboot if asked, then:
echo   cd /d %ROOT%
echo   python tools\interception_status.py
echo Optional smoke:
echo   python tools\interception_status.py --smoke --confirm-smoke --key tab
echo.
echo Press any key to close...
pause >nul
exit /b %EC%
