@echo off
set "GATHERION_HOME=F:\a2"
if not exist "%GATHERION_HOME%\main.py" set "GATHERION_HOME=F:\Gatherion"
if not exist "%GATHERION_HOME%\dataset\data.yaml" (
    echo [HATA] Egitim hedefi bulunamadi: "%GATHERION_HOME%\dataset\data.yaml"
    exit /b 1
)
cd /d "%GATHERION_HOME%"
if errorlevel 1 (
    echo [HATA] Calisma dizinine gecilemedi: "%GATHERION_HOME%"
    exit /b 1
)
REM fix/v20-egitim-devam: sabit checkpoint yolu yok. En son yazilmis
REM weights\last.pt bulunur (zaman damgali kosu klasorleri + eski sabit
REM isimli klasor taninir); yoksa acik hata ile cikilir.
set "RUNS_DIR=%GATHERION_HOME%\models\rtx_runs"
set "LAST_PT="
for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-ChildItem -Path '%RUNS_DIR%' -Directory -Filter 'gatherion_rtx_fp16*' -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName 'weights\last.pt' } | Where-Object { Test-Path $_ } | Sort-Object { (Get-Item $_).LastWriteTime } -Descending | Select-Object -First 1"') do set "LAST_PT=%%P"
if not defined LAST_PT (
    echo [HATA] Devam icin checkpoint yok: "%RUNS_DIR%\gatherion_rtx_fp16*\weights\last.pt" bulunamadi.
    echo [INFO] Ilk egitim icin train_rtx_yolo.bat kullanin.
    exit /b 1
)
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
set "PYTHONUTF8=1"
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
title Gatherion RTX 3070 - Resume latest run ^& ONNX Exporter
echo ============================================================================
echo   GATHERION AI - RESUME RTX CUDA FP16 TRAINING
echo   Checkpoint   : %LAST_PT%
echo   Target ONNX  : F:/a2/models/yolov8n_gatherion.onnx (egitim tamamlaninca)
echo   Config       : Ultralytics resume=True (args run klasorunden)
echo ============================================================================
echo.

"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto :CHECK_ULTRA
echo [INFO] PyTorch CUDA bulunamadi veya CPU surumu kurulu. RTX 3070 icin PyTorch CUDA 12.4 kuruluyor...
"%PYTHON_EXE%" -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [HATA] PyTorch CUDA kurulumu basarisiz oldu.
    exit /b 1
)

:CHECK_ULTRA
"%PYTHON_EXE%" -c "import ultralytics" >nul 2>&1
if not errorlevel 1 goto :START_RESUME
echo [INFO] Installing Ultralytics YOLO and ONNX Runtime GPU dependencies...
"%PYTHON_EXE%" -m pip install -U ultralytics onnx onnxslim onnxruntime-gpu
if errorlevel 1 (
    echo [HATA] Ultralytics kurulumu basarisiz oldu.
    exit /b 1
)

:START_RESUME
echo [INFO] Resuming RTX FP16 training from last.pt...
"%PYTHON_EXE%" -c "from pathlib import Path; import shutil, torch; from ultralytics import YOLO; last_p = Path(r'%LAST_PT%'); use_cuda = torch.cuda.is_available(); dev = 0 if use_cuda else 'cpu'; cpu='CPU'; print(f'[INFO] Active Training Device: {dev} (CUDA={use_cuda}, GPU={torch.cuda.get_device_name(0) if use_cuda else cpu})'); print('[INFO] Resume checkpoint:', last_p); m = YOLO(str(last_p)); m.train(resume=True); exp = m.export(format='onnx', half=use_cuda, dynamic=False, imgsz=960); out_p = Path(r'F:/a2/models/yolov8n_gatherion.onnx'); out_p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(str(exp), str(out_p)); print('[SUCCESS] Exported ONNX model to:', str(out_p))"
if not errorlevel 1 goto :TRAIN_DONE

echo [HATA] Ultralytics egitimi (resume) basarisiz oldu; model uretilmedi.
exit /b 1

:TRAIN_DONE
if not exist "F:\a2\models\yolov8n_gatherion.onnx" (
    echo [HATA] Egitim bitti gorunuyor ama ONNX dosyasi yok: F:\a2\models\yolov8n_gatherion.onnx
    echo [HATA] Model uretilmedi; bot eski modelle calismaya devam eder.
    exit /b 1
)
echo.
echo ============================================================================
echo   TRAINING COMPLETE! Model saved to: models/yolov8n_gatherion.onnx
echo ============================================================================
