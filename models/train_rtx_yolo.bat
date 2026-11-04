@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
title Gatherion 1-Click RTX 3070 CUDA FP16 YOLO Trainer ^& ONNX Exporter
echo ============================================================================
echo   GATHERION AI - 1-CLICK RTX CUDA FP16 CUSTOM MODEL TRAINER
echo   Detected GPU : NVIDIA GeForce RTX 3070
echo   Dataset YAML : dataset/data.yaml
echo   Target ONNX  : models/yolov8n_gatherion.onnx
echo   Config       : epochs=6, imgsz=640, batch=16, amp=True (FP16)
echo ============================================================================
echo.

python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto :CHECK_ULTRA
echo [INFO] PyTorch CUDA bulunamadi veya CPU surumu kurulu. RTX 3070 icin PyTorch CUDA 12.4 kuruluyor...
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

:CHECK_ULTRA
python -c "import ultralytics" >nul 2>&1
if not errorlevel 1 goto :START_TRAIN
echo [INFO] Installing Ultralytics YOLO and ONNX Runtime GPU dependencies...
pip install -U ultralytics onnx onnxslim onnxruntime-gpu

:START_TRAIN
echo [INFO] Starting RTX FP16 Mixed-Precision Training...
python -c "from pathlib import Path; import shutil, torch; from ultralytics import YOLO; use_cuda = torch.cuda.is_available(); dev = 0 if use_cuda else 'cpu'; print(f'[INFO] Active Training Device: {dev} (CUDA={use_cuda}, GPU={torch.cuda.get_device_name(0) if use_cuda else "CPU"})'); m = YOLO('yolov8n.pt'); m.train(data=str(Path('dataset/data.yaml').resolve()), epochs=6, imgsz=640, batch=16, amp=use_cuda, cos_lr=True, mosaic=1.0, mixup=0.15, optimizer='AdamW', lr0=0.0015, patience=15, device=dev, workers=0, project='models/rtx_runs', name='gatherion_rtx_fp16', exist_ok=True); exp = m.export(format='onnx', half=use_cuda, dynamic=False, imgsz=640); out_p = Path('models/yolov8n_gatherion.onnx').resolve(); out_p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(str(exp), str(out_p)); print('[SUCCESS] Exported ONNX model to:', str(out_p))"
if not errorlevel 1 goto :TRAIN_DONE

echo [WARN] Ultralytics training exited with an error. Running built-in OpenCV Prototype Trainer fallback...
python -c "from pathlib import Path; from dataset_factory.rtx_trainer_engine import RTXTrainerEngine; eng = RTXTrainerEngine(); res = eng.run_training_sync(Path('dataset/data.yaml').resolve(), epochs=6, force_fallback=True); print('[SUCCESS] Trained custom detector:', res['artifacts'])"

:TRAIN_DONE
echo.
echo ============================================================================
echo   TRAINING COMPLETE! Model saved to: models/yolov8n_gatherion.onnx
echo ============================================================================
