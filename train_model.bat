@echo off
cd /d "%~dp0"
title Train AI Model - Recyclable Materials Identification
color 0B

echo =====================================================================
echo  AI MODEL TRAINING PIPELINE - MOBILENETV2 TRANSFER LEARNING
echo  Department of Materials Science and Engineering Laboratory
echo =====================================================================
echo.

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [INFO] Step 1: Downloading & Preprocessing Recyclable Materials Dataset...
"%PYTHON_EXE%" download_dataset.py
if errorlevel 1 (
    echo [ERROR] Dataset setup failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Step 2: Training MobileNetV2 Neural Network...
"%PYTHON_EXE%" train_model.py
if errorlevel 1 (
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Step 3: Evaluating Model on Test Dataset...
"%PYTHON_EXE%" evaluate_model.py
if errorlevel 1 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Model training and evaluation successfully completed!
echo Metrics and graphs are saved in the outputs/ directory.
echo =====================================================================
pause
