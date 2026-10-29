@echo off
cd /d "%~dp0"
title AI-Based Recyclable Materials Identification and Sorting System
color 0A

echo =====================================================================
echo  AI-BASED IDENTIFICATION AND SORTING OF RECYCLABLE MATERIALS
echo  Department of Materials Science and Engineering Laboratory
echo =====================================================================
echo.

:: Check for virtual environment Python
if exist "%~dp0venv\Scripts\python.exe" (
    echo [INFO] Found project virtual environment.
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    echo [WARNING] Virtual environment not found in %~dp0venv. Using system Python...
    set "PYTHON_EXE=python"
)

:: Check if AI model is trained
if not exist "%~dp0models\mobilenetv2_recyclable_material.pth" (
    echo [WARNING] Trained AI model not found at models\mobilenetv2_recyclable_material.pth.
    echo [INFO] Training model now...
    "%PYTHON_EXE%" train_model.py
    if errorlevel 1 (
        echo [ERROR] Training failed. Please check the error above.
        pause
        exit /b 1
    )
    echo [INFO] Model training complete!
)

echo.
echo [INFO] Starting Streamlit Web Application...
echo [INFO] URL: http://localhost:8501
echo.

:: Open default browser automatically
start "" http://localhost:8501

:: Launch Streamlit directly using the virtual environment python
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501 --server.headless false

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)
