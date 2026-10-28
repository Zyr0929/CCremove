@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  SubClean: Launching Interactive Web UI Studio...
echo =====================================================================

if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo Starting Gradio server on http://localhost:7860...
%PY_EXE% app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application stopped with an error code.
    pause
)
