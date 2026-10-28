@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  SubClean: Deep Generative Video Subtitle Removal - Setup Script
echo =====================================================================
echo.

set "PYTHON_CMD="

:: 1. Check if 'python' works in current PATH
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
    goto :FoundPython
)

:: 2. Check if 'py' Windows launcher works
py -3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=py -3"
    goto :FoundPython
)

:: 3. Check AppData Local Python install directory
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    goto :FoundPython
)

:: 4. Check WindowsApps Store Python alias
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
    goto :FoundPython
)

:: 5. Check user Programs/Python directories
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_CMD=%%D\python.exe"
        goto :FoundPython
    )
)

:: 6. Check C:\Program Files\Python*
for /d %%D in ("C:\Program Files\Python*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_CMD=%%D\python.exe"
        goto :FoundPython
    )
)

:: 7. If still not found
echo [ERROR] Python executable could not be automatically located on your system.
echo Please install Python (3.10, 3.11, or 3.12) from: https://www.python.org/downloads/
echo Make sure to check the box "Add python.exe to PATH" during installation.
echo.
pause
exit /b 1

:FoundPython
echo [INFO] Detected Python engine: !PYTHON_CMD!
!PYTHON_CMD! --version

:: Add Python and Scripts directory to this session's PATH
for /f "delims=" %%I in ('!PYTHON_CMD! -c "import sys, os; print(os.path.dirname(sys.executable))"') do (
    set "PY_DIR=%%I"
    set "PATH=!PY_DIR!;!PY_DIR!\Scripts;!PATH!"
)

:: Ensure pip is up to date
echo.
echo [INFO] Verifying pip package manager...
!PYTHON_CMD! -m ensurepip --default-pip >nul 2>&1

:: Create virtual environment if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment in .\venv...
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        echo [WARNING] venv creation failed; will install dependencies directly into active Python.
        set "ACTIVE_PY=!PYTHON_CMD!"
    ) else (
        set "ACTIVE_PY=venv\Scripts\python.exe"
    )
) else (
    echo [INFO] Existing virtual environment found in .\venv.
    set "ACTIVE_PY=venv\Scripts\python.exe"
)

:: Upgrade pip and wheel
echo [INFO] Upgrading pip...
!ACTIVE_PY! -m pip install --upgrade pip

:: Install PyTorch and Torchvision
echo [INFO] Installing PyTorch with GPU support...
!ACTIVE_PY! -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || !ACTIVE_PY! -m pip install torch torchvision

:: Install dependencies from requirements.txt
echo [INFO] Installing remaining dependencies from requirements.txt...
!ACTIVE_PY! -m pip install -r requirements.txt

:: Download pretrained model weights
echo.
echo [INFO] Checking / Downloading pretrained model weights...
!ACTIVE_PY! scripts\download_weights.py

echo.
echo =====================================================================
echo  SETUP COMPLETED SUCCESSFULLY!
echo =====================================================================
echo.
echo  To launch the Gradio Web UI Studio:
echo      run_app.bat
echo.
echo  Or via terminal:
echo      venv\Scripts\activate
echo      python app.py
echo.
pause
