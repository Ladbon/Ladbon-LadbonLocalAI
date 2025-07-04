@echo off
REM This script ensures the CUDA paths are set correctly before launching the application
echo ===================================
echo = Launching with CUDA 12.9 Support =
echo ===================================
echo.

REM Add all possible CUDA versions to PATH, prioritizing 12.9
set "CUDA_PATH_V12_9=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH_V12_0=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
set "CUDA_FOUND=0"

REM Check for CUDA 12.9 (preferred)
if exist "%CUDA_PATH_V12_9%" (
    echo Found CUDA 12.9 installation
    set "PATH=%CUDA_PATH_V12_9%\bin;%PATH%"
    set "CUDA_PATH=%CUDA_PATH_V12_9%"
    set "CUDA_FOUND=1"
    echo Added CUDA 12.9 to PATH
)

REM Check for CUDA 12.0 if 12.9 not found
if %CUDA_FOUND%==0 (
    if exist "%CUDA_PATH_V12_0%" (
        echo Found CUDA 12.0 installation
        set "PATH=%CUDA_PATH_V12_0%\bin;%PATH%"
        set "CUDA_PATH=%CUDA_PATH_V12_0%"
        set "CUDA_FOUND=1"
        echo Added CUDA 12.0 to PATH
    )
)

REM No CUDA found
if %CUDA_FOUND%==0 (
    echo WARNING: No CUDA installation found!
    echo The application will run in CPU-only mode.
)

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%dist\Ladbon AI Desktop"

if not exist "%APP_DIR%\Ladbon AI Desktop.exe" (
    echo ERROR: Application not found at:
    echo %APP_DIR%
    echo.
    echo Please build the application first with package.py
    pause
    exit /b 1
)

echo.
echo Launching Ladbon AI Desktop with CUDA support...
echo Application directory: %APP_DIR%
echo Current PATH=%PATH%
echo.
echo If you experience "access violation" errors:
echo 1. Run copy_cuda_dlls.bat to copy CUDA 12.9 DLLs to the application
echo 2. Or use install_cpu_llamacpp.py to switch to CPU-only mode
echo.

REM Change to the application directory
cd /d "%APP_DIR%"
echo Current directory: %CD%
echo.

REM Launch the application
"Ladbon AI Desktop.exe"

echo.
echo Application closed.
pause
