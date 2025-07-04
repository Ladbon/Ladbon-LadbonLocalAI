@echo off
echo ===================================
echo = CUDA 12.9 DLL Copy Tool         =
echo ===================================
echo.

REM Set paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
set "DEST_DIR=%~dp0dist\Ladbon AI Desktop\_internal\llama_cpp\lib"

echo Checking for CUDA 12.9 installation...
if not exist "%CUDA_PATH%" (
    echo ERROR: CUDA 12.9 not found at:
    echo %CUDA_PATH%
    echo.
    echo Please install CUDA 12.9 or modify this script to point to your CUDA installation.
    pause
    exit /b 1
)

echo CUDA 12.9 found at: %CUDA_PATH%
echo.

echo Checking for packaged application...
if not exist "%DEST_DIR%" (
    echo WARNING: Application lib directory not found at:
    echo %DEST_DIR%
    echo.
    echo Creating directory...
    mkdir "%DEST_DIR%" 2>nul
    if not exist "%DEST_DIR%" (
        echo Failed to create directory.
        pause
        exit /b 1
    )
)

echo Destination directory: %DEST_DIR%
echo.

REM Copy CUDA DLLs
echo Copying CUDA 12.9 DLLs from %CUDA_PATH% to %DEST_DIR%
copy "%CUDA_PATH%\cudart64_*.dll" "%DEST_DIR%" /y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy CUDA Runtime DLLs.
    pause
    exit /b 1
)
echo - Copied CUDA Runtime DLLs

copy "%CUDA_PATH%\cublas64_*.dll" "%DEST_DIR%" /y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy cuBLAS DLLs.
    pause
    exit /b 1
)
echo - Copied cuBLAS DLLs

copy "%CUDA_PATH%\cublasLt64_*.dll" "%DEST_DIR%" /y
echo - Copied cuBLAS LT DLLs

copy "%CUDA_PATH%\curand64_*.dll" "%DEST_DIR%" /y
echo - Copied cuRAND DLLs

copy "%CUDA_PATH%\cudnn64_*.dll" "%DEST_DIR%" /y 2>nul
echo - Copied cuDNN DLLs (if available)

copy "%CUDA_PATH%\nvrtc64_*.dll" "%DEST_DIR%" /y 2>nul
echo - Copied NVRTC DLLs (if available)

echo.
echo ===================================
echo = CUDA DLL COPY COMPLETE         =
echo ===================================
echo.
echo Successfully copied CUDA 12.9 DLLs to:
echo %DEST_DIR%
echo.
echo Your application should now work with CUDA support.
echo Launch the application using the "Launch with CUDA.bat" script.
echo.
pause
