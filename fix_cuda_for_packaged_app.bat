@echo off
echo ============================================================
echo CUDA SETUP FOR PACKAGED APPLICATION
echo ============================================================
echo.

REM Detect application directory
set "APP_DIR=%~dp0dist\Ladbon AI Desktop"
if not exist "%APP_DIR%" (
    echo ERROR: Cannot find application directory at %APP_DIR%
    echo Please run this script from the src directory
    pause
    exit /b 1
)

echo Found application directory: %APP_DIR%
echo.

REM Create cuda_dlls directory if it doesn't exist
set "CUDA_DLLS_DIR=%APP_DIR%\_internal\cuda_dlls"
if not exist "%CUDA_DLLS_DIR%" (
    echo Creating CUDA DLLs directory: %CUDA_DLLS_DIR%
    mkdir "%CUDA_DLLS_DIR%"
) else (
    echo CUDA DLLs directory already exists
)

echo.
echo Detecting CUDA installations...
echo.

REM Find installed CUDA versions
set "FOUND_CUDA=0"
for /l %%v in (0, 1, 9) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.%%v\bin" (
        echo Found CUDA 12.%%v at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.%%v\bin
        set "FOUND_CUDA=1"
        set "CUDA_VERSION=12.%%v"
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.%%v"
    )
)

if "%FOUND_CUDA%"=="0" (
    echo ERROR: No CUDA installations found
    echo Please install CUDA Toolkit from NVIDIA website
    pause
    exit /b 1
)

echo.
echo Using CUDA %CUDA_VERSION% installation
echo.

REM Copy CUDA DLLs to application directory
echo Copying CUDA DLLs to application directory...

set "CUDA_BIN=%CUDA_PATH%\bin"
echo Source: %CUDA_BIN%
echo Destination: %CUDA_DLLS_DIR%

REM Copy critical CUDA DLLs
copy "%CUDA_BIN%\cudart64*.dll" "%CUDA_DLLS_DIR%\" /Y
copy "%CUDA_BIN%\cublas64*.dll" "%CUDA_DLLS_DIR%\" /Y
copy "%CUDA_BIN%\cublasLt64*.dll" "%CUDA_DLLS_DIR%\" /Y
copy "%CUDA_BIN%\curand64*.dll" "%CUDA_DLLS_DIR%\" /Y

echo.
echo Creating enhanced launcher batch file...
echo.

REM Create an enhanced launcher batch file
set "LAUNCHER_PATH=%~dp0dist\Launch with CUDA Enhanced.bat"

echo @echo off > "%LAUNCHER_PATH%"
echo echo Setting up CUDA environment for Ladbon AI Desktop... >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo echo Using CUDA %CUDA_VERSION% >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo set "CUDA_PATH=%CUDA_PATH%" >> "%LAUNCHER_PATH%"
echo set "PATH=%CUDA_PATH%\bin;%%PATH%%" >> "%LAUNCHER_PATH%"
echo set "CUDA_VISIBLE_DEVICES=0" >> "%LAUNCHER_PATH%"
echo. >> "%LAUNCHER_PATH%"
echo echo Starting Ladbon AI Desktop with CUDA support... >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo cd "%%~dp0Ladbon AI Desktop" >> "%LAUNCHER_PATH%"
echo set "PATH=%%CD%%\_internal\cuda_dlls;%%CD%%\_internal\llama_cpp\lib;%%PATH%%" >> "%LAUNCHER_PATH%"
echo start "" "Ladbon AI Desktop.exe" >> "%LAUNCHER_PATH%"

echo Created enhanced CUDA launcher: %LAUNCHER_PATH%

echo.
echo Creating patch for llamacpp_loader.py...
echo.

REM Create a DLL verification script to run inside the application
set "VERIFY_SCRIPT=%APP_DIR%\_internal\check_cuda_dlls.py"

echo import os, sys, glob > "%VERIFY_SCRIPT%"
echo print("Checking for CUDA DLLs...") >> "%VERIFY_SCRIPT%"
echo cuda_dlls_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_internal", "cuda_dlls") >> "%VERIFY_SCRIPT%"
echo if os.path.exists(cuda_dlls_dir): >> "%VERIFY_SCRIPT%"
echo     print(f"Found CUDA DLLs directory: {cuda_dlls_dir}") >> "%VERIFY_SCRIPT%"
echo     dlls = glob.glob(os.path.join(cuda_dlls_dir, "*.dll")) >> "%VERIFY_SCRIPT%"
echo     print(f"Found {len(dlls)} CUDA DLLs:") >> "%VERIFY_SCRIPT%"
echo     for dll in dlls: >> "%VERIFY_SCRIPT%"
echo         print(f"  - {os.path.basename(dll)}") >> "%VERIFY_SCRIPT%"
echo     # Add DLL directory to PATH >> "%VERIFY_SCRIPT%"
echo     os.environ["PATH"] = cuda_dlls_dir + os.pathsep + os.environ.get("PATH", "") >> "%VERIFY_SCRIPT%"
echo     try: >> "%VERIFY_SCRIPT%"
echo         os.add_dll_directory(cuda_dlls_dir) >> "%VERIFY_SCRIPT%"
echo         print(f"Added DLL directory: {cuda_dlls_dir}") >> "%VERIFY_SCRIPT%"
echo     except Exception as e: >> "%VERIFY_SCRIPT%"
echo         print(f"Error adding DLL directory: {e}") >> "%VERIFY_SCRIPT%"
echo else: >> "%VERIFY_SCRIPT%"
echo     print(f"CUDA DLLs directory not found: {cuda_dlls_dir}") >> "%VERIFY_SCRIPT%"

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Please use the new launcher "Launch with CUDA Enhanced.bat" 
echo to start the application with CUDA support.
echo.
pause
