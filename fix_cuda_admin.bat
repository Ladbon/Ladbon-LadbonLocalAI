@echo off
:: Check for admin rights and re-launch with admin if needed
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    goto UACPrompt
) else (
    goto gotAdmin
)

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

echo ================================================
echo CUDA DLL Fix Script (Administrator Mode)
echo ================================================

:: Create the target directories if they don't exist
if not exist "dist\Ladbon AI Desktop\_internal\cuda_dlls" (
    mkdir "dist\Ladbon AI Desktop\_internal\cuda_dlls"
    echo Created cuda_dlls directory.
)

:: Copy CUDA DLLs with admin rights
echo Copying CUDA DLLs from NVIDIA CUDA Toolkit...
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudart64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\"
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublas64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\"
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublasLt64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\"
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvrtc64_120*.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudnn64_8.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudnn_ops_infer64_8.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul

:: Create a new launcher batch file with proper CUDA environment
echo Creating optimized CUDA launcher...
(
echo @echo off
echo setlocal EnableDelayedExpansion
echo.
echo :: Set up logging
echo set LOG_FILE=logs\cuda_launch_%date:~-4,4%%date:~-7,2%%date:~-10,2%-%time:~0,2%%time:~3,2%%time:~6,2%.log
echo set LOG_FILE=%%LOG_FILE: =0%%
echo.
echo if not exist logs mkdir logs
echo.
echo echo ================================================ ^> "%%LOG_FILE%%"
echo echo CUDA-enabled launcher for Ladbon AI Desktop ^>^> "%%LOG_FILE%%"
echo echo Started at: %%date%% %%time%% ^>^> "%%LOG_FILE%%"
echo echo ================================================ ^>^> "%%LOG_FILE%%"
echo.
echo :: Add CUDA DLLs directory to PATH
echo set ORIGINAL_PATH=%%PATH%%
echo echo Original PATH: %%PATH%% ^>^> "%%LOG_FILE%%"
echo.
echo echo Adding CUDA paths to PATH... ^>^> "%%LOG_FILE%%"
echo set PATH=%%~dp0_internal\cuda_dlls;%%PATH%%
echo set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%%PATH%%
echo echo New PATH: %%PATH%% ^>^> "%%LOG_FILE%%"
echo.
echo :: Create a simple CUDA test to verify DLLs can be loaded
echo echo Testing CUDA DLL loading... ^>^> "%%LOG_FILE%%"
echo where cudart64_12.dll ^>^> "%%LOG_FILE%%" 2^>^&1
echo where cublas64_12.dll ^>^> "%%LOG_FILE%%" 2^>^&1
echo.
echo :: Launch the application
echo echo Launching Ladbon AI Desktop with CUDA support... ^>^> "%%LOG_FILE%%"
echo echo. ^>^> "%%LOG_FILE%%"
echo.
echo start "" "Ladbon AI Desktop.exe" --use-cuda --verbose
echo.
echo endlocal
) > "dist\Ladbon AI Desktop\Launch_with_CUDA_Admin.bat"

echo Creating CUDA test script...
(
echo import os
echo import sys
echo import ctypes
echo import platform
echo import logging
echo from datetime import datetime
echo.
echo # Set up logging
echo log_file = f"logs/cuda_test_{datetime.now().strftime('%%Y%%m%%d-%%H%%M%%S')}.log"
echo os.makedirs("logs", exist_ok=True)
echo logging.basicConfig(
echo     level=logging.DEBUG,
echo     format='%%(asctime)s [%%(levelname)s] %%(message)s',
echo     handlers=[
echo         logging.FileHandler(log_file),
echo         logging.StreamHandler(sys.stdout)
echo     ]
echo )
echo.
echo logging.info("=" * 50)
echo logging.info("CUDA DLL Test Script")
echo logging.info("=" * 50)
echo.
echo # Log system info
echo logging.info(f"Python version: {platform.python_version()}")
echo logging.info(f"Platform: {platform.platform()}")
echo logging.info(f"Architecture: {platform.architecture()}")
echo logging.info(f"Machine: {platform.machine()}")
echo.
echo # Log environment variables
echo logging.info("Environment variables:")
echo for key, value in os.environ.items():
echo     if "PATH" in key or "CUDA" in key:
echo         logging.info(f"  {key}: {value}")
echo.
echo # Check PATH for DLLs
echo path_dirs = os.environ.get("PATH", "").split(os.pathsep)
echo logging.info("Checking PATH directories for CUDA DLLs:")
echo cuda_dlls_found = False
echo for directory in path_dirs:
echo     if os.path.exists(directory):
echo         dlls = [f for f in os.listdir(directory) if f.lower().endswith(".dll") and ("cuda" in f.lower() or "cublas" in f.lower())]
echo         if dlls:
echo             cuda_dlls_found = True
echo             logging.info(f"  {directory}:")
echo             for dll in dlls:
echo                 logging.info(f"    - {dll}")
echo.
echo if not cuda_dlls_found:
echo     logging.warning("No CUDA DLLs found in PATH directories!")
echo.
echo # Try loading CUDA DLLs directly
echo logging.info("Attempting to load CUDA DLLs:")
echo dlls_to_try = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
echo.
echo for dll in dlls_to_try:
echo     try:
echo         lib = ctypes.CDLL(dll)
echo         logging.info(f"  ✓ Successfully loaded {dll}")
echo     except Exception as e:
echo         logging.error(f"  ✗ Failed to load {dll}: {str(e)}")
echo.
echo # Try importing llama_cpp
echo logging.info("Attempting to import llama_cpp:")
echo try:
echo     import llama_cpp
echo     logging.info(f"  ✓ Successfully imported llama_cpp (version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'})")
echo     
echo     # Try initializing CUDA backend
echo     logging.info("Attempting to initialize CUDA backend:")
echo     try:
echo         # Try different initialization methods
echo         try:
echo             logging.info("  Trying llama_cpp.llama_backend_init(True)")
echo             llama_cpp.llama_backend_init(True)
echo             logging.info("  ✓ Successfully initialized CUDA backend with llama_backend_init(True)")
echo         except (AttributeError, TypeError) as e1:
echo             logging.warning(f"  Method 1 failed: {e1}")
echo             try:
echo                 logging.info("  Trying llama_cpp.llama_backend_init(use_cuda=True)")
echo                 llama_cpp.llama_backend_init(use_cuda=True)
echo                 logging.info("  ✓ Successfully initialized CUDA backend with llama_backend_init(use_cuda=True)")
echo             except (AttributeError, TypeError) as e2:
echo                 logging.warning(f"  Method 2 failed: {e2}")
echo                 logging.info("  Trying llama_cpp._lib.llama_backend_init()")
echo                 if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_backend_init'):
echo                     llama_cpp._lib.llama_backend_init()
echo                     logging.info("  ✓ Successfully initialized CUDA backend with _lib.llama_backend_init()")
echo                 else:
echo                     logging.error("  ✗ Could not find llama_backend_init method")
echo     except Exception as e:
echo         logging.error(f"  ✗ Failed to initialize CUDA backend: {str(e)}")
echo except Exception as e:
echo     logging.error(f"  ✗ Failed to import llama_cpp: {str(e)}")
echo.
echo logging.info("=" * 50)
echo logging.info("Test completed")
echo logging.info("=" * 50)
) > "dist\Ladbon AI Desktop\_internal\cuda_test.py"

echo Creating CUDA test launcher...
(
echo @echo off
echo setlocal EnableDelayedExpansion
echo.
echo :: Add CUDA DLLs to PATH
echo set PATH=%%~dp0_internal\cuda_dlls;%%PATH%%
echo set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%%PATH%%
echo.
echo :: Run the test script
echo "%%~dp0_internal\python.exe" "%%~dp0_internal\cuda_test.py"
echo.
echo echo.
echo echo Test completed. Press any key to exit...
echo pause > nul
echo endlocal
) > "dist\Ladbon AI Desktop\Run_CUDA_Test.bat"

echo ================================================
echo CUDA fix completed successfully!
echo.
echo The following files have been created:
echo - dist\Ladbon AI Desktop\Launch_with_CUDA_Admin.bat
echo - dist\Ladbon AI Desktop\Run_CUDA_Test.bat
echo.
echo Instructions:
echo 1. First run "Run_CUDA_Test.bat" to verify CUDA DLLs can be loaded
echo 2. Then use "Launch_with_CUDA_Admin.bat" to start the app with CUDA support
echo.
echo Check logs\cuda_test_*.log for test results
echo Check logs\cuda_launch_*.log for launcher logs
echo ================================================

pause
