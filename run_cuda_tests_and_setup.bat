@echo off
setlocal EnableDelayedExpansion

echo ==========================================================
echo CUDA Test Runner for Packaged App
echo ==========================================================

:: Set up logging directory
if not exist "logs" mkdir logs

:: Ensure CUDA DLLs directory exists
if not exist "dist\Ladbon AI Desktop\_internal\cuda_dlls" (
    mkdir "dist\Ladbon AI Desktop\_internal\cuda_dlls"
    echo Created CUDA DLLs directory.
)

:: Copy the test script to the packaged app
echo Copying test script to packaged app...
copy /Y "cuda_test_app.py" "dist\Ladbon AI Desktop\_internal\cuda_test_app.py"

:: Copy CUDA DLLs if present in CUDA toolkit
echo Checking for CUDA DLLs in NVIDIA toolkit...
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" (
    echo Found CUDA v12.9 toolkit
    copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudart64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul
    copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublas64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul
    copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublasLt64_12.dll" "dist\Ladbon AI Desktop\_internal\cuda_dlls\" 2>nul
    echo Copied CUDA DLLs to packaged app
) else (
    echo CUDA toolkit not found at expected location
)

:: Create a CUDA test runner batch file
echo Creating test runner batch file...
(
echo @echo off
echo setlocal EnableDelayedExpansion
echo.
echo :: Set up PATH to include CUDA DLLs
echo set PATH=%%~dp0_internal\cuda_dlls;%%PATH%%
echo set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%%PATH%%
echo.
echo :: Run the CUDA test
echo echo Running CUDA test...
echo "%%~dp0_internal\python.exe" "%%~dp0_internal\cuda_test_app.py"
echo.
echo echo.
echo echo Test completed. Press any key to exit...
echo pause ^> nul
) > "dist\Ladbon AI Desktop\Run_CUDA_Test_Full.bat"

echo Test runner batch file created: dist\Ladbon AI Desktop\Run_CUDA_Test_Full.bat

:: Create a batch file for fixing DLLs and launching
echo Creating comprehensive CUDA launcher...
(
echo @echo off
echo setlocal EnableDelayedExpansion
echo.
echo :: Set up logging
echo set LOG_FILE=logs\comprehensive_cuda_%%date:~-4,4%%%%date:~-7,2%%%%date:~-10,2%%-%%time:~0,2%%%%time:~3,2%%%%time:~6,2%%.log
echo set LOG_FILE=%%LOG_FILE: =0%%
echo.
echo if not exist logs mkdir logs
echo.
echo echo ================================================ ^> "%%LOG_FILE%%"
echo echo Comprehensive CUDA launcher for Ladbon AI Desktop ^>^> "%%LOG_FILE%%"
echo echo Started at: %%date%% %%time%% ^>^> "%%LOG_FILE%%"
echo echo ================================================ ^>^> "%%LOG_FILE%%"
echo.
echo :: First ensure all CUDA DLLs are copied
echo echo Copying latest CUDA DLLs... ^>^> "%%LOG_FILE%%"
echo.
echo if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudart64_12.dll" (
echo     copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudart64_12.dll" "%%~dp0_internal\cuda_dlls\" ^>^> "%%LOG_FILE%%" 2^>^&1
echo     copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublas64_12.dll" "%%~dp0_internal\cuda_dlls\" ^>^> "%%LOG_FILE%%" 2^>^&1
echo     copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cublasLt64_12.dll" "%%~dp0_internal\cuda_dlls\" ^>^> "%%LOG_FILE%%" 2^>^&1
echo     copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvrtc64_120*.dll" "%%~dp0_internal\cuda_dlls\" ^>^> "%%LOG_FILE%%" 2^>^&1
echo     echo Copied latest CUDA DLLs ^>^> "%%LOG_FILE%%"
echo ^) else (
echo     echo CUDA toolkit not found at expected location ^>^> "%%LOG_FILE%%"
echo ^)
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
echo :: Set environment variables that might help with CUDA detection
echo set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
echo set CUDA_VISIBLE_DEVICES=0
echo echo Set CUDA environment variables ^>^> "%%LOG_FILE%%"
echo.
echo :: Create a modified Python hook to ensure CUDA is loaded
echo echo Creating Python hook for CUDA... ^>^> "%%LOG_FILE%%"
echo ^(
echo import os, sys, ctypes
echo print^("CUDA Pre-init hook running"^)
echo 
echo # Try to pre-load CUDA DLLs
echo cuda_dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
echo for dll in cuda_dlls:
echo     try:
echo         lib = ctypes.CDLL^(dll^)
echo         print^(f"Pre-loaded {dll}"^)
echo     except Exception as e:
echo         print^(f"Failed to pre-load {dll}: {e}"^)
echo ^) ^> "%%~dp0_internal\cuda_preload.py"
echo.
echo :: Test CUDA DLL loading
echo echo Testing CUDA DLL loading... ^>^> "%%LOG_FILE%%"
echo where cudart64_12.dll ^>^> "%%LOG_FILE%%" 2^>^&1
echo where cublas64_12.dll ^>^> "%%LOG_FILE%%" 2^>^&1
echo.
echo :: Run the Python preload hook
echo echo Running Python CUDA preload hook... ^>^> "%%LOG_FILE%%"
echo "%%~dp0_internal\python.exe" "%%~dp0_internal\cuda_preload.py" ^>^> "%%LOG_FILE%%" 2^>^&1
echo.
echo :: Launch the application with CUDA enabled
echo echo Launching Ladbon AI Desktop with CUDA support... ^>^> "%%LOG_FILE%%"
echo echo. ^>^> "%%LOG_FILE%%"
echo.
echo start "" "Ladbon AI Desktop.exe" --use-cuda --verbose --n-gpu-layers 1
echo.
echo endlocal
) > "dist\Ladbon AI Desktop\Launch_with_Comprehensive_CUDA.bat"

echo Created comprehensive CUDA launcher: dist\Ladbon AI Desktop\Launch_with_Comprehensive_CUDA.bat

echo ==========================================================
echo Setup Complete
echo ==========================================================
echo.
echo Next steps:
echo 1. Run "dist\Ladbon AI Desktop\Run_CUDA_Test_Full.bat" to test CUDA support
echo 2. Use "dist\Ladbon AI Desktop\Launch_with_Comprehensive_CUDA.bat" to launch with CUDA
echo.

pause
