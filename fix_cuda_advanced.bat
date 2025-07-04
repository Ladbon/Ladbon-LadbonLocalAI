@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo    Advanced CUDA Fix for Ladbon AI Desktop
echo ====================================================
echo This script provides an advanced fix for CUDA support
echo in the packaged Ladbon AI Desktop application.
echo.

REM Get directory of this script
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Find the dist directory and packaged app
set "DIST_DIR=%SCRIPT_DIR%\dist"
set "APP_DIR=%DIST_DIR%\Ladbon AI Desktop"
set "LOG_DIR=%APP_DIR%\logs"

if not exist "%APP_DIR%" (
    echo Could not find the packaged app in %DIST_DIR%
    echo Please make sure you've built the app with PyInstaller first.
    goto :error
)

echo Found packaged app at: %APP_DIR%
echo.

REM Create logs directory if it doesn't exist
if not exist "%LOG_DIR%" (
    echo Creating logs directory...
    mkdir "%LOG_DIR%"
)

REM Create CUDA DLLs directory in the package
set "CUDA_DLL_DIR=%APP_DIR%\_internal\cuda_dlls"
if not exist "%CUDA_DLL_DIR%" (
    echo Creating CUDA DLLs directory: %CUDA_DLL_DIR%
    mkdir "%CUDA_DLL_DIR%"
)

REM Check if init_cuda.py exists, if not, create it
set "INIT_CUDA_PATH=%APP_DIR%\_internal\init_cuda.py"
if not exist "%INIT_CUDA_PATH%" (
    echo Copying init_cuda.py to the app...
    copy "%SCRIPT_DIR%\init_cuda.py" "%INIT_CUDA_PATH%" /Y
    if !errorlevel! neq 0 (
        echo Failed to copy init_cuda.py
        goto :error
    )
    echo init_cuda.py copied successfully.
) else (
    echo Found existing init_cuda.py in the app.
)

REM Find the Python environment's llama_cpp directory
set "VENV_SITE_PACKAGES="
for /f "tokens=*" %%a in ('python -c "import site; print(site.getsitepackages()[0])"') do (
    set "VENV_SITE_PACKAGES=%%a"
)

if "!VENV_SITE_PACKAGES!" == "" (
    echo Could not determine Python site-packages directory.
    echo Will continue without copying from site-packages.
) else (
    echo Found site-packages at: !VENV_SITE_PACKAGES!

    REM Check if llama_cpp is installed
    set "VENV_LLAMA_CPP=!VENV_SITE_PACKAGES!\llama_cpp"
    if exist "!VENV_LLAMA_CPP!" (
        echo Found llama_cpp at: !VENV_LLAMA_CPP!
        
        REM Copy DLLs from Python llama_cpp to the packaged app
        echo Copying DLLs from Python llama_cpp to the packaged app...
        if exist "!VENV_LLAMA_CPP!\lib" (
            xcopy "!VENV_LLAMA_CPP!\lib\*.dll" "%APP_DIR%\_internal\llama_cpp\lib\" /Y /I
            echo Copied DLLs from llama_cpp\lib
        )
    ) else (
        echo Could not find llama_cpp in site-packages.
        echo Will continue without copying from site-packages.
    )
)

REM Copy all CUDA DLLs from CUDA_PATH
echo Searching for CUDA installation...
set "CUDA_PATHS="
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "CUDA_PATHS=!CUDA_PATHS! C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

if defined CUDA_PATH (
    set "CUDA_PATHS=!CUDA_PATH!\bin !CUDA_PATHS!"
)

set "FOUND_CUDA=0"
for %%p in (!CUDA_PATHS!) do (
    if exist "%%p" (
        echo Found CUDA directory: %%p
        set "CUDA_BIN=%%p"
        set "FOUND_CUDA=1"
        
        echo Copying CUDA DLLs from %%p to %CUDA_DLL_DIR%...
        copy "%%p\cudart64_*.dll" "%CUDA_DLL_DIR%" /Y
        copy "%%p\cublas64_*.dll" "%CUDA_DLL_DIR%" /Y
        copy "%%p\cublasLt64_*.dll" "%CUDA_DLL_DIR%" /Y
        copy "%%p\nvrtc64_*.dll" "%CUDA_DLL_DIR%" /Y
        copy "%%p\nvrtc-builtins64_*.dll" "%CUDA_DLL_DIR%" /Y
    )
)

if "%FOUND_CUDA%" == "0" (
    echo No CUDA installation found in common paths.
    echo CUDA support may not work without the necessary DLLs.
)

REM Create advanced CUDA hook
echo Creating advanced CUDA hook...
set "CUDA_HOOK_PATH=%APP_DIR%\_internal\cuda_hook.py"

echo import os > "%CUDA_HOOK_PATH%"
echo import sys >> "%CUDA_HOOK_PATH%"
echo import ctypes >> "%CUDA_HOOK_PATH%"
echo import logging >> "%CUDA_HOOK_PATH%"
echo import importlib >> "%CUDA_HOOK_PATH%"
echo from ctypes import WinDLL >> "%CUDA_HOOK_PATH%"
echo import glob >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo # Set up logging >> "%CUDA_HOOK_PATH%"
echo logging.basicConfig( >> "%CUDA_HOOK_PATH%"
echo     level=logging.INFO, >> "%CUDA_HOOK_PATH%"
echo     format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s', >> "%CUDA_HOOK_PATH%"
echo     handlers=[ >> "%CUDA_HOOK_PATH%"
echo         logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "cuda_hook.log")), >> "%CUDA_HOOK_PATH%"
echo         logging.StreamHandler() >> "%CUDA_HOOK_PATH%"
echo     ] >> "%CUDA_HOOK_PATH%"
echo ) >> "%CUDA_HOOK_PATH%"
echo logger = logging.getLogger("cuda_hook") >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo logger.info("CUDA hook initializing...") >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo # Add DLL directories to search path >> "%CUDA_HOOK_PATH%"
echo app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) >> "%CUDA_HOOK_PATH%"
echo logger.info(f"App directory: {app_dir}") >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo # Directories to search for DLLs >> "%CUDA_HOOK_PATH%"
echo search_dirs = [ >> "%CUDA_HOOK_PATH%"
echo     os.path.join(os.path.dirname(__file__), "cuda_dlls"), >> "%CUDA_HOOK_PATH%"
echo     os.path.join(os.path.dirname(__file__), "llama_cpp", "lib"), >> "%CUDA_HOOK_PATH%"
echo ] >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo for search_dir in search_dirs: >> "%CUDA_HOOK_PATH%"
echo     if os.path.exists(search_dir): >> "%CUDA_HOOK_PATH%"
echo         logger.info(f"Adding DLL search path: {search_dir}") >> "%CUDA_HOOK_PATH%"
echo         try: >> "%CUDA_HOOK_PATH%"
echo             if hasattr(os, 'add_dll_directory'): >> "%CUDA_HOOK_PATH%"
echo                 os.add_dll_directory(search_dir) >> "%CUDA_HOOK_PATH%"
echo             elif search_dir not in os.environ["PATH"]: >> "%CUDA_HOOK_PATH%"
echo                 os.environ["PATH"] = search_dir + os.pathsep + os.environ["PATH"] >> "%CUDA_HOOK_PATH%"
echo         except Exception as e: >> "%CUDA_HOOK_PATH%"
echo             logger.warning(f"Failed to add DLL directory: {e}") >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo # Try to import init_cuda >> "%CUDA_HOOK_PATH%"
echo try: >> "%CUDA_HOOK_PATH%"
echo     logger.info("Importing init_cuda...") >> "%CUDA_HOOK_PATH%"
echo     import init_cuda >> "%CUDA_HOOK_PATH%"
echo     cuda_success = init_cuda.initialize_cuda() >> "%CUDA_HOOK_PATH%"
echo     logger.info(f"CUDA initialization {'succeeded' if cuda_success else 'failed'}") >> "%CUDA_HOOK_PATH%"
echo except Exception as e: >> "%CUDA_HOOK_PATH%"
echo     logger.error(f"Failed to import/run init_cuda: {e}") >> "%CUDA_HOOK_PATH%"
echo. >> "%CUDA_HOOK_PATH%"
echo logger.info("CUDA hook initialization complete") >> "%CUDA_HOOK_PATH%"

echo Advanced CUDA hook created at: %CUDA_HOOK_PATH%

REM Create an enhanced launcher
echo Creating enhanced CUDA launcher...
set "ENHANCED_LAUNCHER=%APP_DIR%\Launch with Advanced CUDA Support.bat"

echo @echo off > "%ENHANCED_LAUNCHER%"
echo setlocal enabledelayedexpansion >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"
echo echo ====================================================== >> "%ENHANCED_LAUNCHER%"
echo echo       Ladbon AI Desktop - Advanced CUDA Support        >> "%ENHANCED_LAUNCHER%"
echo echo ====================================================== >> "%ENHANCED_LAUNCHER%"
echo echo. >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Set up environment variables
echo set "SCRIPT_DIR=%%~dp0" >> "%ENHANCED_LAUNCHER%"
echo set "INTERNAL_DIR=%%SCRIPT_DIR%%\_internal" >> "%ENHANCED_LAUNCHER%"
echo set "CUDA_DLL_DIR=%%INTERNAL_DIR%%\cuda_dlls" >> "%ENHANCED_LAUNCHER%"
echo set "LLAMA_CPP_DIR=%%INTERNAL_DIR%%\llama_cpp\lib" >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Add CUDA paths dynamically
echo echo Setting up CUDA environment... >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"
echo REM Add CUDA paths to PATH >> "%ENHANCED_LAUNCHER%"
echo for %%p in ( >> "%ENHANCED_LAUNCHER%"
for %%p in (!CUDA_PATHS!) do (
    echo     "%%p" >> "%ENHANCED_LAUNCHER%"
)
echo ) do ( >> "%ENHANCED_LAUNCHER%"
echo     if exist %%p ( >> "%ENHANCED_LAUNCHER%"
echo         echo Found CUDA directory: %%p >> "%ENHANCED_LAUNCHER%"
echo         set "PATH=%%p;%%PATH%%" >> "%ENHANCED_LAUNCHER%"
echo     ) >> "%ENHANCED_LAUNCHER%"
echo ) >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Add local DLL dirs to PATH
echo if exist "%%CUDA_DLL_DIR%%" ( >> "%ENHANCED_LAUNCHER%"
echo     echo Adding CUDA DLLs directory to PATH: %%CUDA_DLL_DIR%% >> "%ENHANCED_LAUNCHER%"
echo     set "PATH=%%CUDA_DLL_DIR%%;%%PATH%%" >> "%ENHANCED_LAUNCHER%"
echo ) >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"
echo if exist "%%LLAMA_CPP_DIR%%" ( >> "%ENHANCED_LAUNCHER%"
echo     echo Adding llama_cpp lib directory to PATH: %%LLAMA_CPP_DIR%% >> "%ENHANCED_LAUNCHER%"
echo     set "PATH=%%LLAMA_CPP_DIR%%;%%PATH%%" >> "%ENHANCED_LAUNCHER%"
echo ) >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Set Python-related environment variables
echo set "PYTHONPATH=%%INTERNAL_DIR%%;%%PYTHONPATH%%" >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Run the CUDA hook script
echo echo Initializing CUDA support... >> "%ENHANCED_LAUNCHER%"
echo cd /d "%%SCRIPT_DIR%%" >> "%ENHANCED_LAUNCHER%"
echo "%%SCRIPT_DIR%%\Python\python.exe" "%%INTERNAL_DIR%%\cuda_hook.py" >> "%ENHANCED_LAUNCHER%"
echo if !errorlevel! neq 0 ( >> "%ENHANCED_LAUNCHER%"
echo     echo WARNING: CUDA initialization script had errors. >> "%ENHANCED_LAUNCHER%"
echo     echo The application will still start, but GPU support might be limited. >> "%ENHANCED_LAUNCHER%"
echo     echo See logs\cuda_hook.log for details. >> "%ENHANCED_LAUNCHER%"
echo     pause >> "%ENHANCED_LAUNCHER%"
echo ) >> "%ENHANCED_LAUNCHER%"
echo. >> "%ENHANCED_LAUNCHER%"

REM Launch the app
echo echo Starting Ladbon AI Desktop with advanced CUDA support... >> "%ENHANCED_LAUNCHER%"
echo "%%SCRIPT_DIR%%\Ladbon AI Desktop.exe" >> "%ENHANCED_LAUNCHER%"

echo Enhanced CUDA launcher created at: %ENHANCED_LAUNCHER%

REM Create a direct launcher
echo Creating direct CUDA launcher...
set "DIRECT_LAUNCHER=%APP_DIR%\Launch with Direct CUDA Support.bat"

echo @echo off > "%DIRECT_LAUNCHER%"
echo setlocal enabledelayedexpansion >> "%DIRECT_LAUNCHER%"
echo. >> "%DIRECT_LAUNCHER%"
echo echo ====================================================== >> "%DIRECT_LAUNCHER%"
echo echo       Ladbon AI Desktop - Direct CUDA Support          >> "%DIRECT_LAUNCHER%"
echo echo ====================================================== >> "%DIRECT_LAUNCHER%"
echo echo. >> "%DIRECT_LAUNCHER%"
echo. >> "%DIRECT_LAUNCHER%"

REM Add environment variables to trigger CUDA hooks directly
echo set "SCRIPT_DIR=%%~dp0" >> "%DIRECT_LAUNCHER%"
echo set "INTERNAL_DIR=%%SCRIPT_DIR%%\_internal" >> "%DIRECT_LAUNCHER%"
echo set "CUDA_DLL_DIR=%%INTERNAL_DIR%%\cuda_dlls" >> "%DIRECT_LAUNCHER%"
echo set "LLAMA_CPP_DIR=%%INTERNAL_DIR%%\llama_cpp\lib" >> "%DIRECT_LAUNCHER%"
echo. >> "%DIRECT_LAUNCHER%"

REM Add CUDA paths to PATH
echo echo Adding CUDA paths to PATH... >> "%DIRECT_LAUNCHER%"
echo set "PATH=%%CUDA_DLL_DIR%%;%%LLAMA_CPP_DIR%%;%%PATH%%" >> "%DIRECT_LAUNCHER%"
echo. >> "%DIRECT_LAUNCHER%"

REM Launch the app directly
echo echo Starting Ladbon AI Desktop with direct CUDA support... >> "%DIRECT_LAUNCHER%"
echo cd /d "%%SCRIPT_DIR%%" >> "%DIRECT_LAUNCHER%"
echo "%%SCRIPT_DIR%%\Ladbon AI Desktop.exe" >> "%DIRECT_LAUNCHER%"

echo Direct CUDA launcher created at: %DIRECT_LAUNCHER%

REM Create a README file with instructions
echo Creating README file with instructions...
set "README_PATH=%APP_DIR%\CUDA_SUPPORT_README.txt"

echo ====================================================== > "%README_PATH%"
echo        CUDA Support for Ladbon AI Desktop              >> "%README_PATH%"
echo ====================================================== >> "%README_PATH%"
echo. >> "%README_PATH%"
echo This application has been patched with enhanced CUDA support. >> "%README_PATH%"
echo. >> "%README_PATH%"
echo To start the application with GPU support, use one of the following launchers: >> "%README_PATH%"
echo. >> "%README_PATH%"
echo 1. "Launch with Advanced CUDA Support.bat" >> "%README_PATH%"
echo    - Recommended for most users >> "%README_PATH%"
echo    - Sets up CUDA paths and runs initialization script >> "%README_PATH%"
echo    - Provides detailed logging of CUDA initialization >> "%README_PATH%"
echo. >> "%README_PATH%"
echo 2. "Launch with Direct CUDA Support.bat" >> "%README_PATH%"
echo    - Simpler launcher that just adds CUDA paths to PATH >> "%README_PATH%"
echo    - Use if the advanced launcher doesn't work >> "%README_PATH%"
echo. >> "%README_PATH%"
echo If you still experience issues with CUDA/GPU support: >> "%README_PATH%"
echo. >> "%README_PATH%"
echo - Check the logs directory for cuda_hook.log and other logs >> "%README_PATH%"
echo - Make sure your NVIDIA drivers are up to date >> "%README_PATH%"
echo - Verify that CUDA 12.x is installed on your system >> "%README_PATH%"
echo. >> "%README_PATH%"

echo README created at: %README_PATH%

echo.
echo ====================================================
echo             Advanced CUDA Fix Complete!
echo ====================================================
echo.
echo The packaged application has been updated with:
echo.
echo 1. CUDA initialization module (init_cuda.py)
echo 2. CUDA hook for automatic DLL loading (cuda_hook.py)
echo 3. CUDA DLLs in _internal\cuda_dlls directory
echo 4. Enhanced CUDA launchers:
echo    - Launch with Advanced CUDA Support.bat
echo    - Launch with Direct CUDA Support.bat
echo 5. Instructions in CUDA_SUPPORT_README.txt
echo.
echo Please use one of the new launchers to start the application.
echo.
goto :eof

:error
echo.
echo ERROR: An error occurred during the CUDA fix process.
echo Please try running the script again as administrator.
exit /b 1
