@echo off
echo ============================================================
echo PATCHING LLAMACPP_LOADER FOR PACKAGED APPLICATION
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

REM Find the llamacpp_loader.py file in the packaged app
set "LOADER_DIR=%APP_DIR%\_internal\utils"
set "LOADER_FILE=%LOADER_DIR%\llamacpp_loader.py"
set "LOADER_PYCA=%LOADER_DIR%\llamacpp_loader.pyc"
set "LOADER_PYCB=%LOADER_DIR%\__pycache__\llamacpp_loader.cpython-*.pyc"

REM Check if Python source is available
if not exist "%LOADER_FILE%" (
    echo WARNING: Could not find llamacpp_loader.py in the packaged app
    echo This is expected with PyInstaller, will create new file for patching...
    echo.
    echo import os, sys, ctypes, glob > "%LOADER_FILE%"
    echo import logging >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo logger = logging.getLogger("llamacpp_loader_patch") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo def safe_initialize_backend(): >> "%LOADER_FILE%"
    echo     """Safely initialize the llama-cpp backend with proper CUDA support""" >> "%LOADER_FILE%"
    echo     logger.info("Patched llamacpp_loader.py initializing backend...") >> "%LOADER_FILE%"
    echo     # Add code to initialize llama-cpp backend >> "%LOADER_FILE%"
    echo     try: >> "%LOADER_FILE%"
    echo         # Setup CUDA paths >> "%LOADER_FILE%"
    echo         cuda_dlls_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda_dlls") >> "%LOADER_FILE%"
    echo         if os.path.exists(cuda_dlls_dir): >> "%LOADER_FILE%"
    echo             logger.info(f"Found CUDA DLLs directory: {cuda_dlls_dir}") >> "%LOADER_FILE%"
    echo             dlls = glob.glob(os.path.join(cuda_dlls_dir, "*.dll")) >> "%LOADER_FILE%"
    echo             logger.info(f"Found {len(dlls)} CUDA DLLs") >> "%LOADER_FILE%"
    echo             os.environ["PATH"] = cuda_dlls_dir + os.pathsep + os.environ.get("PATH", "") >> "%LOADER_FILE%"
    echo             try: >> "%LOADER_FILE%"
    echo                 os.add_dll_directory(cuda_dlls_dir) >> "%LOADER_FILE%"
    echo                 logger.info(f"Added DLL directory: {cuda_dlls_dir}") >> "%LOADER_FILE%"
    echo             except Exception as e: >> "%LOADER_FILE%"
    echo                 logger.error(f"Error adding DLL directory: {e}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         import llama_cpp >> "%LOADER_FILE%"
    echo         logger.info("Successfully imported llama_cpp") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         # First try with no parameters (newer versions) >> "%LOADER_FILE%"
    echo         try: >> "%LOADER_FILE%"
    echo             logger.info("Trying to initialize backend with no parameters") >> "%LOADER_FILE%"
    echo             llama_cpp.llama_backend_init() >> "%LOADER_FILE%"
    echo             logger.info("Successfully initialized backend with no parameters") >> "%LOADER_FILE%"
    echo             return True >> "%LOADER_FILE%"
    echo         except Exception as e1: >> "%LOADER_FILE%"
    echo             logger.warning(f"Failed to initialize backend with no parameters: {e1}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         # Try with False parameter (older versions) >> "%LOADER_FILE%"
    echo         try: >> "%LOADER_FILE%"
    echo             logger.info("Trying to initialize backend with False parameter") >> "%LOADER_FILE%"
    echo             # Use getattr to avoid linting errors >> "%LOADER_FILE%"
    echo             backend_init_func = getattr(llama_cpp, "llama_backend_init") >> "%LOADER_FILE%"
    echo             backend_init_func.__call__(False) >> "%LOADER_FILE%"
    echo             logger.info("Successfully initialized backend with False parameter") >> "%LOADER_FILE%"
    echo             return True >> "%LOADER_FILE%"
    echo         except Exception as e2: >> "%LOADER_FILE%"
    echo             logger.warning(f"Failed to initialize backend with False parameter: {e2}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         # Try direct C library call as last resort >> "%LOADER_FILE%"
    echo         try: >> "%LOADER_FILE%"
    echo             logger.info("Trying to access C library directly") >> "%LOADER_FILE%"
    echo             # Try all known library paths >> "%LOADER_FILE%"
    echo             lib = None >> "%LOADER_FILE%"
    echo             if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"): >> "%LOADER_FILE%"
    echo                 lib = llama_cpp.llama_cpp._lib >> "%LOADER_FILE%"
    echo                 logger.info("Found library at llama_cpp.llama_cpp._lib") >> "%LOADER_FILE%"
    echo             elif hasattr(llama_cpp, "_lib"): >> "%LOADER_FILE%"
    echo                 lib = llama_cpp._lib >> "%LOADER_FILE%"
    echo                 logger.info("Found library at llama_cpp._lib") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo             if lib and hasattr(lib, "llama_backend_init"): >> "%LOADER_FILE%"
    echo                 logger.info("Calling C function directly") >> "%LOADER_FILE%"
    echo                 lib.llama_backend_init() >> "%LOADER_FILE%"
    echo                 logger.info("Successfully initialized backend with direct C call") >> "%LOADER_FILE%"
    echo                 return True >> "%LOADER_FILE%"
    echo             else: >> "%LOADER_FILE%"
    echo                 logger.warning("Could not find C library or llama_backend_init function") >> "%LOADER_FILE%"
    echo         except Exception as e3: >> "%LOADER_FILE%"
    echo             logger.warning(f"Failed to initialize backend with direct C call: {e3}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         # Force CPU-only mode as last resort >> "%LOADER_FILE%"
    echo         logger.warning("All GPU initialization attempts failed, trying CPU-only mode") >> "%LOADER_FILE%"
    echo         os.environ["CUDA_VISIBLE_DEVICES"] = "-1" >> "%LOADER_FILE%"
    echo         os.environ["DISABLE_CUDA"] = "1" >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         try: >> "%LOADER_FILE%"
    echo             llama_cpp.llama_backend_init() >> "%LOADER_FILE%"
    echo             logger.info("Successfully initialized backend in CPU-only mode") >> "%LOADER_FILE%"
    echo             return True >> "%LOADER_FILE%"
    echo         except Exception as cpu_err: >> "%LOADER_FILE%"
    echo             logger.error(f"Failed to initialize backend even in CPU-only mode: {cpu_err}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo     except Exception as e: >> "%LOADER_FILE%"
    echo         logger.error(f"Fatal error in backend initialization: {e}") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo     return False >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo def check_cuda_availability(): >> "%LOADER_FILE%"
    echo     """Check if CUDA is available and properly set up""" >> "%LOADER_FILE%"
    echo     try: >> "%LOADER_FILE%"
    echo         # Check for CUDA in PATH >> "%LOADER_FILE%"
    echo         cuda_in_path = False >> "%LOADER_FILE%"
    echo         paths = os.environ.get("PATH", "").split(os.pathsep) >> "%LOADER_FILE%"
    echo         cuda_paths = [p for p in paths if "cuda" in p.lower()] >> "%LOADER_FILE%"
    echo         logger.info(f"Found {len(cuda_paths)} CUDA directories in PATH") >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         # Check for packaged CUDA DLLs >> "%LOADER_FILE%"
    echo         cuda_dlls_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda_dlls") >> "%LOADER_FILE%"
    echo         if os.path.exists(cuda_dlls_dir): >> "%LOADER_FILE%"
    echo             dlls = glob.glob(os.path.join(cuda_dlls_dir, "*.dll")) >> "%LOADER_FILE%"
    echo             logger.info(f"Found {len(dlls)} packaged CUDA DLLs") >> "%LOADER_FILE%"
    echo             if len(dlls) > 0: >> "%LOADER_FILE%"
    echo                 return True >> "%LOADER_FILE%"
    echo. >> "%LOADER_FILE%"
    echo         return len(cuda_paths) > 0 >> "%LOADER_FILE%"
    echo     except Exception as e: >> "%LOADER_FILE%"
    echo         logger.error(f"Error checking CUDA availability: {e}") >> "%LOADER_FILE%"
    echo         return False >> "%LOADER_FILE%"
) else (
    echo Found llamacpp_loader.py: %LOADER_FILE%
    echo Creating backup...
    copy "%LOADER_FILE%" "%LOADER_FILE%.bak"
    echo.
)

echo Creating custom CUDA initialization runtime hook...
echo.

REM Create runtime hook for CUDA initialization
set "RUNTIME_HOOK=%APP_DIR%\init_cuda.py"

echo import os, sys, ctypes, glob > "%RUNTIME_HOOK%"
echo import logging >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo # Set up logging >> "%RUNTIME_HOOK%"
echo log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") >> "%RUNTIME_HOOK%"
echo os.makedirs(log_dir, exist_ok=True) >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo logging.basicConfig( >> "%RUNTIME_HOOK%"
echo     level=logging.INFO, >> "%RUNTIME_HOOK%"
echo     format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s', >> "%RUNTIME_HOOK%"
echo     handlers=[ >> "%RUNTIME_HOOK%"
echo         logging.FileHandler(os.path.join(log_dir, "cuda_init.log"), mode='w') >> "%RUNTIME_HOOK%"
echo     ] >> "%RUNTIME_HOOK%"
echo ) >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo logger = logging.getLogger("cuda_init") >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo logger.info("CUDA initialization hook running...") >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo # Find and add CUDA DLLs directory to PATH >> "%RUNTIME_HOOK%"
echo cuda_dlls_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_internal", "cuda_dlls") >> "%RUNTIME_HOOK%"
echo if os.path.exists(cuda_dlls_dir): >> "%RUNTIME_HOOK%"
echo     logger.info(f"Found CUDA DLLs directory: {cuda_dlls_dir}") >> "%RUNTIME_HOOK%"
echo     dlls = glob.glob(os.path.join(cuda_dlls_dir, "*.dll")) >> "%RUNTIME_HOOK%"
echo     logger.info(f"Found {len(dlls)} CUDA DLLs:") >> "%RUNTIME_HOOK%"
echo     for dll in dlls: >> "%RUNTIME_HOOK%"
echo         logger.info(f"  - {os.path.basename(dll)}") >> "%RUNTIME_HOOK%"
echo     # Add DLL directory to PATH >> "%RUNTIME_HOOK%"
echo     os.environ["PATH"] = cuda_dlls_dir + os.pathsep + os.environ.get("PATH", "") >> "%RUNTIME_HOOK%"
echo     try: >> "%RUNTIME_HOOK%"
echo         os.add_dll_directory(cuda_dlls_dir) >> "%RUNTIME_HOOK%"
echo         logger.info(f"Added DLL directory: {cuda_dlls_dir}") >> "%RUNTIME_HOOK%"
echo     except Exception as e: >> "%RUNTIME_HOOK%"
echo         logger.warning(f"Error adding DLL directory: {e}") >> "%RUNTIME_HOOK%"
echo else: >> "%RUNTIME_HOOK%"
echo     logger.warning(f"CUDA DLLs directory not found: {cuda_dlls_dir}") >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo # Find and add llama_cpp lib directory to PATH >> "%RUNTIME_HOOK%"
echo llama_cpp_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_internal", "llama_cpp", "lib") >> "%RUNTIME_HOOK%"
echo if os.path.exists(llama_cpp_lib_dir): >> "%RUNTIME_HOOK%"
echo     logger.info(f"Found llama_cpp lib directory: {llama_cpp_lib_dir}") >> "%RUNTIME_HOOK%"
echo     dlls = glob.glob(os.path.join(llama_cpp_lib_dir, "*.dll")) >> "%RUNTIME_HOOK%"
echo     logger.info(f"Found {len(dlls)} llama_cpp DLLs") >> "%RUNTIME_HOOK%"
echo     # Add DLL directory to PATH >> "%RUNTIME_HOOK%"
echo     os.environ["PATH"] = llama_cpp_lib_dir + os.pathsep + os.environ.get("PATH", "") >> "%RUNTIME_HOOK%"
echo     try: >> "%RUNTIME_HOOK%"
echo         os.add_dll_directory(llama_cpp_lib_dir) >> "%RUNTIME_HOOK%"
echo         logger.info(f"Added DLL directory: {llama_cpp_lib_dir}") >> "%RUNTIME_HOOK%"
echo     except Exception as e: >> "%RUNTIME_HOOK%"
echo         logger.warning(f"Error adding DLL directory: {e}") >> "%RUNTIME_HOOK%"
echo else: >> "%RUNTIME_HOOK%"
echo     logger.warning(f"llama_cpp lib directory not found: {llama_cpp_lib_dir}") >> "%RUNTIME_HOOK%"
echo. >> "%RUNTIME_HOOK%"
echo logger.info("CUDA initialization hook completed") >> "%RUNTIME_HOOK%"

echo.
echo Creating custom launcher script...
echo.

REM Create an enhanced launcher batch file
set "LAUNCHER_PATH=%~dp0dist\Launch with CUDA Fixed.bat"

echo @echo off > "%LAUNCHER_PATH%"
echo echo Setting up CUDA environment for Ladbon AI Desktop... >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo cd "%%~dp0Ladbon AI Desktop" >> "%LAUNCHER_PATH%"
echo echo Running custom CUDA initialization... >> "%LAUNCHER_PATH%"
echo python init_cuda.py >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo echo Setting environment variables... >> "%LAUNCHER_PATH%"
echo set "CUDA_VISIBLE_DEVICES=0" >> "%LAUNCHER_PATH%"
echo set "PATH=%%CD%%\_internal\cuda_dlls;%%CD%%\_internal\llama_cpp\lib;%%PATH%%" >> "%LAUNCHER_PATH%"
echo echo Starting application... >> "%LAUNCHER_PATH%"
echo echo. >> "%LAUNCHER_PATH%"
echo start "" "Ladbon AI Desktop.exe" >> "%LAUNCHER_PATH%"

echo Created fixed CUDA launcher: %LAUNCHER_PATH%

echo.
echo ============================================================
echo PATCHING COMPLETE!
echo ============================================================
echo.
echo Please use the new launcher "Launch with CUDA Fixed.bat"
echo to start the application with proper CUDA support.
echo.
echo IMPORTANT: Run fix_cuda_for_packaged_app.bat first to ensure
echo CUDA DLLs are properly copied to the application directory.
echo.
pause
