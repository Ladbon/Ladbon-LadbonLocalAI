@echo off
echo ===================================================
echo Launching Ladbon AI Desktop with proper DLL loading
echo ===================================================

:: Set the path to the only location where DLLs should be
set "LLAMA_LIB=%~dp0_internal\llama_cpp\lib"

:: Basic verification that we're set up correctly
if not exist "%LLAMA_LIB%" (
    echo [ERROR] Critical folder not found: %LLAMA_LIB%
    echo This indicates a broken installation. Please reinstall.
    pause
    exit /b 1
)

if not exist "%LLAMA_LIB%\llama.dll" (
    echo [ERROR] Critical DLL not found: %LLAMA_LIB%\llama.dll
    echo This indicates a broken installation. Please reinstall.
    pause
    exit /b 1
)

:: Ensure our DLLs are loaded first (and only these copies)
echo [INFO] Setting PATH to prioritize: %LLAMA_LIB%
set "PATH=%LLAMA_LIB%;%PATH%"

:: Debug output for troubleshooting
echo [INFO] Launching from: %~dp0
echo [INFO] DLL directory: %LLAMA_LIB%
echo [INFO] Starting executable...
echo.

:: Change to app directory and start the app
cd /d "%~dp0"
start "" "%~dp0Ladbon AI Desktop.exe" %*

:: End of script
exit /b 0
