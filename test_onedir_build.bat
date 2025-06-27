@echo off
echo Testing Ladbon AI Desktop Onedir Build
echo.

set DIST_DIR=%~dp0dist\Ladbon AI Desktop
if not exist "%DIST_DIR%" (
    echo ERROR: Build directory not found at: %DIST_DIR%
    goto :end
)

echo Build directory found at: %DIST_DIR%
echo.

echo === Directory Contents ===
dir "%DIST_DIR%"
echo.

echo === Checking for llama_cpp lib directory ===
if exist "%DIST_DIR%\llama_cpp\lib\" (
    echo llama_cpp\lib directory found!
    echo Contents:
    dir "%DIST_DIR%\llama_cpp\lib\"
) else (
    echo llama_cpp\lib directory NOT FOUND!
)
echo.

echo === Checking for DLLs in root directory ===
echo DLLs in root directory:
dir "%DIST_DIR%\*.dll" 2>nul || echo No DLLs found in root directory
echo.

echo === Starting application ===
echo Starting Ladbon AI Desktop...
start "" /wait "%DIST_DIR%\Ladbon AI Desktop.exe"
echo.

echo Application should be starting now.
echo Check for error dialogs or logs in the logs directory.

:end
pause
