@echo off
echo ================================
echo Ladbon AI Desktop DLL Diagnostics
echo ================================

echo.
echo Checking for llama.dll in installed location...
echo.

set APP_DIR=%ProgramFiles%\Ladbon AI Desktop
if not exist "%APP_DIR%" (
    set APP_DIR=%ProgramFiles(x86)%\Ladbon AI Desktop
)

echo Looking in: %APP_DIR%
echo.

echo Searching for llama.dll in application directories...
dir /s /b "%APP_DIR%\llama.dll" 2>nul
dir /s /b "%APP_DIR%\_internal\llama_cpp\lib\llama.dll" 2>nul

echo.
echo Checking for loaded llama.dll in running processes...
echo.

tasklist /m llama.dll
tasklist /fi "imagename eq Ladbon AI Desktop.exe" /m llama.dll

echo.
echo Environment variables:
echo.
echo PATH=%PATH%

echo.
echo ================================
echo Diagnostic information complete
echo ================================
echo.
echo You can run this script after installing the app to check DLL loading issues.
echo If the DLL is found in multiple locations, that could cause conflicts.
echo The DLL should only be present in _internal\llama_cpp\lib directory.

pause
