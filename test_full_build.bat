@echo off
echo ==============================================
echo Testing Ladbon AI Desktop with NumPy CPU Dispatcher Fix
echo ==============================================

rem Create necessary directories if they don't exist
if not exist "dist\Ladbon AI Desktop\logs" mkdir "dist\Ladbon AI Desktop\logs"
if not exist "dist\Ladbon AI Desktop\models" mkdir "dist\Ladbon AI Desktop\models"
if not exist "dist\Ladbon AI Desktop\docs" mkdir "dist\Ladbon AI Desktop\docs"
if not exist "dist\Ladbon AI Desktop\img" mkdir "dist\Ladbon AI Desktop\img"

echo Directories created/checked.
echo.
echo Running application...
cd "dist\Ladbon AI Desktop"

rem Launch the application and capture its PID
start /B "" "Ladbon AI Desktop.exe"
echo Application started.
echo.

rem Wait a moment to allow the app to start and create log files
echo Waiting for 10 seconds to allow app to initialize...
timeout /t 10 > nul

echo Checking for log files...
echo.
dir /B logs\*.log 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo No log files found. Application may have crashed or logs are in a different location.
) else (
    echo Log files found! Application is likely running.
)

echo.
echo Checking for crash dumps or error reports...
dir /B *.dmp 2>nul
dir /B *crash*.txt 2>nul

echo.
echo ==============================================
echo Test completed
echo ==============================================
cd ..\..
