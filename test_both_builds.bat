@echo off
echo ==============================================
echo Testing Ladbon AI Desktop Bundle
echo ==============================================

echo Creating a timestamp for logs...
for /f "tokens=1-6 delims=/: " %%a in ("%date% %time%") do (
  set TIMESTAMP=%%c%%a%%b-%%d%%e%%f
)

echo.
echo Testing onedir version (recommended)...
echo Starting: dist\Ladbon AI Desktop\Ladbon AI Desktop.exe
cd "dist\Ladbon AI Desktop"
start "Onedir Test" "Ladbon AI Desktop.exe"
cd ..\..
echo Onedir version launched!
echo.

echo Waiting 10 seconds before testing onefile version...
timeout /t 10 > nul

echo.
echo Testing onefile version...
start "Onefile Test" "dist\Ladbon AI Desktop.exe"
echo Onefile version launched!
echo.

echo ==============================================
echo Both versions have been launched for testing
echo Check for any error messages or log files
echo created in the application directories
echo ==============================================

echo.
echo After testing, check these log files:
echo - numpy_fix_debug_*.log
echo - numpy_hook_*.log 
echo - onefile_loader_*.log

pause
