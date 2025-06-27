@echo off
echo Running test for packaged application
echo.

set LOG_DIR=%~dp0logs
set DIST_DIR=%~dp0dist

echo Log directory: %LOG_DIR%
echo.

echo Counting bootstrap logs before running...
dir /b "%LOG_DIR%\gui_bootstrap_*.log" | find /c /v "" > count_before.txt
set /p COUNT_BEFORE=<count_before.txt
echo Found %COUNT_BEFORE% bootstrap logs before running
echo.

echo Starting the application...
start "" "%DIST_DIR%\Ladbon AI Desktop.exe"
echo Waiting 10 seconds for application to start...
timeout /t 10 /nobreak > nul

echo Counting bootstrap logs after running...
dir /b "%LOG_DIR%\gui_bootstrap_*.log" | find /c /v "" > count_after.txt
set /p COUNT_AFTER=<count_after.txt
echo Found %COUNT_AFTER% bootstrap logs after running
echo.

if %COUNT_AFTER% GTR %COUNT_BEFORE% (
    echo SUCCESS: New log file(s) were created!
    echo Getting the newest log file...
    for /f "tokens=*" %%a in ('dir /b /od "%LOG_DIR%\gui_bootstrap_*.log"') do set NEWEST_LOG=%%a
    echo Newest log file: %NEWEST_LOG%
    echo.
    echo --- Log Contents ---
    type "%LOG_DIR%\%NEWEST_LOG%"
    echo -------------------
) else (
    echo ERROR: No new log files were created.
    echo Something might be wrong with the application.
)

echo.
echo Test completed.
del count_before.txt
del count_after.txt
pause
