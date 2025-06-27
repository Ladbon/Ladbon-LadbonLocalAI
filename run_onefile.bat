@echo off
echo ========================================
echo Launching Ladbon AI Desktop (Single EXE)
echo ========================================
echo.
echo If you encounter issues with this build, try using
echo the directory version instead (dist\Ladbon AI Desktop\Ladbon AI Desktop.exe)
echo.
echo Starting application...

:: Create a timestamp for the log filename
for /f "tokens=1-6 delims=/: " %%a in ("%date% %time%") do (
  set TIMESTAMP=%%c%%a%%b-%%d%%e%%f
)

:: Run the application and capture any errors
"dist\Ladbon AI Desktop.exe" 2> "ladbon-error-%TIMESTAMP%.log"

:: Check if the application exited with an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%
    echo Checking for error logs...
    
    if exist "ladbon-error-%TIMESTAMP%.log" (
        echo Error log found. Contents:
        echo ----------------------------------------
        type "ladbon-error-%TIMESTAMP%.log"
        echo ----------------------------------------
    ) else (
        echo No error log found.
    )
    
    echo.
    echo Please check the application directory for log files.
    echo If you see errors related to "llama_cpp\lib", try using
    echo the directory version instead: dist\Ladbon AI Desktop\Ladbon AI Desktop.exe
) else (
    echo Application exited successfully.
)

echo.
pause
