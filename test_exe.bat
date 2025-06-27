@echo off
echo Testing Ladbon AI Desktop executable...
echo Running from: %~dp0
echo.

if not exist "%~dp0dist\Ladbon AI Desktop.exe" (
  echo ERROR: Executable not found at "%~dp0dist\Ladbon AI Desktop.exe"
  goto :end
)

echo Starting application...
cd "%~dp0dist"
start "" "Ladbon AI Desktop.exe"

echo.
echo Application started. Check for any error dialogs.
echo Look in the logs directory for detailed log files.

:end
pause
