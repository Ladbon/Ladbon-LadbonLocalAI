@echo off
echo Running Ladbon AI Desktop from command line...
cd "dist\Ladbon AI Desktop"
echo Current directory: %CD%
echo.
echo Showing any direct console output (errors should appear here):
echo ========================================
"Ladbon AI Desktop.exe"
echo ========================================
echo.
echo Application execution completed.
cd ..\..
