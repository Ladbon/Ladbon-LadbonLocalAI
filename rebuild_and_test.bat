@echo off
echo ======================================
echo Building Ladbon AI Desktop from scratch
echo ======================================

echo.
echo Cleaning up old builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo.
echo Building the application...
python package.py

echo.
echo Testing the built application...
start "" "dist\Ladbon AI Desktop\Ladbon AI Desktop.exe"

echo.
echo Done! Check the application window to verify it works correctly.
