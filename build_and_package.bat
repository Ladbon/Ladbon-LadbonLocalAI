@echo off
echo ===========================================================
echo Building Ladbon AI Desktop with DLL fixes and path changes
echo ===========================================================

echo.
echo 1. Cleaning up old builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo 2. Building the application...
python package.py

echo.
echo 3. Creating installer...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" LadboAIDesktop.iss

echo.
echo Done! 
echo The installer has been created in the installer directory.
echo.
echo You can now test the installer. Models and user data will be stored in:
echo C:\Users\[username]\Ladbon AI Desktop
echo.
pause
