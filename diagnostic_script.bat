@echo off
setlocal

REM Get the path to the virtual environment's site-packages
call venv\Scripts\activate.bat
for /f "tokens=*" %%i in ('python -c "import site; print(site.getsitepackages()[0])"') do set "SITE_PACKAGES=%%i"
deactivate

REM Define the destination directory for the redistributables
set "DEST_DIR=%~dp0redist"

REM Clean the destination directory
if exist "%DEST_DIR%" (
    echo Cleaning destination directory: %DEST_DIR%
    rmdir /s /q "%DEST_DIR%"
)
mkdir "%DEST_DIR%"

REM Run the Python script to copy the necessary files
echo Running Python script to copy dependencies...
python "%~dp0diagnostic_script.py" "%SITE_PACKAGES%" "%DEST_DIR%"

endlocal
echo.
echo Dependency copying process complete.
echo Files are in the `redist` directory.
pause
