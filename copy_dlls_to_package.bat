@echo off
REM This script copies the llama_cpp lib DLLs from the venv to the packaged app

echo Copying llama_cpp DLLs from venv to packaged app...

set "VENV_PATH=C:\Users\ladfr\source\localai\src\.venv"
set "PACKAGE_PATH=C:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop"

REM Check if venv exists
if not exist "%VENV_PATH%\Lib\site-packages\llama_cpp\lib" (
    echo ERROR: Could not find llama_cpp lib directory in venv
    echo Expected at: %VENV_PATH%\Lib\site-packages\llama_cpp\lib
    goto :error
)

REM Check if packaged app exists
if not exist "%PACKAGE_PATH%" (
    echo ERROR: Could not find packaged app directory
    echo Expected at: %PACKAGE_PATH%
    goto :error
)

REM Create destination directory if it doesn't exist
if not exist "%PACKAGE_PATH%\_internal\llama_cpp\lib" (
    echo Creating destination directory...
    mkdir "%PACKAGE_PATH%\_internal\llama_cpp\lib"
)

REM Copy all DLLs from venv to packaged app
echo Copying DLLs...
copy /Y "%VENV_PATH%\Lib\site-packages\llama_cpp\lib\*.dll" "%PACKAGE_PATH%\_internal\llama_cpp\lib\"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy DLLs
    goto :error
)

REM List copied files
echo Listing copied DLLs:
dir "%PACKAGE_PATH%\_internal\llama_cpp\lib\*.dll"

echo.
echo SUCCESS! All DLLs copied successfully.
echo Try running the packaged app now.
exit /b 0

:error
echo.
echo Failed to copy DLLs. Please check the errors above.
exit /b 1
