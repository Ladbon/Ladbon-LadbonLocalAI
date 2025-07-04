@echo off
echo Setting up environment for building Ladbon AI Desktop with CUDA support...

REM First, install required packages for diagnostics
call .venv\Scripts\activate.bat
pip install pywin32

REM Build the application with enhanced CUDA support
echo Building the application...
python package.py

REM Set up CUDA environment for the packaged application
echo Setting up CUDA environment...
call setup_cuda_for_app.bat

echo.
echo Build complete! You can now:
echo 1. Run "dist\Ladbon AI Desktop\Launch_with_CUDA.bat" to start the application with CUDA support
echo 2. Run "test_cuda_integration.bat" to verify CUDA works in both environments
echo 3. Run "dist\Ladbon AI Desktop\Check_CUDA.bat" if you encounter any issues

pause
