@echo off
echo =========================================================
echo             Ladbon AI Desktop - CPU-Only Mode
echo =========================================================
echo.
echo This launcher forces the application to run in CPU-only mode.
echo Models will still work but will be slower without GPU acceleration.
echo.
echo Setting environment variables to force CPU-only mode...
set FORCE_CPU_ONLY=1
set CUDA_VISIBLE_DEVICES=-1
set DISABLE_CUDA=1

echo.
echo Launching Ladbon AI Desktop in CPU-only mode...
echo (This window can be closed once the application opens)
echo.
cd "%~dp0dist\Ladbon AI Desktop"
start "" "Ladbon AI Desktop.exe"
echo.
echo If the application works in CPU-only mode but not with GPU,
echo you may be missing the required CUDA runtime libraries.
echo.
pause
