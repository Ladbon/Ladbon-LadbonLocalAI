@echo off
echo Testing llama_cpp with CUDA support...

REM Add CUDA paths to PATH
for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*") do (
    echo Found CUDA: %%G
    set "PATH=%%G\bin;%PATH%"
)

REM Set up model for testing if available
if exist "%~dp0models" (
    for /f "delims=" %%a in ('dir /b /s "%~dp0models\*.gguf" 2^>nul') do (
        echo Found model: %%a
        set "LLAMA_TEST_MODEL=%%a"
        goto :run_test
    )
)

:run_test
echo Running test with Python from the venv...
call .venv\Scripts\activate.bat
python test_llama_cpp_cuda.py
pause

echo Running test with packaged version...
cd dist
cd "Ladbon AI Desktop"
echo Setting environment variables for packaged app...
set "PATH=%~dp0dist\Ladbon AI Desktop\_internal\llama_cpp\lib;%PATH%"
set "PATH=%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls;%PATH%"
"_internal\python.exe" "%~dp0test_llama_cpp_cuda.py"
cd ..
cd ..

pause
