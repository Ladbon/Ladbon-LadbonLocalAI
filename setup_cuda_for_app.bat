@echo off
echo Setting up CUDA environment for Ladbon AI Desktop...

REM Add known CUDA paths to PATH
for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*") do (
    echo Found CUDA: %%G
    set "PATH=%%G\bin;%PATH%"
)

REM Add the application's llama_cpp/lib directory to PATH
set "PATH=%~dp0dist\Ladbon AI Desktop\_internal\llama_cpp\lib;%PATH%"
set "PATH=%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls;%PATH%"

REM Create a copy of the CUDA DLLs in the app's directory if needed
if not exist "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls" (
    mkdir "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls"
    echo Created cuda_dlls directory
)

REM Copy CUDA DLLs from system to the app if they don't exist
for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*") do (
    if exist "%%G\bin\cudart64*.dll" (
        if not exist "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls\cudart64*.dll" (
            echo Copying CUDA DLLs from %%G\bin to cuda_dlls...
            copy "%%G\bin\cudart64*.dll" "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls\" /y
            copy "%%G\bin\cublas64*.dll" "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls\" /y
            copy "%%G\bin\cublasLt64*.dll" "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls\" /y
            copy "%%G\bin\curand64*.dll" "%~dp0dist\Ladbon AI Desktop\_internal\cuda_dlls\" /y
        )
    )
)

REM Create a launcher batch file in the dist directory
set "LAUNCHER=%~dp0dist\Ladbon AI Desktop\Launch_with_CUDA.bat"
echo Creating CUDA launcher: %LAUNCHER%

echo @echo off > "%LAUNCHER%"
echo echo Setting up CUDA environment for Ladbon AI Desktop... >> "%LAUNCHER%"
echo. >> "%LAUNCHER%"

REM Add CUDA paths to the launcher
for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*") do (
    echo set "PATH=%%G\bin;%%PATH%%" >> "%LAUNCHER%"
)

REM Add the application's directories to the launcher
echo set "PATH=%%~dp0_internal\llama_cpp\lib;%%PATH%%" >> "%LAUNCHER%"
echo set "PATH=%%~dp0_internal\cuda_dlls;%%PATH%%" >> "%LAUNCHER%"
echo. >> "%LAUNCHER%"
echo echo Launching application with CUDA support... >> "%LAUNCHER%"
echo "%%~dp0Ladbon AI Desktop.exe" >> "%LAUNCHER%"

REM Copy the CUDA diagnostics script to the app
copy "%~dp0cuda_diagnostics.py" "%~dp0dist\Ladbon AI Desktop\_internal\cuda_diagnostics.py" /y

REM Create a diagnostic batch file in the dist directory
set "DIAG=%~dp0dist\Ladbon AI Desktop\Check_CUDA.bat"
echo Creating CUDA diagnostics: %DIAG%

echo @echo off > "%DIAG%"
echo echo Checking CUDA setup... >> "%DIAG%"
echo. >> "%DIAG%"

REM Add CUDA paths to the diagnostic script
for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*") do (
    echo set "PATH=%%G\bin;%%PATH%%" >> "%DIAG%"
)

REM Add the application's directories to the diagnostic script
echo set "PATH=%%~dp0_internal\llama_cpp\lib;%%PATH%%" >> "%DIAG%"
echo set "PATH=%%~dp0_internal\cuda_dlls;%%PATH%%" >> "%DIAG%"
echo. >> "%DIAG%"
echo echo Current PATH: >> "%DIAG%"
echo echo %%PATH%% >> "%DIAG%"
echo. >> "%DIAG%"
echo echo Checking for CUDA DLLs: >> "%DIAG%"
echo where cudart64*.dll >> "%DIAG%"
echo where cublas64*.dll >> "%DIAG%"
echo. >> "%DIAG%"
echo echo Running CUDA diagnostics... >> "%DIAG%"
echo "%%~dp0_internal\python.exe" "%%~dp0_internal\cuda_diagnostics.py" >> "%DIAG%"
echo. >> "%DIAG%"
echo pause >> "%DIAG%"

echo Done! 
echo You can now:
echo 1. Run "%~dp0dist\Ladbon AI Desktop\Launch_with_CUDA.bat" to start the app with CUDA enabled
echo 2. Run "%~dp0dist\Ladbon AI Desktop\Check_CUDA.bat" to diagnose CUDA issues
