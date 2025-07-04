@echo off
echo ==== Enhanced Copying of llama_cpp packages to packaged app ====
echo.

:: Change to the directory where the batch file is located
cd /d "%~dp0"

:: Check if we're in the right directory structure
if not exist "dist\Ladbon AI Desktop\_internal" (
    echo ERROR: Cannot find dist\Ladbon AI Desktop\_internal
    echo Make sure you're running this script from the source directory.
    pause
    exit /b 1
)

:: Find the Python executable and site-packages directory
:: First check for virtualenv
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    for /f "tokens=*" %%i in ('where python') do set PYTHON_EXE=%%i
)
echo Found Python executable: %PYTHON_EXE%

:: Get the site-packages directory
for /f "tokens=*" %%i in ('%PYTHON_EXE% -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%i
echo Site-packages directory: %SITE_PACKAGES%

:: Adjust for venv structure if needed
if not exist "%SITE_PACKAGES%\llama_cpp" (
    if exist "%SITE_PACKAGES%\Lib\site-packages\llama_cpp" (
        set SITE_PACKAGES=%SITE_PACKAGES%\Lib\site-packages
        echo Adjusted site-packages directory: %SITE_PACKAGES%
    )
)

:: Create diagnostic log
echo =================================================== > copy_package_log.txt
echo Python Executable: %PYTHON_EXE% >> copy_package_log.txt
echo Site-packages: %SITE_PACKAGES% >> copy_package_log.txt
echo =================================================== >> copy_package_log.txt

:: Check for both the llama_cpp module and llama_cpp_python dist-info
if not exist "%SITE_PACKAGES%\llama_cpp" (
    echo ERROR: llama_cpp not found in %SITE_PACKAGES%
    echo Make sure llama-cpp-python is installed in your virtual environment.
    echo ERROR: llama_cpp not found in %SITE_PACKAGES% >> copy_package_log.txt
    pause
    exit /b 1
)

:: Find the llama_cpp_python dist-info directory
set DIST_INFO=
for /d %%d in ("%SITE_PACKAGES%\llama_cpp_python-*.dist-info") do (
    set DIST_INFO=%%d
    echo Found llama_cpp_python dist-info: %%d >> copy_package_log.txt
)

if "%DIST_INFO%" == "" (
    echo WARNING: Could not find llama_cpp_python dist-info directory
    echo WARNING: Could not find llama_cpp_python dist-info directory >> copy_package_log.txt
) else (
    echo Found llama_cpp_python dist-info: %DIST_INFO%
)

:: Create backup of existing llama_cpp in package
if exist "dist\Ladbon AI Desktop\_internal\llama_cpp" (
    echo Creating backup of existing llama_cpp package...
    echo Creating backup of existing llama_cpp package... >> copy_package_log.txt
    if exist "dist\Ladbon AI Desktop\_internal\llama_cpp.bak" (
        rmdir /s /q "dist\Ladbon AI Desktop\_internal\llama_cpp.bak"
    )
    move "dist\Ladbon AI Desktop\_internal\llama_cpp" "dist\Ladbon AI Desktop\_internal\llama_cpp.bak"
)

:: Also backup any existing llama_cpp_python dist-info
if exist "dist\Ladbon AI Desktop\_internal\llama_cpp_python-*.dist-info" (
    echo Creating backup of existing llama_cpp_python dist-info...
    echo Creating backup of existing llama_cpp_python dist-info... >> copy_package_log.txt
    for /d %%d in ("dist\Ladbon AI Desktop\_internal\llama_cpp_python-*.dist-info") do (
        echo Backing up: %%d >> copy_package_log.txt
        if exist "%%d.bak" rmdir /s /q "%%d.bak"
        move "%%d" "%%d.bak"
    )
)

:: Copy entire llama_cpp package from site-packages to _internal
echo Copying llama_cpp from %SITE_PACKAGES% to package...
echo Copying llama_cpp from %SITE_PACKAGES% to package... >> copy_package_log.txt
xcopy /E /I /Y "%SITE_PACKAGES%\llama_cpp" "dist\Ladbon AI Desktop\_internal\llama_cpp"

:: Copy llama_cpp_python dist-info if found
if not "%DIST_INFO%" == "" (
    echo Copying %DIST_INFO% to package...
    echo Copying %DIST_INFO% to package... >> copy_package_log.txt
    
    :: Extract the base filename without path
    for /f "delims=\" %%i in ("%DIST_INFO%") do set DIST_INFO_NAME=%%i
    
    xcopy /E /I /Y "%DIST_INFO%" "dist\Ladbon AI Desktop\_internal\%DIST_INFO_NAME%"
)

:: Copy any CUDA DLLs that might be needed
echo Checking for CUDA DLLs in system...
echo Checking for CUDA DLLs in system... >> copy_package_log.txt

:: Create CUDA DLLs directory
if not exist "dist\Ladbon AI Desktop\_internal\cuda_dlls" (
    mkdir "dist\Ladbon AI Desktop\_internal\cuda_dlls"
)

:: Search for CUDA DLLs in common locations and copy them
set CUDA_PATHS=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin
set CUDA_DLLS=cudart64_12.dll cudart64_120.dll cudart64_121.dll cudart64_122.dll cudart64_123.dll cudart64_124.dll cudart64_125.dll cudart64_126.dll cudart64_127.dll cudart64_128.dll cudart64_129.dll cublas64_12.dll cublasLt64_12.dll

for %%p in (%CUDA_PATHS%) do (
    if exist "%%p" (
        echo Found CUDA path: %%p >> copy_package_log.txt
        for %%d in (%CUDA_DLLS%) do (
            if exist "%%p\%%d" (
                echo Found and copying CUDA DLL: %%p\%%d >> copy_package_log.txt
                copy /Y "%%p\%%d" "dist\Ladbon AI Desktop\_internal\cuda_dlls"
                copy /Y "%%p\%%d" "dist\Ladbon AI Desktop\_internal\llama_cpp\lib"
            )
        )
    )
)

:: List what we've copied
echo.
echo Files copied to _internal\llama_cpp:
dir "dist\Ladbon AI Desktop\_internal\llama_cpp" /s /b >> copy_package_log.txt

echo Files copied to _internal\cuda_dlls:
dir "dist\Ladbon AI Desktop\_internal\cuda_dlls" /s /b >> copy_package_log.txt

:: Create a special launcher batch file that ensures CUDA DLLs are in PATH
echo Creating enhanced launcher with CUDA paths...
echo @echo off > "dist\Launch with CUDA paths.bat"
echo echo Setting up environment for Ladbon AI Desktop... >> "dist\Launch with CUDA paths.bat"
echo. >> "dist\Launch with CUDA paths.bat"
echo :: Add CUDA DLLs directory to PATH >> "dist\Launch with CUDA paths.bat"
echo set "PATH=%%~dp0Ladbon AI Desktop\_internal\cuda_dlls;%%~dp0Ladbon AI Desktop\_internal\llama_cpp\lib;%%PATH%%" >> "dist\Launch with CUDA paths.bat"
echo. >> "dist\Launch with CUDA paths.bat"
echo echo Starting Ladbon AI Desktop with enhanced PATH... >> "dist\Launch with CUDA paths.bat"
echo echo PATH = %%PATH%% >> "dist\Launch with CUDA paths.bat"
echo. >> "dist\Launch with CUDA paths.bat"
echo start "" "%%~dp0Ladbon AI Desktop\Ladbon AI Desktop.exe" >> "dist\Launch with CUDA paths.bat"

echo.
echo =================================================== >> copy_package_log.txt
echo Copying complete! >> copy_package_log.txt
echo.
echo Copying complete!
echo 1. Now try running the diagnostic_script.py in the packaged app directory
echo 2. Then try running the app using "Launch with CUDA paths.bat" in the dist directory
echo.
echo Details have been logged to copy_package_log.txt
echo.
pause
