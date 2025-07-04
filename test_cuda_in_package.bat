@echo off
echo ============================================================
echo TEST CUDA IN PACKAGED APPLICATION
echo ============================================================
echo.

REM Detect application directory
set "APP_DIR=%~dp0dist\Ladbon AI Desktop"
if not exist "%APP_DIR%" (
    echo ERROR: Cannot find application directory at %APP_DIR%
    echo Make sure you've built the application first.
    goto :error
)

echo Found application directory at: %APP_DIR%

REM Copy the test script to the app directory
echo Copying test script to app directory...
copy "%~dp0test_cuda_in_packaged_app.py" "%APP_DIR%\test_cuda.py" /Y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy test script.
    goto :error
)

REM Copy the init_cuda.py if it exists
if exist "%~dp0init_cuda.py" (
    echo Copying init_cuda.py to app directory...
    copy "%~dp0init_cuda.py" "%APP_DIR%\_internal\init_cuda.py" /Y
    if %errorlevel% neq 0 (
        echo WARNING: Failed to copy init_cuda.py. The test may still work.
    )
)

REM Create a launcher to run the test
echo Creating test launcher...
set "TEST_LAUNCHER=%APP_DIR%\Run CUDA Test.bat"

echo @echo off > "%TEST_LAUNCHER%"
echo echo ====================================================== >> "%TEST_LAUNCHER%"
echo echo           Testing CUDA in Packaged Application          >> "%TEST_LAUNCHER%"
echo echo ====================================================== >> "%TEST_LAUNCHER%"
echo echo. >> "%TEST_LAUNCHER%"
echo set "SCRIPT_DIR=%%~dp0" >> "%TEST_LAUNCHER%"
echo cd /d "%%SCRIPT_DIR%%" >> "%TEST_LAUNCHER%"
echo "%%SCRIPT_DIR%%\Python\python.exe" "%%SCRIPT_DIR%%\test_cuda.py" >> "%TEST_LAUNCHER%"
echo echo. >> "%TEST_LAUNCHER%"
echo echo Test complete. Check cuda_test.log for results. >> "%TEST_LAUNCHER%"
echo pause >> "%TEST_LAUNCHER%"

echo Test launcher created at: %TEST_LAUNCHER%

echo.
echo ============================================================
echo TEST SETUP COMPLETE
echo ============================================================
echo.
echo To run the test:
echo 1. Navigate to %APP_DIR%
echo 2. Run "Run CUDA Test.bat"
echo 3. Check the results in cuda_test.log
echo.
echo This test will check if CUDA is properly set up in the packaged app
echo and report any issues found.
echo.
goto :eof

:error
echo.
echo ERROR: Setup failed. Please fix the errors and try again.
exit /b 1
