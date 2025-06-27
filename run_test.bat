@echo off
echo Running packaged application and capturing output...
echo.

set LOG_FILE=app_test_log.txt

echo Test started: %date% %time% > %LOG_FILE%
echo Executable: "%~dp0dist\Ladbon AI Desktop.exe" >> %LOG_FILE%
echo. >> %LOG_FILE%

echo Attempting to run the application... >> %LOG_FILE%
"%~dp0dist\Ladbon AI Desktop.exe" 2>> %LOG_FILE%

echo. >> %LOG_FILE%
echo Test completed: %date% %time% >> %LOG_FILE%

echo Log file created at: %~dp0%LOG_FILE%
echo.
echo Check the log file for any error messages.
pause
