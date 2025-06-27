@echo off
echo Running test of Ladbon AI Desktop...
cd "%~dp0\dist\Ladbon AI Desktop"
start "" "Ladbon AI Desktop.exe"
echo Application launched!
echo Waiting for 5 seconds to check for log files...
timeout /t 5
echo Checking for log files...
dir logs\gui_bootstrap_*.log
echo Done!
