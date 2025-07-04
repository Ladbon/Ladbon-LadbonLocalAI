@echo off
echo ===================================
echo = Git LFS Setup Helper           =
echo ===================================
echo.
echo This script will help you set up Git LFS (Large File Storage)
echo for handling large files in your repository.
echo.
echo Prerequisites:
echo - Git LFS must be installed: https://git-lfs.github.com/
echo.
if /i "%SKIP_PAUSE%" neq "1" pause

echo.
echo Checking if Git LFS is installed...
git lfs version
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Git LFS is not installed.
    echo Please download and install from: https://git-lfs.github.com/
    echo Then run this script again.
    if /i "%SKIP_PAUSE%" neq "1" pause
    exit /b 1
)

echo.
echo Step 1: Initializing Git LFS...
git lfs install

echo.
echo Step 2: Setting up tracking for large file types...
git lfs track "installer/*.bin"
git lfs track "installer/*.exe"
git lfs track "installer/*.msi"
git lfs track "dist/*.exe"
git lfs track "dist/*.msi"
git lfs track "models/*.gguf"

echo.
echo Step 3: Adding .gitattributes file...
git add .gitattributes

echo.
echo ===================================
echo = NEXT STEPS                     =
echo ===================================
echo.
echo 1. Commit the .gitattributes file:
echo    git commit -m "Add Git LFS tracking for large files"
echo.
echo 2. Add and commit your large files normally:
echo    git add installer/*.bin
echo    git commit -m "Add installer binaries"
echo.
echo 3. Push to GitHub:
echo    git push origin main
echo.
echo Note: You may need a paid GitHub plan or GitHub Enterprise
echo       for storing very large files with Git LFS.
echo.
if /i "%SKIP_PAUSE%" neq "1" pause
