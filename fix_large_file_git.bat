@echo off
echo ===================================
echo = Git Large File Removal Helper   =
echo ===================================
echo.
echo This script will help you remove large files from your Git repository.
echo Usage: %0 <file-to-remove>
echo.
echo WARNING: This will rewrite your Git history!
echo Make sure you have pushed all other changes first.
echo.
if "%~1"=="" (
	echo ERROR: Please specify the file to remove as a parameter.
	echo Example: %0 installer\Ladbon_AI_Desktop_Setup-1.bin
	pause
	exit /b 1
)
set FILE_TO_REMOVE=%~1
pause

echo.
echo Step 1: Removing large file from Git history...
git filter-repo --path "%FILE_TO_REMOVE%" --invert-paths

echo.
echo Step 2: Cleaning up Git repository...
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo.
echo Step 3: Updating .gitignore to exclude the removed file...
findstr /x /c:"%FILE_TO_REMOVE%" .gitignore >nul 2>&1
if errorlevel 1 (
	echo %FILE_TO_REMOVE%>>.gitignore
	echo Added %FILE_TO_REMOVE% to .gitignore.
) else (
	echo %FILE_TO_REMOVE% is already in .gitignore.
)

echo.
echo ===================================
echo = NEXT STEPS                     =
echo ===================================
echo.
echo 1. Force-push to your repository with:
echo    git push origin main --force
echo.
echo 2. Ask your collaborators to re-clone the repository.
echo.
echo 3. For future large files, consider using Git LFS:
echo    https://git-lfs.github.com/
echo.
pause
