@echo off
echo ========================================
echo Connecting to GitHub: mndolo2x/BlasterOPT
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed on your system.
    echo Please install Git from https://git-scm.com/download/win
    echo.
    echo After installing Git:
    echo 1. Restart this command prompt
    echo 2. Run this script again
    echo.
    pause
    exit /b 1
)

echo Git is installed.
echo.

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    echo Git repository initialized.
) else (
    echo Git repository already exists.
)

echo.
echo Adding all files to git...
git add .

echo.
echo Committing changes...
git commit -m "Initial commit: BlastOpt Botswana platform with all modules - Import errors fixed and deployment ready"

echo.
echo Setting remote repository to mndolo2x/BlasterOPT...
git remote add origin https://github.com/mndolo2x/BlasterOPT.git
git remote set-url origin https://github.com/mndolo2x/BlasterOPT.git

echo.
echo Setting main branch...
git branch -M main

echo.
echo ========================================
echo PUSHING TO GITHUB
echo ========================================
echo.
echo Repository: https://github.com/mndolo2x/BlasterOPT
echo.
echo You may be asked for your GitHub credentials.
echo If prompted, use your GitHub username and personal access token.
echo Create a token at: https://github.com/settings/tokens
echo.
pause

echo.
echo Pushing to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo PUSH FAILED
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Authentication failed - Use personal access token instead of password
    echo 2. Repository not found - Verify the repository exists at https://github.com/mndolo2x/BlasterOPT
    echo 3. Permission denied - Ensure you have push access to the repository
    echo.
    echo Try running: git push -u origin main
    echo.
) else (
    echo.
    echo ========================================
    echo SUCCESS!
    echo ========================================
    echo.
    echo Your code has been pushed to: https://github.com/mndolo2x/BlasterOPT
    echo.
    echo Next step: Deploy to Streamlit Cloud
    echo 1. Go to https://share.streamlit.io
    echo 2. Click "New app"
    echo 3. Select mndolo2x/BlasterOPT repository
    echo 4. Main file: app.py
    echo 5. Click "Deploy"
    echo.
)

pause
