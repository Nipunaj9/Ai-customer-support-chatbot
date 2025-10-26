@echo off
echo Starting AI Customer Support Chatbot...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "chatbot_model.h5" (
    echo Model not found. Running setup first...
    echo.
    python setup.py
    if errorlevel 1 (
        echo Setup failed. Please check the error messages above.
        pause
        exit /b 1
    )
)

REM Start the Flask application
echo Starting the chatbot server...
echo.
echo The chatbot will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
