@echo off
REM Development Environment Startup Script for AI Principles Gym (Windows)
REM This script starts both the backend API and frontend development servers

echo Starting AI Principles Gym Development Environment...

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install backend dependencies if needed
echo Checking backend dependencies...
pip install -q -r requirements.txt

REM Create data directory if it doesn't exist
if not exist "data" mkdir data

REM Start backend server in a new window
echo Starting backend server on http://localhost:8000
start "AI Principles Gym - Backend" cmd /k "python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Check if backend is running
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% == 0 (
    echo Backend is running
) else (
    echo Backend health check failed, but continuing...
)

REM Start frontend server in a new window
echo Starting frontend server on http://localhost:5173
cd frontend
call npm install
start "AI Principles Gym - Frontend" cmd /k "npm run dev"
cd ..

REM Wait for frontend to start
echo Waiting for frontend to start...
timeout /t 5 /nobreak > nul

echo.
echo Development environment is ready!
echo Frontend: http://localhost:5173
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Close the terminal windows to stop the servers
pause
