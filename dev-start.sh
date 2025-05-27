#!/bin/bash

# Development Environment Startup Script for AI Principles Gym
# This script starts both the backend API and frontend development servers

echo "Starting AI Principles Gym Development Environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    # Kill all child processes
    pkill -P $$
    exit 0
}

# Set up trap to call cleanup on CTRL+C
trap cleanup INT TERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python -m venv venv
fi

# Activate virtual environment (Windows-compatible)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install backend dependencies if needed
echo -e "${BLUE}Checking backend dependencies...${NC}"
pip install -q -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p data

# Start backend server
echo -e "${GREEN}Starting backend server on http://localhost:8000${NC}"
cd "$(dirname "$0")"
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}Waiting for backend to start...${NC}"
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ Backend is running${NC}"
else
    echo -e "${YELLOW}⚠ Backend health check failed, but continuing...${NC}"
fi

# Start frontend server
echo -e "${GREEN}Starting frontend server on http://localhost:5173${NC}"
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo -e "${BLUE}Waiting for frontend to start...${NC}"
sleep 5

echo -e "${GREEN}✓ Development environment is ready!${NC}"
echo -e "${GREEN}Frontend: http://localhost:5173${NC}"
echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Press CTRL+C to stop all servers${NC}"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
