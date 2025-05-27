# AI Principles Gym - Development Environment Setup Guide

This guide provides instructions for setting up and running the AI Principles Gym development environment with both frontend and backend properly connected.

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ and npm installed
- Git installed
- curl or similar tool for testing endpoints (optional)

## Quick Start

### Windows
```bash
# Clone the repository
git clone <repository-url>
cd ai-principles-gym

# Run the development environment
dev-start.bat
```

### macOS/Linux
```bash
# Clone the repository
git clone <repository-url>
cd ai-principles-gym

# Make the script executable
chmod +x dev-start.sh

# Run the development environment
./dev-start.sh
```

## Manual Setup

If you prefer to set up the environment manually or need more control:

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Create .env file (copy from example)
cp .env.example .env

# Start the backend server
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup (in a new terminal)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Environment Configuration

### Frontend Configuration (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENABLE_DEVTOOLS=true
```

### Backend Configuration (`.env`)

Create a `.env` file in the root directory with:

```env
# Environment
ENVIRONMENT=development

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
API_LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/principles.db

# CORS (already configured in config.py)
ENABLE_CORS=true

# Optional: API Key for authentication
# API_KEY=your-dev-api-key

# Optional: LLM Configuration
# ANALYSIS_LLM_PROVIDER=anthropic
# ANALYSIS_LLM_API_KEY=your-api-key
```

## Verifying the Setup

### 1. Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": 1234567890,
  "environment": "development",
  "version": "1.0.0",
  "components": {...}
}
```

### 2. Check API Documentation

Open in browser: http://localhost:8000/docs

### 3. Check Frontend

Open in browser: http://localhost:5173

You should see the AI Principles Gym interface.

### 4. Test API Connection from Frontend

The frontend will automatically proxy API requests through Vite's dev server. Test by:
1. Opening the browser console (F12)
2. Checking for any CORS errors
3. Verifying WebSocket connections in the Network tab

## Development Workflow

### Hot Reloading

- **Backend**: Automatically reloads on Python file changes (uvicorn --reload)
- **Frontend**: Automatically reloads on TypeScript/React file changes (Vite HMR)

### Making API Calls

Frontend API calls should use relative paths:
```typescript
// Good - will be proxied to backend
fetch('/api/agents')

// Bad - hardcoded URL
fetch('http://localhost:8000/api/agents')
```

### WebSocket Connections

WebSocket connections are automatically proxied:
```typescript
// Good - will be proxied to backend
new WebSocket('/ws/training/session-id')

// Bad - hardcoded URL
new WebSocket('ws://localhost:8000/ws/training/session-id')
```

## Troubleshooting

### Port Already in Use

If you get "Address already in use" errors:

```bash
# Find processes using the ports
# Windows
netstat -ano | findstr :8000
netstat -ano | findstr :5173

# macOS/Linux
lsof -i :8000
lsof -i :5173

# Kill the processes
# Windows
taskkill /PID <process-id> /F

# macOS/Linux
kill -9 <process-id>
```

### CORS Errors

If you see CORS errors in the browser console:

1. Verify backend is running on port 8000
2. Check that ENABLE_CORS=true in .env
3. Ensure frontend is accessing API through proxy (/api/...)
4. Clear browser cache and cookies

### Module Import Errors

If you get Python import errors:

```bash
# Ensure virtual environment is activated
# Install in development mode
pip install -e .

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Build Errors

If frontend fails to build:

```bash
cd frontend
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Database Connection Issues

If you get database errors:

```bash
# Create data directory
mkdir -p data

# Check database file permissions
ls -la data/

# Remove corrupted database (development only!)
rm data/principles.db
```

## Development Tips

### 1. API Testing with curl

```bash
# Generate API key
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{"user_id": "dev-user", "usage_limit": 1000}'

# Use the API key in subsequent requests
curl http://localhost:8000/api/agents \
  -H "X-API-Key: your-api-key"
```

### 2. Database Inspection

```bash
# Install SQLite browser (optional)
# Windows: Download from https://sqlitebrowser.org/
# macOS: brew install --cask db-browser-for-sqlite
# Linux: sudo apt-get install sqlitebrowser

# Open database
sqlite3 data/principles.db
.tables
.schema
SELECT * FROM agents LIMIT 5;
.quit
```

### 3. Log Monitoring

Backend logs appear in the terminal. For structured log analysis:

```bash
# Filter logs by level
python -m uvicorn src.api.app:app 2>&1 | grep "ERROR"

# Save logs to file
python -m uvicorn src.api.app:app 2>&1 | tee backend.log
```

### 4. Performance Profiling

Enable profiling in development:

```env
# In .env
PROFILE_PERFORMANCE=true
DEBUG_MODE=true
```

## Next Steps

1. Review the [API Documentation](http://localhost:8000/docs)
2. Explore the [Frontend Components](./frontend/src/components/)
3. Run the test suite: `pytest`
4. Check the [Production Deployment Guide](./deployment/production-guide.md)

## Support

If you encounter issues not covered here:

1. Check existing [GitHub Issues]
2. Review logs for error messages
3. Ensure all prerequisites are correctly installed
4. Try the setup in a fresh virtual environment
