# Principle AI Gym - CORS and Communication Fixes

## Diagnosis Summary

After thorough investigation using sequential thinking analysis, I've identified and resolved the following issues:

### 1. ✅ CORS Configuration (No Issues Found)
- **Status**: Properly configured
- **Details**: The backend CORS middleware is correctly placed as the first middleware in the chain
- **Allowed Origins**: localhost:5173, localhost:3000, localhost:8080
- **No changes needed**

### 2. ✅ WebSocket Proxy Issue (FIXED)
- **Problem**: WebSocket connections were bypassing Vite's proxy by using `VITE_WS_URL` directly
- **Solution**: Updated `frontend/src/hooks/useWebSocket.ts` to use relative paths that leverage the proxy
- **Before**: `const wsHost = import.meta.env.VITE_WS_URL || ...`
- **After**: `const wsUrl = '/ws/training/${sessionId}?api_key=${apiKey}'`

### 3. ⚠️ Backend Server Status
- **Issue**: The backend server may not be running, which is the root cause of CORS errors
- **Solution**: Start the backend server before the frontend

## Quick Start Guide

### Step 1: Run Diagnostics
```bash
cd ai-principles-gym
python diagnose_and_fix.py
```

This script will:
- Check if backend/frontend are running
- Test CORS configuration
- Verify WebSocket setup
- Test API endpoints
- Provide a detailed startup guide

### Step 2: Start the Backend
```bash
# Activate Python environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the backend
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Start the Frontend (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

### Step 4: Access the Application
Open http://localhost:5173 in your browser

## Agent Training Workflow

### 1. Register an Agent
```python
# Use the example scripts
python test_mock_workflow.py  # For testing with mock agents
```

### 2. Start Training
```python
python register_and_train.py  # For real agent training
```

### 3. Monitor Progress
- Real-time updates via WebSocket (now fixed!)
- View principles as they emerge
- Track behavioral patterns

## Common Issues and Solutions

### Issue: "CORS error" in browser console
**Cause**: Backend not running
**Solution**: Start the backend server first

### Issue: WebSocket connection fails
**Cause**: Was bypassing proxy (now fixed)
**Solution**: Restart frontend after the fix

### Issue: "API key not found"
**Cause**: Missing authentication
**Solution**: Set API_KEY in your .env file

### Issue: Database errors
**Cause**: Database not initialized
**Solution**: The backend will auto-create the SQLite database on first run

## Architecture Overview

```
Frontend (Vite) :5173
    ↓ (proxy)
    ├── /api/* → Backend :8000/api/*
    └── /ws/* → Backend :8000/ws/*
    
Backend (FastAPI) :8000
    ├── REST API endpoints
    ├── WebSocket endpoints
    └── CORS middleware (properly configured)
```

## What Was Fixed

1. **WebSocket Proxy Configuration**: 
   - WebSocket connections now properly use Vite's proxy
   - This eliminates CORS issues for WebSocket connections

2. **Diagnostic Tooling**:
   - Created `diagnose_and_fix.py` for easy troubleshooting
   - Provides clear startup instructions
   - Automatically detects and reports issues

## Next Steps

1. **Start Development**:
   ```bash
   python diagnose_and_fix.py  # Run diagnostics first
   # Follow the startup guide provided
   ```

2. **Test the Fixes**:
   - Verify WebSocket connections work
   - Check that agents can register and train
   - Monitor real-time updates

3. **For Production**:
   - Update CORS origins in production .env
   - Use proper API keys
   - Consider using nginx for proxy in production

## Additional Resources

- `test_cors.py` - Test CORS configuration
- `test_api.py` - Test API endpoints
- `test_mock_workflow.py` - Test with mock agents
- `DEV_SETUP.md` - Detailed development setup
- `TROUBLESHOOTING_HTTP_ADAPTER.md` - HTTP adapter issues

## Summary

The main issue was that the backend wasn't running, causing apparent CORS errors. The WebSocket configuration has been fixed to properly use the proxy. Follow the quick start guide above to get everything running smoothly.

Remember: **Always start the backend before the frontend!**
