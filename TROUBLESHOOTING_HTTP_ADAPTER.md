# Troubleshooting HTTP Adapter Issues

## Quick Diagnostic

First, run the diagnostic script:
```bash
cd ai-principles-gym
python check_setup.py
```

This will check all components and tell you what's working and what needs fixing.

## Common Issues and Solutions

### 1. Not Seeing "HTTP Endpoint" in Register Agent

**Symptoms:**
- The dropdown doesn't show "HTTP Endpoint" option
- Only see OpenAI, Anthropic, LangChain, Custom

**Solutions:**

**A. Rebuild the frontend:**
```bash
cd ai-principles-gym/frontend
npm install
npm run build
npm run dev
```

**B. Clear browser cache:**
- Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Or open Developer Tools (F12) → Network tab → Check "Disable cache"

**C. Check if files were updated:**
```bash
# Check if HTTP is in the endpoints file
grep -i "http" frontend/src/api/endpoints.ts

# Should show: { value: 'http', label: 'HTTP Endpoint' }
```

### 2. Network Errors

**Symptoms:**
- "Failed to fetch" errors
- "Network error" toasts
- Can't register agents or see agent list

**Solutions:**

**A. Ensure both servers are running:**

Terminal 1 - Backend:
```bash
cd ai-principles-gym
python -m src.api.app
```
Should see: "Uvicorn running on http://0.0.0.0:8000"

Terminal 2 - Frontend:
```bash
cd ai-principles-gym/frontend
npm run dev
```
Should see: "Local: http://localhost:5173"

**B. Check frontend environment configuration:**
```bash
cat frontend/.env
```

Should contain:
```
VITE_API_URL=http://localhost:8000
```

If not, create it:
```bash
echo 'VITE_API_URL=http://localhost:8000' > frontend/.env
```

**C. Test backend directly:**
```bash
curl http://localhost:8000/api/health
```

Should return: `{"status":"healthy","version":"..."}`

### 3. TypeScript Compilation Errors

**Symptoms:**
- Red underlines in VS Code
- Build errors mentioning 'http' type

**Solution:**
```bash
cd ai-principles-gym/frontend
npm run type-check
```

If errors, ensure all files are saved and rebuild.

## Complete Reset Procedure

If nothing else works, try a complete reset:

```bash
# 1. Stop all running servers (Ctrl+C in both terminals)

# 2. Clean and reinstall
cd ai-principles-gym
rm -rf frontend/node_modules frontend/dist
cd frontend
npm install

# 3. Ensure environment files exist
cd ..
cp .env.example .env  # If .env doesn't exist
echo 'VITE_API_URL=http://localhost:8000' > frontend/.env

# 4. Start backend
python -m src.api.app

# 5. In new terminal, start frontend
cd frontend
npm run dev

# 6. Open browser to http://localhost:5173
```

## Verify HTTP Adapter is Working

Once everything is running:

1. Go to http://localhost:5173
2. Navigate to "Agents" page
3. Click "Register Agent"
4. In the Framework dropdown, you should see:
   - OpenAI
   - Anthropic
   - LangChain
   - Custom
   - **HTTP Endpoint** ← This should appear

5. Select "HTTP Endpoint" and you should see:
   - Endpoint URL field
   - Method dropdown (POST/GET/PUT)
   - Timeout field
   - Request/Response format dropdowns
   - Authorization token field
   - Custom headers textarea

## Testing with a Mock HTTP Agent

To test the HTTP adapter, you can create a simple mock agent:

```python
# Save as mock_agent.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class AgentRequest(BaseModel):
    scenario: dict
    history: list
    metadata: dict

@app.post("/chat")
async def chat(request: AgentRequest):
    # Simple agent that always chooses the first option
    choice = request.scenario["choice_options"][0]["id"]
    return {
        "action": choice,
        "reasoning": "I always choose the first option",
        "confidence": 0.9
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

Run it: `python mock_agent.py`

Then register it in the Gym:
- Name: Test HTTP Agent
- Framework: HTTP Endpoint
- Endpoint URL: http://localhost:8001/chat
- Method: POST
- Request Format: JSON
- Response Format: JSON

## Still Having Issues?

If you're still experiencing problems:

1. Check the browser console (F12) for errors
2. Check the terminal running the backend for error messages
3. Check the terminal running the frontend for compilation errors
4. Ensure you're using a modern browser (Chrome, Firefox, Edge)
5. Make sure no firewall is blocking localhost connections

The HTTP adapter is fully implemented and should work once the frontend is properly rebuilt and both servers are running correctly.
