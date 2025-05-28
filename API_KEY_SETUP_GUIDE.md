# API Key Setup Guide for AI Principles Gym

## Overview
The AI Principles Gym uses API keys for authentication. You have two options:
1. Generate a new API key through the UI
2. Use an existing API key

## Quick Start

### Step 1: Initialize the Database
First, make sure the database is initialized:

```bash
cd ai-principles-gym
python -m alembic upgrade head
```

### Step 2: Restart the Backend
Stop and restart the backend to ensure proper initialization:

```bash
# Kill the existing backend process (Ctrl+C in the terminal running it)
# Then restart:
python -m src.api.app
```

### Step 3: Access the Platform

1. Open your browser to http://localhost:5174
2. You'll see the login page with two options:
   - **Generate New API Key**: Click this to create a new key via the API
   - **Use Existing API Key**: Click this if you already have a key

### Step 4: Using Your API Key

Once you have an API key:
1. The key will be automatically stored in your browser
2. All API requests will include this key in the `X-API-Key` header
3. You can view/copy your key in Settings

### Step 5: Register an HTTP Endpoint Agent

1. Navigate to the Agents page
2. Click "Register Agent"
3. Select "HTTP Endpoint" from the Framework dropdown
4. Configure your endpoint:
   - **Name**: Your agent's name
   - **Endpoint URL**: Where your AI agent is hosted (e.g., `http://localhost:8001/chat`)
   - **Method**: POST, GET, or PUT
   - **Headers**: Add any required headers (e.g., Authorization)
   - **Timeout**: Request timeout in seconds
   - **Request/Response Format**: JSON, form data, or plain text

## Testing with a Local AI Agent

### Example: Simple HTTP Echo Agent
Create a simple test agent to verify the connection:

```python
# test_agent.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    # Echo back the scenario with a simple decision
    return jsonify({
        "decision": "cooperate",
        "reasoning": f"Received scenario: {data.get('scenario', 'unknown')}"
    })

if __name__ == '__main__':
    app.run(port=8001)
```

Run it: `python test_agent.py`

Then register it in the UI with:
- Endpoint URL: `http://localhost:8001/chat`
- Method: POST
- Request Format: JSON
- Response Format: JSON

## Troubleshooting

### "Connection Error" when registering agents
1. Make sure the database is initialized (Step 1)
2. Check that the backend is running on http://localhost:8000
3. Check browser console for specific errors

### "Invalid API Key" errors
1. Make sure you're using a valid API key
2. Check that the key is being sent in requests (browser DevTools > Network tab)
3. Try generating a new key

### Backend returns 500 errors
1. The database likely needs initialization
2. Run: `python -m alembic upgrade head`
3. Restart the backend

## API Key Management

### Via UI
- **View Key**: Settings > Authentication
- **Copy Key**: Click "Copy API Key" button
- **Logout**: Clears the key from browser storage

### Via API
```bash
# Generate a new API key
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{"usage_limit": null, "expires_in_days": null}'

# Use the API key in requests
curl -X GET http://localhost:8000/api/agents \
  -H "X-API-Key: sk-your-api-key-here"
```

## Security Notes

1. **Never commit API keys** to version control
2. **Rotate keys regularly** in production
3. **Use environment variables** for keys in your agents
4. **Set usage limits** for keys when appropriate

## Next Steps

1. Register your AI agents (supports OpenAI, Anthropic, LangChain, Custom, and HTTP endpoints)
2. Start training sessions to test behavioral principles
3. Monitor real-time progress via WebSocket connections
4. Analyze principle emergence in the Reports section
