"""Register the Adaptive Bridge Builder using the JSON-RPC adapter."""

import requests
import json

# API endpoint
url = "http://localhost:8000/api/agents/register"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Agent data pointing to our adapter
agent_data = {
    "name": "Adaptive Bridge Builder (via Adapter)",
    "framework": "http",
    "config": {
        "endpoint_url": "http://localhost:8090/adapter",  # Point to adapter instead of agent
        "method": "POST",
        "timeout": 30
    },
    "description": "Principle-based decision maker with JSON-RPC adapter"
}

try:
    # Make the registration request
    response = requests.post(url, headers=headers, json=agent_data)
    
    # Check if successful
    if response.status_code == 201:
        result = response.json()
        print("✅ Agent registered successfully!")
        print(f"Agent ID: {result['agent_id']}")
        print(f"Name: {result['name']}")
        print(f"Framework: {result['framework']}")
        print(f"Status: {result['status']}")
        print(f"Registered at: {result['registered_at']}")
        
        # Save the agent ID for later use
        with open("adapter_agent_id.txt", "w") as f:
            f.write(result['agent_id'])
        print(f"\n💾 Agent ID saved to adapter_agent_id.txt")
        
    else:
        print(f"❌ Registration failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the AI Principles Gym API at http://localhost:8000")
    print("Please make sure the gym is running (run dev-start.bat)")
except Exception as e:
    print(f"❌ Error: {str(e)}")
