"""Register the Adaptive Bridge Builder agent with the AI Principles Gym."""

import requests
import json

# API endpoint
url = "http://localhost:8000/api/agents/register"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Agent data
agent_data = {
    "name": "Adaptive Bridge Builder",
    "framework": "http",
    "config": {
        "endpoint_url": "http://localhost:8080/process",
        "method": "POST",
        "timeout": 30
    },
    "description": "Principle-based decision maker with emotional intelligence"
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
        with open("ai-principles-gym/adaptive_bridge_builder_agent_id.txt", "w") as f:
            f.write(result['agent_id'])
        print(f"\n💾 Agent ID saved to adaptive_bridge_builder_agent_id.txt")
        
    else:
        print(f"❌ Registration failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the AI Principles Gym API at http://localhost:8000")
    print("Please make sure the gym is running (run dev-start.bat)")
except Exception as e:
    print(f"❌ Error: {str(e)}")
