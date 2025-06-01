"""Register the decision wrapper for Adaptive Bridge Builder."""

import requests
import json

# API endpoint
url = "http://localhost:8000/api/agents/register"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Agent data pointing to our decision wrapper
agent_data = {
    "name": "Adaptive Bridge Builder (Decision Wrapper)",
    "framework": "http",
    "config": {
        "endpoint_url": "http://localhost:8091/wrapper",  # Point to decision wrapper
        "method": "POST",
        "timeout": 30
    },
    "description": "Bridge Builder with added decision-making capabilities for training scenarios"
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
        with open("decision_wrapper_agent_id.txt", "w") as f:
            f.write(result['agent_id'])
        print(f"\n💾 Agent ID saved to decision_wrapper_agent_id.txt")
        
    else:
        print(f"❌ Registration failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the AI Principles Gym API at http://localhost:8000")
    print("Please make sure the gym is running")
except Exception as e:
    print(f"❌ Error: {str(e)}")
