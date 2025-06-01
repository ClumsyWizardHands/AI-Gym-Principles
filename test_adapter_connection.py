"""Test the HTTP adapter connection to the Adaptive Bridge Builder."""

import requests
import json

# Test if the agent is accessible
agent_url = "http://localhost:8080/process"

# Create a test scenario request
test_data = {
    "scenario": {
        "execution_id": "test-123",
        "description": "Test scenario",
        "actors": ["tester"],
        "resources": ["test_resource"],
        "constraints": [],
        "choice_options": [
            {
                "id": "option_a",
                "name": "Option A",
                "description": "Test option A"
            },
            {
                "id": "option_b", 
                "name": "Option B",
                "description": "Test option B"
            }
        ],
        "time_limit": 30,
        "archetype": "RESOURCE_ALLOCATION",
        "stress_level": 0.5
    },
    "history": [],
    "metadata": {
        "framework": "principles_gym",
        "version": "1.0.0",
        "request_id": "test-123"
    }
}

try:
    print("🔍 Testing connection to Adaptive Bridge Builder at", agent_url)
    response = requests.post(
        agent_url,
        json=test_data,
        headers={"Content-Type": "application/json"},
        timeout=5
    )
    
    if response.status_code == 200:
        print("✅ Agent is responding!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Agent returned status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to your agent at", agent_url)
    print("Make sure your Adaptive Bridge Builder is running on port 8080")
except requests.exceptions.Timeout:
    print("❌ Agent did not respond within 5 seconds")
except Exception as e:
    print(f"❌ Error: {str(e)}")
