"""Test the JSON-RPC adapter directly."""

import requests
import json

# Test if the adapter is accessible
adapter_url = "http://localhost:8090/adapter"

# Create a test scenario request (same format the gym would send)
test_data = {
    "scenario": {
        "execution_id": "test-adapter-123",
        "description": "Test scenario for adapter",
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
        "request_id": "test-adapter-123"
    }
}

try:
    print("🔍 Testing JSON-RPC adapter at", adapter_url)
    response = requests.post(
        adapter_url,
        json=test_data,
        headers={"Content-Type": "application/json"},
        timeout=5
    )
    
    if response.status_code == 200:
        print("✅ Adapter is responding!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Adapter returned status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the adapter at", adapter_url)
    print("The adapter may not be running. Check the window where you started json_rpc_adapter.py")
except requests.exceptions.Timeout:
    print("❌ Adapter did not respond within 5 seconds")
except Exception as e:
    print(f"❌ Error: {str(e)}")
