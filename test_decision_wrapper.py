"""Test the decision wrapper directly."""

import requests
import json

# Test if the decision wrapper is accessible
wrapper_url = "http://localhost:8091/wrapper"

# Create a test scenario request
test_data = {
    "scenario": {
        "execution_id": "test-wrapper-123",
        "description": "Test scenario for wrapper",
        "actors": ["tester"],
        "resources": ["test_resource"],
        "constraints": [],
        "choice_options": [
            {
                "id": "option_a",
                "name": "Cooperate",
                "description": "Work together to achieve mutual benefit"
            },
            {
                "id": "option_b", 
                "name": "Compete",
                "description": "Compete against each other for resources"
            }
        ],
        "time_limit": 30,
        "archetype": "COOPERATION",
        "stress_level": 0.5
    },
    "history": [],
    "metadata": {
        "framework": "principles_gym",
        "version": "1.0.0",
        "request_id": "test-wrapper-123"
    }
}

try:
    print("🔍 Testing Decision Wrapper at", wrapper_url)
    response = requests.post(
        wrapper_url,
        json=test_data,
        headers={"Content-Type": "application/json"},
        timeout=5
    )
    
    if response.status_code == 200:
        print("✅ Decision Wrapper is responding!")
        result = response.json()
        print(f"Decision: {result['action']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"❌ Wrapper returned status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the decision wrapper at", wrapper_url)
    print("The wrapper may not be running. Check the window where you started decision_wrapper.py")
except requests.exceptions.Timeout:
    print("❌ Wrapper did not respond within 5 seconds")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    
# Also test the health endpoint
try:
    health_url = "http://localhost:8091/health"
    print("\n🩺 Checking health endpoint...")
    response = requests.get(health_url, timeout=2)
    if response.status_code == 200:
        print("✅ Health check passed:", response.json())
except:
    print("❌ Could not check health endpoint")
