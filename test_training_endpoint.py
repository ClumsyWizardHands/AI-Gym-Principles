"""Test the training endpoint to diagnose CORS issues."""
import requests
import json

def test_training_start():
    """Test the training start endpoint."""
    url = "http://localhost:8000/api/training/start"
    
    # Test with minimal headers
    headers = {
        "X-API-Key": "test-key",
        "Content-Type": "application/json",
        "Origin": "http://localhost:5173"
    }
    
    # Sample training request
    payload = {
        "agent_id": "test-agent-123",
        "scenario_types": [],
        "num_scenarios": 10,
        "adaptive": True,
        "use_branching": False,
        "branching_types": ["trust_building", "resource_cascade"]
    }
    
    try:
        # First try OPTIONS request (CORS preflight)
        print("Testing OPTIONS request (CORS preflight)...")
        options_response = requests.options(url, headers=headers)
        print(f"OPTIONS Status: {options_response.status_code}")
        print(f"OPTIONS Headers: {dict(options_response.headers)}")
        print()
        
        # Then try POST request
        print("Testing POST request...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"POST Status: {response.status_code}")
        print(f"POST Headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"Response body: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_training_start()
