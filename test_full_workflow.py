"""Test the full workflow: register agent, then start training."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_full_workflow():
    """Test registering an agent and starting training."""
    
    # Headers for all requests
    headers = {
        "X-API-Key": "test-key-123",
        "Content-Type": "application/json"
    }
    
    print("=== Testing AI Principles Gym Workflow ===\n")
    
    # Step 1: Register an agent
    print("1. Registering an agent...")
    agent_data = {
        "name": "Test Agent",
        "framework": "openai",
        "config": {
            "model": "gpt-3.5-turbo",
            "api_key": "your-openai-key-here"  # Replace with actual key
        },
        "description": "Test agent for demo"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/agents/register",
        json=agent_data,
        headers=headers
    )
    
    if response.status_code == 201:
        agent_info = response.json()
        agent_id = agent_info["agent_id"]
        print(f"✅ Agent registered successfully!")
        print(f"   Agent ID: {agent_id}")
        print(f"   Name: {agent_info['name']}")
        print(f"   Framework: {agent_info['framework']}")
    else:
        print(f"❌ Failed to register agent: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n2. Starting training session...")
    
    # Step 2: Start training
    training_data = {
        "agent_id": agent_id,
        "scenario_types": [],  # Empty for all types
        "num_scenarios": 5,    # Small number for testing
        "adaptive": True,
        "use_branching": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/training/start",
        json=training_data,
        headers=headers
    )
    
    if response.status_code == 202:
        training_info = response.json()
        session_id = training_info["session_id"]
        print(f"✅ Training started successfully!")
        print(f"   Session ID: {session_id}")
        print(f"   Status: {training_info['status']}")
        print(f"   Estimated duration: {training_info['estimated_duration_seconds']}s")
    else:
        print(f"❌ Failed to start training: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n3. Checking training status...")
    
    # Step 3: Check status
    time.sleep(2)  # Wait a bit
    
    response = requests.get(
        f"{BASE_URL}/api/training/status/{session_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        status_info = response.json()
        print(f"✅ Training status retrieved!")
        print(f"   Status: {status_info['status']}")
        print(f"   Progress: {status_info['progress']*100:.1f}%")
        print(f"   Scenarios completed: {status_info['scenarios_completed']}/{status_info['scenarios_total']}")
    else:
        print(f"❌ Failed to get status: {response.status_code}")
        print(f"   Response: {response.text}")
    
    print("\n=== Workflow test complete! ===")
    print("\nNext steps:")
    print("1. Replace 'your-openai-key-here' with your actual OpenAI API key")
    print("2. Monitor the training progress using the session ID")
    print("3. Once complete, get the report using /api/reports/{session_id}")

if __name__ == "__main__":
    test_full_workflow()
