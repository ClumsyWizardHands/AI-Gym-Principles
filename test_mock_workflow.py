"""Test workflow with mock adapter for testing without API keys."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_mock_workflow():
    """Test registering a mock agent and starting training."""
    
    # Headers for all requests
    headers = {
        "X-API-Key": "test-key-123",
        "Content-Type": "application/json"
    }
    
    print("=== Testing AI Principles Gym with Mock Adapter ===\n")
    
    # Step 1: Register a mock agent
    print("1. Registering a mock agent...")
    agent_data = {
        "name": "Mock Test Agent",
        "framework": "mock",
        "config": {
            "response_delay": 0.1  # Fast responses for testing
        },
        "description": "Mock agent for testing without API keys"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/agents/register",
        json=agent_data,
        headers=headers
    )
    
    if response.status_code == 201:
        agent_info = response.json()
        agent_id = agent_info["agent_id"]
        print(f"✅ Mock agent registered successfully!")
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
        "scenario_types": ["trust_game", "resource_sharing"],
        "num_scenarios": 3,    # Small number for quick testing
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
    
    print("\n3. Monitoring training progress...")
    
    # Step 3: Monitor progress
    for i in range(10):  # Check status 10 times
        time.sleep(2)  # Wait 2 seconds between checks
        
        response = requests.get(
            f"{BASE_URL}/api/training/status/{session_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            status_info = response.json()
            print(f"\r   Status: {status_info['status']} | Progress: {status_info['progress']*100:.1f}% | Scenarios: {status_info['scenarios_completed']}/{status_info['scenarios_total']}", end="")
            
            if status_info['status'] == 'completed':
                print("\n✅ Training completed!")
                break
            elif status_info['status'] == 'failed':
                print(f"\n❌ Training failed: {status_info.get('error_message', 'Unknown error')}")
                return
        else:
            print(f"\n❌ Failed to get status: {response.status_code}")
            print(f"   Response: {response.text}")
            return
    
    print("\n\n4. Getting training report...")
    
    # Step 4: Get report
    response = requests.get(
        f"{BASE_URL}/api/reports/{session_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        report = response.json()
        print(f"✅ Report retrieved successfully!")
        print(f"\n=== Training Report ===")
        print(f"Duration: {report['duration_seconds']:.1f} seconds")
        print(f"Scenarios Completed: {report['scenarios_completed']}")
        print(f"Principles Discovered: {len(report['principles_discovered'])}")
        print(f"Behavioral Entropy: {report['behavioral_entropy']:.2f}")
        print(f"\nSummary: {report['summary']}")
        
        if report['principles_discovered']:
            print(f"\nDiscovered Principles:")
            for p in report['principles_discovered']:
                print(f"- {p['name']}: {p['description']} (strength: {p['strength']:.2f})")
    else:
        print(f"❌ Failed to get report: {response.status_code}")
        print(f"   Response: {response.text}")
    
    print("\n=== Mock workflow test complete! ===")
    print("\nThis demonstrates that the system works without real API keys.")
    print("The mock adapter simulates LLM responses for testing purposes.")

if __name__ == "__main__":
    test_mock_workflow()
