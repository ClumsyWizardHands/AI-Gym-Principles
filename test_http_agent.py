"""Test script for HTTP agent at localhost:8080/process"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_http_agent():
    """Test registering and training an HTTP agent."""
    
    # Headers for all requests
    headers = {
        "X-API-Key": "test-key-123",
        "Content-Type": "application/json"
    }
    
    print("=== Testing HTTP Agent with AI Principles Gym ===\n")
    
    # Step 1: Test connection to HTTP agent
    print("1. Testing connection to your agent at http://localhost:8080/process...")
    try:
        test_request = {
            "test": True,
            "message": "Connection test from Principles Gym"
        }
        response = requests.post("http://localhost:8080/process", json=test_request, timeout=5)
        print(f"   Agent responded with status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Warning: Expected status 200, got {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Could not connect to agent: {e}")
        print("   Make sure your agent is running at http://localhost:8080/process")
        print("   The gym will still try to register it.")
    
    print("\n2. Registering HTTP agent...")
    agent_data = {
        "name": "My HTTP Agent",
        "framework": "http",
        "config": {
            "endpoint_url": "http://localhost:8080/process",
            "method": "POST",
            "request_format": "json",
            "response_format": "json",
            "timeout": 30
        },
        "description": "HTTP agent running at localhost:8080"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/agents/register",
        json=agent_data,
        headers=headers
    )
    
    if response.status_code == 201:
        agent_info = response.json()
        agent_id = agent_info["agent_id"]
        print(f"✅ HTTP agent registered successfully!")
        print(f"   Agent ID: {agent_id}")
        print(f"   Name: {agent_info['name']}")
        print(f"   Endpoint: http://localhost:8080/process")
    else:
        print(f"❌ Failed to register agent: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n3. Starting training session...")
    print("   Your agent will receive JSON requests like:")
    print(json.dumps({
        "scenario": {
            "description": "You and another agent must decide...",
            "choice_options": [
                {"id": "cooperate", "description": "Work together"},
                {"id": "defect", "description": "Act selfishly"}
            ]
        },
        "history": []
    }, indent=2))
    
    print("\n   Expected response format:")
    print(json.dumps({
        "action": "cooperate",
        "reasoning": "I choose to cooperate because...",
        "confidence": 0.8
    }, indent=2))
    
    # Step 3: Start training
    training_data = {
        "agent_id": agent_id,
        "scenario_types": ["trust_game", "resource_sharing", "prisoner_dilemma"],
        "num_scenarios": 5,    # Start with 5 scenarios
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
        print(f"\n✅ Training started successfully!")
        print(f"   Session ID: {session_id}")
        print(f"   Status: {training_info['status']}")
        print(f"   Scenarios: 5")
    else:
        print(f"❌ Failed to start training: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n4. Monitoring training progress...")
    print("   (Press Ctrl+C to stop monitoring)\n")
    
    # Step 4: Monitor progress
    completed = False
    last_progress = -1
    
    try:
        while not completed:
            time.sleep(2)  # Check every 2 seconds
            
            response = requests.get(
                f"{BASE_URL}/api/training/status/{session_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                status_info = response.json()
                progress = int(status_info['progress'] * 100)
                
                # Only print if progress changed
                if progress != last_progress:
                    print(f"\r   Progress: {progress}% | Scenarios: {status_info['scenarios_completed']}/{status_info['scenarios_total']} | Status: {status_info['status']}", end="", flush=True)
                    last_progress = progress
                
                if status_info['status'] == 'completed':
                    completed = True
                    print("\n\n✅ Training completed!")
                elif status_info['status'] == 'failed':
                    print(f"\n\n❌ Training failed: {status_info.get('error_message', 'Unknown error')}")
                    return
            else:
                print(f"\n❌ Failed to get status: {response.status_code}")
                return
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped. Training continues in background.")
        print(f"   Check status at: {BASE_URL}/api/training/status/{session_id}")
        return
    
    # Step 5: Get report
    print("\n5. Getting training report...")
    
    response = requests.get(
        f"{BASE_URL}/api/reports/{session_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        report = response.json()
        print(f"\n=== Training Report ===")
        print(f"Duration: {report['duration_seconds']:.1f} seconds")
        print(f"Scenarios Completed: {report['scenarios_completed']}")
        print(f"Behavioral Entropy: {report['behavioral_entropy']:.2f}")
        
        print(f"\nDiscovered Principles ({len(report['principles_discovered'])}):")
        for p in report['principles_discovered']:
            print(f"- {p['name']}: {p['description']}")
            print(f"  Strength: {p['strength']:.2f}, Consistency: {p['consistency']:.2f}")
        
        print(f"\nSummary: {report['summary']}")
        
        # Performance metrics
        perf = report.get('performance_metrics', {})
        if perf.get('action_timeouts', 0) > 0:
            print(f"\n⚠️  Timeouts: {perf['action_timeouts']} / {perf['total_actions']} actions")
    else:
        print(f"❌ Failed to get report: {response.status_code}")
        print(f"   Response: {response.text}")
    
    print("\n=== Test complete! ===")
    print(f"\nView detailed results at: {BASE_URL}")
    print(f"Session ID: {session_id}")

if __name__ == "__main__":
    print("Make sure:")
    print("1. Your agent is running at http://localhost:8080/process")
    print("2. The AI Principles Gym is running (use dev-start.bat)")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    test_http_agent()
