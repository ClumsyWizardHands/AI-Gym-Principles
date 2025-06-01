"""Start a simple training session for the Adaptive Bridge Builder agent."""

import requests
import json

# Read the agent ID
try:
    with open("adaptive_bridge_builder_agent_id.txt", "r") as f:
        agent_id = f.read().strip()
except FileNotFoundError:
    print("❌ Error: Could not find agent ID file.")
    exit(1)

# API endpoint
url = "http://localhost:8000/api/training/start"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Simple training configuration
training_data = {
    "agent_id": agent_id,
    "num_scenarios": 5,
    "adaptive": True
}

try:
    # Start the training session
    response = requests.post(url, headers=headers, json=training_data)
    
    if response.status_code == 202:  # Accepted
        result = response.json()
        print("🎯 Training session started successfully!")
        print(f"Session ID: {result['session_id']}")
        print(f"Agent ID: {result['agent_id']}")
        print(f"Status: {result['status']}")
        print(f"Started at: {result['started_at']}")
        print(f"Estimated duration: {result['estimated_duration_seconds']} seconds")
        
        # Save session ID for monitoring
        with open("current_training_session_id.txt", "w") as f:
            f.write(result['session_id'])
        print("\n💾 Session ID saved to current_training_session_id.txt")
        
        print("\n📊 The gym will now send scenarios to your agent at http://localhost:8080/process")
        print("Make sure your Adaptive Bridge Builder is running and ready to receive requests!")
        
        print("\n🔍 To monitor progress:")
        print(f"- Web UI: http://localhost:5173")
        print(f"- API Status: GET http://localhost:8000/api/training/status/{result['session_id']}")
        
    else:
        print(f"❌ Failed to start training session. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the AI Principles Gym API")
except Exception as e:
    print(f"❌ Error: {str(e)}")
