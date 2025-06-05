"""Start training session with the Adaptive Bridge Builder agent."""
import requests
import json

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "sk-dev-key"  # Development key

# Read the agent ID from file
with open('adaptive_bridge_builder_agent_id.txt', 'r') as f:
    agent_id = f.read().strip()

print(f"🎯 Starting training session for Adaptive Bridge Builder agent...")
print(f"   Agent ID: {agent_id}")
print()

# Start training session
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

training_data = {
    "agent_id": agent_id,
    "scenario_types": ["LOYALTY", "SCARCITY", "BETRAYAL", "TRADEOFFS", "TIME_PRESSURE"],
    "num_scenarios": 10,
    "adaptive": True,
    "use_branching": True,
    "branching_types": ["trust_building", "resource_cascade"]
}

response = requests.post(
    f"{API_URL}/api/training/start",
    headers=headers,
    json=training_data
)

if response.status_code == 202:
    result = response.json()
    print("✅ Training session started successfully!")
    print(f"   Session ID: {result['session_id']}")
    print(f"   Status: {result['status']}")
    print(f"   Estimated duration: {result['estimated_duration_seconds']} seconds")
else:
    print(f"❌ Failed to start training session. Status code: {response.status_code}")
    print(f"Response: {response.text}")
