"""List all registered agents."""
import requests

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "sk-dev-key"  # Development key

headers = {
    "X-API-Key": API_KEY
}

# Get all agents
response = requests.get(f"{API_URL}/api/agents", headers=headers)

if response.status_code == 200:
    agents = response.json()
    print(f"📋 Found {len(agents)} registered agents:\n")
    
    for agent in agents:
        print(f"Agent ID: {agent['agent_id']}")
        print(f"Name: {agent['name']}")
        print(f"Framework: {agent['framework']}")
        print(f"Status: {agent['status']}")
        print(f"Registered: {agent['registered_at']}")
        print("-" * 50)
else:
    print(f"❌ Failed to get agents. Status code: {response.status_code}")
    print(f"Response: {response.text}")
