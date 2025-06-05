"""Test basic workflow with mock adapter."""
import asyncio
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test if API is healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def register_mock_agent():
    """Register a mock agent."""
    try:
        # Simple registration with mock adapter
        data = {
            "name": f"Test Agent {datetime.now().strftime('%H%M%S')}",
            "framework": "mock",
            "config": {
                "model": "mock-v1"
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "sk-dev-key"  # Use the dev key from .env
        }
        
        response = requests.post(
            f"{BASE_URL}/api/agents/register",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            agent_data = response.json()
            print(f"✅ Agent registered: {agent_data['agent_id']}")
            return agent_data['agent_id']
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return None

def list_agents():
    """List all agents."""
    try:
        headers = {
            "X-API-Key": "sk-dev-key"
        }
        
        response = requests.get(
            f"{BASE_URL}/api/agents",
            headers=headers
        )
        
        if response.status_code == 200:
            agents = response.json()
            print(f"✅ Found {len(agents)} agents:")
            for agent in agents:
                print(f"   - {agent['agent_id']}: {agent['name']}")
            return True
        else:
            print(f"❌ List agents failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ List agents error: {e}")
        return False

def main():
    """Run basic workflow test."""
    print("=== Testing Basic AI Principles Gym Workflow ===\n")
    
    # Test health
    if not test_health():
        print("\nAPI server is not running. Please start it first.")
        return
    
    print("\n1. Registering a mock agent...")
    agent_id = register_mock_agent()
    
    if agent_id:
        print("\n2. Listing all agents...")
        list_agents()
        
        print("\n✅ Basic workflow test completed successfully!")
    else:
        print("\n❌ Basic workflow test failed.")

if __name__ == "__main__":
    main()
