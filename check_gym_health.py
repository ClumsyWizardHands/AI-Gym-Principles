"""Check the health of the AI Principles Gym API."""

import requests

print("🩺 Checking AI Principles Gym health...\n")

# Check main API health
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("✅ API is healthy!")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ API health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Could not connect to API: {str(e)}")

# Check available agents
print("\n📋 Checking registered agents...")
try:
    headers = {"X-API-Key": "sk-dev-key"}
    response = requests.get("http://localhost:8000/api/agents", headers=headers, timeout=5)
    if response.status_code == 200:
        agents = response.json()
        print(f"✅ Found {len(agents)} registered agents:")
        for agent in agents[:5]:  # Show first 5
            print(f"   - {agent['name']} ({agent['framework']}) - Status: {agent['status']}")
    else:
        print(f"❌ Could not get agents: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Check if we can list training sessions
print("\n📊 Checking training sessions...")
try:
    response = requests.get("http://localhost:8000/api/training/sessions", headers=headers, timeout=5)
    if response.status_code == 200:
        sessions = response.json()
        print(f"✅ Found {len(sessions)} training sessions")
        if sessions:
            print("   Recent sessions:")
            for session in sessions[:3]:
                print(f"   - {session['id']}: {session['status']} (Agent: {session['agent_id'][:8]}...)")
    else:
        print(f"❌ Could not get sessions: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n💡 Tip: If you're seeing 500 errors, try:")
print("   1. Restart the gym: Ctrl+C in the terminal running dev-start.bat")
print("   2. Check the gym's console for error messages")
print("   3. Ensure all dependencies are installed")
