"""
Simple test to debug training issues with HTTP agent
"""
import asyncio
import aiohttp
import json
import os

# Configuration
GYM_URL = "http://localhost:8000"
AGENT_URL = "http://localhost:8080"
API_KEY = os.getenv("PRINCIPLES_API_KEY", "test-api-key-123")

async def test_agent_direct():
    """Test communicating with agent directly"""
    print("\n1️⃣ Testing direct agent communication...")
    
    # Test health endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{AGENT_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Agent health: {data}")
            else:
                print(f"❌ Agent health failed: {response.status}")
                return False
    
    # Test agent card endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{AGENT_URL}/agent-card") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Agent card: {json.dumps(data, indent=2)}")
            else:
                print(f"❌ Agent card failed: {response.status}")
                return False
    
    # Test process endpoint with a simple message
    test_message = {
        "jsonrpc": "2.0",
        "method": "process",
        "params": {
            "message": "Hello, can you hear me?"
        },
        "id": "test-1"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{AGENT_URL}/process", 
                              json=test_message,
                              headers={"Content-Type": "application/json"}) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Agent response: {json.dumps(data, indent=2)}")
                return True
            else:
                error = await response.text()
                print(f"❌ Agent process failed: {response.status} - {error}")
                return False

async def get_latest_agent_id():
    """Get the most recently registered agent"""
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY}
        async with session.get(f"{GYM_URL}/api/agents", headers=headers) as response:
            if response.status == 200:
                agents = await response.json()
                if agents:
                    # Get the last agent (most recent)
                    latest = agents[-1]
                    agent_id = latest.get('agent_id') or latest.get('id')
                    print(f"📋 Latest agent ID: {agent_id}")
                    print(f"   Name: {latest.get('name')}")
                    print(f"   Framework: {latest.get('framework')}")
                    return agent_id
    return None

async def test_simple_training(agent_id):
    """Test starting a simple training session"""
    print(f"\n2️⃣ Testing training start with agent {agent_id}...")
    
    # Try with just one scenario
    training_data = {
        "agent_id": agent_id,
        "scenario_types": ["trust_game"],
        "num_scenarios": 1,
        "config": {
            "timeout": 60,
            "collect_metrics": True,
            "verbose": True
        }
    }
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        
        print(f"📤 Sending training request: {json.dumps(training_data, indent=2)}")
        
        async with session.post(f"{GYM_URL}/api/training/start",
                               headers=headers,
                               json=training_data) as response:
            if response.status == 200:
                result = await response.json()
                session_id = result.get('session_id')
                print(f"✅ Training session started: {session_id}")
                return session_id
            else:
                error = await response.text()
                print(f"❌ Failed to start training: {response.status}")
                print(f"   Error: {error}")
                
                # Try to get more details from gym logs
                print("\n🔍 Checking gym status...")
                async with session.get(f"{GYM_URL}/health", headers=headers) as health_resp:
                    if health_resp.status == 200:
                        health_data = await health_resp.json()
                        print(f"   Gym health: {health_data}")
                
                return None

async def monitor_session(session_id):
    """Monitor a training session"""
    print(f"\n3️⃣ Monitoring session {session_id}...")
    
    for i in range(30):  # Monitor for up to 30 seconds
        await asyncio.sleep(1)
        
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": API_KEY}
            async with session.get(f"{GYM_URL}/api/training/{session_id}/status",
                                 headers=headers) as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"📊 Status: {status.get('status')} - "
                          f"Scenarios: {status.get('completed_scenarios', 0)}/{status.get('total_scenarios', 0)}")
                    
                    if status.get('status') == 'completed':
                        print("✅ Training completed!")
                        return True
                    elif status.get('status') == 'error':
                        print(f"❌ Training error: {status.get('error')}")
                        return False
                else:
                    print(f"❌ Failed to get status: {response.status}")
                    return False
    
    print("⏱️ Timeout waiting for completion")
    return False

async def main():
    """Main test function"""
    print("🔍 HTTP Agent Training Debug Test")
    print("=" * 50)
    
    # Test agent directly
    if not await test_agent_direct():
        print("\n❌ Agent is not responding correctly")
        return
    
    # Get latest agent ID
    agent_id = await get_latest_agent_id()
    if not agent_id:
        print("\n❌ No agents found in gym")
        return
    
    # Try to start training
    session_id = await test_simple_training(agent_id)
    if session_id:
        # Monitor the session
        await monitor_session(session_id)
    
    print("\n✅ Debug test completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
