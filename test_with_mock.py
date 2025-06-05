"""
Test training with a mock adapter to verify gym functionality
"""
import asyncio
import aiohttp
import json
import os

# Configuration
GYM_URL = "http://localhost:8000"
API_KEY = os.getenv("PRINCIPLES_API_KEY", "test-api-key-123")

async def register_mock_agent():
    """Register a mock agent for testing"""
    print("🔄 Registering mock agent...")
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        agent_data = {
            "name": "Mock Test Agent",
            "framework": "mock",
            "config": {},
            "metadata": {
                "description": "Simple mock agent for testing gym functionality"
            }
        }
        
        async with session.post(f"{GYM_URL}/api/agents/register", 
                               headers=headers, 
                               json=agent_data) as response:
            if response.status in [200, 201]:
                result = await response.json()
                agent_id = result['agent_id']
                print(f"✅ Mock agent registered: {agent_id}")
                return agent_id
            else:
                error = await response.text()
                print(f"❌ Failed to register: {error}")
                return None

async def test_mock_training(agent_id):
    """Test training with the mock agent"""
    print(f"\n🧪 Testing training with mock agent {agent_id}...")
    
    training_data = {
        "agent_id": agent_id,
        "scenario_types": ["trust_game"],
        "num_scenarios": 2,
        "config": {
            "timeout": 30,
            "collect_metrics": True,
            "verbose": True
        }
    }
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        
        async with session.post(f"{GYM_URL}/api/training/start",
                               headers=headers,
                               json=training_data) as response:
            if response.status == 200:
                result = await response.json()
                session_id = result.get('session_id')
                print(f"✅ Training started: {session_id}")
                return session_id
            else:
                error = await response.text()
                print(f"❌ Failed to start training: {error}")
                return None

async def monitor_training(session_id):
    """Monitor the training session"""
    print(f"\n📊 Monitoring session {session_id}...")
    
    for i in range(30):
        await asyncio.sleep(2)
        
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": API_KEY}
            async with session.get(f"{GYM_URL}/api/training/{session_id}/status",
                                 headers=headers) as response:
                if response.status == 200:
                    status = await response.json()
                    state = status.get('status', 'unknown')
                    completed_count = status.get('completed_scenarios', 0)
                    total_count = status.get('total_scenarios', 0)
                    
                    print(f"   Status: {state} - Scenarios: {completed_count}/{total_count}")
                    
                    if state == 'completed':
                        print("\n✅ Training completed successfully!")
                        return True
                    elif state == 'error':
                        print(f"\n❌ Training error: {status.get('error', 'Unknown error')}")
                        return False
    
    print("\n⏱️ Training timeout")
    return False

async def main():
    """Main test function"""
    print("🧪 Mock Agent Test - Verify Gym Functionality")
    print("=" * 50)
    
    # Register mock agent
    agent_id = await register_mock_agent()
    if not agent_id:
        print("\n❌ Failed to register mock agent")
        return
    
    # Test training
    session_id = await test_mock_training(agent_id)
    if session_id:
        success = await monitor_training(session_id)
        if success:
            print("\n✅ Gym is working correctly with mock adapter!")
        else:
            print("\n⚠️ Training did not complete successfully")
    
    print("\n✅ Mock test completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
