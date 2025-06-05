"""
Register the bridge adapter as an agent and test training
"""
import asyncio
import aiohttp
import json
import os

# Configuration
GYM_URL = "http://localhost:8000"
BRIDGE_URL = "http://localhost:8085"
API_KEY = os.getenv("PRINCIPLES_API_KEY", "test-api-key-123")

async def register_bridge_agent():
    """Register the bridge as an HTTP agent"""
    print("🔄 Registering bridge adapter as agent...")
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        agent_data = {
            "name": "Adaptive Bridge Builder (Bridged)",
            "framework": "http",
            "config": {
                "endpoint_url": BRIDGE_URL
            },
            "metadata": {
                "description": "Bridge adapter for Adaptive Bridge Builder agent",
                "bridge_info": {
                    "bridge_port": 8085,
                    "actual_agent_port": 8080,
                    "protocol": "JSON-RPC to HTTP translation"
                }
            }
        }
        
        async with session.post(f"{GYM_URL}/api/agents/register", 
                               headers=headers, 
                               json=agent_data) as response:
            if response.status in [200, 201]:
                result = await response.json()
                agent_id = result['agent_id']
                print(f"✅ Bridge agent registered: {agent_id}")
                return agent_id
            else:
                error = await response.text()
                print(f"❌ Failed to register: {error}")
                return None

async def test_bridge_training(agent_id):
    """Test training with the bridged agent"""
    print(f"\n🧪 Testing training with bridged agent {agent_id}...")
    
    # Simple training configuration
    training_data = {
        "agent_id": agent_id,
        "scenario_types": ["trust_game"],
        "num_scenarios": 2,
        "config": {
            "timeout": 60,
            "collect_metrics": True,
            "verbose": True
        }
    }
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        
        print(f"📤 Starting training session...")
        
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
    
    completed = False
    for i in range(60):  # Monitor for up to 60 seconds
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
                        
                        # Show some results
                        if 'actions' in status:
                            print("\n📝 Agent decisions:")
                            for action in status.get('actions', [])[:3]:
                                print(f"   - {action.get('action', 'unknown')}: {action.get('intent', 'No reason')[:100]}")
                        
                        if 'principles' in status:
                            print(f"\n💡 Principles inferred: {len(status['principles'])}")
                            for principle in status['principles'][:3]:
                                print(f"   - {principle.get('description', 'N/A')}")
                        
                        completed = True
                        break
                    elif state == 'error':
                        print(f"\n❌ Training error: {status.get('error', 'Unknown error')}")
                        break
    
    if not completed:
        print("\n⏱️ Training timeout")
    
    return completed

async def main():
    """Main test function"""
    print("🌉 Bridge Adapter Test")
    print("=" * 50)
    
    # Check bridge health
    print("\n🔍 Checking bridge health...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BRIDGE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Bridge is healthy: {data}")
                else:
                    print(f"❌ Bridge health check failed: {response.status}")
                    return
    except Exception as e:
        print(f"❌ Could not connect to bridge: {e}")
        print("   Make sure the bridge is running on port 8085")
        return
    
    # Register bridge as agent
    agent_id = await register_bridge_agent()
    if not agent_id:
        print("\n❌ Failed to register bridge agent")
        return
    
    # Test training
    session_id = await test_bridge_training(agent_id)
    if session_id:
        success = await monitor_training(session_id)
        if success:
            print("\n🎉 Bridge adapter is working correctly!")
        else:
            print("\n⚠️ Training did not complete successfully")
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
