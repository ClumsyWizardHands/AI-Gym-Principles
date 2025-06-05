"""
Comprehensive test for Adaptive Bridge Builder agent
Tests the agent through various scenario types with thorough training
"""
import asyncio
import aiohttp
import json
import os
from datetime import datetime
import sys

# Configuration
GYM_URL = "http://localhost:8000"
AGENT_URL = "http://localhost:8080"
AGENT_ID_FILE = "adaptive_bridge_builder_agent_id.txt"

# Load API key from environment
API_KEY = os.getenv("PRINCIPLES_API_KEY", "test-api-key-123")

async def check_agent_health():
    """Check if the agent is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{AGENT_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Agent health check: {data}")
                    return True
                else:
                    print(f"❌ Agent health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Could not connect to agent: {e}")
        return False

async def check_gym_health():
    """Check if the gym is running"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": API_KEY}
            async with session.get(f"{GYM_URL}/health", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Gym health check: {data}")
                    return True
                else:
                    print(f"❌ Gym health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Could not connect to gym: {e}")
        return False

async def get_or_register_agent():
    """Get existing agent ID or register new one"""
    # Check if we have a saved agent ID
    if os.path.exists(AGENT_ID_FILE):
        with open(AGENT_ID_FILE, 'r') as f:
            agent_id = f.read().strip()
            print(f"📋 Using existing agent ID: {agent_id}")
            
            # Verify it exists in the gym
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-Key": API_KEY}
                async with session.get(f"{GYM_URL}/api/agents", headers=headers) as response:
                    if response.status == 200:
                        agents = await response.json()
                        # Handle different response formats
                        if isinstance(agents, list):
                            # Check both 'id' and 'agent_id' fields
                            found = False
                            for agent in agents:
                                agent_id_field = agent.get('id') or agent.get('agent_id')
                                if agent_id_field == agent_id:
                                    found = True
                                    break
                            if found:
                                return agent_id
                            else:
                                print("⚠️ Saved agent ID not found in gym, will register new one")
                        else:
                            # Agents might be in a different format, just use the saved ID
                            print("⚠️ Could not verify agent in gym, proceeding with saved ID")
                            return agent_id
    
    # Register new agent
    print("🔄 Registering new agent...")
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        agent_data = {
            "name": "Adaptive Bridge Builder",
            "framework": "http",
            "config": {
                "endpoint_url": AGENT_URL
            },
            "metadata": {
                "description": "An agent designed to facilitate communication and collaboration between different agents and systems using the A2A Protocol",
                "profile": "Empire of the Adaptive Hero",
                "capabilities": [
                    "multi-modal communication",
                    "emoji understanding", 
                    "emotional intelligence",
                    "task coordination",
                    "principle-based reasoning"
                ],
                "version": "2.0"
            }
        }
        
        async with session.post(f"{GYM_URL}/api/agents/register", 
                               headers=headers, 
                               json=agent_data) as response:
            if response.status in [200, 201]:
                result = await response.json()
                agent_id = result['agent_id']
                print(f"✅ Agent registered successfully: {agent_id}")
                
                # Save agent ID
                with open(AGENT_ID_FILE, 'w') as f:
                    f.write(agent_id)
                
                return agent_id
            else:
                error = await response.text()
                # Check if it's actually a success response with different structure
                try:
                    result = json.loads(error)
                    if 'agent_id' in result and result.get('status') == 'active':
                        agent_id = result['agent_id']
                        print(f"✅ Agent registered successfully: {agent_id}")
                        
                        # Save agent ID
                        with open(AGENT_ID_FILE, 'w') as f:
                            f.write(agent_id)
                        
                        return agent_id
                except:
                    pass
                    
                print(f"❌ Failed to register agent: {error}")
                return None

async def start_comprehensive_training(agent_id):
    """Start a comprehensive training session with multiple scenario types"""
    print("\n🏋️ Starting comprehensive training session...")
    
    training_configs = [
        {
            "name": "Ethical Dilemmas",
            "scenario_types": ["ethical_dilemma"],
            "num_scenarios": 5,
            "description": "Testing moral reasoning and principle development"
        },
        {
            "name": "Resource Management",
            "scenario_types": ["resource_allocation"],
            "num_scenarios": 5,
            "description": "Testing fairness and optimization principles"
        },
        {
            "name": "Trust Building",
            "scenario_types": ["trust_game"],
            "num_scenarios": 5,
            "description": "Testing cooperation and relationship building"
        },
        {
            "name": "Strategic Planning",
            "scenario_types": ["strategic_planning"],
            "num_scenarios": 5,
            "description": "Testing long-term thinking and adaptation"
        },
        {
            "name": "Mixed Scenarios",
            "scenario_types": ["ethical_dilemma", "resource_allocation", "trust_game"],
            "num_scenarios": 10,
            "description": "Testing adaptability across different contexts"
        }
    ]
    
    all_sessions = []
    
    for config in training_configs:
        print(f"\n📚 {config['name']}: {config['description']}")
        
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            training_data = {
                "agent_id": agent_id,
                "scenario_types": config["scenario_types"],
                "num_scenarios": config["num_scenarios"],
                "config": {
                    "timeout": 30,
                    "collect_metrics": True,
                    "verbose": True
                }
            }
            
            async with session.post(f"{GYM_URL}/api/training/start",
                                   headers=headers,
                                   json=training_data) as response:
                if response.status == 200:
                    result = await response.json()
                    session_id = result.get('session_id')
                    print(f"✅ Training session started: {session_id}")
                    all_sessions.append({
                        "session_id": session_id,
                        "name": config["name"],
                        "config": config
                    })
                    
                    # Wait a bit between sessions
                    await asyncio.sleep(2)
                else:
                    error = await response.text()
                    print(f"❌ Failed to start training: {error}")
    
    return all_sessions

async def monitor_sessions(sessions):
    """Monitor training sessions and display results"""
    print("\n📊 Monitoring training sessions...")
    
    completed_sessions = []
    
    while sessions:
        await asyncio.sleep(5)  # Check every 5 seconds
        
        remaining_sessions = []
        
        for session_info in sessions:
            session_id = session_info["session_id"]
            
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-Key": API_KEY}
                async with session.get(f"{GYM_URL}/api/training/{session_id}/status",
                                     headers=headers) as response:
                    if response.status == 200:
                        status = await response.json()
                        
                        if status.get("status") == "completed":
                            print(f"\n✅ Session completed: {session_info['name']}")
                            print(f"   - Scenarios: {status.get('completed_scenarios', 0)}/{status.get('total_scenarios', 0)}")
                            print(f"   - Duration: {status.get('duration', 'N/A')}s")
                            
                            # Get principles
                            if 'principles' in status:
                                print(f"   - Principles inferred: {len(status['principles'])}")
                                for principle in status['principles'][:3]:  # Show first 3
                                    print(f"     • {principle.get('description', 'N/A')}")
                            
                            completed_sessions.append(session_info)
                        else:
                            print(f"⏳ {session_info['name']}: {status.get('status', 'unknown')} "
                                  f"({status.get('completed_scenarios', 0)}/{status.get('total_scenarios', 0)})")
                            remaining_sessions.append(session_info)
                    else:
                        print(f"❌ Failed to get status for {session_id}")
                        remaining_sessions.append(session_info)
        
        sessions = remaining_sessions
    
    return completed_sessions

async def generate_report(agent_id):
    """Generate a comprehensive report of the agent's training"""
    print("\n📄 Generating comprehensive report...")
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY}
        
        # Get agent profile
        async with session.get(f"{GYM_URL}/api/agents", headers=headers) as response:
            if response.status == 200:
                agents = await response.json()
                agent = next((a for a in agents if a['id'] == agent_id), None)
                
                if agent:
                    print(f"\n🤖 Agent: {agent['name']}")
                    print(f"   Framework: {agent['framework']}")
                    print(f"   Created: {agent.get('created_at', 'N/A')}")
        
        # Get all training sessions
        async with session.get(f"{GYM_URL}/api/training/sessions", headers=headers) as response:
            if response.status == 200:
                sessions = await response.json()
                agent_sessions = [s for s in sessions if s.get('agent_id') == agent_id]
                
                print(f"\n📊 Training Summary:")
                print(f"   Total sessions: {len(agent_sessions)}")
                
                # Count scenarios by type
                scenario_counts = {}
                total_scenarios = 0
                for sess in agent_sessions:
                    for scenario_type in sess.get('scenario_types', []):
                        scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
                    total_scenarios += sess.get('completed_scenarios', 0)
                
                print(f"   Total scenarios completed: {total_scenarios}")
                print(f"   Scenario breakdown:")
                for stype, count in scenario_counts.items():
                    print(f"     - {stype}: {count}")
        
        # Get principles
        async with session.get(f"{GYM_URL}/api/agents/{agent_id}/principles", 
                             headers=headers) as response:
            if response.status == 200:
                principles = await response.json()
                
                print(f"\n💡 Principles Developed: {len(principles)}")
                
                # Group by type
                principle_types = {}
                for principle in principles:
                    ptype = principle.get('type', 'unknown')
                    if ptype not in principle_types:
                        principle_types[ptype] = []
                    principle_types[ptype].append(principle)
                
                for ptype, prins in principle_types.items():
                    print(f"\n   {ptype.title()} Principles ({len(prins)}):")
                    for p in prins[:3]:  # Show first 3 of each type
                        print(f"     • {p.get('description', 'N/A')}")
                        print(f"       Confidence: {p.get('confidence', 0):.2f}")

async def main():
    """Main test function"""
    print("🚀 AI Principles Gym - Adaptive Bridge Builder Comprehensive Test")
    print("=" * 60)
    
    # Check services
    if not await check_agent_health():
        print("\n⚠️ Agent is not running. Please start it on port 8080.")
        return
    
    if not await check_gym_health():
        print("\n⚠️ Gym is not running. Please start it with dev-start.bat")
        return
    
    # Get or register agent
    agent_id = await get_or_register_agent()
    if not agent_id:
        print("\n❌ Failed to get agent ID")
        return
    
    # Start comprehensive training
    sessions = await start_comprehensive_training(agent_id)
    
    if sessions:
        # Monitor progress
        completed = await monitor_sessions(sessions)
        
        # Generate report
        await generate_report(agent_id)
        
        print("\n✅ Comprehensive training completed!")
        print(f"   Completed sessions: {len(completed)}")
    else:
        print("\n❌ No training sessions were started")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
