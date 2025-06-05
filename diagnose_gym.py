"""
Diagnose AI Principles Gym issues
"""
import asyncio
import aiohttp
import json
import os

# Configuration
GYM_URL = "http://localhost:8000"
API_KEY = os.getenv("PRINCIPLES_API_KEY", "test-api-key-123")

async def check_health():
    """Check gym health"""
    print("1️⃣ Checking gym health...")
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY}
        async with session.get(f"{GYM_URL}/health", headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Gym health: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"❌ Health check failed: {response.status}")
                return False

async def list_agents():
    """List all registered agents"""
    print("\n2️⃣ Listing registered agents...")
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY}
        async with session.get(f"{GYM_URL}/api/agents", headers=headers) as response:
            if response.status == 200:
                agents = await response.json()
                print(f"✅ Found {len(agents)} agents:")
                for agent in agents:
                    agent_id = agent.get('agent_id') or agent.get('id')
                    print(f"   - {agent.get('name')} ({agent_id}) - {agent.get('framework')}")
                return True
            else:
                error = await response.text()
                print(f"❌ Failed to list agents: {response.status} - {error}")
                return False

async def test_minimal_registration():
    """Try minimal agent registration"""
    print("\n3️⃣ Testing minimal agent registration...")
    
    # Try the absolute minimum required fields
    agent_data = {
        "name": "Minimal Test Agent",
        "framework": "mock"
    }
    
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        
        print(f"   Sending: {json.dumps(agent_data, indent=2)}")
        
        async with session.post(f"{GYM_URL}/api/agents/register", 
                               headers=headers, 
                               json=agent_data) as response:
            if response.status in [200, 201]:
                result = await response.json()
                print(f"✅ Registration successful: {json.dumps(result, indent=2)}")
                return True
            else:
                error = await response.text()
                print(f"❌ Registration failed ({response.status}): {error}")
                
                # Try to parse error for more details
                try:
                    error_json = json.loads(error)
                    if 'error' in error_json:
                        print(f"   Error details: {json.dumps(error_json['error'], indent=2)}")
                except:
                    pass
                
                return False

async def check_logs():
    """Check if there are any log files we can read"""
    print("\n4️⃣ Checking for log files...")
    log_paths = [
        "logs/app.log",
        "logs/training.log",
        "ai-principles-gym.log",
        "app.log"
    ]
    
    for path in log_paths:
        if os.path.exists(path):
            print(f"✅ Found log file: {path}")
            # Read last few lines
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    print(f"   Last 5 lines:")
                    for line in lines[-5:]:
                        print(f"     {line.strip()}")
            except Exception as e:
                print(f"   ❌ Could not read: {e}")
        else:
            print(f"❌ No log file at: {path}")

async def main():
    """Main diagnostic function"""
    print("🔍 AI Principles Gym Diagnostics")
    print("=" * 50)
    
    # Run diagnostics
    await check_health()
    await list_agents()
    await test_minimal_registration()
    await check_logs()
    
    print("\n✅ Diagnostics completed")
    print("\n💡 Next steps:")
    print("   - Check the dev-start.bat terminal for error logs")
    print("   - Look for database connection issues")
    print("   - Verify all required services are running")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnostics interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
