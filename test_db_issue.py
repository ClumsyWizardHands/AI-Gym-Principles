"""Test database issue."""
import asyncio
from src.core.database import get_db_manager

async def test_create_agent():
    """Test creating an agent in the database."""
    try:
        db_manager = await get_db_manager()
        
        # Use a unique ID for each test run
        import uuid
        agent_id = f"test-agent-{uuid.uuid4()}"
        agent_name = "Test Agent"
        metadata = {
            "framework": "http",
            "config": {},
            "description": "Test agent",
            "api_key": "sk-dev-key"
        }
        
        print("Creating agent in database...")
        agent = await db_manager.create_agent(agent_id, agent_name, metadata)
        print(f"✅ Agent created successfully: {agent.agent_id}")
        
        # Try to retrieve it
        print("\nRetrieving agent from database...")
        retrieved = await db_manager.get_agent(agent_id)
        if retrieved:
            print(f"✅ Agent retrieved: {retrieved.name}")
            print(f"✅ Agent metadata: {retrieved.meta_data}")
        else:
            print("❌ Agent not found")
            
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_create_agent())
