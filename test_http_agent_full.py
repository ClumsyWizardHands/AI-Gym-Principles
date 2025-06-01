#!/usr/bin/env python3
"""
Full test of HTTP agent integration with AI Principles Gym.
This script:
1. Registers your HTTP agent
2. Runs a complete training session
3. Generates behavioral principles report
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
import requests
from typing import Dict, Any

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import settings
from src.core.database import DatabaseManager
from src.scenarios.engine import ScenarioEngine
from src.core.tracking import ProgressTracker
from src.core.models import Agent, TrainingSession
from src.core.inference import PrincipleInference


def test_http_agent_connection(endpoint: str) -> bool:
    """Test if HTTP agent is responding."""
    print(f"\n📡 Testing connection to HTTP agent at {endpoint}...")
    
    test_request = {
        "scenario": {
            "id": "test",
            "description": "Connection test",
            "choice_options": [
                {"id": "option1", "description": "Test option"}
            ]
        },
        "history": []
    }
    
    try:
        response = requests.post(endpoint, json=test_request, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Agent responded: {data}")
            return True
        else:
            print(f"❌ Agent returned status {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to agent at {endpoint}")
        return False
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return False


async def register_http_agent(db_manager: DatabaseManager, endpoint: str, name: str = "My HTTP Agent") -> Agent:
    """Register HTTP agent with the gym."""
    print(f"\n📝 Registering HTTP agent '{name}'...")
    
    async with db_manager.get_session() as session:
        # Check if agent already exists
        existing = await session.execute(
            f"SELECT * FROM agents WHERE name = '{name}'"
        )
        if existing.first():
            print(f"⚠️  Agent '{name}' already exists, using existing registration")
            result = await session.execute(
                f"SELECT * FROM agents WHERE name = '{name}'"
            )
            agent_data = result.first()
            return Agent(
                id=agent_data.id,
                name=agent_data.name,
                framework=agent_data.framework,
                config=json.loads(agent_data.config) if agent_data.config else {}
            )
        
        # Create new agent
        agent = Agent(
            id=f"agent_{int(time.time())}",
            name=name,
            framework="http",
            config={
                "endpoint": endpoint,
                "timeout": 30,
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        )
        
        # Save to database
        await session.execute(
            """INSERT INTO agents (id, name, framework, config, created_at)
               VALUES (:id, :name, :framework, :config, :created_at)""",
            {
                "id": agent.id,
                "name": agent.name,
                "framework": agent.framework,
                "config": json.dumps(agent.config),
                "created_at": datetime.utcnow()
            }
        )
        await session.commit()
        
        print(f"✅ Agent registered with ID: {agent.id}")
        return agent


async def run_training_session(
    db_manager: DatabaseManager,
    agent: Agent,
    num_scenarios: int = 10
) -> TrainingSession:
    """Run a full training session."""
    print(f"\n🏋️ Starting training session with {num_scenarios} scenarios...")
    
    # Create training session
    session = TrainingSession(
        id=f"session_{int(time.time())}",
        agent_id=agent.id,
        scenario_type="mixed",  # Use mixed scenarios
        total_scenarios=num_scenarios,
        config={
            "include_ethical": True,
            "include_strategic": True,
            "include_creative": True,
            "randomize": True
        }
    )
    
    # Initialize components
    engine = ScenarioEngine()
    tracker = ProgressTracker(session.id)
    inference = PrincipleInference()
    
    # Save session to database
    async with db_manager.get_session() as db_session:
        await db_session.execute(
            """INSERT INTO training_sessions 
               (id, agent_id, scenario_type, total_scenarios, status, config, created_at)
               VALUES (:id, :agent_id, :scenario_type, :total_scenarios, :status, :config, :created_at)""",
            {
                "id": session.id,
                "agent_id": session.agent_id,
                "scenario_type": session.scenario_type,
                "total_scenarios": session.total_scenarios,
                "status": "in_progress",
                "config": json.dumps(session.config),
                "created_at": datetime.utcnow()
            }
        )
        await db_session.commit()
    
    # Run scenarios
    for i in range(num_scenarios):
        print(f"\n📊 Scenario {i+1}/{num_scenarios}")
        
        # Get scenario
        scenario = engine.generate_scenario(
            scenario_type="mixed",
            difficulty=min(i / num_scenarios, 0.8)  # Gradually increase difficulty
        )
        
        print(f"📖 {scenario.description}")
        print(f"   Options:")
        for opt in scenario.choice_options:
            print(f"   - {opt.id}: {opt.description}")
        
        # Get agent response
        request_data = {
            "scenario": {
                "id": scenario.id,
                "description": scenario.description,
                "choice_options": [
                    {"id": opt.id, "description": opt.description}
                    for opt in scenario.choice_options
                ]
            },
            "history": tracker.get_history()
        }
        
        try:
            response = requests.post(
                agent.config["endpoint"],
                json=request_data,
                timeout=agent.config.get("timeout", 30)
            )
            
            if response.status_code == 200:
                agent_response = response.json()
                action = agent_response.get("action")
                reasoning = agent_response.get("reasoning", "")
                confidence = agent_response.get("confidence", 0.5)
                
                print(f"🤖 Agent chose: {action}")
                if reasoning:
                    print(f"   Reasoning: {reasoning}")
                print(f"   Confidence: {confidence:.2f}")
                
                # Track the decision
                tracker.record_decision(
                    scenario_id=scenario.id,
                    action=action,
                    reasoning=reasoning,
                    outcome={"confidence": confidence}
                )
                
                # Update inference with new data
                inference.update(scenario, action, reasoning)
                
            else:
                print(f"❌ Agent error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error communicating with agent: {e}")
        
        # Small delay between scenarios
        await asyncio.sleep(0.5)
    
    # Generate principles report
    print("\n🔍 Analyzing behavioral patterns...")
    principles = inference.extract_principles()
    
    # Update session status
    async with db_manager.get_session() as db_session:
        await db_session.execute(
            """UPDATE training_sessions 
               SET status = 'completed', completed_at = :completed_at,
                   decisions_made = :decisions, principles = :principles
               WHERE id = :id""",
            {
                "id": session.id,
                "completed_at": datetime.utcnow(),
                "decisions": len(tracker.get_history()),
                "principles": json.dumps(principles)
            }
        )
        await db_session.commit()
    
    # Display results
    print("\n" + "="*60)
    print("📊 TRAINING COMPLETE!")
    print("="*60)
    print(f"Session ID: {session.id}")
    print(f"Scenarios completed: {num_scenarios}")
    print(f"Decisions tracked: {len(tracker.get_history())}")
    
    print("\n🧠 DISCOVERED BEHAVIORAL PRINCIPLES:")
    print("-"*60)
    
    if principles.get("ethical_principles"):
        print("\n⚖️ Ethical Principles:")
        for principle in principles["ethical_principles"]:
            print(f"  • {principle['description']} (strength: {principle['strength']:.2f})")
    
    if principles.get("strategic_principles"):
        print("\n♟️ Strategic Principles:")
        for principle in principles["strategic_principles"]:
            print(f"  • {principle['description']} (strength: {principle['strength']:.2f})")
    
    if principles.get("consistency_metrics"):
        print("\n📈 Consistency Metrics:")
        metrics = principles["consistency_metrics"]
        print(f"  • Overall consistency: {metrics.get('overall', 0):.2%}")
        print(f"  • Ethical consistency: {metrics.get('ethical', 0):.2%}")
        print(f"  • Strategic consistency: {metrics.get('strategic', 0):.2%}")
    
    print("\n" + "="*60)
    
    return session


async def main():
    """Main test function."""
    print("🎯 AI Principles Gym - HTTP Agent Full Test")
    print("="*60)
    
    # Configuration
    AGENT_ENDPOINT = "http://localhost:8080/process"  # Your agent endpoint
    AGENT_NAME = "My HTTP Agent"
    NUM_SCENARIOS = 10  # Number of scenarios to run
    
    # Check if agent is running
    if not test_http_agent_connection(AGENT_ENDPOINT):
        print("\n⚠️ Please make sure your HTTP agent is running at", AGENT_ENDPOINT)
        print("The agent should accept POST requests with scenario data and return action choices.")
        return
    
    # Initialize database
    print("\n🗄️ Initializing database...")
    db_manager = DatabaseManager(settings.DATABASE_URL)
    await db_manager.initialize()
    
    try:
        # Register agent
        agent = await register_http_agent(db_manager, AGENT_ENDPOINT, AGENT_NAME)
        
        # Run training session
        session = await run_training_session(db_manager, agent, NUM_SCENARIOS)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results saved to database")
        print(f"🌐 View in web UI at http://localhost:5173")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await db_manager.close()


if __name__ == "__main__":
    # Check for API keys if needed
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ No API keys found in environment.")
        print("The gym will use the HTTP agent adapter, which doesn't require API keys.")
    
    # Run the test
    asyncio.run(main())
