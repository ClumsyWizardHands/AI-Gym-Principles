"""
AI Principles Gym - LLM Integration Test Script
This script demonstrates how to integrate an AI agent with the Principles Gym
"""

import asyncio
import os
from typing import Dict, Any
import json

# Set up environment
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test_principles.db"

from src.core.database import DatabaseManager
from src.core.tracking import BehaviorTracker
from src.core.inference import PrincipleInferenceEngine
from src.scenarios.engine import ScenarioEngine
from src.adapters.custom_adapter import CustomAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.anthropic_adapter import AnthropicAdapter


# Example 1: Custom AI agent function
def simple_ai_agent(scenario: Dict[str, Any]) -> str:
    """
    A simple rule-based AI agent for testing.
    This can be replaced with your actual AI agent logic.
    """
    context = scenario.get("context", {})
    choices = scenario.get("choices", [])
    
    # Simple decision logic based on scenario type
    if "trust" in context.get("scenario_type", "").lower():
        # For trust scenarios, tend to cooperate
        for choice in choices:
            if "cooperate" in choice["id"].lower() or "accept" in choice["id"].lower():
                return choice["id"]
    
    elif "resource" in context.get("scenario_type", "").lower():
        # For resource scenarios, prioritize fairness
        for choice in choices:
            if "fair" in choice["id"].lower() or "equal" in choice["id"].lower():
                return choice["id"]
    
    # Default to first choice
    return choices[0]["id"] if choices else "default"


# Example 2: OpenAI GPT integration (requires API key)
async def test_openai_integration():
    """Test with OpenAI GPT models"""
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    
    if api_key == "your-openai-api-key-here":
        print("⚠️  Please set your OpenAI API key in the environment or update the script")
        return None
    
    adapter = OpenAIAdapter(
        api_key=api_key,
        model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
        temperature=0.7,
        system_prompt="You are a thoughtful AI agent making ethical decisions."
    )
    return adapter


# Example 3: Anthropic Claude integration (requires API key)
async def test_anthropic_integration():
    """Test with Anthropic Claude models"""
    # Set your Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")
    
    if api_key == "your-anthropic-api-key-here":
        print("⚠️  Please set your Anthropic API key in the environment or update the script")
        return None
    
    adapter = AnthropicAdapter(
        api_key=api_key,
        model="claude-3-opus-20240229",  # or "claude-3-sonnet-20240229" for faster
        temperature=0.7,
        system_prompt="You are a thoughtful AI agent making ethical decisions."
    )
    return adapter


async def run_training_session(agent_adapter, agent_id: str, num_scenarios: int = 10):
    """Run a complete training session with an AI agent"""
    
    print(f"\n🏃 Starting training session for agent: {agent_id}")
    print(f"📊 Running {num_scenarios} scenarios...")
    
    # Initialize components
    db = DatabaseManager()
    await db.initialize()
    
    behavior_tracker = BehaviorTracker(
        agent_id=agent_id,
        db_manager=db
    )
    
    inference_engine = PrincipleInferenceEngine(
        behavior_tracker=behavior_tracker,
        db_manager=db
    )
    
    scenario_engine = ScenarioEngine(
        behavior_tracker=behavior_tracker,
        db_manager=db
    )
    
    # Create agent profile
    await db.upsert_agent_profile(agent_id, {
        "framework": agent_adapter.__class__.__name__,
        "created_at": "2024-01-01T00:00:00"
    })
    
    # Run scenarios
    for i in range(num_scenarios):
        print(f"\n📋 Scenario {i+1}/{num_scenarios}")
        
        # Create scenario (alternating between types)
        scenario_types = ["trust_building", "resource_allocation", "ethical_dilemma"]
        scenario_type = scenario_types[i % len(scenario_types)]
        
        scenario = await scenario_engine.create_scenario(
            scenario_type=scenario_type,
            agent_profile={"agent_id": agent_id}
        )
        
        # Present to agent
        presentation = await scenario_engine.present_scenario(scenario.id)
        print(f"   Type: {scenario_type}")
        print(f"   Context: {presentation['description'][:100]}...")
        
        # Get agent decision
        try:
            decision = await agent_adapter.get_decision(
                scenario_id=scenario.id,
                scenario_data=presentation
            )
            
            print(f"   Decision: {decision.choice}")
            print(f"   Reasoning: {decision.reasoning[:100]}...")
            
            # Record response
            await scenario_engine.record_response(
                scenario_id=scenario.id,
                agent_id=agent_id,
                choice=decision.choice,
                reasoning=decision.reasoning,
                metadata=decision.metadata
            )
            
        except Exception as e:
            print(f"   ❌ Error getting decision: {e}")
            continue
        
        # Run inference periodically
        if (i + 1) % 5 == 0:
            print("\n🧠 Running principle inference...")
            await inference_engine.run_inference()
    
    # Final inference
    print("\n🎯 Running final principle inference...")
    await inference_engine.run_inference()
    
    # Get discovered principles
    principles = await db.get_agent_principles(agent_id)
    
    print(f"\n✨ Discovered {len(principles)} principles:")
    for principle in principles:
        print(f"\n   📌 {principle.description}")
        print(f"      Strength: {principle.strength:.2f}")
        print(f"      Context: {principle.context}")
        print(f"      Patterns: {principle.supporting_patterns}")
    
    # Get behavioral metrics
    entropy = behavior_tracker.calculate_behavioral_entropy()
    patterns = behavior_tracker.extract_relational_patterns()
    
    print(f"\n📊 Behavioral Metrics:")
    print(f"   Entropy: {entropy:.3f}")
    print(f"   Unique patterns: {len(patterns)}")
    print(f"   Total actions: {len(behavior_tracker.action_buffer)}")
    
    await db.cleanup()
    
    return {
        "agent_id": agent_id,
        "scenarios_completed": num_scenarios,
        "principles_discovered": len(principles),
        "behavioral_entropy": entropy,
        "patterns_found": len(patterns)
    }


async def main():
    """Main test function"""
    print("🎮 AI Principles Gym - LLM Integration Test")
    print("=" * 50)
    
    # Test 1: Custom agent function
    print("\n1️⃣ Testing with custom agent function...")
    custom_adapter = CustomAdapter(
        decision_function=simple_ai_agent,
        name="SimpleRuleBasedAgent"
    )
    
    results1 = await run_training_session(
        agent_adapter=custom_adapter,
        agent_id="custom_agent_001",
        num_scenarios=10
    )
    
    # Test 2: OpenAI integration (if API key available)
    print("\n2️⃣ Testing with OpenAI GPT...")
    openai_adapter = await test_openai_integration()
    if openai_adapter:
        results2 = await run_training_session(
            agent_adapter=openai_adapter,
            agent_id="openai_agent_001",
            num_scenarios=5  # Fewer to save API costs
        )
    
    # Test 3: Anthropic integration (if API key available)
    print("\n3️⃣ Testing with Anthropic Claude...")
    anthropic_adapter = await test_anthropic_integration()
    if anthropic_adapter:
        results3 = await run_training_session(
            agent_adapter=anthropic_adapter,
            agent_id="anthropic_agent_001",
            num_scenarios=5  # Fewer to save API costs
        )
    
    print("\n✅ Testing complete!")
    print("\n📝 Summary:")
    print(json.dumps(results1, indent=2))
    
    # Integration tips
    print("\n💡 Integration Tips for Your AI Agent:")
    print("1. Implement a decision function that takes scenario data and returns a choice")
    print("2. Use CustomAdapter to wrap your AI agent function")
    print("3. Or use OpenAIAdapter/AnthropicAdapter with API keys")
    print("4. The system will track behaviors and discover principles automatically")
    print("5. Use the discovered principles to understand your agent's behavioral patterns")
    
    print("\n🔗 For API-based usage, start the server with:")
    print("   cd ai-principles-gym && python -m uvicorn src.api.app:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
