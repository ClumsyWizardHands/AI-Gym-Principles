"""Integration tests for the full AI Principles Gym system."""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
import tempfile
import os

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import Action, AgentProfile, DecisionContext
from src.core.tracking import BehaviorTracker
from src.core.inference import PrincipleInferenceEngine
from src.core.database import DatabaseManager
from src.scenarios.engine import ScenarioEngine
from src.scenarios.archetypes import ScenarioArchetype, generate_scenario_from_archetype
from src.adapters.custom_adapter import CustomAdapter
from src.api.routes import TrainingRequest, TrainingSession


class TestFullTrainingPipeline:
    """Test the complete training pipeline from start to finish."""

    @pytest.mark.asyncio
    async def test_basic_training_flow(self):
        """Test basic training flow with scenario presentation and principle inference."""
        # Setup components
        tracker = BehaviorTracker()
        inference_engine = PrincipleInferenceEngine()
        scenario_engine = ScenarioEngine()
        
        # Create test agent
        agent = AgentProfile(
            agent_id="test_pipeline",
            framework="custom",
            model_name="test_model",
            behaviors=[]
        )
        
        # Track agent
        tracker.track_agent(agent)
        
        # Present 10 scenarios
        for i in range(10):
            # Generate scenario
            archetype = list(ScenarioArchetype)[i % len(ScenarioArchetype)]
            scenario = generate_scenario_from_archetype(archetype)
            
            # Present scenario
            execution = scenario_engine.present_scenario(agent.agent_id, scenario)
            
            # Simulate agent response
            action = Action(
                agent_id=agent.agent_id,
                action_type="scenario_response",
                description=f"Response to {scenario.archetype.value}",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.8 + (i * 0.01),  # Slightly increasing confidence
                timestamp=datetime.now()
            )
            
            # Record action
            await tracker.record_action(action)
            
            # Process response
            outcome = scenario_engine.process_response(
                agent_id=agent.agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
            
            assert outcome is not None
            assert outcome.scenario_id == scenario.scenario_id
        
        # Run inference
        await inference_engine.continuous_inference()
        
        # Check that patterns were extracted
        snapshot = tracker.get_snapshot(agent.agent_id)
        assert snapshot is not None
        assert len(snapshot["actions"]) == 10
        assert snapshot["entropy"] is not None
        
        # Check inference results
        assert len(inference_engine._principle_candidates) > 0 or \
               len(inference_engine._active_principles) > 0, \
               "Should have discovered some principle candidates"

    @pytest.mark.asyncio
    async def test_principle_evolution_over_time(self):
        """Test that principles evolve as agent behavior changes."""
        tracker = BehaviorTracker()
        inference_engine = PrincipleInferenceEngine()
        
        agent = AgentProfile(
            agent_id="test_evolution",
            framework="custom",
            model_name="test_model",
            behaviors=[]
        )
        
        tracker.track_agent(agent)
        
        # Phase 1: Cooperative behavior
        for i in range(20):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Cooperate with others",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            await tracker.record_action(action)
        
        # Run inference
        await inference_engine.continuous_inference()
        initial_principles = len(inference_engine._active_principles.get(agent.agent_id, []))
        
        # Phase 2: Shift to competitive behavior
        for i in range(20):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Compete aggressively",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            await tracker.record_action(action)
        
        # Run inference again
        await inference_engine.continuous_inference()
        
        # Check for principle evolution
        final_principles = inference_engine._active_principles.get(agent.agent_id, [])
        
        # Should have detected behavior change
        assert len(final_principles) >= initial_principles, \
            "Should maintain or discover new principles after behavior shift"

    @pytest.mark.asyncio
    async def test_scenario_adaptation(self):
        """Test that scenarios adapt based on agent performance."""
        tracker = BehaviorTracker()
        scenario_engine = ScenarioEngine()
        
        agent = AgentProfile(
            agent_id="test_adaptation",
            framework="custom",
            model_name="test_model",
            behaviors=[]
        )
        
        tracker.track_agent(agent)
        
        stress_levels = []
        
        # Simulate poor performance
        for i in range(5):
            scenario = scenario_engine.generate_adaptive_scenario(agent)
            stress_levels.append(scenario.stress_level)
            
            # Poor response (low confidence)
            action = Action(
                agent_id=agent.agent_id,
                action_type="scenario_response",
                description="Uncertain response",
                context_type=scenario.context,
                confidence=0.3,  # Low confidence
                timestamp=datetime.now()
            )
            
            await tracker.record_action(action)
            
            # Process with poor outcome
            execution = scenario_engine.present_scenario(agent.agent_id, scenario)
            outcome = scenario_engine.process_response(
                agent_id=agent.agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
            
            # Record poor performance
            scenario_engine._recent_performances[agent.agent_id].append(outcome.success_score)
        
        # Stress should increase
        assert stress_levels[-1] > stress_levels[0], \
            "Stress should increase with poor performance"
        
        # Now simulate good performance
        for i in range(5):
            scenario = scenario_engine.generate_adaptive_scenario(agent)
            stress_levels.append(scenario.stress_level)
            
            # Good response (high confidence)
            action = Action(
                agent_id=agent.agent_id,
                action_type="scenario_response",
                description="Confident response",
                context_type=scenario.context,
                confidence=0.95,  # High confidence
                timestamp=datetime.now()
            )
            
            await tracker.record_action(action)
            
            # Process with good outcome
            execution = scenario_engine.present_scenario(agent.agent_id, scenario)
            outcome = scenario_engine.process_response(
                agent_id=agent.agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
            
            scenario_engine._recent_performances[agent.agent_id].append(outcome.success_score)
        
        # Stress should stabilize or decrease
        assert stress_levels[-1] <= stress_levels[5], \
            "Stress should stabilize or decrease with good performance"


class TestMultiAgentConcurrency:
    """Test concurrent training of multiple agents."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_tracking(self):
        """Test tracking multiple agents concurrently."""
        tracker = BehaviorTracker()
        num_agents = 10
        actions_per_agent = 50
        
        # Create agents
        agents = []
        for i in range(num_agents):
            agent = AgentProfile(
                agent_id=f"concurrent_agent_{i}",
                framework="custom",
                model_name="test_model",
                behaviors=[]
            )
            tracker.track_agent(agent)
            agents.append(agent)
        
        # Record actions concurrently
        async def record_agent_actions(agent: AgentProfile):
            for j in range(actions_per_agent):
                action = Action(
                    agent_id=agent.agent_id,
                    action_type="choice",
                    description=f"Action {j}",
                    context_type=list(DecisionContext)[j % len(DecisionContext)],
                    confidence=0.5 + (j * 0.01),
                    timestamp=datetime.now()
                )
                await tracker.record_action(action)
                await asyncio.sleep(0.001)  # Small delay to simulate real timing
        
        # Start timing
        start_time = time.time()
        
        # Run all agents concurrently
        await asyncio.gather(*[record_agent_actions(agent) for agent in agents])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify all actions recorded
        for agent in agents:
            snapshot = tracker.get_snapshot(agent.agent_id)
            assert len(snapshot["actions"]) == actions_per_agent
        
        # Should complete reasonably fast
        assert duration < 5.0, f"Concurrent tracking took {duration:.2f}s, expected < 5s"

    @pytest.mark.asyncio
    async def test_concurrent_scenario_processing(self):
        """Test processing scenarios for multiple agents concurrently."""
        scenario_engine = ScenarioEngine()
        num_agents = 20
        scenarios_per_agent = 5
        
        agent_ids = [f"scenario_agent_{i}" for i in range(num_agents)]
        
        async def process_agent_scenarios(agent_id: str):
            scenarios_completed = 0
            
            for i in range(scenarios_per_agent):
                # Generate and present scenario
                scenario = generate_scenario_from_archetype(
                    list(ScenarioArchetype)[i % len(ScenarioArchetype)]
                )
                execution = scenario_engine.present_scenario(agent_id, scenario)
                
                # Simulate response
                action = Action(
                    agent_id=agent_id,
                    action_type="scenario_response",
                    description=f"Response {i}",
                    context_type=scenario.context,
                    confidence=0.8,
                    timestamp=datetime.now()
                )
                
                # Process response
                outcome = scenario_engine.process_response(
                    agent_id=agent_id,
                    scenario_id=scenario.scenario_id,
                    action=action,
                    choice="option_a"
                )
                
                if outcome:
                    scenarios_completed += 1
                
                await asyncio.sleep(0.01)  # Small delay
            
            return scenarios_completed
        
        # Process all agents concurrently
        start_time = time.time()
        
        results = await asyncio.gather(
            *[process_agent_scenarios(agent_id) for agent_id in agent_ids]
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify all scenarios completed
        assert all(count == scenarios_per_agent for count in results), \
            "All agents should complete all scenarios"
        
        # Should handle concurrent load efficiently
        assert duration < 10.0, \
            f"Concurrent scenario processing took {duration:.2f}s, expected < 10s"

    @pytest.mark.asyncio
    async def test_inference_with_multiple_agents(self):
        """Test principle inference with multiple agents."""
        tracker = BehaviorTracker()
        inference_engine = PrincipleInferenceEngine()
        num_agents = 5
        
        # Create agents with different behavior patterns
        patterns = [
            ("cooperative", DecisionContext.COOPERATION_VS_DEFECTION, "Always cooperate"),
            ("competitive", DecisionContext.COOPERATION_VS_DEFECTION, "Always compete"),
            ("fair", DecisionContext.RESOURCE_ALLOCATION, "Distribute equally"),
            ("efficient", DecisionContext.RESOURCE_ALLOCATION, "Maximize output"),
            ("loyal", DecisionContext.LOYALTY_VS_PRAGMATISM, "Stay loyal"),
        ]
        
        agents = []
        for i in range(num_agents):
            pattern_name, context, description = patterns[i]
            agent = AgentProfile(
                agent_id=f"inference_agent_{pattern_name}",
                framework="custom",
                model_name="test_model",
                behaviors=[]
            )
            tracker.track_agent(agent)
            agents.append((agent, context, description))
        
        # Generate actions for each agent
        for agent, context, description in agents:
            for j in range(30):
                action = Action(
                    agent_id=agent.agent_id,
                    action_type="choice",
                    description=description,
                    context_type=context,
                    confidence=0.85 + (j * 0.005),
                    timestamp=datetime.now()
                )
                await tracker.record_action(action)
        
        # Run inference
        await inference_engine.continuous_inference()
        
        # Each agent should have discovered principles
        for agent, _, _ in agents:
            agent_principles = inference_engine._active_principles.get(agent.agent_id, [])
            assert len(agent_principles) > 0 or \
                   agent.agent_id in inference_engine._principle_candidates, \
                   f"Agent {agent.agent_id} should have discovered principles"


class TestDatabaseTransactions:
    """Test database transaction handling."""

    @pytest.mark.asyncio
    async def test_database_persistence(self):
        """Test that data persists correctly to database."""
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Override database URL
            db_url = f"sqlite+aiosqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            
            # Create and save agent
            async with db_manager.get_session() as session:
                agent_data = {
                    "agent_id": "test_db_agent",
                    "framework": "custom",
                    "model_name": "test_model",
                    "behavioral_entropy": 0.5,
                    "metadata": {"test": "data"}
                }
                agent = await db_manager.create_agent(session, agent_data)
                assert agent.id is not None
            
            # Record actions
            for i in range(100):
                action_data = {
                    "agent_id": "test_db_agent",
                    "action_type": "choice",
                    "description": f"Action {i}",
                    "context_type": "cooperation_vs_defection",
                    "confidence": 0.8,
                    "metadata": {"index": i}
                }
                await db_manager.record_action(action_data)
            
            # Force flush
            await db_manager.flush_action_buffer()
            
            # Retrieve actions
            async with db_manager.get_session() as session:
                actions = await db_manager.get_agent_actions(
                    session, 
                    "test_db_agent",
                    limit=50
                )
                assert len(actions) == 50
                
                # Check ordering (most recent first)
                for i in range(1, len(actions)):
                    assert actions[i-1].timestamp >= actions[i].timestamp
            
            # Create principle
            async with db_manager.get_session() as session:
                principle_data = {
                    "principle_id": "test_principle",
                    "agent_id": "test_db_agent",
                    "description": "Test principle",
                    "strength": 0.8,
                    "confidence": 0.9,
                    "metadata": {"test": "principle"}
                }
                principle = await db_manager.create_principle(session, principle_data)
                assert principle.id is not None
            
            # Update principle
            async with db_manager.get_session() as session:
                updated = await db_manager.update_principle_strength(
                    session,
                    "test_principle",
                    0.95
                )
                assert updated.strength == 0.95
            
        finally:
            # Cleanup
            await db_manager.close()
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_transaction_rollback(self):
        """Test that failed transactions rollback properly."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            
            # Attempt transaction that will fail
            try:
                async with db_manager.get_session() as session:
                    # Create agent
                    agent_data = {
                        "agent_id": "rollback_test",
                        "framework": "custom",
                        "model_name": "test_model",
                        "behavioral_entropy": 0.5
                    }
                    agent = await db_manager.create_agent(session, agent_data)
                    
                    # Force an error by trying to create duplicate
                    duplicate = await db_manager.create_agent(session, agent_data)
                    
            except Exception:
                # Expected to fail
                pass
            
            # Verify rollback - agent should not exist
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                from src.core.database import AgentProfile as DBAgent
                
                result = await session.execute(
                    select(DBAgent).where(DBAgent.agent_id == "rollback_test")
                )
                agent = result.scalar_one_or_none()
                assert agent is None, "Transaction should have rolled back"
            
        finally:
            await db_manager.close()
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_concurrent_database_access(self):
        """Test concurrent database operations."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            
            # Create multiple agents
            agent_ids = [f"concurrent_db_{i}" for i in range(10)]
            
            async def create_agent_with_actions(agent_id: str):
                # Create agent
                async with db_manager.get_session() as session:
                    agent_data = {
                        "agent_id": agent_id,
                        "framework": "custom",
                        "model_name": "test_model",
                        "behavioral_entropy": 0.5
                    }
                    await db_manager.create_agent(session, agent_data)
                
                # Record actions
                for i in range(20):
                    action_data = {
                        "agent_id": agent_id,
                        "action_type": "choice",
                        "description": f"Action {i}",
                        "context_type": "cooperation_vs_defection",
                        "confidence": 0.8
                    }
                    await db_manager.record_action(action_data)
                
                return agent_id
            
            # Create all agents concurrently
            results = await asyncio.gather(
                *[create_agent_with_actions(aid) for aid in agent_ids]
            )
            
            # Force flush
            await db_manager.flush_action_buffer()
            
            # Verify all agents and actions created
            for agent_id in agent_ids:
                async with db_manager.get_session() as session:
                    actions = await db_manager.get_agent_actions(
                        session,
                        agent_id,
                        limit=30
                    )
                    assert len(actions) == 20, \
                        f"Agent {agent_id} should have 20 actions"
            
        finally:
            await db_manager.close()
            os.unlink(db_path)


class TestAPIIntegration:
    """Test API integration with core components."""

    @pytest.mark.asyncio
    async def test_training_session_flow(self):
        """Test complete training session through API models."""
        # Create training request
        request = TrainingRequest(
            agent_id="api_test_agent",
            framework="custom",
            model_name="test_model",
            num_scenarios=10,
            scenario_types=["loyalty", "scarcity", "tradeoffs"]
        )
        
        # Simulate training session
        session = TrainingSession(
            session_id="test_session_123",
            agent_id=request.agent_id,
            start_time=datetime.now(),
            status="in_progress",
            progress=0.0
        )
        
        # Update progress
        for i in range(11):
            session.progress = i / 10.0
            if i == 10:
                session.status = "completed"
                session.end_time = datetime.now()
            
            assert 0.0 <= session.progress <= 1.0
            assert session.status in ["in_progress", "completed", "failed"]
        
        # Verify completion
        assert session.status == "completed"
        assert session.end_time is not None
        assert session.progress == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
