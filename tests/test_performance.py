"""Performance benchmark tests for AI Principles Gym."""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
from httpx import AsyncClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import Action, AgentProfile, DecisionContext
from src.core.tracking import BehaviorTracker
from src.core.inference import PrincipleInferenceEngine
from src.core.database import DatabaseManager
from src.scenarios.engine import ScenarioEngine
from src.scenarios.archetypes import ScenarioArchetype, generate_scenario_from_archetype
from src.api.app import app


class TestInferencePerformance:
    """Test inference engine performance benchmarks."""

    def test_entropy_calculation_performance(self):
        """Test that entropy calculation on 1000 actions completes in < 1 second."""
        inference_engine = PrincipleInferenceEngine()
        
        # Create large action set
        actions = []
        contexts = list(DecisionContext)
        
        for i in range(1000):
            action = Action(
                agent_id="perf_test",
                action_type="choice",
                description=f"Action {i}",
                context_type=contexts[i % len(contexts)],
                confidence=np.random.uniform(0.3, 1.0),
                timestamp=datetime.now()
            )
            actions.append(action)
        
        # Time entropy calculation
        start_time = time.time()
        entropy = inference_engine._calculate_behavioral_entropy(actions)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 1.0, f"Entropy calculation took {duration:.3f}s, expected < 1s"
        assert 0 <= entropy <= 1.0, "Entropy should be normalized between 0 and 1"

    @pytest.mark.asyncio
    async def test_dtw_performance_scaling(self):
        """Test DTW performance with increasing sequence lengths."""
        inference_engine = PrincipleInferenceEngine()
        
        sequence_lengths = [10, 50, 100, 200, 500]
        max_times = [0.01, 0.05, 0.2, 0.8, 5.0]  # Expected max times
        
        for length, max_time in zip(sequence_lengths, max_times):
            # Create sequences
            seq1 = []
            seq2 = []
            
            for i in range(length):
                action = Action(
                    agent_id="dtw_test",
                    action_type="choice",
                    description=f"Action {i}",
                    context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                    confidence=0.8,
                    timestamp=datetime.now()
                )
                seq1.append(action)
                seq2.append(action)
            
            # Time DTW calculation
            start_time = time.time()
            distance = await inference_engine._calculate_dtw_distance_async(seq1, seq2)
            end_time = time.time()
            
            duration = end_time - start_time
            assert duration < max_time, \
                f"DTW for {length} actions took {duration:.3f}s, expected < {max_time}s"

    def test_pattern_extraction_performance(self):
        """Test pattern extraction performance with large action sets."""
        inference_engine = PrincipleInferenceEngine()
        
        # Create agent with many actions
        agent = AgentProfile(
            agent_id="pattern_perf",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Add 5000 actions with patterns
        for i in range(5000):
            # Create repeating pattern every 10 actions
            pattern_phase = i % 10
            if pattern_phase < 7:
                description = "Cooperate"
                context = DecisionContext.COOPERATION_VS_DEFECTION
            else:
                description = "Defect"
                context = DecisionContext.BETRAYAL_EXPLOITATION
            
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description=description,
                context_type=context,
                confidence=0.85,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Time pattern extraction
        start_time = time.time()
        patterns = inference_engine._extract_temporal_patterns(agent.behaviors[:1000])
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 2.0, f"Pattern extraction took {duration:.3f}s, expected < 2s"
        assert len(patterns) > 0, "Should extract at least one pattern"


class TestScenarioPerformance:
    """Test scenario generation and execution performance."""

    def test_scenario_generation_benchmark(self):
        """Scenario generation should complete in < 10ms."""
        total_time = 0
        num_iterations = 1000
        
        for i in range(num_iterations):
            archetype = list(ScenarioArchetype)[i % len(ScenarioArchetype)]
            
            start_time = time.time()
            scenario = generate_scenario_from_archetype(archetype)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            # Validate scenario
            assert scenario.scenario_id is not None
            assert len(scenario.options) >= 2
        
        avg_time = (total_time / num_iterations) * 1000  # Convert to ms
        assert avg_time < 10, f"Average scenario generation took {avg_time:.2f}ms, expected < 10ms"

    def test_concurrent_scenario_execution(self):
        """Test performance with many concurrent scenarios."""
        engine = ScenarioEngine()
        num_concurrent = 100
        
        start_time = time.time()
        
        # Create and execute scenarios
        for i in range(num_concurrent):
            agent_id = f"concurrent_{i}"
            scenario = generate_scenario_from_archetype(
                ScenarioArchetype(i % len(ScenarioArchetype))
            )
            
            # Present scenario
            execution = engine.present_scenario(agent_id, scenario)
            
            # Immediate response
            action = Action(
                agent_id=agent_id,
                action_type="scenario_response",
                description="Quick response",
                context_type=scenario.context,
                confidence=0.8,
                timestamp=datetime.now()
            )
            
            engine.process_response(
                agent_id=agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Processing {num_concurrent} scenarios took {duration:.2f}s"
        
        # Verify cleanup
        active_count = sum(len(s) for s in engine._active_scenarios.values())
        assert active_count == 0, "All scenarios should be completed and cleaned up"


class TestAPIPerformance:
    """Test API response time benchmarks."""

    @pytest.mark.asyncio
    async def test_api_response_times(self):
        """Test that API endpoints meet response time requirements."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test health endpoint (should be < 50ms)
            start = time.time()
            response = await client.get("/health")
            duration = (time.time() - start) * 1000
            
            assert response.status_code == 200
            assert duration < 50, f"Health check took {duration:.1f}ms, expected < 50ms"
            
            # Test metrics endpoint (should be < 100ms)
            start = time.time()
            response = await client.get("/metrics")
            duration = (time.time() - start) * 1000
            
            assert response.status_code == 200
            assert duration < 100, f"Metrics endpoint took {duration:.1f}ms, expected < 100ms"

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        """Test API performance under concurrent load."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            num_requests = 50
            
            async def make_request(i: int):
                start = time.time()
                response = await client.get("/health")
                duration = (time.time() - start) * 1000
                return response.status_code, duration
            
            # Make concurrent requests
            start_time = time.time()
            results = await asyncio.gather(
                *[make_request(i) for i in range(num_requests)]
            )
            total_duration = time.time() - start_time
            
            # Verify all succeeded
            statuses = [r[0] for r in results]
            assert all(s == 200 for s in statuses), "All requests should succeed"
            
            # Check individual response times
            durations = [r[1] for r in results]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            assert avg_duration < 100, f"Average response time {avg_duration:.1f}ms, expected < 100ms"
            assert max_duration < 500, f"Max response time {max_duration:.1f}ms, expected < 500ms"
            assert total_duration < 2.0, f"Total time {total_duration:.2f}s, expected < 2s"

    @pytest.mark.asyncio
    async def test_training_endpoint_performance(self):
        """Test training endpoint returns quickly with async processing."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Generate API key first
            key_response = await client.post("/api/keys", json={})
            api_key = key_response.json()["key"]
            
            # Register agent
            register_data = {
                "agent_id": "perf_test_agent",
                "framework": "custom",
                "model_name": "test_model"
            }
            headers = {"X-API-Key": api_key}
            
            await client.post("/api/agents/register", json=register_data, headers=headers)
            
            # Start training (should return immediately)
            training_data = {
                "agent_id": "perf_test_agent",
                "framework": "custom",
                "model_name": "test_model",
                "num_scenarios": 100
            }
            
            start = time.time()
            response = await client.post("/api/training/start", json=training_data, headers=headers)
            duration = (time.time() - start) * 1000
            
            assert response.status_code == 202  # Accepted
            assert "session_id" in response.json()
            assert duration < 100, f"Training start took {duration:.1f}ms, expected < 100ms"


class TestDatabasePerformance:
    """Test database operation performance."""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self):
        """Test performance of bulk action inserts."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            
            # Create agent
            async with db_manager.get_session() as session:
                agent_data = {
                    "agent_id": "bulk_test",
                    "framework": "test",
                    "model_name": "test",
                    "behavioral_entropy": 0.5
                }
                await db_manager.create_agent(session, agent_data)
            
            # Time bulk insert of 10,000 actions
            start_time = time.time()
            
            for i in range(10000):
                action_data = {
                    "agent_id": "bulk_test",
                    "action_type": "choice",
                    "description": f"Action {i}",
                    "context_type": "cooperation_vs_defection",
                    "confidence": 0.8
                }
                await db_manager.record_action(action_data)
            
            # Force flush
            await db_manager.flush_action_buffer()
            
            end_time = time.time()
            duration = end_time - start_time
            
            assert duration < 5.0, f"Bulk insert of 10k actions took {duration:.2f}s, expected < 5s"
            
            # Verify all inserted
            async with db_manager.get_session() as session:
                from sqlalchemy import select, func
                from src.core.database import Action as DBAction
                
                result = await session.execute(
                    select(func.count(DBAction.id)).where(DBAction.agent_id == "bulk_test")
                )
                count = result.scalar()
                assert count == 10000, f"Expected 10000 actions, found {count}"
            
        finally:
            await db_manager.close()
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_query_performance(self):
        """Test database query performance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            
            # Create agents and actions
            agent_ids = [f"query_agent_{i}" for i in range(10)]
            
            for agent_id in agent_ids:
                async with db_manager.get_session() as session:
                    agent_data = {
                        "agent_id": agent_id,
                        "framework": "test",
                        "model_name": "test",
                        "behavioral_entropy": np.random.uniform(0.3, 0.9)
                    }
                    await db_manager.create_agent(session, agent_data)
                
                # Add 1000 actions per agent
                for j in range(1000):
                    action_data = {
                        "agent_id": agent_id,
                        "action_type": "choice",
                        "description": f"Action {j}",
                        "context_type": list(DecisionContext)[j % len(DecisionContext)].value,
                        "confidence": np.random.uniform(0.5, 1.0)
                    }
                    await db_manager.record_action(action_data)
            
            await db_manager.flush_action_buffer()
            
            # Test query performance
            query_times = []
            
            for agent_id in agent_ids:
                start = time.time()
                async with db_manager.get_session() as session:
                    actions = await db_manager.get_agent_actions(
                        session,
                        agent_id,
                        limit=100
                    )
                duration = time.time() - start
                query_times.append(duration)
                
                assert len(actions) == 100
            
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            
            assert avg_query_time < 0.1, f"Average query time {avg_query_time:.3f}s, expected < 0.1s"
            assert max_query_time < 0.2, f"Max query time {max_query_time:.3f}s, expected < 0.2s"
            
        finally:
            await db_manager.close()
            os.unlink(db_path)


class TestMemoryUsage:
    """Test memory efficiency and limits."""

    def test_action_buffer_memory_limit(self):
        """Test that action buffers respect memory limits."""
        tracker = BehaviorTracker()
        
        agent = AgentProfile(
            agent_id="memory_test",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        tracker.track_agent(agent)
        
        # Add more than max actions (10,000)
        for i in range(15000):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description=f"Action {i}",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.8,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Should be capped at 10,000
        assert len(agent.behaviors) == 10000, "Action buffer should be capped at 10,000"
        
        # Most recent actions should be kept
        assert agent.behaviors[-1].description == "Action 14999"

    def test_cache_memory_efficiency(self):
        """Test that caches don't grow unbounded."""
        inference_engine = PrincipleInferenceEngine()
        
        # Generate many unique action sequences
        for i in range(100):
            seq1 = []
            seq2 = []
            
            for j in range(50):
                action = Action(
                    agent_id=f"cache_test_{i}",
                    action_type="choice",
                    description=f"Action {i}_{j}",
                    context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                    confidence=0.8,
                    timestamp=datetime.now()
                )
                seq1.append(action)
                seq2.append(action)
            
            # Calculate DTW (will be cached)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                inference_engine._calculate_dtw_distance_async(seq1, seq2)
            )
            loop.close()
        
        # Cache should have evicted old entries
        cache_info = inference_engine._dtw_cache.cache_info()
        assert cache_info.currsize <= 128, "DTW cache should not exceed max size"


class TestConcurrencyLimits:
    """Test system behavior at concurrency limits."""

    @pytest.mark.asyncio
    async def test_max_concurrent_agents(self):
        """Test system with maximum concurrent agents."""
        tracker = BehaviorTracker()
        num_agents = 1000
        
        # Create many agents
        agents = []
        for i in range(num_agents):
            agent = AgentProfile(
                agent_id=f"limit_agent_{i}",
                framework="test",
                model_name="test",
                behaviors=[]
            )
            tracker.track_agent(agent)
            agents.append(agent)
        
        # Record one action per agent concurrently
        async def record_action(agent: AgentProfile):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Test action",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.8,
                timestamp=datetime.now()
            )
            await tracker.record_action(action)
        
        start_time = time.time()
        await asyncio.gather(*[record_action(agent) for agent in agents])
        duration = time.time() - start_time
        
        # Should handle 1000 agents efficiently
        assert duration < 10.0, f"Recording actions for 1000 agents took {duration:.2f}s"
        
        # Verify all recorded
        for agent in agents[:10]:  # Check sample
            snapshot = tracker.get_snapshot(agent.agent_id)
            assert len(snapshot["actions"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
