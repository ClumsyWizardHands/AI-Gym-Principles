"""Tests for scenario generation and execution."""

import sys
from datetime import datetime
from pathlib import Path
from typing import List
import time

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import Action, AgentProfile, DecisionContext
from src.scenarios.archetypes import (
    ScenarioArchetype,
    ScenarioTemplate,
    generate_scenario_from_archetype,
)
from src.scenarios.engine import (
    ScenarioEngine,
    ScenarioExecution,
    ScenarioLifecycle,
    ScenarioOutcome,
)


class TestScenarioArchetypes:
    """Test scenario archetype coverage and generation."""

    def test_all_archetypes_represented(self):
        """Ensure all defined archetypes are available."""
        expected_archetypes = {
            ScenarioArchetype.LOYALTY,
            ScenarioArchetype.SCARCITY,
            ScenarioArchetype.BETRAYAL,
            ScenarioArchetype.TRADEOFFS,
            ScenarioArchetype.TIME_PRESSURE,
            ScenarioArchetype.OBEDIENCE_AUTONOMY,
            ScenarioArchetype.INFO_ASYMMETRY,
            ScenarioArchetype.REPUTATION_MGMT,
            ScenarioArchetype.POWER_DYNAMICS,
            ScenarioArchetype.MORAL_HAZARD,
        }
        
        actual_archetypes = set(ScenarioArchetype)
        assert actual_archetypes == expected_archetypes, \
            f"Missing archetypes: {expected_archetypes - actual_archetypes}"

    def test_scenario_generation_for_each_archetype(self):
        """Test that each archetype can generate valid scenarios."""
        for archetype in ScenarioArchetype:
            scenario = generate_scenario_from_archetype(archetype)
            
            # Validate scenario structure
            assert scenario.scenario_id is not None
            assert scenario.archetype == archetype
            assert scenario.description is not None and len(scenario.description) > 20
            assert len(scenario.options) >= 2
            assert scenario.context is not None
            assert scenario.stress_level >= 0.0 and scenario.stress_level <= 1.0

    def test_loyalty_scenario_specifics(self):
        """Test LOYALTY archetype generates appropriate scenarios."""
        scenario = generate_scenario_from_archetype(ScenarioArchetype.LOYALTY)
        
        # Should involve trust or loyalty concepts
        description_lower = scenario.description.lower()
        assert any(word in description_lower for word in 
                  ["trust", "loyal", "betray", "promise", "commitment"]), \
            "LOYALTY scenario should contain trust-related concepts"
        
        # Should have clear loyalty vs pragmatism options
        option_texts = [opt.lower() for opt in scenario.options.values()]
        assert len(option_texts) >= 2, "Should have at least 2 options"

    def test_scarcity_scenario_specifics(self):
        """Test SCARCITY archetype generates resource-based scenarios."""
        scenario = generate_scenario_from_archetype(ScenarioArchetype.SCARCITY)
        
        # Should involve resource concepts
        description_lower = scenario.description.lower()
        assert any(word in description_lower for word in 
                  ["resource", "scarce", "limited", "shortage", "allocate", "distribute"]), \
            "SCARCITY scenario should contain resource-related concepts"
        
        # Check constraints include resources
        assert "resources" in scenario.constraints or "resource" in scenario.description.lower()

    def test_time_pressure_scenario_specifics(self):
        """Test TIME_PRESSURE archetype includes urgency elements."""
        scenario = generate_scenario_from_archetype(ScenarioArchetype.TIME_PRESSURE)
        
        # Should involve time concepts
        description_lower = scenario.description.lower()
        assert any(word in description_lower for word in 
                  ["urgent", "immediate", "deadline", "time", "quick", "fast"]), \
            "TIME_PRESSURE scenario should contain urgency-related concepts"
        
        # Check constraints include time limits
        assert any("time" in constraint.lower() for constraint in scenario.constraints)


class TestStressProgression:
    """Test stress-based scenario adaptation."""

    def test_stress_affects_resources(self):
        """Test that higher stress reduces available resources."""
        low_stress_scenario = generate_scenario_from_archetype(
            ScenarioArchetype.RESOURCE_ALLOCATION,
            stress_level=0.2
        )
        high_stress_scenario = generate_scenario_from_archetype(
            ScenarioArchetype.RESOURCE_ALLOCATION,
            stress_level=0.9
        )
        
        # Higher stress should have more constraints
        assert len(high_stress_scenario.constraints) >= len(low_stress_scenario.constraints)

    def test_stress_progression_sequence(self):
        """Test generating scenarios with increasing stress."""
        engine = ScenarioEngine()
        agent = AgentProfile(
            agent_id="test_stress",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        stress_levels = []
        for i in range(5):
            scenario = engine.generate_adaptive_scenario(agent)
            stress_levels.append(scenario.stress_level)
            
            # Simulate poor performance to increase stress
            execution = ScenarioExecution(
                scenario=scenario,
                agent_id=agent.agent_id,
                start_time=datetime.now()
            )
            
            # Add action with low confidence
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Uncertain choice",
                context_type=DecisionContext.RESOURCE_ALLOCATION,
                confidence=0.4,  # Low confidence
                timestamp=datetime.now()
            )
            agent.add_action(action)
            
            outcome = engine._calculate_outcome(
                execution=execution,
                action=action,
                choice="option_a"
            )
            
            # Record outcome for next iteration
            engine._recent_performances[agent.agent_id].append(outcome.success_score)
        
        # Stress should generally increase with poor performance
        # Allow for some variation but trend should be upward
        avg_first_half = sum(stress_levels[:2]) / 2
        avg_second_half = sum(stress_levels[3:]) / 2
        assert avg_second_half > avg_first_half, \
            "Stress should increase with continued poor performance"


class TestAdversarialGeneration:
    """Test adversarial scenario generation."""

    def test_generate_adversarial_scenario(self):
        """Test generation of adversarial scenarios."""
        engine = ScenarioEngine()
        
        # Create agent with established principle
        agent = AgentProfile(
            agent_id="test_adversarial",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Add consistent cooperation actions
        for i in range(10):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Always cooperate",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.95,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Generate adversarial scenario
        scenario = engine.generate_adversarial_scenario(
            agent=agent,
            target_principle="cooperation"
        )
        
        # Should challenge cooperation
        assert scenario is not None
        assert any(word in scenario.description.lower() for word in 
                  ["betray", "defect", "compete", "against", "conflict"])

    def test_adversarial_targets_weak_principles(self):
        """Test that adversarial scenarios target weak principles."""
        engine = ScenarioEngine()
        agent = AgentProfile(
            agent_id="test_weak_target",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Add mixed behavior (weak principle)
        contexts = [DecisionContext.COOPERATION_VS_DEFECTION, 
                   DecisionContext.LOYALTY_VS_PRAGMATISM]
        for i in range(10):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Sometimes cooperate" if i % 2 == 0 else "Sometimes defect",
                context_type=contexts[i % 2],
                confidence=0.6,  # Medium confidence
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Generate adversarial scenario
        scenario = engine.generate_adversarial_scenario(agent)
        
        # Should generate a scenario (even without specific target)
        assert scenario is not None
        assert scenario.archetype in [
            ScenarioArchetype.BETRAYAL,
            ScenarioArchetype.LOYALTY,
            ScenarioArchetype.TRADEOFFS
        ]


class TestScenarioExecution:
    """Test scenario execution engine."""

    def test_scenario_lifecycle(self):
        """Test scenario progresses through lifecycle states."""
        engine = ScenarioEngine()
        scenario = generate_scenario_from_archetype(ScenarioArchetype.LOYALTY)
        
        # Start execution
        execution = engine.present_scenario("test_agent", scenario)
        assert execution.status == ScenarioLifecycle.PRESENTED
        assert execution.scenario.scenario_id == scenario.scenario_id
        
        # Process response
        action = Action(
            agent_id="test_agent",
            action_type="scenario_response",
            description="Choose to remain loyal",
            context_type=DecisionContext.LOYALTY_VS_PRAGMATISM,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        outcome = engine.process_response(
            agent_id="test_agent",
            scenario_id=scenario.scenario_id,
            action=action,
            choice="option_a"
        )
        
        assert outcome is not None
        assert outcome.scenario_id == scenario.scenario_id
        assert execution.status == ScenarioLifecycle.COMPLETED

    def test_timeout_handling(self):
        """Test scenario timeout after 5 minutes."""
        engine = ScenarioEngine()
        scenario = generate_scenario_from_archetype(ScenarioArchetype.TIME_PRESSURE)
        
        # Create execution with past start time
        execution = ScenarioExecution(
            scenario=scenario,
            agent_id="test_timeout",
            start_time=datetime.now()
        )
        
        # Manually set to past time
        import datetime as dt
        execution.start_time = datetime.now() - dt.timedelta(minutes=6)
        
        engine._active_scenarios["test_timeout"] = {scenario.scenario_id: execution}
        
        # Process timeout
        action = Action(
            agent_id="test_timeout",
            action_type="scenario_response",
            description="Late response",
            context_type=DecisionContext.TIME_PRESSURE,
            confidence=0.5,
            timestamp=datetime.now()
        )
        
        outcome = engine.process_response(
            agent_id="test_timeout",
            scenario_id=scenario.scenario_id,
            action=action,
            choice="option_a"
        )
        
        # Should be marked as timeout
        assert outcome.outcome_type == "timeout"
        assert outcome.success_score < 0.5  # Penalized for timeout

    def test_principle_alignment_scoring(self):
        """Test that choices aligned with principles score higher."""
        engine = ScenarioEngine()
        
        # Create agent with cooperation principle
        agent = AgentProfile(
            agent_id="test_alignment",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Establish cooperation pattern
        for i in range(5):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Cooperate",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Present cooperation scenario
        scenario = generate_scenario_from_archetype(ScenarioArchetype.LOYALTY)
        execution = engine.present_scenario(agent.agent_id, scenario)
        
        # Aligned response
        aligned_action = Action(
            agent_id=agent.agent_id,
            action_type="scenario_response",
            description="Choose cooperation",
            context_type=DecisionContext.COOPERATION_VS_DEFECTION,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        outcome = engine._calculate_outcome(
            execution=execution,
            action=aligned_action,
            choice="option_a"  # Assuming this is cooperative option
        )
        
        assert outcome.principle_alignment > 0.7, \
            "Aligned choice should have high principle alignment score"


class TestPerformanceBenchmarks:
    """Test performance requirements for scenarios."""

    def test_scenario_generation_speed(self):
        """Scenario generation should complete in < 10ms."""
        start_time = time.time()
        
        # Generate 100 scenarios
        for _ in range(100):
            archetype = list(ScenarioArchetype)[_ % len(ScenarioArchetype)]
            scenario = generate_scenario_from_archetype(archetype)
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to ms
        avg_time = duration / 100
        
        assert avg_time < 10, f"Scenario generation took {avg_time:.2f}ms average, expected < 10ms"

    def test_concurrent_scenario_handling(self):
        """Test handling multiple concurrent scenarios."""
        engine = ScenarioEngine()
        
        # Create 50 concurrent scenarios
        agent_ids = [f"agent_{i}" for i in range(50)]
        scenarios = []
        
        start_time = time.time()
        
        for agent_id in agent_ids:
            scenario = generate_scenario_from_archetype(
                ScenarioArchetype(hash(agent_id) % len(ScenarioArchetype))
            )
            execution = engine.present_scenario(agent_id, scenario)
            scenarios.append((agent_id, scenario))
        
        # Process all responses
        for agent_id, scenario in scenarios:
            action = Action(
                agent_id=agent_id,
                action_type="scenario_response",
                description="Response",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.8,
                timestamp=datetime.now()
            )
            
            outcome = engine.process_response(
                agent_id=agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
            
            assert outcome is not None
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Processing 50 scenarios took {duration:.2f}s, expected < 5s"

    def test_memory_efficiency(self):
        """Test that completed scenarios are cleaned up."""
        engine = ScenarioEngine()
        
        initial_scenario_count = len(engine._active_scenarios)
        
        # Create and complete 100 scenarios
        for i in range(100):
            agent_id = f"agent_{i}"
            scenario = generate_scenario_from_archetype(ScenarioArchetype.LOYALTY)
            
            # Present
            engine.present_scenario(agent_id, scenario)
            
            # Respond
            action = Action(
                agent_id=agent_id,
                action_type="scenario_response",
                description="Response",
                context_type=DecisionContext.LOYALTY_VS_PRAGMATISM,
                confidence=0.8,
                timestamp=datetime.now()
            )
            
            engine.process_response(
                agent_id=agent_id,
                scenario_id=scenario.scenario_id,
                action=action,
                choice="option_a"
            )
        
        # Check that scenarios are cleaned up
        final_scenario_count = sum(len(scenarios) for scenarios in engine._active_scenarios.values())
        
        # Should have cleaned up completed scenarios
        assert final_scenario_count == 0, \
            f"Expected 0 active scenarios after completion, found {final_scenario_count}"


class TestDiagnosticSequences:
    """Test diagnostic scenario sequences."""

    def test_generate_diagnostic_sequence(self):
        """Test generation of diagnostic sequences for specific principles."""
        engine = ScenarioEngine()
        
        # Generate diagnostic for cooperation
        sequence = engine.generate_diagnostic_sequence(
            target_principle="cooperation",
            num_scenarios=5
        )
        
        assert len(sequence) == 5
        
        # All scenarios should test cooperation
        for scenario in sequence:
            assert scenario.archetype in [
                ScenarioArchetype.LOYALTY,
                ScenarioArchetype.BETRAYAL,
                ScenarioArchetype.COOPERATION_VS_DEFECTION  # If exists
            ] or "cooperat" in scenario.description.lower()

    def test_diagnostic_sequence_variety(self):
        """Test that diagnostic sequences have variety."""
        engine = ScenarioEngine()
        
        sequence = engine.generate_diagnostic_sequence(
            target_principle="fairness",
            num_scenarios=10
        )
        
        # Should use multiple archetypes
        archetypes_used = set(scenario.archetype for scenario in sequence)
        assert len(archetypes_used) >= 3, \
            "Diagnostic sequence should use at least 3 different archetypes"
        
        # Should have varying stress levels
        stress_levels = [scenario.stress_level for scenario in sequence]
        assert min(stress_levels) < 0.3 and max(stress_levels) > 0.7, \
            "Diagnostic sequence should include both low and high stress scenarios"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
