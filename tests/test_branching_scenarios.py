"""
Tests for branching scenario functionality.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any

from src.scenarios.branching import (
    BranchingScenario, ScenarioNode, Choice, DecisionPath,
    BranchingScenarioBuilder, BranchType,
    create_trust_building_scenario, create_resource_cascade_scenario
)
from src.scenarios.engine import ScenarioEngine, BranchingScenarioExecution
from src.scenarios.archetypes import ScenarioArchetype
from src.core.models import DecisionContext


class TestBranchingScenarioStructures:
    """Test branching scenario data structures."""
    
    def test_choice_availability(self):
        """Test choice requirement checking."""
        choice = Choice(
            id="test_choice",
            description="Test choice",
            requirements={
                "min_resource": ("gold", 50),
                "previous_choice": "choice_1",
                "principle_strength": ("fairness", 0.7),
                "relationship": ("partner", 0.5)
            }
        )
        
        # Context that meets all requirements
        valid_context = {
            "resources": {"gold": 100},
            "previous_choices": ["choice_1"],
            "principles": {"fairness": 0.8},
            "relationships": {"partner": 0.6}
        }
        assert choice.is_available(valid_context)
        
        # Context missing resource
        invalid_context_1 = {
            "resources": {"gold": 30},
            "previous_choices": ["choice_1"],
            "principles": {"fairness": 0.8},
            "relationships": {"partner": 0.6}
        }
        assert not choice.is_available(invalid_context_1)
        
        # Context missing previous choice
        invalid_context_2 = {
            "resources": {"gold": 100},
            "previous_choices": [],
            "principles": {"fairness": 0.8},
            "relationships": {"partner": 0.6}
        }
        assert not choice.is_available(invalid_context_2)
    
    def test_scenario_node_context_updates(self):
        """Test node context application."""
        node = ScenarioNode(
            description="Test node",
            context_updates={
                "resources": {"gold": -20, "reputation": 10},
                "relationships": {"partner": 0.2},
                "flags": {"made_deal": True}
            }
        )
        
        initial_context = {
            "resources": {"gold": 100, "reputation": 50},
            "relationships": {"partner": 0.3, "rival": 0.1},
            "flags": {"started": True}
        }
        
        updated = node.apply_context_updates(initial_context)
        
        assert updated["resources"]["gold"] == 80
        assert updated["resources"]["reputation"] == 60
        assert updated["relationships"]["partner"] == 0.5
        assert updated["relationships"]["rival"] == 0.1
        assert updated["flags"]["made_deal"] is True
        assert updated["flags"]["started"] is True
    
    def test_decision_path_tracking(self):
        """Test decision path accumulation."""
        path = DecisionPath()
        
        choice1 = Choice(
            id="choice_1",
            impacts={"gold": -10, "reputation": 5},
            principle_alignment={"fairness": 0.8, "trust": 0.6}
        )
        
        choice2 = Choice(
            id="choice_2", 
            impacts={"gold": 20, "reputation": -3},
            principle_alignment={"fairness": 0.7, "greed": 0.8}
        )
        
        path.add_decision("node_1", choice1)
        path.add_decision("node_2", choice2)
        
        assert len(path.decisions) == 2
        assert path.total_impacts["gold"] == 10
        assert path.total_impacts["reputation"] == 2
        assert len(path.principle_scores["fairness"]) == 2
        assert path.principle_scores["fairness"] == [0.8, 0.7]
        
        # Test consistency calculation
        consistency = path.get_consistency_score()
        assert 0 <= consistency <= 1
    
    def test_branching_scenario_builder(self):
        """Test scenario builder functionality."""
        builder = BranchingScenarioBuilder()
        
        # Build a simple scenario
        root_choices = [
            Choice(id="left", description="Go left"),
            Choice(id="right", description="Go right")
        ]
        
        left_choices = [
            Choice(id="continue", description="Continue forward"),
            Choice(id="return", description="Go back")
        ]
        
        scenario = (builder
            .set_metadata(
                name="Test Scenario",
                archetype=ScenarioArchetype.TRADEOFFS,
                initial_context={"position": "start"},
                expected_principles=["exploration", "caution"]
            )
            .add_root_node("You're at a crossroads", root_choices)
            .add_branch(
                "left",
                "You went left and found a forest",
                BranchType.CONSEQUENCE,
                left_choices,
                is_terminal=False
            )
            .build()
        )
        
        assert scenario.name == "Test Scenario"
        assert scenario.root is not None
        assert len(scenario.nodes) == 2
        assert scenario.max_depth == 1
        assert scenario.total_paths == 3  # right + left->continue + left->return


class TestBranchingScenarioExecution:
    """Test branching scenario execution in engine."""
    
    @pytest.mark.asyncio
    async def test_create_branching_scenario(self):
        """Test creating branching scenarios."""
        engine = ScenarioEngine()
        
        # Create trust building scenario
        execution = await engine.create_branching_scenario("trust_building")
        assert execution.scenario is not None
        assert execution.scenario.name == "Trust Building Collaboration"
        assert execution.state.value == "initialized"
        
        # Create resource cascade scenario
        execution2 = await engine.create_branching_scenario("resource_cascade")
        assert execution2.scenario.name == "Cascading Crisis Management"
        
        # Test invalid type
        with pytest.raises(ValueError):
            await engine.create_branching_scenario("invalid_type")
    
    @pytest.mark.asyncio
    async def test_present_branching_scenario(self):
        """Test presenting branching scenario nodes."""
        engine = ScenarioEngine()
        execution = await engine.create_branching_scenario("trust_building")
        
        # Present initial node
        presentation = await engine.present_branching_scenario(
            execution.execution_id,
            "test_agent"
        )
        
        assert "execution_id" in presentation
        assert "description" in presentation
        assert "choices" in presentation
        assert len(presentation["choices"]) > 0
        assert presentation["path_depth"] == 0
        assert not presentation["is_terminal"]
        
        # Verify context was initialized
        assert "resources" in presentation["context"]
        assert "relationships" in presentation["context"]
    
    @pytest.mark.asyncio 
    async def test_record_branching_response(self):
        """Test recording responses in branching scenarios."""
        engine = ScenarioEngine()
        execution = await engine.create_branching_scenario("trust_building")
        
        # Present initial node
        presentation = await engine.present_branching_scenario(
            execution.execution_id,
            "test_agent"
        )
        
        # Make a choice
        choice_id = presentation["choices"][0]["id"]
        result = await engine.record_branching_response(
            execution.execution_id,
            choice_id
        )
        
        # Should either present next node or complete
        assert result["status"] in ["completed", "error"] or "choices" in result
        
        # Test invalid choice
        invalid_result = await engine.record_branching_response(
            execution.execution_id,
            "invalid_choice_id"
        )
        assert invalid_result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_branching_scenario_completion(self):
        """Test completing a branching scenario."""
        engine = ScenarioEngine()
        
        # Create a simple test scenario that completes quickly
        builder = BranchingScenarioBuilder()
        scenario = (builder
            .set_metadata(
                name="Quick Test",
                archetype=ScenarioArchetype.LOYALTY,
                initial_context={},
                expected_principles=["test"]
            )
            .add_root_node(
                "Test scenario",
                [Choice(id="end", description="End scenario")]
            )
            .build()
        )
        
        # Manually create execution with our test scenario
        execution = BranchingScenarioExecution(
            scenario=scenario,
            decision_path=DecisionPath()
        )
        execution.current_context = scenario.initial_context.copy()
        engine.active_branching_executions[execution.execution_id] = execution
        
        # Present and complete
        await engine.present_branching_scenario(execution.execution_id, "test_agent")
        
        # Make the terminal choice
        scenario.root.choices[0].next_node_id = None  # Make it terminal
        scenario.root.is_terminal = True
        
        result = await engine.record_branching_response(
            execution.execution_id,
            "end"
        )
        
        assert result["status"] == "completed"
        assert "consistency_score" in result["outcome"]
        assert "path_depth" in result["outcome"]
        assert "decision_path" in result["outcome"]
        
        # Verify metrics were updated
        assert engine.metrics["completed_scenarios"] > 0
        assert engine.metrics["branching_scenarios"] > 0
    
    @pytest.mark.asyncio
    async def test_context_based_choices(self):
        """Test that choices are filtered based on context."""
        engine = ScenarioEngine()
        
        # Create scenario with conditional choices
        builder = BranchingScenarioBuilder()
        
        restricted_choice = Choice(
            id="restricted",
            description="Need resources",
            requirements={"min_resource": ("gold", 100)}
        )
        
        open_choice = Choice(
            id="open",
            description="Always available"
        )
        
        scenario = (builder
            .set_metadata(
                name="Conditional Test",
                archetype=ScenarioArchetype.SCARCITY,
                initial_context={"resources": {"gold": 50}},
                expected_principles=["resource_management"]
            )
            .add_root_node(
                "Choose your path",
                [restricted_choice, open_choice]
            )
            .build()
        )
        
        execution = BranchingScenarioExecution(
            scenario=scenario,
            decision_path=DecisionPath()
        )
        execution.current_context = scenario.initial_context.copy()
        engine.active_branching_executions[execution.execution_id] = execution
        
        # Present scenario
        presentation = await engine.present_branching_scenario(
            execution.execution_id,
            "test_agent"
        )
        
        # Only open choice should be available
        assert len(presentation["choices"]) == 1
        assert presentation["choices"][0]["id"] == "open"
        
        # Try to make restricted choice (should fail)
        result = await engine.record_branching_response(
            execution.execution_id,
            "restricted"
        )
        assert result["status"] == "error"
        assert "not available" in result["message"]


class TestPrebuiltScenarios:
    """Test the prebuilt branching scenarios."""
    
    def test_trust_building_scenario_structure(self):
        """Test trust building scenario structure."""
        scenario = create_trust_building_scenario()
        
        assert scenario.name == "Trust Building Collaboration"
        assert scenario.archetype == ScenarioArchetype.LOYALTY
        assert scenario.root is not None
        
        # Test initial choices
        root_choices = scenario.root.choices
        assert len(root_choices) == 3  # accept, decline, negotiate
        
        # Test that choices lead to different paths
        choice_ids = {c.id for c in root_choices}
        assert "accept" in choice_ids
        assert "decline" in choice_ids
        assert "negotiate" in choice_ids
        
        # Verify scenario has multiple paths
        assert scenario.total_paths > 3
        
        # Check expected principles
        assert "fairness" in scenario.expected_principles
        assert "reciprocity" in scenario.expected_principles
    
    def test_resource_cascade_scenario_structure(self):
        """Test resource cascade scenario structure."""
        scenario = create_resource_cascade_scenario()
        
        assert scenario.name == "Cascading Crisis Management"
        assert scenario.archetype == ScenarioArchetype.SCARCITY
        
        # Test initial context has resources
        assert "food" in scenario.initial_context["resources"]
        assert "medicine" in scenario.initial_context["resources"]
        
        # Test cascading structure exists
        assert scenario.max_depth >= 2  # At least 2 levels of decisions
        
        # Check that choices have resource impacts
        root_choices = scenario.root.choices
        for choice in root_choices:
            assert "food" in choice.impacts or "medicine" in choice.impacts


class TestBranchingIntegration:
    """Test integration between branching scenarios and core systems."""
    
    @pytest.mark.asyncio
    async def test_behavioral_tracking_integration(self):
        """Test that branching choices are tracked as actions."""
        from src.core.tracking import BehavioralTracker
        from src.core.database import DatabaseManager
        
        # Create mock database
        db = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        
        tracker = BehavioralTracker(db)
        engine = ScenarioEngine(behavioral_tracker=tracker)
        
        # Create and run scenario
        execution = await engine.create_branching_scenario("trust_building")
        await engine.present_branching_scenario(execution.execution_id, "test_agent")
        
        # Get available choices
        current_node = execution.get_current_node()
        choice_id = current_node.choices[0].id
        
        # Make choice
        await engine.record_branching_response(execution.execution_id, choice_id)
        
        # Verify action was tracked
        profile = await tracker.get_agent_profile("test_agent")
        assert profile is not None
        assert profile.total_actions > 0
        
        # Check action metadata
        actions = await db.get_agent_actions("test_agent", limit=10)
        assert len(actions) > 0
        assert actions[0].action_type.startswith("branching_choice_")
    
    @pytest.mark.asyncio
    async def test_principle_consistency_tracking(self):
        """Test that principle consistency is tracked across branches."""
        engine = ScenarioEngine()
        
        # Create scenario with multiple principle-testing choices
        builder = BranchingScenarioBuilder()
        
        choices_1 = [
            Choice(
                id="fair",
                description="Fair choice",
                principle_alignment={"fairness": 0.9, "greed": 0.1}
            ),
            Choice(
                id="selfish",
                description="Selfish choice",
                principle_alignment={"fairness": 0.1, "greed": 0.9}
            )
        ]
        
        choices_2 = [
            Choice(
                id="fair_again",
                description="Another fair choice",
                principle_alignment={"fairness": 0.8, "greed": 0.2}
            ),
            Choice(
                id="selfish_again",
                description="Another selfish choice",
                principle_alignment={"fairness": 0.2, "greed": 0.8}
            )
        ]
        
        scenario = (builder
            .set_metadata(
                name="Consistency Test",
                archetype=ScenarioArchetype.TRADEOFFS,
                initial_context={},
                expected_principles=["fairness", "greed"]
            )
            .add_root_node("First choice", choices_1)
            .add_branch(
                "fair",
                "Second choice after being fair",
                BranchType.CONSEQUENCE,
                choices_2,
                is_terminal=True
            )
            .add_branch(
                "selfish",
                "Second choice after being selfish",
                BranchType.CONSEQUENCE,
                choices_2,
                is_terminal=True
            )
            .build()
        )
        
        execution = BranchingScenarioExecution(
            scenario=scenario,
            decision_path=DecisionPath()
        )
        execution.current_context = scenario.initial_context.copy()
        engine.active_branching_executions[execution.execution_id] = execution
        
        # Make consistent fair choices
        await engine.present_branching_scenario(execution.execution_id, "test_agent")
        await engine.record_branching_response(execution.execution_id, "fair")
        result = await engine.record_branching_response(execution.execution_id, "fair_again")
        
        # Check consistency score
        assert result["status"] == "completed"
        consistency = result["outcome"]["consistency_score"]
        assert consistency > 0.5  # Should be high for consistent choices
        
        # Check principle scores
        principle_scores = result["outcome"]["principle_scores"]
        assert "fairness" in principle_scores
        assert principle_scores["fairness"]["mean"] > 0.7
        assert principle_scores["fairness"]["consistency"] > 0.5
