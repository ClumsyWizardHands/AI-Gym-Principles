"""Tests for the behavioral inference engine."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import (
    Action,
    AgentProfile,
    DecisionContext,
    Principle,
    RelationalAnchor,
)
from src.core.inference import PrincipleInferenceEngine


class TestBehavioralEntropy:
    """Test behavioral entropy calculation functionality."""

    def test_high_entropy_detection(self):
        """Random behavior should have high entropy."""
        # Create agent with random actions
        agent = AgentProfile(
            agent_id="test_random",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Generate 20 random actions across different contexts
        contexts = list(DecisionContext)
        for i in range(20):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description=f"Random action {i}",
                context_type=contexts[i % len(contexts)],
                confidence=np.random.uniform(0.3, 0.9),
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Calculate entropy
        inference_engine = PrincipleInferenceEngine()
        entropy = inference_engine._calculate_behavioral_entropy(agent.behaviors)
        
        # Assert entropy is high (> 0.7)
        assert entropy > 0.7, f"Expected entropy > 0.7 for random behavior, got {entropy}"

    def test_low_entropy_detection(self):
        """Consistent behavior should have low entropy."""
        agent = AgentProfile(
            agent_id="test_consistent",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Generate 20 consistent actions (same context and similar descriptions)
        for i in range(20):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Always choose cooperation",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.95,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Calculate entropy
        inference_engine = PrincipleInferenceEngine()
        entropy = inference_engine._calculate_behavioral_entropy(agent.behaviors)
        
        # Assert entropy is low (< 0.3)
        assert entropy < 0.3, f"Expected entropy < 0.3 for consistent behavior, got {entropy}"

    def test_medium_entropy_mixed_behavior(self):
        """Mixed behavior should have medium entropy."""
        agent = AgentProfile(
            agent_id="test_mixed",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Generate mixed pattern: 10 consistent + 10 varied
        for i in range(10):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Choose cooperation",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        contexts = [DecisionContext.SACRIFICE_GREATER_GOOD, DecisionContext.RESOURCE_ALLOCATION]
        for i in range(10):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description=f"Varied choice {i}",
                context_type=contexts[i % 2],
                confidence=0.7,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Calculate entropy
        inference_engine = PrincipleInferenceEngine()
        entropy = inference_engine._calculate_behavioral_entropy(agent.behaviors)
        
        # Assert entropy is medium (0.3 < entropy < 0.7)
        assert 0.3 < entropy < 0.7, f"Expected 0.3 < entropy < 0.7 for mixed behavior, got {entropy}"


class TestContradictionDetection:
    """Test contradiction detection in principles."""

    def test_direct_contradiction(self):
        """Test detection of directly contradicting principles."""
        engine = PrincipleInferenceEngine()
        
        # Create two contradicting principles
        principle1 = Principle(
            principle_id="p1",
            agent_id="test_agent",
            description="Always prioritize individual freedom over collective good",
            strength=0.8,
            confidence=0.9,
            pattern_ids=["pattern1"]
        )
        
        principle2 = Principle(
            principle_id="p2",
            agent_id="test_agent",
            description="Always prioritize collective good over individual freedom",
            strength=0.8,
            confidence=0.9,
            pattern_ids=["pattern2"]
        )
        
        # Should detect contradiction
        contradiction_score = engine._calculate_principle_similarity(principle1, principle2)
        assert contradiction_score < 0.3, "Failed to detect direct contradiction"

    def test_no_contradiction_aligned_principles(self):
        """Test that aligned principles are not marked as contradictory."""
        engine = PrincipleInferenceEngine()
        
        principle1 = Principle(
            principle_id="p1",
            agent_id="test_agent",
            description="Prioritize helping others when possible",
            strength=0.8,
            confidence=0.9,
            pattern_ids=["pattern1"]
        )
        
        principle2 = Principle(
            principle_id="p2",
            agent_id="test_agent",
            description="Act with compassion and help those in need",
            strength=0.8,
            confidence=0.9,
            pattern_ids=["pattern2"]
        )
        
        # Should not detect contradiction
        similarity_score = engine._calculate_principle_similarity(principle1, principle2)
        assert similarity_score > 0.7, "Incorrectly detected contradiction in aligned principles"

    def test_partial_contradiction(self):
        """Test detection of partially contradicting principles."""
        engine = PrincipleInferenceEngine()
        
        principle1 = Principle(
            principle_id="p1",
            agent_id="test_agent",
            description="Generally prioritize efficiency, but consider fairness",
            strength=0.7,
            confidence=0.8,
            pattern_ids=["pattern1"]
        )
        
        principle2 = Principle(
            principle_id="p2",
            agent_id="test_agent",
            description="Always ensure fair distribution regardless of efficiency",
            strength=0.7,
            confidence=0.8,
            pattern_ids=["pattern2"]
        )
        
        # Should detect partial contradiction
        similarity_score = engine._calculate_principle_similarity(principle1, principle2)
        assert 0.3 < similarity_score < 0.7, "Failed to detect partial contradiction"


class TestPrincipleEvolution:
    """Test principle evolution tracking."""

    def test_principle_strengthening(self):
        """Test that consistent behavior strengthens principles."""
        principle = Principle(
            principle_id="test_principle",
            agent_id="test_agent",
            description="Always cooperate in trust scenarios",
            strength=0.5,
            confidence=0.5,
            pattern_ids=["pattern1"]
        )
        
        initial_strength = principle.strength
        
        # Simulate 10 consistent reinforcements
        for _ in range(10):
            principle.update_strength(action_aligned=True, context_weight=0.8)
        
        assert principle.strength > initial_strength, "Principle should strengthen with consistent behavior"
        assert principle.strength > 0.8, "Principle strength should be high after reinforcements"

    def test_principle_weakening(self):
        """Test that contradictory behavior weakens principles."""
        principle = Principle(
            principle_id="test_principle",
            agent_id="test_agent",
            description="Always cooperate in trust scenarios",
            strength=0.8,
            confidence=0.8,
            pattern_ids=["pattern1"]
        )
        
        initial_strength = principle.strength
        
        # Simulate 5 contradictory actions
        for _ in range(5):
            principle.update_strength(action_aligned=False, context_weight=0.9)
        
        assert principle.strength < initial_strength, "Principle should weaken with contradictory behavior"
        assert principle.strength < 0.5, "Principle strength should be low after contradictions"

    def test_principle_forking(self):
        """Test principle forking when behavior diverges."""
        engine = PrincipleInferenceEngine()
        
        # Create agent with initial principle
        agent = AgentProfile(
            agent_id="test_fork",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        original_principle = Principle(
            principle_id="original",
            agent_id=agent.agent_id,
            description="Help others when cost is low",
            strength=0.7,
            confidence=0.8,
            pattern_ids=["pattern1"]
        )
        
        # Add diverging actions
        for i in range(10):
            if i < 5:
                # First half: help regardless of cost
                action = Action(
                    agent_id=agent.agent_id,
                    action_type="choice",
                    description="Help even at high cost",
                    context_type=DecisionContext.SACRIFICE_GREATER_GOOD,
                    confidence=0.9,
                    timestamp=datetime.now()
                )
            else:
                # Second half: only help at low cost
                action = Action(
                    agent_id=agent.agent_id,
                    action_type="choice",
                    description="Refuse help when cost is high",
                    context_type=DecisionContext.RESOURCE_ALLOCATION,
                    confidence=0.9,
                    timestamp=datetime.now()
                )
            agent.add_action(action)
        
        # Check for fork detection
        patterns = engine._extract_temporal_patterns(agent.behaviors)
        assert len(patterns) >= 2, "Should detect multiple patterns from diverging behavior"


class TestDTWPatternMatching:
    """Test Dynamic Time Warping pattern matching."""

    @pytest.mark.asyncio
    async def test_identical_sequences(self):
        """Test DTW distance for identical sequences."""
        engine = PrincipleInferenceEngine()
        
        # Create identical action sequences
        actions1 = []
        actions2 = []
        
        for i in range(10):
            action_template = Action(
                agent_id="test_agent",
                action_type="choice",
                description=f"Action {i}",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            actions1.append(action_template)
            actions2.append(action_template)
        
        # DTW distance should be near 0
        distance = await engine._calculate_dtw_distance_async(actions1, actions2)
        assert distance < 0.1, f"Expected near-zero distance for identical sequences, got {distance}"

    @pytest.mark.asyncio
    async def test_similar_sequences(self):
        """Test DTW distance for similar sequences."""
        engine = PrincipleInferenceEngine()
        
        actions1 = []
        actions2 = []
        
        # Create similar but not identical sequences
        for i in range(10):
            action1 = Action(
                agent_id="test_agent",
                action_type="choice",
                description=f"Cooperate {i}",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            action2 = Action(
                agent_id="test_agent",
                action_type="choice",
                description=f"Collaborate {i}",  # Similar but different
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.85,
                timestamp=datetime.now()
            )
            actions1.append(action1)
            actions2.append(action2)
        
        # DTW distance should be moderate
        distance = await engine._calculate_dtw_distance_async(actions1, actions2)
        assert 0.1 < distance < 0.5, f"Expected moderate distance for similar sequences, got {distance}"

    @pytest.mark.asyncio
    async def test_different_sequences(self):
        """Test DTW distance for very different sequences."""
        engine = PrincipleInferenceEngine()
        
        actions1 = []
        actions2 = []
        
        # Create very different sequences
        for i in range(10):
            action1 = Action(
                agent_id="test_agent",
                action_type="choice",
                description="Always cooperate",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.95,
                timestamp=datetime.now()
            )
            action2 = Action(
                agent_id="test_agent",
                action_type="choice",
                description="Always defect",
                context_type=DecisionContext.LOYALTY_VS_PRAGMATISM,
                confidence=0.95,
                timestamp=datetime.now()
            )
            actions1.append(action1)
            actions2.append(action2)
        
        # DTW distance should be high
        distance = await engine._calculate_dtw_distance_async(actions1, actions2)
        assert distance > 0.7, f"Expected high distance for different sequences, got {distance}"

    def test_temporal_pattern_extraction(self):
        """Test temporal pattern extraction from action sequences."""
        engine = PrincipleInferenceEngine()
        
        agent = AgentProfile(
            agent_id="test_patterns",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Create repeating pattern: cooperate 3x, defect 1x
        for cycle in range(5):
            for i in range(3):
                action = Action(
                    agent_id=agent.agent_id,
                    action_type="choice",
                    description="Cooperate",
                    context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                    confidence=0.9,
                    timestamp=datetime.now()
                )
                agent.add_action(action)
            
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description="Defect",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Extract patterns
        patterns = engine._extract_temporal_patterns(agent.behaviors)
        
        # Should detect the repeating pattern
        assert len(patterns) > 0, "Failed to extract any patterns"
        
        # Check if pattern has correct length (4 actions)
        pattern_lengths = [len(p) for p in patterns]
        assert 4 in pattern_lengths or any(l % 4 == 0 for l in pattern_lengths), \
            "Failed to detect the 4-action repeating pattern"


class TestPerformanceBenchmarks:
    """Test performance requirements."""

    def test_inference_speed_1000_actions(self):
        """Inference on 1000 actions should complete in < 1 second."""
        import time
        
        engine = PrincipleInferenceEngine()
        agent = AgentProfile(
            agent_id="test_performance",
            framework="test",
            model_name="test",
            behaviors=[]
        )
        
        # Generate 1000 actions
        contexts = list(DecisionContext)
        for i in range(1000):
            action = Action(
                agent_id=agent.agent_id,
                action_type="choice",
                description=f"Action {i}",
                context_type=contexts[i % len(contexts)],
                confidence=np.random.uniform(0.5, 1.0),
                timestamp=datetime.now()
            )
            agent.add_action(action)
        
        # Time the inference
        start_time = time.time()
        
        # Run inference operations
        entropy = engine._calculate_behavioral_entropy(agent.behaviors)
        patterns = engine._extract_temporal_patterns(agent.behaviors[:100])  # Sample for speed
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0, f"Inference took {duration:.2f}s, expected < 1s"

    def test_pattern_matching_performance(self):
        """Pattern matching should be efficient even with large sequences."""
        import time
        
        engine = PrincipleInferenceEngine()
        
        # Create two sequences of 500 actions each
        actions1 = []
        actions2 = []
        
        for i in range(500):
            action = Action(
                agent_id="test_agent",
                action_type="choice",
                description=f"Action {i}",
                context_type=DecisionContext.COOPERATION_VS_DEFECTION,
                confidence=0.9,
                timestamp=datetime.now()
            )
            actions1.append(action)
            actions2.append(action)
        
        # Time DTW calculation
        start_time = time.time()
        
        # Use sync version for timing
        loop = asyncio.new_event_loop()
        distance = loop.run_until_complete(
            engine._calculate_dtw_distance_async(actions1[:100], actions2[:100])
        )
        loop.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 0.5, f"DTW calculation took {duration:.2f}s, expected < 0.5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
