"""Core behavioral tracking models for the AI Principles Gym.

These models capture WHO does WHAT to WHOM and WHY, enabling
pattern detection and principle inference from agent behaviors.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import json
from collections import deque

import structlog

logger = structlog.get_logger()


class DecisionContext(Enum):
    """Context categories with weights for decision importance."""
    
    TRUST_BETRAYAL = ("trust_betrayal", 2.0)  # High weight for ethical decisions
    EFFICIENCY_MAX = ("efficiency_max", 1.0)  # Standard weight for optimization
    POWER_DYNAMICS = ("power_dynamics", 1.7)  # Elevated weight for dominance
    RESOURCE_ALLOCATION = ("resource_allocation", 1.3)  # Moderate weight for resources
    COOPERATION_CONFLICT = ("cooperation_conflict", 1.5)  # Important for social dynamics
    RISK_REWARD = ("risk_reward", 1.2)  # Slightly elevated for strategic choices
    FAIRNESS_EQUITY = ("fairness_equity", 1.8)  # High weight for justice decisions
    EXPLORATION_EXPLOITATION = ("exploration_exploitation", 1.1)  # Standard+ for learning
    
    def __init__(self, value: str, weight: float):
        self._value_ = value
        self.weight = weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "context": self.value,
            "weight": self.weight
        }


@dataclass
class RelationalAnchor:
    """Captures the relational dynamics of an action.
    
    This is KEY for pattern detection - tracks WHO affects WHOM and HOW.
    """
    
    actor: str  # Who performs the action
    target: str  # Who/what is affected
    relationship_type: str  # ally/adversary/neutral/resource
    impact_magnitude: float  # -1 to 1 (harm to benefit)
    
    def __post_init__(self):
        """Validate relational anchor data."""
        # Validate relationship type
        valid_types = {"ally", "adversary", "neutral", "resource", "self", "environment"}
        if self.relationship_type not in valid_types:
            raise ValueError(f"relationship_type must be one of {valid_types}")
        
        # Validate impact magnitude
        if not -1.0 <= self.impact_magnitude <= 1.0:
            raise ValueError("impact_magnitude must be between -1 and 1")
        
        # Validate actor and target are non-empty
        if not self.actor or not self.target:
            raise ValueError("actor and target must be non-empty strings")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Action:
    """Complete record of a decision made by an agent.
    
    Captures the full context of WHO did WHAT to WHOM and WHY.
    """
    
    id: str
    timestamp: datetime
    relational_anchor: RelationalAnchor
    decision_context: DecisionContext
    action_type: str  # The specific action taken
    outcome_valence: float  # -1 to 1 (negative to positive outcome)
    decision_entropy: float  # 0 to 1 (certainty to uncertainty)
    latency_ms: int  # Faster decisions = more internalized principles
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate action data."""
        # Validate outcome valence
        if not -1.0 <= self.outcome_valence <= 1.0:
            raise ValueError("outcome_valence must be between -1 and 1")
        
        # Validate decision entropy
        if not 0.0 <= self.decision_entropy <= 1.0:
            raise ValueError("decision_entropy must be between 0 and 1")
        
        # Validate latency
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        
        # Convert timestamp to datetime if string
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "relational_anchor": self.relational_anchor.to_dict(),
            "decision_context": self.decision_context.to_dict(),
            "action_type": self.action_type,
            "outcome_valence": self.outcome_valence,
            "decision_entropy": self.decision_entropy,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata
        }


@dataclass
class Principle:
    """A discovered behavioral pattern that guides agent decisions.
    
    Principles are inferred from consistent patterns in actions.
    """
    
    id: str
    name: str
    description: str
    strength_score: float = 0.0  # Bayesian updated strength
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # Based on evidence
    volatility: float = 0.0  # High volatility = opportunistic behavior
    evidence_count: int = 0  # Number of supporting actions
    contradictions_count: int = 0  # Number of contradicting actions
    contexts: List[DecisionContext] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate principle data."""
        # Validate strength score
        if not 0.0 <= self.strength_score <= 1.0:
            raise ValueError("strength_score must be between 0 and 1")
        
        # Validate confidence interval
        low, high = self.confidence_interval
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("confidence_interval must be valid with 0 <= low <= high <= 1")
        
        # Validate volatility
        if not 0.0 <= self.volatility <= 1.0:
            raise ValueError("volatility must be between 0 and 1")
        
        # Validate counts
        if self.evidence_count < 0 or self.contradictions_count < 0:
            raise ValueError("evidence and contradiction counts must be non-negative")
    
    def update_with_evidence(self, supporting: bool, context: DecisionContext):
        """Update principle strength with new evidence using Bayesian approach."""
        # Update counts
        if supporting:
            self.evidence_count += 1
        else:
            self.contradictions_count += 1
        
        # Update contexts if new
        if context not in self.contexts:
            self.contexts.append(context)
        
        # Bayesian update of strength score
        total_observations = self.evidence_count + self.contradictions_count
        if total_observations > 0:
            # Simple Bayesian update with prior
            alpha = self.evidence_count + 1  # Add 1 for prior
            beta = self.contradictions_count + 1
            self.strength_score = alpha / (alpha + beta)
            
            # Update confidence interval (using Wilson score interval)
            import math
            z = 1.96  # 95% confidence
            n = total_observations
            p = self.strength_score
            
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * math.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
            
            self.confidence_interval = (
                max(0.0, center - margin),
                min(1.0, center + margin)
            )
            
            # Update volatility (standard deviation of recent changes)
            # This is simplified - in production would track history
            confidence_width = self.confidence_interval[1] - self.confidence_interval[0]
            self.volatility = min(1.0, confidence_width * 2)  # Scale to 0-1
        
        self.last_updated = datetime.utcnow()
        
        logger.info(
            "principle_updated",
            principle_id=self.id,
            supporting=supporting,
            new_strength=self.strength_score,
            confidence_interval=self.confidence_interval,
            volatility=self.volatility
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strength_score": self.strength_score,
            "confidence_interval": self.confidence_interval,
            "volatility": self.volatility,
            "evidence_count": self.evidence_count,
            "contradictions_count": self.contradictions_count,
            "contexts": [c.to_dict() for c in self.contexts],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class PrincipleLineage:
    """Tracks the evolution and relationships between principles.
    
    Captures how principles fork, merge, and evolve over time.
    """
    
    principle_id: str
    parent_ids: List[str] = field(default_factory=list)  # Can have multiple parents (merge)
    child_ids: List[str] = field(default_factory=list)  # Can fork into multiple children
    lineage_type: str = "root"  # root/fork/merge/evolution
    transformation_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate lineage data."""
        valid_types = {"root", "fork", "merge", "evolution"}
        if self.lineage_type not in valid_types:
            raise ValueError(f"lineage_type must be one of {valid_types}")
        
        # Validate lineage logic
        if self.lineage_type == "root" and self.parent_ids:
            raise ValueError("Root principles cannot have parents")
        
        if self.lineage_type == "fork" and len(self.parent_ids) != 1:
            raise ValueError("Fork must have exactly one parent")
        
        if self.lineage_type == "merge" and len(self.parent_ids) < 2:
            raise ValueError("Merge must have at least two parents")
    
    def add_child(self, child_id: str):
        """Add a child principle to this lineage."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "principle_id": self.principle_id,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "lineage_type": self.lineage_type,
            "transformation_reason": self.transformation_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentProfile:
    """Tracks an agent's behavioral profile and action history.
    
    Maintains a capped timeline to prevent memory bloat.
    """
    
    agent_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    action_timeline: deque = field(default_factory=lambda: deque(maxlen=10000))  # Cap at 10k
    active_principles: Dict[str, Principle] = field(default_factory=dict)
    principle_lineages: Dict[str, PrincipleLineage] = field(default_factory=dict)
    total_actions: int = 0  # Track total even if timeline is capped
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: Action):
        """Add an action to the agent's timeline."""
        self.action_timeline.append(action)
        self.total_actions += 1
        
        logger.debug(
            "action_added",
            agent_id=self.agent_id,
            action_id=action.id,
            total_actions=self.total_actions,
            timeline_size=len(self.action_timeline)
        )
    
    def add_principle(self, principle: Principle, lineage: PrincipleLineage):
        """Add a new principle to the agent's profile."""
        self.active_principles[principle.id] = principle
        self.principle_lineages[principle.id] = lineage
        
        logger.info(
            "principle_added",
            agent_id=self.agent_id,
            principle_id=principle.id,
            principle_name=principle.name,
            lineage_type=lineage.lineage_type
        )
    
    def get_recent_actions(self, count: int = 100) -> List[Action]:
        """Get the most recent actions (up to count)."""
        return list(self.action_timeline)[-count:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "total_actions": self.total_actions,
            "recent_actions": [a.to_dict() for a in self.get_recent_actions(100)],
            "active_principles": {k: v.to_dict() for k, v in self.active_principles.items()},
            "principle_lineages": {k: v.to_dict() for k, v in self.principle_lineages.items()},
            "metadata": self.metadata
        }


# Utility functions for JSON serialization
def serialize_model(obj: Any) -> str:
    """Serialize a model instance to JSON string."""
    if hasattr(obj, 'to_dict'):
        return json.dumps(obj.to_dict(), indent=2)
    else:
        raise ValueError(f"Object {type(obj)} does not have a to_dict method")


def deserialize_action(data: Dict[str, Any]) -> Action:
    """Deserialize an Action from a dictionary."""
    # Reconstruct RelationalAnchor
    anchor_data = data["relational_anchor"]
    anchor = RelationalAnchor(**anchor_data)
    
    # Reconstruct DecisionContext
    context_data = data["decision_context"]
    context = next(dc for dc in DecisionContext if dc.value == context_data["context"])
    
    return Action(
        id=data["id"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        relational_anchor=anchor,
        decision_context=context,
        action_type=data["action_type"],
        outcome_valence=data["outcome_valence"],
        decision_entropy=data["decision_entropy"],
        latency_ms=data["latency_ms"],
        metadata=data.get("metadata", {})
    )
