"""High-performance behavioral tracking with entropy analysis.

This module provides real-time tracking of agent behaviors with:
- Thread-safe action buffering
- Periodic database flushing
- Behavioral entropy calculation
- Relational pattern extraction using clustering
- Performance optimizations and caching
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import structlog

from .models import Action, DecisionContext, RelationalAnchor, AgentProfile

logger = structlog.get_logger()


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for database failures."""
    
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    reset_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    failure_threshold: int = 5
    
    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )
    
    def record_success(self):
        """Record a success and reset the circuit."""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None
    
    def should_attempt(self) -> bool:
        """Check if we should attempt an operation."""
        if not self.is_open:
            return True
        
        # Check if enough time has passed to retry
        if self.last_failure_time:
            elapsed = datetime.utcnow() - self.last_failure_time
            if elapsed >= self.reset_timeout:
                self.is_open = False
                self.failure_count = 0
                logger.info("circuit_breaker_reset")
                return True
        
        return False


@dataclass
class EntropyCache:
    """Cache for entropy calculations with invalidation."""
    
    cache: Dict[str, Tuple[float, datetime]] = field(default_factory=dict)
    max_age: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    invalidation_threshold: int = 100  # Invalidate after N new actions
    
    def get(self, key: str) -> Optional[float]:
        """Get cached entropy value if still valid."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.max_age:
                return value
            else:
                # Remove stale entry
                del self.cache[key]
        return None
    
    def set(self, key: str, value: float):
        """Cache an entropy value."""
        self.cache[key] = (value, datetime.utcnow())
    
    def invalidate(self, key: str):
        """Invalidate a specific cache entry."""
        self.cache.pop(key, None)
    
    def clear_stale(self):
        """Remove all stale cache entries."""
        now = datetime.utcnow()
        stale_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.max_age
        ]
        for key in stale_keys:
            del self.cache[key]


class BehavioralTracker:
    """High-performance behavioral tracking with entropy analysis."""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval: float = 30.0,
        database_handler: Optional[Any] = None
    ):
        """Initialize the behavioral tracker.
        
        Args:
            buffer_size: Maximum actions to buffer before forcing flush
            flush_interval: Seconds between periodic flushes
            database_handler: Async database handler for persistence
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.database_handler = database_handler
        
        # Thread-safe action buffer
        self._action_buffer = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Agent profiles cache
        self._agent_profiles: Dict[str, AgentProfile] = {}
        self._profiles_lock = threading.Lock()
        
        # Performance optimization caches
        self._entropy_cache = EntropyCache()
        self._pattern_cache: Dict[str, Dict] = {}
        self._pattern_cache_lock = threading.Lock()
        
        # Circuit breaker for database failures
        self._circuit_breaker = CircuitBreakerState()
        
        # Metrics
        self._total_actions_tracked = 0
        self._last_flush_time = datetime.utcnow()
        self._flush_task: Optional[asyncio.Task] = None
        
        # State snapshot management
        self._state_snapshots: deque = deque(maxlen=100)  # Keep last 100 snapshots
        
        logger.info(
            "behavioral_tracker_initialized",
            buffer_size=buffer_size,
            flush_interval=flush_interval
        )
    
    async def start(self):
        """Start the periodic flush task."""
        if not self._flush_task:
            self._flush_task = asyncio.create_task(self._periodic_flush())
            logger.info("periodic_flush_started")
    
    async def stop(self):
        """Stop the tracker and flush remaining actions."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_buffer()
        logger.info("behavioral_tracker_stopped")
    
    def track_action(self, action: Action) -> bool:
        """Track a new action with validation.
        
        Args:
            action: The action to track
            
        Returns:
            bool: True if action was successfully tracked
        """
        try:
            # Validate action
            self._validate_action(action)
            
            # Add to buffer
            with self._buffer_lock:
                self._action_buffer.append(action)
                self._total_actions_tracked += 1
                
                # Update agent profile
                self._update_agent_profile(action)
                
                # Check if buffer is full
                should_flush = len(self._action_buffer) >= self.buffer_size
            
            # Invalidate relevant caches
            self._invalidate_caches(action.relational_anchor.actor)
            
            # Async flush if needed
            if should_flush:
                asyncio.create_task(self._flush_buffer())
            
            logger.debug(
                "action_tracked",
                action_id=action.id,
                agent_id=action.relational_anchor.actor,
                buffer_size=len(self._action_buffer)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "action_tracking_failed",
                action_id=action.id,
                error=str(e),
                exc_info=True
            )
            return False
    
    def calculate_behavioral_entropy(self, actions: List[Action]) -> float:
        """Calculate Shannon entropy of agent behavior.
        
        High entropy (>0.7) indicates inconsistent/random behavior.
        Low entropy (<0.3) indicates principled behavior.
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            float: Entropy value between 0 and 1
        """
        if not actions:
            return 0.0
        
        # Create cache key from action IDs
        cache_key = f"entropy_{hash(tuple(a.id for a in actions[:100]))}"
        
        # Check cache
        cached_value = self._entropy_cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        try:
            # Group actions by intent (combination of context and relationship)
            intent_groups = defaultdict(int)
            
            for action in actions:
                # Create intent signature
                intent = (
                    action.decision_context.value,
                    action.relational_anchor.relationship_type,
                    # Discretize impact magnitude into bins
                    round(action.relational_anchor.impact_magnitude * 5) / 5
                )
                intent_groups[intent] += 1
            
            # Calculate probability distribution
            total_actions = len(actions)
            probabilities = [count / total_actions for count in intent_groups.values()]
            
            # Calculate Shannon entropy
            # Normalize by log(n) to get value between 0 and 1
            raw_entropy = entropy(probabilities, base=2)
            max_entropy = np.log2(len(intent_groups)) if len(intent_groups) > 1 else 1
            normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0
            
            # Cache the result
            self._entropy_cache.set(cache_key, normalized_entropy)
            
            logger.debug(
                "behavioral_entropy_calculated",
                num_actions=len(actions),
                num_intents=len(intent_groups),
                entropy=normalized_entropy
            )
            
            return normalized_entropy
            
        except Exception as e:
            logger.error(
                "entropy_calculation_failed",
                error=str(e),
                exc_info=True
            )
            return 0.5  # Return neutral entropy on error
    
    def extract_relational_patterns(self, actions: List[Action]) -> Dict[str, Any]:
        """Extract relational patterns using DBSCAN clustering.
        
        Finds patterns like "always helps allies" or "exploits neutrals".
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            Dict containing extracted patterns
        """
        if len(actions) < 10:  # Need minimum actions for clustering
            return {"patterns": [], "clusters": 0}
        
        # Create cache key
        cache_key = f"patterns_{hash(tuple(a.id for a in actions[:100]))}"
        
        # Check cache
        with self._pattern_cache_lock:
            if cache_key in self._pattern_cache:
                return self._pattern_cache[cache_key]
        
        try:
            # Create feature vectors for clustering
            features = []
            action_indices = []
            
            for i, action in enumerate(actions):
                # Feature vector: [relationship_type_encoded, impact_magnitude, 
                #                  decision_context_weight, outcome_valence]
                rel_type_encoding = {
                    "ally": 0, "adversary": 1, "neutral": 2, 
                    "resource": 3, "self": 4, "environment": 5
                }
                
                feature = [
                    rel_type_encoding.get(action.relational_anchor.relationship_type, 2),
                    action.relational_anchor.impact_magnitude,
                    action.decision_context.weight,
                    action.outcome_valence
                ]
                features.append(feature)
                action_indices.append(i)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=5)
            labels = clustering.fit_predict(features_scaled)
            
            # Extract patterns from clusters
            patterns = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                # Get actions in this cluster
                cluster_indices = [i for i, l in enumerate(labels) if l == label]
                cluster_actions = [actions[i] for i in cluster_indices]
                
                # Analyze cluster characteristics
                pattern = self._analyze_cluster_pattern(cluster_actions)
                if pattern:
                    patterns.append(pattern)
            
            result = {
                "patterns": patterns,
                "clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
                "noise_ratio": (labels == -1).sum() / len(labels) if len(labels) > 0 else 0
            }
            
            # Cache the result
            with self._pattern_cache_lock:
                self._pattern_cache[cache_key] = result
            
            logger.info(
                "relational_patterns_extracted",
                num_actions=len(actions),
                num_patterns=len(patterns),
                clusters=result["clusters"],
                noise_ratio=result["noise_ratio"]
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "pattern_extraction_failed",
                error=str(e),
                exc_info=True
            )
            return {"patterns": [], "clusters": 0, "error": str(e)}
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile with thread safety."""
        with self._profiles_lock:
            return self._agent_profiles.get(agent_id)
    
    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get current tracking metrics."""
        with self._buffer_lock:
            buffer_size = len(self._action_buffer)
        
        return {
            "total_actions_tracked": self._total_actions_tracked,
            "buffer_size": buffer_size,
            "buffer_utilization": buffer_size / self.buffer_size,
            "last_flush_time": self._last_flush_time.isoformat(),
            "circuit_breaker_open": self._circuit_breaker.is_open,
            "circuit_breaker_failures": self._circuit_breaker.failure_count,
            "entropy_cache_size": len(self._entropy_cache.cache),
            "pattern_cache_size": len(self._pattern_cache),
            "active_agents": len(self._agent_profiles)
        }
    
    async def create_state_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current tracking state."""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.get_tracking_metrics(),
            "agent_summaries": {}
        }
        
        # Add agent summaries
        with self._profiles_lock:
            for agent_id, profile in self._agent_profiles.items():
                recent_actions = profile.get_recent_actions(50)
                if recent_actions:
                    snapshot["agent_summaries"][agent_id] = {
                        "total_actions": profile.total_actions,
                        "active_principles": len(profile.active_principles),
                        "recent_entropy": self.calculate_behavioral_entropy(recent_actions)
                    }
        
        # Store snapshot
        self._state_snapshots.append(snapshot)
        
        return snapshot
    
    # Private methods
    
    def _validate_action(self, action: Action):
        """Validate an action before tracking."""
        if not action.id:
            raise ValueError("Action must have an ID")
        
        if not isinstance(action.relational_anchor, RelationalAnchor):
            raise ValueError("Action must have a valid RelationalAnchor")
        
        if not isinstance(action.decision_context, DecisionContext):
            raise ValueError("Action must have a valid DecisionContext")
        
        # Additional validation is handled by the Action model itself
    
    def _update_agent_profile(self, action: Action):
        """Update agent profile with new action (called within lock)."""
        agent_id = action.relational_anchor.actor
        
        if agent_id not in self._agent_profiles:
            self._agent_profiles[agent_id] = AgentProfile(
                agent_id=agent_id,
                name=f"Agent-{agent_id}"
            )
        
        self._agent_profiles[agent_id].add_action(action)
    
    def _invalidate_caches(self, agent_id: str):
        """Invalidate caches related to an agent."""
        # Invalidate entropy cache entries for this agent
        keys_to_invalidate = [k for k in self._entropy_cache.cache.keys() if agent_id in k]
        for key in keys_to_invalidate:
            self._entropy_cache.invalidate(key)
        
        # Clear pattern cache entries for this agent
        with self._pattern_cache_lock:
            keys_to_remove = [k for k in self._pattern_cache.keys() if agent_id in k]
            for key in keys_to_remove:
                del self._pattern_cache[key]
    
    async def _periodic_flush(self):
        """Periodically flush the action buffer."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
                
                # Cleanup stale cache entries
                self._entropy_cache.clear_stale()
                
                # Create periodic state snapshot
                await self.create_state_snapshot()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "periodic_flush_error",
                    error=str(e),
                    exc_info=True
                )
    
    async def _flush_buffer(self):
        """Flush action buffer to database."""
        if not self.database_handler:
            return
        
        # Check circuit breaker
        if not self._circuit_breaker.should_attempt():
            logger.warning("flush_skipped_circuit_open")
            return
        
        # Extract actions to flush
        with self._buffer_lock:
            if not self._action_buffer:
                return
            
            actions_to_flush = list(self._action_buffer)
            self._action_buffer.clear()
        
        try:
            # Attempt database write
            start_time = time.time()
            await self.database_handler.bulk_insert_actions(actions_to_flush)
            elapsed_ms = (time.time() - start_time) * 1000
            
            self._last_flush_time = datetime.utcnow()
            self._circuit_breaker.record_success()
            
            logger.info(
                "buffer_flushed",
                num_actions=len(actions_to_flush),
                elapsed_ms=elapsed_ms
            )
            
        except Exception as e:
            # Put actions back in buffer on failure
            with self._buffer_lock:
                self._action_buffer.extendleft(actions_to_flush)
            
            self._circuit_breaker.record_failure()
            
            logger.error(
                "buffer_flush_failed",
                error=str(e),
                num_actions=len(actions_to_flush),
                exc_info=True
            )
    
    def _analyze_cluster_pattern(self, cluster_actions: List[Action]) -> Optional[Dict[str, Any]]:
        """Analyze a cluster of actions to extract pattern."""
        if not cluster_actions:
            return None
        
        # Calculate cluster statistics
        relationship_types = defaultdict(int)
        impact_magnitudes = []
        contexts = defaultdict(int)
        
        for action in cluster_actions:
            relationship_types[action.relational_anchor.relationship_type] += 1
            impact_magnitudes.append(action.relational_anchor.impact_magnitude)
            contexts[action.decision_context.value] += 1
        
        # Find dominant characteristics
        dominant_relationship = max(relationship_types.items(), key=lambda x: x[1])
        avg_impact = np.mean(impact_magnitudes)
        dominant_context = max(contexts.items(), key=lambda x: x[1])
        
        # Generate pattern description
        impact_desc = "positive" if avg_impact > 0.3 else "negative" if avg_impact < -0.3 else "neutral"
        
        pattern = {
            "description": f"{impact_desc} actions towards {dominant_relationship[0]} in {dominant_context[0]} contexts",
            "dominant_relationship": dominant_relationship[0],
            "relationship_frequency": dominant_relationship[1] / len(cluster_actions),
            "average_impact": avg_impact,
            "impact_std": np.std(impact_magnitudes),
            "dominant_context": dominant_context[0],
            "context_frequency": dominant_context[1] / len(cluster_actions),
            "cluster_size": len(cluster_actions),
            "consistency_score": min(
                dominant_relationship[1] / len(cluster_actions),
                dominant_context[1] / len(cluster_actions)
            )
        }
        
        # Only return patterns with reasonable consistency
        if pattern["consistency_score"] > 0.5:
            return pattern
        
        return None


# Factory function for creating tracker with database
async def create_behavioral_tracker(
    database_url: Optional[str] = None,
    **kwargs
) -> BehavioralTracker:
    """Create a behavioral tracker with optional database connection.
    
    Args:
        database_url: Optional database connection URL
        **kwargs: Additional arguments for BehavioralTracker
        
    Returns:
        Configured BehavioralTracker instance
    """
    database_handler = None
    
    if database_url:
        # Import database handler dynamically to avoid circular imports
        from .database import DatabaseHandler
        database_handler = await DatabaseHandler.create(database_url)
    
    tracker = BehavioralTracker(
        database_handler=database_handler,
        **kwargs
    )
    
    await tracker.start()
    
    return tracker
