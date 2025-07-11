"""Behavioral principle discovery engine using temporal pattern matching.

This module implements the core inference algorithms for discovering
behavioral principles from action sequences using:
- Dynamic Time Warping (DTW) for sequence comparison
- Context-weighted inference for principle extraction
- Bayesian updates for principle strength
- Evolution tracking for principle lifecycle
"""

import asyncio
import hashlib
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
import structlog

from .models import (
    Action, DecisionContext, Principle, PrincipleLineage, 
    AgentProfile, RelationalAnchor
)
from .tracking import BehavioralTracker
from .config import settings
from .llm_analysis import LLMAnalyzer, create_llm_analyzer

logger = structlog.get_logger()


# Critical thresholds from the task specification
MIN_PATTERN_LENGTH = 20
CONSISTENCY_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 0.7
EVOLUTION_DIVERGENCE_THRESHOLD = 0.3


@dataclass
class TemporalPattern:
    """Represents a discovered temporal pattern in action sequences."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_sequence: List[Action] = field(default_factory=list)
    pattern_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    context_weights: Dict[str, float] = field(default_factory=dict)
    consistency_score: float = 0.0
    support_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate pattern data."""
        if len(self.action_sequence) < MIN_PATTERN_LENGTH:
            raise ValueError(f"Pattern must have at least {MIN_PATTERN_LENGTH} actions")
        
        if not 0.0 <= self.consistency_score <= 1.0:
            raise ValueError("consistency_score must be between 0 and 1")


@dataclass
class PrincipleCandidate:
    """A candidate principle extracted from patterns."""
    
    patterns: List[TemporalPattern] = field(default_factory=list)
    description: str = ""
    contexts: Set[DecisionContext] = field(default_factory=set)
    weighted_importance: float = 0.0
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None


class PrincipleInferenceEngine:
    """Core engine for discovering behavioral principles from actions."""
    
    def __init__(
        self,
        behavioral_tracker: BehavioralTracker,
        min_pattern_length: int = MIN_PATTERN_LENGTH,
        consistency_threshold: float = CONSISTENCY_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        evolution_divergence_threshold: float = EVOLUTION_DIVERGENCE_THRESHOLD,
        llm_analyzer: Optional[LLMAnalyzer] = None
    ):
        """Initialize the inference engine.
        
        Args:
            behavioral_tracker: The behavioral tracker instance
            min_pattern_length: Minimum actions for a valid pattern
            consistency_threshold: Threshold for principle consistency
            entropy_threshold: Max entropy for principled behavior
            evolution_divergence_threshold: Threshold for principle divergence
            llm_analyzer: Optional LLM analyzer for enhanced analysis
        """
        self.tracker = behavioral_tracker
        self.min_pattern_length = min_pattern_length
        self.consistency_threshold = consistency_threshold
        self.entropy_threshold = entropy_threshold
        self.evolution_divergence_threshold = evolution_divergence_threshold
        self.llm_analyzer = llm_analyzer
        
        # Pattern and principle caches
        self._pattern_cache: Dict[str, List[TemporalPattern]] = {}
        self._principle_embeddings: Dict[str, np.ndarray] = {}
        self._dtw_distance_cache: Dict[Tuple[str, str], float] = {}
        
        # Active inference tasks
        self._inference_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(
            "inference_engine_initialized",
            min_pattern_length=min_pattern_length,
            consistency_threshold=consistency_threshold,
            llm_enabled=llm_analyzer is not None
        )
    
    async def infer_principles(self, agent_id: str) -> List[Principle]:
        """Main entry point for principle inference for an agent.
        
        Args:
            agent_id: The agent to analyze
            
        Returns:
            List of inferred principles
        """
        try:
            # Get agent profile
            profile = self.tracker.get_agent_profile(agent_id)
            if not profile:
                logger.warning("agent_not_found", agent_id=agent_id)
                return []
            
            # Get recent actions
            actions = profile.get_recent_actions(1000)  # Analyze last 1000 actions
            if len(actions) < self.min_pattern_length:
                logger.info(
                    "insufficient_actions",
                    agent_id=agent_id,
                    action_count=len(actions),
                    required=self.min_pattern_length
                )
                return []
            
            # Step 1: Extract patterns using DTW
            patterns = self.extract_patterns_dtw(actions)
            if not patterns:
                logger.info("no_patterns_found", agent_id=agent_id)
                return []
            
            # Step 2: Apply context-weighted inference
            candidates = await self._context_weighted_inference(patterns)
            
            # Step 3: Check for multiple personalities
            personalities = self._detect_multiple_personalities(candidates)
            if len(personalities) > 1:
                logger.warning(
                    "multiple_personalities_detected",
                    agent_id=agent_id,
                    personality_count=len(personalities)
                )
            
            # Step 4: Convert candidates to principles
            principles = []
            for candidate in candidates:
                if candidate.confidence >= self.consistency_threshold:
                    principle = self._create_principle_from_candidate(candidate, agent_id)
                    
                    # Check for evolution
                    evolved = await self._check_principle_evolution(
                        principle, profile.active_principles.values()
                    )
                    
                    if evolved:
                        # Handle principle evolution (fork/merge)
                        principle, lineage = evolved
                        profile.add_principle(principle, lineage)
                    
                    principles.append(principle)
            
            # Update pattern cache
            self._pattern_cache[agent_id] = patterns
            
            logger.info(
                "principles_inferred",
                agent_id=agent_id,
                pattern_count=len(patterns),
                principle_count=len(principles)
            )
            
            return principles
            
        except Exception as e:
            logger.error(
                "principle_inference_failed",
                agent_id=agent_id,
                error=str(e),
                exc_info=True
            )
            return []
    
    def extract_patterns_dtw(self, actions: List[Action]) -> List[TemporalPattern]:
        """Extract temporal patterns using Dynamic Time Warping.
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            List of discovered temporal patterns
        """
        if len(actions) < self.min_pattern_length:
            return []
        
        # Convert actions to time series
        time_series = self._actions_to_time_series(actions)
        
        # Use sliding windows to find candidate sequences
        window_size = self.min_pattern_length
        sequences = []
        
        for i in range(len(time_series) - window_size + 1):
            window = time_series[i:i + window_size]
            sequences.append((i, window))
        
        if len(sequences) < 2:
            return []
        
        # Calculate DTW distance matrix
        n = len(sequences)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check cache first
                cache_key = (f"seq_{i}", f"seq_{j}")
                if cache_key in self._dtw_distance_cache:
                    dist = self._dtw_distance_cache[cache_key]
                else:
                    # Calculate DTW distance with error handling
                    try:
                        dist = dtw.distance(sequences[i][1], sequences[j][1])
                        self._dtw_distance_cache[cache_key] = dist
                    except Exception as e:
                        logger.warning(
                            "DTW calculation failed",
                            error=str(e),
                            sequence_i=i,
                            sequence_j=j
                        )
                        # Use maximum distance as fallback to indicate dissimilarity
                        dist = float('inf')
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Cluster sequences using KMeans
        # k = min(5, n//3) as specified
        k = min(5, max(2, n // 3))
        
        # Use distance matrix for clustering
        # Convert distance matrix to similarity for KMeans
        max_dist = np.max(distance_matrix)
        similarity_matrix = max_dist - distance_matrix if max_dist > 0 else distance_matrix
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(similarity_matrix)
        
        # Extract patterns from clusters
        patterns = []
        
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < 2:
                continue
            
            # Get actions for this cluster
            cluster_actions = []
            for idx in cluster_indices:
                start_idx = sequences[idx][0]
                cluster_actions.extend(actions[start_idx:start_idx + window_size])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_actions = []
            for action in cluster_actions:
                if action.id not in seen:
                    seen.add(action.id)
                    unique_actions.append(action)
            
            if len(unique_actions) >= self.min_pattern_length:
                # Calculate pattern metrics
                pattern = self._create_temporal_pattern(unique_actions, len(cluster_indices))
                patterns.append(pattern)
        
        logger.debug(
            "dtw_patterns_extracted",
            num_sequences=len(sequences),
            num_clusters=k,
            num_patterns=len(patterns)
        )
        
        return patterns
    
    async def _context_weighted_inference(
        self, 
        patterns: List[TemporalPattern]
    ) -> List[PrincipleCandidate]:
        """Apply context-weighted inference to extract principle candidates.
        
        High-weight contexts reveal core principles.
        
        Args:
            patterns: Discovered temporal patterns
            
        Returns:
            List of principle candidates
        """
        candidates = []
        
        for pattern in patterns:
            # Calculate weighted importance based on contexts
            total_weight = 0.0
            context_counts = defaultdict(int)
            
            for action in pattern.action_sequence:
                context = action.decision_context
                total_weight += context.weight
                context_counts[context] += 1
            
            # Normalize by number of actions
            avg_weight = total_weight / len(pattern.action_sequence)
            
            # High-weight contexts (>1.5) indicate important principles
            if avg_weight > 1.5:
                # Extract dominant contexts
                dominant_contexts = [
                    ctx for ctx, count in context_counts.items()
                    if count / len(pattern.action_sequence) > 0.3
                ]
                
                # Create candidate
                candidate = PrincipleCandidate(
                    patterns=[pattern],
                    contexts=set(dominant_contexts),
                    weighted_importance=avg_weight,
                    confidence=pattern.consistency_score
                )
                
                # Generate description
                candidate.description = await self._generate_principle_description_async(
                    pattern, dominant_contexts
                )
                
                # Generate embedding for evolution tracking
                candidate.embedding = self._generate_pattern_embedding(pattern)
                
                candidates.append(candidate)
        
        # Merge similar candidates
        merged_candidates = self._merge_similar_candidates(candidates)
        
        logger.info(
            "context_weighted_inference_complete",
            input_patterns=len(patterns),
            candidates=len(candidates),
            merged_candidates=len(merged_candidates)
        )
        
        return merged_candidates
    
    def _detect_multiple_personalities(
        self, 
        candidates: List[PrincipleCandidate]
    ) -> List[List[PrincipleCandidate]]:
        """Detect multiple personalities - stable but contradictory patterns.
        
        Args:
            candidates: Principle candidates to analyze
            
        Returns:
            List of personality groups
        """
        if len(candidates) < 2:
            return [candidates]
        
        # Create similarity matrix based on embeddings
        embeddings = [c.embedding for c in candidates if c.embedding is not None]
        if len(embeddings) < 2:
            return [candidates]
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Use DBSCAN for personality clustering
        # Invert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        # Group candidates by personality
        personalities = defaultdict(list)
        for i, label in enumerate(labels):
            personalities[label].append(candidates[i])
        
        # Check for contradictions between personalities
        personality_groups = []
        for label, group in personalities.items():
            if label != -1:  # Skip noise
                # Check if group has high internal consistency
                group_consistency = np.mean([c.confidence for c in group])
                if group_consistency > self.consistency_threshold:
                    personality_groups.append(group)
        
        # Add unclustered high-confidence candidates as individual personalities
        for i, label in enumerate(labels):
            if label == -1 and candidates[i].confidence > self.consistency_threshold:
                personality_groups.append([candidates[i]])
        
        return personality_groups if personality_groups else [candidates]
    
    async def _check_principle_evolution(
        self,
        new_principle: Principle,
        existing_principles: List[Principle]
    ) -> Optional[Tuple[Principle, PrincipleLineage]]:
        """Check if a principle represents evolution of existing principles.
        
        Detects forks (principle splits) and merges (principles combine).
        
        Args:
            new_principle: The newly inferred principle
            existing_principles: Current active principles
            
        Returns:
            Updated principle and lineage if evolution detected
        """
        if not existing_principles:
            # Root principle
            lineage = PrincipleLineage(
                principle_id=new_principle.id,
                lineage_type="root"
            )
            return (new_principle, lineage)
        
        # Generate embedding for new principle
        new_embedding = self._principle_to_embedding(new_principle)
        
        # Compare with existing principles
        similarities = []
        for existing in existing_principles:
            existing_embedding = self._principle_embeddings.get(existing.id)
            if existing_embedding is None:
                existing_embedding = self._principle_to_embedding(existing)
                self._principle_embeddings[existing.id] = existing_embedding
            
            similarity = cosine_similarity(
                new_embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0, 0]
            
            similarities.append((existing, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Check for evolution patterns
        if similarities[0][1] > 0.8:
            # Very similar - this is an evolution of existing principle
            parent = similarities[0][0]
            
            # Check if parent is weakening (potential fork)
            if parent.strength_score < 0.5 and new_principle.strength_score > 0.7:
                lineage = PrincipleLineage(
                    principle_id=new_principle.id,
                    parent_ids=[parent.id],
                    lineage_type="fork",
                    transformation_reason="Parent principle weakened, new branch emerged"
                )
                return (new_principle, lineage)
        
        # Check for merge (multiple weak principles -> one strong)
        weak_parents = [
            p for p, sim in similarities 
            if sim > 0.6 and p.strength_score < 0.4
        ]
        
        if len(weak_parents) >= 2 and new_principle.strength_score > 0.7:
            lineage = PrincipleLineage(
                principle_id=new_principle.id,
                parent_ids=[p.id for p in weak_parents],
                lineage_type="merge",
                transformation_reason="Multiple weak principles consolidated"
            )
            return (new_principle, lineage)
        
        # Check divergence using embedding distance
        max_similarity = similarities[0][1] if similarities else 0
        if max_similarity < self.evolution_divergence_threshold:
            # Too different - new root principle
            lineage = PrincipleLineage(
                principle_id=new_principle.id,
                lineage_type="root",
                transformation_reason="Significantly different from existing principles"
            )
            return (new_principle, lineage)
        
        # Regular evolution
        lineage = PrincipleLineage(
            principle_id=new_principle.id,
            parent_ids=[similarities[0][0].id],
            lineage_type="evolution"
        )
        return (new_principle, lineage)
    
    # Private helper methods
    
    def _actions_to_time_series(self, actions: List[Action]) -> np.ndarray:
        """Convert actions to numerical time series for DTW."""
        series = []
        
        for action in actions:
            # Create feature vector for each action
            features = [
                action.relational_anchor.impact_magnitude,
                action.outcome_valence,
                action.decision_entropy,
                action.decision_context.weight,
                # Encode relationship type
                hash(action.relational_anchor.relationship_type) % 10 / 10,
                # Normalize latency
                min(action.latency_ms / 1000, 1.0)
            ]
            series.append(features)
        
        return np.array(series)
    
    def _create_temporal_pattern(
        self, 
        actions: List[Action], 
        support_count: int
    ) -> TemporalPattern:
        """Create a temporal pattern from action sequence."""
        # Calculate pattern vector (mean of action features)
        time_series = self._actions_to_time_series(actions)
        pattern_vector = np.mean(time_series, axis=0)
        
        # Calculate context weights
        context_weights = defaultdict(float)
        for action in actions:
            context_weights[action.decision_context.value] += action.decision_context.weight
        
        # Normalize weights
        total_weight = sum(context_weights.values())
        if total_weight > 0:
            context_weights = {k: v/total_weight for k, v in context_weights.items()}
        
        # Calculate consistency score
        entropy_val = self.tracker.calculate_behavioral_entropy(actions)
        consistency_score = 1.0 - entropy_val  # Low entropy = high consistency
        
        return TemporalPattern(
            action_sequence=actions,
            pattern_vector=pattern_vector,
            context_weights=dict(context_weights),
            consistency_score=consistency_score,
            support_count=support_count
        )
    
    async def _generate_principle_description_async(
        self, 
        pattern: TemporalPattern,
        dominant_contexts: List[DecisionContext]
    ) -> str:
        """Generate human-readable principle description using LLM if available."""
        # Use LLM if available and enabled
        if (self.llm_analyzer and self.llm_analyzer.enabled and 
            settings.ENABLE_LLM_PRINCIPLE_GENERATION):
            try:
                return await self.llm_analyzer.generate_principle_description(pattern)
            except Exception as e:
                logger.warning(
                    "llm_description_generation_failed",
                    error=str(e),
                    pattern_id=pattern.id
                )
        
        # Fallback to template-based generation
        return self._generate_principle_description(pattern, dominant_contexts)
    
    def _generate_principle_description(
        self, 
        pattern: TemporalPattern,
        dominant_contexts: List[DecisionContext]
    ) -> str:
        """Generate human-readable principle description."""
        # Analyze pattern characteristics
        actions = pattern.action_sequence
        
        # Get dominant relationship types
        relationship_counts = defaultdict(int)
        impact_sum = 0.0
        
        for action in actions:
            relationship_counts[action.relational_anchor.relationship_type] += 1
            impact_sum += action.relational_anchor.impact_magnitude
        
        dominant_relationship = max(relationship_counts.items(), key=lambda x: x[1])[0]
        avg_impact = impact_sum / len(actions)
        
        # Generate description
        impact_desc = "positive" if avg_impact > 0.3 else "negative" if avg_impact < -0.3 else "neutral"
        context_names = [ctx.value for ctx in dominant_contexts]
        
        description = (
            f"Tends to take {impact_desc} actions towards {dominant_relationship}s "
            f"in {', '.join(context_names)} contexts"
        )
        
        return description
    
    def _generate_pattern_embedding(self, pattern: TemporalPattern) -> np.ndarray:
        """Generate embedding vector for a pattern."""
        # Combine multiple features into embedding
        features = []
        
        # Pattern vector features
        features.extend(pattern.pattern_vector.tolist())
        
        # Context weight features
        for context in DecisionContext:
            features.append(pattern.context_weights.get(context.value, 0.0))
        
        # Statistical features
        actions = pattern.action_sequence
        if actions:
            impacts = [a.relational_anchor.impact_magnitude for a in actions]
            outcomes = [a.outcome_valence for a in actions]
            
            features.extend([
                np.mean(impacts),
                np.std(impacts),
                np.mean(outcomes),
                np.std(outcomes),
                pattern.consistency_score,
                len(actions) / 1000  # Normalized length
            ])
        
        return np.array(features)
    
    def _principle_to_embedding(self, principle: Principle) -> np.ndarray:
        """Generate embedding for a principle."""
        features = []
        
        # Basic principle features
        features.extend([
            principle.strength_score,
            principle.volatility,
            principle.evidence_count / 1000,  # Normalized
            principle.contradictions_count / 1000,
            principle.confidence_interval[0],
            principle.confidence_interval[1]
        ])
        
        # Context features
        context_vector = np.zeros(len(DecisionContext))
        for i, context in enumerate(DecisionContext):
            if context in principle.contexts:
                context_vector[i] = context.weight
        features.extend(context_vector.tolist())
        
        # Text features (simplified - in production would use proper NLP)
        text_hash = int(hashlib.md5(principle.description.encode()).hexdigest()[:8], 16)
        features.append(text_hash / 1e10)  # Normalized hash
        
        return np.array(features)
    
    def _merge_similar_candidates(
        self, 
        candidates: List[PrincipleCandidate]
    ) -> List[PrincipleCandidate]:
        """Merge similar principle candidates."""
        if len(candidates) <= 1:
            return candidates
        
        # Create similarity matrix
        embeddings = [c.embedding for c in candidates if c.embedding is not None]
        if not embeddings:
            return candidates
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Merge candidates with similarity > 0.8
        merged = []
        used = set()
        
        for i in range(len(candidates)):
            if i in used:
                continue
                
            # Find similar candidates
            similar_indices = [
                j for j in range(len(candidates))
                if j != i and similarity_matrix[i, j] > 0.8 and j not in used
            ]
            
            if similar_indices:
                # Merge candidates
                merged_candidate = candidates[i]
                for j in similar_indices:
                    # Combine patterns
                    merged_candidate.patterns.extend(candidates[j].patterns)
                    # Combine contexts
                    merged_candidate.contexts.update(candidates[j].contexts)
                    # Update importance (weighted average)
                    total_patterns = len(merged_candidate.patterns) + len(candidates[j].patterns)
                    merged_candidate.weighted_importance = (
                        merged_candidate.weighted_importance * len(merged_candidate.patterns) +
                        candidates[j].weighted_importance * len(candidates[j].patterns)
                    ) / total_patterns
                    
                    used.add(j)
                
                # Recalculate confidence
                all_scores = [p.consistency_score for p in merged_candidate.patterns]
                merged_candidate.confidence = np.mean(all_scores)
                
                # Regenerate embedding
                merged_candidate.embedding = np.mean(
                    [self._generate_pattern_embedding(p) for p in merged_candidate.patterns],
                    axis=0
                )
            
            merged.append(merged_candidate)
            used.add(i)
        
        return merged
    
    def _create_principle_from_candidate(
        self,
        candidate: PrincipleCandidate,
        agent_id: str
    ) -> Principle:
        """Create a Principle from a PrincipleCandidate."""
        # Calculate initial strength based on pattern support
        total_actions = sum(p.support_count for p in candidate.patterns)
        initial_strength = min(total_actions / 100, 0.9)  # Cap at 0.9
        
        principle = Principle(
            id=str(uuid.uuid4()),
            name=f"Principle-{agent_id}-{len(self._principle_embeddings)+1}",
            description=candidate.description,
            strength_score=initial_strength,
            confidence_interval=(
                max(0, initial_strength - 0.1),
                min(1, initial_strength + 0.1)
            ),
            contexts=list(candidate.contexts),
            evidence_count=total_actions,
            metadata={
                "pattern_count": len(candidate.patterns),
                "weighted_importance": candidate.weighted_importance,
                "inferred_from_actions": sum(len(p.action_sequence) for p in candidate.patterns)
            }
        )
        
        # Store embedding
        if candidate.embedding is not None:
            self._principle_embeddings[principle.id] = candidate.embedding
        
        return principle
    
    async def continuous_inference(self, agent_id: str, interval: float = 60.0):
        """Run continuous principle inference for an agent.
        
        Args:
            agent_id: The agent to monitor
            interval: Seconds between inference runs
        """
        task_key = f"inference_{agent_id}"
        
        # Cancel existing task if any
        if task_key in self._inference_tasks:
            self._inference_tasks[task_key].cancel()
        
        async def inference_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.infer_principles(agent_id)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "continuous_inference_error",
                        agent_id=agent_id,
                        error=str(e),
                        exc_info=True
                    )
        
        self._inference_tasks[task_key] = asyncio.create_task(inference_loop())
        logger.info(
            "continuous_inference_started",
            agent_id=agent_id,
            interval=interval
        )
    
    def stop_continuous_inference(self, agent_id: str):
        """Stop continuous inference for an agent."""
        task_key = f"inference_{agent_id}"
        if task_key in self._inference_tasks:
            self._inference_tasks[task_key].cancel()
            del self._inference_tasks[task_key]
            logger.info("continuous_inference_stopped", agent_id=agent_id)


# Factory function
async def create_inference_engine(
    behavioral_tracker: BehavioralTracker,
    **kwargs
) -> PrincipleInferenceEngine:
    """Create and initialize a principle inference engine.
    
    Args:
        behavioral_tracker: The behavioral tracker to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PrincipleInferenceEngine instance
    """
    engine = PrincipleInferenceEngine(
        behavioral_tracker=behavioral_tracker,
        **kwargs
    )
    
    logger.info("inference_engine_created")
    
    return engine
