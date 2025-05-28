"""
Hierarchical clustering-based inference plugin.

This plugin uses hierarchical clustering to identify behavioral patterns
and extract principles from agent actions.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import structlog

from ..decorators import register_plugin, plugin_config, cached_method
from ..base import InferencePlugin
from ...core.models import Action, Pattern, Principle

logger = structlog.get_logger()


@register_plugin(
    name="hierarchical_clustering_inference",
    version="1.0.0",
    author="AI Gym Team",
    description="Uses hierarchical clustering to discover behavioral patterns and principles",
    dependencies=["numpy", "scikit-learn"],
    tags=["clustering", "unsupervised", "pattern-detection"]
)
@plugin_config(
    n_clusters=None,  # None for automatic determination
    distance_threshold=0.5,
    linkage="ward",
    min_cluster_size=5,
    confidence_threshold=0.75,
    feature_extraction="context_action"  # or "embedding" if available
)
class HierarchicalClusteringInferencePlugin(InferencePlugin):
    """
    Inference plugin that uses hierarchical clustering to identify patterns.
    
    This plugin extracts features from actions, clusters them hierarchically,
    and infers principles from the resulting clusters.
    """
    
    def initialize(self) -> None:
        """Initialize the plugin and validate dependencies."""
        super().initialize()
        
        # Initialize clustering model
        self.clusterer = AgglomerativeClustering(
            n_clusters=self.config.get("n_clusters"),
            distance_threshold=self.config.get("distance_threshold"),
            linkage=self.config.get("linkage")
        )
        self.scaler = StandardScaler()
        
    def extract_patterns(self, actions: List[Action]) -> List[Pattern]:
        """
        Extract patterns using hierarchical clustering.
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            List of discovered patterns
        """
        if len(actions) < self.config["min_cluster_size"]:
            logger.warning(f"Not enough actions ({len(actions)}) for clustering")
            return []
        
        # Extract features from actions
        features = self._extract_features(actions)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(scaled_features)
        
        # Extract patterns from clusters
        patterns = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) >= self.config["min_cluster_size"]:
                cluster_actions = [actions[i] for i in cluster_indices]
                pattern = self._create_pattern_from_cluster(cluster_actions, label)
                patterns.append(pattern)
        
        logger.info(f"Extracted {len(patterns)} patterns from {len(actions)} actions")
        return patterns
    
    def infer_principles(self, patterns: List[Pattern]) -> List[Principle]:
        """
        Infer principles from discovered patterns.
        
        Args:
            patterns: List of patterns to analyze
            
        Returns:
            List of inferred principles
        """
        principles = []
        
        for pattern in patterns:
            # Analyze pattern for principle extraction
            principle_strength = self._calculate_principle_strength(pattern)
            
            if principle_strength >= self.config["confidence_threshold"]:
                principle = self._create_principle_from_pattern(pattern, principle_strength)
                principles.append(principle)
        
        # Merge similar principles
        merged_principles = self._merge_similar_principles(principles)
        
        logger.info(f"Inferred {len(merged_principles)} principles from {len(patterns)} patterns")
        return merged_principles
    
    @cached_method(ttl=600)
    def _extract_features(self, actions: List[Action]) -> np.ndarray:
        """Extract numerical features from actions."""
        feature_type = self.config["feature_extraction"]
        features = []
        
        for action in actions:
            if feature_type == "context_action":
                # Use context and action type as features
                feature_vector = [
                    action.context.value,  # Numeric context value
                    hash(action.action_taken) % 1000,  # Simple action hash
                    len(action.options),  # Number of options
                    action.confidence or 0.5,  # Confidence score
                ]
            elif feature_type == "embedding" and hasattr(action, "embedding"):
                # Use pre-computed embeddings if available
                feature_vector = action.embedding
            else:
                # Fallback to basic features
                feature_vector = [
                    action.context.value,
                    len(action.action_taken),
                    len(action.options)
                ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_pattern_from_cluster(self, actions: List[Action], cluster_id: int) -> Pattern:
        """Create a Pattern object from a cluster of actions."""
        # Calculate pattern frequency and confidence
        timestamps = [a.timestamp for a in actions]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        frequency = len(actions) / max(time_span, 1.0)
        
        # Find most common context and action
        contexts = [a.context for a in actions]
        most_common_context = max(set(contexts), key=contexts.count)
        
        action_types = [a.action_taken for a in actions]
        most_common_action = max(set(action_types), key=action_types.count)
        
        return Pattern(
            pattern_id=f"cluster_{cluster_id}",
            action_sequence=[most_common_action],  # Simplified for clustering
            frequency=frequency,
            confidence=len(actions) / self.config["min_cluster_size"],
            context_distribution={most_common_context: 1.0},
            first_seen=min(timestamps),
            last_seen=max(timestamps),
            occurrence_count=len(actions)
        )
    
    def _calculate_principle_strength(self, pattern: Pattern) -> float:
        """Calculate the strength of a principle from a pattern."""
        # Combine frequency, confidence, and occurrence count
        base_strength = pattern.confidence
        
        # Boost for high frequency patterns
        if pattern.frequency > 0.1:  # More than once per 10 seconds
            base_strength *= 1.2
        
        # Boost for patterns seen many times
        if pattern.occurrence_count > 10:
            base_strength *= 1.1
        
        return min(base_strength, 1.0)
    
    def _create_principle_from_pattern(self, pattern: Pattern, strength: float) -> Principle:
        """Create a Principle object from a pattern."""
        # Generate principle description based on pattern
        context_name = list(pattern.context_distribution.keys())[0].name
        action = pattern.action_sequence[0] if pattern.action_sequence else "unknown"
        
        description = f"In {context_name} contexts, prefer action: {action}"
        
        return Principle(
            principle_id=f"principle_{pattern.pattern_id}",
            description=description,
            strength=strength,
            consistency=pattern.confidence,
            first_observed=pattern.first_seen,
            last_updated=pattern.last_seen,
            supporting_patterns=[pattern.pattern_id],
            context_weights={context_name: 1.0}
        )
    
    def _merge_similar_principles(self, principles: List[Principle]) -> List[Principle]:
        """Merge principles that are similar."""
        if len(principles) <= 1:
            return principles
        
        # Simple merging based on description similarity
        merged = []
        used = set()
        
        for i, p1 in enumerate(principles):
            if i in used:
                continue
                
            similar_group = [p1]
            
            for j, p2 in enumerate(principles[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check if principles are similar (simplified check)
                if self._are_principles_similar(p1, p2):
                    similar_group.append(p2)
                    used.add(j)
            
            # Merge the group
            if len(similar_group) > 1:
                merged_principle = self._merge_principle_group(similar_group)
                merged.append(merged_principle)
            else:
                merged.append(p1)
        
        return merged
    
    def _are_principles_similar(self, p1: Principle, p2: Principle) -> bool:
        """Check if two principles are similar enough to merge."""
        # Check context overlap
        contexts1 = set(p1.context_weights.keys())
        contexts2 = set(p2.context_weights.keys())
        
        if contexts1.intersection(contexts2):
            # Check description similarity (simple word overlap)
            words1 = set(p1.description.lower().split())
            words2 = set(p2.description.lower().split())
            overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
            
            return overlap > 0.5
        
        return False
    
    def _merge_principle_group(self, principles: List[Principle]) -> Principle:
        """Merge a group of similar principles."""
        # Average strengths and consistencies
        avg_strength = np.mean([p.strength for p in principles])
        avg_consistency = np.mean([p.consistency for p in principles])
        
        # Combine descriptions
        combined_desc = "; ".join(set(p.description for p in principles))
        
        # Merge context weights
        all_contexts = {}
        for p in principles:
            for context, weight in p.context_weights.items():
                if context in all_contexts:
                    all_contexts[context] = max(all_contexts[context], weight)
                else:
                    all_contexts[context] = weight
        
        # Combine supporting patterns
        all_patterns = []
        for p in principles:
            all_patterns.extend(p.supporting_patterns)
        
        return Principle(
            principle_id=f"merged_{principles[0].principle_id}",
            description=combined_desc,
            strength=avg_strength,
            consistency=avg_consistency,
            first_observed=min(p.first_observed for p in principles),
            last_updated=max(p.last_updated for p in principles),
            supporting_patterns=list(set(all_patterns)),
            context_weights=all_contexts
        )
