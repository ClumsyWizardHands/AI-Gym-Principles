"""
Neural Network-based inference plugin.

This plugin uses neural networks to identify complex behavioral patterns
and extract high-level principles from agent actions.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import structlog
from datetime import datetime, timedelta

from ..decorators import register_plugin, plugin_config, cached_method, validate_input
from ..base import InferencePlugin
from ...core.models import Action, Principle, DecisionContext

logger = structlog.get_logger()


@register_plugin(
    name="neural_network_inference",
    version="1.0.0",
    author="AI Gym Team",
    description="Uses neural networks to discover complex behavioral patterns and infer principles",
    dependencies=["numpy", "tensorflow"],
    tags=["neural-network", "deep-learning", "pattern-detection", "advanced"]
)
@plugin_config(
    model_architecture="lstm",  # lstm, transformer, or cnn
    hidden_dim=128,
    num_layers=3,
    sequence_length=20,
    batch_size=32,
    learning_rate=0.001,
    confidence_threshold=0.8,
    min_pattern_length=5,
    use_attention=True,
    embedding_dim=64
)
class NeuralNetworkInferencePlugin(InferencePlugin):
    """
    Advanced inference plugin using neural networks for pattern recognition.
    
    This plugin employs deep learning techniques to:
    - Learn complex temporal dependencies in action sequences
    - Extract hierarchical patterns from behavior
    - Infer abstract principles from learned representations
    """
    
    def initialize(self) -> None:
        """Initialize the neural network model."""
        super().initialize()
        
        # Import TensorFlow only when needed
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            logger.error(
                "TensorFlow not installed. This is an optional dependency for the Neural Network Inference Plugin. "
                "Install with: pip install -r requirements-tensorflow.txt"
            )
            raise ImportError(
                "TensorFlow is required for NeuralNetworkInferencePlugin. "
                "Please install it with: pip install -r requirements-tensorflow.txt"
            )
        
        # Build the model based on configuration
        self.model = self._build_model()
        self.action_encoder = {}  # Cache for action encodings
        self.pattern_embeddings = {}  # Store learned pattern embeddings
        
    def extract_patterns(self, actions: List[Action]) -> List[Dict[str, Any]]:
        """
        Extract patterns using neural network analysis.
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            List of discovered patterns as dictionaries with confidence scores
        """
        if len(actions) < self.config["min_pattern_length"]:
            logger.warning(f"Not enough actions ({len(actions)}) for neural analysis")
            return []
        
        # Encode actions into sequences
        sequences = self._create_action_sequences(actions)
        
        # Process through neural network
        pattern_representations = self._process_sequences(sequences)
        
        # Cluster similar representations to form patterns
        patterns = self._extract_patterns_from_representations(
            pattern_representations, actions
        )
        
        logger.info(f"Extracted {len(patterns)} neural patterns from {len(actions)} actions")
        return patterns
    
    def infer_principles(self, patterns: List[Dict[str, Any]]) -> List[Principle]:
        """
        Infer high-level principles from neural pattern analysis.
        
        Args:
            patterns: List of patterns (as dictionaries) to analyze
            
        Returns:
            List of inferred principles with neural confidence scores
        """
        if not patterns:
            return []
        
        # Extract high-level features from patterns
        pattern_features = self._extract_pattern_features(patterns)
        
        # Use neural network to identify principle clusters
        principle_clusters = self._identify_principle_clusters(pattern_features)
        
        # Generate principles from clusters
        principles = []
        for cluster_id, cluster_data in enumerate(principle_clusters):
            principle = self._generate_principle_from_cluster(
                cluster_data, patterns
            )
            if principle.strength >= self.config["confidence_threshold"]:
                principles.append(principle)
        
        # Apply neural refinement to principles
        refined_principles = self._refine_principles_neurally(principles)
        
        logger.info(f"Inferred {len(refined_principles)} principles from {len(patterns)} patterns")
        return refined_principles
    
    def _build_model(self) -> Any:
        """Build the neural network model based on configuration."""
        architecture = self.config["model_architecture"]
        
        if architecture == "lstm":
            return self._build_lstm_model()
        elif architecture == "transformer":
            return self._build_transformer_model()
        elif architecture == "cnn":
            return self._build_cnn_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_lstm_model(self) -> Any:
        """Build an LSTM-based model for sequence analysis."""
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.Embedding(
                input_dim=10000,  # Vocabulary size
                output_dim=self.config["embedding_dim"]
            ),
            *[self.tf.keras.layers.LSTM(
                self.config["hidden_dim"],
                return_sequences=(i < self.config["num_layers"] - 1),
                dropout=0.2
            ) for i in range(self.config["num_layers"])],
            self.tf.keras.layers.Dense(128, activation='relu'),
            self.tf.keras.layers.Dense(64, activation='relu')
        ])
        
        if self.config["use_attention"]:
            # Add attention mechanism
            attention_layer = self.tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.config["hidden_dim"] // 4
            )
            model.add(attention_layer)
        
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(self.config["learning_rate"]),
            loss='mse'
        )
        
        return model
    
    def _build_transformer_model(self) -> Any:
        """Build a Transformer-based model."""
        # Simplified transformer for demonstration
        inputs = self.tf.keras.Input(shape=(self.config["sequence_length"],))
        
        # Embedding
        x = self.tf.keras.layers.Embedding(
            input_dim=10000,
            output_dim=self.config["embedding_dim"]
        )(inputs)
        
        # Positional encoding
        positions = self.tf.range(start=0, limit=self.config["sequence_length"], delta=1)
        pos_encoding = self.tf.keras.layers.Embedding(
            input_dim=self.config["sequence_length"],
            output_dim=self.config["embedding_dim"]
        )(positions)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.config["num_layers"]):
            # Multi-head attention
            attn_output = self.tf.keras.layers.MultiHeadAttention(
                num_heads=8,
                key_dim=self.config["embedding_dim"] // 8
            )(x, x)
            x = self.tf.keras.layers.Add()([x, attn_output])
            x = self.tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed forward
            ff_output = self.tf.keras.Sequential([
                self.tf.keras.layers.Dense(self.config["hidden_dim"], activation='relu'),
                self.tf.keras.layers.Dense(self.config["embedding_dim"])
            ])(x)
            x = self.tf.keras.layers.Add()([x, ff_output])
            x = self.tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = self.tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = self.tf.keras.layers.Dense(128)(x)
        
        model = self.tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(self.config["learning_rate"]),
            loss='mse'
        )
        
        return model
    
    def _build_cnn_model(self) -> Any:
        """Build a CNN-based model for pattern recognition."""
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.Embedding(
                input_dim=10000,
                output_dim=self.config["embedding_dim"]
            ),
            self.tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation='relu',
                padding='same'
            ),
            self.tf.keras.layers.MaxPooling1D(pool_size=2),
            self.tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                activation='relu',
                padding='same'
            ),
            self.tf.keras.layers.GlobalMaxPooling1D(),
            self.tf.keras.layers.Dense(128, activation='relu'),
            self.tf.keras.layers.Dropout(0.3),
            self.tf.keras.layers.Dense(64)
        ])
        
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(self.config["learning_rate"]),
            loss='mse'
        )
        
        return model
    
    @cached_method(ttl=300)
    def _create_action_sequences(self, actions: List[Action]) -> np.ndarray:
        """Create sequences of encoded actions for neural processing."""
        sequence_length = self.config["sequence_length"]
        sequences = []
        
        # Encode actions
        encoded_actions = [self._encode_action(action) for action in actions]
        
        # Create overlapping sequences
        for i in range(len(encoded_actions) - sequence_length + 1):
            sequence = encoded_actions[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _encode_action(self, action: Action) -> int:
        """Encode an action into a numerical representation."""
        # Create a unique key for the action
        action_key = f"{action.decision_context.name}_{action.action_type}"
        
        if action_key not in self.action_encoder:
            # Assign new encoding
            self.action_encoder[action_key] = len(self.action_encoder) + 1
        
        return self.action_encoder[action_key]
    
    def _process_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Process action sequences through the neural network."""
        # Process in batches
        batch_size = self.config["batch_size"]
        representations = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # Get neural representations
            batch_representations = self.model.predict(batch, verbose=0)
            representations.extend(batch_representations)
        
        return np.array(representations)
    
    def _extract_patterns_from_representations(
        self,
        representations: np.ndarray,
        original_actions: List[Action]
    ) -> List[Dict[str, Any]]:
        """Extract patterns from neural representations."""
        from sklearn.cluster import DBSCAN
        
        # Cluster similar representations
        clustering = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = clustering.fit_predict(representations)
        
        patterns = []
        unique_labels = set(cluster_labels) - {-1}  # Exclude noise
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            # Create pattern from cluster
            pattern = self._create_neural_pattern(
                cluster_indices,
                representations[cluster_indices],
                original_actions
            )
            patterns.append(pattern)
        
        return patterns
    
    def _create_neural_pattern(
        self,
        indices: np.ndarray,
        representations: np.ndarray,
        actions: List[Action]
    ) -> Dict[str, Any]:
        """Create a pattern dictionary from neural cluster analysis."""
        # Calculate pattern properties
        sequence_length = self.config["sequence_length"]
        
        # Get action sequences for this pattern
        action_sequences = []
        for idx in indices:
            start_idx = idx
            end_idx = min(idx + sequence_length, len(actions))
            action_seq = [actions[i].action_type for i in range(start_idx, end_idx)]
            action_sequences.append(action_seq)
        
        # Find most common sequence
        from collections import Counter
        seq_counter = Counter(tuple(seq) for seq in action_sequences)
        most_common_seq = list(seq_counter.most_common(1)[0][0])
        
        # Calculate pattern metrics
        avg_representation = np.mean(representations, axis=0)
        pattern_variance = np.var(representations, axis=0).mean()
        confidence = 1.0 / (1.0 + pattern_variance)  # Lower variance = higher confidence
        
        # Store embedding for later use
        pattern_id = f"neural_pattern_{len(self.pattern_embeddings)}"
        self.pattern_embeddings[pattern_id] = avg_representation
        
        # Get temporal information
        pattern_actions = [actions[i] for idx in indices for i in range(idx, min(idx + sequence_length, len(actions)))]
        timestamps = [a.timestamp for a in pattern_actions]
        
        return {
            "pattern_id": pattern_id,
            "action_sequence": most_common_seq,
            "frequency": len(indices) / len(actions),
            "confidence": float(confidence),
            "context_distribution": self._calculate_context_distribution(pattern_actions),
            "first_seen": min(timestamps),
            "last_seen": max(timestamps),
            "occurrence_count": len(indices),
            "metadata": {
                "neural_confidence": float(confidence),
                "embedding_variance": float(pattern_variance),
                "cluster_size": len(indices)
            }
        }
    
    def _calculate_context_distribution(self, actions: List[Action]) -> Dict[str, float]:
        """Calculate the distribution of contexts for actions."""
        from collections import Counter
        context_counts = Counter(action.decision_context.name for action in actions)
        total = sum(context_counts.values())
        
        return {
            context: count / total
            for context, count in context_counts.items()
        }
    
    def _extract_pattern_features(self, patterns: List[Dict[str, Any]]) -> np.ndarray:
        """Extract neural features from patterns for principle inference."""
        features = []
        
        for pattern in patterns:
            # Use stored embeddings if available
            pattern_id = pattern.get("pattern_id", "")
            if pattern_id in self.pattern_embeddings:
                embedding = self.pattern_embeddings[pattern_id]
            else:
                # Create feature vector from pattern properties
                embedding = np.array([
                    pattern.get("frequency", 0.0),
                    pattern.get("confidence", 0.0),
                    pattern.get("occurrence_count", 0),
                    len(pattern.get("action_sequence", [])),
                    len(pattern.get("context_distribution", {}))
                ])
            
            features.append(embedding)
        
        return np.array(features)
    
    def _identify_principle_clusters(self, pattern_features: np.ndarray) -> List[Dict[str, Any]]:
        """Identify principle clusters using neural analysis."""
        from sklearn.cluster import KMeans
        
        # Determine optimal number of clusters
        n_clusters = min(5, len(pattern_features) // 3)
        
        if n_clusters < 2:
            # Not enough data for clustering
            return [{
                "features": pattern_features,
                "indices": list(range(len(pattern_features))),
                "confidence": 0.5
            }]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pattern_features)
        
        # Organize clusters
        clusters = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_features = pattern_features[cluster_indices]
            
            # Calculate cluster confidence based on cohesion
            distances = np.linalg.norm(
                cluster_features - kmeans.cluster_centers_[i],
                axis=1
            )
            cohesion = 1.0 / (1.0 + np.mean(distances))
            
            clusters.append({
                "features": cluster_features,
                "indices": cluster_indices.tolist(),
                "center": kmeans.cluster_centers_[i],
                "confidence": float(cohesion)
            })
        
        return clusters
    
    def _generate_principle_from_cluster(
        self,
        cluster_data: Dict[str, Any],
        patterns: List[Pattern]
    ) -> Principle:
        """Generate a principle from a neural cluster analysis."""
        cluster_patterns = [patterns[i] for i in cluster_data["indices"]]
        
        # Analyze common elements in cluster patterns
        all_actions = []
        all_contexts = {}
        
        for pattern in cluster_patterns:
            all_actions.extend(pattern.action_sequence)
            for context, weight in pattern.context_distribution.items():
                if context not in all_contexts:
                    all_contexts[context] = []
                all_contexts[context].append(weight)
        
        # Find dominant action and context
        from collections import Counter
        action_counter = Counter(all_actions)
        dominant_action = action_counter.most_common(1)[0][0] if action_counter else "adaptive behavior"
        
        # Calculate context weights
        context_weights = {
            context.name: np.mean(weights)
            for context, weights in all_contexts.items()
        }
        
        # Generate principle description using neural insights
        description = self._generate_neural_principle_description(
            dominant_action,
            context_weights,
            cluster_data["confidence"]
        )
        
        # Calculate principle strength
        avg_pattern_confidence = np.mean([p.confidence for p in cluster_patterns])
        strength = avg_pattern_confidence * cluster_data["confidence"]
        
        return Principle(
            principle_id=f"neural_principle_{hash(description) % 10000}",
            description=description,
            strength=float(strength),
            consistency=float(avg_pattern_confidence),
            first_observed=min(p.first_seen for p in cluster_patterns),
            last_updated=max(p.last_seen for p in cluster_patterns),
            supporting_patterns=[p.pattern_id for p in cluster_patterns],
            context_weights=context_weights,
            metadata={
                "neural_cluster_confidence": cluster_data["confidence"],
                "cluster_size": len(cluster_patterns),
                "inference_method": "neural_network"
            }
        )
    
    def _generate_neural_principle_description(
        self,
        dominant_action: str,
        context_weights: Dict[str, float],
        confidence: float
    ) -> str:
        """Generate a principle description based on neural analysis."""
        # Sort contexts by weight
        sorted_contexts = sorted(
            context_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if len(sorted_contexts) > 1 and sorted_contexts[0][1] > 0.7:
            # Strong context preference
            return f"Strongly prefer '{dominant_action}' in {sorted_contexts[0][0]} contexts (neural confidence: {confidence:.2f})"
        elif len(sorted_contexts) > 2:
            # Multiple contexts
            top_contexts = ", ".join([c[0] for c in sorted_contexts[:3]])
            return f"Exhibit '{dominant_action}' behavior across multiple contexts: {top_contexts}"
        else:
            # General principle
            return f"Demonstrate consistent '{dominant_action}' behavior pattern (neural confidence: {confidence:.2f})"
    
    def _refine_principles_neurally(self, principles: List[Principle]) -> List[Principle]:
        """Apply neural refinement to improve principle quality."""
        if len(principles) <= 1:
            return principles
        
        # Create principle embeddings
        principle_features = []
        for principle in principles:
            feature_vec = self._create_principle_embedding(principle)
            principle_features.append(feature_vec)
        
        principle_features = np.array(principle_features)
        
        # Use neural network to identify redundant or conflicting principles
        similarity_matrix = self._calculate_neural_similarity(principle_features)
        
        # Merge similar principles
        merged_principles = []
        merged_indices = set()
        
        for i, principle in enumerate(principles):
            if i in merged_indices:
                continue
            
            # Find similar principles
            similar_indices = np.where(similarity_matrix[i] > 0.8)[0]
            similar_indices = [idx for idx in similar_indices if idx != i and idx not in merged_indices]
            
            if similar_indices:
                # Merge with similar principles
                similar_principles = [principles[idx] for idx in similar_indices]
                merged_principle = self._merge_principles_neurally(
                    [principle] + similar_principles
                )
                merged_principles.append(merged_principle)
                merged_indices.update([i] + list(similar_indices))
            else:
                merged_principles.append(principle)
        
        return merged_principles
    
    def _create_principle_embedding(self, principle: Principle) -> np.ndarray:
        """Create a neural embedding for a principle."""
        # Combine various principle features
        embedding = []
        
        # Numerical features
        embedding.extend([
            principle.strength,
            principle.consistency,
            len(principle.supporting_patterns),
            len(principle.context_weights)
        ])
        
        # Context distribution
        context_vector = np.zeros(10)  # Fixed size vector
        for i, (_, weight) in enumerate(principle.context_weights.items()):
            if i < 10:
                context_vector[i] = weight
        embedding.extend(context_vector)
        
        # Text features (simplified)
        text_hash = hash(principle.description) % 1000
        embedding.append(text_hash / 1000.0)
        
        return np.array(embedding)
    
    def _calculate_neural_similarity(self, features: np.ndarray) -> np.ndarray:
        """Calculate pairwise similarity using neural distance metrics."""
        n = len(features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Cosine similarity
                    dot_product = np.dot(features[i], features[j])
                    norm_product = np.linalg.norm(features[i]) * np.linalg.norm(features[j])
                    
                    if norm_product > 0:
                        similarity = dot_product / norm_product
                        similarity_matrix[i, j] = max(0, similarity)
        
        return similarity_matrix
    
    def _merge_principles_neurally(self, principles: List[Principle]) -> Principle:
        """Merge similar principles using neural analysis."""
        # Weight principles by their neural confidence
        weights = [p.strength * p.consistency for p in principles]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average of strengths
        merged_strength = sum(p.strength * w for p, w in zip(principles, normalized_weights))
        merged_consistency = sum(p.consistency * w for p, w in zip(principles, normalized_weights))
        
        # Combine descriptions intelligently
        descriptions = [p.description for p in principles]
        merged_description = self._synthesize_descriptions(descriptions, normalized_weights)
        
        # Merge context weights
        all_contexts = {}
        for principle, weight in zip(principles, normalized_weights):
            for context, context_weight in principle.context_weights.items():
                if context in all_contexts:
                    all_contexts[context] += context_weight * weight
                else:
                    all_contexts[context] = context_weight * weight
        
        # Normalize context weights
        total_context_weight = sum(all_contexts.values())
        if total_context_weight > 0:
            all_contexts = {k: v / total_context_weight for k, v in all_contexts.items()}
        
        # Combine supporting patterns
        all_patterns = []
        for p in principles:
            all_patterns.extend(p.supporting_patterns)
        
        return Principle(
            principle_id=f"neural_merged_{principles[0].principle_id}",
            description=merged_description,
            strength=float(merged_strength),
            consistency=float(merged_consistency),
            first_observed=min(p.first_observed for p in principles),
            last_updated=max(p.last_updated for p in principles),
            supporting_patterns=list(set(all_patterns)),
            context_weights=all_contexts,
            metadata={
                "merge_count": len(principles),
                "merge_method": "neural_weighted",
                "inference_method": "neural_network"
            }
        )
    
    def _synthesize_descriptions(
        self,
        descriptions: List[str],
        weights: List[float]
    ) -> str:
        """Synthesize multiple principle descriptions into one."""
        # Extract key terms from descriptions
        from collections import Counter
        
        all_words = []
        for desc, weight in zip(descriptions, weights):
            words = desc.lower().split()
            # Weight word frequency by principle weight
            all_words.extend(words * int(weight * 10))
        
        # Find most important words
        word_counter = Counter(all_words)
        important_words = [word for word, _ in word_counter.most_common(5)]
        
        # Generate synthesized description
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"Balanced principle combining: {descriptions[0]} and {descriptions[1]}"
        else:
            key_terms = " ".join(important_words[:3])
            return f"Unified principle emphasizing {key_terms} (synthesized from {len(descriptions)} patterns)"
