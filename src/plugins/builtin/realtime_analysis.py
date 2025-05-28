"""
Real-time Analysis Plugin.

This plugin provides real-time analysis of agent behaviors during training sessions,
including pattern detection, anomaly identification, and performance monitoring.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import asyncio
import structlog

from ..decorators import register_plugin, plugin_config, cached_method, validate_input
from ..base import AnalysisPlugin
from ...core.models import Action, Principle, AgentProfile
from ...core.inference import PatternMatcher

logger = structlog.get_logger()


@register_plugin(
    name="realtime_analysis",
    version="1.0.0",
    author="AI Gym Team",
    description="Provides real-time analysis and monitoring of agent behaviors during training",
    dependencies=["numpy", "scipy"],
    tags=["realtime", "monitoring", "anomaly-detection", "performance", "streaming"]
)
@plugin_config(
    window_size=50,  # Number of actions to keep in sliding window
    update_frequency=5,  # Analyze every N actions
    anomaly_threshold=2.5,  # Standard deviations for anomaly detection
    pattern_min_frequency=0.1,  # Minimum frequency for pattern detection
    enable_predictive_analysis=True,
    enable_comparative_analysis=True,
    alert_thresholds={
        "consistency_drop": 0.3,
        "entropy_spike": 0.8,
        "principle_conflict": 0.5
    },
    export_formats=["json", "csv", "html", "websocket"]
)
class RealtimeAnalysisPlugin(AnalysisPlugin):
    """
    Real-time analysis plugin for monitoring agent behaviors.
    
    Features:
    - Sliding window analysis of recent actions
    - Anomaly detection using statistical methods
    - Real-time pattern emergence tracking
    - Performance metrics calculation
    - Predictive behavior modeling
    - Multi-agent comparative analysis
    """
    
    def initialize(self) -> None:
        """Initialize real-time analysis components."""
        super().initialize()
        
        # Sliding windows for different metrics
        self.action_window = deque(maxlen=self.config["window_size"])
        self.pattern_window = deque(maxlen=self.config["window_size"] // 2)
        self.principle_window = deque(maxlen=self.config["window_size"] // 3)
        
        # Real-time metrics storage
        self.metrics_history = {
            "behavioral_entropy": deque(maxlen=100),
            "consistency_score": deque(maxlen=100),
            "pattern_diversity": deque(maxlen=100),
            "decision_time": deque(maxlen=100),
            "principle_alignment": deque(maxlen=100)
        }
        
        # Anomaly detection baselines
        self.baselines = {}
        self.anomaly_counts = defaultdict(int)
        
        # Pattern matcher for real-time pattern detection
        self.pattern_matcher = PatternMatcher()
        
        # Agent profiles for comparative analysis
        self.agent_profiles: Dict[str, AgentProfile] = {}
        
        # Alert system
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Streaming connections
        self.stream_handlers: List[Any] = []
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform real-time analysis on incoming data.
        
        Args:
            data: Dictionary containing:
                - actions: Recent actions to analyze
                - patterns: Detected patterns (optional)
                - principles: Active principles (optional)
                - agent_id: Agent identifier
                - timestamp: Current timestamp
                
        Returns:
            Real-time analysis results
        """
        actions = data.get("actions", [])
        agent_id = data.get("agent_id", "unknown")
        timestamp = data.get("timestamp", datetime.now())
        
        # Update sliding windows
        self._update_windows(actions, data.get("patterns", []), data.get("principles", []))
        
        # Perform real-time analyses
        analysis_results = {
            "timestamp": timestamp.isoformat(),
            "agent_id": agent_id,
            "window_size": len(self.action_window),
            "metrics": self._calculate_realtime_metrics(),
            "anomalies": self._detect_anomalies(),
            "emerging_patterns": self._identify_emerging_patterns(),
            "performance_indicators": self._calculate_performance_indicators(),
            "alerts": self._check_alert_conditions()
        }
        
        # Predictive analysis if enabled
        if self.config["enable_predictive_analysis"]:
            analysis_results["predictions"] = self._generate_predictions()
        
        # Comparative analysis if enabled and multiple agents
        if self.config["enable_comparative_analysis"] and len(self.agent_profiles) > 1:
            analysis_results["comparative"] = self._perform_comparative_analysis(agent_id)
        
        # Stream results if handlers are connected
        self._stream_results(analysis_results)
        
        return analysis_results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """
        Generate a real-time analysis report.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            Report in the requested format
        """
        format_type = analysis_results.get("format", "json")
        
        if format_type == "json":
            return self._generate_json_report(analysis_results)
        elif format_type == "csv":
            return self._generate_csv_report(analysis_results)
        elif format_type == "html":
            return self._generate_html_report(analysis_results)
        elif format_type == "websocket":
            return self._generate_websocket_message(analysis_results)
        else:
            return analysis_results
    
    def _update_windows(self, actions: List[Action], patterns: List[Dict[str, Any]], 
                        principles: List[Principle]) -> None:
        """Update sliding windows with new data."""
        # Add actions to window
        for action in actions:
            self.action_window.append(action)
        
        # Add patterns (as dictionaries)
        for pattern in patterns:
            self.pattern_window.append(pattern)
        
        # Add principles
        for principle in principles:
            self.principle_window.append(principle)
    
    @cached_method(ttl=10)  # Short TTL for real-time data
    def _calculate_realtime_metrics(self) -> Dict[str, float]:
        """Calculate real-time behavioral metrics."""
        if len(self.action_window) < 5:
            return {
                "behavioral_entropy": 0.0,
                "consistency_score": 0.0,
                "pattern_diversity": 0.0,
                "avg_decision_time": 0.0,
                "principle_alignment": 0.0
            }
        
        # Behavioral entropy
        action_types = [a.action_taken for a in self.action_window]
        action_counts = defaultdict(int)
        for action in action_types:
            action_counts[action] += 1
        
        total_actions = len(action_types)
        entropy = 0.0
        for count in action_counts.values():
            if count > 0:
                p = count / total_actions
                entropy -= p * np.log2(p)
        
        # Consistency score (how similar are recent actions)
        if len(self.action_window) >= 2:
            consistency_pairs = 0
            for i in range(1, len(self.action_window)):
                if self.action_window[i].action_taken == self.action_window[i-1].action_taken:
                    consistency_pairs += 1
            consistency = consistency_pairs / (len(self.action_window) - 1)
        else:
            consistency = 0.0
        
        # Pattern diversity
        unique_patterns = set()
        for pattern in self.pattern_window:
            unique_patterns.add(tuple(pattern.action_sequence))
        pattern_diversity = len(unique_patterns) / max(1, len(self.pattern_window))
        
        # Average decision time
        decision_times = []
        for action in self.action_window:
            if hasattr(action, 'decision_time') and action.decision_time:
                decision_times.append(action.decision_time)
        avg_decision_time = np.mean(decision_times) if decision_times else 0.0
        
        # Principle alignment
        if self.principle_window:
            alignment_scores = [p.strength * p.consistency for p in self.principle_window]
            principle_alignment = np.mean(alignment_scores)
        else:
            principle_alignment = 0.0
        
        metrics = {
            "behavioral_entropy": float(entropy),
            "consistency_score": float(consistency),
            "pattern_diversity": float(pattern_diversity),
            "avg_decision_time": float(avg_decision_time),
            "principle_alignment": float(principle_alignment)
        }
        
        # Update history
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
        
        return metrics
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in real-time behavior."""
        anomalies = []
        
        # Check each metric for anomalies
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:
                continue
            
            # Calculate baseline statistics
            values = list(history)
            mean = np.mean(values[:-1])  # Exclude current value
            std = np.std(values[:-1])
            
            if std > 0:
                current_value = values[-1]
                z_score = abs((current_value - mean) / std)
                
                if z_score > self.config["anomaly_threshold"]:
                    anomaly = {
                        "type": "metric_anomaly",
                        "metric": metric_name,
                        "value": current_value,
                        "expected_range": (mean - 2*std, mean + 2*std),
                        "z_score": float(z_score),
                        "severity": self._calculate_anomaly_severity(z_score),
                        "timestamp": datetime.now().isoformat()
                    }
                    anomalies.append(anomaly)
                    self.anomaly_counts[metric_name] += 1
        
        # Check for action sequence anomalies
        if len(self.action_window) >= 5:
            recent_sequence = [a.action_taken for a in list(self.action_window)[-5:]]
            if self._is_sequence_anomalous(recent_sequence):
                anomalies.append({
                    "type": "sequence_anomaly",
                    "sequence": recent_sequence,
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat()
                })
        
        return anomalies
    
    def _identify_emerging_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns that are emerging in real-time."""
        if len(self.action_window) < 10:
            return []
        
        emerging_patterns = []
        
        # Extract sequences of different lengths
        for seq_length in [3, 4, 5]:
            sequences = self._extract_sequences(seq_length)
            
            for seq, count in sequences.items():
                frequency = count / (len(self.action_window) - seq_length + 1)
                
                if frequency >= self.config["pattern_min_frequency"]:
                    # Check if this is a new or strengthening pattern
                    pattern_id = f"rt_pattern_{hash(seq) % 10000}"
                    
                    pattern_info = {
                        "pattern_id": pattern_id,
                        "sequence": list(seq),
                        "frequency": float(frequency),
                        "count": count,
                        "length": seq_length,
                        "emergence_stage": self._classify_emergence_stage(frequency),
                        "predicted_strength": self._predict_pattern_strength(frequency, count)
                    }
                    
                    emerging_patterns.append(pattern_info)
        
        return emerging_patterns
    
    def _calculate_performance_indicators(self) -> Dict[str, Any]:
        """Calculate real-time performance indicators."""
        indicators = {
            "current_performance": "normal",
            "trend": "stable",
            "efficiency_score": 0.0,
            "adaptability_score": 0.0,
            "learning_rate": 0.0
        }
        
        if len(self.metrics_history["behavioral_entropy"]) < 5:
            return indicators
        
        # Performance classification based on entropy and consistency
        recent_entropy = list(self.metrics_history["behavioral_entropy"])[-5:]
        recent_consistency = list(self.metrics_history["consistency_score"])[-5:]
        
        avg_entropy = np.mean(recent_entropy)
        avg_consistency = np.mean(recent_consistency)
        
        # Classify current performance
        if avg_entropy > 0.8 and avg_consistency < 0.3:
            indicators["current_performance"] = "chaotic"
        elif avg_entropy < 0.2 and avg_consistency > 0.8:
            indicators["current_performance"] = "rigid"
        elif 0.4 <= avg_entropy <= 0.7 and 0.4 <= avg_consistency <= 0.7:
            indicators["current_performance"] = "optimal"
        
        # Trend analysis
        entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
        if entropy_trend > 0.1:
            indicators["trend"] = "increasing_exploration"
        elif entropy_trend < -0.1:
            indicators["trend"] = "increasing_exploitation"
        
        # Efficiency score (quick decisions with good outcomes)
        if self.action_window:
            decision_times = [a.decision_time for a in self.action_window 
                            if hasattr(a, 'decision_time') and a.decision_time]
            if decision_times:
                avg_time = np.mean(decision_times)
                # Normalize to 0-1 (assuming 5 seconds is slow)
                indicators["efficiency_score"] = float(1.0 - min(avg_time / 5.0, 1.0))
        
        # Adaptability score (how well agent responds to context changes)
        context_changes = self._count_context_changes()
        if context_changes > 0:
            adaptation_success = self._measure_adaptation_success()
            indicators["adaptability_score"] = float(adaptation_success)
        
        # Learning rate (improvement in principle strength over time)
        if len(self.principle_window) >= 2:
            early_principles = list(self.principle_window)[:len(self.principle_window)//2]
            recent_principles = list(self.principle_window)[len(self.principle_window)//2:]
            
            early_strength = np.mean([p.strength for p in early_principles]) if early_principles else 0
            recent_strength = np.mean([p.strength for p in recent_principles]) if recent_principles else 0
            
            indicators["learning_rate"] = float(recent_strength - early_strength)
        
        return indicators
    
    def _check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for conditions that should trigger alerts."""
        new_alerts = []
        thresholds = self.config["alert_thresholds"]
        
        # Check consistency drop
        if len(self.metrics_history["consistency_score"]) >= 10:
            recent_consistency = list(self.metrics_history["consistency_score"])[-5:]
            older_consistency = list(self.metrics_history["consistency_score"])[-10:-5]
            
            if np.mean(older_consistency) - np.mean(recent_consistency) > thresholds["consistency_drop"]:
                new_alerts.append({
                    "type": "consistency_drop",
                    "severity": "high",
                    "message": "Significant drop in behavioral consistency detected",
                    "current_value": float(np.mean(recent_consistency)),
                    "previous_value": float(np.mean(older_consistency)),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check entropy spike
        if len(self.metrics_history["behavioral_entropy"]) >= 5:
            recent_entropy = list(self.metrics_history["behavioral_entropy"])[-1]
            baseline_entropy = np.mean(list(self.metrics_history["behavioral_entropy"])[:-1])
            
            if recent_entropy > baseline_entropy + thresholds["entropy_spike"]:
                new_alerts.append({
                    "type": "entropy_spike",
                    "severity": "medium",
                    "message": "Sudden increase in behavioral randomness",
                    "current_value": float(recent_entropy),
                    "baseline_value": float(baseline_entropy),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check principle conflicts
        if len(self.principle_window) >= 2:
            conflicts = self._detect_principle_conflicts()
            if conflicts:
                for conflict in conflicts:
                    if conflict["severity_score"] > thresholds["principle_conflict"]:
                        new_alerts.append({
                            "type": "principle_conflict",
                            "severity": "high",
                            "message": f"Conflicting principles detected: {conflict['principle1']} vs {conflict['principle2']}",
                            "conflict_score": conflict["severity_score"],
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Add new alerts to active list
        self.active_alerts.extend(new_alerts)
        self.alert_history.extend(new_alerts)
        
        # Clean old alerts
        self._clean_old_alerts()
        
        return new_alerts
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions about future behavior."""
        predictions = {
            "next_action_probabilities": {},
            "pattern_emergence_likelihood": {},
            "performance_forecast": {},
            "anomaly_risk": 0.0
        }
        
        if len(self.action_window) < 10:
            return predictions
        
        # Predict next action based on recent patterns
        recent_actions = [a.action_taken for a in list(self.action_window)[-10:]]
        action_counts = defaultdict(int)
        
        # Simple n-gram prediction
        for i in range(len(recent_actions) - 2):
            trigram = tuple(recent_actions[i:i+3])
            if i < len(recent_actions) - 3:
                next_action = recent_actions[i+3]
                action_counts[next_action] += 1
        
        total = sum(action_counts.values())
        if total > 0:
            predictions["next_action_probabilities"] = {
                action: count/total for action, count in action_counts.items()
            }
        
        # Predict pattern emergence
        for pattern in self._identify_emerging_patterns():
            pattern_id = pattern["pattern_id"]
            predictions["pattern_emergence_likelihood"][pattern_id] = {
                "current_frequency": pattern["frequency"],
                "predicted_strength": pattern["predicted_strength"],
                "confidence": 0.7  # Simplified confidence
            }
        
        # Performance forecast
        if len(self.metrics_history["behavioral_entropy"]) >= 20:
            # Simple linear extrapolation
            entropy_values = list(self.metrics_history["behavioral_entropy"])
            x = np.arange(len(entropy_values))
            coeffs = np.polyfit(x, entropy_values, 2)  # Quadratic fit
            
            # Predict next 5 steps
            future_x = np.arange(len(entropy_values), len(entropy_values) + 5)
            future_entropy = np.polyval(coeffs, future_x)
            
            predictions["performance_forecast"] = {
                "entropy_trend": future_entropy.tolist(),
                "expected_performance": self._classify_performance(future_entropy[-1])
            }
        
        # Anomaly risk assessment
        recent_anomalies = sum(1 for a in self.alert_history 
                             if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(minutes=5))
        predictions["anomaly_risk"] = float(min(recent_anomalies / 5, 1.0))
        
        return predictions
    
    def _perform_comparative_analysis(self, current_agent_id: str) -> Dict[str, Any]:
        """Compare current agent with other agents."""
        if current_agent_id not in self.agent_profiles:
            return {}
        
        current_profile = self.agent_profiles[current_agent_id]
        comparisons = {}
        
        for agent_id, profile in self.agent_profiles.items():
            if agent_id == current_agent_id:
                continue
            
            comparison = {
                "behavioral_similarity": self._calculate_behavioral_similarity(
                    current_profile, profile
                ),
                "performance_difference": self._calculate_performance_difference(
                    current_profile, profile
                ),
                "principle_overlap": self._calculate_principle_overlap(
                    current_profile, profile
                ),
                "relative_efficiency": self._calculate_relative_efficiency(
                    current_profile, profile
                )
            }
            
            comparisons[agent_id] = comparison
        
        # Rank agents by similarity
        similarity_ranking = sorted(
            comparisons.items(),
            key=lambda x: x[1]["behavioral_similarity"],
            reverse=True
        )
        
        return {
            "comparisons": comparisons,
            "most_similar_agent": similarity_ranking[0][0] if similarity_ranking else None,
            "performance_rank": self._calculate_performance_rank(current_agent_id),
            "unique_behaviors": self._identify_unique_behaviors(current_agent_id)
        }
    
    def _stream_results(self, results: Dict[str, Any]) -> None:
        """Stream results to connected handlers."""
        for handler in self.stream_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(results))
                else:
                    handler(results)
            except Exception as e:
                logger.error(f"Error streaming results: {e}")
    
    # Helper methods
    
    def _is_sequence_anomalous(self, sequence: List[str]) -> bool:
        """Check if an action sequence is anomalous."""
        # Simple check: all actions are the same (stuck behavior)
        if len(set(sequence)) == 1:
            return True
        
        # Rapid switching between two actions
        if len(set(sequence)) == 2 and len(sequence) >= 4:
            switches = sum(1 for i in range(1, len(sequence)) 
                         if sequence[i] != sequence[i-1])
            if switches >= len(sequence) - 1:
                return True
        
        return False
    
    def _calculate_anomaly_severity(self, z_score: float) -> str:
        """Calculate severity of an anomaly based on z-score."""
        if z_score < 3:
            return "low"
        elif z_score < 4:
            return "medium"
        else:
            return "high"
    
    def _extract_sequences(self, length: int) -> Dict[Tuple[str, ...], int]:
        """Extract action sequences of given length."""
        sequences = defaultdict(int)
        actions = [a.action_taken for a in self.action_window]
        
        for i in range(len(actions) - length + 1):
            seq = tuple(actions[i:i+length])
            sequences[seq] += 1
        
        return sequences
    
    def _classify_emergence_stage(self, frequency: float) -> str:
        """Classify the emergence stage of a pattern."""
        if frequency < 0.2:
            return "nascent"
        elif frequency < 0.4:
            return "emerging"
        elif frequency < 0.6:
            return "establishing"
        else:
            return "established"
    
    def _predict_pattern_strength(self, frequency: float, count: int) -> float:
        """Predict future strength of a pattern."""
        # Simple prediction based on frequency and occurrence count
        base_strength = frequency
        count_factor = min(count / 10, 1.0)  # Saturates at 10 occurrences
        
        return float(base_strength * 0.7 + count_factor * 0.3)
    
    def _count_context_changes(self) -> int:
        """Count number of context changes in action window."""
        if len(self.action_window) < 2:
            return 0
        
        changes = 0
        for i in range(1, len(self.action_window)):
            if self.action_window[i].context != self.action_window[i-1].context:
                changes += 1
        
        return changes
    
    def _measure_adaptation_success(self) -> float:
        """Measure how well agent adapts to context changes."""
        # Simplified: check if actions change appropriately with context
        context_action_pairs = defaultdict(list)
        
        for action in self.action_window:
            context_action_pairs[action.context].append(action.action_taken)
        
        # Good adaptation = different contexts lead to different action distributions
        if len(context_action_pairs) < 2:
            return 0.5  # Not enough data
        
        # Calculate action distribution similarity between contexts
        distributions = []
        for context, actions in context_action_pairs.items():
            action_counts = defaultdict(int)
            for a in actions:
                action_counts[a] += 1
            total = len(actions)
            dist = {a: c/total for a, c in action_counts.items()}
            distributions.append(dist)
        
        # Compare distributions (lower similarity = better adaptation)
        similarities = []
        for i in range(len(distributions)):
            for j in range(i+1, len(distributions)):
                sim = self._calculate_distribution_similarity(distributions[i], distributions[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.5
        adaptation_score = 1.0 - avg_similarity
        
        return adaptation_score
    
    def _detect_principle_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between active principles."""
        conflicts = []
        principles = list(self.principle_window)
        
        for i in range(len(principles)):
            for j in range(i+1, len(principles)):
                p1, p2 = principles[i], principles[j]
                
                # Check for opposing context weights
                context_conflict = 0.0
                common_contexts = set(p1.context_weights.keys()) & set(p2.context_weights.keys())
                
                for context in common_contexts:
                    # If weights differ significantly, there's conflict
                    weight_diff = abs(p1.context_weights[context] - p2.context_weights[context])
                    context_conflict += weight_diff
                
                if context_conflict > 0.5:
                    conflicts.append({
                        "principle1": p1.description,
                        "principle2": p2.description,
                        "severity_score": float(context_conflict),
                        "type": "context_weight_conflict"
                    })
        
        return conflicts
    
    def _clean_old_alerts(self) -> None:
        """Remove old alerts from active list."""
        current_time = datetime.now()
        self.active_alerts = [
            alert for alert in self.active_alerts
            if datetime.fromisoformat(alert["timestamp"]) > current_time - timedelta(minutes=10)
        ]
    
    def _classify_performance(self, entropy_value: float) -> str:
        """Classify performance based on entropy value."""
        if entropy_value < 0.2:
            return "highly_exploitative"
        elif entropy_value < 0.5:
            return "balanced"
        elif entropy_value < 0.8:
            return "exploratory"
        else:
            return "highly_exploratory"
    
    def _calculate_behavioral_similarity(self, profile1: AgentProfile, profile2: AgentProfile) -> float:
        """Calculate behavioral similarity between two agents."""
        # Simplified: compare action distributions
        # In real implementation, would use profile data
        return np.random.uniform(0.3, 0.9)  # Placeholder
    
    def _calculate_performance_difference(self, profile1: AgentProfile, profile2: AgentProfile) -> float:
        """Calculate performance difference between agents."""
        # Placeholder implementation
        return np.random.uniform(-0.5, 0.5)
    
    def _calculate_principle_overlap(self, profile1: AgentProfile, profile2: AgentProfile) -> float:
        """Calculate overlap in discovered principles."""
        # Placeholder implementation
        return np.random.uniform(0.2, 0.8)
    
    def _calculate_relative_efficiency(self, profile1: AgentProfile, profile2: AgentProfile) -> float:
        """Calculate relative efficiency between agents."""
        # Placeholder implementation
        return np.random.uniform(0.8, 1.2)
    
    def _calculate_performance_rank(self, agent_id: str) -> int:
        """Calculate performance rank among all agents."""
        # Placeholder: return random rank
        return np.random.randint(1, len(self.agent_profiles) + 1)
    
    def _identify_unique_behaviors(self, agent_id: str) -> List[str]:
        """Identify behaviors unique to this agent."""
        # Placeholder implementation
        return ["unique_pattern_1", "unique_pattern_2"]
    
    def _calculate_distribution_similarity(self, dist1: Dict[str, float], 
                                         dist2: Dict[str, float]) -> float:
        """Calculate similarity between two probability distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        # Convert to vectors
        vec1 = np.array([dist1.get(k, 0) for k in all_keys])
        vec2 = np.array([dist2.get(k, 0) for k in all_keys])
        
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    # Report generation methods
    
    def _generate_json_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON format report."""
        return {
            "report_type": "realtime_analysis",
            "generated_at": datetime.now().isoformat(),
            "analysis_results": results,
            "summary": {
                "total_actions_analyzed": len(self.action_window),
                "active_alerts": len(self.active_alerts),
                "anomalies_detected": len(results.get("anomalies", [])),
                "emerging_patterns": len(results.get("emerging_patterns", [])),
                "current_performance": results.get("performance_indicators", {}).get("current_performance", "unknown")
            }
        }
    
    def _generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV format report."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["Timestamp", "Metric", "Value", "Category"])
        
        # Metrics
        timestamp = results.get("timestamp", datetime.now().isoformat())
        for metric, value in results.get("metrics", {}).items():
            writer.writerow([timestamp, metric, value, "metric"])
        
        # Anomalies
        for anomaly in results.get("anomalies", []):
            writer.writerow([
                anomaly.get("timestamp", timestamp),
                f"anomaly_{anomaly.get('type', 'unknown')}",
                anomaly.get("severity", "unknown"),
                "anomaly"
            ])
        
        # Alerts
        for alert in results.get("alerts", []):
            writer.writerow([
                alert.get("timestamp", timestamp),
                f"alert_{alert.get('type', 'unknown')}",
                alert.get("severity", "unknown"),
                "alert"
            ])
        
        return output.getvalue()
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML format report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-time Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ padding: 10px; margin: 5px; background: #f0f0f0; }}
                .anomaly {{ padding: 10px; margin: 5px; background: #ffcccc; }}
                .alert {{ padding: 10px; margin: 5px; background: #ffeecc; }}
                .section {{ margin: 20px 0; }}
                h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Real-time Analysis Report</h1>
            <p>Generated at: {timestamp}</p>
            
            <div class="section">
                <h2>Current Metrics</h2>
                {metrics_html}
            </div>
            
            <div class="section">
                <h2>Anomalies Detected</h2>
                {anomalies_html}
            </div>
            
            <div class="section">
                <h2>Active Alerts</h2>
                {alerts_html}
            </div>
            
            <div class="section">
                <h2>Performance Indicators</h2>
                {performance_html}
            </div>
        </body>
        </html>
        """
        
        # Build metrics HTML
        metrics_html = ""
        for metric, value in results.get("metrics", {}).items():
            metrics_html += f'<div class="metric">{metric}: {value:.3f}</div>'
        
        # Build anomalies HTML
        anomalies_html = ""
        for anomaly in results.get("anomalies", []):
            anomalies_html += f'''<div class="anomaly">
                Type: {anomaly.get("type", "unknown")}<br>
                Severity: {anomaly.get("severity", "unknown")}<br>
                Details: {anomaly.get("message", "No details")}
            </div>'''
        
        # Build alerts HTML
        alerts_html = ""
        for alert in results.get("alerts", []):
            alerts_html += f'''<div class="alert">
                Type: {alert.get("type", "unknown")}<br>
                Severity: {alert.get("severity", "unknown")}<br>
                Message: {alert.get("message", "No message")}
            </div>'''
        
        # Build performance HTML
        performance = results.get("performance_indicators", {})
        performance_html = f'''
        <div class="metric">Current Performance: {performance.get("current_performance", "unknown")}</div>
        <div class="metric">Trend: {performance.get("trend", "stable")}</div>
        <div class="metric">Efficiency Score: {performance.get("efficiency_score", 0):.2f}</div>
        <div class="metric">Adaptability Score: {performance.get("adaptability_score", 0):.2f}</div>
        '''
        
        return html_template.format(
            timestamp=results.get("timestamp", datetime.now().isoformat()),
            metrics_html=metrics_html,
            anomalies_html=anomalies_html if anomalies_html else "<p>No anomalies detected</p>",
            alerts_html=alerts_html if alerts_html else "<p>No active alerts</p>",
            performance_html=performance_html
        )
    
    def _generate_websocket_message(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate WebSocket format message."""
        return {
            "type": "realtime_analysis_update",
            "timestamp": results.get("timestamp", datetime.now().isoformat()),
            "data": {
                "metrics": results.get("metrics", {}),
                "anomalies": len(results.get("anomalies", [])),
                "alerts": len(results.get("alerts", [])),
                "performance": results.get("performance_indicators", {}).get("current_performance", "unknown"),
                "predictions": results.get("predictions", {})
            },
            "critical_alerts": [
                alert for alert in results.get("alerts", [])
                if alert.get("severity") == "high"
            ]
        }
