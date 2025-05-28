"""
Comprehensive report analysis plugin.

This plugin generates detailed analysis reports with multiple export formats
and advanced metrics for principle development.
"""

import json
import csv
from io import StringIO
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
import structlog

from ..decorators import register_plugin, plugin_config, cached_method
from ..base import AnalysisPlugin
from ...core.models import Action, Principle

logger = structlog.get_logger()


@register_plugin(
    name="comprehensive_report_analysis",
    version="1.0.0",
    author="AI Gym Team",
    description="Generates comprehensive analysis reports with multiple metrics and export formats",
    tags=["reporting", "metrics", "export", "visualization-ready"]
)
@plugin_config(
    export_formats=["json", "csv", "markdown", "html"],
    metrics=[
        "principle_strength_distribution",
        "behavioral_consistency",
        "context_alignment",
        "temporal_evolution",
        "contradiction_analysis",
        "decision_confidence"
    ],
    include_visualizations=True,
    streaming=False
)
class ComprehensiveReportAnalysisPlugin(AnalysisPlugin):
    """
    Analysis plugin that generates comprehensive reports on agent behavior.
    
    This plugin analyzes actions, principles, and patterns to produce
    detailed reports with multiple metrics and export options.
    """
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on provided data.
        
        Args:
            data: Dictionary containing actions, principles, patterns, etc.
            
        Returns:
            Analysis results with calculated metrics
        """
        results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "plugin_version": self.metadata.version,
                "data_summary": self._generate_data_summary(data)
            },
            "metrics": {},
            "insights": [],
            "visualizations": {} if self.config["include_visualizations"] else None
        }
        
        # Extract data components
        actions = data.get("actions", [])
        principles = data.get("principles", [])
        patterns = data.get("patterns", [])
        agent_id = data.get("agent_id", "unknown")
        
        # Calculate each metric
        for metric in self.config["metrics"]:
            if metric == "principle_strength_distribution":
                results["metrics"][metric] = self._analyze_principle_strength(principles)
            elif metric == "behavioral_consistency":
                results["metrics"][metric] = self._analyze_behavioral_consistency(actions, patterns)
            elif metric == "context_alignment":
                results["metrics"][metric] = self._analyze_context_alignment(actions, principles)
            elif metric == "temporal_evolution":
                results["metrics"][metric] = self._analyze_temporal_evolution(principles, actions)
            elif metric == "contradiction_analysis":
                results["metrics"][metric] = self._analyze_contradictions(principles, actions)
            elif metric == "decision_confidence":
                results["metrics"][metric] = self._analyze_decision_confidence(actions)
        
        # Generate insights based on metrics
        results["insights"] = self._generate_insights(results["metrics"])
        
        # Prepare visualization data if requested
        if self.config["include_visualizations"]:
            results["visualizations"] = self._prepare_visualization_data(data, results["metrics"])
        
        logger.info(f"Completed analysis for agent {agent_id} with {len(results['metrics'])} metrics")
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """
        Generate a report in the requested format.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            Report in the appropriate format
        """
        # Default to JSON if no format specified
        export_format = self.config.get("current_format", "json")
        
        if export_format == "json":
            return self._generate_json_report(analysis_results)
        elif export_format == "csv":
            return self._generate_csv_report(analysis_results)
        elif export_format == "markdown":
            return self._generate_markdown_report(analysis_results)
        elif export_format == "html":
            return self._generate_html_report(analysis_results)
        else:
            logger.warning(f"Unknown format {export_format}, defaulting to JSON")
            return self._generate_json_report(analysis_results)
    
    def _generate_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the input data."""
        return {
            "num_actions": len(data.get("actions", [])),
            "num_principles": len(data.get("principles", [])),
            "num_patterns": len(data.get("patterns", [])),
            "time_span": self._calculate_time_span(data.get("actions", [])),
            "contexts_encountered": self._count_unique_contexts(data.get("actions", []))
        }
    
    @cached_method(ttl=300)
    def _analyze_principle_strength(self, principles: List[Principle]) -> Dict[str, Any]:
        """Analyze the distribution of principle strengths."""
        if not principles:
            return {"status": "no_principles"}
        
        strengths = [p.strength for p in principles]
        
        return {
            "mean": np.mean(strengths),
            "std": np.std(strengths),
            "min": np.min(strengths),
            "max": np.max(strengths),
            "quartiles": {
                "q1": np.percentile(strengths, 25),
                "median": np.percentile(strengths, 50),
                "q3": np.percentile(strengths, 75)
            },
            "distribution": self._create_histogram(strengths, bins=10),
            "strong_principles": [p.principle_id for p in principles if p.strength > 0.8],
            "weak_principles": [p.principle_id for p in principles if p.strength < 0.3]
        }
    
    def _analyze_behavioral_consistency(self, actions: List[Action], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how consistent the agent's behavior is."""
        if not actions:
            return {"status": "no_actions"}
        
        # Calculate action diversity
        action_types = [a.action_type for a in actions]
        unique_actions = len(set(action_types))
        action_entropy = self._calculate_entropy(action_types)
        
        # Pattern consistency
        pattern_coverage = len(patterns) / max(len(actions), 1)
        avg_pattern_confidence = np.mean([p.get("confidence", 0) for p in patterns]) if patterns else 0
        
        return {
            "action_diversity": unique_actions / len(action_types),
            "action_entropy": action_entropy,
            "pattern_coverage": pattern_coverage,
            "average_pattern_confidence": avg_pattern_confidence,
            "consistency_score": (1 - action_entropy) * avg_pattern_confidence,
            "most_common_actions": self._get_top_items(action_types, 5),
            "pattern_stability": self._calculate_pattern_stability(patterns)
        }
    
    def _analyze_context_alignment(self, actions: List[Action], principles: List[Principle]) -> Dict[str, Any]:
        """Analyze how well actions align with principles in different contexts."""
        if not actions or not principles:
            return {"status": "insufficient_data"}
        
        # Group actions by context
        context_actions = {}
        for action in actions:
            context = action.decision_context.name
            if context not in context_actions:
                context_actions[context] = []
            context_actions[context].append(action)
        
        # Calculate alignment scores
        alignment_scores = {}
        for context, ctx_actions in context_actions.items():
            # Find relevant principles for this context
            relevant_principles = [
                p for p in principles 
                if context in p.context_weights and p.context_weights[context] > 0.5
            ]
            
            if relevant_principles:
                # Simple alignment: check if actions match principle descriptions
                alignment_scores[context] = {
                    "num_actions": len(ctx_actions),
                    "num_relevant_principles": len(relevant_principles),
                    "average_principle_strength": np.mean([p.strength for p in relevant_principles]),
                    "alignment_ratio": self._calculate_alignment_ratio(ctx_actions, relevant_principles)
                }
        
        return {
            "context_alignment_scores": alignment_scores,
            "overall_alignment": np.mean([s["alignment_ratio"] for s in alignment_scores.values()]) if alignment_scores else 0,
            "best_aligned_context": max(alignment_scores.items(), key=lambda x: x[1]["alignment_ratio"])[0] if alignment_scores else None,
            "worst_aligned_context": min(alignment_scores.items(), key=lambda x: x[1]["alignment_ratio"])[0] if alignment_scores else None
        }
    
    def _analyze_temporal_evolution(self, principles: List[Principle], actions: List[Action]) -> Dict[str, Any]:
        """Analyze how principles and behavior evolve over time."""
        if not principles or not actions:
            return {"status": "insufficient_data"}
        
        # Sort by time
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)
        sorted_principles = sorted(principles, key=lambda p: p.first_observed)
        
        # Divide timeline into segments
        num_segments = min(10, len(sorted_actions) // 10)
        if num_segments < 2:
            return {"status": "insufficient_timeline"}
        
        segment_size = len(sorted_actions) // num_segments
        evolution_data = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(sorted_actions)
            segment_actions = sorted_actions[start_idx:end_idx]
            
            # Find principles active during this segment
            segment_start = segment_actions[0].timestamp
            segment_end = segment_actions[-1].timestamp
            
            active_principles = [
                p for p in sorted_principles
                if p.first_observed <= segment_end and p.last_updated >= segment_start
            ]
            
            evolution_data.append({
                "segment": i + 1,
                "time_range": {
                    "start": segment_start.isoformat(),
                    "end": segment_end.isoformat()
                },
                "num_actions": len(segment_actions),
                "num_active_principles": len(active_principles),
                "avg_principle_strength": np.mean([p.strength for p in active_principles]) if active_principles else 0,
                "action_diversity": len(set(a.action_type for a in segment_actions)) / len(segment_actions),
                "avg_confidence": np.mean([a.confidence or 0.5 for a in segment_actions])
            })
        
        return {
            "timeline_segments": evolution_data,
            "principle_emergence_rate": len(principles) / num_segments,
            "behavioral_stability_trend": self._calculate_stability_trend(evolution_data),
            "maturity_indicators": self._calculate_maturity_indicators(evolution_data)
        }
    
    def _analyze_contradictions(self, principles: List[Principle], actions: List[Action]) -> Dict[str, Any]:
        """Analyze contradictions between principles and actions."""
        contradictions = []
        
        # Check for conflicting principles
        for i, p1 in enumerate(principles):
            for p2 in principles[i+1:]:
                conflict_score = self._calculate_principle_conflict(p1, p2)
                if conflict_score > 0.7:
                    contradictions.append({
                        "type": "principle_conflict",
                        "principle1": p1.principle_id,
                        "principle2": p2.principle_id,
                        "conflict_score": conflict_score,
                        "description": f"Principles '{p1.description}' and '{p2.description}' appear contradictory"
                    })
        
        # Check for action-principle contradictions
        for principle in principles:
            conflicting_actions = self._find_conflicting_actions(principle, actions)
            if conflicting_actions:
                contradictions.append({
                    "type": "action_principle_conflict",
                    "principle": principle.principle_id,
                    "num_conflicting_actions": len(conflicting_actions),
                    "conflict_rate": len(conflicting_actions) / len(actions),
                    "description": f"Found {len(conflicting_actions)} actions that contradict principle '{principle.description}'"
                })
        
        return {
            "num_contradictions": len(contradictions),
            "contradiction_details": contradictions,
            "consistency_score": 1 - (len(contradictions) / max(len(principles) + len(actions), 1)),
            "most_problematic_principle": self._find_most_problematic_principle(contradictions) if contradictions else None
        }
    
    def _analyze_decision_confidence(self, actions: List[Action]) -> Dict[str, Any]:
        """Analyze the confidence levels in decision-making."""
        if not actions:
            return {"status": "no_actions"}
        
        confidences = [a.confidence or 0.5 for a in actions]
        
        # Group by context
        context_confidence = {}
        for action in actions:
            context = action.decision_context.name
            if context not in context_confidence:
                context_confidence[context] = []
            context_confidence[context].append(action.confidence or 0.5)
        
        return {
            "overall_confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "trend": self._calculate_confidence_trend(actions)
            },
            "confidence_by_context": {
                ctx: {
                    "mean": np.mean(conf_list),
                    "std": np.std(conf_list),
                    "num_decisions": len(conf_list)
                }
                for ctx, conf_list in context_confidence.items()
            },
            "low_confidence_decisions": sum(1 for c in confidences if c < 0.3),
            "high_confidence_decisions": sum(1 for c in confidences if c > 0.8),
            "confidence_distribution": self._create_histogram(confidences, bins=10)
        }
    
    def _generate_insights(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate insights based on calculated metrics."""
        insights = []
        
        # Principle strength insights
        if "principle_strength_distribution" in metrics:
            strength_data = metrics["principle_strength_distribution"]
            if strength_data.get("status") != "no_principles":
                if strength_data["mean"] > 0.7:
                    insights.append({
                        "type": "principle_strength",
                        "severity": "positive",
                        "message": "Agent has developed strong, well-established principles"
                    })
                elif strength_data["mean"] < 0.3:
                    insights.append({
                        "type": "principle_strength",
                        "severity": "warning",
                        "message": "Agent's principles are weak and may need more training"
                    })
        
        # Consistency insights
        if "behavioral_consistency" in metrics:
            consistency_data = metrics["behavioral_consistency"]
            if consistency_data.get("status") != "no_actions":
                if consistency_data["consistency_score"] > 0.8:
                    insights.append({
                        "type": "consistency",
                        "severity": "positive",
                        "message": "Agent demonstrates highly consistent behavior"
                    })
                elif consistency_data["action_entropy"] > 0.9:
                    insights.append({
                        "type": "consistency",
                        "severity": "warning",
                        "message": "Agent's behavior appears random or unpredictable"
                    })
        
        # Contradiction insights
        if "contradiction_analysis" in metrics:
            contradiction_data = metrics["contradiction_analysis"]
            if contradiction_data["num_contradictions"] > 5:
                insights.append({
                    "type": "contradictions",
                    "severity": "critical",
                    "message": f"Found {contradiction_data['num_contradictions']} contradictions that need resolution"
                })
        
        # Evolution insights
        if "temporal_evolution" in metrics:
            evolution_data = metrics["temporal_evolution"]
            if evolution_data.get("status") != "insufficient_data":
                if evolution_data.get("behavioral_stability_trend", 0) > 0:
                    insights.append({
                        "type": "evolution",
                        "severity": "positive",
                        "message": "Agent's behavior is becoming more stable over time"
                    })
        
        return insights
    
    def _prepare_visualization_data(self, data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualization components."""
        viz_data = {}
        
        # Principle emergence data
        if "principles" in data:
            viz_data["principle_emergence"] = [
                {
                    "principle_id": p.principle_id,
                    "name": p.description[:50] + "..." if len(p.description) > 50 else p.description,
                    "timeline": self._create_principle_timeline(p, data.get("actions", []))
                }
                for p in data["principles"]
            ]
        
        # Behavioral pattern network
        if "patterns" in data:
            viz_data["pattern_network"] = {
                "nodes": [
                    {
                        "id": p.get("pattern_id", ""),
                        "frequency": p.get("frequency", 0),
                        "confidence": p.get("confidence", 0),
                        "context": list(p.get("context_distribution", {}).keys())[0] if p.get("context_distribution") else "unknown"
                    }
                    for p in data["patterns"]
                ],
                "edges": self._create_pattern_edges(data["patterns"])
            }
        
        # Metrics for dashboard
        viz_data["dashboard_metrics"] = {
            "principle_strength": metrics.get("principle_strength_distribution", {}).get("mean", 0),
            "behavioral_consistency": metrics.get("behavioral_consistency", {}).get("consistency_score", 0),
            "overall_confidence": metrics.get("decision_confidence", {}).get("overall_confidence", {}).get("mean", 0),
            "contradiction_count": metrics.get("contradiction_analysis", {}).get("num_contradictions", 0)
        }
        
        return viz_data
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON format report."""
        return json.dumps(results, indent=2, default=str)
    
    def _generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV format report."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write metadata
        writer.writerow(["AI Principles Gym Analysis Report"])
        writer.writerow(["Generated", results["metadata"]["analysis_timestamp"]])
        writer.writerow([])
        
        # Write metrics
        writer.writerow(["Metrics"])
        for metric_name, metric_data in results["metrics"].items():
            writer.writerow([metric_name])
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if not isinstance(value, (dict, list)):
                        writer.writerow(["", key, value])
            writer.writerow([])
        
        # Write insights
        writer.writerow(["Insights"])
        for insight in results["insights"]:
            writer.writerow([insight["type"], insight["severity"], insight["message"]])
        
        return output.getvalue()
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown format report."""
        lines = []
        
        # Header
        lines.append("# AI Principles Gym Analysis Report")
        lines.append(f"\n**Generated:** {results['metadata']['analysis_timestamp']}")
        lines.append("\n## Data Summary")
        
        summary = results['metadata']['data_summary']
        lines.append(f"- **Actions analyzed:** {summary['num_actions']}")
        lines.append(f"- **Principles discovered:** {summary['num_principles']}")
        lines.append(f"- **Patterns identified:** {summary['num_patterns']}")
        lines.append(f"- **Time span:** {summary['time_span']}")
        lines.append(f"- **Contexts encountered:** {summary['contexts_encountered']}")
        
        # Metrics
        lines.append("\n## Key Metrics")
        
        for metric_name, metric_data in results["metrics"].items():
            lines.append(f"\n### {metric_name.replace('_', ' ').title()}")
            
            if isinstance(metric_data, dict) and metric_data.get("status") == "no_data":
                lines.append("*No data available for this metric*")
            elif isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if not isinstance(value, (dict, list)):
                        lines.append(f"- **{key}:** {value}")
        
        # Insights
        lines.append("\n## Insights")
        
        for insight in results["insights"]:
            emoji = "✅" if insight["severity"] == "positive" else "⚠️" if insight["severity"] == "warning" else "❌"
            lines.append(f"\n{emoji} **{insight['type'].replace('_', ' ').title()}**")
            lines.append(f"   {insight['message']}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML format report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Principles Gym Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .insight {{ padding: 10px; margin: 10px 0; border-left: 4px solid; }}
                .positive {{ border-color: #4CAF50; background: #e8f5e9; }}
                .warning {{ border-color: #ff9800; background: #fff3e0; }}
                .critical {{ border-color: #f44336; background: #ffebee; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>AI Principles Gym Analysis Report</h1>
            <p><strong>Generated:</strong> {results['metadata']['analysis_timestamp']}</p>
            
            <h2>Data Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for key, value in results['metadata']['data_summary'].items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += """
            </table>
            
            <h2>Analysis Results</h2>
        """
        
        # Add metrics
        for metric_name, metric_data in results["metrics"].items():
            html += f'<div class="metric"><h3>{metric_name.replace("_", " ").title()}</h3>'
            
            if isinstance(metric_data, dict):
                html += "<ul>"
                for key, value in metric_data.items():
                    if not isinstance(value, (dict, list)):
                        html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        # Add insights
        html += "<h2>Insights</h2>"
        
        for insight in results["insights"]:
            html += f'<div class="insight {insight["severity"]}">'
            html += f'<strong>{insight["type"].replace("_", " ").title()}:</strong> '
            html += f'{insight["message"]}</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    # Helper methods
    def _calculate_time_span(self, actions: List[Action]) -> str:
        """Calculate the time span covered by actions."""
        if not actions:
            return "N/A"
        
        timestamps = [a.timestamp for a in actions]
        time_diff = max(timestamps) - min(timestamps)
        
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        
        if days > 0:
            return f"{days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    def _count_unique_contexts(self, actions: List[Action]) -> int:
        """Count unique contexts encountered."""
        return len(set(a.decision_context for a in actions))
    
    def _create_histogram(self, values: List[float], bins: int = 10) -> List[Dict[str, Any]]:
        """Create histogram data."""
        hist, edges = np.histogram(values, bins=bins)
        
        return [
            {
                "range": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                "count": int(hist[i]),
                "percentage": float(hist[i] / len(values))
            }
            for i in range(len(hist))
        ]
    
    def _calculate_entropy(self, items: List[Any]) -> float:
        """Calculate Shannon entropy of a list."""
        from collections import Counter
        
        counts = Counter(items)
        total = len(items)
        
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy / np.log2(len(counts)) if len(counts) > 1 else 0
    
    def _get_top_items(self, items: List[Any], n: int = 5) -> List[Dict[str, Any]]:
        """Get top N most common items."""
        from collections import Counter
        
        counts = Counter(items)
        return [
            {"item": item, "count": count, "percentage": count / len(items)}
            for item, count in counts.most_common(n)
        ]
    
    def _calculate_pattern_stability(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate how stable patterns are over time."""
        if not patterns:
            return 0
        
        # Simple stability: ratio of patterns that appear consistently
        stable_patterns = [
            p for p in patterns 
            if p.get("occurrence_count", 0) > 5 and p.get("confidence", 0) > 0.7
        ]
        
        return len(stable_patterns) / len(patterns)
    
    def _calculate_alignment_ratio(self, actions: List[Action], principles: List[Principle]) -> float:
        """Calculate how well actions align with principles."""
        # Simplified alignment calculation
        # In practice, this would involve more sophisticated matching
        return min(len(principles) / max(len(actions), 1), 1.0)
    
    def _calculate_stability_trend(self, evolution_data: List[Dict[str, Any]]) -> float:
        """Calculate trend in behavioral stability."""
        if len(evolution_data) < 2:
            return 0
        
        diversities = [seg["action_diversity"] for seg in evolution_data]
        
        # Calculate trend using linear regression
        x = np.arange(len(diversities))
        y = np.array(diversities)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Negative slope means decreasing diversity = increasing stability
        return -slope
    
    def _calculate_maturity_indicators(self, evolution_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate indicators of agent maturity."""
        if not evolution_data:
            return {}
        
        latest = evolution_data[-1]
        earliest = evolution_data[0]
        
        return {
            "principle_growth": latest["num_active_principles"] / max(earliest["num_active_principles"], 1),
            "confidence_improvement": latest["avg_confidence"] - earliest["avg_confidence"],
            "stability_achieved": latest["action_diversity"] < 0.3,
            "maturity_score": (latest["avg_principle_strength"] + latest["avg_confidence"] + (1 - latest["action_diversity"])) / 3
        }
    
    def _calculate_principle_conflict(self, p1: Principle, p2: Principle) -> float:
        """Calculate conflict score between two principles."""
        # Check for opposite actions in similar contexts
        context_overlap = set(p1.context_weights.keys()).intersection(set(p2.context_weights.keys()))
        
        if not context_overlap:
            return 0
        
        # Simple conflict detection based on description
        # In practice, this would use more sophisticated NLP
        words1 = set(p1.description.lower().split())
        words2 = set(p2.description.lower().split())
        
        # Check for negation patterns
        negation_words = {"not", "never", "avoid", "don't", "shouldn't"}
        
        if (words1.intersection(negation_words) and not words2.intersection(negation_words)) or \
           (words2.intersection(negation_words) and not words1.intersection(negation_words)):
            return 0.8
        
        return 0
    
    def _find_conflicting_actions(self, principle: Principle, actions: List[Action]) -> List[Action]:
        """Find actions that conflict with a principle."""
        # Simplified conflict detection
        # In practice, this would involve more sophisticated analysis
        conflicting = []
        
        for action in actions:
            # Check if action is in a context where principle applies
            if action.decision_context.name in principle.context_weights:
                # Simple check: low confidence actions in high-weight contexts
                if action.confidence and action.confidence < 0.3 and principle.context_weights[action.decision_context.name] > 0.7:
                    conflicting.append(action)
        
        return conflicting
    
    def _find_most_problematic_principle(self, contradictions: List[Dict[str, Any]]) -> Optional[str]:
        """Find the principle involved in most contradictions."""
        principle_counts = {}
        
        for contradiction in contradictions:
            if "principle" in contradiction:
                principle_id = contradiction["principle"]
                principle_counts[principle_id] = principle_counts.get(principle_id, 0) + 1
            elif "principle1" in contradiction:
                principle_counts[contradiction["principle1"]] = principle_counts.get(contradiction["principle1"], 0) + 1
                principle_counts[contradiction["principle2"]] = principle_counts.get(contradiction["principle2"], 0) + 1
        
        if principle_counts:
            return max(principle_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def _calculate_confidence_trend(self, actions: List[Action]) -> float:
        """Calculate trend in decision confidence over time."""
        if len(actions) < 2:
            return 0
        
        # Sort actions by timestamp
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)
        confidences = [a.confidence or 0.5 for a in sorted_actions]
        
        # Calculate trend using linear regression
        x = np.arange(len(confidences))
        y = np.array(confidences)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _create_principle_timeline(self, principle: Principle, actions: List[Action]) -> List[Dict[str, Any]]:
        """Create timeline data for a principle."""
        # Group actions by time periods
        timeline = []
        
        # Find actions related to this principle's contexts
        relevant_actions = [
            a for a in actions
            if a.decision_context.name in principle.context_weights
        ]
        
        if not relevant_actions:
            return timeline
        
        # Sort by timestamp
        relevant_actions.sort(key=lambda a: a.timestamp)
        
        # Create time buckets (e.g., hourly)
        from datetime import timedelta
        
        start_time = relevant_actions[0].timestamp
        end_time = relevant_actions[-1].timestamp
        current_time = start_time
        
        while current_time <= end_time:
            bucket_end = current_time + timedelta(hours=1)
            
            # Count actions in this bucket
            bucket_actions = [
                a for a in relevant_actions
                if current_time <= a.timestamp < bucket_end
            ]
            
            if bucket_actions:
                timeline.append({
                    "timestamp": current_time.isoformat(),
                    "strength": principle.strength,  # Could be interpolated
                    "consistency": np.mean([a.confidence or 0.5 for a in bucket_actions]),
                    "action_count": len(bucket_actions)
                })
            
            current_time = bucket_end
        
        return timeline
    
    def _create_pattern_edges(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create edges between related patterns."""
        edges = []
        
        # Simple edge creation based on context overlap
        for i, p1 in enumerate(patterns):
            for p2 in patterns[i+1:]:
                # Check for context overlap
                contexts1 = set(p1.get("context_distribution", {}).keys())
                contexts2 = set(p2.get("context_distribution", {}).keys())
                
                overlap = contexts1.intersection(contexts2)
                if overlap:
                    edges.append({
                        "source": p1.get("pattern_id", ""),
                        "target": p2.get("pattern_id", ""),
                        "weight": len(overlap) / max(len(contexts1), len(contexts2))
                    })
        
        return edges
