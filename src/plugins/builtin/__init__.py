"""
Built-in plugins for AI Principles Gym.

This package contains the default plugins that ship with the gym.
"""

# Import all built-in plugins
from .clustering_inference import HierarchicalClusteringInferencePlugin
from .ethical_dilemma_scenarios import EthicalDilemmaScenarioPlugin
from .comprehensive_report_analysis import ComprehensiveReportAnalysisPlugin
from .neural_network_inference import NeuralNetworkInferencePlugin
from .game_theory_scenarios import GameTheoryScenarioPlugin
from .realtime_analysis import RealtimeAnalysisPlugin

__all__ = [
    'HierarchicalClusteringInferencePlugin',
    'EthicalDilemmaScenarioPlugin',
    'ComprehensiveReportAnalysisPlugin',
    'NeuralNetworkInferencePlugin',
    'GameTheoryScenarioPlugin',
    'RealtimeAnalysisPlugin',
]
