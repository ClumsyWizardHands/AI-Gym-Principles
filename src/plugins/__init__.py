"""
Plugin system for AI Principles Gym.

This module provides a flexible plugin architecture for extending the gym with custom:
- Inference algorithms
- Scenario generators
- Analysis tools
"""

from .base import (
    InferencePlugin,
    ScenarioPlugin,
    AnalysisPlugin,
    PluginMetadata,
    PluginRegistry,
    PluginType,
    PLUGIN_REGISTRY,
)
from .loader import PluginLoader, PLUGIN_LOADER
from .decorators import register_plugin

__all__ = [
    "InferencePlugin",
    "ScenarioPlugin",
    "AnalysisPlugin",
    "PluginMetadata",
    "PluginRegistry",
    "PluginType",
    "PLUGIN_REGISTRY",
    "PluginLoader",
    "PLUGIN_LOADER",
    "register_plugin",
]
