"""
Base classes for the plugin system.

This module defines abstract base classes for different plugin types and the
core infrastructure for plugin management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Type, Callable, Union
import structlog

from ..core.models import Action, Principle

logger = structlog.get_logger()


class PluginType(Enum):
    """Types of plugins supported by the system."""
    INFERENCE = "inference"
    SCENARIO = "scenario"
    ANALYSIS = "analysis"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with optional configuration."""
        self.config = config or {}
        self._metadata: Optional[PluginMetadata] = None
        self._initialized = False
        
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def initialize(self) -> None:
        """Initialize the plugin. Override for custom initialization logic."""
        self._initialized = True
        logger.info(f"Initialized plugin: {self.metadata.name}")
    
    def cleanup(self) -> None:
        """Cleanup plugin resources. Override for custom cleanup logic."""
        self._initialized = False
        logger.info(f"Cleaned up plugin: {self.metadata.name}")
    
    def validate_config(self) -> bool:
        """Validate plugin configuration against schema."""
        if not self.metadata.config_schema:
            return True
        
        # Simple validation - can be enhanced with jsonschema
        required_keys = [k for k, v in self.metadata.config_schema.items() 
                        if v.get("required", False)]
        
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True


class InferencePlugin(BasePlugin):
    """Base class for inference plugins that extract patterns and infer principles."""
    
    @abstractmethod
    def extract_patterns(self, actions: List[Action]) -> List[Dict[str, Any]]:
        """
        Extract behavioral patterns from a list of actions.
        
        Args:
            actions: List of agent actions to analyze
            
        Returns:
            List of detected patterns as dictionaries
        """
        pass
    
    @abstractmethod
    def infer_principles(self, patterns: List[Dict[str, Any]]) -> List[Principle]:
        """
        Infer principles from detected patterns.
        
        Args:
            patterns: List of patterns to analyze
            
        Returns:
            List of inferred principles
        """
        pass
    
    def confidence_threshold(self) -> float:
        """Return minimum confidence threshold for principle extraction."""
        return self.config.get("confidence_threshold", 0.7)
    
    def supports_incremental_learning(self) -> bool:
        """Whether this plugin supports incremental learning."""
        return self.config.get("incremental_learning", False)


class ScenarioPlugin(BasePlugin):
    """Base class for scenario generation plugins."""
    
    @abstractmethod
    def generate_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single scenario.
        
        Args:
            context: Context information for scenario generation
            
        Returns:
            Generated scenario data
        """
        pass
    
    @abstractmethod
    def generate_batch(self, count: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple scenarios.
        
        Args:
            count: Number of scenarios to generate
            context: Context information for scenario generation
            
        Returns:
            List of generated scenarios
        """
        pass
    
    def get_difficulty_range(self) -> tuple[float, float]:
        """Return min and max difficulty levels this plugin can generate."""
        return (
            self.config.get("min_difficulty", 0.0),
            self.config.get("max_difficulty", 1.0)
        )
    
    def supports_procedural_generation(self) -> bool:
        """Whether this plugin supports procedural generation."""
        return self.config.get("procedural_generation", False)
    
    def get_domain(self) -> Optional[str]:
        """Return the domain this plugin specializes in, if any."""
        return self.config.get("domain", None)


class AnalysisPlugin(BasePlugin):
    """Base class for analysis and reporting plugins."""
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis on provided data.
        
        Args:
            data: Data to analyze (actions, principles, patterns, etc.)
            
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    def generate_report(self, analysis_results: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """
        Generate a report from analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            Report in the appropriate format (text, binary, or structured data)
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""
        return self.config.get("export_formats", ["json"])
    
    def get_metrics(self) -> List[str]:
        """Return list of metrics this plugin can calculate."""
        return self.config.get("metrics", [])
    
    def supports_streaming(self) -> bool:
        """Whether this plugin supports streaming analysis."""
        return self.config.get("streaming", False)


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[PluginType, Dict[str, Type[BasePlugin]]] = {
            PluginType.INFERENCE: {},
            PluginType.SCENARIO: {},
            PluginType.ANALYSIS: {}
        }
        self._instances: Dict[str, BasePlugin] = {}
        
    def register(self, plugin_class: Type[BasePlugin], 
                 plugin_type: Optional[PluginType] = None) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: The plugin class to register
            plugin_type: Type of plugin (inferred from class if not provided)
        """
        # Create temporary instance to get metadata
        temp_instance = plugin_class()
        metadata = temp_instance.metadata
        
        # Infer plugin type if not provided
        if plugin_type is None:
            plugin_type = metadata.plugin_type
            
        # Validate plugin type matches class type
        if plugin_type == PluginType.INFERENCE and not isinstance(temp_instance, InferencePlugin):
            raise ValueError(f"Plugin {metadata.name} is not an InferencePlugin")
        elif plugin_type == PluginType.SCENARIO and not isinstance(temp_instance, ScenarioPlugin):
            raise ValueError(f"Plugin {metadata.name} is not a ScenarioPlugin")
        elif plugin_type == PluginType.ANALYSIS and not isinstance(temp_instance, AnalysisPlugin):
            raise ValueError(f"Plugin {metadata.name} is not an AnalysisPlugin")
        
        # Register the plugin
        self._plugins[plugin_type][metadata.name] = plugin_class
        logger.info(f"Registered plugin: {metadata.name} (type: {plugin_type.value})")
        
    def unregister(self, plugin_name: str, plugin_type: PluginType) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            plugin_type: Type of the plugin
        """
        if plugin_name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][plugin_name]
            
            # Clean up instance if exists
            instance_key = f"{plugin_type.value}:{plugin_name}"
            if instance_key in self._instances:
                self._instances[instance_key].cleanup()
                del self._instances[instance_key]
                
            logger.info(f"Unregistered plugin: {plugin_name} (type: {plugin_type.value})")
    
    def get_plugin_class(self, plugin_name: str, 
                        plugin_type: PluginType) -> Optional[Type[BasePlugin]]:
        """Get a registered plugin class."""
        return self._plugins[plugin_type].get(plugin_name)
    
    def create_instance(self, plugin_name: str, plugin_type: PluginType,
                       config: Optional[Dict[str, Any]] = None) -> Optional[BasePlugin]:
        """
        Create an instance of a plugin.
        
        Args:
            plugin_name: Name of the plugin
            plugin_type: Type of the plugin
            config: Configuration for the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        plugin_class = self.get_plugin_class(plugin_name, plugin_type)
        if not plugin_class:
            logger.error(f"Plugin not found: {plugin_name} (type: {plugin_type.value})")
            return None
        
        instance_key = f"{plugin_type.value}:{plugin_name}"
        
        # Return existing instance if already created
        if instance_key in self._instances:
            return self._instances[instance_key]
        
        # Create new instance
        instance = plugin_class(config)
        
        # Validate configuration
        if not instance.validate_config():
            logger.error(f"Invalid configuration for plugin: {plugin_name}")
            return None
        
        # Initialize and cache
        instance.initialize()
        self._instances[instance_key] = instance
        
        return instance
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> Dict[str, List[str]]:
        """
        List all registered plugins.
        
        Args:
            plugin_type: Filter by plugin type (all types if None)
            
        Returns:
            Dictionary mapping plugin types to lists of plugin names
        """
        if plugin_type:
            return {plugin_type.value: list(self._plugins[plugin_type].keys())}
        
        return {
            ptype.value: list(plugins.keys()) 
            for ptype, plugins in self._plugins.items()
        }
    
    def get_plugin_metadata(self, plugin_name: str, 
                           plugin_type: PluginType) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        plugin_class = self.get_plugin_class(plugin_name, plugin_type)
        if plugin_class:
            temp_instance = plugin_class()
            return temp_instance.metadata
        return None
    
    def cleanup_all(self) -> None:
        """Cleanup all plugin instances."""
        for instance in self._instances.values():
            instance.cleanup()
        self._instances.clear()
        logger.info("Cleaned up all plugin instances")


# Global plugin registry instance
PLUGIN_REGISTRY = PluginRegistry()
