"""
Decorators for easy plugin registration.

This module provides decorators that simplify the process of creating
and registering plugins.
"""

from functools import wraps
from typing import Optional, Type, Dict, Any, Callable
import structlog

from .base import (
    BasePlugin, InferencePlugin, ScenarioPlugin, AnalysisPlugin,
    PluginType, PluginMetadata, PLUGIN_REGISTRY
)

logger = structlog.get_logger()


def register_plugin(
    name: str,
    version: str,
    author: str,
    description: str,
    plugin_type: Optional[PluginType] = None,
    dependencies: Optional[list] = None,
    config_schema: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    auto_register: bool = True
) -> Callable:
    """
    Decorator for registering a plugin class.
    
    Args:
        name: Name of the plugin
        version: Version of the plugin
        author: Author of the plugin
        description: Description of what the plugin does
        plugin_type: Type of plugin (inferred if not provided)
        dependencies: List of dependencies (Python packages or plugin:name)
        config_schema: Configuration schema for the plugin
        tags: Tags for categorizing the plugin
        auto_register: Whether to automatically register the plugin
        
    Returns:
        Decorator function
        
    Example:
        @register_plugin(
            name="dtw_inference",
            version="1.0.0",
            author="AI Gym Team",
            description="DTW-based pattern extraction and principle inference"
        )
        class DTWInferencePlugin(InferencePlugin):
            ...
    """
    def decorator(cls: Type[BasePlugin]) -> Type[BasePlugin]:
        # Determine plugin type from class hierarchy if not provided
        if plugin_type is None:
            if issubclass(cls, InferencePlugin):
                inferred_type = PluginType.INFERENCE
            elif issubclass(cls, ScenarioPlugin):
                inferred_type = PluginType.SCENARIO
            elif issubclass(cls, AnalysisPlugin):
                inferred_type = PluginType.ANALYSIS
            else:
                raise ValueError(f"Could not infer plugin type for {cls.__name__}")
        else:
            inferred_type = plugin_type
        
        # Create metadata
        metadata = PluginMetadata(
            name=name,
            version=version,
            author=author,
            description=description,
            plugin_type=inferred_type,
            dependencies=dependencies or [],
            config_schema=config_schema,
            tags=tags or []
        )
        
        # Store metadata on the class
        cls._plugin_metadata = metadata
        
        # Override the metadata property
        @property
        def metadata_property(self) -> PluginMetadata:
            return self._plugin_metadata
        
        cls.metadata = metadata_property
        
        # Auto-register if requested
        if auto_register:
            try:
                PLUGIN_REGISTRY.register(cls, inferred_type)
            except Exception as e:
                logger.error(f"Failed to auto-register plugin {name}: {e}")
        
        return cls
    
    return decorator


def plugin_config(**config_defaults) -> Callable:
    """
    Decorator for setting default configuration values for a plugin.
    
    Args:
        **config_defaults: Default configuration key-value pairs
        
    Returns:
        Decorator function
        
    Example:
        @plugin_config(
            confidence_threshold=0.8,
            min_pattern_length=3,
            incremental_learning=True
        )
        class MyInferencePlugin(InferencePlugin):
            ...
    """
    def decorator(cls: Type[BasePlugin]) -> Type[BasePlugin]:
        # Store original __init__
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, config: Optional[Dict[str, Any]] = None):
            # Merge default config with provided config
            merged_config = config_defaults.copy()
            if config:
                merged_config.update(config)
            
            # Call original init with merged config
            original_init(self, merged_config)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def requires_dependencies(*dependencies: str) -> Callable:
    """
    Decorator for validating plugin dependencies before initialization.
    
    Args:
        *dependencies: Required dependencies (package names or plugin:name)
        
    Returns:
        Decorator function
        
    Example:
        @requires_dependencies("numpy", "scikit-learn", "plugin:base_patterns")
        class AdvancedInferencePlugin(InferencePlugin):
            ...
    """
    def decorator(cls: Type[BasePlugin]) -> Type[BasePlugin]:
        # Store original initialize method
        original_initialize = cls.initialize
        
        @wraps(original_initialize)
        def new_initialize(self):
            # Check dependencies
            for dep in dependencies:
                if dep.startswith("plugin:"):
                    # Check for plugin dependency
                    plugin_name = dep.split(":", 1)[1]
                    if not any(plugin_name in plugins 
                              for plugins in PLUGIN_REGISTRY.list_plugins().values()):
                        raise RuntimeError(f"Required plugin not found: {plugin_name}")
                else:
                    # Check for Python package dependency
                    try:
                        __import__(dep)
                    except ImportError:
                        raise RuntimeError(f"Required package not found: {dep}")
            
            # Call original initialize
            original_initialize(self)
        
        cls.initialize = new_initialize
        return cls
    
    return decorator


def cached_method(ttl: int = 300) -> Callable:
    """
    Decorator for caching plugin method results.
    
    Args:
        ttl: Time to live in seconds (default: 5 minutes)
        
    Returns:
        Decorator function
        
    Example:
        class MyPlugin(InferencePlugin):
            @cached_method(ttl=600)
            def expensive_computation(self, data):
                ...
    """
    def decorator(method: Callable) -> Callable:
        cache_attr = f"_cache_{method.__name__}"
        
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            import time
            import hashlib
            import json
            
            # Create cache key from arguments
            key_data = {
                "args": args,
                "kwargs": kwargs
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check if cache exists
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            
            cache = getattr(self, cache_attr)
            
            # Check if cached value exists and is valid
            if cache_key in cache:
                cached_value, cached_time = cache[cache_key]
                if time.time() - cached_time < ttl:
                    logger.debug(f"Cache hit for {method.__name__}")
                    return cached_value
            
            # Compute and cache result
            result = method(self, *args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cache miss for {method.__name__}, computed and cached")
            
            # Clean old cache entries
            current_time = time.time()
            cache_copy = cache.copy()
            for key, (_, cached_time) in cache_copy.items():
                if current_time - cached_time >= ttl:
                    del cache[key]
            
            return result
        
        return wrapper
    
    return decorator


def validate_input(schema: Dict[str, Any]) -> Callable:
    """
    Decorator for validating method inputs against a schema.
    
    Args:
        schema: Schema definition for validation
        
    Returns:
        Decorator function
        
    Example:
        class MyPlugin(ScenarioPlugin):
            @validate_input({
                "count": {"type": "int", "min": 1, "max": 100},
                "difficulty": {"type": "float", "min": 0.0, "max": 1.0}
            })
            def generate_batch(self, count, context):
                ...
    """
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Simple validation implementation
            # In production, you might use jsonschema or similar
            
            # Get method signature
            import inspect
            sig = inspect.signature(method)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument against schema
            for param_name, param_schema in schema.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    # Type validation
                    if "type" in param_schema:
                        expected_type = {
                            "int": int,
                            "float": float,
                            "str": str,
                            "bool": bool,
                            "list": list,
                            "dict": dict
                        }.get(param_schema["type"])
                        
                        if expected_type and not isinstance(value, expected_type):
                            raise ValueError(
                                f"Parameter {param_name} must be of type {param_schema['type']}"
                            )
                    
                    # Range validation for numbers
                    if isinstance(value, (int, float)):
                        if "min" in param_schema and value < param_schema["min"]:
                            raise ValueError(
                                f"Parameter {param_name} must be >= {param_schema['min']}"
                            )
                        if "max" in param_schema and value > param_schema["max"]:
                            raise ValueError(
                                f"Parameter {param_name} must be <= {param_schema['max']}"
                            )
                    
                    # Length validation for strings and lists
                    if isinstance(value, (str, list)):
                        if "min_length" in param_schema and len(value) < param_schema["min_length"]:
                            raise ValueError(
                                f"Parameter {param_name} must have length >= {param_schema['min_length']}"
                            )
                        if "max_length" in param_schema and len(value) > param_schema["max_length"]:
                            raise ValueError(
                                f"Parameter {param_name} must have length <= {param_schema['max_length']}"
                            )
            
            return method(self, *args, **kwargs)
        
        return wrapper
    
    return decorator
