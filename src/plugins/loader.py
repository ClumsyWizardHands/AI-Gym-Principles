"""
Plugin loader for automatic plugin discovery and loading.

This module handles dynamic loading of plugins from the plugins directory
and external packages.
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import structlog

from .base import (
    BasePlugin, InferencePlugin, ScenarioPlugin, AnalysisPlugin,
    PluginType, PLUGIN_REGISTRY
)

logger = structlog.get_logger()


class PluginLoader:
    """Handles plugin discovery and loading."""
    
    def __init__(self, plugin_dir: Optional[Path] = None, 
                 auto_discover: bool = True):
        """
        Initialize the plugin loader.
        
        Args:
            plugin_dir: Directory to search for plugins (default: src/plugins/builtin)
            auto_discover: Whether to automatically discover plugins on init
        """
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent / "builtin"
        
        self.plugin_dir = plugin_dir
        self.loaded_modules: Dict[str, Any] = {}
        self.external_packages: List[str] = []
        
        if auto_discover and plugin_dir.exists():
            self.discover_plugins()
    
    def discover_plugins(self) -> Dict[str, List[str]]:
        """
        Discover all plugins in the plugin directory.
        
        Returns:
            Dictionary mapping plugin types to lists of discovered plugin names
        """
        discovered = {
            PluginType.INFERENCE.value: [],
            PluginType.SCENARIO.value: [],
            PluginType.ANALYSIS.value: []
        }
        
        # Discover plugins in plugin directory
        if self.plugin_dir.exists():
            for file_path in self.plugin_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                    
                module_name = file_path.stem
                try:
                    module = self._load_module_from_file(file_path, module_name)
                    plugins = self._extract_plugins_from_module(module)
                    
                    for plugin_class in plugins:
                        self._register_plugin_class(plugin_class, discovered)
                        
                except Exception as e:
                    logger.error(f"Failed to load plugin from {file_path}: {e}")
        
        # Discover plugins from external packages
        for package_name in self.external_packages:
            try:
                module = importlib.import_module(package_name)
                plugins = self._extract_plugins_from_module(module)
                
                for plugin_class in plugins:
                    self._register_plugin_class(plugin_class, discovered)
                    
            except Exception as e:
                logger.error(f"Failed to load plugin from package {package_name}: {e}")
        
        logger.info(f"Discovered plugins: {discovered}")
        return discovered
    
    def add_external_package(self, package_name: str) -> None:
        """
        Add an external package to search for plugins.
        
        Args:
            package_name: Name of the package to add
        """
        if package_name not in self.external_packages:
            self.external_packages.append(package_name)
            logger.info(f"Added external plugin package: {package_name}")
    
    def load_plugin_from_file(self, file_path: Path) -> List[str]:
        """
        Load plugins from a specific file.
        
        Args:
            file_path: Path to the plugin file
            
        Returns:
            List of loaded plugin names
        """
        module_name = file_path.stem
        loaded_plugins = []
        
        try:
            module = self._load_module_from_file(file_path, module_name)
            plugins = self._extract_plugins_from_module(module)
            
            for plugin_class in plugins:
                temp_instance = plugin_class()
                PLUGIN_REGISTRY.register(plugin_class)
                loaded_plugins.append(temp_instance.metadata.name)
                
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
        
        return loaded_plugins
    
    def _load_module_from_file(self, file_path: Path, module_name: str) -> Any:
        """Load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.loaded_modules[module_name] = module
        
        return module
    
    def _extract_plugins_from_module(self, module: Any) -> List[Type[BasePlugin]]:
        """Extract all plugin classes from a module."""
        plugins = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BasePlugin) and 
                obj not in [BasePlugin, InferencePlugin, ScenarioPlugin, AnalysisPlugin] and
                not inspect.isabstract(obj)):
                plugins.append(obj)
        
        return plugins
    
    def _register_plugin_class(self, plugin_class: Type[BasePlugin], 
                              discovered: Dict[str, List[str]]) -> None:
        """Register a plugin class and update discovered list."""
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            PLUGIN_REGISTRY.register(plugin_class)
            discovered[metadata.plugin_type.value].append(metadata.name)
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
    
    def reload_plugin(self, plugin_name: str, plugin_type: PluginType) -> bool:
        """
        Reload a specific plugin (useful for development).
        
        Args:
            plugin_name: Name of the plugin to reload
            plugin_type: Type of the plugin
            
        Returns:
            True if successfully reloaded, False otherwise
        """
        # First unregister the plugin
        PLUGIN_REGISTRY.unregister(plugin_name, plugin_type)
        
        # Find and reload the module containing the plugin
        for module_name, module in self.loaded_modules.items():
            plugins = self._extract_plugins_from_module(module)
            
            for plugin_class in plugins:
                temp_instance = plugin_class()
                if temp_instance.metadata.name == plugin_name:
                    # Reload the module
                    try:
                        importlib.reload(module)
                        
                        # Re-extract and register the plugin
                        reloaded_plugins = self._extract_plugins_from_module(module)
                        for reloaded_class in reloaded_plugins:
                            temp = reloaded_class()
                            if temp.metadata.name == plugin_name:
                                PLUGIN_REGISTRY.register(reloaded_class)
                                logger.info(f"Reloaded plugin: {plugin_name}")
                                return True
                                
                    except Exception as e:
                        logger.error(f"Failed to reload plugin {plugin_name}: {e}")
                        return False
        
        logger.error(f"Plugin {plugin_name} not found for reload")
        return False
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about all loaded plugins.
        
        Returns:
            Dictionary containing plugin information
        """
        info = {
            "plugin_directory": str(self.plugin_dir),
            "external_packages": self.external_packages,
            "loaded_modules": list(self.loaded_modules.keys()),
            "registered_plugins": PLUGIN_REGISTRY.list_plugins()
        }
        
        # Add metadata for each plugin
        plugin_metadata = {}
        for plugin_type in PluginType:
            plugin_metadata[plugin_type.value] = {}
            
            for plugin_name in PLUGIN_REGISTRY.list_plugins(plugin_type)[plugin_type.value]:
                metadata = PLUGIN_REGISTRY.get_plugin_metadata(plugin_name, plugin_type)
                if metadata:
                    plugin_metadata[plugin_type.value][plugin_name] = metadata.to_dict()
        
        info["plugin_metadata"] = plugin_metadata
        return info
    
    def validate_dependencies(self, plugin_name: str, 
                            plugin_type: PluginType) -> bool:
        """
        Validate that all plugin dependencies are satisfied.
        
        Args:
            plugin_name: Name of the plugin
            plugin_type: Type of the plugin
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        metadata = PLUGIN_REGISTRY.get_plugin_metadata(plugin_name, plugin_type)
        if not metadata:
            return False
        
        for dependency in metadata.dependencies:
            if dependency.startswith("plugin:"):
                # Check for plugin dependency
                dep_name = dependency.split(":", 1)[1]
                if not any(dep_name in plugins 
                          for plugins in PLUGIN_REGISTRY.list_plugins().values()):
                    logger.error(f"Missing plugin dependency: {dep_name}")
                    return False
            else:
                # Check for Python package dependency
                try:
                    importlib.import_module(dependency)
                except ImportError:
                    logger.error(f"Missing package dependency: {dependency}")
                    return False
        
        return True
    
    def load_plugins_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load plugins based on configuration.
        
        Args:
            config: Configuration dictionary with plugin settings
        """
        # Load external packages
        external_packages = config.get("external_plugin_packages", [])
        for package in external_packages:
            self.add_external_package(package)
        
        # Set custom plugin directory if specified
        custom_dir = config.get("plugin_directory")
        if custom_dir:
            self.plugin_dir = Path(custom_dir)
        
        # Discover plugins
        self.discover_plugins()
        
        # Auto-enable plugins if specified
        enabled_plugins = config.get("enabled_plugins", {})
        for plugin_type_str, plugin_names in enabled_plugins.items():
            try:
                plugin_type = PluginType(plugin_type_str)
                for plugin_name in plugin_names:
                    plugin_config = config.get("plugin_configs", {}).get(plugin_name, {})
                    PLUGIN_REGISTRY.create_instance(plugin_name, plugin_type, plugin_config)
                    
            except ValueError:
                logger.error(f"Invalid plugin type: {plugin_type_str}")


# Global plugin loader instance
PLUGIN_LOADER = PluginLoader(auto_discover=False)
