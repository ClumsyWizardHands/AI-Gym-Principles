"""
Comprehensive tests for the plugin system.

Tests cover plugin loading, unloading, hot reload, configuration,
dependencies, and all decorators.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, call
import pytest
import time

from src.plugins.base import (
    BasePlugin, InferencePlugin, ScenarioPlugin, AnalysisPlugin,
    PluginType, PluginMetadata, PluginRegistry, PLUGIN_REGISTRY
)
from src.plugins.loader import PluginLoader, PLUGIN_LOADER
from src.plugins.decorators import (
    register_plugin, plugin_config, requires_dependencies,
    cached_method, validate_input
)
from src.core.models import Action, Pattern, Principle, DecisionContext


class TestPluginBase:
    """Test base plugin functionality."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata structure."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="Test Description",
            plugin_type=PluginType.INFERENCE,
            dependencies=["numpy"],
            tags=["test", "demo"]
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.INFERENCE
        assert "numpy" in metadata.dependencies
        assert "test" in metadata.tags
        
        # Test to_dict conversion
        metadata_dict = metadata.to_dict()
        assert metadata_dict["name"] == "test_plugin"
        assert metadata_dict["plugin_type"] == "inference"
        assert isinstance(metadata_dict["created_at"], str)
    
    def test_base_plugin_initialization(self):
        """Test base plugin initialization and configuration."""
        class TestPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
        
        # Test with default config
        plugin = TestPlugin()
        assert plugin.config == {}
        assert not plugin._initialized
        
        # Test with custom config
        config = {"param1": "value1", "param2": 42}
        plugin = TestPlugin(config)
        assert plugin.config == config
        
        # Test initialization
        plugin.initialize()
        assert plugin._initialized
        
        # Test cleanup
        plugin.cleanup()
        assert not plugin._initialized
    
    def test_config_validation(self):
        """Test plugin configuration validation."""
        class TestPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE,
                    config_schema={
                        "required_param": {"required": True},
                        "optional_param": {"required": False}
                    }
                )
        
        # Test valid config
        plugin = TestPlugin({"required_param": "value"})
        assert plugin.validate_config()
        
        # Test missing required param
        plugin = TestPlugin({"optional_param": "value"})
        assert not plugin.validate_config()
        
        # Test no schema
        class NoSchemaPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
        
        plugin = NoSchemaPlugin()
        assert plugin.validate_config()


class TestPluginRegistry:
    """Test plugin registry functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        # Clear all registered plugins
        for plugin_type in PluginType:
            PLUGIN_REGISTRY._plugins[plugin_type].clear()
        PLUGIN_REGISTRY._instances.clear()
    
    def test_plugin_registration(self):
        """Test registering plugins in the registry."""
        class TestInferencePlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_inference",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Register plugin
        PLUGIN_REGISTRY.register(TestInferencePlugin)
        
        # Check registration
        assert "test_inference" in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
        plugin_class = PLUGIN_REGISTRY.get_plugin_class("test_inference", PluginType.INFERENCE)
        assert plugin_class == TestInferencePlugin
    
    def test_plugin_type_validation(self):
        """Test that plugin types are validated during registration."""
        class WrongTypePlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="wrong_type",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.SCENARIO  # Wrong type!
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="is not a ScenarioPlugin"):
            PLUGIN_REGISTRY.register(WrongTypePlugin, PluginType.SCENARIO)
    
    def test_plugin_unregistration(self):
        """Test unregistering plugins."""
        class TestPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_unreg",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Register and create instance
        PLUGIN_REGISTRY.register(TestPlugin)
        instance = PLUGIN_REGISTRY.create_instance("test_unreg", PluginType.INFERENCE)
        assert instance is not None
        
        # Unregister
        PLUGIN_REGISTRY.unregister("test_unreg", PluginType.INFERENCE)
        
        # Check removal
        assert "test_unreg" not in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
        assert f"{PluginType.INFERENCE.value}:test_unreg" not in PLUGIN_REGISTRY._instances
    
    def test_instance_creation_and_caching(self):
        """Test plugin instance creation and caching."""
        class TestPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_instance",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        PLUGIN_REGISTRY.register(TestPlugin)
        
        # Create instance
        config = {"param": "value"}
        instance1 = PLUGIN_REGISTRY.create_instance("test_instance", PluginType.INFERENCE, config)
        assert instance1 is not None
        assert instance1.config == config
        assert instance1._initialized
        
        # Get same instance (should be cached)
        instance2 = PLUGIN_REGISTRY.create_instance("test_instance", PluginType.INFERENCE)
        assert instance1 is instance2
    
    def test_list_plugins(self):
        """Test listing registered plugins."""
        # Register multiple plugins
        for i in range(3):
            class TestPlugin(InferencePlugin):
                name = f"test_{i}"
                
                @property
                def metadata(self):
                    return PluginMetadata(
                        name=self.name,
                        version="1.0.0",
                        author="Test",
                        description="Test",
                        plugin_type=PluginType.INFERENCE
                    )
                
                def extract_patterns(self, actions):
                    return []
                
                def infer_principles(self, patterns):
                    return []
            
            TestPlugin.name = f"test_{i}"
            PLUGIN_REGISTRY.register(TestPlugin)
        
        # List all plugins
        all_plugins = PLUGIN_REGISTRY.list_plugins()
        assert len(all_plugins["inference"]) == 3
        
        # List specific type
        inference_plugins = PLUGIN_REGISTRY.list_plugins(PluginType.INFERENCE)
        assert len(inference_plugins["inference"]) == 3


class TestPluginLoader:
    """Test plugin loader functionality."""
    
    def setup_method(self):
        """Clear loader state before each test."""
        PLUGIN_LOADER.loaded_modules.clear()
        PLUGIN_LOADER.external_packages.clear()
        
        # Clear registry
        for plugin_type in PluginType:
            PLUGIN_REGISTRY._plugins[plugin_type].clear()
        PLUGIN_REGISTRY._instances.clear()
    
    def test_plugin_discovery_from_directory(self):
        """Test discovering plugins from directory."""
        # Create temporary plugin file
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            plugin_file = plugin_dir / "test_plugin.py"
            
            plugin_code = '''
from src.plugins.base import InferencePlugin, PluginMetadata, PluginType
from src.core.models import Action, Pattern, Principle

class TestDiscoveryPlugin(InferencePlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test_discovery",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type=PluginType.INFERENCE
        )
    
    def extract_patterns(self, actions):
        return []
    
    def infer_principles(self, patterns):
        return []
'''
            plugin_file.write_text(plugin_code)
            
            # Create loader with test directory
            loader = PluginLoader(plugin_dir=plugin_dir, auto_discover=True)
            
            # Check discovery
            assert "test_discovery" in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
    
    def test_external_package_loading(self):
        """Test loading plugins from external packages."""
        # Mock an external module
        mock_module = MagicMock()
        
        class MockExternalPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="external_plugin",
                    version="1.0.0",
                    author="External",
                    description="External plugin",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        mock_module.MockExternalPlugin = MockExternalPlugin
        
        with patch('importlib.import_module', return_value=mock_module):
            with patch('inspect.getmembers', return_value=[("MockExternalPlugin", MockExternalPlugin)]):
                PLUGIN_LOADER.add_external_package("external.plugin")
                discovered = PLUGIN_LOADER.discover_plugins()
                
                assert "external_plugin" in discovered["inference"]
    
    def test_plugin_reload(self):
        """Test hot reloading of plugins."""
        # Create initial plugin
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            plugin_file = plugin_dir / "reload_plugin.py"
            
            # Initial version
            plugin_code_v1 = '''
from src.plugins.base import InferencePlugin, PluginMetadata, PluginType
from src.core.models import Action, Pattern, Principle

class ReloadPlugin(InferencePlugin):
    VERSION = "1.0.0"
    
    @property
    def metadata(self):
        return PluginMetadata(
            name="reload_test",
            version=self.VERSION,
            author="Test",
            description="Version 1",
            plugin_type=PluginType.INFERENCE
        )
    
    def extract_patterns(self, actions):
        return []
    
    def infer_principles(self, patterns):
        return []
'''
            plugin_file.write_text(plugin_code_v1)
            
            # Load initial version
            loader = PluginLoader(plugin_dir=plugin_dir, auto_discover=False)
            loader.load_plugin_from_file(plugin_file)
            
            # Get initial metadata
            metadata_v1 = PLUGIN_REGISTRY.get_plugin_metadata("reload_test", PluginType.INFERENCE)
            assert metadata_v1.version == "1.0.0"
            assert metadata_v1.description == "Version 1"
            
            # Update plugin code
            plugin_code_v2 = plugin_code_v1.replace('VERSION = "1.0.0"', 'VERSION = "2.0.0"')
            plugin_code_v2 = plugin_code_v2.replace('description="Version 1"', 'description="Version 2"')
            plugin_file.write_text(plugin_code_v2)
            
            # Mock reload to use new code
            with patch('importlib.reload', side_effect=lambda m: exec(plugin_code_v2, m.__dict__)):
                success = loader.reload_plugin("reload_test", PluginType.INFERENCE)
                assert success
    
    def test_dependency_validation(self):
        """Test plugin dependency validation."""
        # Test Python package dependency
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            # Create plugin with missing dependency
            class DependentPlugin(InferencePlugin):
                @property
                def metadata(self):
                    return PluginMetadata(
                        name="dependent",
                        version="1.0.0",
                        author="Test",
                        description="Test",
                        plugin_type=PluginType.INFERENCE,
                        dependencies=["nonexistent_package"]
                    )
                
                def extract_patterns(self, actions):
                    return []
                
                def infer_principles(self, patterns):
                    return []
            
            PLUGIN_REGISTRY.register(DependentPlugin)
            
            # Validation should fail
            assert not PLUGIN_LOADER.validate_dependencies("dependent", PluginType.INFERENCE)
    
    def test_load_plugins_from_config(self):
        """Test loading plugins based on configuration."""
        # Create test plugin
        class ConfigPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="config_plugin",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        PLUGIN_REGISTRY.register(ConfigPlugin)
        
        # Load from config
        config = {
            "enabled_plugins": {
                "inference": ["config_plugin"]
            },
            "plugin_configs": {
                "config_plugin": {
                    "param1": "value1"
                }
            }
        }
        
        PLUGIN_LOADER.load_plugins_from_config(config)
        
        # Check instance was created
        instance_key = f"{PluginType.INFERENCE.value}:config_plugin"
        assert instance_key in PLUGIN_REGISTRY._instances
        instance = PLUGIN_REGISTRY._instances[instance_key]
        assert instance.config["param1"] == "value1"


class TestPluginDecorators:
    """Test plugin decorators."""
    
    def setup_method(self):
        """Clear registry before each test."""
        for plugin_type in PluginType:
            PLUGIN_REGISTRY._plugins[plugin_type].clear()
        PLUGIN_REGISTRY._instances.clear()
    
    def test_register_plugin_decorator(self):
        """Test the @register_plugin decorator."""
        @register_plugin(
            name="decorator_test",
            version="1.0.0",
            author="Test Author",
            description="Test Description",
            dependencies=["numpy"],
            tags=["test", "decorator"]
        )
        class DecoratorTestPlugin(InferencePlugin):
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Check registration
        assert "decorator_test" in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
        
        # Check metadata
        plugin = DecoratorTestPlugin()
        assert plugin.metadata.name == "decorator_test"
        assert plugin.metadata.version == "1.0.0"
        assert plugin.metadata.author == "Test Author"
        assert "numpy" in plugin.metadata.dependencies
        assert "decorator" in plugin.metadata.tags
    
    def test_plugin_config_decorator(self):
        """Test the @plugin_config decorator."""
        @plugin_config(
            threshold=0.8,
            max_iterations=100,
            debug=True
        )
        class ConfigTestPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="config_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Test default config
        plugin = ConfigTestPlugin()
        assert plugin.config["threshold"] == 0.8
        assert plugin.config["max_iterations"] == 100
        assert plugin.config["debug"] is True
        
        # Test config override
        plugin = ConfigTestPlugin({"threshold": 0.9, "new_param": "value"})
        assert plugin.config["threshold"] == 0.9  # Overridden
        assert plugin.config["max_iterations"] == 100  # Default
        assert plugin.config["new_param"] == "value"  # New
    
    def test_requires_dependencies_decorator(self):
        """Test the @requires_dependencies decorator."""
        @requires_dependencies("json", "os")  # Standard library modules
        class DepsTestPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="deps_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Should initialize successfully
        plugin = DepsTestPlugin()
        plugin.initialize()
        assert plugin._initialized
        
        # Test with missing dependency
        @requires_dependencies("nonexistent_module_xyz")
        class FailingDepsPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="failing_deps",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        plugin = FailingDepsPlugin()
        with pytest.raises(RuntimeError, match="Required package not found"):
            plugin.initialize()
    
    def test_cached_method_decorator(self):
        """Test the @cached_method decorator."""
        class CacheTestPlugin(InferencePlugin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.computation_count = 0
            
            @property
            def metadata(self):
                return PluginMetadata(
                    name="cache_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            @cached_method(ttl=1)  # 1 second TTL
            def expensive_computation(self, x, y):
                self.computation_count += 1
                return x + y
            
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        plugin = CacheTestPlugin()
        
        # First call - should compute
        result1 = plugin.expensive_computation(2, 3)
        assert result1 == 5
        assert plugin.computation_count == 1
        
        # Second call with same args - should use cache
        result2 = plugin.expensive_computation(2, 3)
        assert result2 == 5
        assert plugin.computation_count == 1  # Not incremented
        
        # Different args - should compute again
        result3 = plugin.expensive_computation(3, 4)
        assert result3 == 7
        assert plugin.computation_count == 2
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Should compute again after TTL
        result4 = plugin.expensive_computation(2, 3)
        assert result4 == 5
        assert plugin.computation_count == 3
    
    def test_validate_input_decorator(self):
        """Test the @validate_input decorator."""
        class ValidationTestPlugin(ScenarioPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="validation_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.SCENARIO
                )
            
            def generate_scenario(self, context):
                return {}
            
            @validate_input({
                "count": {"type": "int", "min": 1, "max": 100},
                "difficulty": {"type": "float", "min": 0.0, "max": 1.0},
                "tags": {"type": "list", "min_length": 1}
            })
            def generate_batch(self, count, context, difficulty=0.5, tags=None):
                return [{"difficulty": difficulty} for _ in range(count)]
        
        plugin = ValidationTestPlugin()
        
        # Valid inputs
        result = plugin.generate_batch(5, {}, difficulty=0.7, tags=["test"])
        assert len(result) == 5
        
        # Invalid type
        with pytest.raises(ValueError, match="must be of type int"):
            plugin.generate_batch("five", {})
        
        # Out of range
        with pytest.raises(ValueError, match="must be <= 100"):
            plugin.generate_batch(150, {})
        
        # Below minimum
        with pytest.raises(ValueError, match="must be >= 0.0"):
            plugin.generate_batch(5, {}, difficulty=-0.5)
        
        # List too short
        with pytest.raises(ValueError, match="must have length >= 1"):
            plugin.generate_batch(5, {}, tags=[])


class TestInferencePlugin:
    """Test InferencePlugin specific functionality."""
    
    def test_inference_plugin_methods(self):
        """Test InferencePlugin abstract methods and helpers."""
        class TestInference(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_inference",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                # Simple pattern extraction
                if len(actions) >= 2:
                    return [Pattern(
                        pattern_id="test_pattern",
                        action_sequence=[a.action_taken for a in actions[:2]],
                        frequency=0.5,
                        confidence=0.8,
                        context_distribution={DecisionContext.LOYALTY: 1.0},
                        first_seen=actions[0].timestamp,
                        last_seen=actions[-1].timestamp,
                        occurrence_count=len(actions)
                    )]
                return []
            
            def infer_principles(self, patterns):
                # Simple principle inference
                principles = []
                for pattern in patterns:
                    principles.append(Principle(
                        principle_id=f"principle_{pattern.pattern_id}",
                        description="Test principle",
                        strength=pattern.confidence,
                        consistency=0.9,
                        first_observed=pattern.first_seen,
                        last_updated=pattern.last_seen,
                        supporting_patterns=[pattern.pattern_id]
                    ))
                return principles
        
        plugin = TestInference({"confidence_threshold": 0.7, "incremental_learning": True})
        
        # Test helper methods
        assert plugin.confidence_threshold() == 0.7
        assert plugin.supports_incremental_learning() is True
        
        # Test pattern extraction
        actions = [
            Action(
                action_id=f"action_{i}",
                agent_id="test_agent",
                context=DecisionContext.LOYALTY,
                action_taken=f"action_type_{i}",
                options=[f"option_{i}"],
                timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        patterns = plugin.extract_patterns(actions)
        assert len(patterns) == 1
        assert patterns[0].pattern_id == "test_pattern"
        
        # Test principle inference
        principles = plugin.infer_principles(patterns)
        assert len(principles) == 1
        assert principles[0].strength == 0.8


class TestScenarioPlugin:
    """Test ScenarioPlugin specific functionality."""
    
    def test_scenario_plugin_methods(self):
        """Test ScenarioPlugin abstract methods and helpers."""
        class TestScenario(ScenarioPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_scenario",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.SCENARIO
                )
            
            def generate_scenario(self, context):
                difficulty = context.get("difficulty", 0.5)
                return {
                    "id": "test_scenario_1",
                    "description": "Test scenario",
                    "difficulty": difficulty,
                    "domain": self.get_domain()
                }
            
            def generate_batch(self, count, context):
                return [self.generate_scenario(context) for _ in range(count)]
        
        plugin = TestScenario({
            "min_difficulty": 0.2,
            "max_difficulty": 0.8,
            "procedural_generation": True,
            "domain": "ethics"
        })
        
        # Test helper methods
        assert plugin.get_difficulty_range() == (0.2, 0.8)
        assert plugin.supports_procedural_generation() is True
        assert plugin.get_domain() == "ethics"
        
        # Test scenario generation
        scenario = plugin.generate_scenario({"difficulty": 0.6})
        assert scenario["difficulty"] == 0.6
        assert scenario["domain"] == "ethics"
        
        # Test batch generation
        batch = plugin.generate_batch(3, {"difficulty": 0.5})
        assert len(batch) == 3
        assert all(s["difficulty"] == 0.5 for s in batch)


class TestAnalysisPlugin:
    """Test AnalysisPlugin specific functionality."""
    
    def test_analysis_plugin_methods(self):
        """Test AnalysisPlugin abstract methods and helpers."""
        class TestAnalysis(AnalysisPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test_analysis",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.ANALYSIS
                )
            
            def analyze(self, data):
                # Simple analysis
                return {
                    "total_actions": len(data.get("actions", [])),
                    "total_principles": len(data.get("principles", [])),
                    "metrics": {
                        "avg_strength": 0.75,
                        "consistency": 0.85
                    }
                }
            
            def generate_report(self, analysis_results):
                format_type = self.config.get("format", "json")
                
                if format_type == "json":
                    return analysis_results
                elif format_type == "text":
                    return f"Actions: {analysis_results['total_actions']}\n" \
                           f"Principles: {analysis_results['total_principles']}"
                else:
                    return json.dumps(analysis_results).encode()
        
        plugin = TestAnalysis({
            "export_formats": ["json", "text", "binary"],
            "metrics": ["avg_strength", "consistency"],
            "streaming": True,
            "format": "text"
        })
        
        # Test helper methods
        assert plugin.get_supported_formats() == ["json", "text", "binary"]
        assert plugin.get_metrics() == ["avg_strength", "consistency"]
        assert plugin.supports_streaming() is True
        
        # Test analysis
        data = {
            "actions": [{"id": 1}, {"id": 2}],
            "principles": [{"id": 1}]
        }
        
        analysis = plugin.analyze(data)
        assert analysis["total_actions"] == 2
        assert analysis["total_principles"] == 1
        
        # Test report generation
        report = plugin.generate_report(analysis)
        assert "Actions: 2" in report
        assert "Principles: 1" in report


class TestPluginIntegration:
    """Test plugin system integration scenarios."""
    
    def setup_method(self):
        """Clear state before each test."""
        for plugin_type in PluginType:
            PLUGIN_REGISTRY._plugins[plugin_type].clear()
        PLUGIN_REGISTRY._instances.clear()
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self):
        """Test complete plugin lifecycle."""
        # Create a comprehensive plugin
        @register_plugin(
            name="lifecycle_test",
            version="1.0.0",
            author="Test",
            description="Lifecycle test plugin",
            dependencies=["numpy"],
            tags=["lifecycle", "test"]
        )
        @plugin_config(
            threshold=0.8,
            batch_size=10
        )
        @requires_dependencies("json")
        class LifecyclePlugin(InferencePlugin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pattern_count = 0
                self.principle_count = 0
            
            @cached_method(ttl=60)
            def extract_patterns(self, actions):
                self.pattern_count += 1
                if len(actions) >= 2:
                    return [Pattern(
                        pattern_id=f"pattern_{self.pattern_count}",
                        action_sequence=[a.action_taken for a in actions],
                        frequency=0.1 * self.pattern_count,
                        confidence=0.7,
                        context_distribution={DecisionContext.LOYALTY: 1.0},
                        first_seen=actions[0].timestamp,
                        last_seen=actions[-1].timestamp,
                        occurrence_count=len(actions)
                    )]
                return []
            
            def infer_principles(self, patterns):
                self.principle_count += 1
                return [Principle(
                    principle_id=f"principle_{self.principle_count}",
                    description=f"Principle from {len(patterns)} patterns",
                    strength=0.8,
                    consistency=0.9,
                    first_observed=datetime.now(),
                    last_updated=datetime.now(),
                    supporting_patterns=[p.pattern_id for p in patterns]
                ) for p in patterns]
        
        # Verify registration
        assert "lifecycle_test" in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
        
        # Get metadata
        metadata = PLUGIN_REGISTRY.get_plugin_metadata("lifecycle_test", PluginType.INFERENCE)
        assert metadata is not None
        assert metadata.name == "lifecycle_test"
        assert "numpy" in metadata.dependencies
        
        # Create instance
        custom_config = {"threshold": 0.9}
        instance = PLUGIN_REGISTRY.create_instance(
            "lifecycle_test", 
            PluginType.INFERENCE, 
            custom_config
        )
        assert instance is not None
        assert instance.config["threshold"] == 0.9  # Overridden
        assert instance.config["batch_size"] == 10  # Default
        
        # Test pattern extraction with caching
        actions = [
            Action(
                action_id=f"action_{i}",
                agent_id="test",
                context=DecisionContext.LOYALTY,
                action_taken=f"action_{i}",
                options=[f"opt_{i}"],
                timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        patterns1 = instance.extract_patterns(actions)
        assert len(patterns1) == 1
        assert instance.pattern_count == 1
        
        # Second call should use cache
        patterns2 = instance.extract_patterns(actions)
        assert len(patterns2) == 1
        assert instance.pattern_count == 1  # Not incremented due to cache
        
        # Test principle inference
        principles = instance.infer_principles(patterns1)
        assert len(principles) == 1
        assert principles[0].strength == 0.8
        
        # Cleanup
        PLUGIN_REGISTRY.cleanup_all()
        assert len(PLUGIN_REGISTRY._instances) == 0
    
    def test_plugin_error_handling(self):
        """Test error handling in plugin system."""
        # Test invalid plugin type
        class InvalidPlugin:
            pass
        
        with pytest.raises(ValueError):
            PLUGIN_REGISTRY.register(InvalidPlugin)
        
        # Test plugin with runtime error
        class ErrorPlugin(InferencePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="error_plugin",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.INFERENCE
                )
            
            def extract_patterns(self, actions):
                raise RuntimeError("Pattern extraction failed")
            
            def infer_principles(self, patterns):
                raise RuntimeError("Principle inference failed")
        
        PLUGIN_REGISTRY.register(ErrorPlugin)
        instance = PLUGIN_REGISTRY.create_instance("error_plugin", PluginType.INFERENCE)
        
        # Should raise runtime errors
        with pytest.raises(RuntimeError, match="Pattern extraction failed"):
            instance.extract_patterns([])
        
        with pytest.raises(RuntimeError, match="Principle inference failed"):
            instance.infer_principles([])
    
    def test_multiple_plugin_types(self):
        """Test using multiple plugin types together."""
        # Create one of each type
        @register_plugin(
            name="multi_inference",
            version="1.0.0",
            author="Test",
            description="Multi-type test inference"
        )
        class MultiInference(InferencePlugin):
            def extract_patterns(self, actions):
                return [Pattern(
                    pattern_id="multi_pattern",
                    action_sequence=["test"],
                    frequency=0.5,
                    confidence=0.8,
                    context_distribution={DecisionContext.LOYALTY: 1.0},
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    occurrence_count=1
                )]
            
            def infer_principles(self, patterns):
                return [Principle(
                    principle_id="multi_principle",
                    description="Multi-test principle",
                    strength=0.7,
                    consistency=0.8,
                    first_observed=datetime.now(),
                    last_updated=datetime.now(),
                    supporting_patterns=["multi_pattern"]
                )]
        
        @register_plugin(
            name="multi_scenario",
            version="1.0.0",
            author="Test",
            description="Multi-type test scenario"
        )
        class MultiScenario(ScenarioPlugin):
            def generate_scenario(self, context):
                return {
                    "id": "multi_scenario",
                    "description": "Multi-type scenario",
                    "context": context
                }
            
            def generate_batch(self, count, context):
                return [self.generate_scenario(context) for _ in range(count)]
        
        @register_plugin(
            name="multi_analysis",
            version="1.0.0",
            author="Test",
            description="Multi-type test analysis"
        )
        class MultiAnalysis(AnalysisPlugin):
            def analyze(self, data):
                return {
                    "inference_data": data.get("inference", {}),
                    "scenario_data": data.get("scenario", {})
                }
            
            def generate_report(self, analysis_results):
                return json.dumps(analysis_results, indent=2)
        
        # Create instances of each
        inference = PLUGIN_REGISTRY.create_instance("multi_inference", PluginType.INFERENCE)
        scenario = PLUGIN_REGISTRY.create_instance("multi_scenario", PluginType.SCENARIO)
        analysis = PLUGIN_REGISTRY.create_instance("multi_analysis", PluginType.ANALYSIS)
        
        # Use them together
        actions = []  # Empty for this test
        patterns = inference.extract_patterns(actions)
        principles = inference.infer_principles(patterns)
        
        scenario_data = scenario.generate_scenario({"difficulty": 0.5})
        
        analysis_data = {
            "inference": {
                "patterns": len(patterns),
                "principles": len(principles)
            },
            "scenario": scenario_data
        }
        
        analysis_results = analysis.analyze(analysis_data)
        report = analysis.generate_report(analysis_results)
        
        # Verify results
        assert isinstance(report, str)
        report_data = json.loads(report)
        assert "inference_data" in report_data
        assert "scenario_data" in report_data
        assert report_data["scenario_data"]["id"] == "multi_scenario"


class TestBuiltinPlugins:
    """Test the built-in example plugins."""
    
    def test_clustering_inference_plugin(self):
        """Test the hierarchical clustering inference plugin."""
        # Import the plugin to ensure it registers
        from src.plugins.builtin.clustering_inference import HierarchicalClusteringInferencePlugin
        
        # Check registration
        assert "hierarchical_clustering_inference" in PLUGIN_REGISTRY._plugins[PluginType.INFERENCE]
        
        # Create instance
        instance = PLUGIN_REGISTRY.create_instance(
            "hierarchical_clustering_inference",
            PluginType.INFERENCE,
            {"min_cluster_size": 2, "confidence_threshold": 0.6}
        )
        assert instance is not None
        
        # Test with actions
        actions = []
        for i in range(10):
            actions.append(Action(
                action_id=f"action_{i}",
                agent_id="test",
                context=DecisionContext.LOYALTY if i % 2 == 0 else DecisionContext.SCARCITY,
                action_taken="cooperate" if i % 2 == 0 else "compete",
                options=["cooperate", "compete"],
                timestamp=datetime.now(),
                confidence=0.8
            ))
        
        # Extract patterns
        patterns = instance.extract_patterns(actions)
        assert len(patterns) > 0
        
        # Infer principles
        principles = instance.infer_principles(patterns)
        assert len(principles) > 0
        
        # Check principle properties
        for principle in principles:
            assert principle.strength >= 0.6  # Confidence threshold
            assert principle.consistency > 0
            assert len(principle.supporting_patterns) > 0
    
    def test_ethical_dilemma_scenario_plugin(self):
        """Test the ethical dilemma scenario plugin."""
        from src.plugins.builtin.ethical_dilemma_scenarios import EthicalDilemmaScenarioPlugin
        
        # Check registration  
        assert "ethical_dilemma_scenarios" in PLUGIN_REGISTRY._plugins[PluginType.SCENARIO]
        
        # Create instance
        instance = PLUGIN_REGISTRY.create_instance(
            "ethical_dilemma_scenarios",
            PluginType.SCENARIO,
            {"difficulty_range": (0.3, 0.7)}
        )
        assert instance is not None
        
        # Generate single scenario
        scenario = instance.generate_scenario({"difficulty": 0.5})
        assert "scenario_id" in scenario
        assert "description" in scenario
        assert "options" in scenario
        assert len(scenario["options"]) >= 2
        
        # Generate batch
        batch = instance.generate_batch(3, {"difficulty": 0.6})
        assert len(batch) == 3
        
        # Check scenario properties
        for s in batch:
            assert s["difficulty"] == 0.6
            assert "ethical_dimensions" in s
            assert len(s["ethical_dimensions"]) > 0
    
    def test_comprehensive_report_analysis_plugin(self):
        """Test the comprehensive report analysis plugin."""
        from src.plugins.builtin.comprehensive_report_analysis import ComprehensiveReportAnalysisPlugin
        
        # Check registration
        assert "comprehensive_report_analysis" in PLUGIN_REGISTRY._plugins[PluginType.ANALYSIS]
        
        # Create instance
        instance = PLUGIN_REGISTRY.create_instance(
            "comprehensive_report_analysis",
            PluginType.ANALYSIS,
            {"export_format": "json"}
        )
        assert instance is not None
        
        # Prepare test data
        test_data = {
            "principles": [
                {
                    "principle_id": "p1",
                    "description": "Test principle 1",
                    "strength": 0.8,
                    "consistency": 0.9,
                    "context_weights": {"LOYALTY": 0.7, "SCARCITY": 0.3}
                },
                {
                    "principle_id": "p2", 
                    "description": "Test principle 2",
                    "strength": 0.6,
                    "consistency": 0.7,
                    "context_weights": {"LOYALTY": 0.4, "SCARCITY": 0.6}
                }
            ],
            "actions": [
                {"action_id": f"a{i}", "timestamp": datetime.now().isoformat()}
                for i in range(20)
            ],
            "patterns": [
                {"pattern_id": "pat1", "frequency": 0.5},
                {"pattern_id": "pat2", "frequency": 0.3}
            ]
        }
        
        # Analyze
        analysis_results = instance.analyze(test_data)
        assert "summary" in analysis_results
        assert "metrics" in analysis_results
        assert "insights" in analysis_results
        
        # Check metrics
        metrics = analysis_results["metrics"]
        assert "strength_distribution" in metrics
        assert "consistency_metrics" in metrics
        assert "context_alignment" in metrics
        
        # Generate report
        report = instance.generate_report(analysis_results)
        assert isinstance(report, dict)  # JSON format
        assert "timestamp" in report
        assert "analysis" in report


class TestPluginAPIIntegration:
    """Test plugin integration with API routes."""
    
    def setup_method(self):
        """Clear state before each test."""
        for plugin_type in PluginType:
            PLUGIN_REGISTRY._plugins[plugin_type].clear()
        PLUGIN_REGISTRY._instances.clear()
    
    def test_plugin_configuration_schema(self):
        """Test plugin configuration schema validation."""
        @register_plugin(
            name="schema_test",
            version="1.0.0",
            author="Test",
            description="Schema test plugin",
            config_schema={
                "threshold": {
                    "type": "number",
                    "required": True,
                    "min": 0.0,
                    "max": 1.0
                },
                "mode": {
                    "type": "string",
                    "required": False,
                    "enum": ["fast", "accurate"]
                }
            }
        )
        class SchemaTestPlugin(InferencePlugin):
            def extract_patterns(self, actions):
                return []
            
            def infer_principles(self, patterns):
                return []
        
        # Valid config
        instance = PLUGIN_REGISTRY.create_instance(
            "schema_test",
            PluginType.INFERENCE,
            {"threshold": 0.5}
        )
        assert instance is not None
        
        # Invalid config (missing required)
        instance = PLUGIN_REGISTRY.create_instance(
            "schema_test",
            PluginType.INFERENCE,
            {"mode": "fast"}  # Missing threshold
        )
        assert instance is None  # Should fail validation
