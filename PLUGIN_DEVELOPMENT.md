# Plugin Development Guide for AI Principles Gym

This guide explains how to create custom plugins to extend the AI Principles Gym with new algorithms, scenarios, and analysis tools.

## Overview

The plugin system allows you to extend the gym in three key areas:

1. **Inference Plugins** - Custom pattern detection and principle extraction algorithms
2. **Scenario Plugins** - Domain-specific scenario generators
3. **Analysis Plugins** - Custom report generators and metrics

## Quick Start

### 1. Basic Plugin Structure

```python
from principles_gym.plugins import InferencePlugin, register_plugin, plugin_config

@register_plugin(
    name="my_custom_inference",
    version="1.0.0",
    author="Your Name",
    description="Custom inference algorithm for pattern detection"
)
@plugin_config(
    threshold=0.8,
    min_samples=10
)
class MyCustomInferencePlugin(InferencePlugin):
    def extract_patterns(self, actions):
        # Your pattern extraction logic
        patterns = []
        # ... process actions ...
        return patterns
    
    def infer_principles(self, patterns):
        # Your principle inference logic
        principles = []
        # ... process patterns ...
        return principles
```

### 2. Using Decorators

#### @register_plugin
Automatically registers your plugin with metadata:

```python
@register_plugin(
    name="unique_plugin_name",
    version="1.0.0",
    author="Author Name",
    description="What the plugin does",
    dependencies=["numpy", "scikit-learn"],  # Optional
    tags=["ml", "clustering"],  # Optional
    config_schema={  # Optional
        "threshold": {"type": "float", "required": True, "min": 0, "max": 1}
    }
)
```

#### @plugin_config
Sets default configuration values:

```python
@plugin_config(
    confidence_threshold=0.7,
    batch_size=32,
    enable_caching=True
)
```

#### @requires_dependencies
Validates dependencies before initialization:

```python
@requires_dependencies("numpy", "pandas", "plugin:base_inference")
class AdvancedPlugin(InferencePlugin):
    ...
```

#### @cached_method
Caches method results for performance:

```python
class MyPlugin(InferencePlugin):
    @cached_method(ttl=600)  # Cache for 10 minutes
    def expensive_computation(self, data):
        # This result will be cached
        return compute_something(data)
```

#### @validate_input
Validates method inputs:

```python
@validate_input({
    "count": {"type": "int", "min": 1, "max": 100},
    "threshold": {"type": "float", "min": 0.0, "max": 1.0}
})
def generate_batch(self, count, context):
    ...
```

## Plugin Types

### Inference Plugins

Inference plugins analyze agent actions to extract patterns and infer principles.

```python
from typing import List
from principles_gym.plugins import InferencePlugin, register_plugin
from principles_gym.core.models import Action, Pattern, Principle

@register_plugin(
    name="custom_inference",
    version="1.0.0",
    author="Your Name",
    description="Custom pattern detection algorithm"
)
class CustomInferencePlugin(InferencePlugin):
    def extract_patterns(self, actions: List[Action]) -> List[Pattern]:
        """
        Extract behavioral patterns from actions.
        
        Args:
            actions: List of agent actions
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Your pattern detection logic here
        # Example: Find repeated action sequences
        for i in range(len(actions) - 2):
            if actions[i].action_taken == actions[i+2].action_taken:
                pattern = Pattern(
                    pattern_id=f"repeat_{i}",
                    action_sequence=[actions[i].action_taken],
                    frequency=1.0,
                    confidence=0.8,
                    context_distribution={actions[i].context: 1.0},
                    first_seen=actions[i].timestamp,
                    last_seen=actions[i+2].timestamp,
                    occurrence_count=2
                )
                patterns.append(pattern)
        
        return patterns
    
    def infer_principles(self, patterns: List[Pattern]) -> List[Principle]:
        """
        Infer principles from patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            List of inferred principles
        """
        principles = []
        
        # Your principle inference logic here
        for pattern in patterns:
            if pattern.confidence > self.confidence_threshold():
                principle = Principle(
                    principle_id=f"principle_{pattern.pattern_id}",
                    description=f"Prefer {pattern.action_sequence[0]}",
                    strength=pattern.confidence,
                    consistency=pattern.frequency,
                    first_observed=pattern.first_seen,
                    last_updated=pattern.last_seen,
                    supporting_patterns=[pattern.pattern_id],
                    context_weights={}
                )
                principles.append(principle)
        
        return principles
```

### Scenario Plugins

Scenario plugins generate training scenarios for agents.

```python
from typing import List, Dict, Any
from principles_gym.plugins import ScenarioPlugin, register_plugin

@register_plugin(
    name="domain_scenarios",
    version="1.0.0",
    author="Your Name",
    description="Domain-specific scenario generator"
)
class DomainScenarioPlugin(ScenarioPlugin):
    def generate_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single scenario."""
        scenario = {
            "id": f"scenario_{random.randint(1000, 9999)}",
            "type": "custom_type",
            "title": "Scenario Title",
            "description": "Detailed scenario description",
            "options": [
                {
                    "id": "option1",
                    "description": "First option",
                    "consequences": {"outcome": "positive"}
                },
                {
                    "id": "option2", 
                    "description": "Second option",
                    "consequences": {"outcome": "negative"}
                }
            ],
            "metadata": {
                "difficulty": context.get("difficulty", 0.5),
                "tags": ["custom", "domain-specific"]
            }
        }
        return scenario
    
    def generate_batch(self, count: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple scenarios."""
        return [self.generate_scenario(context) for _ in range(count)]
```

### Analysis Plugins

Analysis plugins process training data and generate reports.

```python
from typing import Dict, Any, Union
from principles_gym.plugins import AnalysisPlugin, register_plugin

@register_plugin(
    name="custom_analysis",
    version="1.0.0",
    author="Your Name",
    description="Custom analysis and reporting"
)
class CustomAnalysisPlugin(AnalysisPlugin):
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training data."""
        actions = data.get("actions", [])
        principles = data.get("principles", [])
        
        # Your analysis logic
        results = {
            "metrics": {
                "total_actions": len(actions),
                "total_principles": len(principles),
                "custom_metric": self._calculate_custom_metric(actions)
            },
            "insights": self._generate_insights(actions, principles)
        }
        
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """Generate report from analysis results."""
        format = self.config.get("current_format", "json")
        
        if format == "json":
            return json.dumps(analysis_results, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report(analysis_results)
        else:
            return analysis_results
```

## Plugin Discovery and Loading

### Automatic Discovery

Place your plugin files in the `plugins/builtin/` directory:

```
ai-principles-gym/
└── src/
    └── plugins/
        └── builtin/
            └── my_custom_plugin.py
```

### External Package Loading

1. Create a Python package with your plugins
2. Use the plugin loader to add your package:

```python
from principles_gym.plugins import PLUGIN_LOADER

# Add external package
PLUGIN_LOADER.add_external_package("my_plugin_package")

# Discover plugins
discovered = PLUGIN_LOADER.discover_plugins()
```

### Manual Registration

```python
from principles_gym.plugins import PLUGIN_REGISTRY, PluginType

# Register plugin class
PLUGIN_REGISTRY.register(MyCustomPlugin, PluginType.INFERENCE)

# Create instance
instance = PLUGIN_REGISTRY.create_instance(
    "my_custom_plugin",
    PluginType.INFERENCE,
    config={"threshold": 0.9}
)
```

## Using Plugins via API

### List Available Plugins

```bash
GET /api/plugins/list?plugin_type=inference
```

### Get Plugin Metadata

```bash
GET /api/plugins/metadata/inference/my_custom_plugin
```

### Load a Plugin

```bash
POST /api/plugins/load
{
    "plugin_name": "my_custom_plugin",
    "plugin_type": "inference",
    "config": {
        "threshold": 0.8
    }
}
```

### Use a Plugin

```bash
POST /api/plugins/use
{
    "plugin_name": "my_custom_plugin",
    "plugin_type": "inference",
    "method": "extract_patterns",
    "data": {
        "actions": [...]
    }
}
```

## Best Practices

### 1. Configuration Management

- Use `@plugin_config` for defaults
- Document all configuration options in `config_schema`
- Validate configuration in `validate_config()`

### 2. Error Handling

```python
def extract_patterns(self, actions):
    if not actions:
        logger.warning("No actions provided")
        return []
    
    try:
        patterns = self._process_actions(actions)
    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        return []
    
    return patterns
```

### 3. Performance Optimization

- Use `@cached_method` for expensive computations
- Implement `supports_incremental_learning()` for online learning
- Batch process when possible

### 4. Testing

```python
import pytest
from principles_gym.plugins import PLUGIN_REGISTRY, PluginType

def test_my_plugin():
    # Register plugin
    PLUGIN_REGISTRY.register(MyCustomPlugin, PluginType.INFERENCE)
    
    # Create instance
    plugin = PLUGIN_REGISTRY.create_instance(
        "my_custom_plugin",
        PluginType.INFERENCE,
        config={"threshold": 0.7}
    )
    
    # Test pattern extraction
    actions = [...]  # Test data
    patterns = plugin.extract_patterns(actions)
    
    assert len(patterns) > 0
    assert all(p.confidence >= 0.7 for p in patterns)
```

### 5. Documentation

Always include:
- Clear docstrings for all methods
- Usage examples
- Configuration options
- Dependencies

## Advanced Features

### Hot Reload (Development)

```python
# Reload a plugin during development
from principles_gym.plugins import PLUGIN_LOADER, PluginType

success = PLUGIN_LOADER.reload_plugin("my_plugin", PluginType.INFERENCE)
```

### Plugin Dependencies

Specify dependencies on other plugins:

```python
@register_plugin(
    name="advanced_inference",
    version="1.0.0",
    author="Your Name",
    description="Advanced inference using base patterns",
    dependencies=["plugin:base_pattern_extractor"]
)
```

### Streaming Analysis

For large datasets, implement streaming support:

```python
class StreamingAnalysisPlugin(AnalysisPlugin):
    def supports_streaming(self) -> bool:
        return True
    
    def analyze_stream(self, data_iterator):
        """Process data in chunks."""
        for chunk in data_iterator:
            yield self._process_chunk(chunk)
```

## Example: Complete Custom Plugin

Here's a complete example of a custom inference plugin using machine learning:

```python
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any
from principles_gym.plugins import (
    InferencePlugin, 
    register_plugin, 
    plugin_config,
    cached_method,
    requires_dependencies
)
from principles_gym.core.models import Action, Pattern, Principle

@register_plugin(
    name="ml_inference",
    version="1.0.0",
    author="AI Gym Team",
    description="Machine learning-based inference using DBSCAN clustering",
    dependencies=["numpy", "scikit-learn"],
    tags=["ml", "clustering", "dbscan"]
)
@plugin_config(
    eps=0.3,
    min_samples=5,
    confidence_threshold=0.75,
    feature_dims=10
)
@requires_dependencies("numpy", "scikit-learn")
class MLInferencePlugin(InferencePlugin):
    """
    Advanced inference plugin using DBSCAN clustering.
    """
    
    def initialize(self) -> None:
        """Initialize the ML models."""
        super().initialize()
        
        self.clusterer = DBSCAN(
            eps=self.config["eps"],
            min_samples=self.config["min_samples"]
        )
        
    @cached_method(ttl=300)
    def _extract_features(self, actions: List[Action]) -> np.ndarray:
        """Extract feature vectors from actions."""
        features = []
        
        for action in actions:
            # Create feature vector
            feature = np.zeros(self.config["feature_dims"])
            
            # Encode action type
            action_hash = hash(action.action_taken) % self.config["feature_dims"]
            feature[action_hash] = 1.0
            
            # Add context information
            feature[0] = action.context.value
            feature[1] = action.confidence or 0.5
            
            features.append(feature)
        
        return np.array(features)
    
    def extract_patterns(self, actions: List[Action]) -> List[Pattern]:
        """Extract patterns using DBSCAN clustering."""
        if len(actions) < self.config["min_samples"]:
            return []
        
        # Extract features
        features = self._extract_features(actions)
        
        # Cluster actions
        labels = self.clusterer.fit_predict(features)
        
        # Create patterns from clusters
        patterns = []
        unique_labels = set(labels) - {-1}  # Exclude noise
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_actions = [actions[i] for i in cluster_indices]
            
            pattern = self._create_pattern(cluster_actions, f"cluster_{label}")
            patterns.append(pattern)
        
        return patterns
    
    def infer_principles(self, patterns: List[Pattern]) -> List[Principle]:
        """Infer principles from clustered patterns."""
        principles = []
        
        # Group similar patterns
        pattern_groups = self._group_similar_patterns(patterns)
        
        # Create principles from pattern groups
        for group_id, group_patterns in pattern_groups.items():
            if len(group_patterns) >= 2:  # Need multiple patterns
                principle = self._create_principle_from_group(
                    group_patterns,
                    f"ml_principle_{group_id}"
                )
                principles.append(principle)
        
        return principles
    
    def _create_pattern(self, actions: List[Action], pattern_id: str) -> Pattern:
        """Create a pattern from clustered actions."""
        # Implementation details...
        pass
    
    def _group_similar_patterns(self, patterns: List[Pattern]) -> Dict[str, List[Pattern]]:
        """Group patterns by similarity."""
        # Implementation details...
        pass
    
    def _create_principle_from_group(self, patterns: List[Pattern], principle_id: str) -> Principle:
        """Create a principle from a group of patterns."""
        # Implementation details...
        pass
```

## Troubleshooting

### Common Issues

1. **Plugin not found**
   - Check plugin name is unique
   - Ensure file is in correct directory
   - Verify no syntax errors

2. **Dependencies not satisfied**
   - Install required packages: `pip install package_name`
   - Check plugin dependencies are loaded first

3. **Configuration errors**
   - Validate against config_schema
   - Check required parameters are provided

### Debug Mode

Enable debug logging:

```python
import structlog
logger = structlog.get_logger()
logger.setLevel("DEBUG")
```

## Contributing Plugins

To contribute a plugin to the AI Principles Gym:

1. Fork the repository
2. Create your plugin following the guidelines
3. Add tests for your plugin
4. Update documentation
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
