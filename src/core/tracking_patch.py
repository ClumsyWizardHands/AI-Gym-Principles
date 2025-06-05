"""
Monkey patch for BehavioralTracker to fix compatibility with training_integration.py

This patch modifies the BehavioralTracker class to:
1. Accept agent_id and db_manager parameters in __init__
2. Provide calculate_entropy() and extract_patterns() methods
"""

from . import tracking

# Store original class
_OriginalBehavioralTracker = tracking.BehavioralTracker

# Create patched version
class PatchedBehavioralTracker(_OriginalBehavioralTracker):
    """Patched version of BehavioralTracker with compatibility fixes."""
    
    def __init__(self, agent_id=None, db_manager=None, **kwargs):
        """Initialize with compatibility for training_integration.py parameters."""
        # Store agent_id for reference
        self.agent_id = agent_id
        
        # Map db_manager to database_handler
        if db_manager is not None:
            kwargs['database_handler'] = db_manager
        
        # Call parent with mapped parameters
        super().__init__(**kwargs)
    
    async def calculate_entropy(self):
        """Compatibility method for training_integration.py."""
        # Get recent actions from buffer
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        # Call the real method
        return self.calculate_behavioral_entropy(actions)
    
    async def extract_patterns(self):
        """Compatibility method for training_integration.py."""
        # Get recent actions from buffer
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        # Call the real method and extract patterns list
        result = self.extract_relational_patterns(actions)
        return result.get("patterns", [])


# Apply the monkey patch
tracking.BehavioralTracker = PatchedBehavioralTracker

# Also update in the module's namespace
import sys
if 'src.core.tracking' in sys.modules:
    sys.modules['src.core.tracking'].BehavioralTracker = PatchedBehavioralTracker

print("BehavioralTracker patched successfully for training_integration.py compatibility")
