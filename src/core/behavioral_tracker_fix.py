"""
Fix for BehavioralTracker compatibility issues in training_integration.py

This module provides a fixed version of BehavioralTracker that:
1. Accepts the parameters that training_integration.py is trying to use
2. Provides the methods that training_integration.py expects
"""

from typing import Optional, Any, Dict, List
from .tracking import BehavioralTracker
from .models import Action


class BehavioralTrackerFixed(BehavioralTracker):
    """Fixed version of BehavioralTracker with compatibility patches."""
    
    def __init__(self, agent_id: str = None, db_manager: Any = None, **kwargs):
        """Initialize with compatibility for training_integration.py.
        
        Args:
            agent_id: Agent identifier (stored for reference)
            db_manager: Database manager (passed as database_handler)
            **kwargs: Additional arguments for BehavioralTracker
        """
        # Store agent_id for reference
        self.agent_id = agent_id
        
        # Initialize parent with correct parameter name
        super().__init__(
            database_handler=db_manager,
            **kwargs
        )
    
    async def calculate_entropy(self) -> float:
        """Compatibility method that provides expected interface."""
        # Get recent actions from buffer
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        # Call the actual method
        return self.calculate_behavioral_entropy(actions)
    
    async def extract_patterns(self) -> List[Dict[str, Any]]:
        """Compatibility method that provides expected interface."""
        # Get recent actions from buffer
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        # Call the actual method and extract patterns list
        result = self.extract_relational_patterns(actions)
        return result.get("patterns", [])


# Export the fixed class with the name expected by training_integration.py
BehaviorTracker = BehavioralTrackerFixed
