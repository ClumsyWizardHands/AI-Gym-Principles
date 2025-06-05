"""Wrapper for BehavioralTracker to provide compatibility with training_integration.py"""

from typing import Optional, Any, Dict, List
from .tracking import BehavioralTracker
from .models import Action


class BehavioralTrackerWrapper(BehavioralTracker):
    """Wrapper that provides the expected interface for training_integration.py"""
    
    def __init__(self, agent_id: str = None, db_manager: Any = None, **kwargs):
        """Initialize with compatibility for old interface.
        
        Args:
            agent_id: Agent identifier (stored but not used by base class)
            db_manager: Database manager (passed as database_handler)
            **kwargs: Additional arguments for BehavioralTracker
        """
        # Store agent_id for potential future use
        self.agent_id = agent_id
        
        # Initialize parent with correct parameter name
        super().__init__(
            database_handler=db_manager,
            **kwargs
        )
    
    async def calculate_entropy(self) -> float:
        """Compatibility method that calls calculate_behavioral_entropy."""
        # Get recent actions for the agent
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        return self.calculate_behavioral_entropy(actions)
    
    async def extract_patterns(self) -> List[Dict[str, Any]]:
        """Compatibility method that calls extract_relational_patterns."""
        # Get recent actions for the agent
        actions = []
        if hasattr(self, '_action_buffer'):
            with self._buffer_lock:
                actions = list(self._action_buffer)
        
        result = self.extract_relational_patterns(actions)
        return result.get("patterns", [])
