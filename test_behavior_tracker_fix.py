"""Test file to verify BehaviorTracker initialization fix."""

import asyncio
from src.core.tracking import BehavioralTracker as BehaviorTracker
from src.core.database import DatabaseManager

async def test_behavior_tracker_init():
    """Test the correct initialization of BehaviorTracker."""
    
    # This is the current incorrect initialization
    try:
        tracker = BehaviorTracker(
            agent_id="test_agent",
            db_manager=DatabaseManager("sqlite+aiosqlite:///test.db")
        )
        print("ERROR: Incorrect initialization succeeded (should have failed)")
    except TypeError as e:
        print(f"Expected error: {e}")
    
    # This is the correct initialization
    try:
        db_manager = DatabaseManager("sqlite+aiosqlite:///test.db")
        tracker = BehaviorTracker(
            database_handler=db_manager  # Correct parameter name
        )
        print("SUCCESS: Correct initialization succeeded")
        
        # Check if the tracker has the expected methods
        print(f"Has calculate_behavioral_entropy: {hasattr(tracker, 'calculate_behavioral_entropy')}")
        print(f"Has extract_relational_patterns: {hasattr(tracker, 'extract_relational_patterns')}")
        
        # These are the methods used in training_integration.py
        print(f"Has calculate_entropy: {hasattr(tracker, 'calculate_entropy')}")
        print(f"Has extract_patterns: {hasattr(tracker, 'extract_patterns')}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_behavior_tracker_init())
