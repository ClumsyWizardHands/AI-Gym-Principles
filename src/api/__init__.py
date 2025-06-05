"""API module initialization with BehavioralTracker compatibility patch."""

# Apply the BehavioralTracker patch before any imports
import sys

# Check if we need to apply the patch
if 'src.core.tracking' not in sys.modules:
    # Import and apply the patch
    from src.core import tracking_patch
