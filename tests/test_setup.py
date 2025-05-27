"""Basic tests to verify project setup."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that core imports work correctly."""
    from src.core import settings, get_logger, setup_logging
    
    assert settings is not None
    assert get_logger is not None
    assert setup_logging is not None


def test_settings_defaults():
    """Test that settings have correct defaults."""
    from src.core import settings
    
    assert settings.API_PORT == 8000
    assert settings.MIN_PATTERN_LENGTH == 20
    assert settings.CONSISTENCY_THRESHOLD == 0.85
    assert settings.ENTROPY_THRESHOLD == 0.7


def test_logger_creation():
    """Test that logger can be created."""
    from src.core import get_logger
    
    logger = get_logger(__name__)
    assert logger is not None
    
    # Test that logger can log without errors
    logger.info("test_message", test_data="value")


def test_project_structure():
    """Test that project structure is correct."""
    assert project_root.exists()
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "deployment").exists()
    assert (project_root / "client").exists()
    assert (project_root / "memory-bank").exists()
    
    # Check core modules
    assert (project_root / "src" / "core").exists()
    assert (project_root / "src" / "scenarios").exists()
    assert (project_root / "src" / "adapters").exists()
    assert (project_root / "src" / "api").exists()
