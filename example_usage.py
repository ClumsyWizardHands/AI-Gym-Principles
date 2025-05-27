"""Example usage of AI Principles Gym setup."""

import asyncio
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core import setup_logging, get_logger, settings


async def main():
    """Example of using the configured system."""
    # Setup logging
    setup_logging()
    
    # Get a logger
    logger = get_logger(__name__)
    
    # Log some information
    logger.info(
        "ai_principles_gym_started",
        environment=settings.ENVIRONMENT,
        api_port=settings.API_PORT,
        min_pattern_length=settings.MIN_PATTERN_LENGTH,
        consistency_threshold=settings.CONSISTENCY_THRESHOLD,
    )
    
    # Show configuration
    print("AI Principles Gym Configuration:")
    print(f"  Environment: {settings.ENVIRONMENT}")
    print(f"  API Port: {settings.API_PORT}")
    print(f"  Min Pattern Length: {settings.MIN_PATTERN_LENGTH}")
    print(f"  Consistency Threshold: {settings.CONSISTENCY_THRESHOLD}")
    print(f"  Entropy Threshold: {settings.ENTROPY_THRESHOLD}")
    print(f"  Max Scenarios per Session: {settings.MAX_SCENARIOS_PER_SESSION}")
    print(f"  Database URL: {settings.DATABASE_URL}")
    
    logger.info("configuration_displayed", status="success")


if __name__ == "__main__":
    asyncio.run(main())
