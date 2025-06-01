"""Mock LLM adapter for testing without API keys."""
import asyncio
import random
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    """Mock adapter that simulates LLM responses for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock adapter.
        
        Args:
            config: Configuration dictionary (ignored for mock)
        """
        super().__init__(config)
        self.response_delay = config.get("response_delay", 0.5)
        self.mock_responses = [
            "I'll cooperate with you.",
            "Let me think about this carefully.",
            "I prefer to work together.",
            "That seems fair to me.",
            "I'll take the collaborative approach.",
            "Let's find a win-win solution.",
            "I'm considering all options.",
            "Trust is important here.",
            "I'll share resources equally.",
            "Let's build something together."
        ]
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response.
        
        Args:
            prompt: The prompt to respond to
            **kwargs: Additional generation parameters (ignored)
            
        Returns:
            A mock response string
        """
        # Simulate processing time
        await asyncio.sleep(self.response_delay)
        
        # Generate contextual response based on prompt content
        if "cooperate" in prompt.lower() or "collaborate" in prompt.lower():
            responses = [r for r in self.mock_responses if "cooperate" in r.lower() or "together" in r.lower()]
        elif "trust" in prompt.lower():
            responses = [r for r in self.mock_responses if "trust" in r.lower() or "fair" in r.lower()]
        elif "resource" in prompt.lower():
            responses = [r for r in self.mock_responses if "share" in r.lower() or "resource" in r.lower()]
        else:
            responses = self.mock_responses
        
        return random.choice(responses)
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Additional generation parameters (ignored)
            
        Returns:
            List of mock responses
        """
        # Process in parallel with slight delays
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": "mock-model",
            "version": "1.0.0",
            "capabilities": ["text-generation", "testing"],
            "max_tokens": 1000,
            "temperature_range": [0.0, 1.0]
        }
    
    async def health_check(self) -> bool:
        """Check if the mock adapter is healthy.
        
        Returns:
            Always True for mock adapter
        """
        return True
