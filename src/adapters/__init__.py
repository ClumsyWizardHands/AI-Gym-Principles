"""Multi-framework adapters for AI agent integration."""

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .langchain_adapter import LangChainAdapter
from .custom_adapter import CustomAdapter

__all__ = [
    # Base classes
    "AgentInterface",
    "TrainingScenario", 
    "AgentDecision",
    "ParseError",
    
    # Framework adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LangChainAdapter",
    "CustomAdapter",
]
