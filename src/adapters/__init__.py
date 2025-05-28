"""Multi-framework adapters for AI agent integration."""

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .custom_adapter import CustomAdapter
from .http_adapter import HTTPAdapter

# Optional LangChain import
try:
    from .langchain_adapter import LangChainAdapter
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Create a placeholder that raises an informative error
    class LangChainAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain is not installed. Please install it with:\n"
                "pip install -r requirements-langchain.txt\n"
                "Note: LangChain packages may have dependency conflicts. "
                "Test thoroughly before using in production."
            )

__all__ = [
    # Base classes
    "AgentInterface",
    "TrainingScenario", 
    "AgentDecision",
    "ParseError",
    
    # Framework adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LangChainAdapter",  # Will be placeholder if LangChain not installed
    "CustomAdapter",
    "HTTPAdapter",
]
