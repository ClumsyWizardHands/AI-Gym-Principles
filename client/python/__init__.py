"""AI Principles Gym Python Client.

Easy-to-use client library for interacting with the AI Principles Gym training system.
"""

from .principles_gym_client import (
    # Main clients
    PrinciplesGymClient,
    AsyncPrinciplesGymClient,
    
    # Exceptions
    APIError,
    AuthenticationError,
    RateLimitError,
    ResourceNotFoundError,
    TrainingError,
    
    # Response models
    APIKeyResponse,
    AgentRegistrationResponse,
    TrainingResponse,
    TrainingStatus,
    PrincipleReport,
    TrainingReport
)

__version__ = "0.1.0"
__author__ = "AI Principles Gym Team"

__all__ = [
    # Clients
    "PrinciplesGymClient",
    "AsyncPrinciplesGymClient",
    
    # Exceptions
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "ResourceNotFoundError",
    "TrainingError",
    
    # Response models
    "APIKeyResponse",
    "AgentRegistrationResponse",
    "TrainingResponse",
    "TrainingStatus",
    "PrincipleReport",
    "TrainingReport"
]
