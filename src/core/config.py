"""Application configuration using Pydantic settings."""

from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    ENVIRONMENT: str = Field(default="development")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=True)
    API_LOG_LEVEL: str = Field(default="INFO")
    
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data/principles.db")
    DATABASE_ECHO: bool = Field(default=False)
    DATABASE_POOL_SIZE: int = Field(default=5)
    DATABASE_MAX_OVERFLOW: int = Field(default=10)
    
    # Inference Configuration
    MIN_PATTERN_LENGTH: int = Field(default=20)
    CONSISTENCY_THRESHOLD: float = Field(default=0.85)
    ENTROPY_THRESHOLD: float = Field(default=0.7)
    PRINCIPLE_CONFIDENCE_THRESHOLD: float = Field(default=0.8)
    TEMPORAL_WINDOW_SIZE: int = Field(default=50)
    
    # Performance Limits
    MAX_SCENARIOS_PER_SESSION: int = Field(default=500)
    MAX_CONCURRENT_SESSIONS: int = Field(default=10)
    ACTION_BUFFER_SIZE: int = Field(default=1000)
    MEMORY_CACHE_SIZE: int = Field(default=100)
    SESSION_TIMEOUT_MINUTES: int = Field(default=30)
    ACTION_TIMEOUT_SECONDS: int = Field(default=30)
    WEBSOCKET_RATE_LIMIT: int = Field(default=120)  # Messages per minute
    
    # Scenario Engine Configuration
    DEFAULT_SCENARIO_TIMEOUT: int = Field(default=300)
    SCENARIO_STEP_DELAY: float = Field(default=0.1)
    MAX_SCENARIO_RETRIES: int = Field(default=3)
    SCENARIO_RANDOM_SEED: int = Field(default=42)
    
    # Agent Configuration
    AGENT_DEFAULT_LEARNING_RATE: float = Field(default=0.01)
    AGENT_EXPLORATION_RATE: float = Field(default=0.1)
    AGENT_MEMORY_CAPACITY: int = Field(default=10000)
    AGENT_UPDATE_FREQUENCY: int = Field(default=100)
    
    # Logging Configuration
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE_PATH: Optional[str] = Field(default="./logs/ai_principles.log")
    LOG_FILE_ROTATION: str = Field(default="100MB")
    LOG_FILE_RETENTION: int = Field(default=7)
    LOG_INCLUDE_TIMESTAMPS: bool = Field(default=True)
    LOG_INCLUDE_CONTEXT: bool = Field(default=True)
    
    # Monitoring and Metrics
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    METRICS_COLLECTION_INTERVAL: int = Field(default=60)
    ENABLE_HEALTH_CHECK: bool = Field(default=True)
    HEALTH_CHECK_INTERVAL: int = Field(default=30)
    
    # Feature Flags
    ENABLE_PRINCIPLE_VISUALIZATION: bool = Field(default=True)
    ENABLE_REAL_TIME_UPDATES: bool = Field(default=True)
    ENABLE_BATCH_PROCESSING: bool = Field(default=False)
    ENABLE_EXPERIMENTAL_FEATURES: bool = Field(default=False)
    
    # External Services
    VECTOR_DB_URL: Optional[str] = Field(default=None)
    CACHE_REDIS_URL: Optional[str] = Field(default=None)
    MESSAGE_QUEUE_URL: Optional[str] = Field(default=None)
    
    # Security
    API_KEY: Optional[str] = Field(default=None)
    ENABLE_CORS: bool = Field(default=True)
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"]
    )
    JWT_SECRET_KEY: Optional[str] = Field(default=None)
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_MINUTES: int = Field(default=60)
    
    # Development Settings
    DEBUG_MODE: bool = Field(default=False)
    PROFILE_PERFORMANCE: bool = Field(default=False)
    MOCK_EXTERNAL_SERVICES: bool = Field(default=False)
    TESTING_MODE: bool = Field(default=False)
    
    # LLM Analysis Configuration
    ANALYSIS_LLM_PROVIDER: str = Field(default="none")  # "anthropic", "openai", or "none"
    ANALYSIS_LLM_MODEL: str = Field(default="claude-opus-4-20250514")
    ANALYSIS_LLM_API_KEY: Optional[str] = Field(default=None)
    ANALYSIS_LLM_TEMPERATURE: float = Field(default=0.3)
    ANALYSIS_LLM_MAX_TOKENS: int = Field(default=2000)
    ANALYSIS_LLM_TIMEOUT: int = Field(default=30)
    
    # LLM Feature Flags
    ENABLE_LLM_PRINCIPLE_GENERATION: bool = Field(default=True)
    ENABLE_LLM_CONTRADICTION_DETECTION: bool = Field(default=True)
    ENABLE_LLM_SCENARIO_ENHANCEMENT: bool = Field(default=True)
    ENABLE_LLM_PERSONALITY_INSIGHTS: bool = Field(default=True)
    
    # LLM Performance Settings
    LLM_CACHE_TTL_SECONDS: int = Field(default=3600)
    LLM_MAX_RETRIES: int = Field(default=3)
    LLM_RETRY_DELAY: float = Field(default=1.0)
    LLM_BATCH_SIZE: int = Field(default=10)
    
    @field_validator("CONSISTENCY_THRESHOLD", "ENTROPY_THRESHOLD", "PRINCIPLE_CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure thresholds are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v
    
    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Ensure log format is valid."""
        if v not in ["json", "text"]:
            raise ValueError("LOG_FORMAT must be 'json' or 'text'")
        return v
    
    @field_validator("API_LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"API_LOG_LEVEL must be one of {valid_levels}")
        return v.upper()


# Create a singleton instance
settings = Settings()
