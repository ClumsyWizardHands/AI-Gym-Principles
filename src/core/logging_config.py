"""Structured logging configuration using structlog."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder
from structlog.types import EventDict, Processor

from .config import settings


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application context to all log entries."""
    event_dict["app"] = "ai-principles-gym"
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create logs directory if it doesn't exist
    if settings.LOG_FILE_PATH:
        log_dir = Path(settings.LOG_FILE_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure processors
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_app_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.LOG_INCLUDE_CONTEXT:
        processors.append(
            CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ]
            )
        )
    
    # Configure output format
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.API_LOG_LEVEL.upper()),
    )
    
    # Add file handler if configured
    if settings.LOG_FILE_PATH:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE_PATH,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=settings.LOG_FILE_RETENTION,
        )
        file_handler.setLevel(getattr(logging, settings.API_LOG_LEVEL.upper()))
        
        # Add formatter for file output
        if settings.LOG_FORMAT == "json":
            file_handler.setFormatter(
                logging.Formatter("%(message)s")
            )
        
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# Usage example:
# from src.core.logging_config import get_logger
# logger = get_logger(__name__)
# logger.info("principle_inferred", principle_id="123", confidence=0.95)
