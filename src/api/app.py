"""FastAPI application for AI Principles Gym."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import settings
from src.core.logging_config import setup_logging
from src.core.database import DatabaseManager
from src.core.monitoring import MetricsCollector, check_system_health as get_system_health
from src.api.routes import router as api_router
from src.api.plugin_routes import router as plugin_router
from src.api.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    LoggingMiddleware,
    TimeoutMiddleware
)
from src.api.training_integration import (
    initialize_training_manager,
    shutdown_training_manager
)
from src.api.websocket import websocket_endpoint

# Setup structured logging
setup_logging()
logger = structlog.get_logger()

# Global storage for active training sessions (replace with Redis in production)
training_sessions: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(
        "starting_application",
        environment=settings.ENVIRONMENT,
        host=settings.API_HOST,
        port=settings.API_PORT
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(settings.DATABASE_URL)
    await db_manager.initialize()
    
    # Initialize training manager
    await initialize_training_manager(db_manager)
    
    # Initialize cache
    # TODO: Initialize Redis connection if configured
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    logger.info("application_started")
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")
    
    # Shutdown training manager
    await shutdown_training_manager()
    
    # Close database connections
    await db_manager.close()
    
    # Close cache connections
    # TODO: Close Redis connection
    
    logger.info("application_shutdown_complete")


# Create FastAPI app
app = FastAPI(
    title="AI Principles Gym",
    description="Framework for training AI agents to develop behavioral principles through experience",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None
)

# Add CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add custom middleware
app.add_middleware(TimeoutMiddleware, timeout=300)  # 5 minute timeout
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(LoggingMiddleware)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "http_exception",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": request_id
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "validation_error",
        request_id=request_id,
        errors=exc.errors(),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "status_code": 422,
                "request_id": request_id,
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log the full exception for debugging
    logger.exception(
        "unexpected_error",
        request_id=request_id,
        path=request.url.path,
        exc_info=exc
    )
    
    # Never expose stack traces to clients
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "status_code": 500,
                "request_id": request_id
            }
        }
    )


# Health check endpoint
@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint for monitoring."""
    # Simplified health check for now
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0"
    }

# Also add health endpoint at /api/health for compatibility
@app.get("/api/health", tags=["monitoring"])
async def api_health_check():
    """Health check endpoint for monitoring (API path)."""
    return await health_check()


# Metrics endpoint (Prometheus format)
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    # TODO: Implement proper Prometheus metrics
    return {
        "active_sessions": len(training_sessions),
        "total_requests": 0,  # TODO: Track this
        "error_rate": 0.0,  # TODO: Calculate this
    }


# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(plugin_router)  # Plugin routes already have /api/plugins prefix

# Add WebSocket endpoint
app.add_websocket_route("/ws/training/{session_id}", websocket_endpoint)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.API_LOG_LEVEL.lower()
    )
