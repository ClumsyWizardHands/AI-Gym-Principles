"""Custom middleware for the API."""

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, Optional

import structlog
from fastapi import Request, Response, status, Query, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(duration)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check and metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier (API key or IP)
        api_key = request.headers.get("X-API-Key")
        client_id = api_key if api_key else (request.client.host if request.client else "unknown")
        
        # Clean old requests
        current_time = time.time()
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.request_counts[client_id]) >= self.requests_per_minute:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                "rate_limit_exceeded",
                request_id=request_id,
                client_id=client_id,
                limit=self.requests_per_minute
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "status_code": 429,
                        "request_id": request_id,
                        "retry_after": 60
                    }
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.request_counts[client_id].append(current_time)
        
        # Process request
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Request timeout middleware."""
    
    def __init__(self, app: ASGIApp, timeout: int = 300):
        super().__init__(app)
        self.timeout = timeout
        
    async def dispatch(self, request: Request, call_next):
        # Skip timeout for training endpoints which may take longer
        if request.url.path.startswith("/api/training"):
            return await call_next(request)
        
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                "request_timeout",
                request_id=request_id,
                path=request.url.path,
                timeout_seconds=self.timeout
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": {
                        "message": "Request timeout",
                        "status_code": 504,
                        "request_id": request_id
                    }
                }
            )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""
    
    def __init__(self, app: ASGIApp, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "message": "Missing API key",
                        "status_code": 401,
                        "request_id": request_id
                    }
                },
                headers={"WWW-Authenticate": "API-Key"}
            )
        
        # TODO: Validate API key against database
        # For now, just check if it's not empty
        if not api_key.strip():
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "message": "Invalid API key",
                        "status_code": 401,
                        "request_id": request_id
                    }
                }
            )
        
        # Store API key in request state for later use
        request.state.api_key = api_key
        
        return await call_next(request)


async def validate_api_key_ws(api_key: Optional[str] = Query(None)) -> str:
    """Validate API key for WebSocket connections."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    # TODO: Validate API key against database
    # For now, just check if it's not empty
    if not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
