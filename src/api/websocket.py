"""WebSocket implementation for real-time training updates."""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Set, Optional, Any
from collections import defaultdict
from contextlib import asynccontextmanager

import structlog
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.exceptions import WebSocketException

from src.core.config import settings
from src.api.middleware import validate_api_key_ws

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for training sessions."""
    
    def __init__(self):
        # Track connections by session_id
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Track connection metadata
        self._connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        # Rate limiting per connection
        self._message_counts: Dict[WebSocket, Dict[str, int]] = defaultdict(
            lambda: {"count": 0, "window_start": time.time()}
        )
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(
        self, 
        websocket: WebSocket, 
        session_id: str,
        user_id: str
    ):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        async with self._lock:
            self._connections[session_id].add(websocket)
            self._connection_info[websocket] = {
                "session_id": session_id,
                "user_id": user_id,
                "connected_at": datetime.utcnow(),
                "messages_sent": 0
            }
        
        logger.info(
            "websocket_connected",
            session_id=session_id,
            user_id=user_id,
            total_connections=len(self._connections[session_id])
        )
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            info = self._connection_info.get(websocket)
            if info:
                session_id = info["session_id"]
                self._connections[session_id].discard(websocket)
                
                # Clean up empty session sets
                if not self._connections[session_id]:
                    del self._connections[session_id]
                
                del self._connection_info[websocket]
                
                # Clean up rate limiting data
                if websocket in self._message_counts:
                    del self._message_counts[websocket]
                
                logger.info(
                    "websocket_disconnected",
                    session_id=session_id,
                    user_id=info["user_id"],
                    duration_seconds=(
                        datetime.utcnow() - info["connected_at"]
                    ).total_seconds(),
                    messages_sent=info["messages_sent"]
                )
    
    def _check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check if connection has exceeded rate limit."""
        now = time.time()
        window = self._message_counts[websocket]
        
        # Reset window if expired (1 minute window)
        if now - window["window_start"] > 60:
            window["count"] = 0
            window["window_start"] = now
        
        # Check limit (max 120 messages per minute)
        if window["count"] >= settings.WEBSOCKET_RATE_LIMIT:
            return False
        
        window["count"] += 1
        return True
    
    async def send_message(
        self,
        websocket: WebSocket,
        message_type: str,
        data: Dict[str, Any]
    ):
        """Send a message to a specific WebSocket connection."""
        try:
            # Check rate limit
            if not self._check_rate_limit(websocket):
                logger.warning(
                    "websocket_rate_limit_exceeded",
                    websocket_id=id(websocket)
                )
                return
            
            # Prepare message
            message = {
                "type": message_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Send message
            await websocket.send_json(message)
            
            # Update stats
            if websocket in self._connection_info:
                self._connection_info[websocket]["messages_sent"] += 1
        
        except Exception as e:
            logger.error(
                "websocket_send_error",
                error=str(e),
                message_type=message_type
            )
            # Connection is likely closed, remove it
            await self.disconnect(websocket)
    
    async def broadcast_update(
        self,
        session_id: str,
        message_type: str,
        data: Dict[str, Any]
    ):
        """Broadcast an update to all connections for a session."""
        connections = list(self._connections.get(session_id, []))
        
        if not connections:
            return
        
        logger.debug(
            "broadcasting_update",
            session_id=session_id,
            message_type=message_type,
            connection_count=len(connections)
        )
        
        # Send to all connections concurrently
        tasks = [
            self.send_message(websocket, message_type, data)
            for websocket in connections
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_progress(
        self,
        session_id: str,
        current_scenario: int,
        total_scenarios: int,
        progress_percentage: float
    ):
        """Broadcast training progress update."""
        await self.broadcast_update(
            session_id,
            "progress",
            {
                "current_scenario": current_scenario,
                "total_scenarios": total_scenarios,
                "progress_percentage": progress_percentage,
                "message": f"Processing scenario {current_scenario}/{total_scenarios}"
            }
        )
    
    async def broadcast_principle_discovered(
        self,
        session_id: str,
        principle_name: str,
        principle_description: str,
        strength: float,
        evidence_count: int
    ):
        """Broadcast when a new principle is discovered."""
        await self.broadcast_update(
            session_id,
            "principle_discovered",
            {
                "principle_name": principle_name,
                "principle_description": principle_description,
                "strength": strength,
                "evidence_count": evidence_count,
                "message": f"Discovered principle: {principle_name}"
            }
        )
    
    async def broadcast_status_change(
        self,
        session_id: str,
        old_status: str,
        new_status: str,
        message: Optional[str] = None
    ):
        """Broadcast training status change."""
        status_messages = {
            "pending": "Training session queued",
            "running": "Training session started",
            "completed": "Training session completed successfully",
            "failed": "Training session failed"
        }
        
        await self.broadcast_update(
            session_id,
            "status_change",
            {
                "old_status": old_status,
                "new_status": new_status,
                "message": message or status_messages.get(
                    new_status,
                    f"Status changed to {new_status}"
                )
            }
        )
    
    async def broadcast_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        recoverable: bool = True
    ):
        """Broadcast error during training."""
        await self.broadcast_update(
            session_id,
            "error",
            {
                "error_type": error_type,
                "error_message": error_message,
                "recoverable": recoverable,
                "message": f"Error: {error_message}"
            }
        )
    
    def get_connection_count(self, session_id: str) -> int:
        """Get number of active connections for a session."""
        return len(self._connections.get(session_id, []))
    
    def get_all_sessions(self) -> Set[str]:
        """Get all sessions with active connections."""
        return set(self._connections.keys())
    
    async def close_session_connections(self, session_id: str):
        """Close all connections for a session."""
        connections = list(self._connections.get(session_id, []))
        
        for websocket in connections:
            try:
                await websocket.close(
                    code=status.WS_1000_NORMAL_CLOSURE,
                    reason="Training session ended"
                )
            except Exception:
                pass
            
            await self.disconnect(websocket)


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_websocket_user(
    api_key: str = Depends(validate_api_key_ws)
) -> Dict[str, Any]:
    """Get user information from WebSocket API key."""
    # In a real implementation, this would look up the user from the API key
    # For now, we'll return a mock user
    return {
        "user_id": api_key[:8],  # Use first 8 chars of API key as user ID
        "api_key": api_key
    }


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    user: Dict[str, Any] = Depends(get_websocket_user)
):
    """
    WebSocket endpoint for real-time training updates.
    
    Connect using: ws://localhost:8000/ws/training/{session_id}?api_key=YOUR_KEY
    
    Message types sent by server:
    - progress: Training progress updates
    - principle_discovered: New principle found
    - status_change: Training status changes
    - error: Error occurred during training
    
    Each message has format:
    {
        "type": "message_type",
        "timestamp": "ISO 8601 timestamp",
        "data": {
            // Type-specific data
        }
    }
    """
    # Validate session ownership
    # In a real implementation, check if user owns this session
    # For now, we'll allow any authenticated user
    
    try:
        # Connect client
        await connection_manager.connect(
            websocket,
            session_id,
            user["user_id"]
        )
        
        # Send initial connection success message
        await connection_manager.send_message(
            websocket,
            "connection_established",
            {
                "session_id": session_id,
                "message": "Connected to training session updates"
            }
        )
        
        # Keep connection alive
        try:
            while True:
                # Wait for any client messages (ping/pong)
                message = await websocket.receive_json()
                
                # Handle ping messages
                if message.get("type") == "ping":
                    await connection_manager.send_message(
                        websocket,
                        "pong",
                        {"timestamp": datetime.utcnow().isoformat()}
                    )
                
                # Ignore other client messages for now
        
        except WebSocketDisconnect:
            pass
    
    finally:
        # Clean up connection
        await connection_manager.disconnect(websocket)


# Integration hooks for training process
async def notify_training_progress(
    session_id: str,
    current_scenario: int,
    total_scenarios: int
):
    """Helper function to notify training progress."""
    progress = (current_scenario / total_scenarios) * 100
    await connection_manager.broadcast_progress(
        session_id,
        current_scenario,
        total_scenarios,
        progress
    )


async def notify_principle_discovered(
    session_id: str,
    principle: Any  # Would be Principle model
):
    """Helper function to notify principle discovery."""
    await connection_manager.broadcast_principle_discovered(
        session_id,
        principle.name,
        principle.description,
        principle.strength,
        len(principle.supporting_actions)
    )


async def notify_status_change(
    session_id: str,
    old_status: str,
    new_status: str,
    message: Optional[str] = None
):
    """Helper function to notify status change."""
    await connection_manager.broadcast_status_change(
        session_id,
        old_status,
        new_status,
        message
    )


async def notify_training_error(
    session_id: str,
    error_type: str,
    error_message: str,
    recoverable: bool = True
):
    """Helper function to notify training error."""
    await connection_manager.broadcast_error(
        session_id,
        error_type,
        error_message,
        recoverable
    )
