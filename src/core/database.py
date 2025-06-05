"""Database models and persistence layer for AI Principles Gym.

Provides async SQLAlchemy models with performance optimizations including:
- Batch inserts with configurable buffer size
- Connection pooling
- Composite indexes for efficient queries
- JSON columns for flexible metadata storage
- Query monitoring and metrics
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import json
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, JSON, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    select, and_, func, event
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import declarative_base, relationship, selectinload
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.dialects.postgresql import UUID
import asyncio
from collections import deque
import time

import structlog
from src.core.config import settings

logger = structlog.get_logger()

# Create declarative base
Base = declarative_base()

# Query monitoring metrics
query_metrics = {
    "total_queries": 0,
    "slow_queries": 0,
    "query_times": deque(maxlen=1000)  # Keep last 1000 query times
}


class AgentProfile(Base):
    """SQLAlchemy model for agent profiles."""
    
    __tablename__ = "agent_profiles"
    
    agent_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    behavioral_entropy = Column(Float, default=0.0, nullable=False)
    total_actions = Column(Integer, default=0, nullable=False)
    meta_data = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    actions = relationship("Action", back_populates="agent", cascade="all, delete-orphan")
    principles = relationship("Principle", back_populates="agent", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('behavioral_entropy >= 0 AND behavioral_entropy <= 1', 
                       name='check_entropy_range'),
        CheckConstraint('total_actions >= 0', name='check_positive_actions'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "behavioral_entropy": self.behavioral_entropy,
            "total_actions": self.total_actions,
            "metadata": self.meta_data
        }


class Action(Base):
    """SQLAlchemy model for agent actions."""
    
    __tablename__ = "actions"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey("agent_profiles.agent_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Relational data as JSON
    relational_data = Column(JSON, nullable=False)
    # Contains: actor, target, relationship_type, impact_magnitude
    
    # Decision context
    decision_context = Column(String, nullable=False)
    context_weight = Column(Float, nullable=False)
    
    # Action details
    action_type = Column(String, nullable=False)
    outcome_valence = Column(Float, nullable=False)
    decision_entropy = Column(Float, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    
    # Additional context
    context = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    agent = relationship("AgentProfile", back_populates="actions")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_agent_timestamp', 'agent_id', 'timestamp'),
        Index('idx_decision_context', 'decision_context'),
        Index('idx_timestamp_partition', 'timestamp'),  # For date partitioning
        CheckConstraint('outcome_valence >= -1 AND outcome_valence <= 1', 
                       name='check_outcome_range'),
        CheckConstraint('decision_entropy >= 0 AND decision_entropy <= 1', 
                       name='check_entropy_range'),
        CheckConstraint('latency_ms >= 0', name='check_positive_latency'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "relational_data": self.relational_data,
            "decision_context": self.decision_context,
            "context_weight": self.context_weight,
            "action_type": self.action_type,
            "outcome_valence": self.outcome_valence,
            "decision_entropy": self.decision_entropy,
            "latency_ms": self.latency_ms,
            "context": self.context
        }


class Principle(Base):
    """SQLAlchemy model for discovered principles."""
    
    __tablename__ = "principles"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey("agent_profiles.agent_id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    
    # Strength and confidence
    strength_score = Column(Float, default=0.0, nullable=False)
    confidence_lower = Column(Float, default=0.0, nullable=False)
    confidence_upper = Column(Float, default=0.0, nullable=False)
    volatility = Column(Float, default=0.0, nullable=False)
    
    # Evidence tracking
    evidence_count = Column(Integer, default=0, nullable=False)
    contradictions_count = Column(Integer, default=0, nullable=False)
    
    # Lineage data as JSON
    lineage_data = Column(JSON, default=dict, nullable=False)
    # Contains: parent_ids, child_ids, lineage_type, transformation_reason
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional metadata
    meta_data = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    agent = relationship("AgentProfile", back_populates="principles")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_agent_principle', 'agent_id', 'name'),
        Index('idx_strength_score', 'strength_score'),
        CheckConstraint('strength_score >= 0 AND strength_score <= 1', 
                       name='check_strength_range'),
        CheckConstraint('volatility >= 0 AND volatility <= 1', 
                       name='check_volatility_range'),
        CheckConstraint('confidence_lower <= confidence_upper', 
                       name='check_confidence_order'),
        CheckConstraint('evidence_count >= 0 AND contradictions_count >= 0', 
                       name='check_positive_counts'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "strength_score": self.strength_score,
            "confidence_interval": (self.confidence_lower, self.confidence_upper),
            "volatility": self.volatility,
            "evidence_count": self.evidence_count,
            "contradictions_count": self.contradictions_count,
            "lineage_data": self.lineage_data,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.meta_data
        }


class DatabaseManager:
    """Manages database connections and provides high-level operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager with connection pooling."""
        self.database_url = database_url or settings.DATABASE_URL
        
        # Configure connection pool based on database type
        if "sqlite" in self.database_url:
            # SQLite doesn't support connection pooling well
            pool_class = NullPool
            pool_kwargs = {}
        else:
            # PostgreSQL/MySQL can handle connection pooling
            pool_class = AsyncAdaptedQueuePool
            pool_kwargs = {
                "pool_size": 20,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 3600,  # Recycle connections after 1 hour
            }
        
        # Create async engine with query logging
        self.engine = create_async_engine(
            self.database_url,
            poolclass=pool_class,
            echo=settings.DEBUG_MODE,  # Enable SQL logging in debug mode
            future=True,
            **pool_kwargs
        )
        
        # Create async session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Action buffer for batch inserts
        self.action_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = asyncio.Lock()
        self.flush_task: Optional[asyncio.Task] = None
        
        # Set up query monitoring
        self._setup_query_monitoring()
    
    def _setup_query_monitoring(self):
        """Set up monitoring for slow queries."""
        @event.listens_for(self.engine.sync_engine, "before_execute")
        def receive_before_execute(conn, clauseelement, multiparams, params, execution_options):
            conn.info["query_start_time"] = time.time()
        
        @event.listens_for(self.engine.sync_engine, "after_execute")
        def receive_after_execute(conn, clauseelement, multiparams, params, execution_options, result):
            query_time = time.time() - conn.info.get("query_start_time", time.time())
            query_metrics["total_queries"] += 1
            query_metrics["query_times"].append(query_time)
            
            # Log slow queries (> 100ms)
            if query_time > 0.1:
                query_metrics["slow_queries"] += 1
                logger.warning(
                    "slow_query_detected",
                    query_time=query_time,
                    query=str(clauseelement)[:200]  # Truncate long queries
                )
    
    async def initialize(self):
        """Initialize database schema and start background tasks."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Start periodic flush task
        self.flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info("database_initialized", url=self.database_url)
    
    async def close(self):
        """Close database connections and stop background tasks."""
        # Stop flush task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining actions
        await self.flush_actions()
        
        # Close engine
        await self.engine.dispose()
        
        logger.info("database_closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional session context."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def buffer_action(self, action_data: Dict[str, Any]):
        """Add action to buffer for batch insert."""
        async with self.buffer_lock:
            self.action_buffer.append(action_data)
            
            # Flush if buffer is full (1000 actions)
            if len(self.action_buffer) >= 1000:
                await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush action buffer to database."""
        if not self.action_buffer:
            return
        
        actions_to_insert = self.action_buffer.copy()
        self.action_buffer.clear()
        
        try:
            async with self.session() as session:
                # Bulk insert actions
                await session.execute(
                    Action.__table__.insert(),
                    actions_to_insert
                )
                
                # Update agent action counts
                agent_counts = {}
                for action in actions_to_insert:
                    agent_id = action["agent_id"]
                    agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
                
                for agent_id, count in agent_counts.items():
                    await session.execute(
                        AgentProfile.__table__.update()
                        .where(AgentProfile.agent_id == agent_id)
                        .values(total_actions=AgentProfile.total_actions + count)
                    )
            
            logger.info(
                "actions_flushed",
                count=len(actions_to_insert),
                agents=len(agent_counts)
            )
        
        except Exception as e:
            logger.error(
                "action_flush_failed",
                error=str(e),
                count=len(actions_to_insert)
            )
            # Re-add actions to buffer for retry
            async with self.buffer_lock:
                self.action_buffer.extend(actions_to_insert)
    
    async def _periodic_flush(self):
        """Periodically flush action buffer every 30 seconds."""
        while True:
            try:
                await asyncio.sleep(30)
                async with self.buffer_lock:
                    if self.action_buffer:
                        await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("periodic_flush_error", error=str(e))
    
    async def flush_actions(self):
        """Force flush all buffered actions."""
        async with self.buffer_lock:
            await self._flush_buffer()
    
    # High-level database operations
    
    async def create_agent(self, agent_id: str, name: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> AgentProfile:
        """Create a new agent profile."""
        async with self.session() as session:
            agent = AgentProfile(
                agent_id=agent_id,
                name=name,
                meta_data=metadata or {}
            )
            session.add(agent)
            await session.flush()
            await session.refresh(agent)
            
            logger.info("agent_created", agent_id=agent_id, name=name)
            return agent
    
    async def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(AgentProfile)
                .where(AgentProfile.agent_id == agent_id)
                .options(selectinload(AgentProfile.principles))
            )
            return result.scalar_one_or_none()
    
    async def get_agent_by_id(self, agent_id: str, session: AsyncSession) -> Optional[AgentProfile]:
        """Get agent profile by ID using provided session."""
        result = await session.execute(
            select(AgentProfile)
            .where(AgentProfile.agent_id == agent_id)
            .options(selectinload(AgentProfile.principles))
        )
        return result.scalar_one_or_none()
    
    async def record_action(self, action_data: Dict[str, Any]):
        """Record an action (uses buffering for efficiency)."""
        # Prepare action data for database
        db_action = {
            "id": action_data["id"],
            "agent_id": action_data["agent_id"],
            "timestamp": action_data["timestamp"],
            "relational_data": action_data["relational_anchor"],
            "decision_context": action_data["decision_context"]["context"],
            "context_weight": action_data["decision_context"]["weight"],
            "action_type": action_data["action_type"],
            "outcome_valence": action_data["outcome_valence"],
            "decision_entropy": action_data["decision_entropy"],
            "latency_ms": action_data["latency_ms"],
            "context": action_data.get("metadata", {})
        }
        
        await self.buffer_action(db_action)
    
    async def get_agent_actions(self, agent_id: str, limit: int = 100,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[Action]:
        """Get agent actions with optional time filtering."""
        async with self.session() as session:
            query = (
                select(Action)
                .where(Action.agent_id == agent_id)
                .order_by(Action.timestamp.desc())
                .limit(limit)
            )
            
            if start_time:
                query = query.where(Action.timestamp >= start_time)
            if end_time:
                query = query.where(Action.timestamp <= end_time)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def upsert_principle(self, principle_data: Dict[str, Any]) -> Principle:
        """Create or update a principle."""
        async with self.session() as session:
            # Check if principle exists
            result = await session.execute(
                select(Principle).where(Principle.id == principle_data["id"])
            )
            principle = result.scalar_one_or_none()
            
            if principle:
                # Update existing principle
                for key, value in principle_data.items():
                    if key == "confidence_interval":
                        principle.confidence_lower = value[0]
                        principle.confidence_upper = value[1]
                    elif hasattr(principle, key):
                        setattr(principle, key, value)
                principle.last_updated = datetime.utcnow()
            else:
                # Create new principle
                confidence = principle_data.pop("confidence_interval", (0.0, 0.0))
                principle = Principle(
                    **principle_data,
                    confidence_lower=confidence[0],
                    confidence_upper=confidence[1]
                )
                session.add(principle)
            
            await session.flush()
            await session.refresh(principle)
            
            logger.info(
                "principle_upserted",
                principle_id=principle.id,
                agent_id=principle.agent_id,
                strength=principle.strength_score
            )
            return principle
    
    async def get_agent_principles(self, agent_id: str,
                                 min_strength: Optional[float] = None) -> List[Principle]:
        """Get agent principles with optional strength filtering."""
        async with self.session() as session:
            query = (
                select(Principle)
                .where(Principle.agent_id == agent_id)
                .order_by(Principle.strength_score.desc())
            )
            
            if min_strength is not None:
                query = query.where(Principle.strength_score >= min_strength)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def update_behavioral_entropy(self, agent_id: str, entropy: float):
        """Update agent's behavioral entropy."""
        async with self.session() as session:
            await session.execute(
                AgentProfile.__table__.update()
                .where(AgentProfile.agent_id == agent_id)
                .values(behavioral_entropy=entropy)
            )
    
    async def get_query_metrics(self) -> Dict[str, Any]:
        """Get database query performance metrics."""
        avg_time = (
            sum(query_metrics["query_times"]) / len(query_metrics["query_times"])
            if query_metrics["query_times"] else 0
        )
        
        return {
            "total_queries": query_metrics["total_queries"],
            "slow_queries": query_metrics["slow_queries"],
            "average_query_time": avg_time,
            "recent_query_count": len(query_metrics["query_times"])
        }
    
    async def cleanup_old_actions(self, days: int = 30):
        """Clean up actions older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.session() as session:
            result = await session.execute(
                Action.__table__.delete()
                .where(Action.timestamp < cutoff_date)
            )
            
            logger.info(
                "old_actions_cleaned",
                deleted_count=result.rowcount,
                cutoff_date=cutoff_date.isoformat()
            )


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()
    return db_manager


# Alembic migration template
ALEMBIC_TEMPLATE = """
# Alembic Migration Setup

To set up database migrations:

1. Install Alembic:
   ```bash
   pip install alembic
   ```

2. Initialize Alembic:
   ```bash
   alembic init alembic
   ```

3. Update alembic.ini:
   ```ini
   sqlalchemy.url = driver://user:pass@localhost/dbname
   # Or use environment variable:
   # sqlalchemy.url = 
   ```

4. Update alembic/env.py:
   ```python
   from src.core.database import Base
   target_metadata = Base.metadata
   ```

5. Create initial migration:
   ```bash
   alembic revision --autogenerate -m "initial schema"
   ```

6. Apply migrations:
   ```bash
   alembic upgrade head
   ```

For production deployments, run migrations as part of deployment:
```bash
alembic upgrade head && uvicorn src.api.app:app
```
"""

if __name__ == "__main__":
    # Print Alembic setup instructions
    print(ALEMBIC_TEMPLATE)
