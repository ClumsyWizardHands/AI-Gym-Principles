"""Alembic environment configuration."""

from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.declarative import declarative_base

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# For now, we'll create a temporary Base to get migrations working
# In production, this should import from your actual models
Base = declarative_base()

# Import table definitions directly to avoid circular imports
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, JSON, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime

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


class Action(Base):
    """SQLAlchemy model for agent actions."""
    
    __tablename__ = "actions"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey("agent_profiles.agent_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Relational data as JSON
    relational_data = Column(JSON, nullable=False)
    
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
        Index('idx_timestamp_partition', 'timestamp'),
        CheckConstraint('outcome_valence >= -1 AND outcome_valence <= 1', 
                       name='check_outcome_range'),
        CheckConstraint('decision_entropy >= 0 AND decision_entropy <= 1', 
                       name='check_entropy_range'),
        CheckConstraint('latency_ms >= 0', name='check_positive_latency'),
    )


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

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


def get_database_url() -> str:
    """Get the database URL from environment or default."""
    # Get from environment variable or use default
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///./data/principles.db')
    
    # For synchronous operations (like migrations), we need sync drivers
    # Make sure we're using the sync version of the driver
    if database_url.startswith("sqlite+aiosqlite://"):
        database_url = database_url.replace("sqlite+aiosqlite://", "sqlite:///")
    elif database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    elif database_url.startswith("mysql+aiomysql://"):
        database_url = database_url.replace("mysql+aiomysql://", "mysql://")
    
    return database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
