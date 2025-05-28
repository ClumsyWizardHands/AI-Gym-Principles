"""Tests for database migrations."""

import pytest
import os
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def alembic_config():
    """Create a test Alembic configuration."""
    # Create a temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_migrations.db")
    
    # Create Alembic config
    config = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_engine(alembic_config):
    """Create a test database engine."""
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    engine = create_engine(db_url)
    yield engine
    engine.dispose()


class TestDatabaseMigrations:
    """Test database migration procedures."""
    
    def test_initial_migration_creates_all_tables(self, alembic_config, test_engine):
        """Test that the initial migration creates all required tables."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        # Check tables exist
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        
        assert "agent_profiles" in tables
        assert "actions" in tables
        assert "principles" in tables
        assert "alembic_version" in tables  # Alembic tracking table
    
    def test_initial_migration_creates_correct_columns(self, alembic_config, test_engine):
        """Test that tables have the correct columns."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        inspector = inspect(test_engine)
        
        # Check agent_profiles columns
        agent_columns = {col['name'] for col in inspector.get_columns('agent_profiles')}
        expected_agent_columns = {
            'agent_id', 'name', 'created_at', 
            'behavioral_entropy', 'total_actions', 'meta_data'
        }
        assert expected_agent_columns.issubset(agent_columns)
        
        # Check actions columns
        action_columns = {col['name'] for col in inspector.get_columns('actions')}
        expected_action_columns = {
            'id', 'agent_id', 'timestamp', 'relational_data',
            'decision_context', 'context_weight', 'action_type',
            'outcome_valence', 'decision_entropy', 'latency_ms', 'context'
        }
        assert expected_action_columns.issubset(action_columns)
        
        # Check principles columns
        principle_columns = {col['name'] for col in inspector.get_columns('principles')}
        expected_principle_columns = {
            'id', 'agent_id', 'name', 'description', 'strength_score',
            'confidence_lower', 'confidence_upper', 'volatility',
            'evidence_count', 'contradictions_count', 'lineage_data',
            'created_at', 'last_updated', 'meta_data'
        }
        assert expected_principle_columns.issubset(principle_columns)
    
    def test_initial_migration_creates_indexes(self, alembic_config, test_engine):
        """Test that proper indexes are created."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        inspector = inspect(test_engine)
        
        # Check indexes on actions table
        action_indexes = inspector.get_indexes('actions')
        index_names = {idx['name'] for idx in action_indexes}
        
        assert 'idx_agent_timestamp' in index_names
        assert 'idx_decision_context' in index_names
        assert 'idx_timestamp_partition' in index_names
        
        # Check indexes on principles table
        principle_indexes = inspector.get_indexes('principles')
        principle_index_names = {idx['name'] for idx in principle_indexes}
        
        assert 'idx_agent_principle' in principle_index_names
        assert 'idx_strength_score' in principle_index_names
    
    def test_migration_rollback(self, alembic_config, test_engine):
        """Test that migrations can be rolled back."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        # Verify tables exist
        inspector = inspect(test_engine)
        assert len(inspector.get_table_names()) > 1
        
        # Rollback
        command.downgrade(alembic_config, "base")
        
        # Verify tables are gone (except alembic_version)
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        assert len(tables) == 1
        assert tables[0] == "alembic_version"
    
    def test_migration_idempotency(self, alembic_config, test_engine):
        """Test that running migrations multiple times is safe."""
        # Run migrations twice
        command.upgrade(alembic_config, "head")
        command.upgrade(alembic_config, "head")  # Should not fail
        
        # Verify state is correct
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        
        assert "agent_profiles" in tables
        assert "actions" in tables
        assert "principles" in tables
    
    def test_foreign_key_constraints(self, alembic_config, test_engine):
        """Test that foreign key constraints are properly set up."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        # Try to insert an action with non-existent agent_id
        # This should fail due to foreign key constraint
        Session = sessionmaker(bind=test_engine)
        session = Session()
        
        try:
            # This should fail
            session.execute(text("""
                INSERT INTO actions (
                    id, agent_id, timestamp, relational_data,
                    decision_context, context_weight, action_type,
                    outcome_valence, decision_entropy, latency_ms, context
                ) VALUES (
                    'test-id', 'non-existent-agent', '2024-01-01 00:00:00', '{}',
                    'test', 1.0, 'test', 0.5, 0.5, 100, '{}'
                )
            """))
            session.commit()
            assert False, "Foreign key constraint should have prevented insertion"
        except Exception:
            # Expected behavior
            session.rollback()
        finally:
            session.close()
    
    def test_check_constraints(self, alembic_config, test_engine):
        """Test that check constraints are properly enforced."""
        # Run migrations
        command.upgrade(alembic_config, "head")
        
        Session = sessionmaker(bind=test_engine)
        session = Session()
        
        # First insert a valid agent
        session.execute(text("""
            INSERT INTO agent_profiles (
                agent_id, name, created_at, behavioral_entropy, 
                total_actions, meta_data
            ) VALUES (
                'test-agent', 'Test Agent', '2024-01-01 00:00:00', 
                0.5, 0, '{}'
            )
        """))
        session.commit()
        
        # Try to insert a principle with invalid strength_score
        try:
            session.execute(text("""
                INSERT INTO principles (
                    id, agent_id, name, description, strength_score,
                    confidence_lower, confidence_upper, volatility,
                    evidence_count, contradictions_count, lineage_data,
                    created_at, last_updated, meta_data
                ) VALUES (
                    'test-principle', 'test-agent', 'Test', 'Test principle', 
                    1.5,  -- Invalid: should be between 0 and 1
                    0.8, 0.9, 0.1, 10, 0, '{}',
                    '2024-01-01 00:00:00', '2024-01-01 00:00:00', '{}'
                )
            """))
            session.commit()
            assert False, "Check constraint should have prevented insertion"
        except Exception:
            # Expected behavior
            session.rollback()
        finally:
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
