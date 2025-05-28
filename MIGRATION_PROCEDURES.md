# Database Migration Procedures

This document outlines the procedures for managing database migrations using Alembic in the AI Principles Gym project.

## Overview

We use Alembic for database schema versioning and migrations. This ensures that:
- Database schema changes are tracked and version-controlled
- Migrations can be applied consistently across environments
- Database upgrades and downgrades are automated and repeatable

## Setup Requirements

1. **Dependencies**: Ensure Alembic is installed
   ```bash
   pip install alembic==1.13.1
   ```

2. **Configuration**: Alembic is configured in:
   - `alembic.ini` - Main configuration file
   - `alembic/env.py` - Environment configuration that integrates with our models

## Migration Procedures

### 1. Creating a New Migration

#### Automatic Migration Generation (Recommended)
```bash
# Generate migration by detecting model changes
alembic revision --autogenerate -m "description of changes"
```

Example:
```bash
alembic revision --autogenerate -m "add user preferences table"
```

#### Manual Migration
```bash
# Create empty migration file for manual editing
alembic revision -m "description of changes"
```

### 2. Reviewing Migrations

Always review generated migrations before applying:

1. Check the generated file in `alembic/versions/`
2. Verify:
   - All intended changes are included
   - No unintended changes are present
   - Downgrade operations are correct
   - Data migrations (if any) are safe

### 3. Applying Migrations

#### Development Environment
```bash
# Apply all pending migrations
alembic upgrade head

# Apply to specific revision
alembic upgrade <revision>

# Show current revision
alembic current

# Show migration history
alembic history
```

#### Production Environment
```bash
# Always backup database first!
pg_dump production_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Apply migrations
alembic upgrade head

# Verify migration
alembic current
```

### 4. Rolling Back Migrations

```bash
# Downgrade by one revision
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision>

# Downgrade to initial state
alembic downgrade base
```

## Migration Testing Process

### 1. Local Testing

Before applying migrations to any shared environment:

```bash
# 1. Create test database
cp data/principles.db data/principles_test.db

# 2. Apply migration to test database
DATABASE_URL=sqlite:///./data/principles_test.db alembic upgrade head

# 3. Run tests
pytest tests/test_database_migrations.py

# 4. Verify schema
sqlite3 data/principles_test.db ".schema"
```

### 2. Migration Test Checklist

- [ ] Migration applies cleanly from current production schema
- [ ] Downgrade works correctly
- [ ] No data loss occurs
- [ ] Indexes are created/dropped appropriately
- [ ] Constraints are properly handled
- [ ] Performance impact is acceptable
- [ ] Application code is compatible with new schema

### 3. Automated Migration Tests

Run the migration test suite:
```bash
python -m pytest tests/test_database_migrations.py -v
```

## Common Scenarios

### Adding a New Column

1. Add column to model in `src/core/database.py`
2. Generate migration:
   ```bash
   alembic revision --autogenerate -m "add column_name to table_name"
   ```
3. Review and adjust if needed (especially for NOT NULL columns)
4. Apply migration:
   ```bash
   alembic upgrade head
   ```

### Adding an Index

1. Add index to model:
   ```python
   __table_args__ = (
       Index('idx_column_name', 'column_name'),
   )
   ```
2. Generate and apply migration

### Renaming a Column

Alembic may not detect renames automatically:

1. Create manual migration:
   ```bash
   alembic revision -m "rename old_column to new_column"
   ```
2. Edit migration file:
   ```python
   def upgrade():
       op.alter_column('table_name', 'old_column', new_column_name='new_column')
   
   def downgrade():
       op.alter_column('table_name', 'new_column', new_column_name='old_column')
   ```

### Data Migrations

For migrations that modify data:

1. Create migration with data operations:
   ```python
   from alembic import op
   import sqlalchemy as sa
   from sqlalchemy.sql import table, column
   
   def upgrade():
       # Create connection
       conn = op.get_bind()
       
       # Define table
       my_table = table('my_table',
           column('id', sa.Integer),
           column('data', sa.String)
       )
       
       # Perform data migration
       conn.execute(
           my_table.update().values(data='new_value')
       )
   ```

## Best Practices

### 1. Migration Naming
- Use descriptive names: `add_user_preferences_table`
- Include ticket numbers if applicable: `TICKET-123_add_audit_fields`

### 2. Migration Size
- Keep migrations focused and atomic
- Separate schema changes from data migrations
- Large data migrations should be done in batches

### 3. Backwards Compatibility
- Ensure application can work with both old and new schema during deployment
- Use feature flags for schema-dependent features
- Plan for zero-downtime deployments

### 4. Testing
- Always test migrations on a copy of production data
- Include both upgrade and downgrade in tests
- Test with realistic data volumes

### 5. Documentation
- Document complex migrations in the migration file
- Update this document with new procedures
- Keep a migration log for production deployments

## Troubleshooting

### Migration Conflicts

If multiple developers create migrations:

1. Merge branches
2. Resolve conflicts:
   ```bash
   alembic merge -m "merge branch migrations"
   ```

### Failed Migrations

If a migration fails:

1. Check error message
2. Fix the issue
3. If partially applied, may need to manually clean up
4. Rerun migration

### Lost Migration Head

If Alembic loses track of current version:

1. Check actual schema
2. Manually update alembic_version table:
   ```sql
   UPDATE alembic_version SET version_num = 'correct_revision_id';
   ```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
- name: Run migrations
  run: |
    alembic upgrade head
    alembic current
```

### Pre-deployment Checks

1. Verify migration files are included in deployment
2. Check database connectivity
3. Ensure proper permissions for schema changes
4. Backup database

### Post-deployment Verification

1. Verify migration applied: `alembic current`
2. Run smoke tests
3. Monitor application logs
4. Check performance metrics

## Migration Commands Reference

```bash
# Initialize Alembic (already done)
alembic init alembic

# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
alembic upgrade +1
alembic upgrade <revision>

# Rollback migrations
alembic downgrade -1
alembic downgrade <revision>
alembic downgrade base

# Information commands
alembic current
alembic history
alembic show <revision>

# Offline migrations (generate SQL)
alembic upgrade head --sql
alembic downgrade -1 --sql

# Merge migrations
alembic merge -m "merge description"

# Stamp database with specific revision (use carefully!)
alembic stamp <revision>
```

## Environment-Specific Notes

### Development
- Use SQLite for rapid development
- Migrations are applied automatically on startup
- Reset database: Delete `data/principles.db` and run migrations

### Testing
- Use in-memory SQLite for unit tests
- Apply migrations in test setup
- Rollback after each test

### Production
- Use PostgreSQL or MySQL
- Apply migrations during deployment
- Always backup before migrations
- Monitor migration duration

## Contact

For questions about migrations or database schema:
- Check existing migrations in `alembic/versions/`
- Review models in `src/core/database.py`
- Consult team lead before major schema changes
