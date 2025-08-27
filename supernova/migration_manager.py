"""
Advanced Migration Management System for SuperNova AI

Provides comprehensive database migration management including:
- Alembic integration
- Environment-specific migrations
- Rollback procedures
- Migration validation and testing
- Multi-database support (PostgreSQL + TimescaleDB)
"""

from __future__ import annotations
import os
import sys
import logging
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import tempfile

from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.runtime.environment import EnvironmentContext
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import settings
    from .database_config import db_config
    from .db import Base
    from .sentiment_models import TimescaleBase
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    raise e

logger = logging.getLogger(__name__)

class MigrationManager:
    """
    Comprehensive migration management system.
    
    Handles database schema migrations with support for:
    - Multiple databases (PostgreSQL, TimescaleDB)
    - Environment-specific migrations
    - Rollback procedures
    - Migration validation and testing
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.alembic_cfg_path = self.project_root / "alembic.ini"
        self.alembic_dir = self.project_root / "alembic"
        
        if not self.alembic_cfg_path.exists():
            raise FileNotFoundError(f"Alembic configuration not found: {self.alembic_cfg_path}")
        
        self.environment = getattr(settings, 'SUPERNOVA_ENV', 'development')
        self._alembic_configs: Dict[str, Config] = {}
        
        # Migration tracking
        self.migration_history: List[Dict[str, Any]] = []
        
    def get_alembic_config(self, database: str = 'primary') -> Config:
        """Get Alembic configuration for specified database."""
        if database in self._alembic_configs:
            return self._alembic_configs[database]
        
        # Create Alembic configuration
        config = Config(str(self.alembic_cfg_path))
        
        # Set database-specific configuration
        if database == 'timescale':
            os.environ['ALEMBIC_TARGET_DB'] = 'timescale'
        else:
            os.environ['ALEMBIC_TARGET_DB'] = 'primary'
        
        # Set environment variables for migration
        os.environ['SUPERNOVA_ENV'] = self.environment
        
        # Get database URL
        urls = db_config.get_database_urls()
        if database in urls:
            config.set_main_option("sqlalchemy.url", urls[database])
        else:
            raise ValueError(f"Database '{database}' not configured")
        
        # Cache configuration
        self._alembic_configs[database] = config
        
        return config
    
    def create_migration(self, message: str, database: str = 'primary', 
                        autogenerate: bool = True) -> Optional[str]:
        """
        Create a new migration.
        
        Args:
            message: Migration description
            database: Target database ('primary' or 'timescale')
            autogenerate: Whether to auto-generate migration content
            
        Returns:
            Migration revision ID if successful, None otherwise
        """
        try:
            config = self.get_alembic_config(database)
            
            # Create migration
            if autogenerate:
                revision = command.revision(
                    config, 
                    message=message, 
                    autogenerate=True
                )
            else:
                revision = command.revision(
                    config, 
                    message=message
                )
            
            revision_id = revision.revision if revision else None
            
            if revision_id:
                logger.info(f"Created migration {revision_id} for {database}: {message}")
                
                # Log migration creation
                self._log_migration_event({
                    'action': 'create',
                    'database': database,
                    'revision': revision_id,
                    'message': message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'environment': self.environment
                })
            
            return revision_id
            
        except Exception as e:
            logger.error(f"Failed to create migration for {database}: {e}")
            return None
    
    def upgrade(self, database: str = 'primary', revision: str = 'head') -> bool:
        """
        Upgrade database to specified revision.
        
        Args:
            database: Target database
            revision: Target revision (default: 'head')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.get_alembic_config(database)
            
            # Get current revision before upgrade
            current_rev = self.get_current_revision(database)
            
            # Perform upgrade
            command.upgrade(config, revision)
            
            # Get new revision after upgrade
            new_rev = self.get_current_revision(database)
            
            logger.info(f"Upgraded {database} from {current_rev} to {new_rev}")
            
            # Log migration event
            self._log_migration_event({
                'action': 'upgrade',
                'database': database,
                'from_revision': current_rev,
                'to_revision': new_rev,
                'target_revision': revision,
                'timestamp': datetime.utcnow().isoformat(),
                'environment': self.environment
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upgrade {database}: {e}")
            return False
    
    def downgrade(self, database: str = 'primary', revision: str = '-1') -> bool:
        """
        Downgrade database to specified revision.
        
        Args:
            database: Target database
            revision: Target revision (default: '-1' for previous)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.get_alembic_config(database)
            
            # Get current revision before downgrade
            current_rev = self.get_current_revision(database)
            
            # Perform downgrade
            command.downgrade(config, revision)
            
            # Get new revision after downgrade
            new_rev = self.get_current_revision(database)
            
            logger.info(f"Downgraded {database} from {current_rev} to {new_rev}")
            
            # Log migration event
            self._log_migration_event({
                'action': 'downgrade',
                'database': database,
                'from_revision': current_rev,
                'to_revision': new_rev,
                'target_revision': revision,
                'timestamp': datetime.utcnow().isoformat(),
                'environment': self.environment
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade {database}: {e}")
            return False
    
    def get_current_revision(self, database: str = 'primary') -> Optional[str]:
        """Get current database revision."""
        try:
            config = self.get_alembic_config(database)
            
            # Get database URL
            urls = db_config.get_database_urls()
            engine = create_engine(urls[database])
            
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
                
        except Exception as e:
            logger.error(f"Failed to get current revision for {database}: {e}")
            return None
    
    def get_migration_history(self, database: str = 'primary') -> List[Dict[str, Any]]:
        """Get migration history for database."""
        try:
            config = self.get_alembic_config(database)
            script = ScriptDirectory.from_config(config)
            
            history = []
            for revision in script.walk_revisions():
                history.append({
                    'revision': revision.revision,
                    'down_revision': revision.down_revision,
                    'branch_labels': revision.branch_labels,
                    'depends_on': revision.depends_on,
                    'message': revision.doc,
                    'is_current': revision.revision == self.get_current_revision(database)
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get migration history for {database}: {e}")
            return []
    
    def get_pending_migrations(self, database: str = 'primary') -> List[str]:
        """Get pending migrations for database."""
        try:
            config = self.get_alembic_config(database)
            script = ScriptDirectory.from_config(config)
            
            # Get current revision
            current_rev = self.get_current_revision(database)
            
            # Get all revisions from current to head
            pending = []
            if current_rev:
                for revision in script.iterate_revisions('head', current_rev):
                    if revision.revision != current_rev:
                        pending.append(revision.revision)
            else:
                # No current revision, all migrations are pending
                for revision in script.walk_revisions():
                    pending.append(revision.revision)
            
            return pending
            
        except Exception as e:
            logger.error(f"Failed to get pending migrations for {database}: {e}")
            return []
    
    def validate_migration(self, database: str = 'primary', 
                          revision: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate migration without applying it.
        
        Args:
            database: Target database
            revision: Specific revision to validate (default: head)
            
        Returns:
            Tuple of (success, list_of_issues)
        """
        issues = []
        
        try:
            config = self.get_alembic_config(database)
            
            # Create temporary test database
            test_db_url = self._create_test_database(database)
            if not test_db_url:
                return False, ["Failed to create test database"]
            
            try:
                # Apply migration to test database
                test_config = Config(str(self.alembic_cfg_path))
                test_config.set_main_option("sqlalchemy.url", test_db_url)
                
                # Run migration
                command.upgrade(test_config, revision or 'head')
                
                # Validate schema
                validation_issues = self._validate_schema(test_db_url, database)
                issues.extend(validation_issues)
                
                # Test rollback capability
                if revision:
                    rollback_issues = self._test_rollback(test_config, revision)
                    issues.extend(rollback_issues)
                
            finally:
                # Cleanup test database
                self._cleanup_test_database(test_db_url)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Migration validation failed for {database}: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def migrate_all_databases(self, revision: str = 'head') -> Dict[str, bool]:
        """
        Migrate all configured databases to specified revision.
        
        Args:
            revision: Target revision for all databases
            
        Returns:
            Dictionary mapping database names to success status
        """
        results = {}
        
        # Get all configured databases
        urls = db_config.get_database_urls()
        
        for database in urls.keys():
            if database in ['test']:  # Skip test database
                continue
            
            logger.info(f"Migrating {database} to {revision}")
            results[database] = self.upgrade(database, revision)
        
        # Log overall migration status
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        self._log_migration_event({
            'action': 'migrate_all',
            'results': results,
            'success_count': success_count,
            'total_count': total_count,
            'target_revision': revision,
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment
        })
        
        return results
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status for all databases."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'databases': {}
        }
        
        # Get status for each database
        urls = db_config.get_database_urls()
        
        for database in urls.keys():
            if database == 'test':
                continue
            
            try:
                current_rev = self.get_current_revision(database)
                pending = self.get_pending_migrations(database)
                history = self.get_migration_history(database)
                
                status['databases'][database] = {
                    'current_revision': current_rev,
                    'pending_migrations': pending,
                    'pending_count': len(pending),
                    'total_migrations': len(history),
                    'is_up_to_date': len(pending) == 0,
                    'history': history[:5]  # Last 5 migrations
                }
                
            except Exception as e:
                status['databases'][database] = {
                    'error': str(e),
                    'current_revision': None,
                    'pending_migrations': [],
                    'pending_count': 0,
                    'is_up_to_date': False
                }
        
        return status
    
    def create_initial_migration(self) -> Dict[str, Optional[str]]:
        """Create initial migrations for all databases."""
        results = {}
        
        # Create migration for primary database
        primary_rev = self.create_migration(
            "Initial schema setup",
            database='primary',
            autogenerate=True
        )
        results['primary'] = primary_rev
        
        # Create migration for TimescaleDB if configured
        if 'timescale' in db_config.get_database_urls():
            timescale_rev = self.create_migration(
                "Initial TimescaleDB schema setup",
                database='timescale',
                autogenerate=True
            )
            results['timescale'] = timescale_rev
        
        return results
    
    def backup_before_migration(self, database: str = 'primary') -> Optional[str]:
        """Create database backup before migration."""
        try:
            urls = db_config.get_database_urls()
            url = urls[database]
            
            if url.startswith('sqlite'):
                # For SQLite, simply copy the file
                import shutil
                db_path = url.replace('sqlite:///', '')
                backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(db_path, backup_path)
                logger.info(f"Created SQLite backup: {backup_path}")
                return backup_path
                
            elif url.startswith('postgresql'):
                # For PostgreSQL, use pg_dump
                backup_path = f"/tmp/{database}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                
                # Extract connection parameters
                from urllib.parse import urlparse
                parsed = urlparse(url)
                
                cmd = [
                    'pg_dump',
                    f'--host={parsed.hostname}',
                    f'--port={parsed.port or 5432}',
                    f'--username={parsed.username}',
                    f'--dbname={parsed.path[1:]}',
                    f'--file={backup_path}',
                    '--verbose',
                    '--no-password'
                ]
                
                # Set password via environment
                env = os.environ.copy()
                env['PGPASSWORD'] = parsed.password or ''
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Created PostgreSQL backup: {backup_path}")
                    return backup_path
                else:
                    logger.error(f"Backup failed: {result.stderr}")
                    return None
            
        except Exception as e:
            logger.error(f"Failed to create backup for {database}: {e}")
            return None
    
    def _create_test_database(self, database: str) -> Optional[str]:
        """Create temporary test database for validation."""
        try:
            # Create temporary database URL
            if database == 'primary':
                return f"sqlite:///{tempfile.mktemp(suffix='.db')}"
            elif database == 'timescale':
                # For TimescaleDB, would need actual PostgreSQL instance
                # For now, use SQLite for testing
                return f"sqlite:///{tempfile.mktemp(suffix='.db')}"
            
        except Exception as e:
            logger.error(f"Failed to create test database: {e}")
            return None
    
    def _cleanup_test_database(self, test_url: str) -> None:
        """Clean up test database."""
        try:
            if test_url.startswith('sqlite:///'):
                db_path = test_url.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    os.remove(db_path)
        except Exception as e:
            logger.error(f"Failed to cleanup test database: {e}")
    
    def _validate_schema(self, db_url: str, database: str) -> List[str]:
        """Validate schema after migration."""
        issues = []
        
        try:
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Get table names
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
                
                # Basic validation
                if database == 'primary':
                    required_tables = ['users', 'profiles', 'assets']
                    missing_tables = [t for t in required_tables if t not in tables]
                    if missing_tables:
                        issues.append(f"Missing required tables: {missing_tables}")
                
                elif database == 'timescale':
                    required_tables = ['sentiment_data', 'sentiment_aggregates']
                    missing_tables = [t for t in required_tables if t not in tables]
                    if missing_tables:
                        issues.append(f"Missing required tables: {missing_tables}")
            
        except Exception as e:
            issues.append(f"Schema validation error: {str(e)}")
        
        return issues
    
    def _test_rollback(self, config: Config, revision: str) -> List[str]:
        """Test rollback capability."""
        issues = []
        
        try:
            # Try to rollback
            command.downgrade(config, '-1')
            
            # Try to upgrade again
            command.upgrade(config, revision)
            
        except Exception as e:
            issues.append(f"Rollback test failed: {str(e)}")
        
        return issues
    
    def _log_migration_event(self, event: Dict[str, Any]) -> None:
        """Log migration event."""
        self.migration_history.append(event)
        
        # Also log to file if configured
        log_file = getattr(settings, 'MIGRATION_LOG_FILE', None)
        if log_file:
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
            except Exception as e:
                logger.error(f"Failed to write to migration log: {e}")

# Global migration manager instance
migration_manager = MigrationManager()

# Convenience functions
def create_migration(message: str, database: str = 'primary') -> Optional[str]:
    """Create a new migration."""
    return migration_manager.create_migration(message, database)

def upgrade_database(database: str = 'primary', revision: str = 'head') -> bool:
    """Upgrade database to specified revision."""
    return migration_manager.upgrade(database, revision)

def downgrade_database(database: str = 'primary', revision: str = '-1') -> bool:
    """Downgrade database to specified revision."""
    return migration_manager.downgrade(database, revision)

def get_migration_status() -> Dict[str, Any]:
    """Get migration status for all databases."""
    return migration_manager.get_migration_status()

def migrate_all(revision: str = 'head') -> Dict[str, bool]:
    """Migrate all databases."""
    return migration_manager.migrate_all_databases(revision)