#!/usr/bin/env python3
"""
SuperNova AI Database Migration Manager
Handles database schema migrations and version control
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the parent directory to the path to import supernova modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from supernova.config import settings
    from supernova.db import engine, SessionLocal
    from sqlalchemy import create_engine, text
except ImportError as e:
    print(f"Warning: Could not import supernova modules: {e}")
    print("Running in standalone mode...")
    settings = None

class MigrationManager:
    """Database migration management"""
    
    def __init__(self, database_url: str = None):
        self.migrations_dir = Path(__file__).parent
        
        # Use provided database URL or fallback to settings or default
        if database_url:
            self.database_url = database_url
        elif settings and hasattr(settings, 'DATABASE_URL'):
            self.database_url = settings.DATABASE_URL
        else:
            self.database_url = "sqlite:///./data/supernova.db"
        
        # Create data directory if it doesn't exist
        if "sqlite:///" in self.database_url:
            db_path = self.database_url.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    def get_connection(self):
        """Get database connection"""
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.replace("sqlite:///", "")
            return sqlite3.connect(db_path)
        else:
            # For other databases, use SQLAlchemy
            engine = create_engine(self.database_url)
            return engine.connect()
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations"""
        try:
            with self.get_connection() as conn:
                if self.database_url.startswith("sqlite"):
                    cursor = conn.cursor()
                    # Check if migrations table exists
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='schema_migrations'
                    """)
                    if not cursor.fetchone():
                        return []
                    
                    # Get applied migrations
                    cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
                    return [row[0] for row in cursor.fetchall()]
                else:
                    # For other databases
                    result = conn.execute(text("SELECT version FROM schema_migrations ORDER BY version"))
                    return [row[0] for row in result.fetchall()]
        except Exception as e:
            self.logger.warning(f"Could not read applied migrations: {e}")
            return []
    
    def get_available_migrations(self) -> List[str]:
        """Get list of available migration files"""
        migration_files = []
        for file in self.migrations_dir.glob("*.sql"):
            if file.name != "migrate.py":  # Skip this script
                migration_files.append(file.stem)
        
        return sorted(migration_files)
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations"""
        applied = set(self.get_applied_migrations())
        available = set(self.get_available_migrations())
        pending = available - applied
        return sorted(list(pending))
    
    def apply_migration(self, migration_name: str) -> bool:
        """Apply a single migration"""
        migration_file = self.migrations_dir / f"{migration_name}.sql"
        
        if not migration_file.exists():
            self.logger.error(f"Migration file not found: {migration_file}")
            return False
        
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            with self.get_connection() as conn:
                if self.database_url.startswith("sqlite"):
                    cursor = conn.cursor()
                    # Execute migration SQL
                    cursor.executescript(migration_sql)
                    conn.commit()
                else:
                    # For other databases, execute each statement separately
                    statements = migration_sql.split(';')
                    for statement in statements:
                        statement = statement.strip()
                        if statement:
                            conn.execute(text(statement))
                    conn.commit()
            
            self.logger.info(f"✅ Applied migration: {migration_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply migration {migration_name}: {e}")
            return False
    
    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a migration (if rollback script exists)"""
        rollback_file = self.migrations_dir / f"{migration_name}_rollback.sql"
        
        if not rollback_file.exists():
            self.logger.warning(f"No rollback script found for: {migration_name}")
            return False
        
        try:
            with open(rollback_file, 'r') as f:
                rollback_sql = f.read()
            
            with self.get_connection() as conn:
                if self.database_url.startswith("sqlite"):
                    cursor = conn.cursor()
                    cursor.executescript(rollback_sql)
                    # Remove from migrations table
                    cursor.execute("DELETE FROM schema_migrations WHERE version = ?", (migration_name,))
                    conn.commit()
                else:
                    statements = rollback_sql.split(';')
                    for statement in statements:
                        statement = statement.strip()
                        if statement:
                            conn.execute(text(statement))
                    conn.execute(text("DELETE FROM schema_migrations WHERE version = :version"), 
                               {"version": migration_name})
                    conn.commit()
            
            self.logger.info(f"⏪ Rolled back migration: {migration_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rollback migration {migration_name}: {e}")
            return False
    
    def migrate_up(self) -> bool:
        """Apply all pending migrations"""
        pending = self.get_pending_migrations()
        
        if not pending:
            self.logger.info("No pending migrations")
            return True
        
        self.logger.info(f"Found {len(pending)} pending migrations")
        
        success_count = 0
        for migration in pending:
            if self.apply_migration(migration):
                success_count += 1
            else:
                self.logger.error(f"Migration failed, stopping at: {migration}")
                break
        
        self.logger.info(f"Applied {success_count}/{len(pending)} migrations")
        return success_count == len(pending)
    
    def migrate_down(self, steps: int = 1) -> bool:
        """Rollback specified number of migrations"""
        applied = self.get_applied_migrations()
        
        if not applied:
            self.logger.info("No migrations to rollback")
            return True
        
        # Get the last N migrations to rollback
        to_rollback = applied[-steps:] if steps < len(applied) else applied
        to_rollback.reverse()  # Rollback in reverse order
        
        success_count = 0
        for migration in to_rollback:
            if self.rollback_migration(migration):
                success_count += 1
            else:
                self.logger.error(f"Rollback failed, stopping at: {migration}")
                break
        
        self.logger.info(f"Rolled back {success_count}/{len(to_rollback)} migrations")
        return success_count == len(to_rollback)
    
    def status(self) -> Dict[str, Any]:
        """Get migration status"""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        pending = self.get_pending_migrations()
        
        return {
            "database_url": self.database_url.replace("sqlite:///", "").replace("://", "://***:***@") if "://" in self.database_url else self.database_url,
            "applied_count": len(applied),
            "available_count": len(available),
            "pending_count": len(pending),
            "applied": applied,
            "pending": pending,
            "status": "up_to_date" if not pending else "pending_migrations"
        }
    
    def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_name = f"{timestamp}_{name}"
        migration_file = self.migrations_dir / f"{migration_name}.sql"
        
        template = f"""-- SuperNova AI Database Migration
-- Version: {migration_name}
-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- ============================================================================
-- Migration: {name}
-- ============================================================================

-- Add your SQL statements here

-- Record this migration
INSERT OR IGNORE INTO schema_migrations (version) VALUES ('{migration_name}');
"""
        
        with open(migration_file, 'w') as f:
            f.write(template)
        
        self.logger.info(f"Created migration file: {migration_file}")
        return str(migration_file)
    
    def validate_database(self) -> bool:
        """Validate database structure"""
        try:
            with self.get_connection() as conn:
                if self.database_url.startswith("sqlite"):
                    cursor = conn.cursor()
                    # Check if core tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    required_tables = ['users', 'profiles', 'assets', 'portfolios', 'schema_migrations']
                    missing_tables = [table for table in required_tables if table not in tables]
                    
                    if missing_tables:
                        self.logger.error(f"Missing required tables: {missing_tables}")
                        return False
                    
                    self.logger.info("✅ Database structure validation passed")
                    return True
                else:
                    # For other databases, similar validation
                    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                    tables = [row[0] for row in result.fetchall()]
                    
                    required_tables = ['users', 'profiles', 'assets', 'portfolios', 'schema_migrations']
                    missing_tables = [table for table in required_tables if table not in tables]
                    
                    if missing_tables:
                        self.logger.error(f"Missing required tables: {missing_tables}")
                        return False
                    
                    self.logger.info("✅ Database structure validation passed")
                    return True
                    
        except Exception as e:
            self.logger.error(f"❌ Database validation failed: {e}")
            return False

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI Database Migration Manager')
    parser.add_argument('command', choices=['status', 'up', 'down', 'create', 'validate'], 
                       help='Migration command')
    parser.add_argument('--name', help='Migration name (for create command)')
    parser.add_argument('--description', help='Migration description (for create command)')
    parser.add_argument('--steps', type=int, default=1, help='Number of steps for down command')
    parser.add_argument('--database-url', help='Database URL (overrides settings)')
    
    args = parser.parse_args()
    
    # Initialize migration manager
    manager = MigrationManager(database_url=args.database_url)
    
    if args.command == 'status':
        status = manager.status()
        print("\\nDatabase Migration Status:")
        print("=" * 50)
        print(f"Database: {status['database_url']}")
        print(f"Applied migrations: {status['applied_count']}")
        print(f"Available migrations: {status['available_count']}")
        print(f"Pending migrations: {status['pending_count']}")
        print(f"Status: {status['status']}")
        
        if status['pending']:
            print(f"\\nPending migrations:")
            for migration in status['pending']:
                print(f"  - {migration}")
    
    elif args.command == 'up':
        print("Applying pending migrations...")
        success = manager.migrate_up()
        sys.exit(0 if success else 1)
    
    elif args.command == 'down':
        print(f"Rolling back {args.steps} migration(s)...")
        success = manager.migrate_down(args.steps)
        sys.exit(0 if success else 1)
    
    elif args.command == 'create':
        if not args.name:
            print("Error: --name is required for create command")
            sys.exit(1)
        
        migration_file = manager.create_migration(args.name, args.description or "")
        print(f"Created migration: {migration_file}")
    
    elif args.command == 'validate':
        print("Validating database structure...")
        success = manager.validate_database()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()