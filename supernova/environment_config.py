"""
Environment-Specific Database Configuration

Manages database configurations for different environments:
- Development, staging, production environment setup
- Environment-specific connection parameters
- Configuration validation and security
- Database seeding for development environments
- Environment synchronization procedures
"""

from __future__ import annotations
import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import subprocess

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import settings
    from .database_config import DatabaseConfig
    from .migration_manager import migration_manager
    from .db import Base
    from .sentiment_models import TimescaleBase
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseEnvironmentConfig:
    """Database configuration for a specific environment."""
    environment: Environment
    database_name: str
    
    # Connection parameters
    host: str
    port: int
    database: str
    username: str
    password: str
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Security settings
    ssl_mode: str = "prefer"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_root_cert: Optional[str] = None
    
    # Performance settings
    connection_timeout: int = 30
    statement_timeout: int = 30000  # milliseconds
    
    # Feature flags
    enable_logging: bool = False
    enable_metrics: bool = True
    enable_backup: bool = True
    
    # Additional parameters
    additional_params: Dict[str, Any] = None

@dataclass
class EnvironmentSyncConfig:
    """Configuration for environment synchronization."""
    source_env: Environment
    target_env: Environment
    sync_schema: bool = True
    sync_data: bool = False
    exclude_tables: List[str] = None
    include_tables: List[str] = None
    anonymize_data: bool = True

class EnvironmentManager:
    """Manage database configurations across different environments."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path(__file__).parent.parent / "database_environments.yaml"
        self.configurations: Dict[str, Dict[str, DatabaseEnvironmentConfig]] = {}
        self.current_environment = Environment(getattr(settings, 'SUPERNOVA_ENV', 'development'))
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load environment configurations from file or create defaults."""
        if self.config_file.exists() and YAML_AVAILABLE:
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self._parse_config_data(config_data)
                logger.info(f"Loaded environment configurations from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Error loading configuration file: {e}")
                self._create_default_configurations()
        else:
            self._create_default_configurations()
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> None:
        """Parse configuration data from YAML."""
        for env_name, env_config in config_data.items():
            try:
                env = Environment(env_name)
                self.configurations[env_name] = {}
                
                for db_name, db_config in env_config.items():
                    self.configurations[env_name][db_name] = DatabaseEnvironmentConfig(
                        environment=env,
                        database_name=db_name,
                        **db_config
                    )
                    
            except ValueError:
                logger.warning(f"Unknown environment: {env_name}")
    
    def _create_default_configurations(self) -> None:
        """Create default configurations for all environments."""
        logger.info("Creating default environment configurations")
        
        # Development configuration
        dev_configs = {
            'primary': DatabaseEnvironmentConfig(
                environment=Environment.DEVELOPMENT,
                database_name='primary',
                host='localhost',
                port=5432,
                database='supernova_dev',
                username='supernova_dev',
                password='dev_password',
                pool_size=5,
                enable_logging=True,
                enable_backup=False
            ),
            'timescale': DatabaseEnvironmentConfig(
                environment=Environment.DEVELOPMENT,
                database_name='timescale',
                host='localhost',
                port=5432,
                database='supernova_timescale_dev',
                username='supernova_dev',
                password='dev_password',
                pool_size=3,
                enable_logging=True,
                enable_backup=False
            )
        }
        
        # Testing configuration
        test_configs = {
            'primary': DatabaseEnvironmentConfig(
                environment=Environment.TESTING,
                database_name='primary',
                host='localhost',
                port=5432,
                database='supernova_test',
                username='supernova_test',
                password='test_password',
                pool_size=2,
                enable_logging=False,
                enable_backup=False,
                enable_metrics=False
            ),
            'timescale': DatabaseEnvironmentConfig(
                environment=Environment.TESTING,
                database_name='timescale',
                host='localhost',
                port=5432,
                database='supernova_timescale_test',
                username='supernova_test',
                password='test_password',
                pool_size=2,
                enable_logging=False,
                enable_backup=False,
                enable_metrics=False
            )
        }
        
        # Staging configuration
        staging_configs = {
            'primary': DatabaseEnvironmentConfig(
                environment=Environment.STAGING,
                database_name='primary',
                host=os.getenv('STAGING_DB_HOST', 'staging-db.internal'),
                port=int(os.getenv('STAGING_DB_PORT', '5432')),
                database='supernova_staging',
                username='supernova_staging',
                password=os.getenv('STAGING_DB_PASSWORD', 'staging_password'),
                pool_size=10,
                ssl_mode='require',
                enable_logging=True,
                enable_backup=True
            ),
            'timescale': DatabaseEnvironmentConfig(
                environment=Environment.STAGING,
                database_name='timescale',
                host=os.getenv('STAGING_TIMESCALE_HOST', 'staging-timescale.internal'),
                port=int(os.getenv('STAGING_TIMESCALE_PORT', '5432')),
                database='supernova_timescale_staging',
                username='supernova_staging',
                password=os.getenv('STAGING_TIMESCALE_PASSWORD', 'staging_password'),
                pool_size=5,
                ssl_mode='require',
                enable_logging=True,
                enable_backup=True
            )
        }
        
        # Production configuration
        production_configs = {
            'primary': DatabaseEnvironmentConfig(
                environment=Environment.PRODUCTION,
                database_name='primary',
                host=os.getenv('PROD_DB_HOST', 'prod-db.internal'),
                port=int(os.getenv('PROD_DB_PORT', '5432')),
                database='supernova_production',
                username='supernova_prod',
                password=os.getenv('PROD_DB_PASSWORD', ''),
                pool_size=20,
                max_overflow=30,
                ssl_mode='require',
                ssl_cert=os.getenv('PROD_DB_SSL_CERT'),
                ssl_key=os.getenv('PROD_DB_SSL_KEY'),
                ssl_root_cert=os.getenv('PROD_DB_SSL_ROOT_CERT'),
                enable_logging=False,
                enable_backup=True,
                enable_metrics=True
            ),
            'timescale': DatabaseEnvironmentConfig(
                environment=Environment.PRODUCTION,
                database_name='timescale',
                host=os.getenv('PROD_TIMESCALE_HOST', 'prod-timescale.internal'),
                port=int(os.getenv('PROD_TIMESCALE_PORT', '5432')),
                database='supernova_timescale_production',
                username='supernova_prod',
                password=os.getenv('PROD_TIMESCALE_PASSWORD', ''),
                pool_size=15,
                max_overflow=25,
                ssl_mode='require',
                ssl_cert=os.getenv('PROD_TIMESCALE_SSL_CERT'),
                ssl_key=os.getenv('PROD_TIMESCALE_SSL_KEY'),
                ssl_root_cert=os.getenv('PROD_TIMESCALE_SSL_ROOT_CERT'),
                enable_logging=False,
                enable_backup=True,
                enable_metrics=True
            )
        }
        
        self.configurations = {
            'development': dev_configs,
            'testing': test_configs,
            'staging': staging_configs,
            'production': production_configs
        }
        
        # Save default configurations
        self.save_configurations()
    
    def save_configurations(self) -> None:
        """Save current configurations to file."""
        if not YAML_AVAILABLE:
            logger.warning("YAML not available, cannot save configurations")
            return
        
        try:
            config_data = {}
            
            for env_name, env_configs in self.configurations.items():
                config_data[env_name] = {}
                
                for db_name, db_config in env_configs.items():
                    # Convert to dict and remove environment field
                    config_dict = asdict(db_config)
                    config_dict.pop('environment', None)
                    config_dict.pop('database_name', None)
                    
                    config_data[env_name][db_name] = config_dict
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved environment configurations to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configurations: {e}")
    
    def get_config(self, environment: Environment, database: str) -> Optional[DatabaseEnvironmentConfig]:
        """Get configuration for specific environment and database."""
        env_configs = self.configurations.get(environment.value)
        if env_configs:
            return env_configs.get(database)
        return None
    
    def get_current_config(self, database: str) -> Optional[DatabaseEnvironmentConfig]:
        """Get configuration for current environment."""
        return self.get_config(self.current_environment, database)
    
    def build_connection_url(self, config: DatabaseEnvironmentConfig) -> str:
        """Build database connection URL from configuration."""
        # Base URL
        url = f"postgresql+psycopg2://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        
        # Add SSL parameters
        params = []
        params.append(f"sslmode={config.ssl_mode}")
        
        if config.ssl_cert:
            params.append(f"sslcert={config.ssl_cert}")
        if config.ssl_key:
            params.append(f"sslkey={config.ssl_key}")
        if config.ssl_root_cert:
            params.append(f"sslrootcert={config.ssl_root_cert}")
        
        # Add connection timeout
        params.append(f"connect_timeout={config.connection_timeout}")
        
        # Add additional parameters
        if config.additional_params:
            for key, value in config.additional_params.items():
                params.append(f"{key}={value}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    def validate_configuration(self, config: DatabaseEnvironmentConfig) -> Tuple[bool, List[str]]:
        """Validate database configuration."""
        issues = []
        
        # Check required fields
        if not config.host:
            issues.append("Host is required")
        if not config.database:
            issues.append("Database name is required")
        if not config.username:
            issues.append("Username is required")
        
        # Check port range
        if not (1 <= config.port <= 65535):
            issues.append("Port must be between 1 and 65535")
        
        # Check pool settings
        if config.pool_size <= 0:
            issues.append("Pool size must be positive")
        if config.max_overflow < 0:
            issues.append("Max overflow cannot be negative")
        
        # Check SSL configuration
        if config.ssl_mode not in ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full']:
            issues.append("Invalid SSL mode")
        
        # Check timeout values
        if config.connection_timeout <= 0:
            issues.append("Connection timeout must be positive")
        if config.pool_timeout <= 0:
            issues.append("Pool timeout must be positive")
        
        # Environment-specific checks
        if config.environment == Environment.PRODUCTION:
            if not config.password:
                issues.append("Password is required for production")
            if config.ssl_mode in ['disable', 'allow']:
                issues.append("SSL should be required for production")
            if config.enable_logging:
                issues.append("Logging should be disabled in production for performance")
        
        return len(issues) == 0, issues
    
    def test_connection(self, config: DatabaseEnvironmentConfig) -> Tuple[bool, Optional[str]]:
        """Test database connection with given configuration."""
        try:
            url = self.build_connection_url(config)
            engine = create_engine(url, pool_pre_ping=True)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
            
            engine.dispose()
            logger.info(f"Connection test successful for {config.database_name} in {config.environment.value}")
            return True, None
            
        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def test_all_connections(self, environment: Environment = None) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Test all database connections for an environment."""
        env = environment or self.current_environment
        results = {}
        
        env_configs = self.configurations.get(env.value, {})
        
        for db_name, config in env_configs.items():
            results[db_name] = self.test_connection(config)
        
        return results
    
    def create_database(self, config: DatabaseEnvironmentConfig) -> bool:
        """Create database if it doesn't exist."""
        try:
            # Connect to default database (postgres)
            admin_config = DatabaseEnvironmentConfig(
                environment=config.environment,
                database_name='admin',
                host=config.host,
                port=config.port,
                database='postgres',  # Default database
                username=config.username,
                password=config.password,
                ssl_mode=config.ssl_mode,
                ssl_cert=config.ssl_cert,
                ssl_key=config.ssl_key,
                ssl_root_cert=config.ssl_root_cert
            )
            
            admin_url = self.build_connection_url(admin_config)
            engine = create_engine(admin_url, isolation_level='AUTOCOMMIT')
            
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text(
                    f"SELECT 1 FROM pg_database WHERE datname = '{config.database}'"
                ))
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text(f"CREATE DATABASE {config.database}"))
                    logger.info(f"Created database: {config.database}")
                else:
                    logger.info(f"Database already exists: {config.database}")
            
            engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database {config.database}: {e}")
            return False
    
    def setup_environment(self, environment: Environment, create_databases: bool = True) -> Dict[str, bool]:
        """Setup all databases for an environment."""
        results = {}
        
        env_configs = self.configurations.get(environment.value, {})
        
        for db_name, config in env_configs.items():
            try:
                success = True
                
                # Validate configuration
                valid, issues = self.validate_configuration(config)
                if not valid:
                    logger.error(f"Configuration validation failed for {db_name}: {issues}")
                    success = False
                
                # Create database if requested
                if success and create_databases:
                    success = self.create_database(config)
                
                # Test connection
                if success:
                    success, error = self.test_connection(config)
                
                results[db_name] = success
                
            except Exception as e:
                logger.error(f"Error setting up {db_name}: {e}")
                results[db_name] = False
        
        return results
    
    def migrate_environment(self, environment: Environment) -> Dict[str, bool]:
        """Run migrations for all databases in an environment."""
        results = {}
        
        # Set environment for migration
        original_env = os.environ.get('SUPERNOVA_ENV')
        os.environ['SUPERNOVA_ENV'] = environment.value
        
        try:
            env_configs = self.configurations.get(environment.value, {})
            
            for db_name in env_configs.keys():
                try:
                    success = migration_manager.upgrade(db_name, 'head')
                    results[db_name] = success
                    
                    if success:
                        logger.info(f"Migrations completed for {db_name} in {environment.value}")
                    else:
                        logger.error(f"Migrations failed for {db_name} in {environment.value}")
                        
                except Exception as e:
                    logger.error(f"Error running migrations for {db_name}: {e}")
                    results[db_name] = False
            
        finally:
            # Restore original environment
            if original_env:
                os.environ['SUPERNOVA_ENV'] = original_env
            else:
                os.environ.pop('SUPERNOVA_ENV', None)
        
        return results
    
    def seed_development_data(self) -> bool:
        """Seed development databases with sample data."""
        if self.current_environment != Environment.DEVELOPMENT:
            logger.warning("Data seeding is only allowed in development environment")
            return False
        
        try:
            # Get development configuration
            primary_config = self.get_current_config('primary')
            if not primary_config:
                logger.error("Primary database configuration not found for development")
                return False
            
            # Create connection
            url = self.build_connection_url(primary_config)
            engine = create_engine(url)
            
            with engine.connect() as conn:
                # Insert sample users
                conn.execute(text("""
                    INSERT INTO users (name, email) VALUES 
                    ('John Doe', 'john@example.com'),
                    ('Jane Smith', 'jane@example.com')
                    ON CONFLICT (email) DO NOTHING
                """))
                
                # Insert sample assets
                conn.execute(text("""
                    INSERT INTO assets (symbol, asset_class, description) VALUES 
                    ('AAPL', 'stock', 'Apple Inc.'),
                    ('GOOGL', 'stock', 'Alphabet Inc.'),
                    ('BTC', 'crypto', 'Bitcoin')
                    ON CONFLICT (symbol) DO NOTHING
                """))
                
                conn.commit()
            
            engine.dispose()
            logger.info("Development data seeding completed")
            return True
            
        except Exception as e:
            logger.error(f"Error seeding development data: {e}")
            return False
    
    def sync_environments(self, sync_config: EnvironmentSyncConfig) -> bool:
        """Synchronize data/schema between environments."""
        try:
            source_configs = self.configurations.get(sync_config.source_env.value)
            target_configs = self.configurations.get(sync_config.target_env.value)
            
            if not source_configs or not target_configs:
                logger.error("Source or target environment configuration not found")
                return False
            
            success = True
            
            for db_name in source_configs.keys():
                if db_name not in target_configs:
                    continue
                
                logger.info(f"Syncing {db_name} from {sync_config.source_env.value} to {sync_config.target_env.value}")
                
                if sync_config.sync_schema:
                    # Use pg_dump for schema sync
                    success &= self._sync_schema(
                        source_configs[db_name],
                        target_configs[db_name],
                        sync_config
                    )
                
                if sync_config.sync_data and success:
                    # Sync data with filtering
                    success &= self._sync_data(
                        source_configs[db_name],
                        target_configs[db_name],
                        sync_config
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Error synchronizing environments: {e}")
            return False
    
    def _sync_schema(
        self,
        source_config: DatabaseEnvironmentConfig,
        target_config: DatabaseEnvironmentConfig,
        sync_config: EnvironmentSyncConfig
    ) -> bool:
        """Synchronize schema between databases."""
        try:
            # Use pg_dump to export schema
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.sql', delete=False) as temp_file:
                dump_cmd = [
                    'pg_dump',
                    f'--host={source_config.host}',
                    f'--port={source_config.port}',
                    f'--username={source_config.username}',
                    f'--dbname={source_config.database}',
                    '--schema-only',
                    '--no-owner',
                    '--no-privileges',
                    f'--file={temp_file.name}'
                ]
                
                # Set password via environment
                env = os.environ.copy()
                env['PGPASSWORD'] = source_config.password
                
                result = subprocess.run(dump_cmd, env=env, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Schema dump failed: {result.stderr}")
                    return False
                
                # Import schema to target
                import_cmd = [
                    'psql',
                    f'--host={target_config.host}',
                    f'--port={target_config.port}',
                    f'--username={target_config.username}',
                    f'--dbname={target_config.database}',
                    f'--file={temp_file.name}'
                ]
                
                env['PGPASSWORD'] = target_config.password
                
                result = subprocess.run(import_cmd, env=env, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Schema import failed: {result.stderr}")
                    return False
                
                # Cleanup
                os.unlink(temp_file.name)
                
                return True
                
        except Exception as e:
            logger.error(f"Error syncing schema: {e}")
            return False
    
    def _sync_data(
        self,
        source_config: DatabaseEnvironmentConfig,
        target_config: DatabaseEnvironmentConfig,
        sync_config: EnvironmentSyncConfig
    ) -> bool:
        """Synchronize data between databases."""
        try:
            # This is a simplified implementation
            # In production, you'd want more sophisticated data sync
            logger.info("Data synchronization not fully implemented")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing data: {e}")
            return False
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get status of all environments."""
        status = {
            'current_environment': self.current_environment.value,
            'environments': {}
        }
        
        for env_name, env_configs in self.configurations.items():
            env_status = {
                'databases': {},
                'total_databases': len(env_configs),
                'healthy_databases': 0
            }
            
            for db_name, config in env_configs.items():
                # Test connection
                healthy, error = self.test_connection(config)
                
                env_status['databases'][db_name] = {
                    'healthy': healthy,
                    'error': error,
                    'host': config.host,
                    'port': config.port,
                    'database': config.database
                }
                
                if healthy:
                    env_status['healthy_databases'] += 1
            
            status['environments'][env_name] = env_status
        
        return status

# Global environment manager
environment_manager = EnvironmentManager()

# Convenience functions
def get_current_environment() -> Environment:
    """Get current environment."""
    return environment_manager.current_environment

def get_database_config(database: str, environment: Environment = None) -> Optional[DatabaseEnvironmentConfig]:
    """Get database configuration for environment."""
    env = environment or environment_manager.current_environment
    return environment_manager.get_config(env, database)

def test_database_connections(environment: Environment = None) -> Dict[str, Tuple[bool, Optional[str]]]:
    """Test all database connections."""
    return environment_manager.test_all_connections(environment)

def setup_environment(environment: Environment) -> Dict[str, bool]:
    """Setup databases for environment."""
    return environment_manager.setup_environment(environment)

def migrate_environment(environment: Environment) -> Dict[str, bool]:
    """Run migrations for environment."""
    return environment_manager.migrate_environment(environment)

def get_environment_status() -> Dict[str, Any]:
    """Get status of all environments."""
    return environment_manager.get_environment_status()