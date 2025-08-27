"""
Production Database Configuration and Connection Management

This module provides comprehensive database configuration for production environments,
including PostgreSQL setup, connection pooling, health monitoring, and multi-database
architecture support (PostgreSQL + TimescaleDB).
"""

from __future__ import annotations
import os
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import urlparse, parse_qs
import time
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
import redis
from redis.exceptions import RedisError

try:
    from .config import settings
except ImportError:
    # Fallback for testing
    class MockSettings:
        SUPERNOVA_ENV = "development"
        DATABASE_URL = "sqlite:///./supernova.db"
    settings = MockSettings()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Production database configuration and connection management."""
    
    def __init__(self):
        self.environment = getattr(settings, 'SUPERNOVA_ENV', 'development')
        self.engines: Dict[str, Engine] = {}
        self.async_engines: Dict[str, AsyncEngine] = {}
        self.session_factories: Dict[str, sessionmaker] = {}
        self.async_session_factories: Dict[str, async_sessionmaker] = {}
        self.redis_client: Optional[redis.Redis] = None
        self._health_status: Dict[str, Any] = {}
        
        # Connection monitoring
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        
    def get_database_urls(self) -> Dict[str, str]:
        """
        Get database URLs for different databases in the system.
        
        Returns:
            Dictionary mapping database names to connection URLs
        """
        urls = {}
        
        # Primary application database
        if self.environment == 'production':
            # Production PostgreSQL
            urls['primary'] = self._build_postgresql_url('primary')
            # Read replica (if configured)
            if hasattr(settings, 'POSTGRES_READ_REPLICA_HOST'):
                urls['replica'] = self._build_postgresql_url('replica')
        elif self.environment == 'staging':
            # Staging PostgreSQL
            urls['primary'] = self._build_postgresql_url('staging')
        else:
            # Development SQLite
            urls['primary'] = getattr(settings, 'DATABASE_URL', 'sqlite:///./supernova.db')
        
        # TimescaleDB for time-series data
        if self._has_timescale_config():
            urls['timescale'] = self._build_timescale_url()
        
        # Test database
        if hasattr(settings, 'TEST_DATABASE_URL'):
            urls['test'] = getattr(settings, 'TEST_DATABASE_URL')
        
        return urls
    
    def _build_postgresql_url(self, db_type: str = 'primary') -> str:
        """Build PostgreSQL connection URL."""
        if db_type == 'primary':
            host = getattr(settings, 'POSTGRES_HOST', 'localhost')
            port = getattr(settings, 'POSTGRES_PORT', 5432)
            db = getattr(settings, 'POSTGRES_DB', 'supernova_prod')
            user = getattr(settings, 'POSTGRES_USER', 'supernova')
            password = getattr(settings, 'POSTGRES_PASSWORD', '')
        elif db_type == 'replica':
            host = getattr(settings, 'POSTGRES_READ_REPLICA_HOST', 'localhost')
            port = getattr(settings, 'POSTGRES_READ_REPLICA_PORT', 5432)
            db = getattr(settings, 'POSTGRES_DB', 'supernova_prod')
            user = getattr(settings, 'POSTGRES_READ_REPLICA_USER', getattr(settings, 'POSTGRES_USER', 'supernova'))
            password = getattr(settings, 'POSTGRES_READ_REPLICA_PASSWORD', getattr(settings, 'POSTGRES_PASSWORD', ''))
        elif db_type == 'staging':
            host = getattr(settings, 'POSTGRES_STAGING_HOST', 'localhost')
            port = getattr(settings, 'POSTGRES_STAGING_PORT', 5432)
            db = getattr(settings, 'POSTGRES_STAGING_DB', 'supernova_staging')
            user = getattr(settings, 'POSTGRES_STAGING_USER', 'supernova')
            password = getattr(settings, 'POSTGRES_STAGING_PASSWORD', '')
        
        # SSL configuration
        ssl_mode = getattr(settings, 'POSTGRES_SSL_MODE', 'prefer')
        ssl_params = f"?sslmode={ssl_mode}"
        
        if hasattr(settings, 'POSTGRES_SSL_CERT'):
            ssl_params += f"&sslcert={getattr(settings, 'POSTGRES_SSL_CERT')}"
        if hasattr(settings, 'POSTGRES_SSL_KEY'):
            ssl_params += f"&sslkey={getattr(settings, 'POSTGRES_SSL_KEY')}"
        if hasattr(settings, 'POSTGRES_SSL_ROOT_CERT'):
            ssl_params += f"&sslrootcert={getattr(settings, 'POSTGRES_SSL_ROOT_CERT')}"
        
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}{ssl_params}"
    
    def _build_timescale_url(self) -> str:
        """Build TimescaleDB connection URL."""
        host = getattr(settings, 'TIMESCALE_HOST', 'localhost')
        port = getattr(settings, 'TIMESCALE_PORT', 5432)
        db = getattr(settings, 'TIMESCALE_DB', 'supernova_timescale')
        user = getattr(settings, 'TIMESCALE_USER', 'supernova')
        password = getattr(settings, 'TIMESCALE_PASSWORD', '')
        
        # SSL configuration
        ssl_mode = getattr(settings, 'TIMESCALE_SSL_MODE', 'prefer')
        ssl_params = f"?sslmode={ssl_mode}"
        
        if hasattr(settings, 'TIMESCALE_SSL_CERT'):
            ssl_params += f"&sslcert={getattr(settings, 'TIMESCALE_SSL_CERT')}"
        if hasattr(settings, 'TIMESCALE_SSL_KEY'):
            ssl_params += f"&sslkey={getattr(settings, 'TIMESCALE_SSL_KEY')}"
        if hasattr(settings, 'TIMESCALE_SSL_ROOT_CERT'):
            ssl_params += f"&sslrootcert={getattr(settings, 'TIMESCALE_SSL_ROOT_CERT')}"
        
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}{ssl_params}"
    
    def _has_timescale_config(self) -> bool:
        """Check if TimescaleDB is configured."""
        required_settings = ['TIMESCALE_HOST', 'TIMESCALE_DB', 'TIMESCALE_USER', 'TIMESCALE_PASSWORD']
        return all(hasattr(settings, setting) and getattr(settings, setting) 
                  for setting in required_settings)
    
    def create_engine_config(self, db_name: str, url: str, is_async: bool = False) -> Dict[str, Any]:
        """Create engine configuration for database."""
        config = {
            'echo': getattr(settings, 'DATABASE_ECHO', False) and self.environment != 'production',
            'echo_pool': getattr(settings, 'DATABASE_ECHO_POOL', False) and self.environment != 'production',
            'future': True,
        }
        
        # Connection pool configuration
        if not url.startswith('sqlite'):
            if db_name == 'timescale':
                # TimescaleDB specific settings
                config.update({
                    'pool_size': getattr(settings, 'TIMESCALE_POOL_SIZE', 10),
                    'max_overflow': getattr(settings, 'TIMESCALE_MAX_OVERFLOW', 20),
                    'pool_timeout': getattr(settings, 'TIMESCALE_POOL_TIMEOUT', 30),
                    'pool_recycle': getattr(settings, 'TIMESCALE_POOL_RECYCLE', 3600),
                    'pool_pre_ping': True,
                })
            else:
                # PostgreSQL specific settings
                config.update({
                    'pool_size': getattr(settings, 'POSTGRES_POOL_SIZE', 20),
                    'max_overflow': getattr(settings, 'POSTGRES_MAX_OVERFLOW', 30),
                    'pool_timeout': getattr(settings, 'POSTGRES_POOL_TIMEOUT', 30),
                    'pool_recycle': getattr(settings, 'POSTGRES_POOL_RECYCLE', 3600),
                    'pool_pre_ping': True,
                })
            
            # Use QueuePool for production, NullPool for testing
            if self.environment == 'test':
                config['poolclass'] = NullPool
            else:
                config['poolclass'] = QueuePool
        
        # Connection arguments
        connect_args = {}
        if url.startswith('postgresql'):
            connect_args.update({
                'connect_timeout': getattr(settings, f'{db_name.upper()}_CONNECTION_TIMEOUT', 30),
                'application_name': f'supernova_{self.environment}_{db_name}',
                'options': f'-c default_transaction_isolation=read_committed'
            })
        
        if connect_args:
            config['connect_args'] = connect_args
        
        return config
    
    def create_engine(self, db_name: str, is_async: bool = False) -> Engine | AsyncEngine:
        """Create database engine with production configuration."""
        urls = self.get_database_urls()
        
        if db_name not in urls:
            raise ValueError(f"Database '{db_name}' not configured")
        
        url = urls[db_name]
        config = self.create_engine_config(db_name, url, is_async)
        
        # Create engine
        if is_async:
            # Convert to async URL
            if url.startswith('postgresql+psycopg2://'):
                url = url.replace('postgresql+psycopg2://', 'postgresql+asyncpg://')
            engine = create_async_engine(url, **config)
        else:
            engine = create_engine(url, **config)
        
        # Add event listeners for monitoring
        self._add_engine_listeners(engine, db_name)
        
        # Store engine
        if is_async:
            self.async_engines[db_name] = engine
        else:
            self.engines[db_name] = engine
        
        logger.info(f"Created {'async' if is_async else 'sync'} engine for database: {db_name}")
        return engine
    
    def _add_engine_listeners(self, engine: Engine | AsyncEngine, db_name: str) -> None:
        """Add event listeners for connection monitoring."""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas if applicable."""
            if hasattr(dbapi_connection, 'execute'):
                # SQLite specific settings
                if 'sqlite' in str(engine.url):
                    dbapi_connection.execute('PRAGMA foreign_keys=ON')
                    dbapi_connection.execute('PRAGMA journal_mode=WAL')
                    dbapi_connection.execute('PRAGMA synchronous=NORMAL')
                    dbapi_connection.execute('PRAGMA temp_store=memory')
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Track connection checkouts."""
            if db_name not in self._connection_stats:
                self._connection_stats[db_name] = {
                    'checkouts': 0,
                    'checkins': 0,
                    'active_connections': 0,
                    'last_checkout': None
                }
            
            self._connection_stats[db_name]['checkouts'] += 1
            self._connection_stats[db_name]['active_connections'] += 1
            self._connection_stats[db_name]['last_checkout'] = datetime.utcnow()
        
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Track connection checkins."""
            if db_name in self._connection_stats:
                self._connection_stats[db_name]['checkins'] += 1
                self._connection_stats[db_name]['active_connections'] -= 1
        
        @event.listens_for(engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            logger.warning(f"Connection invalidated for {db_name}: {exception}")
    
    def create_session_factory(self, db_name: str, is_async: bool = False) -> sessionmaker | async_sessionmaker:
        """Create session factory for database."""
        # Get or create engine
        if is_async:
            if db_name not in self.async_engines:
                self.create_engine(db_name, is_async=True)
            engine = self.async_engines[db_name]
            
            factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            self.async_session_factories[db_name] = factory
        else:
            if db_name not in self.engines:
                self.create_engine(db_name, is_async=False)
            engine = self.engines[db_name]
            
            factory = sessionmaker(
                bind=engine,
                autoflush=False,
                autocommit=False
            )
            
            self.session_factories[db_name] = factory
        
        return factory
    
    @contextmanager
    def get_session(self, db_name: str = 'primary', readonly: bool = False):
        """Get database session with automatic cleanup."""
        # Use replica for readonly if available
        if readonly and 'replica' in self.get_database_urls():
            db_name = 'replica'
        
        if db_name not in self.session_factories:
            self.create_session_factory(db_name)
        
        session = self.session_factories[db_name]()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error for {db_name}: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self, db_name: str = 'primary', readonly: bool = False):
        """Get async database session with automatic cleanup."""
        # Use replica for readonly if available
        if readonly and 'replica' in self.get_database_urls():
            db_name = 'replica'
        
        if db_name not in self.async_session_factories:
            self.create_session_factory(db_name, is_async=True)
        
        session = self.async_session_factories[db_name]()
        
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error for {db_name}: {e}")
            raise
        finally:
            await session.close()
    
    def test_connection(self, db_name: str = 'primary') -> Tuple[bool, Optional[str]]:
        """Test database connection."""
        try:
            with self.get_session(db_name) as session:
                session.execute(text("SELECT 1"))
            
            logger.info(f"Database connection test passed for: {db_name}")
            return True, None
            
        except Exception as e:
            error_msg = f"Database connection test failed for {db_name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def test_async_connection(self, db_name: str = 'primary') -> Tuple[bool, Optional[str]]:
        """Test async database connection."""
        try:
            async with self.get_async_session(db_name) as session:
                await session.execute(text("SELECT 1"))
            
            logger.info(f"Async database connection test passed for: {db_name}")
            return True, None
            
        except Exception as e:
            error_msg = f"Async database connection test failed for {db_name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for caching."""
        if not hasattr(settings, 'REDIS_URL'):
            logger.info("Redis not configured, skipping initialization")
            return None
        
        try:
            redis_url = getattr(settings, 'REDIS_URL')
            
            # Parse Redis URL for connection parameters
            parsed = urlparse(redis_url)
            
            self.redis_client = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                password=parsed.password,
                db=int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0,
                decode_responses=True,
                socket_connect_timeout=getattr(settings, 'REDIS_CONNECT_TIMEOUT', 5),
                socket_timeout=getattr(settings, 'REDIS_SOCKET_TIMEOUT', 5),
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection initialized successfully")
            return self.redis_client
            
        except (RedisError, Exception) as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for all databases."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'databases': {},
            'redis': {'available': False},
            'connection_stats': self._connection_stats.copy()
        }
        
        # Test each database connection
        for db_name in self.get_database_urls().keys():
            success, error = self.test_connection(db_name)
            
            db_status = {
                'healthy': success,
                'error': error,
                'last_check': datetime.utcnow().isoformat()
            }
            
            # Add pool information if available
            if db_name in self.engines:
                engine = self.engines[db_name]
                if hasattr(engine.pool, 'size'):
                    db_status['pool'] = {
                        'size': engine.pool.size(),
                        'checked_out': engine.pool.checkedout(),
                        'overflow': engine.pool.overflow(),
                        'checked_in': engine.pool.checkedin()
                    }
            
            status['databases'][db_name] = db_status
        
        # Test Redis if available
        if self.redis_client:
            try:
                self.redis_client.ping()
                status['redis'] = {
                    'available': True,
                    'info': self.redis_client.info('server')
                }
            except Exception as e:
                status['redis'] = {
                    'available': False,
                    'error': str(e)
                }
        
        return status
    
    def close_all_connections(self) -> None:
        """Close all database connections and cleanup resources."""
        # Close sync engines
        for db_name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed sync engine for: {db_name}")
            except Exception as e:
                logger.error(f"Error closing sync engine {db_name}: {e}")
        
        # Close async engines
        for db_name, engine in self.async_engines.items():
            try:
                asyncio.create_task(engine.aclose())
                logger.info(f"Closed async engine for: {db_name}")
            except Exception as e:
                logger.error(f"Error closing async engine {db_name}: {e}")
        
        # Close Redis
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")
        
        # Clear all references
        self.engines.clear()
        self.async_engines.clear()
        self.session_factories.clear()
        self.async_session_factories.clear()
        self._connection_stats.clear()

# Global database configuration instance
db_config = DatabaseConfig()

# Convenience functions
def get_session(db_name: str = 'primary', readonly: bool = False):
    """Get database session."""
    return db_config.get_session(db_name, readonly)

def get_async_session(db_name: str = 'primary', readonly: bool = False):
    """Get async database session."""
    return db_config.get_async_session(db_name, readonly)

def initialize_databases() -> Dict[str, bool]:
    """Initialize all configured databases."""
    results = {}
    
    for db_name in db_config.get_database_urls().keys():
        try:
            # Create sync and async engines
            db_config.create_engine(db_name, is_async=False)
            db_config.create_engine(db_name, is_async=True)
            
            # Test connections
            success, error = db_config.test_connection(db_name)
            results[db_name] = success
            
            if not success:
                logger.error(f"Failed to initialize database {db_name}: {error}")
            else:
                logger.info(f"Successfully initialized database: {db_name}")
                
        except Exception as e:
            logger.error(f"Error initializing database {db_name}: {e}")
            results[db_name] = False
    
    # Initialize Redis
    redis_client = db_config.initialize_redis()
    results['redis'] = redis_client is not None
    
    return results

def get_database_health() -> Dict[str, Any]:
    """Get database health status."""
    return db_config.get_health_status()

def close_all_connections() -> None:
    """Close all database connections."""
    db_config.close_all_connections()