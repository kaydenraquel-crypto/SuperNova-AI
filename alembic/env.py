import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, create_engine, MetaData
from sqlalchemy import pool
from alembic import context

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import SuperNova models and configuration
    from supernova.db import Base
    from supernova.sentiment_models import TimescaleBase
    from supernova.config import settings
    from supernova.database_config import db_config
except ImportError as e:
    print(f"Warning: Could not import supernova modules: {e}")
    print("Running in standalone mode...")
    Base = None
    TimescaleBase = None
    settings = None
    db_config = None

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Environment configuration
environment = os.getenv('SUPERNOVA_ENV', 'development')
target_database = os.getenv('ALEMBIC_TARGET_DB', 'primary')

# Configure target metadata based on database
if target_database == 'timescale':
    target_metadata = TimescaleBase.metadata if TimescaleBase else None
else:
    target_metadata = Base.metadata if Base else None


def get_database_url():
    """Get database URL based on environment and target database."""
    if db_config:
        urls = db_config.get_database_urls()
        if target_database in urls:
            return urls[target_database]
    
    # Fallback to environment variables or config
    if target_database == 'timescale':
        if settings and hasattr(settings, 'TIMESCALE_HOST'):
            return (
                f"postgresql+psycopg2://"
                f"{settings.TIMESCALE_USER}:{settings.TIMESCALE_PASSWORD}@"
                f"{settings.TIMESCALE_HOST}:{getattr(settings, 'TIMESCALE_PORT', 5432)}/"
                f"{settings.TIMESCALE_DB}"
            )
    else:
        # Primary database
        if environment == 'production':
            return os.getenv('DATABASE_URL') or config.get_main_option("sqlalchemy.url")
        elif settings and hasattr(settings, 'DATABASE_URL'):
            return settings.DATABASE_URL
    
    # Final fallback
    return config.get_main_option("sqlalchemy.url") or "sqlite:///./supernova.db"

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
    # Get database URL
    url = get_database_url()
    
    # Create engine configuration
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url
    
    # Configure engine based on database type
    if target_database == 'timescale':
        # TimescaleDB specific settings
        configuration.update({
            "sqlalchemy.pool_size": "5",
            "sqlalchemy.max_overflow": "10",
            "sqlalchemy.pool_timeout": "30",
            "sqlalchemy.pool_recycle": "3600",
        })
    else:
        # PostgreSQL specific settings for primary database
        if not url.startswith('sqlite'):
            configuration.update({
                "sqlalchemy.pool_size": "10",
                "sqlalchemy.max_overflow": "20", 
                "sqlalchemy.pool_timeout": "30",
                "sqlalchemy.pool_recycle": "3600",
            })

    # Create engine
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Use NullPool for migrations
    )

    with connectable.connect() as connection:
        # Configure migration context
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            render_as_batch=url.startswith('sqlite'),  # Batch mode for SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
