"""TimescaleDB Setup and Management Utilities

Database initialization, hypertable creation, continuous aggregates,
data retention policies, and comprehensive database management for
SuperNova's sentiment time-series data.
"""

from __future__ import annotations
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

from sqlalchemy import text, create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

# Local imports
try:
    from .sentiment_models import (
        TimescaleBase, SentimentData, SentimentAggregates, SentimentAlerts, SentimentMetadata,
        get_timescale_engine, get_timescale_session, get_timescale_url
    )
    from .config import settings
    from .db import is_timescale_available
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    raise e

# Configure logging
logger = logging.getLogger(__name__)

class TimescaleSetupManager:
    """
    Comprehensive manager for TimescaleDB setup, maintenance, and monitoring.
    
    Handles database initialization, hypertable creation, continuous aggregates,
    compression policies, retention policies, and health monitoring.
    """
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._setup_status = {}
        
    def get_engine(self):
        """Get or create database engine."""
        if self.engine is None:
            self.engine = get_timescale_engine(async_mode=False)
            if self.engine is not None:
                self.session_factory = get_timescale_session(async_mode=False)
        return self.engine
    
    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test database connection and TimescaleDB availability.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            engine = self.get_engine()
            if engine is None:
                return False, "Could not create database engine. Check configuration."
            
            with engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT version()"))
                postgres_version = result.scalar()
                
                # Test TimescaleDB extension
                try:
                    result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"))
                    timescale_version = result.scalar()
                    
                    if timescale_version is None:
                        return False, "TimescaleDB extension not installed in database"
                    
                    logger.info(f"Connected to PostgreSQL {postgres_version}")
                    logger.info(f"TimescaleDB version: {timescale_version}")
                    return True, None
                    
                except Exception as e:
                    return False, f"TimescaleDB extension not available: {str(e)}"
                    
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"
    
    def create_database_schema(self) -> bool:
        """
        Create all database tables and schema objects.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self.get_engine()
            if engine is None:
                logger.error("No database engine available")
                return False
            
            logger.info("Creating TimescaleDB schema...")
            
            # Create all tables
            TimescaleBase.metadata.create_all(bind=engine)
            
            logger.info("Successfully created TimescaleDB schema")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            return False
    
    def create_hypertables(self) -> bool:
        """
        Convert regular tables to TimescaleDB hypertables for time-series optimization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self.get_engine()
            if engine is None:
                return False
            
            with engine.connect() as conn:
                # Create hypertable for sentiment_data
                logger.info("Creating hypertable for sentiment_data...")
                
                # Check if already a hypertable
                check_sql = """
                SELECT * FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'sentiment_data'
                """
                result = conn.execute(text(check_sql))
                
                if result.fetchone() is None:
                    # Create hypertable
                    chunk_interval = getattr(settings, 'TIMESCALE_CHUNK_INTERVAL', '1 day')
                    
                    hypertable_sql = f"""
                    SELECT create_hypertable(
                        'sentiment_data', 
                        'timestamp',
                        chunk_time_interval => INTERVAL '{chunk_interval}',
                        if_not_exists => TRUE
                    )
                    """
                    conn.execute(text(hypertable_sql))
                    logger.info(f"Created hypertable for sentiment_data with chunk interval: {chunk_interval}")
                else:
                    logger.info("sentiment_data hypertable already exists")
                
                # Create hypertable for sentiment_aggregates
                logger.info("Creating hypertable for sentiment_aggregates...")
                
                check_agg_sql = """
                SELECT * FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'sentiment_aggregates'
                """
                result = conn.execute(text(check_agg_sql))
                
                if result.fetchone() is None:
                    # Use longer chunk interval for aggregates
                    agg_chunk_interval = getattr(settings, 'TIMESCALE_CHUNK_INTERVAL', '7 days')
                    
                    hypertable_agg_sql = f"""
                    SELECT create_hypertable(
                        'sentiment_aggregates', 
                        'time_bucket',
                        chunk_time_interval => INTERVAL '{agg_chunk_interval}',
                        if_not_exists => TRUE
                    )
                    """
                    conn.execute(text(hypertable_agg_sql))
                    logger.info(f"Created hypertable for sentiment_aggregates with chunk interval: {agg_chunk_interval}")
                else:
                    logger.info("sentiment_aggregates hypertable already exists")
                
                # Create hypertable for sentiment_alerts
                logger.info("Creating hypertable for sentiment_alerts...")
                
                check_alerts_sql = """
                SELECT * FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'sentiment_alerts'
                """
                result = conn.execute(text(check_alerts_sql))
                
                if result.fetchone() is None:
                    alerts_chunk_interval = '30 days'  # Alerts are less frequent
                    
                    hypertable_alerts_sql = f"""
                    SELECT create_hypertable(
                        'sentiment_alerts', 
                        'triggered_at',
                        chunk_time_interval => INTERVAL '{alerts_chunk_interval}',
                        if_not_exists => TRUE
                    )
                    """
                    conn.execute(text(hypertable_alerts_sql))
                    logger.info(f"Created hypertable for sentiment_alerts with chunk interval: {alerts_chunk_interval}")
                else:
                    logger.info("sentiment_alerts hypertable already exists")
                
                conn.commit()
                logger.info("Successfully created all hypertables")
                return True
                
        except Exception as e:
            logger.error(f"Error creating hypertables: {e}")
            return False
    
    def create_continuous_aggregates(self) -> bool:
        """
        Create continuous aggregates for common time-based queries.
        
        These pre-compute hourly, daily, and weekly aggregations for fast queries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not getattr(settings, 'TIMESCALE_ENABLE_AGGREGATES', True):
                logger.info("Continuous aggregates disabled in configuration")
                return True
            
            engine = self.get_engine()
            if engine is None:
                return False
            
            with engine.connect() as conn:
                # Hourly aggregates
                if getattr(settings, 'TIMESCALE_HOURLY_AGGREGATES', True):
                    logger.info("Creating hourly continuous aggregates...")
                    
                    hourly_agg_sql = """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_hourly_agg
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        symbol,
                        time_bucket('1 hour', timestamp) AS hour_bucket,
                        AVG(overall_score) as avg_score,
                        MIN(overall_score) as min_score,
                        MAX(overall_score) as max_score,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) as data_points,
                        COUNT(*) FILTER (WHERE confidence > 0.7) as high_confidence_points,
                        AVG(twitter_sentiment) as twitter_avg,
                        AVG(reddit_sentiment) as reddit_avg,
                        AVG(news_sentiment) as news_avg,
                        STDDEV(overall_score) as score_stddev
                    FROM sentiment_data
                    GROUP BY symbol, hour_bucket
                    """
                    
                    try:
                        conn.execute(text(hourly_agg_sql))
                        logger.info("Created hourly continuous aggregate")
                    except ProgrammingError as e:
                        if "already exists" in str(e):
                            logger.info("Hourly continuous aggregate already exists")
                        else:
                            raise e
                
                # Daily aggregates
                if getattr(settings, 'TIMESCALE_DAILY_AGGREGATES', True):
                    logger.info("Creating daily continuous aggregates...")
                    
                    daily_agg_sql = """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_daily_agg
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        symbol,
                        time_bucket('1 day', timestamp) AS day_bucket,
                        AVG(overall_score) as avg_score,
                        MIN(overall_score) as min_score,
                        MAX(overall_score) as max_score,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) as data_points,
                        COUNT(*) FILTER (WHERE confidence > 0.7) as high_confidence_points,
                        AVG(twitter_sentiment) as twitter_avg,
                        AVG(reddit_sentiment) as reddit_avg,
                        AVG(news_sentiment) as news_avg,
                        STDDEV(overall_score) as score_stddev,
                        -- Momentum calculation: later half vs earlier half sentiment
                        (AVG(CASE WHEN timestamp >= day_bucket + interval '12 hours' THEN overall_score END) -
                         AVG(CASE WHEN timestamp < day_bucket + interval '12 hours' THEN overall_score END)) as momentum_change
                    FROM sentiment_data
                    GROUP BY symbol, day_bucket
                    """
                    
                    try:
                        conn.execute(text(daily_agg_sql))
                        logger.info("Created daily continuous aggregate")
                    except ProgrammingError as e:
                        if "already exists" in str(e):
                            logger.info("Daily continuous aggregate already exists")
                        else:
                            raise e
                
                # Weekly aggregates
                if getattr(settings, 'TIMESCALE_WEEKLY_AGGREGATES', True):
                    logger.info("Creating weekly continuous aggregates...")
                    
                    weekly_agg_sql = """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_weekly_agg
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        symbol,
                        time_bucket('1 week', timestamp) AS week_bucket,
                        AVG(overall_score) as avg_score,
                        MIN(overall_score) as min_score,
                        MAX(overall_score) as max_score,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) as data_points,
                        COUNT(*) FILTER (WHERE confidence > 0.7) as high_confidence_points,
                        AVG(twitter_sentiment) as twitter_avg,
                        AVG(reddit_sentiment) as reddit_avg,
                        AVG(news_sentiment) as news_avg,
                        STDDEV(overall_score) as score_stddev
                    FROM sentiment_data
                    GROUP BY symbol, week_bucket
                    """
                    
                    try:
                        conn.execute(text(weekly_agg_sql))
                        logger.info("Created weekly continuous aggregate")
                    except ProgrammingError as e:
                        if "already exists" in str(e):
                            logger.info("Weekly continuous aggregate already exists")
                        else:
                            raise e
                
                conn.commit()
                logger.info("Successfully created continuous aggregates")
                return True
                
        except Exception as e:
            logger.error(f"Error creating continuous aggregates: {e}")
            return False
    
    def setup_compression_policies(self) -> bool:
        """
        Set up data compression policies to save storage space.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self.get_engine()
            if engine is None:
                return False
            
            compression_after = getattr(settings, 'TIMESCALE_COMPRESSION_AFTER', '7 days')
            
            with engine.connect() as conn:
                logger.info(f"Setting up compression policy (compress after {compression_after})...")
                
                # Enable compression on sentiment_data hypertable
                enable_compression_sql = """
                ALTER TABLE sentiment_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                )
                """
                
                try:
                    conn.execute(text(enable_compression_sql))
                    logger.info("Enabled compression on sentiment_data table")
                except ProgrammingError as e:
                    if "already compressed" in str(e) or "already set" in str(e):
                        logger.info("Compression already enabled on sentiment_data")
                    else:
                        logger.warning(f"Could not enable compression: {e}")
                
                # Add compression policy
                compression_policy_sql = f"""
                SELECT add_compression_policy('sentiment_data', INTERVAL '{compression_after}', if_not_exists => TRUE)
                """
                
                try:
                    conn.execute(text(compression_policy_sql))
                    logger.info(f"Added compression policy for sentiment_data (compress after {compression_after})")
                except Exception as e:
                    logger.warning(f"Could not add compression policy: {e}")
                
                # Enable compression on aggregates table
                enable_agg_compression_sql = """
                ALTER TABLE sentiment_aggregates SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol,interval_type',
                    timescaledb.compress_orderby = 'time_bucket DESC'
                )
                """
                
                try:
                    conn.execute(text(enable_agg_compression_sql))
                    logger.info("Enabled compression on sentiment_aggregates table")
                except ProgrammingError as e:
                    if "already compressed" in str(e) or "already set" in str(e):
                        logger.info("Compression already enabled on sentiment_aggregates")
                    else:
                        logger.warning(f"Could not enable compression on aggregates: {e}")
                
                # Add compression policy for aggregates (longer interval)
                agg_compression_after = getattr(settings, 'TIMESCALE_COMPRESSION_AFTER', '30 days')
                agg_compression_policy_sql = f"""
                SELECT add_compression_policy('sentiment_aggregates', INTERVAL '{agg_compression_after}', if_not_exists => TRUE)
                """
                
                try:
                    conn.execute(text(agg_compression_policy_sql))
                    logger.info(f"Added compression policy for sentiment_aggregates (compress after {agg_compression_after})")
                except Exception as e:
                    logger.warning(f"Could not add aggregates compression policy: {e}")
                
                conn.commit()
                logger.info("Successfully set up compression policies")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up compression policies: {e}")
            return False
    
    def setup_retention_policies(self) -> bool:
        """
        Set up data retention policies to automatically drop old data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self.get_engine()
            if engine is None:
                return False
            
            retention_period = getattr(settings, 'TIMESCALE_RETENTION_POLICY', '1 year')
            
            with engine.connect() as conn:
                logger.info(f"Setting up retention policy (drop data older than {retention_period})...")
                
                # Add retention policy for sentiment_data
                retention_policy_sql = f"""
                SELECT add_retention_policy('sentiment_data', INTERVAL '{retention_period}', if_not_exists => TRUE)
                """
                
                try:
                    conn.execute(text(retention_policy_sql))
                    logger.info(f"Added retention policy for sentiment_data (drop after {retention_period})")
                except Exception as e:
                    logger.warning(f"Could not add retention policy: {e}")
                
                # Add retention policy for aggregates (longer period)
                agg_retention_period = '2 years'  # Keep aggregates longer
                agg_retention_policy_sql = f"""
                SELECT add_retention_policy('sentiment_aggregates', INTERVAL '{agg_retention_period}', if_not_exists => TRUE)
                """
                
                try:
                    conn.execute(text(agg_retention_policy_sql))
                    logger.info(f"Added retention policy for sentiment_aggregates (drop after {agg_retention_period})")
                except Exception as e:
                    logger.warning(f"Could not add aggregates retention policy: {e}")
                
                # Add retention policy for alerts (shorter period)
                alerts_retention_period = '6 months'
                alerts_retention_policy_sql = f"""
                SELECT add_retention_policy('sentiment_alerts', INTERVAL '{alerts_retention_period}', if_not_exists => TRUE)
                """
                
                try:
                    conn.execute(text(alerts_retention_policy_sql))
                    logger.info(f"Added retention policy for sentiment_alerts (drop after {alerts_retention_period})")
                except Exception as e:
                    logger.warning(f"Could not add alerts retention policy: {e}")
                
                conn.commit()
                logger.info("Successfully set up retention policies")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up retention policies: {e}")
            return False
    
    def create_additional_indexes(self) -> bool:
        """
        Create additional indexes for query optimization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self.get_engine()
            if engine is None:
                return False
            
            with engine.connect() as conn:
                logger.info("Creating additional performance indexes...")
                
                # Additional indexes for common query patterns
                indexes = [
                    # Multi-symbol queries with time range
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_symbol_timestamp_score ON sentiment_data (symbol, timestamp DESC, overall_score)",
                    
                    # High confidence data queries
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_high_confidence ON sentiment_data (confidence, timestamp DESC) WHERE confidence > 0.7",
                    
                    # Market regime analysis
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_regime_symbol_time ON sentiment_data (market_regime, symbol, timestamp DESC) WHERE market_regime IS NOT NULL",
                    
                    # Source-specific queries
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_twitter_time ON sentiment_data (timestamp DESC) WHERE twitter_sentiment IS NOT NULL",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_reddit_time ON sentiment_data (timestamp DESC) WHERE reddit_sentiment IS NOT NULL",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_news_time ON sentiment_data (timestamp DESC) WHERE news_sentiment IS NOT NULL",
                    
                    # Aggregates optimization
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agg_symbol_interval_bucket ON sentiment_aggregates (symbol, interval_type, time_bucket DESC)",
                    
                    # Alerts optimization
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_active ON sentiment_alerts (symbol, alert_type, triggered_at DESC) WHERE resolved_at IS NULL",
                ]
                
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0] if 'idx_' in index_sql else 'unknown'}")
                    except Exception as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Could not create index: {e}")
                
                conn.commit()
                logger.info("Successfully created additional indexes")
                return True
                
        except Exception as e:
            logger.error(f"Error creating additional indexes: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics and health metrics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "connection_status": "unknown",
            "tables": {},
            "hypertables": [],
            "compression": {},
            "policies": {},
            "performance": {}
        }
        
        try:
            engine = self.get_engine()
            if engine is None:
                stats["connection_status"] = "no_engine"
                return stats
            
            with engine.connect() as conn:
                stats["connection_status"] = "connected"
                
                # Table statistics
                table_stats_sql = """
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples
                FROM pg_stat_user_tables 
                WHERE tablename IN ('sentiment_data', 'sentiment_aggregates', 'sentiment_alerts', 'sentiment_metadata')
                """
                
                result = conn.execute(text(table_stats_sql))
                for row in result:
                    stats["tables"][row.tablename] = {
                        "inserts": row.inserts,
                        "updates": row.updates,
                        "deletes": row.deletes,
                        "live_tuples": row.live_tuples,
                        "dead_tuples": row.dead_tuples
                    }
                
                # Hypertable information
                hypertable_sql = """
                SELECT 
                    hypertable_name,
                    num_dimensions,
                    num_chunks,
                    table_size,
                    index_size,
                    total_size
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
                """
                
                try:
                    result = conn.execute(text(hypertable_sql))
                    for row in result:
                        stats["hypertables"].append({
                            "name": row.hypertable_name,
                            "num_chunks": row.num_chunks,
                            "table_size": row.table_size,
                            "index_size": row.index_size,
                            "total_size": row.total_size
                        })
                except Exception as e:
                    logger.warning(f"Could not get hypertable stats: {e}")
                
                # Compression statistics
                compression_sql = """
                SELECT 
                    hypertable_name,
                    compression_status,
                    uncompressed_heap_size,
                    compressed_heap_size,
                    uncompressed_index_size,
                    compressed_index_size
                FROM timescaledb_information.compression_settings cs
                JOIN timescaledb_information.hypertables h ON cs.hypertable_name = h.hypertable_name
                WHERE h.hypertable_schema = 'public'
                """
                
                try:
                    result = conn.execute(text(compression_sql))
                    for row in result:
                        stats["compression"][row.hypertable_name] = {
                            "status": row.compression_status,
                            "uncompressed_heap_size": row.uncompressed_heap_size,
                            "compressed_heap_size": row.compressed_heap_size,
                            "compression_ratio": (
                                row.uncompressed_heap_size / row.compressed_heap_size 
                                if row.compressed_heap_size and row.compressed_heap_size > 0 
                                else None
                            )
                        }
                except Exception as e:
                    logger.warning(f"Could not get compression stats: {e}")
                
                # Recent performance metrics
                performance_sql = """
                SELECT 
                    NOW() as check_time,
                    COUNT(*) as active_connections
                FROM pg_stat_activity
                """
                
                result = conn.execute(text(performance_sql))
                row = result.fetchone()
                if row:
                    stats["performance"] = {
                        "check_time": row.check_time.isoformat(),
                        "active_connections": row.active_connections
                    }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            stats["connection_status"] = f"error: {str(e)}"
        
        return stats
    
    def run_full_setup(self) -> bool:
        """
        Run complete TimescaleDB setup process.
        
        Returns:
            True if all setup steps completed successfully, False otherwise
        """
        logger.info("Starting full TimescaleDB setup...")
        
        setup_steps = [
            ("Test Connection", self.test_connection),
            ("Create Schema", self.create_database_schema),
            ("Create Hypertables", self.create_hypertables),
            ("Create Continuous Aggregates", self.create_continuous_aggregates),
            ("Setup Compression Policies", self.setup_compression_policies),
            ("Setup Retention Policies", self.setup_retention_policies),
            ("Create Additional Indexes", self.create_additional_indexes),
        ]
        
        all_success = True
        
        for step_name, step_function in setup_steps:
            logger.info(f"Running setup step: {step_name}")
            
            try:
                if step_name == "Test Connection":
                    success, error_msg = step_function()
                    if not success:
                        logger.error(f"{step_name} failed: {error_msg}")
                        all_success = False
                        continue
                else:
                    success = step_function()
                    if not success:
                        logger.error(f"{step_name} failed")
                        all_success = False
                        continue
                
                logger.info(f"{step_name} completed successfully")
                self._setup_status[step_name] = "success"
                
            except Exception as e:
                logger.error(f"{step_name} failed with exception: {e}")
                self._setup_status[step_name] = f"error: {str(e)}"
                all_success = False
        
        if all_success:
            logger.info("Full TimescaleDB setup completed successfully!")
        else:
            logger.warning("TimescaleDB setup completed with some failures. Check logs for details.")
        
        return all_success
    
    def get_setup_status(self) -> Dict[str, str]:
        """Get status of all setup steps."""
        return self._setup_status.copy()

# ================================
# STANDALONE FUNCTIONS
# ================================

def initialize_timescale_database() -> bool:
    """
    Initialize TimescaleDB database with full setup.
    
    This is the main entry point for database initialization.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if TimescaleDB is available
        if not is_timescale_available():
            logger.warning("TimescaleDB not available or not configured")
            return False
        
        # Run full setup
        manager = TimescaleSetupManager()
        return manager.run_full_setup()
        
    except Exception as e:
        logger.error(f"Error initializing TimescaleDB: {e}")
        return False

def get_database_health() -> Dict[str, Any]:
    """
    Get comprehensive database health and statistics.
    
    Returns:
        Dictionary with health information
    """
    try:
        manager = TimescaleSetupManager()
        
        # Test connection first
        connection_ok, error_msg = manager.test_connection()
        
        if not connection_ok:
            return {
                "status": "unhealthy",
                "connection_error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get detailed stats
        stats = manager.get_database_stats()
        stats["status"] = "healthy"
        stats["timestamp"] = datetime.utcnow().isoformat()
        
        return stats
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def run_maintenance() -> Dict[str, Any]:
    """
    Run database maintenance tasks.
    
    Returns:
        Dictionary with maintenance results
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "tasks": {}
    }
    
    try:
        manager = TimescaleSetupManager()
        engine = manager.get_engine()
        
        if engine is None:
            results["error"] = "No database engine available"
            return results
        
        with engine.connect() as conn:
            # Refresh continuous aggregates
            logger.info("Refreshing continuous aggregates...")
            
            refresh_tasks = [
                ("sentiment_hourly_agg", "1 hour"),
                ("sentiment_daily_agg", "1 day"), 
                ("sentiment_weekly_agg", "1 week")
            ]
            
            for view_name, interval in refresh_tasks:
                try:
                    # Refresh the last period
                    refresh_sql = f"""
                    CALL refresh_continuous_aggregate('{view_name}', 
                        NOW() - INTERVAL '{interval}', NOW())
                    """
                    conn.execute(text(refresh_sql))
                    results["tasks"][f"refresh_{view_name}"] = "success"
                    logger.info(f"Refreshed {view_name}")
                except Exception as e:
                    results["tasks"][f"refresh_{view_name}"] = f"error: {str(e)}"
                    logger.warning(f"Could not refresh {view_name}: {e}")
            
            # Analyze tables for query optimization
            logger.info("Running table analysis...")
            analyze_tables = ['sentiment_data', 'sentiment_aggregates', 'sentiment_alerts']
            
            for table in analyze_tables:
                try:
                    conn.execute(text(f"ANALYZE {table}"))
                    results["tasks"][f"analyze_{table}"] = "success"
                except Exception as e:
                    results["tasks"][f"analyze_{table}"] = f"error: {str(e)}"
            
            conn.commit()
            logger.info("Database maintenance completed")
            
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error during database maintenance: {e}")
    
    return results

# Export main functions
__all__ = [
    'TimescaleSetupManager',
    'initialize_timescale_database',
    'get_database_health',
    'run_maintenance'
]