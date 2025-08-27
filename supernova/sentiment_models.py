"""TimescaleDB Models for Sentiment Data

Time-series optimized models for storing and querying sentiment analysis data.
Uses TimescaleDB hypertables for efficient time-series operations and querying.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import (
    create_engine, String, Integer, Float, DateTime, Text, JSON, Index,
    CheckConstraint, UniqueConstraint, func, text
)
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Local imports for configuration
try:
    from .config import settings
except ImportError:
    # Fallback settings if config module not available
    class MockSettings:
        TIMESCALE_HOST: str = "localhost"
        TIMESCALE_PORT: int = 5432
        TIMESCALE_DB: str = "supernova_timeseries"
        TIMESCALE_USER: str = "postgres"
        TIMESCALE_PASSWORD: str = "password"
        TIMESCALE_POOL_SIZE: int = 5
        TIMESCALE_MAX_OVERFLOW: int = 10
        TIMESCALE_POOL_TIMEOUT: int = 30
        TIMESCALE_POOL_RECYCLE: int = 3600
    settings = MockSettings()

class TimescaleBase(DeclarativeBase):
    """Base class for TimescaleDB models"""
    pass

class SentimentData(TimescaleBase):
    """
    Main sentiment data table optimized for time-series queries.
    
    This table stores individual sentiment data points with full metadata
    and is designed to be partitioned as a TimescaleDB hypertable.
    """
    __tablename__ = "sentiment_data"
    
    # Primary key components for time-series data
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), 
        primary_key=True,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    
    # Core sentiment metrics
    overall_score: Mapped[float] = mapped_column(
        Float, 
        CheckConstraint('overall_score >= -1.0 AND overall_score <= 1.0'),
        nullable=False,
        index=True
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0'),
        nullable=False,
        index=True
    )
    
    # Social media sentiment breakdown
    social_momentum: Mapped[float] = mapped_column(Float, nullable=True, index=True)
    news_sentiment: Mapped[float] = mapped_column(Float, nullable=True, index=True)
    twitter_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    reddit_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Source metadata and counts
    source_counts: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Market regime context
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    
    # Additional analytics
    figure_influence: Mapped[float] = mapped_column(Float, default=0.0)
    contrarian_indicator: Mapped[float] = mapped_column(Float, default=0.0)
    regime_adjusted_score: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Data quality and processing metadata
    total_data_points: Mapped[int] = mapped_column(Integer, default=0)
    processing_version: Mapped[str] = mapped_column(String(20), default="1.0")
    
    # Constraints
    __table_args__ = (
        # Ensure no duplicate entries for same symbol at same time
        UniqueConstraint('symbol', 'timestamp', name='uq_sentiment_symbol_timestamp'),
        
        # Optimize common query patterns
        Index('ix_sentiment_symbol_time_desc', 'symbol', 'timestamp', postgresql_using='btree'),
        Index('ix_sentiment_time_score', 'timestamp', 'overall_score'),
        Index('ix_sentiment_confidence_time', 'confidence', 'timestamp'),
        Index('ix_sentiment_regime_time', 'market_regime', 'timestamp'),
        
        # GIN index for JSON source_counts for efficient JSON queries
        Index('ix_sentiment_source_counts_gin', 'source_counts', postgresql_using='gin'),
    )

class SentimentAggregates(TimescaleBase):
    """
    Pre-computed sentiment aggregates for common time ranges.
    
    This table stores hourly, daily, and weekly aggregations to speed up
    common queries and dashboard displays.
    """
    __tablename__ = "sentiment_aggregates"
    
    # Primary key and partitioning
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, index=True)
    time_bucket: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), 
        primary_key=True,
        index=True
    )
    interval_type: Mapped[str] = mapped_column(
        String(10), 
        CheckConstraint("interval_type IN ('1h', '6h', '1d', '1w')"),
        primary_key=True,
        index=True
    )
    
    # Aggregated metrics
    avg_score: Mapped[float] = mapped_column(Float)
    min_score: Mapped[float] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float)
    avg_confidence: Mapped[float] = mapped_column(Float)
    
    # Count statistics
    data_points: Mapped[int] = mapped_column(Integer)
    high_confidence_points: Mapped[int] = mapped_column(Integer, default=0)
    
    # Source distribution
    twitter_avg: Mapped[Optional[float]] = mapped_column(Float)
    reddit_avg: Mapped[Optional[float]] = mapped_column(Float)
    news_avg: Mapped[Optional[float]] = mapped_column(Float)
    
    # Volatility and trend metrics
    score_stddev: Mapped[Optional[float]] = mapped_column(Float)
    momentum_change: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    
    __table_args__ = (
        Index('ix_aggregates_symbol_interval_time', 'symbol', 'interval_type', 'time_bucket'),
        Index('ix_aggregates_time_interval', 'time_bucket', 'interval_type'),
    )

class SentimentAlerts(TimescaleBase):
    """
    Sentiment-based alerts and threshold breaches.
    
    Stores alerts triggered by sentiment threshold breaches,
    unusual patterns, or significant changes.
    """
    __tablename__ = "sentiment_alerts"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Alert details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(
        String(50), 
        CheckConstraint("alert_type IN ('threshold_breach', 'volatility_spike', 'momentum_change', 'confidence_drop', 'source_divergence')"),
        nullable=False,
        index=True
    )
    
    # Timing
    triggered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    
    # Alert data
    trigger_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Alert metadata
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(
        String(10), 
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')"),
        default="medium"
    )
    acknowledged: Mapped[bool] = mapped_column(Integer, default=0)  # Use Integer for boolean in PostgreSQL
    
    # Context data
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    __table_args__ = (
        Index('ix_alerts_symbol_time', 'symbol', 'triggered_at'),
        Index('ix_alerts_type_time', 'alert_type', 'triggered_at'),
        Index('ix_alerts_severity_time', 'severity', 'triggered_at'),
        Index('ix_alerts_unresolved', 'resolved_at', postgresql_where=text('resolved_at IS NULL')),
    )

class SentimentMetadata(TimescaleBase):
    """
    Metadata about sentiment data processing and quality.
    
    Tracks processing runs, data quality metrics, and system health.
    """
    __tablename__ = "sentiment_metadata"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Processing run info
    run_timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    processing_type: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("processing_type IN ('realtime', 'batch', 'backfill', 'aggregation')"),
        index=True
    )
    
    # Symbols processed
    symbols_processed: Mapped[List[str]] = mapped_column(JSON)
    
    # Quality metrics
    total_records_processed: Mapped[int] = mapped_column(Integer, default=0)
    successful_records: Mapped[int] = mapped_column(Integer, default=0)
    failed_records: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Source statistics
    twitter_records: Mapped[int] = mapped_column(Integer, default=0)
    reddit_records: Mapped[int] = mapped_column(Integer, default=0)
    news_records: Mapped[int] = mapped_column(Integer, default=0)
    
    # Processing performance
    processing_duration_seconds: Mapped[float] = mapped_column(Float)
    records_per_second: Mapped[float] = mapped_column(Float)
    
    # Error information
    errors: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    warnings: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # System context
    processing_version: Mapped[str] = mapped_column(String(20), default="1.0")
    config_hash: Mapped[Optional[str]] = mapped_column(String(64))  # Hash of processing config
    
    __table_args__ = (
        Index('ix_metadata_timestamp', 'run_timestamp'),
        Index('ix_metadata_type_timestamp', 'processing_type', 'run_timestamp'),
    )

# ================================
# DATABASE CONNECTION MANAGEMENT
# ================================

def get_timescale_url(async_mode: bool = False) -> str:
    """
    Generate TimescaleDB connection URL.
    
    Args:
        async_mode: If True, returns asyncpg URL for async operations
        
    Returns:
        Database connection URL string
    """
    driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"
    
    return (
        f"{driver}://"
        f"{settings.TIMESCALE_USER}:{settings.TIMESCALE_PASSWORD}@"
        f"{settings.TIMESCALE_HOST}:{settings.TIMESCALE_PORT}/"
        f"{settings.TIMESCALE_DB}"
    )

# Synchronous engine for setup and management operations
timescale_engine = None
TimescaleSessionLocal = None

# Asynchronous engine for high-performance operations
async_timescale_engine = None
AsyncTimescaleSessionLocal = None

def get_timescale_engine(async_mode: bool = False):
    """Get TimescaleDB engine, creating if necessary."""
    global timescale_engine, async_timescale_engine, TimescaleSessionLocal, AsyncTimescaleSessionLocal
    
    if async_mode:
        if async_timescale_engine is None:
            async_timescale_engine = create_async_engine(
                get_timescale_url(async_mode=True),
                echo=False,
                pool_size=getattr(settings, 'TIMESCALE_POOL_SIZE', 5),
                max_overflow=getattr(settings, 'TIMESCALE_MAX_OVERFLOW', 10),
                pool_timeout=getattr(settings, 'TIMESCALE_POOL_TIMEOUT', 30),
                pool_recycle=getattr(settings, 'TIMESCALE_POOL_RECYCLE', 3600),
            )
            AsyncTimescaleSessionLocal = async_sessionmaker(
                bind=async_timescale_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        return async_timescale_engine
    else:
        if timescale_engine is None:
            timescale_engine = create_engine(
                get_timescale_url(async_mode=False),
                echo=False,
                pool_size=getattr(settings, 'TIMESCALE_POOL_SIZE', 5),
                max_overflow=getattr(settings, 'TIMESCALE_MAX_OVERFLOW', 10),
                pool_timeout=getattr(settings, 'TIMESCALE_POOL_TIMEOUT', 30),
                pool_recycle=getattr(settings, 'TIMESCALE_POOL_RECYCLE', 3600),
            )
            TimescaleSessionLocal = sessionmaker(
                bind=timescale_engine,
                autoflush=False,
                autocommit=False
            )
        return timescale_engine

def get_timescale_session(async_mode: bool = False):
    """Get TimescaleDB session factory."""
    get_timescale_engine(async_mode=async_mode)  # Ensure engine is created
    
    if async_mode:
        return AsyncTimescaleSessionLocal
    else:
        return TimescaleSessionLocal

# ================================
# UTILITY FUNCTIONS
# ================================

def to_dict(obj) -> dict:
    """Convert SQLAlchemy model to dictionary."""
    if obj is None:
        return {}
    
    result = {}
    for column in obj.__table__.columns:
        value = getattr(obj, column.name)
        
        # Handle datetime objects
        if isinstance(value, datetime):
            result[column.name] = value.isoformat()
        # Handle JSON/JSONB fields
        elif isinstance(value, (dict, list)):
            result[column.name] = value
        else:
            result[column.name] = value
    
    return result

def from_sentiment_signal(signal, symbol: str) -> SentimentData:
    """
    Convert SentimentSignal dataclass to SentimentData model.
    
    Args:
        signal: SentimentSignal object from sentiment analysis
        symbol: Stock symbol
        
    Returns:
        SentimentData model instance
    """
    return SentimentData(
        symbol=symbol,
        timestamp=signal.timestamp,
        overall_score=signal.overall_score,
        confidence=signal.confidence,
        social_momentum=signal.social_momentum,
        news_sentiment=signal.news_impact,  # Map news_impact to news_sentiment
        twitter_sentiment=signal.source_breakdown.get('twitter'),
        reddit_sentiment=signal.source_breakdown.get('reddit'),
        source_counts=getattr(signal, 'source_counts', {}),
        market_regime=getattr(signal, 'market_regime', None),
        figure_influence=signal.figure_influence,
        contrarian_indicator=signal.contrarian_indicator,
        regime_adjusted_score=signal.regime_adjusted_score,
        total_data_points=getattr(signal, 'total_data_points', 0)
    )

# Export main classes and functions
__all__ = [
    'TimescaleBase',
    'SentimentData',
    'SentimentAggregates', 
    'SentimentAlerts',
    'SentimentMetadata',
    'get_timescale_engine',
    'get_timescale_session',
    'get_timescale_url',
    'to_dict',
    'from_sentiment_signal'
]