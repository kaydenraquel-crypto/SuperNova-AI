from sqlalchemy import create_engine, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.sql import func
from typing import Optional
import logging
from .config import settings

# Import analytics models for registration
try:
    from .analytics_models import (
        Portfolio, Position, Transaction, PerformanceRecord, RiskMetric,
        MarketSentiment, TechnicalIndicator, BacktestAnalysis, AnalyticsReport
    )
    ANALYTICS_MODELS_AVAILABLE = True
except ImportError:
    ANALYTICS_MODELS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# SQLite engine for application data (users, profiles, etc.)
engine = create_engine(settings.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# TimescaleDB engine for time-series sentiment data (optional)
timescale_engine = None
TimescaleSessionLocal = None

def get_timescale_connection_url() -> Optional[str]:
    """
    Build TimescaleDB connection URL from settings.
    Returns None if TimescaleDB is not configured.
    """
    required_settings = ['TIMESCALE_HOST', 'TIMESCALE_DB', 'TIMESCALE_USER', 'TIMESCALE_PASSWORD']
    
    # Check if all required TimescaleDB settings are present
    for setting in required_settings:
        if not hasattr(settings, setting) or not getattr(settings, setting):
            logger.info(f"TimescaleDB not configured: missing {setting}")
            return None
    
    port = getattr(settings, 'TIMESCALE_PORT', 5432)
    
    return (
        f"postgresql+psycopg2://"
        f"{settings.TIMESCALE_USER}:{settings.TIMESCALE_PASSWORD}@"
        f"{settings.TIMESCALE_HOST}:{port}/"
        f"{settings.TIMESCALE_DB}"
    )

def init_timescale_db():
    """
    Initialize TimescaleDB connection and session factory.
    This is optional and only runs if TimescaleDB is properly configured.
    """
    global timescale_engine, TimescaleSessionLocal
    
    if timescale_engine is not None:
        return timescale_engine  # Already initialized
    
    connection_url = get_timescale_connection_url()
    if not connection_url:
        logger.info("TimescaleDB not configured, skipping time-series database initialization")
        return None
    
    try:
        # Create TimescaleDB engine with connection pooling
        timescale_engine = create_engine(
            connection_url,
            echo=False,
            future=True,
            pool_size=getattr(settings, 'TIMESCALE_POOL_SIZE', 5),
            max_overflow=getattr(settings, 'TIMESCALE_MAX_OVERFLOW', 10),
            pool_timeout=getattr(settings, 'TIMESCALE_POOL_TIMEOUT', 30),
            pool_recycle=getattr(settings, 'TIMESCALE_POOL_RECYCLE', 3600),
        )
        
        # Test connection
        with timescale_engine.connect() as conn:
            result = conn.execute(func.version())
            version_info = result.scalar()
            logger.info(f"Connected to TimescaleDB: {version_info}")
        
        # Create session factory
        TimescaleSessionLocal = sessionmaker(
            bind=timescale_engine,
            autoflush=False,
            autocommit=False
        )
        
        logger.info("TimescaleDB connection initialized successfully")
        return timescale_engine
        
    except Exception as e:
        logger.error(f"Failed to initialize TimescaleDB connection: {e}")
        logger.info("Continuing with SQLite-only mode")
        timescale_engine = None
        TimescaleSessionLocal = None
        return None

def is_timescale_available() -> bool:
    """Check if TimescaleDB connection is available."""
    return timescale_engine is not None and TimescaleSessionLocal is not None

def get_timescale_session():
    """
    Get TimescaleDB session factory.
    Returns None if TimescaleDB is not available.
    """
    if not is_timescale_available():
        init_timescale_db()  # Try to initialize if not already done
    
    return TimescaleSessionLocal

class Base(DeclarativeBase): pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str | None] = mapped_column(String(200), unique=True, index=True)
    
    # Authentication fields
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # MFA fields
    mfa_secret: Mapped[str | None] = mapped_column(String(255))
    mfa_backup_codes: Mapped[str | None] = mapped_column(Text)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    profiles: Mapped[list["Profile"]] = relationship(back_populates="user")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")
    context: Mapped["UserContext"] = relationship(back_populates="user", uselist=False)
    
    # Collaboration relationships (will be added when collaboration_models is imported)
    teams: Mapped[list["Team"]] = relationship(
        "Team", 
        secondary="team_members",
        back_populates="members",
        overlaps="teams"
    )

class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    risk_score: Mapped[int] = mapped_column(Integer, default=50)  # 0-100
    time_horizon_yrs: Mapped[int] = mapped_column(Integer, default=5)
    objectives: Mapped[str] = mapped_column(Text, default="growth")
    constraints: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    user: Mapped["User"] = relationship(back_populates="profiles")

class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(40), index=True)
    asset_class: Mapped[str] = mapped_column(String(20))  # stock, crypto, fx, futures, option
    description: Mapped[str | None] = mapped_column(String(200))

class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"))
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

class Strategy(Base):
    __tablename__ = "strategies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    template: Mapped[str] = mapped_column(String(100))  # e.g., 'ma_crossover', 'rsi_breakout'
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    profile_id: Mapped[int | None] = mapped_column(Integer)  # optional owner

class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(40))
    timeframe: Mapped[str] = mapped_column(String(20))
    metrics_json: Mapped[str] = mapped_column(Text)  # CAGR, Sharpe, MaxDD, WinRate
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    watchlist_item_id: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)

# ================================
# CONVERSATION & MEMORY MODELS
# ================================

class Conversation(Base):
    """Conversation/chat session model"""
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Conversation UUID
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    personality: Mapped[str] = mapped_column(String(50), default="balanced_analyst")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    last_activity: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Status and settings
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Context data (JSON)
    context_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Statistics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

class ConversationMessage(Base):
    """Individual conversation message model"""
    __tablename__ = "conversation_messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    
    # Message content
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    
    # Context and metadata
    memory_type: Mapped[str] = mapped_column(String(50), default="user_message")
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    context_tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array as string
    metadata: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object as string
    
    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Processing info
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

class UserContext(Base):
    """User context and preferences model"""
    __tablename__ = "user_contexts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    
    # Preferences (JSON)
    preferences: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    financial_profile: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    interaction_patterns: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    learned_preferences: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    
    # Personality preferences (JSON)
    personality_affinity: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Learning statistics
    total_interactions: Mapped[int] = mapped_column(Integer, default=0)
    learning_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="context")

class ConversationSession(Base):
    """Session memory model for temporary conversation state"""
    __tablename__ = "conversation_sessions"
    
    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Session UUID
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Session state (JSON)
    session_state: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    active_context: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    conversation_flow: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship()
    user: Mapped["User"] = relationship()

class FinancialContext(Base):
    """Financial context storage for conversations"""
    __tablename__ = "financial_contexts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    
    # Context information
    context_type: Mapped[str] = mapped_column(String(100))  # market_data, analysis_result, etc.
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    
    # Context data (JSON)
    context_data: Mapped[str] = mapped_column(Text)  # JSON object
    
    # Metadata
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    relevance_score: Mapped[float] = mapped_column(Float, default=0.5)
    
    # Tags for retrieval
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array as string
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship()

class AgentInteraction(Base):
    """Agent interaction tracking for learning and analytics"""
    __tablename__ = "agent_interactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    
    # Interaction details
    query_type: Mapped[str] = mapped_column(String(50))
    personality_used: Mapped[str] = mapped_column(String(50))
    tools_used: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    
    # Performance metrics
    response_time_ms: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # User feedback
    user_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5 rating
    
    # Context
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship()
    user: Mapped["User"] = relationship()

class MemoryOptimizationLog(Base):
    """Log of memory optimization operations"""
    __tablename__ = "memory_optimization_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Operation details
    optimization_type: Mapped[str] = mapped_column(String(50))  # conversation, user_context, etc.
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # conversation_id or user_id
    
    # Results
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_removed: Mapped[int] = mapped_column(Integer, default=0)
    memory_freed_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="completed")  # running, completed, failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

# ================================
# AUTHENTICATION MODELS
# ================================

class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Session UUID
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Session information
    ip_address: Mapped[str | None] = mapped_column(String(45))  # IPv6 support
    user_agent: Mapped[str | None] = mapped_column(Text)
    
    # Status and timing
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Security metadata
    login_method: Mapped[str | None] = mapped_column(String(50))  # password, mfa, sso
    device_fingerprint: Mapped[str | None] = mapped_column(String(255))
    location_info: Mapped[str | None] = mapped_column(Text)  # JSON with location data
    
    # Relationships
    user: Mapped["User"] = relationship()

class APIKey(Base):
    """API keys for programmatic access"""
    __tablename__ = "api_keys"
    
    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Key ID
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Key information
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(255))  # Hash of the actual key
    prefix: Mapped[str] = mapped_column(String(20))  # First few chars for display
    
    # Permissions and scopes
    scopes: Mapped[str | None] = mapped_column(Text)  # JSON array of permissions
    rate_limit_tier: Mapped[str] = mapped_column(String(20), default="standard")
    
    # Status and timing
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    rate_limit_hits: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    user: Mapped["User"] = relationship()

class SecurityEvent(Base):
    """Security events and audit log"""
    __tablename__ = "security_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    
    # Event information
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    event_category: Mapped[str] = mapped_column(String(30))  # auth, access, security, etc.
    severity: Mapped[str] = mapped_column(String(20))  # low, medium, high, critical
    
    # Request information
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(Text)
    endpoint: Mapped[str | None] = mapped_column(String(255))
    method: Mapped[str | None] = mapped_column(String(10))
    
    # Event details
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(Text)  # JSON with additional info
    risk_score: Mapped[int | None] = mapped_column(Integer)  # 0-100
    
    # Timing and status
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    
    # Relationships
    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    resolver: Mapped["User"] = relationship(foreign_keys=[resolved_by])

class PasswordResetToken(Base):
    """Password reset tokens"""
    __tablename__ = "password_reset_tokens"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Token information
    token_hash: Mapped[str] = mapped_column(String(255))  # Hash of the actual token
    token_type: Mapped[str] = mapped_column(String(20), default="password_reset")
    
    # Status and timing
    is_used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Security metadata
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(Text)
    
    # Relationships
    user: Mapped["User"] = relationship()

class RateLimitRecord(Base):
    """Rate limiting tracking"""
    __tablename__ = "rate_limit_records"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Identifier (IP, user_id, or API key)
    identifier: Mapped[str] = mapped_column(String(255), index=True)
    identifier_type: Mapped[str] = mapped_column(String(20))  # ip, user, api_key
    
    # Rate limit information
    endpoint: Mapped[str | None] = mapped_column(String(255))
    method: Mapped[str | None] = mapped_column(String(10))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    window_duration: Mapped[int] = mapped_column(Integer)  # in seconds
    
    # Counts
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    limit_exceeded_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Status
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    blocked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Last update
    last_request: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class LoginAttempt(Base):
    """Failed login attempt tracking"""
    __tablename__ = "login_attempts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Attempt information
    email: Mapped[str] = mapped_column(String(200), index=True)
    ip_address: Mapped[str] = mapped_column(String(45), index=True)
    user_agent: Mapped[str | None] = mapped_column(Text)
    
    # Attempt details
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_reason: Mapped[str | None] = mapped_column(String(100))
    mfa_used: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timing
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Security metadata
    risk_score: Mapped[int | None] = mapped_column(Integer)  # 0-100
    suspicious_indicators: Mapped[str | None] = mapped_column(Text)  # JSON array

def init_db():
    """Initialize both SQLite and TimescaleDB databases."""
    # Initialize SQLite database for application data
    Base.metadata.create_all(bind=engine)
    logger.info("SQLite database initialized")
    
    # Initialize TimescaleDB for time-series data (if configured)
    timescale_engine_instance = init_timescale_db()
    
    if timescale_engine_instance is not None:
        try:
            # Import TimescaleDB models
            from .sentiment_models import TimescaleBase
            
            # Create tables (but not hypertables yet - that's done in timescale_setup.py)
            TimescaleBase.metadata.create_all(bind=timescale_engine_instance)
            logger.info("TimescaleDB tables created successfully")
            
        except ImportError as e:
            logger.warning(f"Could not import TimescaleDB models: {e}")
        except Exception as e:
            logger.error(f"Error creating TimescaleDB tables: {e}")
    
    return engine, timescale_engine_instance
