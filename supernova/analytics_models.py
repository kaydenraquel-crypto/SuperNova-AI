"""
Advanced Analytics Database Models for SuperNova AI
Comprehensive models for portfolio analytics, performance tracking, and risk management
"""

from sqlalchemy import (
    String, Integer, Float, DateTime, ForeignKey, Text, Boolean, 
    JSON, Numeric, Index, DECIMAL, create_engine
)
from datetime import datetime, date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
import json
from enum import Enum
from decimal import Decimal

from .db import Base

# Enums for type safety
class PerformancePeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class RiskMetricType(str, Enum):
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAXIMUM_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    TRACKING_ERROR = "tracking_error"

class ReportStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

# ================================
# PORTFOLIO MODELS
# ================================

class Portfolio(Base):
    """Enhanced portfolio model for comprehensive tracking"""
    __tablename__ = "portfolios"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"), nullable=True)
    
    # Basic information
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    
    # Portfolio configuration
    initial_value: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), default=0)
    benchmark_symbol: Mapped[str] = mapped_column(String(20), nullable=True)  # e.g., SPY
    risk_free_rate: Mapped[float] = mapped_column(Float, default=0.02)  # 2% default
    
    # Status and metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_paper_trading: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    positions: Mapped[List["Position"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    performance_records: Mapped[List["PerformanceRecord"]] = relationship(back_populates="portfolio")
    risk_metrics: Mapped[List["RiskMetric"]] = relationship(back_populates="portfolio")
    transactions: Mapped[List["Transaction"]] = relationship(back_populates="portfolio")
    
    # Collaboration relationships (will be added when collaboration_models is imported)
    shared_portfolios: Mapped[List["SharedPortfolio"]] = relationship(back_populates="portfolio")
    comments: Mapped[List["PortfolioComment"]] = relationship(
        "PortfolioComment",
        primaryjoin="and_(Portfolio.id == foreign(PortfolioComment.resource_id), PortfolioComment.resource_type == 'portfolio')",
        viewonly=True
    )

class Position(Base):
    """Individual position within a portfolio"""
    __tablename__ = "positions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    
    # Position details
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    average_cost: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    current_price: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), nullable=True)
    market_value: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), nullable=True)
    
    # P&L tracking
    unrealized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), default=0)
    
    # Position metadata
    sector: Mapped[str] = mapped_column(String(50), nullable=True)
    industry: Mapped[str] = mapped_column(String(50), nullable=True)
    allocation_percent: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")
    transactions: Mapped[List["Transaction"]] = relationship(back_populates="position")

class Transaction(Base):
    """Individual transaction record"""
    __tablename__ = "transactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id"), nullable=True, index=True)
    
    # Transaction details
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    transaction_type: Mapped[str] = mapped_column(String(10))  # BUY, SELL, DIVIDEND, etc.
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    price: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    
    # Financial details
    gross_amount: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    fees: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), default=0)
    taxes: Mapped[Decimal] = mapped_column(DECIMAL(18, 6), default=0)
    net_amount: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    
    # Metadata
    exchange: Mapped[str] = mapped_column(String(10), nullable=True)
    order_id: Mapped[str] = mapped_column(String(50), nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Timestamp
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="transactions")
    position: Mapped["Position"] = relationship(back_populates="transactions")

# ================================
# PERFORMANCE TRACKING MODELS
# ================================

class PerformanceRecord(Base):
    """Portfolio performance snapshots over time"""
    __tablename__ = "performance_records"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    
    # Time period
    period_type: Mapped[str] = mapped_column(String(20), index=True)  # daily, weekly, monthly
    period_date: Mapped[date] = mapped_column(DateTime(timezone=True), index=True)
    
    # Portfolio values
    total_value: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    cash_value: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    equity_value: Mapped[Decimal] = mapped_column(DECIMAL(18, 6))
    
    # Performance metrics
    total_return: Mapped[float] = mapped_column(Float)  # Period return
    cumulative_return: Mapped[float] = mapped_column(Float)  # Since inception
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=True)
    excess_return: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Risk metrics for the period
    volatility: Mapped[float] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Additional metrics
    turnover_rate: Mapped[float] = mapped_column(Float, nullable=True)
    num_positions: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="performance_records")

class RiskMetric(Base):
    """Risk metrics calculated for portfolios"""
    __tablename__ = "risk_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    
    # Metric identification
    metric_type: Mapped[str] = mapped_column(String(50), index=True)
    metric_name: Mapped[str] = mapped_column(String(100))
    
    # Metric values
    value: Mapped[float] = mapped_column(Float)
    confidence_level: Mapped[float] = mapped_column(Float, nullable=True)  # For VaR, etc.
    time_horizon_days: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Calculation metadata
    calculation_method: Mapped[str] = mapped_column(String(50), nullable=True)
    data_points_used: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Time context
    calculation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="risk_metrics")

# ================================
# MARKET DATA & SENTIMENT MODELS
# ================================

class MarketSentiment(Base):
    """Market sentiment analysis results"""
    __tablename__ = "market_sentiment"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Target identification
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=True)
    sector: Mapped[str] = mapped_column(String(50), index=True, nullable=True)
    market: Mapped[str] = mapped_column(String(20), index=True, nullable=True)  # overall, sector-specific
    
    # Sentiment scores
    sentiment_score: Mapped[float] = mapped_column(Float)  # -1 to 1
    confidence_score: Mapped[float] = mapped_column(Float)  # 0 to 1
    volume_weighted_score: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Source breakdown
    social_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    news_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    analyst_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Data sources count
    total_mentions: Mapped[int] = mapped_column(Integer, default=0)
    social_mentions: Mapped[int] = mapped_column(Integer, default=0)
    news_articles: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class TechnicalIndicator(Base):
    """Technical indicator calculations"""
    __tablename__ = "technical_indicators"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Asset identification
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    timeframe: Mapped[str] = mapped_column(String(10), index=True)  # 1m, 5m, 1h, 1d, etc.
    
    # Indicator identification
    indicator_name: Mapped[str] = mapped_column(String(50), index=True)
    indicator_params: Mapped[str] = mapped_column(Text, nullable=True)  # JSON params
    
    # Indicator values
    value: Mapped[float] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(20), nullable=True)  # BUY, SELL, NEUTRAL
    signal_strength: Mapped[float] = mapped_column(Float, nullable=True)  # 0 to 1
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

# ================================
# BACKTESTING & ANALYSIS MODELS
# ================================

class BacktestAnalysis(Base):
    """Enhanced backtest analysis results"""
    __tablename__ = "backtest_analyses"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    backtest_result_id: Mapped[int] = mapped_column(ForeignKey("backtest_results.id"), index=True)
    
    # Analysis metadata
    analysis_type: Mapped[str] = mapped_column(String(50))  # performance, risk, drawdown, etc.
    analysis_name: Mapped[str] = mapped_column(String(100))
    
    # Statistical significance
    p_value: Mapped[float] = mapped_column(Float, nullable=True)
    confidence_interval_lower: Mapped[float] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Detailed results (JSON)
    results: Mapped[str] = mapped_column(Text)  # JSON serialized results
    
    # Visualization data
    chart_data: Mapped[str] = mapped_column(Text, nullable=True)  # JSON chart config
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

# ================================
# REPORTING MODELS
# ================================

class AnalyticsReport(Base):
    """Generated analytics reports"""
    __tablename__ = "analytics_reports"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), nullable=True, index=True)
    
    # Report details
    report_type: Mapped[str] = mapped_column(String(50))  # performance, risk, allocation, etc.
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Report parameters
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    parameters: Mapped[str] = mapped_column(Text, nullable=True)  # JSON parameters
    
    # File information
    file_format: Mapped[str] = mapped_column(String(10))  # PDF, XLSX, CSV
    file_path: Mapped[str] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Status tracking
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Timestamps
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

# ================================
# INDEXES FOR PERFORMANCE
# ================================

# Performance-critical indexes
Index('idx_portfolio_performance', 'portfolio_id', 'period_date')
Index('idx_portfolio_risk_metrics', 'portfolio_id', 'calculation_date')
Index('idx_market_sentiment_symbol_time', 'symbol', 'timestamp')
Index('idx_technical_indicators_symbol_time', 'symbol', 'timeframe', 'timestamp')
Index('idx_transactions_portfolio_time', 'portfolio_id', 'executed_at')
Index('idx_positions_portfolio_active', 'portfolio_id', 'closed_at')

# Composite indexes for common queries
Index('idx_performance_period_lookup', 'portfolio_id', 'period_type', 'period_date')
Index('idx_risk_metric_lookup', 'portfolio_id', 'metric_type', 'calculation_date')
Index('idx_sentiment_time_series', 'symbol', 'timestamp')