from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from typing import List, Optional, Literal, Dict, Any, Union, Tuple
from datetime import datetime
from decimal import Decimal
import re
from enum import Enum

# Import security validators
from .input_validation import (
    validate_sql_safe, validate_xss_safe, validate_financial_symbol,
    validate_decimal_amount, input_validator
)

# ====================================
# AUTHENTICATION SCHEMAS
# ====================================

class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    USER = "user"
    VIEWER = "viewer"

class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")
    remember_me: bool = Field(default=False, description="Remember user session")
    mfa_token: Optional[str] = Field(None, description="MFA token if required")

class RegisterRequest(BaseModel):
    """User registration request"""
    name: str = Field(..., min_length=1, max_length=100, description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=12, description="Password")
    confirm_password: str = Field(..., description="Password confirmation")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: dict = Field(..., description="User information")

class RefreshTokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")

class MFASetupRequest(BaseModel):
    """MFA setup request"""
    password: str = Field(..., description="Current password for verification")

class MFASetupResponse(BaseModel):
    """MFA setup response"""
    secret: str = Field(..., description="MFA secret")
    qr_code: str = Field(..., description="Base64 encoded QR code")
    backup_codes: List[str] = Field(..., description="Backup codes")

class MFAVerifyRequest(BaseModel):
    """MFA verification request"""
    token: str = Field(..., description="MFA token")
    backup_code: Optional[str] = Field(None, description="MFA backup code")

class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=12, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr = Field(..., description="User email address")

class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation"""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=12, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class LogoutRequest(BaseModel):
    """Logout request"""
    all_devices: bool = Field(default=False, description="Logout from all devices")

class UserProfile(BaseModel):
    """User profile information"""
    id: int = Field(..., description="User ID")
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Account status")
    mfa_enabled: bool = Field(..., description="MFA status")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str = Field(..., description="Session ID")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    is_current: bool = Field(..., description="Current session indicator")

class UserSessionsResponse(BaseModel):
    """User sessions response"""
    sessions: List[SessionInfo] = Field(..., description="Active sessions")
    total: int = Field(..., description="Total session count")

class APIKeyRequest(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=1, max_length=100, description="Key name")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")

class APIKeyResponse(BaseModel):
    """API key response"""
    key_id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    key: str = Field(..., description="API key (only shown once)")
    created_at: datetime = Field(..., description="Creation date")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")

class APIKeyInfo(BaseModel):
    """API key information (without actual key)"""
    key_id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    created_at: datetime = Field(..., description="Creation date")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    last_used: Optional[datetime] = Field(None, description="Last used date")
    is_active: bool = Field(..., description="Key status")

class SecurityEvent(BaseModel):
    """Security event information"""
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    details: Dict[str, Any] = Field(..., description="Event details")

class SecurityLog(BaseModel):
    """Security log response"""
    events: List[SecurityEvent] = Field(..., description="Security events")
    total: int = Field(..., description="Total event count")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Events per page")

# ====================================
# EXISTING SCHEMAS
# ====================================

class IntakeRequest(BaseModel):
    name: str
    email: Optional[str] = None
    income: Optional[float] = None
    expenses: Optional[float] = None
    assets: Optional[float] = None
    debts: Optional[float] = None
    time_horizon_yrs: int = 5
    objectives: str = "growth"
    constraints: str = ""
    risk_questions: List[int] = Field(default_factory=list)  # responses 0-4

class ProfileOut(BaseModel):
    profile_id: int
    risk_score: int

class OHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class AdviceRequest(BaseModel):
    profile_id: int
    symbol: str
    asset_class: Literal["stock","crypto","fx","futures","option"] = "stock"
    timeframe: str = "1h"
    bars: List[OHLCVBar]  # pass in recent history to avoid live data dependencies
    sentiment_hint: Optional[float] = None  # -1..+1 precomputed if available
    strategy_template: Optional[str] = None
    params: dict = {}

class AdviceOut(BaseModel):
    symbol: str
    timeframe: str
    action: Literal["buy","sell","hold","reduce","avoid"]
    confidence: float
    rationale: str
    key_indicators: dict
    risk_notes: str

class WatchlistRequest(BaseModel):
    profile_id: int
    symbols: List[str]
    asset_class: str = "stock"

class AlertOut(BaseModel):
    id: int
    symbol: str
    message: str
    triggered_at: str

class BacktestRequest(BaseModel):
    strategy_template: str
    params: dict
    symbol: str
    timeframe: str
    bars: List[OHLCVBar]
    use_vectorbt: bool = True  # Default to VectorBT if available
    start_cash: float = 10000.0
    fees: float = 0.001  # 0.1% transaction fees
    slippage: float = 0.001  # 0.1% slippage

class BacktestOut(BaseModel):
    symbol: str
    timeframe: str
    metrics: dict
    notes: str = ""
    engine: str = "Legacy"  # Indicates which backtesting engine was used

# ================================
# TIMESCALEDB SENTIMENT SCHEMAS
# ================================

class SentimentDataPoint(BaseModel):
    """Individual sentiment data point"""
    symbol: str
    timestamp: datetime
    overall_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment score from -1 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level from 0 to 1")
    social_momentum: Optional[float] = Field(None, description="Social media momentum indicator")
    news_sentiment: Optional[float] = Field(None, description="News-specific sentiment score")
    twitter_sentiment: Optional[float] = Field(None, description="Twitter-specific sentiment score")
    reddit_sentiment: Optional[float] = Field(None, description="Reddit-specific sentiment score")
    source_counts: Optional[Dict[str, Any]] = Field(None, description="Count of data sources used")
    market_regime: Optional[str] = Field(None, description="Market regime context")
    figure_influence: float = Field(0.0, description="Influence of public figures")
    contrarian_indicator: float = Field(0.0, description="Contrarian signal strength")
    regime_adjusted_score: Optional[float] = Field(None, description="Market regime adjusted score")
    total_data_points: int = Field(0, description="Number of data points used")

class SentimentHistoryRequest(BaseModel):
    """Request parameters for historical sentiment data"""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="Stock symbols to query")
    start_date: datetime = Field(..., description="Start date for historical data")
    end_date: datetime = Field(..., description="End date for historical data")
    interval: Optional[Literal["raw", "1h", "6h", "1d", "1w"]] = Field(
        "raw", 
        description="Data aggregation interval"
    )
    min_confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum confidence threshold for filtering results"
    )
    market_regime: Optional[str] = Field(None, description="Filter by market regime")
    limit: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum number of records to return")
    offset: Optional[int] = Field(0, ge=0, description="Number of records to skip for pagination")

class SentimentHistoryResponse(BaseModel):
    """Response for historical sentiment data queries"""
    symbols: List[str] = Field(..., description="Symbols included in response")
    start_date: datetime = Field(..., description="Actual start date of returned data")
    end_date: datetime = Field(..., description="Actual end date of returned data")
    interval: str = Field(..., description="Data aggregation interval used")
    total_records: int = Field(..., description="Total number of records in response")
    data_points: List[SentimentDataPoint] = Field(..., description="Historical sentiment data points")
    
    # Pagination metadata
    has_more: bool = Field(False, description="Whether more data is available")
    next_offset: Optional[int] = Field(None, description="Offset for next page of results")
    
    # Query performance metadata
    query_duration_ms: Optional[float] = Field(None, description="Query execution time in milliseconds")
    
    # Data quality metadata
    avg_confidence: Optional[float] = Field(None, description="Average confidence of returned data")
    source_distribution: Optional[Dict[str, int]] = Field(None, description="Distribution of data sources")

class SentimentSummaryStats(BaseModel):
    """Summary statistics for sentiment data"""
    symbol: str
    period_start: datetime
    period_end: datetime
    total_data_points: int
    avg_score: float
    min_score: float
    max_score: float
    avg_confidence: float
    score_volatility: Optional[float] = Field(None, description="Standard deviation of scores")
    momentum_trend: Optional[float] = Field(None, description="Overall momentum trend")
    dominant_regime: Optional[str] = Field(None, description="Most common market regime")
    source_breakdown: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Statistics by source")

class SentimentAggregateRequest(BaseModel):
    """Request for aggregated sentiment data"""
    symbols: List[str] = Field(..., min_items=1, max_items=20, description="Stock symbols to aggregate")
    start_date: datetime = Field(..., description="Start date for aggregation")
    end_date: datetime = Field(..., description="End date for aggregation")
    aggregation: Literal["1h", "6h", "1d", "1w"] = Field("1d", description="Aggregation interval")
    include_stats: bool = Field(True, description="Whether to include summary statistics")

class SentimentAggregateResponse(BaseModel):
    """Response for aggregated sentiment data"""
    symbols: List[str]
    aggregation_interval: str
    period_start: datetime
    period_end: datetime
    aggregated_data: List[Dict[str, Any]] = Field(..., description="Aggregated sentiment data points")
    summary_stats: Optional[List[SentimentSummaryStats]] = Field(None, description="Summary statistics by symbol")
    
    # Metadata
    total_raw_points: int = Field(0, description="Total raw data points used in aggregation")
    query_duration_ms: Optional[float] = Field(None, description="Query execution time")

class SentimentAlertRequest(BaseModel):
    """Request to create sentiment-based alerts"""
    symbol: str = Field(..., description="Stock symbol to monitor")
    alert_type: Literal["threshold_breach", "volatility_spike", "momentum_change", "confidence_drop", "source_divergence"] = Field(
        ..., 
        description="Type of alert to create"
    )
    threshold_value: float = Field(..., description="Threshold value for the alert")
    comparison: Literal["above", "below", "equals"] = Field("above", description="Comparison operation")
    severity: Literal["low", "medium", "high", "critical"] = Field("medium", description="Alert severity level")
    description: Optional[str] = Field(None, description="Optional alert description")

class SentimentAlertResponse(BaseModel):
    """Response for sentiment alert operations"""
    alert_id: int
    symbol: str
    alert_type: str
    threshold_value: float
    status: Literal["active", "triggered", "resolved", "disabled"]
    created_at: datetime
    triggered_at: Optional[datetime] = None
    message: Optional[str] = None

class TimescaleHealthResponse(BaseModel):
    """TimescaleDB health check response"""
    status: Literal["healthy", "degraded", "unhealthy", "unavailable"]
    connection_status: str
    database_version: Optional[str] = None
    timescale_version: Optional[str] = None
    
    # Performance metrics
    active_connections: Optional[int] = None
    total_sentiment_records: Optional[int] = None
    latest_data_timestamp: Optional[datetime] = None
    oldest_data_timestamp: Optional[datetime] = None
    
    # Storage metrics
    database_size: Optional[str] = None
    compressed_chunks: Optional[int] = None
    uncompressed_chunks: Optional[int] = None
    
    # Health check metadata
    check_timestamp: datetime
    response_time_ms: Optional[float] = None
    
    # Issues (if any)
    warnings: Optional[List[str]] = Field(None, description="Non-critical issues")
    errors: Optional[List[str]] = Field(None, description="Critical issues")

class BulkSentimentInsertRequest(BaseModel):
    """Request for bulk sentiment data insertion"""
    data_points: List[SentimentDataPoint] = Field(
        ..., 
        min_items=1, 
        max_items=10000, 
        description="Sentiment data points to insert"
    )
    on_conflict: Literal["update", "ignore", "error"] = Field(
        "update", 
        description="How to handle conflicts with existing data"
    )
    validate_data: bool = Field(True, description="Whether to validate data before insertion")

class BulkSentimentInsertResponse(BaseModel):
    """Response for bulk sentiment data insertion"""
    total_submitted: int
    total_inserted: int
    total_updated: int
    total_skipped: int
    total_errors: int
    
    processing_duration_ms: float
    records_per_second: float
    
    # Error details (if any)
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed error information")
    
    # Success summary
    symbols_affected: List[str] = Field(..., description="Symbols that were updated")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range of inserted data")

# ================================
# OPTIMIZATION SCHEMAS
# ================================

class OptimizationRequest(BaseModel):
    """Request for strategy optimization"""
    symbol: str = Field(..., description="Stock symbol to optimize")
    strategy_template: str = Field(..., description="Strategy template to optimize")
    bars: List[OHLCVBar] = Field(..., min_items=100, description="Historical OHLCV data")
    
    # Optimization configuration
    n_trials: int = Field(100, ge=10, le=1000, description="Number of optimization trials")
    primary_objective: str = Field("sharpe_ratio", description="Primary optimization objective")
    secondary_objectives: List[str] = Field(default_factory=list, description="Secondary objectives for multi-objective optimization")
    
    # Parameter space override
    parameter_space: Optional[Dict[str, Any]] = Field(None, description="Custom parameter space definition")
    
    # Validation settings
    walk_forward: bool = Field(False, description="Enable walk-forward optimization")
    validation_splits: int = Field(3, ge=2, le=10, description="Number of validation splits")
    
    # Risk constraints
    max_drawdown_limit: float = Field(0.25, ge=0.05, le=0.50, description="Maximum allowed drawdown")
    min_sharpe_ratio: float = Field(0.5, ge=0.0, le=3.0, description="Minimum required Sharpe ratio")
    min_win_rate: float = Field(0.35, ge=0.1, le=0.9, description="Minimum required win rate")
    
    # Transaction costs
    include_transaction_costs: bool = Field(True, description="Include transaction costs in optimization")
    commission: float = Field(0.001, ge=0.0, le=0.01, description="Commission rate")
    slippage: float = Field(0.001, ge=0.0, le=0.01, description="Slippage rate")
    
    # Study settings
    study_name: Optional[str] = Field(None, description="Custom study name")
    resume_study: bool = Field(True, description="Resume existing study if found")

class OptimizationStudy(BaseModel):
    """Optimization study metadata and status"""
    study_id: str = Field(..., description="Unique study identifier")
    study_name: str = Field(..., description="Human-readable study name")
    symbol: str = Field(..., description="Symbol being optimized")
    strategy_template: str = Field(..., description="Strategy template")
    
    # Study configuration
    n_trials: int = Field(..., description="Target number of trials")
    primary_objective: str = Field(..., description="Primary objective")
    secondary_objectives: List[str] = Field(default_factory=list, description="Secondary objectives")
    
    # Study status
    status: Literal["created", "running", "completed", "failed", "paused"] = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Completion progress (0-1)")
    
    # Trial statistics
    n_complete_trials: int = Field(0, description="Number of completed trials")
    n_pruned_trials: int = Field(0, description="Number of pruned trials")
    n_failed_trials: int = Field(0, description="Number of failed trials")
    
    # Performance
    best_value: Optional[float] = Field(None, description="Best objective value found")
    best_params: Optional[Dict[str, Any]] = Field(None, description="Best parameters found")
    best_trial_number: Optional[int] = Field(None, description="Best trial number")
    
    # Timing
    created_at: datetime = Field(..., description="Study creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Study start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Study completion timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Metadata
    user_attributes: Dict[str, Any] = Field(default_factory=dict, description="User-defined attributes")
    system_attributes: Dict[str, Any] = Field(default_factory=dict, description="System attributes")

class OptimizationTrial(BaseModel):
    """Individual optimization trial results"""
    trial_id: int = Field(..., description="Trial identifier")
    study_id: str = Field(..., description="Parent study identifier")
    
    # Trial configuration
    params: Dict[str, Any] = Field(..., description="Parameter values for this trial")
    
    # Results
    value: Optional[float] = Field(None, description="Primary objective value")
    values: Optional[List[float]] = Field(None, description="Multi-objective values")
    
    # Status and timing
    state: Literal["running", "complete", "pruned", "failed"] = Field(..., description="Trial state")
    datetime_start: Optional[datetime] = Field(None, description="Trial start time")
    datetime_complete: Optional[datetime] = Field(None, description="Trial completion time")
    duration: Optional[float] = Field(None, description="Trial duration in seconds")
    
    # Detailed metrics
    metrics: Optional[Dict[str, float]] = Field(None, description="Detailed backtesting metrics")
    
    # Intermediate values (for pruning)
    intermediate_values: List[Tuple[int, float]] = Field(default_factory=list, description="Intermediate objective values")
    
    # Metadata
    user_attrs: Dict[str, Any] = Field(default_factory=dict, description="User attributes")
    system_attrs: Dict[str, Any] = Field(default_factory=dict, description="System attributes")

class OptimizationResults(BaseModel):
    """Comprehensive optimization results"""
    study_id: str = Field(..., description="Study identifier")
    study_name: str = Field(..., description="Study name")
    symbol: str = Field(..., description="Optimized symbol")
    strategy_template: str = Field(..., description="Strategy template")
    
    # Best results
    best_params: Dict[str, Any] = Field(..., description="Best parameter combination")
    best_value: float = Field(..., description="Best objective value")
    best_trial_number: int = Field(..., description="Best trial number")
    
    # Comprehensive metrics
    best_metrics: Dict[str, float] = Field(..., description="Detailed metrics for best trial")
    validation_metrics: Optional[Dict[str, float]] = Field(None, description="Out-of-sample validation metrics")
    
    # Study statistics
    total_trials: int = Field(..., description="Total number of trials")
    completed_trials: int = Field(..., description="Number of completed trials")
    pruned_trials: int = Field(..., description="Number of pruned trials")
    failed_trials: int = Field(..., description="Number of failed trials")
    
    # Performance analysis
    parameter_importance: Optional[Dict[str, float]] = Field(None, description="Parameter importance scores")
    convergence_info: Optional[Dict[str, Any]] = Field(None, description="Convergence analysis")
    
    # Multi-objective results
    pareto_front: Optional[List[Dict[str, Any]]] = Field(None, description="Pareto front for multi-objective optimization")
    
    # Timing and efficiency
    optimization_duration: float = Field(..., description="Total optimization time in seconds")
    trials_per_minute: float = Field(..., description="Optimization speed")
    
    # Quality metrics
    consensus_strength: Optional[float] = Field(None, description="Parameter consensus strength (walk-forward)")
    stability_score: Optional[float] = Field(None, description="Result stability across validation periods")
    
    # Metadata
    completed_at: datetime = Field(..., description="Optimization completion timestamp")
    configuration: Dict[str, Any] = Field(..., description="Optimization configuration used")

class OptimizationProgress(BaseModel):
    """Real-time optimization progress updates"""
    study_id: str = Field(..., description="Study identifier")
    current_trial: int = Field(..., description="Current trial number")
    total_trials: int = Field(..., description="Total planned trials")
    
    # Progress metrics
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Completion percentage")
    elapsed_time: float = Field(..., description="Elapsed time in seconds")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")
    
    # Current best
    current_best_value: Optional[float] = Field(None, description="Current best objective value")
    current_best_params: Optional[Dict[str, Any]] = Field(None, description="Current best parameters")
    current_best_trial: Optional[int] = Field(None, description="Current best trial number")
    
    # Recent performance
    recent_trials: List[OptimizationTrial] = Field(default_factory=list, description="Recent trial results")
    improvement_streak: int = Field(0, description="Consecutive trials with improvement")
    
    # Status
    status: Literal["initializing", "running", "paused", "completed", "failed"] = Field(..., description="Current status")
    message: Optional[str] = Field(None, description="Status message")
    
    # Real-time metrics
    trials_per_minute: float = Field(0.0, description="Current optimization speed")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    
    # Update timestamp
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

class WatchlistOptimizationRequest(BaseModel):
    """Request for optimizing strategies across watchlist symbols"""
    profile_id: int = Field(..., description="Profile ID for watchlist access")
    symbols: Optional[List[str]] = Field(None, description="Specific symbols (if not using full watchlist)")
    strategy_templates: List[str] = Field(..., min_items=1, description="Strategy templates to optimize")
    
    # Data requirements
    data_timeframe: str = Field("1h", description="Timeframe for historical data")
    lookback_days: int = Field(365, ge=30, le=1095, description="Days of historical data to use")
    
    # Optimization configuration
    n_trials_per_symbol: int = Field(50, ge=10, le=500, description="Trials per symbol-strategy combination")
    parallel_jobs: int = Field(4, ge=1, le=16, description="Number of parallel optimization jobs")
    
    # Risk management
    portfolio_risk_limit: float = Field(0.20, ge=0.05, le=0.50, description="Portfolio-level risk limit")
    correlation_threshold: float = Field(0.7, ge=0.3, le=0.95, description="Maximum correlation threshold")
    
    # Scheduling
    schedule_overnight: bool = Field(False, description="Schedule for overnight execution")
    priority: Literal["low", "normal", "high"] = Field("normal", description="Optimization priority")
    
    # Notifications
    notify_on_completion: bool = Field(True, description="Send notification when complete")
    email_results: bool = Field(False, description="Email detailed results")

class WatchlistOptimizationResponse(BaseModel):
    """Response for watchlist optimization request"""
    optimization_id: str = Field(..., description="Unique optimization batch identifier")
    profile_id: int = Field(..., description="Profile ID")
    symbols: List[str] = Field(..., description="Symbols being optimized")
    strategy_templates: List[str] = Field(..., description="Strategy templates")
    
    # Status
    status: Literal["queued", "running", "completed", "partially_completed", "failed"] = Field(..., description="Overall status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Overall progress")
    
    # Individual study tracking
    study_statuses: Dict[str, OptimizationStudy] = Field(..., description="Status of individual studies")
    
    # Estimated timing
    estimated_duration_hours: float = Field(..., description="Estimated total duration in hours")
    estimated_completion: datetime = Field(..., description="Estimated completion time")
    
    # Resource usage
    parallel_jobs_used: int = Field(..., description="Number of parallel jobs allocated")
    priority_level: str = Field(..., description="Assigned priority level")
    
    # Scheduling info
    created_at: datetime = Field(..., description="Request creation time")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    scheduled_for: Optional[datetime] = Field(None, description="Scheduled execution time")

class OptimizationDashboardData(BaseModel):
    """Data structure for optimization dashboard"""
    # Summary statistics
    total_studies: int = Field(..., description="Total number of studies")
    active_studies: int = Field(..., description="Currently active studies")
    completed_studies: int = Field(..., description="Completed studies")
    
    # Recent activity
    recent_completions: List[OptimizationResults] = Field(..., description="Recently completed optimizations")
    active_optimizations: List[OptimizationProgress] = Field(..., description="Currently running optimizations")
    
    # Performance insights
    top_performers: List[Dict[str, Any]] = Field(..., description="Best performing strategy-symbol combinations")
    parameter_insights: Dict[str, Any] = Field(..., description="Parameter importance insights across studies")
    
    # Resource utilization
    compute_usage: Dict[str, float] = Field(..., description="Compute resource utilization")
    storage_usage: Dict[str, float] = Field(..., description="Storage usage statistics")
    
    # Trends
    optimization_trends: Dict[str, List[float]] = Field(..., description="Performance trends over time")
    popular_strategies: Dict[str, int] = Field(..., description="Most optimized strategies")
    
    # Generated timestamp
    generated_at: datetime = Field(default_factory=datetime.now, description="Dashboard generation timestamp")
    refresh_interval: int = Field(30, description="Recommended refresh interval in seconds")

class OptimizationComparison(BaseModel):
    """Compare multiple optimization results"""
    comparison_id: str = Field(..., description="Unique comparison identifier")
    study_ids: List[str] = Field(..., min_items=2, description="Studies being compared")
    
    # Comparison results
    parameter_comparison: Dict[str, Dict[str, Any]] = Field(..., description="Parameter value comparison")
    metric_comparison: Dict[str, Dict[str, float]] = Field(..., description="Performance metric comparison")
    
    # Statistical analysis
    significance_tests: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Statistical significance tests")
    correlation_analysis: Optional[Dict[str, float]] = Field(None, description="Parameter correlation analysis")
    
    # Recommendations
    winner: Optional[str] = Field(None, description="Study ID of best performer")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    
    # Metadata
    compared_at: datetime = Field(default_factory=datetime.now, description="Comparison timestamp")
    comparison_criteria: Dict[str, Any] = Field(..., description="Criteria used for comparison")

# ================================
# CHAT API SCHEMAS
# ================================

class ChatMessage(BaseModel):
    """Individual chat message model"""
    id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Chat session identifier")
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    charts: Optional[List[Dict[str, Any]]] = Field(None, description="Chart data for embedding")
    
    # User interaction
    rating: Optional[int] = Field(None, ge=1, le=5, description="User rating for AI messages")
    helpful: Optional[bool] = Field(None, description="User feedback on helpfulness")

class ChatSession(BaseModel):
    """Chat session metadata"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    profile_id: Optional[int] = Field(None, description="Associated user profile ID")
    
    # Session metadata
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(0, description="Total message count")
    
    # Session data
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")
    preview: Optional[str] = Field(None, description="Session preview text")
    
    # Settings
    settings: Optional[Dict[str, Any]] = Field(None, description="User session settings")

class ChatRequest(BaseModel):
    """Incoming chat request"""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    profile_id: Optional[int] = Field(None, description="User profile ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    # Request options
    stream: bool = Field(False, description="Enable streaming response")
    include_suggestions: bool = Field(True, description="Include follow-up suggestions")
    include_charts: bool = Field(True, description="Include chart recommendations")

class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str = Field(..., description="Session identifier")
    message: ChatMessage = Field(..., description="AI response message")
    
    # Additional response data
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Chart data")
    context_updates: Optional[Dict[str, Any]] = Field(None, description="Context updates")
    
    # Metadata
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    model_info: Optional[Dict[str, str]] = Field(None, description="AI model information")

class ChatHistory(BaseModel):
    """Chat history response"""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    total_messages: int = Field(..., description="Total message count")
    
    # Pagination
    has_more: bool = Field(False, description="More messages available")
    next_offset: Optional[int] = Field(None, description="Next page offset")
    
    # Session context
    context: Dict[str, Any] = Field(default_factory=dict, description="Current session context")

class ChatFeedback(BaseModel):
    """User feedback on chat responses"""
    message_id: str = Field(..., description="Message being rated")
    user_id: str = Field(..., description="User providing feedback")
    
    # Feedback data
    rating: int = Field(..., ge=1, le=5, description="Message rating (1-5)")
    helpful: bool = Field(..., description="Whether response was helpful")
    feedback: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")
    
    # Categories
    categories: Optional[List[str]] = Field(None, description="Feedback categories")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")

class ChatContext(BaseModel):
    """User context for chat sessions"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Context data
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    investment_profile: Optional[Dict[str, Any]] = Field(None, description="Investment profile data")
    portfolio_data: Optional[Dict[str, Any]] = Field(None, description="Portfolio information")
    market_interests: List[str] = Field(default_factory=list, description="Market interests/watchlist")
    
    # Conversation context
    recent_topics: List[str] = Field(default_factory=list, description="Recently discussed topics")
    active_symbols: List[str] = Field(default_factory=list, description="Currently discussed symbols")
    conversation_intent: Optional[str] = Field(None, description="Current conversation intent")
    
    # Metadata
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

class ChatSuggestion(BaseModel):
    """Smart chat suggestions"""
    suggestion_id: str = Field(..., description="Unique suggestion identifier")
    text: str = Field(..., description="Suggestion text")
    category: str = Field(..., description="Suggestion category")
    
    # Relevance
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    personalized: bool = Field(False, description="Whether suggestion is personalized")
    
    # Metadata
    triggers: List[str] = Field(default_factory=list, description="Trigger keywords/phrases")
    context_requirements: Optional[Dict[str, Any]] = Field(None, description="Required context")

class ChatSessionList(BaseModel):
    """List of chat sessions"""
    sessions: List[ChatSession] = Field(..., description="Chat sessions")
    total_count: int = Field(..., description="Total session count")
    
    # Pagination
    limit: int = Field(50, description="Items per page")
    offset: int = Field(0, description="Page offset")
    has_more: bool = Field(False, description="More sessions available")

class ChatAnalytics(BaseModel):
    """Chat analytics and insights"""
    user_id: str = Field(..., description="User identifier")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    
    # Usage statistics
    total_sessions: int = Field(0, description="Total chat sessions")
    total_messages: int = Field(0, description="Total messages sent")
    total_ai_responses: int = Field(0, description="Total AI responses")
    avg_session_length: float = Field(0.0, description="Average session length in minutes")
    
    # Interaction patterns
    most_active_hours: List[int] = Field(default_factory=list, description="Most active hours of day")
    popular_topics: List[str] = Field(default_factory=list, description="Most discussed topics")
    frequent_symbols: List[str] = Field(default_factory=list, description="Most queried symbols")
    
    # Quality metrics
    avg_response_rating: Optional[float] = Field(None, description="Average response rating")
    positive_feedback_rate: Optional[float] = Field(None, description="Positive feedback percentage")
    
    # Generated insights
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    recommendations: List[str] = Field(default_factory=list, description="Usage recommendations")

# WebSocket-specific schemas
class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: Literal["chat", "typing", "presence", "system", "notification", "chart_data", "market_update", "voice_message", "file_share", "screen_share"]
    data: Dict[str, Any] = Field(..., description="Message data")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    message_id: Optional[str] = Field(None, description="Unique message ID")

class PresenceUpdate(BaseModel):
    """User presence update"""
    user_id: str = Field(..., description="User identifier")
    status: Literal["online", "away", "busy", "offline"] = Field(..., description="Presence status")
    session_id: Optional[str] = Field(None, description="Active session")
    last_seen: datetime = Field(..., description="Last activity timestamp")
    custom_status: Optional[str] = Field(None, description="Custom status message")

class VoiceMessage(BaseModel):
    """Voice message data"""
    message_id: str = Field(..., description="Message identifier")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Audio data
    audio_url: str = Field(..., description="Audio file URL")
    duration: float = Field(..., description="Audio duration in seconds")
    transcript: Optional[str] = Field(None, description="Audio transcription")
    
    # Processing info
    transcription_confidence: Optional[float] = Field(None, description="Transcription confidence")
    language_detected: Optional[str] = Field(None, description="Detected language")
    
    # Metadata
    timestamp: datetime = Field(..., description="Message timestamp")

class FileShare(BaseModel):
    """File sharing data"""
    file_id: str = Field(..., description="File identifier")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # File metadata
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="MIME type")
    file_url: str = Field(..., description="File access URL")
    
    # Processing info
    upload_status: Literal["uploading", "completed", "failed"] = Field(..., description="Upload status")
    analysis_status: Optional[Literal["pending", "processing", "completed", "failed"]] = Field(None, description="Analysis status")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="File analysis results")
    
    # Metadata
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration timestamp")

class MarketDataUpdate(BaseModel):
    """Real-time market data update"""
    symbol: str = Field(..., description="Stock symbol")
    data_type: Literal["quote", "trade", "level2", "news", "sentiment"] = Field(..., description="Data type")
    
    # Market data
    data: Dict[str, Any] = Field(..., description="Market data payload")
    
    # Metadata
    timestamp: datetime = Field(..., description="Data timestamp")
    source: Optional[str] = Field(None, description="Data source")
    latency_ms: Optional[float] = Field(None, description="Data latency in milliseconds")

class ChartData(BaseModel):
    """Chart data for embedding in chat"""
    chart_id: str = Field(..., description="Chart identifier")
    symbol: str = Field(..., description="Stock symbol")
    chart_type: Literal["candlestick", "line", "bar", "heatmap", "scatter"] = Field(..., description="Chart type")
    timeframe: str = Field(..., description="Chart timeframe")
    
    # Chart configuration
    indicators: List[str] = Field(default_factory=list, description="Technical indicators")
    overlays: List[str] = Field(default_factory=list, description="Chart overlays")
    
    # Data
    data: Dict[str, Any] = Field(..., description="Chart data")
    layout: Dict[str, Any] = Field(..., description="Chart layout configuration")
    
    # Metadata
    generated_at: datetime = Field(..., description="Chart generation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Chart data expiration")

# Advanced chat features
class ConversationSummary(BaseModel):
    """Conversation summary"""
    session_id: str = Field(..., description="Session identifier")
    summary: str = Field(..., description="Conversation summary")
    key_topics: List[str] = Field(..., description="Key topics discussed")
    symbols_mentioned: List[str] = Field(..., description="Symbols mentioned")
    recommendations_made: List[str] = Field(..., description="Recommendations provided")
    follow_up_items: List[str] = Field(..., description="Follow-up action items")
    
    # Metadata
    generated_at: datetime = Field(..., description="Summary generation timestamp")
    message_count: int = Field(..., description="Number of messages summarized")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Summary confidence")

class ChatNotification(BaseModel):
    """Chat notification"""
    notification_id: str = Field(..., description="Notification identifier")
    user_id: str = Field(..., description="Target user")
    type: Literal["message", "mention", "market_alert", "system", "reminder"] = Field(..., description="Notification type")
    
    # Notification content
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    action_url: Optional[str] = Field(None, description="Action URL")
    
    # Delivery options
    channels: List[Literal["websocket", "email", "push", "sms"]] = Field(..., description="Delivery channels")
    priority: Literal["low", "normal", "high", "urgent"] = Field("normal", description="Notification priority")
    
    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    read_at: Optional[datetime] = Field(None, description="Read timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

# ================================
# CONVERSATIONAL AGENT SCHEMAS
# ================================

class ConversationRole(str, Enum):
    """Conversation participant roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class AgentPersonality(str, Enum):
    """Agent personality types"""
    CONSERVATIVE_ADVISOR = "conservative_advisor"
    AGGRESSIVE_TRADER = "aggressive_trader"
    BALANCED_ANALYST = "balanced_analyst"
    EDUCATIONAL_MODE = "educational_mode"
    RESEARCH_ASSISTANT = "research_assistant"

class QueryType(str, Enum):
    """Types of user queries"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    STRATEGY_ADVICE = "strategy_advice"
    MARKET_RESEARCH = "market_research"
    PORTFOLIO_REVIEW = "portfolio_review"
    EDUCATIONAL = "educational"
    GENERAL_CHAT = "general_chat"

class ChatMessage(BaseModel):
    """Individual chat message"""
    id: Optional[int] = None
    conversation_id: str = Field(..., description="Conversation identifier")
    role: ConversationRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")
    context_tags: List[str] = Field(default_factory=list, description="Context tags for the message")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Message importance score")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

class ConversationRequest(BaseModel):
    """Request for conversation/chat"""
    user_input: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID")
    personality: Optional[AgentPersonality] = Field(AgentPersonality.BALANCED_ANALYST, description="Agent personality")
    streaming: bool = Field(False, description="Enable streaming response")
    include_context: bool = Field(True, description="Include conversation context")
    max_context_messages: int = Field(20, ge=1, le=100, description="Maximum context messages")

class ConversationResponse(BaseModel):
    """Response from conversational agent"""
    content: str = Field(..., description="Agent response content")
    conversation_id: str = Field(..., description="Conversation identifier")
    message_id: Optional[int] = Field(None, description="Message ID if persisted")
    query_type: str = Field(..., description="Detected query type")
    personality: str = Field(..., description="Agent personality used")
    response_time_ms: int = Field(..., description="Response generation time in milliseconds")
    tools_used: List[str] = Field(default_factory=list, description="Tools/functions used by agent")
    context_summary: Optional[str] = Field(None, description="Conversation context summary")
    suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    disclaimer: Optional[str] = Field(None, description="Financial disclaimer if applicable")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence")

class ConversationHistoryRequest(BaseModel):
    """Request for conversation history"""
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[int] = Field(None, description="User ID for access control")
    limit: int = Field(50, ge=1, le=500, description="Maximum messages to return")
    include_system: bool = Field(False, description="Include system messages")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    message_types: Optional[List[ConversationRole]] = Field(None, description="Filter by message types")

class ConversationHistoryResponse(BaseModel):
    """Response with conversation history"""
    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total messages in conversation")
    context_summary: Optional[str] = Field(None, description="Conversation summary")
    participants: List[ConversationRole] = Field(..., description="Conversation participants")
    created_at: Optional[datetime] = Field(None, description="Conversation start time")
    last_activity: Optional[datetime] = Field(None, description="Last message time")

class UserPreference(BaseModel):
    """User preference entry"""
    key: str = Field(..., description="Preference key")
    value: Any = Field(..., description="Preference value")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in preference")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")

class UserContextData(BaseModel):
    """User context and preferences"""
    user_id: int = Field(..., description="User identifier")
    preferences: Dict[str, UserPreference] = Field(default_factory=dict, description="User preferences")
    financial_profile: Dict[str, Any] = Field(default_factory=dict, description="Financial profile data")
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict, description="Learned interaction patterns")
    learned_preferences: Dict[str, Any] = Field(default_factory=dict, description="System-learned preferences")
    personality_affinity: Dict[AgentPersonality, float] = Field(default_factory=dict, description="Personality preferences")

class ConversationSummaryRequest(BaseModel):
    """Request for conversation summary"""
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[int] = Field(None, description="User ID for access control")
    summary_type: Literal["brief", "detailed", "topics", "decisions"] = Field("brief", description="Type of summary")
    include_insights: bool = Field(True, description="Include insights and patterns")
    time_window_hours: Optional[int] = Field(None, ge=1, le=168, description="Time window for summary")

class ConversationSummary(BaseModel):
    """Conversation summary response"""
    conversation_id: str = Field(..., description="Conversation identifier")
    summary_type: str = Field(..., description="Type of summary generated")
    summary: str = Field(..., description="Generated summary")
    key_topics: List[str] = Field(default_factory=list, description="Main discussion topics")
    financial_symbols: List[str] = Field(default_factory=list, description="Financial symbols discussed")
    decisions_made: List[str] = Field(default_factory=list, description="Decisions or recommendations")
    insights: Optional[Dict[str, Any]] = Field(None, description="Additional insights")
    message_count: int = Field(..., description="Number of messages summarized")
    time_span: Optional[Dict[str, datetime]] = Field(None, description="Time span of summarized messages")
    generated_at: datetime = Field(default_factory=datetime.now, description="Summary generation time")

class AgentPersonalityRequest(BaseModel):
    """Request to switch agent personality"""
    personality: AgentPersonality = Field(..., description="Target personality")
    conversation_id: Optional[str] = Field(None, description="Conversation to apply personality to")
    reason: Optional[str] = Field(None, description="Reason for personality change")

class AgentStatsResponse(BaseModel):
    """Agent performance statistics"""
    personality: str = Field(..., description="Current agent personality")
    llm_available: bool = Field(..., description="LLM availability status")
    agent_available: bool = Field(..., description="Agent availability status")
    tools_count: int = Field(..., description="Number of available tools")
    total_queries: int = Field(..., description="Total queries processed")
    query_breakdown: Dict[str, int] = Field(..., description="Queries by type")
    avg_response_time_ms: int = Field(..., description="Average response time in milliseconds")
    error_counts: Dict[str, int] = Field(default_factory=dict, description="Error counts by type")
    memory_stats: Dict[str, Any] = Field(..., description="Memory usage statistics")
    uptime_seconds: Optional[float] = Field(None, description="Agent uptime in seconds")

class MemoryOptimizationRequest(BaseModel):
    """Request for memory optimization"""
    conversation_id: Optional[str] = Field(None, description="Specific conversation to optimize")
    user_id: Optional[int] = Field(None, description="User context to optimize")
    optimization_type: Literal["conversation", "user_context", "financial_context", "all"] = Field("all", description="Type of memory to optimize")
    force_cleanup: bool = Field(False, description="Force immediate cleanup")

class MemoryOptimizationResponse(BaseModel):
    """Memory optimization results"""
    optimization_type: str = Field(..., description="Type of optimization performed")
    items_processed: int = Field(..., description="Items processed during optimization")
    items_removed: int = Field(..., description="Items removed during optimization")
    memory_freed_mb: Optional[float] = Field(None, description="Memory freed in megabytes")
    optimization_duration_ms: int = Field(..., description="Optimization duration in milliseconds")
    next_cleanup_due: datetime = Field(..., description="When next cleanup is due")

class StreamingChatRequest(BaseModel):
    """Request for streaming chat response"""
    user_input: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    personality: Optional[AgentPersonality] = Field(AgentPersonality.BALANCED_ANALYST, description="Agent personality")
    stream_tokens: bool = Field(True, description="Stream individual tokens")
    stream_thoughts: bool = Field(False, description="Stream agent reasoning")
    include_metadata: bool = Field(True, description="Include metadata in stream")

class StreamingChatChunk(BaseModel):
    """Streaming chat response chunk"""
    conversation_id: str = Field(..., description="Conversation identifier")
    chunk_type: Literal["token", "thought", "tool_use", "metadata", "complete"] = Field(..., description="Type of chunk")
    content: str = Field(..., description="Chunk content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Chunk timestamp")
    sequence_number: int = Field(..., description="Sequence number in stream")

class ConversationAnalyticsRequest(BaseModel):
    """Request for conversation analytics"""
    conversation_id: Optional[str] = Field(None, description="Specific conversation to analyze")
    user_id: Optional[int] = Field(None, description="User to analyze")
    start_date: Optional[datetime] = Field(None, description="Analysis start date")
    end_date: Optional[datetime] = Field(None, description="Analysis end date")
    analytics_type: List[Literal["engagement", "topics", "sentiment", "performance", "learning"]] = Field(
        default_factory=lambda: ["engagement", "topics"], 
        description="Types of analytics to generate"
    )

class ConversationAnalytics(BaseModel):
    """Conversation analytics results"""
    analysis_period: Dict[str, datetime] = Field(..., description="Analysis time period")
    total_conversations: int = Field(..., description="Total conversations analyzed")
    total_messages: int = Field(..., description="Total messages analyzed")
    
    # Engagement metrics
    avg_messages_per_conversation: Optional[float] = Field(None, description="Average messages per conversation")
    avg_response_time_ms: Optional[int] = Field(None, description="Average agent response time")
    user_engagement_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="User engagement score")
    
    # Topic analysis
    top_topics: Optional[List[Dict[str, Any]]] = Field(None, description="Most discussed topics")
    financial_symbols_discussed: Optional[List[Dict[str, Any]]] = Field(None, description="Financial symbols and frequency")
    query_type_distribution: Optional[Dict[str, int]] = Field(None, description="Distribution of query types")
    
    # Performance metrics
    tool_usage_stats: Optional[Dict[str, int]] = Field(None, description="Tool usage statistics")
    error_rates: Optional[Dict[str, float]] = Field(None, description="Error rates by category")
    personality_usage: Optional[Dict[AgentPersonality, int]] = Field(None, description="Personality usage statistics")
    
    # Learning insights
    user_learning_patterns: Optional[Dict[str, Any]] = Field(None, description="User learning and adaptation patterns")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggested improvements")
    
    generated_at: datetime = Field(default_factory=datetime.now, description="Analytics generation time")
