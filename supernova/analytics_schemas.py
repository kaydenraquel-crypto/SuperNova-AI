"""
Analytics Schemas and Validation for SuperNova AI
Pydantic models for analytics API request/response validation
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

# Import existing validation functions
from .input_validation import (
    validate_sql_safe, validate_xss_safe, validate_financial_symbol,
    validate_decimal_amount, input_validator
)

# ================================
# ENUMS AND CONSTANTS
# ================================

class ReportType(str, Enum):
    """Supported report types"""
    PERFORMANCE = "performance"
    RISK = "risk"
    ALLOCATION = "allocation"
    SUMMARY = "summary"

class ReportFormat(str, Enum):
    """Supported report formats"""
    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"

class PerformancePeriod(str, Enum):
    """Performance analysis periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class RiskMetricType(str, Enum):
    """Risk metric types"""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAXIMUM_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    VOLATILITY = "volatility"

# ================================
# REQUEST SCHEMAS
# ================================

class PortfolioPerformanceRequest(BaseModel):
    """Request schema for portfolio performance analysis"""
    portfolio_id: int = Field(..., description="Portfolio ID", gt=0)
    start_date: Optional[date] = Field(None, description="Start date for analysis")
    end_date: Optional[date] = Field(None, description="End date for analysis")
    benchmark: Optional[str] = Field(None, description="Benchmark symbol", max_length=10)
    period: PerformancePeriod = Field(PerformancePeriod.DAILY, description="Performance period")
    include_attribution: bool = Field(True, description="Include performance attribution")
    
    @validator('benchmark')
    def validate_benchmark_symbol(cls, v):
        """Validate benchmark symbol"""
        if v is not None:
            validate_financial_symbol(v)
            validate_sql_safe(v)
        return v
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate date range"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date:
            if start_date > end_date:
                raise ValueError('Start date must be before end date')
            
            if (end_date - start_date).days > 1825:  # 5 years
                raise ValueError('Date range cannot exceed 5 years')
                
            if start_date > date.today():
                raise ValueError('Start date cannot be in the future')
        
        return values

class RiskAnalysisRequest(BaseModel):
    """Request schema for risk analysis"""
    portfolio_id: int = Field(..., description="Portfolio ID", gt=0)
    confidence_level: float = Field(0.95, description="Confidence level for VaR", ge=0.5, le=0.99)
    time_horizon: int = Field(1, description="Time horizon in days", ge=1, le=252)
    include_stress_testing: bool = Field(False, description="Include stress testing scenarios")
    include_correlation_analysis: bool = Field(True, description="Include correlation analysis")
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        """Validate confidence level is reasonable"""
        if v < 0.5 or v > 0.99:
            raise ValueError('Confidence level must be between 0.5 and 0.99')
        return v

class MarketSentimentRequest(BaseModel):
    """Request schema for market sentiment analysis"""
    symbols: Optional[List[str]] = Field(None, description="List of symbols to analyze", max_items=50)
    sector: Optional[str] = Field(None, description="Sector to analyze", max_length=50)
    timeframe: str = Field("1d", description="Timeframe for sentiment analysis", regex="^(1d|1w|1m)$")
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    include_social: bool = Field(True, description="Include social media sentiment")
    include_news: bool = Field(True, description="Include news sentiment")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbols list"""
        if v is not None:
            for symbol in v:
                validate_financial_symbol(symbol)
                validate_sql_safe(symbol)
        return v
    
    @validator('sector')
    def validate_sector(cls, v):
        """Validate sector input"""
        if v is not None:
            validate_sql_safe(v)
            validate_xss_safe(v)
        return v

class ReportGenerationRequest(BaseModel):
    """Request schema for report generation"""
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID for portfolio-specific reports", gt=0)
    report_type: ReportType = Field(..., description="Report type")
    format: ReportFormat = Field(ReportFormat.PDF, description="Report format")
    start_date: Optional[date] = Field(None, description="Start date for report")
    end_date: Optional[date] = Field(None, description="End date for report")
    include_benchmarks: bool = Field(True, description="Include benchmark comparisons")
    include_attribution: bool = Field(True, description="Include performance attribution")
    custom_title: Optional[str] = Field(None, description="Custom report title", max_length=200)
    
    @validator('custom_title')
    def validate_title(cls, v):
        """Validate custom title"""
        if v is not None:
            validate_xss_safe(v)
        return v
    
    @root_validator
    def validate_portfolio_requirement(cls, values):
        """Validate portfolio ID is provided for portfolio-specific reports"""
        report_type = values.get('report_type')
        portfolio_id = values.get('portfolio_id')
        
        if report_type in [ReportType.PERFORMANCE, ReportType.RISK, ReportType.ALLOCATION]:
            if portfolio_id is None:
                raise ValueError(f'Portfolio ID is required for {report_type.value} reports')
        
        return values

class BacktestAnalysisRequest(BaseModel):
    """Request schema for backtest analysis"""
    backtest_id: int = Field(..., description="Backtest ID", gt=0)
    include_trades: bool = Field(True, description="Include individual trade analysis")
    include_statistics: bool = Field(True, description="Include statistical significance tests")
    confidence_level: float = Field(0.95, description="Confidence level for statistical tests", ge=0.9, le=0.99)

# ================================
# RESPONSE SCHEMAS
# ================================

class PerformanceMetricsResponse(BaseModel):
    """Response schema for performance metrics"""
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    alpha: float = Field(..., description="Alpha")
    beta: float = Field(..., description="Beta")
    var_95: float = Field(..., description="Value at Risk (95%)")
    win_rate: float = Field(..., description="Win rate")
    
    class Config:
        schema_extra = {
            "example": {
                "total_return": 0.15,
                "annualized_return": 0.12,
                "volatility": 0.18,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.8,
                "max_drawdown": -0.08,
                "alpha": 0.03,
                "beta": 1.1,
                "var_95": -0.035,
                "win_rate": 0.62
            }
        }

class RiskAnalysisResponse(BaseModel):
    """Response schema for risk analysis"""
    portfolio_var: float = Field(..., description="Portfolio Value at Risk")
    diversification_ratio: float = Field(..., description="Diversification ratio")
    concentration_risk: float = Field(..., description="Concentration risk")
    component_var: Dict[str, float] = Field(..., description="Component VaR by asset")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Correlation matrix")
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_var": 0.035,
                "diversification_ratio": 1.25,
                "concentration_risk": 0.15,
                "component_var": {"AAPL": 0.012, "MSFT": 0.008},
                "correlation_matrix": {"AAPL": {"MSFT": 0.65}}
            }
        }

class MarketSentimentResponse(BaseModel):
    """Response schema for market sentiment"""
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
    timeframe: str = Field(..., description="Analysis timeframe")
    symbols_analyzed: List[str] = Field(..., description="Symbols analyzed")
    sentiment_data: Dict[str, Any] = Field(..., description="Sentiment data")
    data_points: int = Field(..., description="Number of data points")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_timestamp": "2024-01-15T10:30:00Z",
                "timeframe": "1d",
                "symbols_analyzed": ["AAPL", "MSFT"],
                "sentiment_data": {
                    "AAPL": {
                        "current_sentiment": 0.65,
                        "confidence": 0.82,
                        "trend": "positive"
                    }
                },
                "data_points": 1500
            }
        }

class ReportStatusResponse(BaseModel):
    """Response schema for report status"""
    report_id: int = Field(..., description="Report ID")
    title: str = Field(..., description="Report title")
    report_type: str = Field(..., description="Report type")
    status: str = Field(..., description="Report status")
    file_format: str = Field(..., description="File format")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    requested_at: datetime = Field(..., description="Request timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    download_url: Optional[str] = Field(None, description="Download URL")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "report_id": 123,
                "title": "Performance Report - 2024-01-15",
                "report_type": "performance",
                "status": "completed",
                "file_format": "PDF",
                "file_size_bytes": 2048576,
                "requested_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:05:00Z",
                "download_url": "/api/analytics/reports/123/download",
                "error_message": None
            }
        }

class TimeSeriesAnalysisResponse(BaseModel):
    """Response schema for time series analysis"""
    trend_strength: float = Field(..., description="Trend strength (0-1)")
    trend_direction: str = Field(..., description="Trend direction")
    volatility: float = Field(..., description="Current volatility")
    data_quality: str = Field(..., description="Data quality assessment")
    seasonality_score: float = Field(..., description="Seasonality score")
    autocorrelation: float = Field(..., description="Autocorrelation coefficient")
    
    class Config:
        schema_extra = {
            "example": {
                "trend_strength": 0.75,
                "trend_direction": "upward",
                "volatility": 0.18,
                "data_quality": "high",
                "seasonality_score": 0.15,
                "autocorrelation": 0.23
            }
        }

class PortfolioAnalyticsResponse(BaseModel):
    """Comprehensive portfolio analytics response"""
    portfolio: Dict[str, Any] = Field(..., description="Portfolio information")
    analysis_period: Dict[str, Any] = Field(..., description="Analysis period details")
    performance_metrics: PerformanceMetricsResponse = Field(..., description="Performance metrics")
    time_series_analysis: TimeSeriesAnalysisResponse = Field(..., description="Time series analysis")
    generated_at: datetime = Field(..., description="Generation timestamp")

class BacktestAnalysisResponse(BaseModel):
    """Response schema for backtest analysis"""
    backtest_id: int = Field(..., description="Backtest ID")
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Symbol analyzed")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
    statistical_significance: Dict[str, Any] = Field(..., description="Statistical tests")
    performance_analysis: Dict[str, Any] = Field(..., description="Performance analysis")
    risk_analysis: Optional[Dict[str, Any]] = Field(None, description="Risk analysis")

# ================================
# ERROR SCHEMAS
# ================================

class ValidationError(BaseModel):
    """Validation error response"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    code: str = Field(..., description="Error code")

class AnalyticsError(BaseModel):
    """Analytics error response"""
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

# ================================
# UTILITY FUNCTIONS
# ================================

def validate_portfolio_access(portfolio_id: int, user_id: int) -> bool:
    """
    Validate user has access to portfolio
    
    Args:
        portfolio_id: Portfolio ID to check
        user_id: User ID making the request
        
    Returns:
        True if user has access, False otherwise
    """
    # This would typically check database permissions
    # For now, return True for demonstration
    return True

def sanitize_analytics_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize analytics input data
    
    Args:
        data: Input data dictionary
        
    Returns:
        Sanitized data dictionary
    """
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Apply XSS and SQL injection protection
            sanitized[key] = validate_xss_safe(validate_sql_safe(value))
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [
                validate_xss_safe(validate_sql_safe(item)) if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, dict):
            sanitized[key] = sanitize_analytics_input(value)
        else:
            sanitized[key] = value
    
    return sanitized

# ================================
# ANALYTICS VALIDATORS
# ================================

class AnalyticsValidationMixin:
    """Mixin class for analytics validation"""
    
    @staticmethod
    def validate_date_range(start_date: Optional[date], end_date: Optional[date], max_range_days: int = 1825):
        """Validate date range parameters"""
        if start_date and end_date:
            if start_date > end_date:
                raise ValueError('Start date must be before end date')
            
            if (end_date - start_date).days > max_range_days:
                raise ValueError(f'Date range cannot exceed {max_range_days} days')
                
            if start_date > date.today():
                raise ValueError('Start date cannot be in the future')
    
    @staticmethod
    def validate_financial_parameters(
        confidence_level: Optional[float] = None,
        time_horizon: Optional[int] = None,
        volatility_threshold: Optional[float] = None
    ):
        """Validate financial calculation parameters"""
        if confidence_level is not None:
            if not 0.5 <= confidence_level <= 0.99:
                raise ValueError('Confidence level must be between 0.5 and 0.99')
        
        if time_horizon is not None:
            if not 1 <= time_horizon <= 252:
                raise ValueError('Time horizon must be between 1 and 252 days')
        
        if volatility_threshold is not None:
            if not 0 <= volatility_threshold <= 2:
                raise ValueError('Volatility threshold must be between 0 and 2')
    
    @staticmethod
    def validate_portfolio_parameters(portfolio_id: int, user_id: int):
        """Validate portfolio access and parameters"""
        if portfolio_id <= 0:
            raise ValueError('Portfolio ID must be positive')
        
        if not validate_portfolio_access(portfolio_id, user_id):
            raise ValueError('Access denied to specified portfolio')
    
    @staticmethod
    def validate_symbol_list(symbols: List[str], max_symbols: int = 50):
        """Validate list of financial symbols"""
        if len(symbols) > max_symbols:
            raise ValueError(f'Cannot analyze more than {max_symbols} symbols at once')
        
        for symbol in symbols:
            validate_financial_symbol(symbol)
            validate_sql_safe(symbol)