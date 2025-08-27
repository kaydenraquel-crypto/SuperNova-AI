"""
Advanced Analytics API Endpoints for SuperNova AI
Comprehensive API routes for portfolio analytics, risk management, and financial reporting
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, date
from decimal import Decimal
import asyncio
import json
import logging
import io
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func, desc

# Import database and authentication
from .db import SessionLocal, get_timescale_session, is_timescale_available
from .analytics_models import (
    Portfolio, Position, Transaction, PerformanceRecord, RiskMetric,
    MarketSentiment, TechnicalIndicator, BacktestAnalysis, AnalyticsReport
)
from .auth import get_current_user, require_permission, Permission
from .schemas import UserProfile

# Import analytics engines
from .analytics_engine import AdvancedAnalyticsEngine, PerformanceMetrics, RiskAnalysis, AttributionAnalysis
from .data_processing_engine import FinancialDataProcessor, TimeSeriesMetrics, RiskModel, MarketRegime

# Import validation and security
from .input_validation import validate_sql_safe, validate_decimal_amount, input_validator
from .financial_validators import validate_portfolio_id, validate_symbol, validate_date_range

logger = logging.getLogger(__name__)

# Initialize analytics components
analytics_engine = AdvancedAnalyticsEngine()
data_processor = FinancialDataProcessor()

# Create router
router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

# Dependency for database session
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================================
# PORTFOLIO PERFORMANCE ENDPOINTS
# ================================

@router.get("/portfolio/{portfolio_id}/performance", response_model=Dict[str, Any])
async def get_portfolio_performance(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    start_date: Optional[date] = Query(None, description="Start date for analysis"),
    end_date: Optional[date] = Query(None, description="End date for analysis"),
    benchmark: Optional[str] = Query(None, description="Benchmark symbol for comparison"),
    period: str = Query("daily", description="Performance period (daily, weekly, monthly)"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive portfolio performance metrics
    
    Returns detailed performance analysis including:
    - Total and annualized returns
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
    - Drawdown analysis
    - Beta and alpha calculations
    - Statistical metrics (VaR, skewness, kurtosis)
    """
    try:
        # Validate inputs
        validate_portfolio_id(portfolio_id)
        if benchmark:
            validate_symbol(benchmark)
        if start_date and end_date:
            validate_date_range(start_date, end_date)
        
        # Check portfolio access
        portfolio = db.query(Portfolio).filter(
            and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        ).first()
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Get portfolio performance records
        query = db.query(PerformanceRecord).filter(PerformanceRecord.portfolio_id == portfolio_id)
        
        if start_date:
            query = query.filter(PerformanceRecord.period_date >= start_date)
        if end_date:
            query = query.filter(PerformanceRecord.period_date <= end_date)
        if period != "daily":
            query = query.filter(PerformanceRecord.period_type == period)
        
        performance_records = query.order_by(PerformanceRecord.period_date).all()
        
        if not performance_records:
            raise HTTPException(status_code=404, detail="No performance data found for the specified period")
        
        # Convert to pandas series for analysis
        portfolio_values = pd.Series(
            [float(record.total_value) for record in performance_records],
            index=[record.period_date for record in performance_records]
        )
        
        # Get benchmark data if specified
        benchmark_values = None
        if benchmark:
            # This would typically fetch from market data provider
            # For now, we'll create a synthetic benchmark
            benchmark_values = portfolio_values * (1 + pd.Series([0.0001] * len(portfolio_values)).cumsum())
        
        # Calculate comprehensive performance metrics
        performance_metrics = analytics_engine.calculate_portfolio_performance(
            portfolio_values, benchmark_values
        )
        
        # Get additional portfolio information
        portfolio_info = {
            "id": portfolio.id,
            "name": portfolio.name,
            "currency": portfolio.currency,
            "initial_value": float(portfolio.initial_value),
            "benchmark_symbol": portfolio.benchmark_symbol,
            "is_paper_trading": portfolio.is_paper_trading
        }
        
        # Calculate time series analysis
        time_series_metrics = data_processor.process_time_series_data(
            pd.DataFrame({'close': portfolio_values})
        )
        
        return {
            "portfolio": portfolio_info,
            "analysis_period": {
                "start_date": start_date or performance_records[0].period_date,
                "end_date": end_date or performance_records[-1].period_date,
                "data_points": len(performance_records)
            },
            "performance_metrics": performance_metrics.to_dict(),
            "time_series_analysis": time_series_metrics.to_dict(),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating portfolio performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/portfolio/{portfolio_id}/risk", response_model=Dict[str, Any])
async def get_portfolio_risk_analysis(
    portfolio_id: int = Path(..., description="Portfolio ID"),
    confidence_level: float = Query(0.95, description="Confidence level for VaR calculations"),
    time_horizon: int = Query(1, description="Time horizon in days"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive portfolio risk analysis
    
    Returns detailed risk metrics including:
    - Value at Risk (VaR) and Conditional VaR
    - Risk decomposition by position
    - Correlation analysis
    - Concentration risk measures
    - Volatility forecasting
    """
    try:
        # Validate inputs
        validate_portfolio_id(portfolio_id)
        if not 0.5 <= confidence_level <= 0.99:
            raise ValueError("Confidence level must be between 0.5 and 0.99")
        if not 1 <= time_horizon <= 252:
            raise ValueError("Time horizon must be between 1 and 252 days")
        
        # Check portfolio access
        portfolio = db.query(Portfolio).filter(
            and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        ).first()
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Get current positions
        positions = db.query(Position).filter(
            and_(Position.portfolio_id == portfolio_id, Position.closed_at.is_(None))
        ).all()
        
        if not positions:
            raise HTTPException(status_code=404, detail="No active positions found")
        
        # Prepare position data
        position_data = {}
        total_value = sum(float(pos.market_value or 0) for pos in positions)
        
        for position in positions:
            market_value = float(position.market_value or 0)
            weight = market_value / total_value if total_value > 0 else 0
            position_data[position.symbol] = {
                'weight': weight,
                'value': market_value,
                'quantity': float(position.quantity),
                'sector': position.sector or 'Unknown'
            }
        
        # Get returns data (simplified - would typically fetch from market data)
        symbols = list(position_data.keys())
        returns_data = pd.DataFrame()
        
        # This would typically fetch actual market data
        # For demonstration, we'll create synthetic returns
        for symbol in symbols:
            returns_data[symbol] = pd.Series(
                np.random.normal(0, 0.02, 252)  # Synthetic daily returns
            )
        
        # Calculate risk analysis
        risk_analysis = analytics_engine.calculate_risk_analysis(
            position_data, returns_data, confidence_level
        )
        
        # Calculate advanced risk metrics
        portfolio_returns = pd.Series([0.001] * 252)  # Synthetic portfolio returns
        risk_model = data_processor.calculate_advanced_risk_metrics(
            portfolio_returns, [confidence_level], time_horizon
        )
        
        # Get recent risk metrics from database
        recent_risk_metrics = db.query(RiskMetric).filter(
            and_(
                RiskMetric.portfolio_id == portfolio_id,
                RiskMetric.calculation_date >= datetime.utcnow() - timedelta(days=30)
            )
        ).order_by(desc(RiskMetric.calculation_date)).limit(10).all()
        
        risk_history = []
        for metric in recent_risk_metrics:
            risk_history.append({
                "date": metric.calculation_date.isoformat(),
                "metric_type": metric.metric_type,
                "value": metric.value,
                "confidence_level": metric.confidence_level
            })
        
        return {
            "portfolio_id": portfolio_id,
            "analysis_date": datetime.utcnow().isoformat(),
            "positions_analyzed": len(positions),
            "total_portfolio_value": total_value,
            "risk_analysis": risk_analysis.to_dict(),
            "risk_model": risk_model.to_dict(),
            "risk_history": risk_history,
            "parameters": {
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================================
# MARKET ANALYSIS ENDPOINTS
# ================================

@router.get("/market/sentiment", response_model=Dict[str, Any])
async def get_market_sentiment(
    symbols: Optional[List[str]] = Query(None, description="List of symbols to analyze"),
    sector: Optional[str] = Query(None, description="Sector to analyze"),
    timeframe: str = Query("1d", description="Timeframe for sentiment analysis"),
    limit: int = Query(100, description="Maximum number of results"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get market sentiment analysis
    
    Returns sentiment scores from multiple sources:
    - Social media sentiment
    - News sentiment  
    - Analyst sentiment
    - Volume-weighted sentiment
    """
    try:
        # Validate inputs
        if symbols:
            for symbol in symbols:
                validate_symbol(symbol)
        if limit > 1000:
            raise ValueError("Limit cannot exceed 1000")
        
        # Build query
        query = db.query(MarketSentiment)
        
        if symbols:
            query = query.filter(MarketSentiment.symbol.in_(symbols))
        elif sector:
            query = query.filter(MarketSentiment.sector == sector)
        
        # Get recent sentiment data
        cutoff_time = datetime.utcnow() - timedelta(days=1 if timeframe == "1d" else 7)
        query = query.filter(MarketSentiment.timestamp >= cutoff_time)
        
        sentiment_records = query.order_by(desc(MarketSentiment.timestamp)).limit(limit).all()
        
        # Process sentiment data
        sentiment_analysis = {}
        
        for record in sentiment_records:
            key = record.symbol or record.sector or 'market'
            
            if key not in sentiment_analysis:
                sentiment_analysis[key] = {
                    'current_sentiment': record.sentiment_score,
                    'confidence': record.confidence_score,
                    'volume_weighted': record.volume_weighted_score,
                    'social_sentiment': record.social_sentiment,
                    'news_sentiment': record.news_sentiment,
                    'analyst_sentiment': record.analyst_sentiment,
                    'total_mentions': record.total_mentions,
                    'last_updated': record.timestamp.isoformat(),
                    'trend': 'neutral'  # Would be calculated from historical data
                }
        
        # Calculate overall market sentiment if no specific symbols requested
        if not symbols and not sector:
            overall_scores = [record.sentiment_score for record in sentiment_records]
            overall_sentiment = {
                'average_sentiment': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                'sentiment_distribution': {
                    'bullish': len([s for s in overall_scores if s > 0.1]),
                    'neutral': len([s for s in overall_scores if -0.1 <= s <= 0.1]),
                    'bearish': len([s for s in overall_scores if s < -0.1])
                },
                'market_mood': 'bullish' if sum(overall_scores) > 0.1 else 'bearish' if sum(overall_scores) < -0.1 else 'neutral'
            }
            sentiment_analysis['overall_market'] = overall_sentiment
        
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "timeframe": timeframe,
            "symbols_analyzed": symbols or ['market'],
            "sentiment_data": sentiment_analysis,
            "data_points": len(sentiment_records)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================================
# REPORTING ENDPOINTS
# ================================

@router.post("/reports/generate", response_model=Dict[str, Any])
async def generate_analytics_report(
    background_tasks: BackgroundTasks,
    portfolio_id: Optional[int] = Query(None, description="Portfolio ID for portfolio-specific reports"),
    report_type: str = Query(..., description="Report type: performance, risk, allocation, summary"),
    format: str = Query("pdf", description="Report format: pdf, xlsx, csv"),
    start_date: Optional[date] = Query(None, description="Start date for report"),
    end_date: Optional[date] = Query(None, description="End date for report"),
    include_benchmarks: bool = Query(True, description="Include benchmark comparisons"),
    include_attribution: bool = Query(True, description="Include performance attribution"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive analytics report
    
    Supported report types:
    - performance: Detailed performance analysis
    - risk: Comprehensive risk assessment  
    - allocation: Asset allocation analysis
    - summary: Executive summary report
    """
    try:
        # Validate inputs
        valid_report_types = ["performance", "risk", "allocation", "summary"]
        if report_type not in valid_report_types:
            raise ValueError(f"Report type must be one of: {valid_report_types}")
        
        valid_formats = ["pdf", "xlsx", "csv"]
        if format not in valid_formats:
            raise ValueError(f"Format must be one of: {valid_formats}")
        
        if portfolio_id:
            validate_portfolio_id(portfolio_id)
            # Check portfolio access
            portfolio = db.query(Portfolio).filter(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
            ).first()
            if not portfolio:
                raise HTTPException(status_code=404, detail="Portfolio not found")
        
        if start_date and end_date:
            validate_date_range(start_date, end_date)
        
        # Create report record
        report = AnalyticsReport(
            user_id=current_user.id,
            portfolio_id=portfolio_id,
            report_type=report_type,
            title=f"{report_type.title()} Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
            period_start=start_date or datetime.utcnow() - timedelta(days=365),
            period_end=end_date or datetime.utcnow(),
            file_format=format.upper(),
            parameters=json.dumps({
                "include_benchmarks": include_benchmarks,
                "include_attribution": include_attribution,
                "generated_by": current_user.name
            }),
            status="pending"
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Queue report generation as background task
        background_tasks.add_task(
            generate_report_background,
            report.id,
            portfolio_id,
            report_type,
            format,
            start_date,
            end_date,
            include_benchmarks,
            include_attribution
        )
        
        return {
            "report_id": report.id,
            "status": "queued",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "download_url": f"/api/analytics/reports/{report.id}/download",
            "message": "Report generation started. You will be notified when complete."
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/reports/{report_id}", response_model=Dict[str, Any])
async def get_report_status(
    report_id: int = Path(..., description="Report ID"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analytics report status and metadata"""
    try:
        report = db.query(AnalyticsReport).filter(
            and_(AnalyticsReport.id == report_id, AnalyticsReport.user_id == current_user.id)
        ).first()
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": report.id,
            "title": report.title,
            "report_type": report.report_type,
            "status": report.status,
            "file_format": report.file_format,
            "file_size_bytes": report.file_size_bytes,
            "requested_at": report.requested_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None,
            "expires_at": report.expires_at.isoformat() if report.expires_at else None,
            "download_url": f"/api/analytics/reports/{report_id}/download" if report.status == "completed" else None,
            "error_message": report.error_message,
            "parameters": json.loads(report.parameters) if report.parameters else {}
        }
        
    except Exception as e:
        logger.error(f"Error getting report status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: int = Path(..., description="Report ID"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download generated analytics report"""
    try:
        report = db.query(AnalyticsReport).filter(
            and_(AnalyticsReport.id == report_id, AnalyticsReport.user_id == current_user.id)
        ).first()
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if report.status != "completed":
            raise HTTPException(status_code=400, detail="Report is not ready for download")
        
        if report.expires_at and datetime.utcnow() > report.expires_at:
            raise HTTPException(status_code=410, detail="Report has expired")
        
        if not report.file_path:
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Return file response
        return FileResponse(
            path=report.file_path,
            filename=f"{report.report_type}_report_{report_id}.{report.file_format.lower()}",
            media_type=f"application/{report.file_format.lower()}"
        )
        
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================================
# BACKTESTING ANALYSIS ENDPOINTS  
# ================================

@router.get("/backtests/{backtest_id}/analysis", response_model=Dict[str, Any])
async def get_backtest_analysis(
    backtest_id: int = Path(..., description="Backtest ID"),
    include_trades: bool = Query(True, description="Include individual trade analysis"),
    include_statistics: bool = Query(True, description="Include statistical significance tests"),
    current_user: UserProfile = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive backtest analysis with statistical significance testing
    
    Returns detailed analysis including:
    - Performance metrics with confidence intervals
    - Trade-by-trade analysis
    - Statistical significance tests
    - Risk-adjusted performance measures
    - Drawdown analysis with recovery periods
    """
    try:
        # Get backtest result (assuming this exists in the original backtester)
        from .backtester import BacktestResult as OriginalBacktestResult
        
        backtest = db.query(OriginalBacktestResult).filter(
            OriginalBacktestResult.id == backtest_id
        ).first()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Get associated analysis records
        analyses = db.query(BacktestAnalysis).filter(
            BacktestAnalysis.backtest_result_id == backtest_id
        ).all()
        
        # Parse backtest metrics
        metrics = json.loads(backtest.metrics_json)
        
        # Perform additional statistical analysis
        returns_data = pd.Series(metrics.get('returns', []))
        
        if len(returns_data) > 30:
            # Statistical significance tests
            statistical_tests = perform_statistical_significance_tests(returns_data)
            
            # Performance metrics with confidence intervals
            performance_analysis = calculate_performance_confidence_intervals(returns_data)
            
            # Risk analysis
            risk_analysis = data_processor.calculate_advanced_risk_metrics(returns_data)
        else:
            statistical_tests = {"error": "Insufficient data for statistical analysis"}
            performance_analysis = {"error": "Insufficient data for performance analysis"}
            risk_analysis = None
        
        # Compile analysis results
        analysis_results = {
            "backtest_id": backtest_id,
            "strategy_name": backtest.strategy_id,
            "symbol": backtest.symbol,
            "timeframe": backtest.timeframe,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "basic_metrics": metrics,
            "statistical_significance": statistical_tests,
            "performance_analysis": performance_analysis,
            "risk_analysis": risk_analysis.to_dict() if risk_analysis else None,
            "detailed_analyses": []
        }
        
        # Add detailed analysis records
        for analysis in analyses:
            analysis_data = {
                "analysis_type": analysis.analysis_type,
                "analysis_name": analysis.analysis_name,
                "p_value": analysis.p_value,
                "confidence_interval": [
                    analysis.confidence_interval_lower,
                    analysis.confidence_interval_upper
                ] if analysis.confidence_interval_lower is not None else None,
                "results": json.loads(analysis.results),
                "chart_data": json.loads(analysis.chart_data) if analysis.chart_data else None
            }
            analysis_results["detailed_analyses"].append(analysis_data)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error getting backtest analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================================
# BACKGROUND TASKS
# ================================

async def generate_report_background(
    report_id: int,
    portfolio_id: Optional[int],
    report_type: str,
    format: str,
    start_date: Optional[date],
    end_date: Optional[date],
    include_benchmarks: bool,
    include_attribution: bool
):
    """Background task for report generation"""
    db = SessionLocal()
    try:
        # Update report status
        report = db.query(AnalyticsReport).filter(AnalyticsReport.id == report_id).first()
        if not report:
            return
        
        report.status = "generating"
        db.commit()
        
        # Simulate report generation (replace with actual implementation)
        await asyncio.sleep(10)  # Simulate processing time
        
        # Generate report file path
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        file_extension = format.lower()
        file_name = f"report_{report_id}.{file_extension}"
        file_path = os.path.join(temp_dir, file_name)
        
        # Create dummy report file (replace with actual report generation)
        if format.lower() == "pdf":
            # Would use libraries like reportlab, matplotlib, etc.
            with open(file_path, "w") as f:
                f.write(f"Dummy {report_type} report content")
        elif format.lower() == "xlsx":
            # Would use pandas, openpyxl, etc.
            df = pd.DataFrame({"metric": ["return", "volatility"], "value": [0.1, 0.15]})
            df.to_excel(file_path, index=False)
        elif format.lower() == "csv":
            df = pd.DataFrame({"metric": ["return", "volatility"], "value": [0.1, 0.15]})
            df.to_csv(file_path, index=False)
        
        # Update report record
        report.status = "completed"
        report.file_path = file_path
        report.file_size_bytes = os.path.getsize(file_path)
        report.completed_at = datetime.utcnow()
        report.expires_at = datetime.utcnow() + timedelta(days=7)  # 7 day expiration
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error generating report {report_id}: {str(e)}")
        report.status = "failed"
        report.error_message = str(e)
        db.commit()
    finally:
        db.close()

# ================================
# HELPER FUNCTIONS
# ================================

def perform_statistical_significance_tests(returns: pd.Series) -> Dict[str, Any]:
    """Perform statistical significance tests on backtest returns"""
    try:
        from scipy.stats import ttest_1samp, jarque_bera, shapiro
        
        # Test if mean return is significantly different from zero
        t_stat, p_value_mean = ttest_1samp(returns, 0)
        
        # Test for normality
        jb_stat, p_value_normality = jarque_bera(returns)
        
        # Sharpe ratio significance test (simplified)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        sharpe_std_error = np.sqrt((1 + (sharpe_ratio**2)/2) / len(returns))
        sharpe_t_stat = sharpe_ratio / sharpe_std_error
        sharpe_p_value = 2 * (1 - stats.t.cdf(abs(sharpe_t_stat), len(returns)-1))
        
        return {
            "mean_return_test": {
                "t_statistic": t_stat,
                "p_value": p_value_mean,
                "significant": p_value_mean < 0.05
            },
            "normality_test": {
                "jarque_bera_statistic": jb_stat,
                "p_value": p_value_normality,
                "normal_distribution": p_value_normality > 0.05
            },
            "sharpe_ratio_test": {
                "sharpe_ratio": sharpe_ratio,
                "standard_error": sharpe_std_error,
                "t_statistic": sharpe_t_stat,
                "p_value": sharpe_p_value,
                "significant": sharpe_p_value < 0.05
            }
        }
    except Exception as e:
        return {"error": f"Statistical test error: {str(e)}"}

def calculate_performance_confidence_intervals(returns: pd.Series, confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculate performance metrics with confidence intervals"""
    try:
        from scipy.stats import bootstrap
        
        # Bootstrap confidence intervals for key metrics
        def calculate_sharpe(data):
            return data.mean() / data.std() * np.sqrt(252)
        
        def calculate_sortino(data):
            downside = data[data < 0]
            return data.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        
        n_bootstrap = 1000
        alpha = 1 - confidence_level
        
        # Bootstrap for Sharpe ratio
        sharpe_samples = []
        sortino_samples = []
        
        for _ in range(n_bootstrap):
            sample = returns.sample(n=len(returns), replace=True)
            sharpe_samples.append(calculate_sharpe(sample))
            sortino_samples.append(calculate_sortino(sample))
        
        sharpe_ci = np.percentile(sharpe_samples, [alpha/2*100, (1-alpha/2)*100])
        sortino_ci = np.percentile(sortino_samples, [alpha/2*100, (1-alpha/2)*100])
        
        return {
            "sharpe_ratio": {
                "point_estimate": calculate_sharpe(returns),
                "confidence_interval": sharpe_ci.tolist(),
                "confidence_level": confidence_level
            },
            "sortino_ratio": {
                "point_estimate": calculate_sortino(returns),
                "confidence_interval": sortino_ci.tolist(),
                "confidence_level": confidence_level
            },
            "bootstrap_samples": n_bootstrap
        }
    except Exception as e:
        return {"error": f"Confidence interval calculation error: {str(e)}"}

# Add numpy import for helper functions
import numpy as np