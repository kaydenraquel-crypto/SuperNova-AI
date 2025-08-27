"""
Unit tests for the Advanced Analytics Engine
Tests for portfolio performance metrics, risk analysis, and financial calculations
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supernova.analytics_engine import (
    AdvancedAnalyticsEngine, 
    PerformanceMetrics, 
    RiskAnalysis,
    AttributionAnalysis
)
from supernova.analytics_error_handler import (
    AnalyticsException,
    InsufficientDataException,
    CalculationException
)

@pytest.fixture
def analytics_engine():
    """Create analytics engine instance for testing"""
    return AdvancedAnalyticsEngine(risk_free_rate=0.02)

@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio value data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic price movements
    initial_value = 100000
    returns = np.random.normal(0.0003, 0.012, len(dates))  # Daily returns
    values = [initial_value]
    
    for ret in returns:
        values.append(values[-1] * (1 + ret))
    
    return pd.Series(values[1:], index=dates)

@pytest.fixture
def sample_benchmark_data():
    """Generate sample benchmark data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(123)  # Different seed for benchmark
    
    initial_value = 100000
    returns = np.random.normal(0.0002, 0.010, len(dates))  # Lower return, lower vol
    values = [initial_value]
    
    for ret in returns:
        values.append(values[-1] * (1 + ret))
    
    return pd.Series(values[1:], index=dates)

@pytest.fixture
def sample_position_data():
    """Generate sample position data for risk analysis"""
    return {
        'AAPL': {'weight': 0.3, 'value': 30000, 'sector': 'Technology'},
        'MSFT': {'weight': 0.25, 'value': 25000, 'sector': 'Technology'},
        'GOOGL': {'weight': 0.2, 'value': 20000, 'sector': 'Technology'},
        'JPM': {'weight': 0.15, 'value': 15000, 'sector': 'Financial'},
        'JNJ': {'weight': 0.1, 'value': 10000, 'sector': 'Healthcare'}
    }

@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for risk analysis"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.0005, 0.025, len(dates)),
        'MSFT': np.random.normal(0.0004, 0.022, len(dates)),
        'GOOGL': np.random.normal(0.0003, 0.028, len(dates)),
        'JPM': np.random.normal(0.0002, 0.030, len(dates)),
        'JNJ': np.random.normal(0.0001, 0.015, len(dates))
    }, index=dates)
    
    return returns_data

class TestAdvancedAnalyticsEngine:
    """Test cases for the Advanced Analytics Engine"""
    
    def test_initialization(self):
        """Test analytics engine initialization"""
        engine = AdvancedAnalyticsEngine(risk_free_rate=0.03)
        assert engine.risk_free_rate == 0.03
        assert engine.trading_days_per_year == 252
        assert engine.executor is not None
    
    def test_calculate_portfolio_performance_basic(self, analytics_engine, sample_portfolio_data):
        """Test basic portfolio performance calculation"""
        metrics = analytics_engine.calculate_portfolio_performance(sample_portfolio_data)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        
        # Check reasonable ranges
        assert -1 <= metrics.total_return <= 5  # Reasonable annual return range
        assert 0 <= metrics.volatility <= 1     # Reasonable volatility range
        assert -3 <= metrics.sharpe_ratio <= 10 # Reasonable Sharpe ratio range
        assert -1 <= metrics.max_drawdown <= 0  # Drawdown should be negative
    
    def test_calculate_portfolio_performance_with_benchmark(
        self, analytics_engine, sample_portfolio_data, sample_benchmark_data
    ):
        """Test portfolio performance calculation with benchmark"""
        metrics = analytics_engine.calculate_portfolio_performance(
            sample_portfolio_data, sample_benchmark_data
        )
        
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.tracking_error, float)
        assert isinstance(metrics.information_ratio, float)
        
        # Beta should be reasonable (typically 0.5 to 2.0)
        assert 0 <= metrics.beta <= 3
        assert 0 <= metrics.tracking_error <= 1
    
    def test_calculate_risk_analysis(
        self, analytics_engine, sample_position_data, sample_returns_data
    ):
        """Test risk analysis calculation"""
        risk_analysis = analytics_engine.calculate_risk_analysis(
            sample_position_data, sample_returns_data
        )
        
        assert isinstance(risk_analysis, RiskAnalysis)
        assert isinstance(risk_analysis.portfolio_var, float)
        assert isinstance(risk_analysis.diversification_ratio, float)
        assert isinstance(risk_analysis.concentration_risk, float)
        assert isinstance(risk_analysis.component_var, dict)
        assert isinstance(risk_analysis.correlation_matrix, dict)
        
        # Check component VaR has correct keys
        assert set(risk_analysis.component_var.keys()) == set(sample_position_data.keys())
        
        # Diversification ratio should be >= 1
        assert risk_analysis.diversification_ratio >= 1
        
        # Concentration risk should be between 0 and 1
        assert 0 <= risk_analysis.concentration_risk <= 1
    
    def test_calculate_attribution_analysis(self, analytics_engine):
        """Test performance attribution analysis"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        portfolio_returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, 252), index=dates)
        
        sector_weights = {'Technology': 0.6, 'Financial': 0.4}
        sector_returns = {
            'Technology': pd.Series(np.random.normal(0.0006, 0.025, 252), index=dates),
            'Financial': pd.Series(np.random.normal(0.0004, 0.018, 252), index=dates)
        }
        
        attribution = analytics_engine.calculate_attribution_analysis(
            portfolio_returns, benchmark_returns, sector_weights, sector_returns
        )
        
        assert isinstance(attribution, AttributionAnalysis)
        assert isinstance(attribution.total_excess_return, float)
        assert isinstance(attribution.asset_allocation, float)
        assert isinstance(attribution.security_selection, float)
        assert isinstance(attribution.sector_attribution, dict)
    
    def test_time_series_analysis(self, analytics_engine, sample_portfolio_data):
        """Test time series analysis"""
        analysis = analytics_engine.calculate_time_series_analysis(sample_portfolio_data)
        
        assert isinstance(analysis, dict)
        assert 'trend_slope' in analysis
        assert 'trend_r_squared' in analysis
        assert 'volatility_persistence' in analysis
        assert 'current_trend' in analysis
        assert 'trend_strength' in analysis
        
        # Check trend direction is valid
        assert analysis['current_trend'] in ['upward', 'downward']
        
        # R-squared should be between 0 and 1
        assert 0 <= analysis['trend_r_squared'] <= 1
    
    def test_correlation_analysis(self, analytics_engine, sample_returns_data):
        """Test correlation analysis"""
        analysis = analytics_engine.calculate_correlation_analysis(sample_returns_data)
        
        assert isinstance(analysis, dict)
        assert 'correlation_matrix' in analysis
        assert 'highest_correlations' in analysis
        assert 'average_correlation' in analysis
        assert 'correlation_distribution' in analysis
        
        # Check correlation matrix structure
        correlation_matrix = analysis['correlation_matrix']
        symbols = list(sample_returns_data.columns)
        assert set(correlation_matrix.keys()) == set(symbols)
        
        # Check correlation values are in valid range
        for symbol1 in symbols:
            for symbol2 in symbols:
                if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                    corr = correlation_matrix[symbol1][symbol2]
                    assert -1 <= corr <= 1
    
    def test_volatility_forecast(self, analytics_engine):
        """Test volatility forecasting"""
        # Generate sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        
        forecast = analytics_engine.calculate_volatility_forecast(returns)
        
        assert isinstance(forecast, dict)
        assert 'forecast_volatility' in forecast
        assert 'historical_volatility' in forecast
        assert 'regime' in forecast
        assert 'model_used' in forecast
        
        # Forecast volatility should be positive
        assert forecast['forecast_volatility'] >= 0
        
        # Regime should be valid
        assert forecast['regime'] in ['high', 'normal', 'low']
    
    def test_insufficient_data_handling(self, analytics_engine):
        """Test handling of insufficient data"""
        # Test with very little data
        small_data = pd.Series([100, 101, 99])
        
        metrics = analytics_engine.calculate_portfolio_performance(small_data)
        
        # Should still return metrics, but they may be less reliable
        assert isinstance(metrics, PerformanceMetrics)
        
        # Some metrics might be zero or NaN due to insufficient data
        # But the function shouldn't crash
    
    def test_invalid_data_handling(self, analytics_engine):
        """Test handling of invalid data"""
        # Test with NaN values
        invalid_data = pd.Series([100, np.nan, 102, 101, np.nan])
        
        # Should handle NaN values gracefully
        try:
            metrics = analytics_engine.calculate_portfolio_performance(invalid_data)
            assert isinstance(metrics, PerformanceMetrics)
        except Exception as e:
            # If it raises an exception, it should be a known analytics exception
            assert isinstance(e, (AnalyticsException, ValueError))
    
    def test_async_portfolio_analytics(self, analytics_engine):
        """Test async portfolio analytics wrapper"""
        import asyncio
        
        portfolio_data = {
            'values': [100000, 101000, 102000, 101500, 103000],
            'benchmark': [100000, 100500, 101000, 100800, 101500],
            'positions': {
                'AAPL': {'weight': 0.5, 'value': 50000},
                'MSFT': {'weight': 0.5, 'value': 50000}
            },
            'returns_data': {
                'AAPL': [0.01, -0.005, 0.015, -0.01, 0.02],
                'MSFT': [0.008, 0.002, -0.01, 0.012, 0.005]
            }
        }
        
        async def run_async_test():
            results = await analytics_engine.calculate_portfolio_analytics_async(portfolio_data)
            return results
        
        results = asyncio.run(run_async_test())
        
        assert isinstance(results, dict)
        # Should contain some results, even if not all due to insufficient data

class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test creation of PerformanceMetrics object"""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_return=0.15,
            excess_return=0.03,
            volatility=0.18,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            calmar_ratio=2.1,
            max_drawdown=-0.08,
            max_drawdown_duration_days=45,
            current_drawdown=-0.02,
            beta=1.1,
            alpha=0.03,
            tracking_error=0.05,
            information_ratio=0.6,
            skewness=-0.2,
            kurtosis=3.5,
            var_95=-0.035,
            cvar_95=-0.045,
            win_rate=0.62,
            profit_factor=1.8,
            avg_win=0.025,
            avg_loss=-0.018
        )
        
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == -0.08
        
        # Test to_dict method
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'total_return' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict

class TestRiskAnalysis:
    """Test cases for RiskAnalysis dataclass"""
    
    def test_risk_analysis_creation(self):
        """Test creation of RiskAnalysis object"""
        risk_analysis = RiskAnalysis(
            portfolio_var=0.035,
            component_var={'AAPL': 0.012, 'MSFT': 0.023},
            marginal_var={'AAPL': 0.015, 'MSFT': 0.028},
            correlation_matrix={'AAPL': {'MSFT': 0.65}},
            diversification_ratio=1.25,
            concentration_risk=0.15
        )
        
        assert risk_analysis.portfolio_var == 0.035
        assert risk_analysis.diversification_ratio == 1.25
        assert 'AAPL' in risk_analysis.component_var
        
        # Test to_dict method
        risk_dict = risk_analysis.to_dict()
        assert isinstance(risk_dict, dict)
        assert 'portfolio_var' in risk_dict

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_information_coefficient(self):
        """Test information coefficient calculation"""
        from supernova.analytics_engine import calculate_information_coefficient
        
        predictions = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        actual_returns = np.array([0.012, 0.018, -0.008, 0.020, -0.003])
        
        ic = calculate_information_coefficient(predictions, actual_returns)
        
        assert isinstance(ic, float)
        assert -1 <= ic <= 1  # Correlation should be between -1 and 1
    
    def test_maximum_adverse_excursion(self):
        """Test MAE calculation"""
        from supernova.analytics_engine import calculate_maximum_adverse_excursion
        
        entry_prices = [100, 105, 98]
        exit_prices = [110, 102, 103]
        low_prices = [[98, 97, 99], [103, 101, 100], [95, 94, 96]]
        
        mae_values = calculate_maximum_adverse_excursion(entry_prices, exit_prices, low_prices)
        
        assert len(mae_values) == len(entry_prices)
        for mae in mae_values:
            assert isinstance(mae, float)
            assert mae >= 0  # MAE should be non-negative
    
    def test_maximum_favorable_excursion(self):
        """Test MFE calculation"""
        from supernova.analytics_engine import calculate_maximum_favorable_excursion
        
        entry_prices = [100, 105, 98]
        exit_prices = [110, 102, 103]
        high_prices = [[108, 112, 115], [107, 108, 106], [102, 105, 108]]
        
        mfe_values = calculate_maximum_favorable_excursion(entry_prices, exit_prices, high_prices)
        
        assert len(mfe_values) == len(entry_prices)
        for mfe in mfe_values:
            assert isinstance(mfe, float)
            assert mfe >= 0  # MFE should be non-negative

class TestErrorHandling:
    """Test error handling in analytics engine"""
    
    def test_empty_data_error(self, analytics_engine):
        """Test handling of empty data"""
        empty_series = pd.Series([], dtype=float)
        
        with pytest.raises((AnalyticsException, ValueError)):
            analytics_engine.calculate_portfolio_performance(empty_series)
    
    def test_calculation_error_handling(self, analytics_engine):
        """Test handling of calculation errors"""
        # Create data that might cause numerical issues
        problematic_data = pd.Series([0, 0, 0, 0, 0])  # No variance
        
        # Should handle gracefully
        try:
            metrics = analytics_engine.calculate_portfolio_performance(problematic_data)
            # If it succeeds, check that metrics are reasonable
            assert isinstance(metrics, PerformanceMetrics)
        except AnalyticsException:
            # If it fails, it should be with a known exception type
            pass
    
    def test_risk_analysis_error_handling(self, analytics_engine):
        """Test risk analysis error handling"""
        # Empty position data
        empty_positions = {}
        empty_returns = pd.DataFrame()
        
        with pytest.raises((AnalyticsException, ValueError)):
            analytics_engine.calculate_risk_analysis(empty_positions, empty_returns)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])