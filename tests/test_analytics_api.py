"""
Unit tests for Analytics API endpoints
Tests for authentication, validation, and API responses
"""

import pytest
import json
from fastapi.testclient import TestClient
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supernova.api import app
from supernova.analytics_schemas import PortfolioPerformanceRequest, ReportGenerationRequest
from supernova.analytics_error_handler import AnalyticsException, AnalyticsErrorCode

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "role": "user"
    }

@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {"Authorization": "Bearer test_token"}

@pytest.fixture
def sample_portfolio():
    """Sample portfolio data"""
    return {
        "id": 1,
        "name": "Test Portfolio",
        "currency": "USD",
        "initial_value": 50000.0,
        "is_paper_trading": True
    }

@pytest.fixture
def sample_performance_data():
    """Sample performance metrics data"""
    return {
        "total_return": 0.15,
        "annualized_return": 0.12,
        "volatility": 0.18,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.08,
        "beta": 1.1,
        "alpha": 0.03,
        "var_95": -0.035,
        "win_rate": 0.62,
        "sortino_ratio": 1.8,
        "calmar_ratio": 2.1,
        "cumulative_return": 0.15,
        "excess_return": 0.03,
        "max_drawdown_duration_days": 45,
        "current_drawdown": -0.02,
        "tracking_error": 0.05,
        "information_ratio": 0.6,
        "skewness": -0.2,
        "kurtosis": 3.5,
        "cvar_95": -0.045,
        "profit_factor": 1.8,
        "avg_win": 0.025,
        "avg_loss": -0.018
    }

@pytest.fixture
def sample_risk_data():
    """Sample risk analysis data"""
    return {
        "portfolio_var": 0.035,
        "diversification_ratio": 1.25,
        "concentration_risk": 0.15,
        "component_var": {"AAPL": 0.012, "MSFT": 0.008},
        "marginal_var": {"AAPL": 0.015, "MSFT": 0.010},
        "correlation_matrix": {"AAPL": {"MSFT": 0.65}}
    }

class TestPortfolioPerformanceAPI:
    """Test portfolio performance API endpoints"""
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    @patch('supernova.analytics_engine.AdvancedAnalyticsEngine.calculate_portfolio_performance')
    def test_get_portfolio_performance_success(
        self, mock_calc_performance, mock_db, mock_get_user, 
        mock_user, sample_portfolio, sample_performance_data, auth_headers
    ):
        """Test successful portfolio performance retrieval"""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Mock database session and queries
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock portfolio query
        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.currency = "USD"
        mock_portfolio.initial_value = 50000.0
        mock_portfolio.is_paper_trading = True
        mock_portfolio.benchmark_symbol = None
        mock_session.query.return_value.filter.return_value.first.return_value = mock_portfolio
        
        # Mock performance records
        mock_records = []
        for i in range(100):  # 100 days of data
            record = Mock()
            record.total_value = 50000 * (1 + i * 0.001)  # Slight upward trend
            record.period_date = date(2023, 1, 1) + timedelta(days=i)
            mock_records.append(record)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_records
        
        # Mock performance calculation
        from supernova.analytics_engine import PerformanceMetrics
        mock_metrics = PerformanceMetrics(**sample_performance_data)
        mock_calc_performance.return_value = mock_metrics
        
        # Make request
        response = client.get(
            "/api/analytics/portfolio/1/performance",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "portfolio" in data
        assert "performance_metrics" in data
        assert "analysis_period" in data
        assert data["portfolio"]["id"] == 1
        assert data["performance_metrics"]["sharpe_ratio"] == 1.2
    
    @patch('supernova.analytics_api.get_current_user')
    def test_get_portfolio_performance_unauthorized(self, mock_get_user):
        """Test unauthorized access"""
        # No authentication
        response = client.get("/api/analytics/portfolio/1/performance")
        assert response.status_code == 401
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    def test_get_portfolio_performance_not_found(
        self, mock_db, mock_get_user, mock_user, auth_headers
    ):
        """Test portfolio not found"""
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        response = client.get(
            "/api/analytics/portfolio/999/performance",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "Portfolio not found" in response.json()["detail"]
    
    def test_get_portfolio_performance_invalid_params(self, auth_headers):
        """Test invalid parameters"""
        # Invalid portfolio ID
        response = client.get(
            "/api/analytics/portfolio/-1/performance",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid date range
        response = client.get(
            "/api/analytics/portfolio/1/performance?start_date=2023-12-31&end_date=2023-01-01",
            headers=auth_headers
        )
        # This should be caught by validation
        assert response.status_code in [400, 422]

class TestRiskAnalysisAPI:
    """Test risk analysis API endpoints"""
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    @patch('supernova.analytics_engine.AdvancedAnalyticsEngine.calculate_risk_analysis')
    def test_get_risk_analysis_success(
        self, mock_calc_risk, mock_db, mock_get_user, 
        mock_user, sample_risk_data, auth_headers
    ):
        """Test successful risk analysis retrieval"""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_session.query.return_value.filter.return_value.first.return_value = mock_portfolio
        
        # Mock positions
        mock_positions = []
        for i, symbol in enumerate(['AAPL', 'MSFT']):
            pos = Mock()
            pos.symbol = symbol
            pos.market_value = 25000.0
            pos.quantity = 100
            pos.sector = 'Technology'
            pos.closed_at = None
            mock_positions.append(pos)
        
        mock_session.query.return_value.filter.return_value.all.return_value = mock_positions
        
        # Mock risk calculation
        from supernova.analytics_engine import RiskAnalysis
        mock_risk = RiskAnalysis(**sample_risk_data)
        mock_calc_risk.return_value = mock_risk
        
        # Make request
        response = client.get(
            "/api/analytics/portfolio/1/risk",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "risk_analysis" in data
        assert "portfolio_id" in data
        assert data["risk_analysis"]["portfolio_var"] == 0.035
    
    def test_get_risk_analysis_invalid_confidence(self, auth_headers):
        """Test invalid confidence level"""
        response = client.get(
            "/api/analytics/portfolio/1/risk?confidence_level=1.5",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_get_risk_analysis_invalid_time_horizon(self, auth_headers):
        """Test invalid time horizon"""
        response = client.get(
            "/api/analytics/portfolio/1/risk?time_horizon=500",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

class TestMarketSentimentAPI:
    """Test market sentiment API endpoints"""
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    def test_get_market_sentiment_success(
        self, mock_db, mock_get_user, mock_user, auth_headers
    ):
        """Test successful market sentiment retrieval"""
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock sentiment records
        mock_records = []
        for symbol in ['AAPL', 'MSFT']:
            record = Mock()
            record.symbol = symbol
            record.sentiment_score = 0.5
            record.confidence_score = 0.8
            record.volume_weighted_score = 0.6
            record.total_mentions = 1000
            record.timestamp = datetime.utcnow()
            mock_records.append(record)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_records
        
        # Make request
        response = client.get(
            "/api/analytics/market/sentiment?symbols=AAPL,MSFT",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "sentiment_data" in data
        assert "timeframe" in data
        assert data["data_points"] == 2
    
    def test_get_market_sentiment_invalid_symbols(self, auth_headers):
        """Test invalid symbols"""
        response = client.get(
            "/api/analytics/market/sentiment?symbols=INVALID$SYMBOL",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_get_market_sentiment_too_many_symbols(self, auth_headers):
        """Test too many symbols"""
        symbols = ",".join([f"SYM{i}" for i in range(100)])  # 100 symbols
        response = client.get(
            f"/api/analytics/market/sentiment?symbols={symbols}",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

class TestReportGenerationAPI:
    """Test report generation API endpoints"""
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    def test_generate_report_success(
        self, mock_db, mock_get_user, mock_user, auth_headers
    ):
        """Test successful report generation"""
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.id = 1
        mock_session.query.return_value.filter.return_value.first.return_value = mock_portfolio
        
        # Mock report creation
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        # Create mock report with ID
        mock_report = Mock()
        mock_report.id = 123
        
        def set_report_id(report):
            report.id = 123
        
        mock_session.refresh.side_effect = set_report_id
        
        # Make request
        response = client.post(
            "/api/analytics/reports/generate",
            json={
                "portfolio_id": 1,
                "report_type": "performance",
                "format": "pdf"
            },
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "report_id" in data
        assert data["status"] == "queued"
        assert "download_url" in data
    
    def test_generate_report_invalid_type(self, auth_headers):
        """Test invalid report type"""
        response = client.post(
            "/api/analytics/reports/generate",
            json={
                "portfolio_id": 1,
                "report_type": "invalid_type",
                "format": "pdf"
            },
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_generate_report_missing_portfolio(self, auth_headers):
        """Test missing portfolio for portfolio-specific report"""
        response = client.post(
            "/api/analytics/reports/generate",
            json={
                "report_type": "performance",
                "format": "pdf"
                # Missing portfolio_id for performance report
            },
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    def test_get_report_status_success(
        self, mock_db, mock_get_user, mock_user, auth_headers
    ):
        """Test successful report status retrieval"""
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock report
        mock_report = Mock()
        mock_report.id = 123
        mock_report.title = "Test Report"
        mock_report.report_type = "performance"
        mock_report.status = "completed"
        mock_report.file_format = "PDF"
        mock_report.file_size_bytes = 2048
        mock_report.requested_at = datetime.utcnow()
        mock_report.completed_at = datetime.utcnow()
        mock_report.expires_at = None
        mock_report.error_message = None
        mock_report.parameters = "{}"
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_report
        
        # Make request
        response = client.get(
            "/api/analytics/reports/123",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["report_id"] == 123
        assert data["status"] == "completed"
        assert data["report_type"] == "performance"
    
    def test_get_report_status_not_found(self, auth_headers):
        """Test report not found"""
        with patch('supernova.analytics_api.get_current_user') as mock_get_user, \
             patch('supernova.analytics_api.SessionLocal') as mock_db:
            
            mock_get_user.return_value = {"id": 1}
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            response = client.get(
                "/api/analytics/reports/999",
                headers=auth_headers
            )
            
            assert response.status_code == 404

class TestBacktestAnalysisAPI:
    """Test backtest analysis API endpoints"""
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_api.SessionLocal')
    def test_get_backtest_analysis_success(
        self, mock_db, mock_get_user, mock_user, auth_headers
    ):
        """Test successful backtest analysis retrieval"""
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock backtest result
        mock_backtest = Mock()
        mock_backtest.id = 1
        mock_backtest.strategy_id = "test_strategy"
        mock_backtest.symbol = "AAPL"
        mock_backtest.timeframe = "1d"
        mock_backtest.metrics_json = json.dumps({
            "returns": [0.01, -0.005, 0.02, -0.01, 0.015],
            "total_return": 0.15,
            "sharpe_ratio": 1.2
        })
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_backtest
        mock_session.query.return_value.filter.return_value.all.return_value = []  # No analysis records
        
        # Make request
        response = client.get(
            "/api/analytics/backtests/1/analysis",
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["backtest_id"] == 1
        assert data["symbol"] == "AAPL"
        assert "statistical_significance" in data
    
    def test_get_backtest_analysis_not_found(self, auth_headers):
        """Test backtest not found"""
        with patch('supernova.analytics_api.get_current_user') as mock_get_user, \
             patch('supernova.analytics_api.SessionLocal') as mock_db:
            
            mock_get_user.return_value = {"id": 1}
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            response = client.get(
                "/api/analytics/backtests/999/analysis",
                headers=auth_headers
            )
            
            assert response.status_code == 404

class TestValidationAndErrorHandling:
    """Test validation and error handling"""
    
    def test_invalid_portfolio_id(self, auth_headers):
        """Test invalid portfolio ID validation"""
        response = client.get(
            "/api/analytics/portfolio/0/performance",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        
        response = client.get(
            "/api/analytics/portfolio/-1/performance", 
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_date_range(self, auth_headers):
        """Test invalid date range validation"""
        # End date before start date
        response = client.get(
            "/api/analytics/portfolio/1/performance?start_date=2023-12-31&end_date=2023-01-01",
            headers=auth_headers
        )
        assert response.status_code in [400, 422]
    
    def test_sql_injection_protection(self, auth_headers):
        """Test SQL injection protection"""
        # Try SQL injection in benchmark parameter
        response = client.get(
            "/api/analytics/portfolio/1/performance?benchmark=SPY'; DROP TABLE portfolios; --",
            headers=auth_headers
        )
        assert response.status_code == 422  # Should be caught by validation
    
    def test_xss_protection(self, auth_headers):
        """Test XSS protection"""
        response = client.post(
            "/api/analytics/reports/generate",
            json={
                "portfolio_id": 1,
                "report_type": "performance",
                "format": "pdf",
                "custom_title": "<script>alert('xss')</script>"
            },
            headers=auth_headers
        )
        assert response.status_code == 422  # Should be caught by validation
    
    @patch('supernova.analytics_api.get_current_user')
    @patch('supernova.analytics_engine.AdvancedAnalyticsEngine.calculate_portfolio_performance')
    def test_analytics_exception_handling(self, mock_calc, mock_get_user, auth_headers):
        """Test analytics exception handling"""
        mock_get_user.return_value = {"id": 1}
        
        # Mock analytics exception
        mock_calc.side_effect = AnalyticsException(
            message="Calculation failed",
            error_code=AnalyticsErrorCode.CALCULATION_FAILED
        )
        
        with patch('supernova.analytics_api.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock portfolio and records
            mock_portfolio = Mock()
            mock_session.query.return_value.filter.return_value.first.return_value = mock_portfolio
            mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [Mock()]
            
            response = client.get(
                "/api/analytics/portfolio/1/performance",
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data["detail"] or "Calculation failed" in data["detail"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])