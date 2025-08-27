import pytest
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from supernova.api import app, _get_risk
from supernova.db import Base, User, Profile, Asset, WatchlistItem
from supernova.schemas import (
    IntakeRequest, AdviceRequest, WatchlistRequest, 
    BacktestRequest, OHLCVBar
)


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_supernova.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=test_engine)
    
    # Override the SessionLocal in the api module
    import supernova.api
    original_session = supernova.api.SessionLocal
    supernova.api.SessionLocal = TestSessionLocal
    
    yield TestSessionLocal()
    
    # Restore original session and clean up
    supernova.api.SessionLocal = original_session
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db_session):
    """Create test client with database override."""
    return TestClient(app)


@pytest.fixture
def sample_bars():
    """Generate sample OHLCV bars for testing."""
    bars = []
    base_price = 100.0
    
    for i in range(100):
        timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
        price_change = (i - 50) * 0.1  # Trending up
        base_price += price_change
        
        bars.append(OHLCVBar(
            timestamp=timestamp,
            open=base_price,
            high=base_price + 1.0,
            low=base_price - 1.0,
            close=base_price + 0.5,
            volume=10000
        ))
    
    return bars


@pytest.fixture
def sample_user_profile(db_session):
    """Create a sample user and profile for testing."""
    db = db_session
    
    user = User(name="Test User", email="test@example.com")
    db.add(user)
    db.flush()
    
    profile = Profile(
        user_id=user.id,
        risk_score=60,
        time_horizon_yrs=5,
        objectives="growth",
        constraints="no crypto"
    )
    db.add(profile)
    db.commit()
    
    return {"user_id": user.id, "profile_id": profile.id}


class TestIntakeEndpoint:
    def test_intake_basic_functionality(self, client):
        """Test basic intake endpoint functionality."""
        request_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "time_horizon_yrs": 10,
            "objectives": "retirement",
            "constraints": "no high risk investments",
            "risk_questions": [2, 3, 2, 4, 2]
        }
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "profile_id" in data
        assert "risk_score" in data
        assert isinstance(data["profile_id"], int)
        assert isinstance(data["risk_score"], int)
        assert 0 <= data["risk_score"] <= 100

    def test_intake_minimal_data(self, client):
        """Test intake with minimal required data."""
        request_data = {
            "name": "Jane Smith"
        }
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "profile_id" in data
        assert data["risk_score"] == 50  # Default score for empty risk questions

    def test_intake_with_financial_data(self, client):
        """Test intake with comprehensive financial data."""
        request_data = {
            "name": "Bob Johnson",
            "email": "bob@example.com",
            "income": 75000.0,
            "expenses": 45000.0,
            "assets": 150000.0,
            "debts": 25000.0,
            "time_horizon_yrs": 15,
            "objectives": "wealth accumulation",
            "constraints": "ESG investing only",
            "risk_questions": [4, 4, 3, 4, 3]
        }
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["risk_score"] > 70  # High risk questions should yield high score

    def test_intake_empty_name_error(self, client):
        """Test intake fails with empty name."""
        request_data = {}
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_intake_extreme_risk_scores(self, client):
        """Test intake with extreme risk question responses."""
        # Very conservative
        conservative_data = {
            "name": "Conservative Investor",
            "risk_questions": [1, 1, 1, 1, 1]
        }
        
        response = client.post("/intake", json=conservative_data)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] == 25
        
        # Very aggressive
        aggressive_data = {
            "name": "Aggressive Investor",
            "risk_questions": [4, 4, 4, 4, 4]
        }
        
        response = client.post("/intake", json=aggressive_data)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] == 100

    def test_intake_duplicate_users(self, client):
        """Test that duplicate users can be created (business rule)."""
        request_data = {
            "name": "Duplicate User",
            "email": "duplicate@example.com"
        }
        
        # Create first user
        response1 = client.post("/intake", json=request_data)
        assert response1.status_code == 200
        
        # Create second user with same data
        response2 = client.post("/intake", json=request_data)
        assert response2.status_code == 200
        
        # Should have different profile IDs
        assert response1.json()["profile_id"] != response2.json()["profile_id"]


class TestAdviceEndpoint:
    def test_advice_basic_functionality(self, client, sample_user_profile, sample_bars):
        """Test basic advice endpoint functionality."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbol": "AAPL",
            "asset_class": "stock",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["timeframe"] == "1h"
        assert data["action"] in ["buy", "sell", "hold", "reduce", "avoid"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["rationale"], str)
        assert isinstance(data["key_indicators"], dict)
        assert isinstance(data["risk_notes"], str)

    def test_advice_with_sentiment(self, client, sample_user_profile, sample_bars):
        """Test advice with sentiment hint."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbol": "TSLA",
            "bars": [bar.model_dump() for bar in sample_bars],
            "sentiment_hint": 0.3  # Positive sentiment
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "TSLA"
        assert 0.0 <= data["confidence"] <= 1.0

    def test_advice_with_strategy_template(self, client, sample_user_profile, sample_bars):
        """Test advice with specific strategy template."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbol": "BTC",
            "asset_class": "crypto",
            "bars": [bar.model_dump() for bar in sample_bars],
            "strategy_template": "ma_crossover",
            "params": {"fast": 5, "slow": 20}
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTC"
        # Should contain MA-specific indicators
        assert "fast" in data["key_indicators"] or "slow" in data["key_indicators"]

    def test_advice_invalid_profile(self, client, sample_bars):
        """Test advice with invalid profile ID."""
        request_data = {
            "profile_id": 99999,  # Non-existent profile
            "symbol": "AAPL",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 404

    def test_advice_empty_bars(self, client, sample_user_profile):
        """Test advice with empty bars list."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbol": "AAPL",
            "bars": []
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 500  # Should fail with empty bars

    def test_advice_different_asset_classes(self, client, sample_user_profile, sample_bars):
        """Test advice for different asset classes."""
        asset_classes = ["stock", "crypto", "fx", "futures", "option"]
        
        for asset_class in asset_classes:
            request_data = {
                "profile_id": sample_user_profile["profile_id"],
                "symbol": f"TEST_{asset_class.upper()}",
                "asset_class": asset_class,
                "bars": [bar.model_dump() for bar in sample_bars[:50]]  # Smaller dataset
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == f"TEST_{asset_class.upper()}"

    def test_advice_invalid_strategy_template(self, client, sample_user_profile, sample_bars):
        """Test advice with invalid strategy template."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbol": "AAPL",
            "bars": [bar.model_dump() for bar in sample_bars],
            "strategy_template": "nonexistent_strategy"
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 200  # Should fall back to ensemble
        
        data = response.json()
        # Should have multiple indicators from ensemble
        assert len(data["key_indicators"]) > 1


class TestWatchlistEndpoint:
    def test_add_watchlist_basic(self, client, sample_user_profile):
        """Test basic watchlist addition."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "added_ids" in data
        assert len(data["added_ids"]) == 3

    def test_add_watchlist_duplicate_symbols(self, client, sample_user_profile):
        """Test adding duplicate symbols to watchlist."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbols": ["AAPL", "AAPL", "GOOGL"],
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["added_ids"]) == 3  # Should create multiple entries

    def test_add_watchlist_crypto_assets(self, client, sample_user_profile):
        """Test adding crypto assets to watchlist."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbols": ["BTC", "ETH", "ADA"],
            "asset_class": "crypto"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200

    def test_add_watchlist_empty_symbols(self, client, sample_user_profile):
        """Test adding empty symbols list."""
        request_data = {
            "profile_id": sample_user_profile["profile_id"],
            "symbols": [],
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["added_ids"] == []

    def test_add_watchlist_invalid_profile(self, client):
        """Test adding watchlist with invalid profile."""
        request_data = {
            "profile_id": 99999,
            "symbols": ["AAPL"],
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        # Should succeed but create orphaned watchlist items
        assert response.status_code == 200


class TestAlertsEndpoint:
    def test_evaluate_alerts_basic(self, client):
        """Test basic alert evaluation."""
        # Create sample watchlist and bars data
        watch_items = [
            {"symbol": "AAPL", "profile_id": 1},
            {"symbol": "GOOGL", "profile_id": 1}
        ]
        
        bars_data = {
            "AAPL": [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "open": 150, "high": 155, "low": 149, "close": 152, "volume": 1000000
                }
            ] * 50,  # Repeat for sufficient data
            "GOOGL": [
                {
                    "timestamp": "2024-01-01T10:00:00Z", 
                    "open": 2500, "high": 2520, "low": 2480, "close": 2510, "volume": 500000
                }
            ] * 50
        }
        
        request_data = {
            "watch": watch_items,
            "bars": bars_data
        }
        
        response = client.post("/alerts/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Each alert should have required fields
        for alert in data:
            assert "id" in alert
            assert "symbol" in alert
            assert "message" in alert
            assert "triggered_at" in alert

    def test_evaluate_alerts_empty_watchlist(self, client):
        """Test alert evaluation with empty watchlist."""
        request_data = {
            "watch": [],
            "bars": {}
        }
        
        response = client.post("/alerts/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data == []

    def test_evaluate_alerts_missing_bars(self, client):
        """Test alert evaluation with missing bar data."""
        watch_items = [{"symbol": "MISSING", "profile_id": 1}]
        
        request_data = {
            "watch": watch_items,
            "bars": {}  # No bar data for MISSING symbol
        }
        
        response = client.post("/alerts/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data == []  # No alerts should trigger

    def test_evaluate_alerts_malformed_data(self, client):
        """Test alert evaluation with malformed data."""
        request_data = {
            "watch": "invalid",  # Should be list
            "bars": "invalid"    # Should be dict
        }
        
        response = client.post("/alerts/evaluate", json=request_data)
        # Should handle gracefully or return error
        assert response.status_code in [200, 422, 500]


class TestBacktestEndpoint:
    def test_backtest_basic_functionality(self, client, sample_bars):
        """Test basic backtest endpoint functionality."""
        request_data = {
            "strategy_template": "ma_crossover",
            "params": {"fast": 10, "slow": 20},
            "symbol": "AAPL",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/backtest", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["timeframe"] == "1h"
        assert isinstance(data["metrics"], dict)
        
        # Check required metrics
        metrics = data["metrics"]
        assert "final_equity" in metrics
        assert "CAGR" in metrics
        assert "Sharpe" in metrics
        assert "MaxDrawdown" in metrics
        assert "WinRate" in metrics

    def test_backtest_all_strategies(self, client, sample_bars):
        """Test backtest with all available strategies."""
        strategies = ["ma_crossover", "rsi_breakout", "macd_trend", 
                     "options_straddle", "fx_breakout", "futures_trend"]
        
        for strategy in strategies:
            request_data = {
                "strategy_template": strategy,
                "params": {},
                "symbol": "TEST",
                "timeframe": "1h",
                "bars": [bar.model_dump() for bar in sample_bars]
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["symbol"] == "TEST"
            assert isinstance(data["metrics"], dict)

    def test_backtest_ensemble_strategy(self, client, sample_bars):
        """Test backtest with ensemble strategy."""
        request_data = {
            "strategy_template": "ensemble",
            "params": {
                "ma_crossover": {"fast": 5, "slow": 15},
                "rsi_breakout": {"length": 10}
            },
            "symbol": "ENSEMBLE_TEST",
            "timeframe": "4h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/backtest", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "ENSEMBLE_TEST"

    def test_backtest_invalid_strategy(self, client, sample_bars):
        """Test backtest with invalid strategy."""
        request_data = {
            "strategy_template": "invalid_strategy",
            "params": {},
            "symbol": "AAPL",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/backtest", json=request_data)
        assert response.status_code == 500  # Should fail with invalid strategy

    def test_backtest_insufficient_data(self, client):
        """Test backtest with insufficient data."""
        # Only a few bars - insufficient for backtesting
        limited_bars = [
            OHLCVBar(
                timestamp="2024-01-01T10:00:00Z",
                open=100, high=101, low=99, close=100, volume=1000
            ) for _ in range(10)
        ]
        
        request_data = {
            "strategy_template": "ma_crossover",
            "params": {},
            "symbol": "LIMITED",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in limited_bars]
        }
        
        response = client.post("/backtest", json=request_data)
        assert response.status_code == 200  # Should handle gracefully
        
        data = response.json()
        metrics = data["metrics"]
        # With insufficient data, no trades should occur
        assert metrics["final_equity"] == 10000.0  # Starting cash
        assert metrics["n_trades_est"] == 0

    def test_backtest_custom_parameters(self, client, sample_bars):
        """Test backtest with custom strategy parameters."""
        request_data = {
            "strategy_template": "rsi_breakout",
            "params": {
                "length": 21,
                "low_th": 25,
                "high_th": 75
            },
            "symbol": "CUSTOM_RSI",
            "timeframe": "2h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        response = client.post("/backtest", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["metrics"], dict)


class TestUtilityFunctions:
    def test_get_risk_valid_profile(self, db_session, sample_user_profile):
        """Test _get_risk with valid profile."""
        risk_score = _get_risk(sample_user_profile["profile_id"])
        assert risk_score == 60  # From fixture

    def test_get_risk_invalid_profile(self, db_session):
        """Test _get_risk with invalid profile."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            _get_risk(99999)
        
        assert exc_info.value.status_code == 404


class TestEndpointIntegration:
    def test_full_workflow_integration(self, client, sample_bars):
        """Test complete workflow from intake to advice."""
        # 1. Create user profile
        intake_data = {
            "name": "Integration Test User",
            "email": "integration@test.com",
            "risk_questions": [3, 3, 3, 2, 3]
        }
        
        intake_response = client.post("/intake", json=intake_data)
        assert intake_response.status_code == 200
        profile_id = intake_response.json()["profile_id"]
        
        # 2. Add to watchlist
        watchlist_data = {
            "profile_id": profile_id,
            "symbols": ["INTEG", "TEST"],
            "asset_class": "stock"
        }
        
        watchlist_response = client.post("/watchlist/add", json=watchlist_data)
        assert watchlist_response.status_code == 200
        
        # 3. Get advice
        advice_data = {
            "profile_id": profile_id,
            "symbol": "INTEG",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        advice_response = client.post("/advice", json=advice_data)
        assert advice_response.status_code == 200
        
        # 4. Run backtest
        backtest_data = {
            "strategy_template": "ma_crossover",
            "params": {},
            "symbol": "INTEG",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_bars]
        }
        
        backtest_response = client.post("/backtest", json=backtest_data)
        assert backtest_response.status_code == 200
        
        # All steps should complete successfully
        assert all([
            intake_response.status_code == 200,
            watchlist_response.status_code == 200,
            advice_response.status_code == 200,
            backtest_response.status_code == 200
        ])

    def test_multiple_users_isolation(self, client, sample_bars):
        """Test that multiple users don't interfere with each other."""
        # Create two users
        user1_data = {"name": "User One", "risk_questions": [1, 2, 1, 2, 1]}
        user2_data = {"name": "User Two", "risk_questions": [4, 4, 3, 4, 4]}
        
        user1_response = client.post("/intake", json=user1_data)
        user2_response = client.post("/intake", json=user2_data)
        
        profile1_id = user1_response.json()["profile_id"]
        profile2_id = user2_response.json()["profile_id"]
        
        # Get advice for both users with same data
        advice_data_base = {
            "symbol": "ISOLATION_TEST",
            "bars": [bar.model_dump() for bar in sample_bars[:50]]
        }
        
        advice1_data = {**advice_data_base, "profile_id": profile1_id}
        advice2_data = {**advice_data_base, "profile_id": profile2_id}
        
        advice1_response = client.post("/advice", json=advice1_data)
        advice2_response = client.post("/advice", json=advice2_data)
        
        assert advice1_response.status_code == 200
        assert advice2_response.status_code == 200
        
        # Risk notes should be different due to different risk profiles
        advice1 = advice1_response.json()
        advice2 = advice2_response.json()
        
        # User1 is conservative, User2 is aggressive
        assert "Conservative" in advice1["risk_notes"] or advice1["risk_notes"] != advice2["risk_notes"]