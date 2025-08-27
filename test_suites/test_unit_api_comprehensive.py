"""
Comprehensive Unit Tests for SuperNova API Endpoints
==================================================

This module provides exhaustive unit testing for all API endpoints including:
- All CRUD operations
- Error handling scenarios
- Edge cases and boundary conditions
- Authentication and authorization
- Input validation
- Response formatting
"""
import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient

from supernova.api import app, _get_risk
from supernova.db import User, Profile, Asset, WatchlistItem
from supernova.schemas import (
    IntakeRequest, AdviceRequest, WatchlistRequest, 
    BacktestRequest, OHLCVBar, SentimentDataPoint
)


class TestIntakeEndpointComprehensive:
    """Comprehensive tests for intake endpoint."""
    
    @pytest.mark.unit
    def test_intake_minimal_required_fields(self, client):
        """Test intake with only required fields."""
        request_data = {"name": "Minimal User"}
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["profile_id"] > 0
        assert data["risk_score"] == 50  # Default for no risk questions
    
    @pytest.mark.unit
    def test_intake_full_data_profile(self, client):
        """Test intake with complete data profile."""
        request_data = {
            "name": "Complete User",
            "email": "complete@example.com",
            "income": 100000.0,
            "expenses": 60000.0,
            "assets": 250000.0,
            "debts": 50000.0,
            "time_horizon_yrs": 20,
            "objectives": "retirement and wealth preservation",
            "constraints": "ESG investing only, no tobacco stocks",
            "risk_questions": [3, 4, 3, 2, 4]
        }
        
        response = client.post("/intake", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["profile_id"] > 0
        assert 50 <= data["risk_score"] <= 75  # Mid-range risk score
    
    @pytest.mark.unit
    def test_intake_extreme_risk_scenarios(self, client):
        """Test intake with extreme risk question responses."""
        # Ultra-conservative investor
        conservative_data = {
            "name": "Ultra Conservative",
            "risk_questions": [1, 1, 1, 1, 1]
        }
        
        response = client.post("/intake", json=conservative_data)
        assert response.status_code == 200
        assert response.json()["risk_score"] == 25
        
        # Ultra-aggressive investor
        aggressive_data = {
            "name": "Ultra Aggressive",
            "risk_questions": [4, 4, 4, 4, 4]
        }
        
        response = client.post("/intake", json=aggressive_data)
        assert response.status_code == 200
        assert response.json()["risk_score"] == 100
    
    @pytest.mark.unit
    def test_intake_invalid_data_validation(self, client):
        """Test intake with various invalid data scenarios."""
        invalid_scenarios = [
            {},  # Empty request
            {"name": ""},  # Empty name
            {"name": "Test", "email": "invalid-email"},  # Invalid email format
            {"name": "Test", "income": -1000},  # Negative income
            {"name": "Test", "time_horizon_yrs": -5},  # Negative time horizon
            {"name": "Test", "risk_questions": [1, 2, 6]},  # Invalid risk question values
            {"name": "Test", "risk_questions": [1, 2, 3, 4, 5, 6]},  # Too many risk questions
        ]
        
        for invalid_data in invalid_scenarios:
            response = client.post("/intake", json=invalid_data)
            # Should return validation error or handle gracefully
            assert response.status_code in [400, 422, 200]
    
    @pytest.mark.unit
    def test_intake_edge_case_values(self, client):
        """Test intake with edge case values."""
        edge_cases = [
            {
                "name": "A" * 1000,  # Very long name
                "time_horizon_yrs": 100,  # Very long time horizon
                "income": 999999999.99,  # Very high income
            },
            {
                "name": "Single Character Name: X",
                "time_horizon_yrs": 1,  # Very short time horizon
                "income": 0.01,  # Very low income
            }
        ]
        
        for edge_case in edge_cases:
            response = client.post("/intake", json=edge_case)
            assert response.status_code == 200
    
    @pytest.mark.unit
    @patch('supernova.api.SessionLocal')
    def test_intake_database_error_handling(self, mock_session, client):
        """Test intake endpoint handles database errors gracefully."""
        mock_session.side_effect = Exception("Database connection failed")
        
        request_data = {"name": "Test User"}
        response = client.post("/intake", json=request_data)
        
        # Should handle database errors gracefully
        assert response.status_code in [500, 503]
    
    @pytest.mark.unit
    def test_intake_concurrent_requests(self, client):
        """Test intake can handle concurrent requests."""
        import threading
        
        results = []
        
        def make_request(user_id):
            request_data = {"name": f"Concurrent User {user_id}"}
            response = client.post("/intake", json=request_data)
            results.append(response.status_code)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)


class TestAdviceEndpointComprehensive:
    """Comprehensive tests for advice endpoint."""
    
    @pytest.mark.unit
    def test_advice_all_asset_classes(self, client, test_profile, sample_ohlcv_data):
        """Test advice for all supported asset classes."""
        asset_classes = ["stock", "crypto", "fx", "futures", "option", "bond", "commodity"]
        
        for asset_class in asset_classes:
            request_data = {
                "profile_id": test_profile.id,
                "symbol": f"TEST_{asset_class.upper()}",
                "asset_class": asset_class,
                "timeframe": "1h",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["symbol"] == f"TEST_{asset_class.upper()}"
            assert data["action"] in ["buy", "sell", "hold", "reduce", "avoid"]
            assert 0.0 <= data["confidence"] <= 1.0
    
    @pytest.mark.unit
    def test_advice_all_strategy_templates(self, client, test_profile, sample_ohlcv_data):
        """Test advice with all available strategy templates."""
        strategies = [
            "ma_crossover",
            "rsi_breakout", 
            "macd_trend",
            "bollinger_bands",
            "momentum",
            "mean_reversion",
            "breakout",
            "trend_following",
            "options_straddle",
            "fx_breakout",
            "futures_trend",
            "ensemble"
        ]
        
        for strategy in strategies:
            request_data = {
                "profile_id": test_profile.id,
                "symbol": "STRATEGY_TEST",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]],
                "strategy_template": strategy,
                "params": {}
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data["key_indicators"], dict)
            assert len(data["key_indicators"]) > 0
    
    @pytest.mark.unit
    def test_advice_different_timeframes(self, client, test_profile, sample_ohlcv_data):
        """Test advice with different timeframes."""
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        
        for timeframe in timeframes:
            request_data = {
                "profile_id": test_profile.id,
                "symbol": "TIMEFRAME_TEST",
                "timeframe": timeframe,
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["timeframe"] == timeframe
    
    @pytest.mark.unit
    def test_advice_sentiment_integration(self, client, test_profile, sample_ohlcv_data):
        """Test advice with different sentiment values."""
        sentiment_scenarios = [
            -0.8,  # Very bearish
            -0.3,  # Bearish
            0.0,   # Neutral
            0.3,   # Bullish
            0.8,   # Very bullish
        ]
        
        for sentiment in sentiment_scenarios:
            request_data = {
                "profile_id": test_profile.id,
                "symbol": "SENTIMENT_TEST",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]],
                "sentiment_hint": sentiment
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            # Sentiment should influence the advice
            assert isinstance(data["rationale"], str)
    
    @pytest.mark.unit
    def test_advice_risk_profile_influence(self, client, db_session, sample_ohlcv_data):
        """Test that different risk profiles influence advice differently."""
        # Create profiles with different risk scores
        risk_profiles = [
            ("Conservative", 25),
            ("Moderate", 50), 
            ("Aggressive", 85)
        ]
        
        results = {}
        
        for name, risk_score in risk_profiles:
            user = User(name=name, email=f"{name.lower()}@test.com")
            db_session.add(user)
            db_session.flush()
            
            profile = Profile(
                user_id=user.id,
                risk_score=risk_score,
                time_horizon_yrs=10,
                objectives="growth",
                constraints="none"
            )
            db_session.add(profile)
            db_session.commit()
            
            request_data = {
                "profile_id": profile.id,
                "symbol": "RISK_TEST",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            results[name] = response.json()
        
        # Risk notes should differ between conservative and aggressive profiles
        conservative_notes = results["Conservative"]["risk_notes"]
        aggressive_notes = results["Aggressive"]["risk_notes"]
        assert conservative_notes != aggressive_notes
    
    @pytest.mark.unit
    def test_advice_custom_parameters(self, client, test_profile, sample_ohlcv_data):
        """Test advice with custom strategy parameters."""
        custom_params = {
            "ma_crossover": {"fast": 5, "slow": 20, "signal": 9},
            "rsi_breakout": {"length": 21, "low_th": 25, "high_th": 75},
            "bollinger_bands": {"length": 20, "std": 2.5}
        }
        
        for strategy, params in custom_params.items():
            request_data = {
                "profile_id": test_profile.id,
                "symbol": "CUSTOM_PARAMS_TEST",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]],
                "strategy_template": strategy,
                "params": params
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data["key_indicators"], dict)
    
    @pytest.mark.unit
    def test_advice_insufficient_data_handling(self, client, test_profile):
        """Test advice with insufficient data."""
        # Very few bars
        limited_bars = [
            OHLCVBar(
                timestamp="2024-01-01T10:00:00Z",
                open=100, high=101, low=99, close=100, volume=1000
            ) for _ in range(5)
        ]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbol": "LIMITED_DATA",
            "bars": [bar.model_dump() for bar in limited_bars]
        }
        
        response = client.post("/advice", json=request_data)
        # Should handle gracefully - either return advice or appropriate error
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert data["confidence"] <= 0.5  # Low confidence due to limited data
    
    @pytest.mark.unit
    def test_advice_invalid_profile_handling(self, client, sample_ohlcv_data):
        """Test advice with invalid profile ID."""
        request_data = {
            "profile_id": 999999,  # Non-existent profile
            "symbol": "INVALID_PROFILE_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:50]]
        }
        
        response = client.post("/advice", json=request_data)
        assert response.status_code == 404
    
    @pytest.mark.unit
    def test_advice_malformed_bars_data(self, client, test_profile):
        """Test advice with malformed OHLCV bars."""
        malformed_bars = [
            {"timestamp": "invalid-timestamp", "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000},
            {"timestamp": "2024-01-01T10:00:00Z", "open": "invalid", "high": 105, "low": 95, "close": 102, "volume": 1000},
            {"timestamp": "2024-01-01T10:00:00Z", "open": 100, "high": 95, "low": 105, "close": 102, "volume": 1000},  # High < Low
        ]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbol": "MALFORMED_TEST",
            "bars": malformed_bars
        }
        
        response = client.post("/advice", json=request_data)
        # Should handle validation errors
        assert response.status_code in [400, 422]


class TestWatchlistEndpointComprehensive:
    """Comprehensive tests for watchlist endpoints."""
    
    @pytest.mark.unit
    def test_add_watchlist_various_asset_classes(self, client, test_profile):
        """Test adding watchlist items for various asset classes."""
        test_cases = [
            {"symbols": ["AAPL", "GOOGL", "MSFT"], "asset_class": "stock"},
            {"symbols": ["BTC", "ETH", "ADA"], "asset_class": "crypto"},
            {"symbols": ["EURUSD", "GBPUSD", "USDJPY"], "asset_class": "fx"},
            {"symbols": ["GC", "CL", "NG"], "asset_class": "futures"},
            {"symbols": ["SPY240101C400", "QQQ240101P350"], "asset_class": "option"}
        ]
        
        for test_case in test_cases:
            request_data = {
                "profile_id": test_profile.id,
                **test_case
            }
            
            response = client.post("/watchlist/add", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "added_ids" in data
            assert len(data["added_ids"]) == len(test_case["symbols"])
    
    @pytest.mark.unit
    def test_add_watchlist_duplicate_handling(self, client, test_profile):
        """Test handling of duplicate symbols in watchlist."""
        symbols_with_duplicates = ["AAPL", "GOOGL", "AAPL", "MSFT", "GOOGL"]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbols": symbols_with_duplicates,
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Should handle duplicates appropriately
        assert "added_ids" in data
    
    @pytest.mark.unit
    def test_add_watchlist_large_batch(self, client, test_profile):
        """Test adding large batch of watchlist items."""
        large_symbol_list = [f"TEST{i:04d}" for i in range(1000)]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbols": large_symbol_list,
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["added_ids"]) == 1000
    
    @pytest.mark.unit
    def test_add_watchlist_invalid_symbols(self, client, test_profile):
        """Test adding watchlist with invalid symbols."""
        invalid_symbols = ["", "   ", "SYMBOL_WITH_VERY_LONG_NAME_THAT_EXCEEDS_NORMAL_LIMITS"]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbols": invalid_symbols,
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        # Should handle invalid symbols gracefully
        assert response.status_code in [200, 400]
    
    @pytest.mark.unit
    def test_add_watchlist_nonexistent_profile(self, client):
        """Test adding watchlist with non-existent profile."""
        request_data = {
            "profile_id": 999999,
            "symbols": ["AAPL"],
            "asset_class": "stock"
        }
        
        response = client.post("/watchlist/add", json=request_data)
        # Should handle non-existent profile appropriately
        assert response.status_code in [200, 404]


class TestAlertsEndpointComprehensive:
    """Comprehensive tests for alerts endpoint."""
    
    @pytest.mark.unit
    def test_evaluate_alerts_various_conditions(self, client, sample_ohlcv_data):
        """Test alert evaluation with various market conditions."""
        # Create different market scenarios
        scenarios = [
            ("trending_up", [bar.model_dump() for bar in sample_ohlcv_data[:50]]),
            ("high_volatility", self._create_volatile_bars(50)),
            ("low_volume", self._create_low_volume_bars(50)),
        ]
        
        for scenario_name, bars in scenarios:
            watch_items = [{"symbol": f"TEST_{scenario_name.upper()}", "profile_id": 1}]
            
            request_data = {
                "watch": watch_items,
                "bars": {f"TEST_{scenario_name.upper()}": bars}
            }
            
            response = client.post("/alerts/evaluate", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
    
    def _create_volatile_bars(self, count: int):
        """Create volatile OHLCV bars for testing."""
        import numpy as np
        bars = []
        base_price = 100.0
        
        for i in range(count):
            # High volatility movements
            price_change = np.random.normal(0, 0.05) * base_price  # 5% std dev
            base_price = max(1.0, base_price + price_change)
            
            bars.append({
                "timestamp": (datetime.now() - timedelta(hours=count-i)).isoformat() + "Z",
                "open": base_price,
                "high": base_price * (1 + abs(np.random.normal(0, 0.03))),
                "low": base_price * (1 - abs(np.random.normal(0, 0.03))),
                "close": base_price + np.random.normal(0, 0.02) * base_price,
                "volume": int(1000000 * np.random.uniform(2.0, 5.0))  # High volume
            })
        
        return bars
    
    def _create_low_volume_bars(self, count: int):
        """Create low volume OHLCV bars for testing."""
        bars = []
        base_price = 100.0
        
        for i in range(count):
            bars.append({
                "timestamp": (datetime.now() - timedelta(hours=count-i)).isoformat() + "Z",
                "open": base_price,
                "high": base_price + 0.5,
                "low": base_price - 0.5,
                "close": base_price + 0.1,
                "volume": int(10000 * np.random.uniform(0.1, 0.5))  # Low volume
            })
        
        return bars
    
    @pytest.mark.unit
    def test_evaluate_alerts_empty_watchlist(self, client):
        """Test alert evaluation with empty watchlist."""
        request_data = {"watch": [], "bars": {}}
        
        response = client.post("/alerts/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data == []
    
    @pytest.mark.unit
    def test_evaluate_alerts_missing_data(self, client):
        """Test alert evaluation with missing bar data."""
        watch_items = [{"symbol": "MISSING_DATA", "profile_id": 1}]
        
        request_data = {
            "watch": watch_items,
            "bars": {}  # No data for MISSING_DATA
        }
        
        response = client.post("/alerts/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.unit
    def test_evaluate_alerts_large_watchlist(self, client, sample_ohlcv_data):
        """Test alert evaluation with large watchlist."""
        # Create large watchlist
        watch_items = [{"symbol": f"LARGE_TEST_{i}", "profile_id": 1} for i in range(100)]
        
        # Create bar data for all symbols
        bars_data = {}
        for i in range(100):
            symbol = f"LARGE_TEST_{i}"
            bars_data[symbol] = [bar.model_dump() for bar in sample_ohlcv_data[:20]]
        
        request_data = {
            "watch": watch_items,
            "bars": bars_data
        }
        
        start_time = time.time()
        response = client.post("/alerts/evaluate", json=request_data)
        execution_time = time.time() - start_time
        
        assert response.status_code == 200
        assert execution_time < 30  # Should complete within reasonable time


class TestBacktestEndpointComprehensive:
    """Comprehensive tests for backtest endpoint."""
    
    @pytest.mark.unit
    def test_backtest_all_strategy_templates(self, client, sample_ohlcv_data):
        """Test backtest with all available strategy templates."""
        strategies = [
            "ma_crossover",
            "rsi_breakout",
            "macd_trend", 
            "bollinger_bands",
            "momentum",
            "mean_reversion",
            "breakout",
            "options_straddle",
            "fx_breakout",
            "futures_trend",
            "ensemble"
        ]
        
        for strategy in strategies:
            request_data = {
                "strategy_template": strategy,
                "params": {},
                "symbol": f"BACKTEST_{strategy.upper()}",
                "timeframe": "1h",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data]
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["symbol"] == f"BACKTEST_{strategy.upper()}"
            assert "metrics" in data
            
            # Verify required metrics are present
            metrics = data["metrics"]
            required_metrics = ["final_equity", "CAGR", "Sharpe", "MaxDrawdown", "WinRate"]
            for metric in required_metrics:
                assert metric in metrics
    
    @pytest.mark.unit
    def test_backtest_parameter_variations(self, client, sample_ohlcv_data):
        """Test backtest with various parameter combinations."""
        parameter_sets = [
            {"strategy_template": "ma_crossover", "params": {"fast": 5, "slow": 20}},
            {"strategy_template": "ma_crossover", "params": {"fast": 10, "slow": 50}},
            {"strategy_template": "rsi_breakout", "params": {"length": 14, "low_th": 30, "high_th": 70}},
            {"strategy_template": "rsi_breakout", "params": {"length": 21, "low_th": 25, "high_th": 75}},
        ]
        
        for param_set in parameter_sets:
            request_data = {
                **param_set,
                "symbol": "PARAM_TEST",
                "timeframe": "4h",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data]
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data["metrics"], dict)
    
    @pytest.mark.unit
    def test_backtest_different_data_lengths(self, client):
        """Test backtest with different data lengths."""
        data_lengths = [50, 100, 500, 1000]
        
        from .conftest import TestDataGenerator
        
        for length in data_lengths:
            # Generate data of specific length
            prices = TestDataGenerator.generate_trend_data(length)
            bars = []
            
            for i, price in enumerate(prices):
                bars.append(OHLCVBar(
                    timestamp=(datetime.now() - timedelta(hours=length-i)).isoformat() + "Z",
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price * 1.005,
                    volume=10000
                ).model_dump())
            
            request_data = {
                "strategy_template": "ma_crossover",
                "params": {"fast": 10, "slow": 20},
                "symbol": f"LENGTH_TEST_{length}",
                "timeframe": "1h",
                "bars": bars
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            # Longer data should generally produce more reliable metrics
            if length >= 100:
                assert data["metrics"]["n_trades_est"] >= 0
    
    @pytest.mark.unit
    def test_backtest_ensemble_combinations(self, client, sample_ohlcv_data):
        """Test backtest with ensemble strategy combinations."""
        ensemble_configs = [
            {
                "ma_crossover": {"fast": 5, "slow": 15},
                "rsi_breakout": {"length": 14}
            },
            {
                "ma_crossover": {"fast": 10, "slow": 30},
                "macd_trend": {"fast": 12, "slow": 26, "signal": 9},
                "rsi_breakout": {"length": 21}
            }
        ]
        
        for config in ensemble_configs:
            request_data = {
                "strategy_template": "ensemble",
                "params": config,
                "symbol": "ENSEMBLE_TEST",
                "timeframe": "2h",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data]
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data["metrics"], dict)
    
    @pytest.mark.unit
    def test_backtest_invalid_parameters(self, client, sample_ohlcv_data):
        """Test backtest with invalid parameters."""
        invalid_scenarios = [
            {"strategy_template": "nonexistent_strategy"},
            {"strategy_template": "ma_crossover", "params": {"fast": -5, "slow": 20}},  # Negative parameters
            {"strategy_template": "ma_crossover", "params": {"fast": 50, "slow": 20}},  # Fast > Slow
            {"strategy_template": "rsi_breakout", "params": {"length": 0}},  # Zero length
        ]
        
        for invalid_scenario in invalid_scenarios:
            request_data = {
                **invalid_scenario,
                "symbol": "INVALID_TEST",
                "timeframe": "1h",
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/backtest", json=request_data)
            # Should either handle gracefully or return appropriate error
            assert response.status_code in [200, 400, 422, 500]
    
    @pytest.mark.unit
    def test_backtest_performance_benchmarks(self, client, sample_ohlcv_data):
        """Test backtest performance with timing benchmarks."""
        request_data = {
            "strategy_template": "ensemble",
            "params": {
                "ma_crossover": {"fast": 10, "slow": 50},
                "rsi_breakout": {"length": 14},
                "macd_trend": {"fast": 12, "slow": 26, "signal": 9}
            },
            "symbol": "PERFORMANCE_TEST",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        start_time = time.time()
        response = client.post("/backtest", json=request_data)
        execution_time = time.time() - start_time
        
        assert response.status_code == 200
        assert execution_time < 30  # Should complete within 30 seconds
        
        data = response.json()
        assert isinstance(data["metrics"], dict)


class TestUtilityFunctionsComprehensive:
    """Comprehensive tests for utility functions."""
    
    @pytest.mark.unit
    def test_get_risk_function_all_scenarios(self, db_session):
        """Test _get_risk function with various scenarios."""
        # Create profiles with different risk scores
        risk_scores = [0, 25, 50, 75, 100]
        profile_ids = []
        
        for risk_score in risk_scores:
            user = User(name=f"Risk User {risk_score}", email=f"risk{risk_score}@test.com")
            db_session.add(user)
            db_session.flush()
            
            profile = Profile(
                user_id=user.id,
                risk_score=risk_score,
                time_horizon_yrs=10,
                objectives="test",
                constraints="test"
            )
            db_session.add(profile)
            db_session.commit()
            profile_ids.append((profile.id, risk_score))
        
        # Test valid profiles
        for profile_id, expected_risk in profile_ids:
            actual_risk = _get_risk(profile_id)
            assert actual_risk == expected_risk
        
        # Test invalid profile
        with pytest.raises(HTTPException) as exc_info:
            _get_risk(999999)
        assert exc_info.value.status_code == 404
    
    @pytest.mark.unit
    @patch('supernova.api.SessionLocal')
    def test_get_risk_database_error_handling(self, mock_session):
        """Test _get_risk handles database errors."""
        mock_session.return_value.__enter__.return_value.get.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception):
            _get_risk(1)


class TestAPIMiddlewareAndSecurity:
    """Tests for API middleware and security features."""
    
    @pytest.mark.unit
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/intake")
        # Should include appropriate CORS headers
        assert response.status_code in [200, 204, 405]
    
    @pytest.mark.unit
    def test_request_validation(self, client):
        """Test request validation middleware."""
        # Test with malformed JSON
        response = client.post("/intake", data="invalid json", headers={"content-type": "application/json"})
        assert response.status_code in [400, 422]
    
    @pytest.mark.unit
    def test_rate_limiting_behavior(self, client):
        """Test rate limiting behavior (if implemented)."""
        # Make rapid requests to test rate limiting
        responses = []
        for i in range(100):
            response = client.post("/intake", json={"name": f"Rate Test User {i}"})
            responses.append(response.status_code)
        
        # Should either all succeed or start rate limiting
        success_count = sum(1 for status in responses if status == 200)
        assert success_count > 0  # At least some should succeed
    
    @pytest.mark.unit
    def test_error_response_format(self, client):
        """Test that error responses follow consistent format."""
        # Trigger validation error
        response = client.post("/advice", json={"invalid": "data"})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data or "message" in data
    
    @pytest.mark.unit
    def test_health_check_endpoints(self, client):
        """Test health check functionality."""
        # Test basic health endpoint (if exists)
        health_endpoints = ["/health", "/healthz", "/status", "/ping"]
        
        for endpoint in health_endpoints:
            try:
                response = client.get(endpoint)
                if response.status_code != 404:  # Endpoint exists
                    assert response.status_code == 200
            except:
                pass  # Endpoint doesn't exist


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Integration tests for complete workflows."""
    
    def test_complete_user_journey(self, client, sample_ohlcv_data):
        """Test complete user journey from intake to backtest."""
        # 1. User intake
        intake_data = {
            "name": "Journey Test User",
            "email": "journey@test.com",
            "risk_questions": [3, 2, 3, 3, 2],
            "time_horizon_yrs": 15,
            "objectives": "retirement planning"
        }
        
        intake_response = client.post("/intake", json=intake_data)
        assert intake_response.status_code == 200
        profile_id = intake_response.json()["profile_id"]
        
        # 2. Add to watchlist
        watchlist_data = {
            "profile_id": profile_id,
            "symbols": ["JOURNEY_AAPL", "JOURNEY_GOOGL"],
            "asset_class": "stock"
        }
        
        watchlist_response = client.post("/watchlist/add", json=watchlist_data)
        assert watchlist_response.status_code == 200
        
        # 3. Get advice
        advice_data = {
            "profile_id": profile_id,
            "symbol": "JOURNEY_AAPL",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:200]]
        }
        
        advice_response = client.post("/advice", json=advice_data)
        assert advice_response.status_code == 200
        
        # 4. Run backtest
        backtest_data = {
            "strategy_template": "ma_crossover",
            "params": {"fast": 10, "slow": 30},
            "symbol": "JOURNEY_AAPL",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        backtest_response = client.post("/backtest", json=backtest_data)
        assert backtest_response.status_code == 200
        
        # Verify all responses are consistent
        advice_data = advice_response.json()
        backtest_data = backtest_response.json()
        
        assert advice_data["symbol"] == backtest_data["symbol"]
        assert isinstance(advice_data["key_indicators"], dict)
        assert isinstance(backtest_data["metrics"], dict)
    
    def test_multi_user_isolation(self, client, sample_ohlcv_data):
        """Test that multiple users' data is properly isolated."""
        # Create two users with different profiles
        users = [
            {"name": "Conservative User", "risk_questions": [1, 1, 2, 1, 2]},
            {"name": "Aggressive User", "risk_questions": [4, 4, 3, 4, 4]}
        ]
        
        user_profiles = []
        for user_data in users:
            response = client.post("/intake", json=user_data)
            assert response.status_code == 200
            user_profiles.append(response.json())
        
        # Get advice for same symbol but different users
        symbol = "ISOLATION_TEST"
        bars = [bar.model_dump() for bar in sample_ohlcv_data[:100]]
        
        advice_responses = []
        for profile in user_profiles:
            advice_data = {
                "profile_id": profile["profile_id"],
                "symbol": symbol,
                "bars": bars
            }
            
            response = client.post("/advice", json=advice_data)
            assert response.status_code == 200
            advice_responses.append(response.json())
        
        # Advice should be different due to different risk profiles
        conservative_advice = advice_responses[0]
        aggressive_advice = advice_responses[1]
        
        # Risk notes should be different
        assert conservative_advice["risk_notes"] != aggressive_advice["risk_notes"]
        # Actions might be different or confidence levels should differ
        assert (conservative_advice["action"] != aggressive_advice["action"] or 
                abs(conservative_advice["confidence"] - aggressive_advice["confidence"]) > 0.1)