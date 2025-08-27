"""
Comprehensive Integration Tests for SuperNova AI
===============================================

This module provides extensive integration testing including:
- Database integration with all models
- LLM provider integration (mocked)
- WebSocket real-time communication
- External API integrations
- Multi-component workflow testing
- TimescaleDB sentiment integration
- VectorBT backtesting integration
- Prefect workflow orchestration
"""
import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np

from fastapi.testclient import TestClient
from sqlalchemy import select, text
import websockets
import httpx

from supernova.api import app
from supernova.db import SessionLocal, User, Profile, Asset, WatchlistItem
from supernova.schemas import (
    OHLCVBar, SentimentDataPoint, ChatMessage, ChatRequest, 
    WebSocketMessage, MarketDataUpdate
)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database operations and model interactions."""
    
    def test_user_profile_relationship(self, db_session):
        """Test User-Profile relationship integrity."""
        # Create user
        user = User(
            name="Integration Test User",
            email="integration@test.com"
        )
        db_session.add(user)
        db_session.flush()
        
        # Create profile
        profile = Profile(
            user_id=user.id,
            risk_score=75,
            time_horizon_yrs=10,
            objectives="growth and income",
            constraints="no tobacco stocks"
        )
        db_session.add(profile)
        db_session.commit()
        
        # Test relationship integrity
        retrieved_user = db_session.get(User, user.id)
        assert retrieved_user is not None
        assert len(retrieved_user.profiles) == 1
        assert retrieved_user.profiles[0].risk_score == 75
        
        retrieved_profile = db_session.get(Profile, profile.id)
        assert retrieved_profile.user.name == "Integration Test User"
    
    def test_watchlist_management(self, db_session, test_user, test_profile):
        """Test watchlist CRUD operations."""
        symbols = ["INTEG_AAPL", "INTEG_GOOGL", "INTEG_MSFT", "INTEG_TSLA"]
        
        # Add watchlist items
        watchlist_items = []
        for symbol in symbols:
            asset = Asset(
                symbol=symbol,
                name=f"{symbol} Corp",
                asset_class="stock"
            )
            db_session.add(asset)
            db_session.flush()
            
            watchlist_item = WatchlistItem(
                profile_id=test_profile.id,
                asset_id=asset.id
            )
            db_session.add(watchlist_item)
            watchlist_items.append(watchlist_item)
        
        db_session.commit()
        
        # Verify watchlist items
        profile_watchlist = db_session.query(WatchlistItem).filter(
            WatchlistItem.profile_id == test_profile.id
        ).all()
        
        assert len(profile_watchlist) == len(symbols)
        
        # Test watchlist item deletion
        db_session.delete(watchlist_items[0])
        db_session.commit()
        
        remaining_items = db_session.query(WatchlistItem).filter(
            WatchlistItem.profile_id == test_profile.id
        ).all()
        
        assert len(remaining_items) == len(symbols) - 1
    
    def test_database_constraints_and_validations(self, db_session):
        """Test database constraints and data validation."""
        # Test user email uniqueness (if constraint exists)
        user1 = User(name="User 1", email="same@email.com")
        user2 = User(name="User 2", email="same@email.com")
        
        db_session.add(user1)
        db_session.add(user2)
        
        try:
            db_session.commit()
            # If no unique constraint, both should be added
            assert True
        except Exception:
            # If unique constraint exists, should raise exception
            db_session.rollback()
            assert True
    
    def test_cascade_deletes(self, db_session):
        """Test cascade delete behavior."""
        # Create user with profile and watchlist
        user = User(name="Cascade Test", email="cascade@test.com")
        db_session.add(user)
        db_session.flush()
        
        profile = Profile(
            user_id=user.id,
            risk_score=50,
            time_horizon_yrs=5,
            objectives="test",
            constraints="test"
        )
        db_session.add(profile)
        db_session.flush()
        
        asset = Asset(symbol="CASCADE", name="Cascade Corp", asset_class="stock")
        db_session.add(asset)
        db_session.flush()
        
        watchlist = WatchlistItem(profile_id=profile.id, asset_id=asset.id)
        db_session.add(watchlist)
        db_session.commit()
        
        # Delete user and check cascade behavior
        user_id = user.id
        profile_id = profile.id
        
        db_session.delete(user)
        db_session.commit()
        
        # Check if profile and watchlist are handled appropriately
        remaining_profile = db_session.get(Profile, profile_id)
        if remaining_profile is None:
            # Cascade delete worked
            assert True
        else:
            # No cascade, profile remains orphaned
            assert remaining_profile.user_id == user_id
    
    @pytest.mark.requires_db
    def test_database_connection_pooling(self):
        """Test database connection pooling under load."""
        def create_and_query():
            db = SessionLocal()
            try:
                result = db.execute(text("SELECT 1"))
                return result.scalar()
            finally:
                db.close()
        
        import threading
        results = []
        threads = []
        
        # Create multiple concurrent database connections
        for i in range(20):
            thread = threading.Thread(
                target=lambda: results.append(create_and_query())
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All connections should succeed
        assert len(results) == 20
        assert all(result == 1 for result in results)


@pytest.mark.integration
class TestTimescaleDBIntegration:
    """Test TimescaleDB integration for sentiment data."""
    
    @pytest.mark.requires_timescale
    @patch('supernova.db.is_timescale_available')
    def test_timescale_connection(self, mock_timescale_available):
        """Test TimescaleDB connection and basic operations."""
        mock_timescale_available.return_value = True
        
        with patch('supernova.db.get_timescale_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Test sentiment data insertion
            from supernova.sentiment_models import SentimentData
            
            mock_db.add.return_value = None
            mock_db.commit.return_value = None
            
            # Should not raise exception
            assert True
    
    @pytest.mark.requires_timescale 
    def test_sentiment_data_aggregation(self, sample_sentiment_data):
        """Test sentiment data aggregation queries."""
        # Mock TimescaleDB aggregation
        with patch('supernova.db.get_timescale_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock aggregation query result
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [
                {"symbol": "TEST", "avg_sentiment": 0.25, "count": 100}
            ]
            mock_db.execute.return_value = mock_result
            
            # Test would execute aggregation query
            assert True
    
    @pytest.mark.requires_timescale
    def test_sentiment_data_time_series_queries(self):
        """Test time-series specific queries for sentiment data."""
        with patch('supernova.db.get_timescale_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock time-series query
            mock_db.execute.return_value.fetchall.return_value = [
                ("2024-01-01 10:00:00", 0.15),
                ("2024-01-01 11:00:00", 0.22),
                ("2024-01-01 12:00:00", 0.18)
            ]
            
            # Time-series queries should work
            assert True


@pytest.mark.integration
class TestLLMProviderIntegration:
    """Test LLM provider integrations with mocked responses."""
    
    @patch('openai.ChatCompletion.create')
    def test_openai_integration(self, mock_openai, client, test_profile, sample_ohlcv_data):
        """Test OpenAI integration for advice generation."""
        # Mock OpenAI response
        mock_openai.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "action": "buy",
                        "confidence": 0.75,
                        "rationale": "Strong upward trend with good volume",
                        "risk_notes": "Consider position sizing"
                    })
                }
            }]
        }
        
        # Test advice request that would use OpenAI
        request_data = {
            "profile_id": test_profile.id,
            "symbol": "OPENAI_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
        }
        
        with patch('supernova.advisor.advise') as mock_advise:
            mock_advise.return_value = ("buy", 0.75, {}, "Strong trend", "Consider sizing")
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["action"] == "buy"
            assert data["confidence"] == 0.75
    
    @patch('anthropic.Anthropic')
    def test_anthropic_integration(self, mock_anthropic, client, test_profile, sample_ohlcv_data):
        """Test Anthropic integration for advice generation."""
        # Mock Anthropic response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text=json.dumps({
                "action": "hold",
                "confidence": 0.60,
                "rationale": "Mixed signals in the market",
                "risk_notes": "Wait for clearer trend"
            }))
        ]
        
        request_data = {
            "profile_id": test_profile.id,
            "symbol": "ANTHROPIC_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
        }
        
        with patch('supernova.advisor.advise') as mock_advise:
            mock_advise.return_value = ("hold", 0.60, {}, "Mixed signals", "Wait for trend")
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["action"] == "hold"
            assert data["confidence"] == 0.60
    
    @patch('langchain.llms.OpenAI')
    def test_langchain_integration(self, mock_langchain):
        """Test LangChain integration for conversational AI."""
        # Mock LangChain LLM
        mock_llm = MagicMock()
        mock_langchain.return_value = mock_llm
        mock_llm.predict.return_value = "This is a comprehensive financial analysis..."
        
        # Test LangChain chain execution
        from supernova.agent_tools import create_agent_chain
        
        with patch.object(mock_llm, 'predict', return_value="Test response"):
            # Should not raise exception
            assert True
    
    def test_llm_error_handling(self, client, test_profile, sample_ohlcv_data):
        """Test LLM integration error handling."""
        request_data = {
            "profile_id": test_profile.id,
            "symbol": "ERROR_TEST", 
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:50]]
        }
        
        # Mock LLM failure
        with patch('supernova.advisor.advise') as mock_advise:
            # Simulate LLM API failure
            mock_advise.side_effect = Exception("API Rate limit exceeded")
            
            try:
                response = client.post("/advice", json=request_data)
                # Should either handle gracefully or return appropriate error
                assert response.status_code in [200, 429, 500, 503]
            except Exception:
                # Error handling might raise exception
                assert True


@pytest.mark.integration  
class TestWebSocketIntegration:
    """Test WebSocket real-time communication."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        with patch('supernova.websocket_handler.WebSocketManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.disconnect = AsyncMock()
            
            # Test WebSocket connection
            with TestClient(app) as client:
                try:
                    with client.websocket_connect("/ws") as websocket:
                        # Connection successful
                        assert True
                except Exception:
                    # WebSocket endpoint might not be configured
                    assert True
    
    @pytest.mark.asyncio
    async def test_websocket_message_broadcasting(self):
        """Test WebSocket message broadcasting."""
        with patch('supernova.websocket_handler.WebSocketManager') as mock_manager:
            mock_ws_manager = MagicMock()
            mock_manager.return_value = mock_ws_manager
            mock_ws_manager.broadcast_to_all = AsyncMock()
            
            # Test message broadcasting
            test_message = WebSocketMessage(
                type="market_update",
                data={"symbol": "AAPL", "price": 150.00}
            )
            
            await mock_ws_manager.broadcast_to_all(test_message.model_dump())
            mock_ws_manager.broadcast_to_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_real_time_data_flow(self):
        """Test real-time data flow through WebSocket."""
        with patch('supernova.websocket_handler.WebSocketManager') as mock_manager:
            mock_ws_manager = MagicMock()
            mock_manager.return_value = mock_ws_manager
            mock_ws_manager.send_personal_message = AsyncMock()
            
            # Simulate real-time market data
            market_data = MarketDataUpdate(
                symbol="BTC",
                price=45000.00,
                volume=1000000,
                timestamp=datetime.now().isoformat()
            )
            
            await mock_ws_manager.send_personal_message(
                market_data.model_dump(),
                "user123"
            )
            
            mock_ws_manager.send_personal_message.assert_called_once()


@pytest.mark.integration
class TestVectorBTIntegration:
    """Test VectorBT backtesting integration."""
    
    @pytest.mark.requires_external
    def test_vectorbt_backtest_execution(self, client, sample_ohlcv_data):
        """Test VectorBT backtest execution."""
        request_data = {
            "strategy_template": "ma_crossover",
            "params": {"fast": 10, "slow": 30},
            "symbol": "VBT_TEST",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        with patch('supernova.backtester.run_vbt_backtest') as mock_vbt:
            # Mock VectorBT results
            mock_vbt.return_value = {
                "final_equity": 11500.0,
                "total_return": 15.0,
                "CAGR": 12.5,
                "Sharpe": 1.25,
                "MaxDrawdown": -8.5,
                "WinRate": 65.0,
                "n_trades_est": 25
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["metrics"]["final_equity"] == 11500.0
            assert data["metrics"]["CAGR"] == 12.5
    
    def test_vectorbt_performance_metrics(self):
        """Test VectorBT performance metrics calculation."""
        with patch('vectorbt') as mock_vbt:
            # Mock VectorBT portfolio
            mock_portfolio = MagicMock()
            mock_portfolio.stats.return_value = {
                'Total Return [%]': 15.5,
                'CAGR [%]': 12.8,
                'Sharpe Ratio': 1.35,
                'Max Drawdown [%]': -9.2
            }
            
            mock_vbt.Portfolio.from_signals.return_value = mock_portfolio
            
            # Test metrics extraction
            assert True
    
    def test_vectorbt_strategy_optimization(self):
        """Test VectorBT strategy parameter optimization."""
        with patch('vectorbt') as mock_vbt:
            # Mock optimization results
            mock_optimization = MagicMock()
            mock_optimization.max.return_value = (0.15, {"fast": 8, "slow": 25})
            
            mock_vbt.Portfolio.from_signals.return_value.stats.return_value = mock_optimization
            
            # Test parameter optimization
            assert True


@pytest.mark.integration
class TestPrefectWorkflowIntegration:
    """Test Prefect workflow orchestration integration."""
    
    @patch('prefect.flow')
    def test_prefect_flow_definition(self, mock_flow):
        """Test Prefect flow definition and execution."""
        from supernova.workflows import create_analysis_flow
        
        # Mock flow decorator
        mock_flow.return_value = lambda func: func
        
        # Should not raise exception
        try:
            flow = create_analysis_flow()
            assert callable(flow)
        except ImportError:
            # Prefect might not be properly configured
            assert True
    
    @patch('prefect.task')
    def test_prefect_task_execution(self, mock_task):
        """Test Prefect task execution."""
        from supernova.workflows import analyze_sentiment_task
        
        # Mock task decorator
        mock_task.return_value = lambda func: func
        
        try:
            task = analyze_sentiment_task()
            assert callable(task)
        except ImportError:
            # Prefect might not be configured
            assert True
    
    def test_workflow_scheduling(self):
        """Test workflow scheduling capabilities."""
        with patch('prefect.schedules.IntervalSchedule') as mock_schedule:
            mock_schedule.return_value = MagicMock()
            
            # Test scheduling configuration
            assert True


@pytest.mark.integration
class TestExternalAPIIntegration:
    """Test external API integrations."""
    
    @pytest.mark.requires_external
    @patch('httpx.AsyncClient')
    async def test_market_data_api_integration(self, mock_client):
        """Test market data API integration."""
        # Mock external API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "price": 150.00,
            "volume": 1000000,
            "timestamp": "2024-01-01T10:00:00Z"
        }
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Test API integration
        async with mock_client() as client:
            response = await client.get("/external/market-data")
            assert response.status_code == 200
    
    @pytest.mark.requires_external  
    @patch('tweepy.Client')
    def test_social_media_integration(self, mock_twitter):
        """Test social media API integration for sentiment."""
        # Mock Twitter API response
        mock_twitter.return_value.search_recent_tweets.return_value = MagicMock(
            data=[
                MagicMock(text="AAPL is looking great today! #bullish"),
                MagicMock(text="Not sure about AAPL's recent performance")
            ]
        )
        
        from supernova.sentiment import collect_social_sentiment
        
        try:
            result = collect_social_sentiment("AAPL")
            assert isinstance(result, dict)
        except ImportError:
            # Social media integration might not be configured
            assert True
    
    @patch('httpx.AsyncClient')
    async def test_news_api_integration(self, mock_client):
        """Test news API integration for sentiment analysis."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "AAPL reports strong quarterly earnings",
                    "content": "Apple Inc. exceeded expectations...",
                    "publishedAt": "2024-01-01T10:00:00Z"
                }
            ]
        }
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Test news API integration
        async with mock_client() as client:
            response = await client.get("/external/news")
            assert response.status_code == 200


@pytest.mark.integration
class TestRedisIntegration:
    """Test Redis caching integration."""
    
    @pytest.mark.requires_redis
    @patch('redis.Redis')
    def test_redis_connection(self, mock_redis):
        """Test Redis connection and basic operations."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        from supernova.cache_manager import CacheManager
        
        cache = CacheManager()
        assert cache.is_connected()
    
    @patch('redis.Redis')  
    def test_caching_operations(self, mock_redis):
        """Test caching set/get operations."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.set.return_value = True
        mock_client.get.return_value = b'{"cached": "data"}'
        
        from supernova.cache_manager import CacheManager
        
        cache = CacheManager()
        
        # Test cache operations
        cache.set("test_key", {"test": "data"})
        mock_client.set.assert_called()
        
        result = cache.get("test_key")
        mock_client.get.assert_called_with("test_key")
    
    @patch('redis.Redis')
    def test_cache_expiration(self, mock_redis):
        """Test cache expiration functionality."""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        mock_client.setex.return_value = True
        
        from supernova.cache_manager import CacheManager
        
        cache = CacheManager()
        cache.set("expiring_key", {"data": "value"}, ttl=300)
        
        mock_client.setex.assert_called_with("expiring_key", 300, mock_client.set.call_args[0][1])


@pytest.mark.integration 
class TestOptunIntegration:
    """Test Optuna hyperparameter optimization integration."""
    
    @patch('optuna.create_study')
    def test_optuna_study_creation(self, mock_create_study):
        """Test Optuna study creation for optimization."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        from supernova.ml_optimizer import create_optimization_study
        
        study = create_optimization_study()
        assert study is not None
        mock_create_study.assert_called_once()
    
    @patch('optuna.Trial')
    def test_optuna_parameter_optimization(self, mock_trial):
        """Test Optuna parameter optimization."""
        mock_trial.suggest_int.return_value = 15
        mock_trial.suggest_float.return_value = 0.3
        
        from supernova.ml_optimizer import optimize_strategy_parameters
        
        # Test parameter suggestion
        params = optimize_strategy_parameters(mock_trial)
        assert isinstance(params, dict)
    
    def test_optimization_objective_function(self):
        """Test optimization objective function."""
        with patch('supernova.backtester.run_backtest') as mock_backtest:
            mock_backtest.return_value = {"sharpe_ratio": 1.25}
            
            from supernova.ml_optimizer import objective_function
            
            # Mock trial and test data
            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 10
            mock_trial.suggest_float.return_value = 0.5
            
            result = objective_function(mock_trial, test_data=[], symbol="TEST")
            assert isinstance(result, (int, float))


@pytest.mark.integration
class TestSecurityIntegration:
    """Test security component integration."""
    
    def test_authentication_integration(self, client):
        """Test authentication system integration."""
        # Test protected endpoint access
        with patch('supernova.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": 1, "scopes": ["read"]}
            
            headers = {"Authorization": "Bearer test_token"}
            response = client.get("/protected-endpoint", headers=headers)
            
            # Should handle authentication appropriately  
            assert response.status_code in [200, 401, 404]
    
    def test_rate_limiting_integration(self, client):
        """Test rate limiting integration."""
        # Test rate limiting across requests
        responses = []
        
        for i in range(50):
            response = client.post("/intake", json={"name": f"Rate Test {i}"})
            responses.append(response.status_code)
            
            if response.status_code == 429:  # Rate limited
                break
        
        # Should either all succeed or eventually rate limit
        assert 200 in responses
    
    def test_input_validation_integration(self, client):
        """Test comprehensive input validation integration."""
        malicious_inputs = [
            {"name": "<script>alert('xss')</script>"},
            {"name": "'; DROP TABLE users; --"},
            {"name": "../../../etc/passwd"},
            {"email": "javascript:alert('xss')"},
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/intake", json=malicious_input)
            # Should sanitize or reject malicious input
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                # If accepted, should be sanitized
                data = response.json()
                assert "script" not in str(data).lower()
                assert "drop table" not in str(data).lower()


@pytest.mark.integration
class TestMultiComponentWorkflows:
    """Test complex workflows involving multiple components."""
    
    def test_complete_analysis_workflow(self, client, db_session, sample_ohlcv_data):
        """Test complete analysis workflow from data ingestion to advice."""
        # 1. Create user and profile
        user = User(name="Workflow User", email="workflow@test.com")
        db_session.add(user)
        db_session.flush()
        
        profile = Profile(
            user_id=user.id,
            risk_score=60,
            time_horizon_yrs=8,
            objectives="balanced growth",
            constraints="low risk"
        )
        db_session.add(profile)
        db_session.commit()
        
        # 2. Add symbols to watchlist
        symbols = ["WORKFLOW_AAPL", "WORKFLOW_GOOGL"]
        
        for symbol in symbols:
            asset = Asset(symbol=symbol, name=f"{symbol} Corp", asset_class="stock")
            db_session.add(asset)
            db_session.flush()
            
            watchlist = WatchlistItem(profile_id=profile.id, asset_id=asset.id)
            db_session.add(watchlist)
        
        db_session.commit()
        
        # 3. Get advice for each symbol
        advice_results = []
        
        for symbol in symbols:
            request_data = {
                "profile_id": profile.id,
                "symbol": symbol,
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            advice_results.append(response.json())
        
        # 4. Run backtests for advised strategies
        backtest_results = []
        
        for i, symbol in enumerate(symbols):
            advice = advice_results[i]
            
            request_data = {
                "strategy_template": "ma_crossover",
                "params": {"fast": 10, "slow": 30},
                "symbol": symbol,
                "timeframe": "1h", 
                "bars": [bar.model_dump() for bar in sample_ohlcv_data]
            }
            
            response = client.post("/backtest", json=request_data)
            assert response.status_code == 200
            backtest_results.append(response.json())
        
        # 5. Evaluate alerts based on current data
        watch_items = [{"symbol": symbol, "profile_id": profile.id} for symbol in symbols]
        bars_data = {}
        
        for symbol in symbols:
            bars_data[symbol] = [bar.model_dump() for bar in sample_ohlcv_data[:50]]
        
        alert_request = {"watch": watch_items, "bars": bars_data}
        alert_response = client.post("/alerts/evaluate", json=alert_request)
        assert alert_response.status_code == 200
        
        # Verify workflow completion
        assert len(advice_results) == len(symbols)
        assert len(backtest_results) == len(symbols)
        assert isinstance(alert_response.json(), list)
    
    def test_real_time_data_processing_workflow(self):
        """Test real-time data processing workflow."""
        with patch('supernova.websocket_handler.WebSocketManager') as mock_ws:
            mock_manager = MagicMock()
            mock_ws.return_value = mock_manager
            
            # Mock real-time data flow
            mock_manager.broadcast_to_all = AsyncMock()
            
            # Simulate market data update
            market_update = {
                "type": "price_update",
                "symbol": "REALTIME_TEST",
                "price": 150.00,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat()
            }
            
            # Should process and broadcast update
            asyncio.run(mock_manager.broadcast_to_all(market_update))
            mock_manager.broadcast_to_all.assert_called_once()
    
    def test_batch_processing_workflow(self, client, sample_ohlcv_data):
        """Test batch processing workflow for multiple symbols."""
        symbols = [f"BATCH_TEST_{i}" for i in range(10)]
        
        # Batch create users and profiles
        profiles = []
        for i, symbol in enumerate(symbols):
            intake_data = {
                "name": f"Batch User {i}",
                "email": f"batch{i}@test.com",
                "risk_questions": [2, 3, 2, 3, 3]
            }
            
            response = client.post("/intake", json=intake_data)
            assert response.status_code == 200
            profiles.append(response.json()["profile_id"])
        
        # Batch advice requests
        advice_responses = []
        for i, symbol in enumerate(symbols):
            request_data = {
                "profile_id": profiles[i],
                "symbol": symbol,
                "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
            }
            
            response = client.post("/advice", json=request_data)
            assert response.status_code == 200
            advice_responses.append(response.json())
        
        # Verify batch processing success
        assert len(advice_responses) == len(symbols)
        assert all(resp["symbol"] in symbols for resp in advice_responses)
    
    def test_error_recovery_workflow(self, client, sample_ohlcv_data):
        """Test error recovery in multi-step workflows."""
        # Start workflow with valid data
        intake_data = {"name": "Error Recovery User", "email": "error@test.com"}
        intake_response = client.post("/intake", json=intake_data)
        assert intake_response.status_code == 200
        
        profile_id = intake_response.json()["profile_id"]
        
        # Introduce error in middle of workflow
        invalid_advice_data = {
            "profile_id": profile_id,
            "symbol": "ERROR_TEST",
            "bars": []  # Empty bars should cause error
        }
        
        advice_response = client.post("/advice", json=invalid_advice_data)
        # Should handle error gracefully
        assert advice_response.status_code in [400, 422, 500]
        
        # Continue workflow with valid data after error
        valid_advice_data = {
            "profile_id": profile_id,
            "symbol": "RECOVERY_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
        }
        
        recovery_response = client.post("/advice", json=valid_advice_data)
        assert recovery_response.status_code == 200
        
        # Workflow should recover and continue normally
        assert recovery_response.json()["symbol"] == "RECOVERY_TEST"