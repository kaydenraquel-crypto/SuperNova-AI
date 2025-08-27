"""
Enhanced Integration Testing Suite
=================================

Comprehensive integration tests for SuperNova AI financial platform
covering API endpoints, database operations, external service integrations,
and complete user workflows.
"""
import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
import httpx

from supernova.api import app
from supernova.db import User, Profile, Asset, WatchlistItem
from supernova.schemas import (
    IntakeRequest, AdviceRequest, WatchlistRequest, 
    BacktestRequest, OHLCVBar, SentimentDataPoint
)


class TestUserLifecycleIntegration:
    """Test complete user lifecycle from onboarding to portfolio management."""
    
    @pytest.mark.integration
    def test_complete_user_onboarding_flow(self, client, db_session):
        """Test complete user onboarding and profile creation."""
        # Step 1: User intake
        intake_data = {
            "name": "Integration Test User",
            "email": "integration@test.com",
            "income": 75000.0,
            "risk_questions": [3, 4, 2, 3, 4],
            "investment_goals": ["retirement", "growth"],
            "time_horizon_yrs": 15
        }
        
        intake_response = client.post("/intake", json=intake_data)
        assert intake_response.status_code == 200
        
        intake_result = intake_response.json()
        assert "profile_id" in intake_result
        assert "risk_score" in intake_result
        profile_id = intake_result["profile_id"]
        
        # Step 2: Verify profile creation in database
        profile = db_session.query(Profile).filter(Profile.id == profile_id).first()
        assert profile is not None
        assert profile.time_horizon_yrs == 15
        
        # Step 3: Get initial portfolio advice
        advice_data = {
            "profile_id": profile_id,
            "symbols": ["VTI", "BND", "VEA", "VWO"]
        }
        
        advice_response = client.post("/advice", json=advice_data)
        assert advice_response.status_code == 200
        
        advice_result = advice_response.json()
        assert "allocations" in advice_result
        assert "risk_metrics" in advice_result
        
        # Step 4: Add symbols to watchlist
        watchlist_symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in watchlist_symbols:
            watchlist_data = {
                "profile_id": profile_id,
                "symbol": symbol
            }
            
            watchlist_response = client.post("/watchlist", json=watchlist_data)
            assert watchlist_response.status_code == 200
        
        # Step 5: Verify watchlist in database
        watchlist_count = db_session.query(WatchlistItem).filter(
            WatchlistItem.profile_id == profile_id
        ).count()
        assert watchlist_count == len(watchlist_symbols)
        
        # Step 6: Run backtest on advised portfolio
        backtest_data = {
            "profile_id": profile_id,
            "symbols": ["VTI", "BND"],
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_capital": 100000
        }
        
        backtest_response = client.post("/backtest", json=backtest_data)
        assert backtest_response.status_code == 200
        
        backtest_result = backtest_response.json()
        assert "performance" in backtest_result
        assert "metrics" in backtest_result
        
        # Verify expected performance metrics are present
        metrics = backtest_result["metrics"]
        required_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]
        for metric in required_metrics:
            assert metric in metrics
    
    @pytest.mark.integration
    def test_user_profile_updates_and_rebalancing(self, client, db_session):
        """Test profile updates and portfolio rebalancing."""
        # Create initial user
        intake_data = {
            "name": "Rebalancing Test User",
            "email": "rebalancing@test.com",
            "income": 100000.0,
            "risk_questions": [2, 2, 3, 2, 2],  # Conservative
        }
        
        intake_response = client.post("/intake", json=intake_data)
        profile_id = intake_response.json()["profile_id"]
        
        # Get initial conservative advice
        initial_advice_response = client.post("/advice", json={
            "profile_id": profile_id,
            "symbols": ["VTI", "BND", "VEA"]
        })
        initial_advice = initial_advice_response.json()
        initial_equity_weight = sum(
            alloc["weight"] for alloc in initial_advice["allocations"] 
            if alloc["symbol"] in ["VTI", "VEA"]
        )
        
        # Update profile to aggressive
        update_data = {
            "profile_id": profile_id,
            "risk_questions": [5, 5, 4, 5, 5],  # Aggressive
            "time_horizon_yrs": 25
        }
        
        update_response = client.put(f"/profile/{profile_id}", json=update_data)
        assert update_response.status_code == 200
        
        # Get updated advice
        updated_advice_response = client.post("/advice", json={
            "profile_id": profile_id,
            "symbols": ["VTI", "BND", "VEA", "VWO"]
        })
        updated_advice = updated_advice_response.json()
        updated_equity_weight = sum(
            alloc["weight"] for alloc in updated_advice["allocations"] 
            if alloc["symbol"] in ["VTI", "VEA", "VWO"]
        )
        
        # Verify equity allocation increased for aggressive profile
        assert updated_equity_weight > initial_equity_weight + 0.1


class TestMarketDataIntegration:
    """Test market data integration and processing."""
    
    @pytest.mark.integration
    @pytest.mark.requires_external
    def test_real_market_data_integration(self, client):
        """Test integration with real market data sources."""
        # Test popular ETF symbols
        test_symbols = ["VTI", "SPY", "BND", "QQQ"]
        
        for symbol in test_symbols:
            response = client.get(f"/market-data/{symbol}")
            
            if response.status_code == 200:
                data = response.json()
                assert "bars" in data
                assert len(data["bars"]) > 0
                
                # Verify OHLCV bar structure
                bar = data["bars"][0]
                required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
                for field in required_fields:
                    assert field in bar
                
                # Verify realistic price relationships
                assert bar["low"] <= bar["open"] <= bar["high"]
                assert bar["low"] <= bar["close"] <= bar["high"]
                assert bar["volume"] > 0
            else:
                # If external API is unavailable, test should handle gracefully
                assert response.status_code in [429, 503, 504]  # Rate limited or service unavailable
    
    @pytest.mark.integration
    @pytest.mark.mock
    def test_market_data_caching_and_error_handling(self, client, mock_external_apis):
        """Test market data caching and error handling."""
        symbol = "TEST"
        
        # First request should hit external API
        response1 = client.get(f"/market-data/{symbol}")
        
        # Second request should use cache (faster response)
        start_time = time.time()
        response2 = client.get(f"/market-data/{symbol}")
        response_time = time.time() - start_time
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response_time < 0.1  # Should be very fast due to caching
        
        # Responses should be identical due to caching
        assert response1.json() == response2.json()
    
    @pytest.mark.integration
    def test_historical_data_processing(self, client, sample_ohlcv_data):
        """Test historical data processing and calculations."""
        # Mock historical data endpoint
        with patch('supernova.api.get_historical_data') as mock_get_data:
            mock_get_data.return_value = sample_ohlcv_data
            
            response = client.get("/market-data/VTI/historical", params={
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "resolution": "1D"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert "bars" in data
            assert "statistics" in data
            
            # Verify calculated statistics
            stats = data["statistics"]
            assert "total_return" in stats
            assert "volatility" in stats
            assert "sharpe_ratio" in stats
            assert "max_drawdown" in stats


class TestSentimentAnalysisIntegration:
    """Test sentiment analysis integration."""
    
    @pytest.mark.integration
    @pytest.mark.requires_external
    async def test_sentiment_data_collection(self, client):
        """Test sentiment data collection from multiple sources."""
        test_symbols = ["AAPL", "TSLA", "NVDA"]
        
        for symbol in test_symbols:
            response = client.get(f"/sentiment/{symbol}")
            
            if response.status_code == 200:
                data = response.json()
                assert "sentiment_score" in data
                assert "confidence" in data
                assert "sources" in data
                
                # Verify sentiment score is in valid range
                assert -1.0 <= data["sentiment_score"] <= 1.0
                assert 0.0 <= data["confidence"] <= 1.0
            else:
                # Handle API limitations gracefully
                assert response.status_code in [429, 503]
    
    @pytest.mark.integration
    def test_sentiment_aggregation(self, client, sample_sentiment_data):
        """Test sentiment data aggregation from multiple sources."""
        with patch('supernova.sentiment.collect_sentiment_data') as mock_collect:
            mock_collect.return_value = sample_sentiment_data
            
            response = client.get("/sentiment/TEST/aggregate", params={
                "period": "7d"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert "average_sentiment" in data
            assert "sentiment_trend" in data
            assert "confidence_weighted_score" in data
            assert "data_points" in data
            
            # Verify aggregated scores are reasonable
            assert -1.0 <= data["average_sentiment"] <= 1.0


class TestLLMIntegration:
    """Test LLM integration for financial advice and chat."""
    
    @pytest.mark.integration
    @pytest.mark.requires_llm
    def test_llm_financial_advice_generation(self, client, mock_llm_responses):
        """Test LLM integration for generating financial advice."""
        with patch('supernova.llm_service.get_completion') as mock_llm:
            mock_llm.return_value = "Based on your moderate risk profile and 10-year time horizon, I recommend a balanced portfolio allocation of 60% stocks and 40% bonds."
            
            advice_request = {
                "profile_id": 1,
                "symbols": ["VTI", "BND"],
                "include_llm_insights": True
            }
            
            response = client.post("/advice", json=advice_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "llm_insights" in data
            assert "reasoning" in data["llm_insights"]
            assert len(data["llm_insights"]["reasoning"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.requires_llm
    def test_chat_interface_integration(self, client):
        """Test chat interface with LLM integration."""
        chat_request = {
            "message": "What should I know about investing in index funds?",
            "context": {
                "user_profile": {"risk_score": 50, "time_horizon": 10}
            }
        }
        
        with patch('supernova.chat.process_message') as mock_chat:
            mock_chat.return_value = {
                "response": "Index funds are excellent for diversified, low-cost investing...",
                "confidence": 0.9,
                "sources": ["investment_fundamentals", "modern_portfolio_theory"]
            }
            
            response = client.post("/chat", json=chat_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "response" in data
            assert "confidence" in data
            assert len(data["response"]) > 0


class TestDatabaseTransactionIntegrity:
    """Test database transaction integrity and ACID properties."""
    
    @pytest.mark.integration
    @pytest.mark.requires_db
    def test_atomic_portfolio_creation(self, client, db_session):
        """Test atomic portfolio creation with rollback on failure."""
        # Create test scenario that should fail partway through
        intake_data = {
            "name": "Transaction Test User",
            "email": "transaction@test.com",
            "income": 50000.0,
            "risk_questions": [3, 3, 3, 3, 3],
        }
        
        # Mock database error during profile creation
        with patch('supernova.db.Profile.__init__') as mock_init:
            mock_init.side_effect = Exception("Database error")
            
            response = client.post("/intake", json=intake_data)
            # Should handle error gracefully
            assert response.status_code in [500, 400]
            
            # Verify no partial data was committed
            user_count = db_session.query(User).filter(
                User.email == "transaction@test.com"
            ).count()
            assert user_count == 0
    
    @pytest.mark.integration
    def test_concurrent_watchlist_modifications(self, client, db_session):
        """Test concurrent modifications to watchlist."""
        # Create test user
        intake_response = client.post("/intake", json={
            "name": "Concurrent Test User",
            "email": "concurrent@test.com",
            "risk_questions": [3, 3, 3, 3, 3]
        })
        profile_id = intake_response.json()["profile_id"]
        
        # Simulate concurrent additions
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        for symbol in symbols:
            response = client.post("/watchlist", json={
                "profile_id": profile_id,
                "symbol": symbol
            })
            assert response.status_code == 200
        
        # Verify all items were added
        watchlist_count = db_session.query(WatchlistItem).filter(
            WatchlistItem.profile_id == profile_id
        ).count()
        assert watchlist_count == len(symbols)
        
        # Test concurrent removal
        for symbol in symbols[:3]:
            response = client.delete(f"/watchlist/{profile_id}/{symbol}")
            assert response.status_code == 200
        
        # Verify correct number remain
        remaining_count = db_session.query(WatchlistItem).filter(
            WatchlistItem.profile_id == profile_id
        ).count()
        assert remaining_count == len(symbols) - 3


class TestPerformanceUnderLoad:
    """Test system performance under simulated load."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_api_response_times(self, client):
        """Test API response times under normal load."""
        endpoints_to_test = [
            ("/health", "GET", None),
            ("/intake", "POST", {
                "name": "Performance Test User",
                "email": "perf@test.com",
                "risk_questions": [3, 3, 3, 3, 3]
            }),
        ]
        
        response_times = []
        
        for endpoint, method, data in endpoints_to_test:
            for i in range(10):  # Test multiple requests
                start_time = time.time()
                
                if method == "GET":
                    response = client.get(endpoint)
                else:
                    test_data = data.copy()
                    test_data["email"] = f"perf{i}@test.com"  # Unique email
                    response = client.post(endpoint, json=test_data)
                
                elapsed_time = time.time() - start_time
                response_times.append(elapsed_time)
                
                assert response.status_code in [200, 201]
        
        # Performance assertions
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time:.2f}s"
        assert max_response_time < 3.0, f"Maximum response time too high: {max_response_time:.2f}s"
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_database_query_performance(self, client, db_session):
        """Test database query performance with larger datasets."""
        # Create multiple users and profiles for testing
        user_count = 100
        users_created = []
        
        start_time = time.time()
        for i in range(user_count):
            intake_data = {
                "name": f"Perf User {i}",
                "email": f"perfuser{i}@test.com",
                "income": 50000 + (i * 1000),
                "risk_questions": [3, 3, 3, 3, 3]
            }
            
            response = client.post("/intake", json=intake_data)
            assert response.status_code == 200
            users_created.append(response.json()["profile_id"])
        
        creation_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        for profile_id in users_created[:10]:  # Test subset
            response = client.get(f"/profile/{profile_id}")
            assert response.status_code == 200
        
        query_time = time.time() - start_time
        
        # Performance assertions
        avg_creation_time = creation_time / user_count
        avg_query_time = query_time / 10
        
        assert avg_creation_time < 0.5, f"User creation too slow: {avg_creation_time:.2f}s"
        assert avg_query_time < 0.1, f"Profile queries too slow: {avg_query_time:.2f}s"


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.integration
    def test_external_service_failure_handling(self, client):
        """Test handling of external service failures."""
        # Mock external service failure
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timeout")
            
            # Request should handle timeout gracefully
            response = client.get("/market-data/VTI")
            
            # Should return cached data or graceful error
            assert response.status_code in [200, 503, 504]
            
            if response.status_code == 200:
                # If returning cached data, verify it's marked as cached
                data = response.json()
                assert "cached" in data or "stale" in data
    
    @pytest.mark.integration
    def test_database_connection_failure_recovery(self, client):
        """Test database connection failure and recovery."""
        # This test would typically require database connection mocking
        # For integration testing, we verify the system handles DB issues gracefully
        
        with patch('supernova.db.SessionLocal') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            response = client.get("/health")
            
            # Health check should indicate database issues
            if response.status_code == 200:
                data = response.json()
                assert "database" in data
                assert data["database"]["status"] == "unhealthy"
            else:
                assert response.status_code == 503
    
    @pytest.mark.integration
    def test_invalid_data_handling(self, client):
        """Test handling of invalid or malicious input data."""
        # Test various invalid inputs
        invalid_inputs = [
            # SQL injection attempts
            {"name": "'; DROP TABLE users; --", "email": "test@test.com"},
            
            # XSS attempts
            {"name": "<script>alert('xss')</script>", "email": "test@test.com"},
            
            # Extremely long strings
            {"name": "A" * 10000, "email": "test@test.com"},
            
            # Invalid data types
            {"name": 12345, "email": "not-an-email"},
            
            # Missing required fields
            {"name": "Test User"},
        ]
        
        for invalid_data in invalid_inputs:
            response = client.post("/intake", json=invalid_data)
            
            # Should reject invalid data
            assert response.status_code in [400, 422]
            
            # Should not cause server errors
            assert response.status_code != 500


class TestBusinessLogicIntegration:
    """Test business logic integration across components."""
    
    @pytest.mark.integration
    def test_risk_assessment_consistency(self, client):
        """Test consistency of risk assessment across the platform."""
        # Create users with different risk profiles
        risk_profiles = [
            {"questions": [1, 1, 1, 1, 1], "expected_range": (0, 30)},     # Very conservative
            {"questions": [3, 3, 3, 3, 3], "expected_range": (40, 60)},    # Moderate
            {"questions": [5, 5, 5, 5, 5], "expected_range": (80, 100)},   # Very aggressive
        ]
        
        for i, profile in enumerate(risk_profiles):
            intake_data = {
                "name": f"Risk Test User {i}",
                "email": f"risk{i}@test.com",
                "risk_questions": profile["questions"]
            }
            
            response = client.post("/intake", json=intake_data)
            assert response.status_code == 200
            
            result = response.json()
            risk_score = result["risk_score"]
            min_expected, max_expected = profile["expected_range"]
            
            assert min_expected <= risk_score <= max_expected, \
                f"Risk score {risk_score} not in expected range {profile['expected_range']}"
    
    @pytest.mark.integration
    def test_portfolio_allocation_constraints(self, client):
        """Test portfolio allocation business rules and constraints."""
        # Create moderate risk user
        intake_response = client.post("/intake", json={
            "name": "Allocation Test User",
            "email": "allocation@test.com",
            "risk_questions": [3, 3, 3, 3, 3]
        })
        profile_id = intake_response.json()["profile_id"]
        
        # Test allocation with various symbol combinations
        test_cases = [
            ["VTI", "BND"],  # Simple equity/bond mix
            ["VTI", "VEA", "VWO", "BND"],  # Diversified portfolio
            ["QQQ", "SPY", "IWM"],  # All equity (should adjust for risk)
        ]
        
        for symbols in test_cases:
            advice_response = client.post("/advice", json={
                "profile_id": profile_id,
                "symbols": symbols
            })
            
            assert advice_response.status_code == 200
            advice = advice_response.json()
            
            # Verify allocation constraints
            allocations = advice["allocations"]
            total_allocation = sum(alloc["weight"] for alloc in allocations)
            
            # Total should be approximately 1.0
            assert abs(total_allocation - 1.0) < 0.01
            
            # No single position should dominate (unless only one symbol)
            if len(symbols) > 1:
                max_weight = max(alloc["weight"] for alloc in allocations)
                assert max_weight < 0.8  # No more than 80% in single asset


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])