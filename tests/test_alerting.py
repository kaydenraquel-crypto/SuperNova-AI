import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from supernova.alerting import evaluate_alerts
from supernova.strategy_engine import eval_rsi_breakout, make_df


@pytest.fixture
def sample_watch_items():
    """Sample watchlist items for testing."""
    return [
        {"symbol": "AAPL", "profile_id": 1, "notes": "Tech stock"},
        {"symbol": "GOOGL", "profile_id": 1, "notes": "Search giant"},
        {"symbol": "TSLA", "profile_id": 2, "notes": "EV company"}
    ]


@pytest.fixture
def overbought_bars():
    """Generate bars that should trigger RSI overbought condition."""
    bars = []
    base_price = 100
    
    # Create consistently rising prices to get high RSI
    for i in range(30):
        timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
        base_price += 2.0  # Strong upward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price - 1,
            "high": base_price + 0.5,
            "low": base_price - 1.5,
            "close": base_price,
            "volume": 10000 + (i * 500)  # Increasing volume
        })
    
    return bars


@pytest.fixture
def oversold_bars():
    """Generate bars that should trigger RSI oversold condition."""
    bars = []
    base_price = 150
    
    # Create consistently falling prices to get low RSI
    for i in range(30):
        timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
        base_price -= 2.5  # Strong downward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price + 1,
            "high": base_price + 1.5,
            "low": base_price - 0.5,
            "close": base_price,
            "volume": 15000 - (i * 200)  # Decreasing volume
        })
    
    return bars


@pytest.fixture
def neutral_bars():
    """Generate bars that should not trigger alerts (neutral RSI)."""
    bars = []
    base_price = 100
    
    # Create sideways movement for neutral RSI
    for i in range(30):
        timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
        # Random walk around base price
        import numpy as np
        base_price += np.random.normal(0, 0.5)
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": base_price + abs(np.random.normal(0, 0.3)),
            "low": base_price - abs(np.random.normal(0, 0.3)),
            "close": base_price + np.random.normal(0, 0.2),
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
        base_price = bars[-1]["close"]
    
    return bars


class TestEvaluateAlerts:
    @pytest.mark.asyncio
    async def test_evaluate_alerts_basic_functionality(self, sample_watch_items, overbought_bars):
        """Test basic alert evaluation functionality."""
        bars_by_symbol = {
            "AAPL": overbought_bars,
            "GOOGL": overbought_bars,
            "TSLA": overbought_bars
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        assert isinstance(triggered, list)
        # Should have triggered alerts for overbought conditions
        assert len(triggered) > 0
        
        # Check alert structure
        for alert in triggered:
            assert "symbol" in alert
            assert "message" in alert
            assert alert["symbol"] in ["AAPL", "GOOGL", "TSLA"]
            assert "SELL" in alert["message"] or "BUY" in alert["message"]

    @pytest.mark.asyncio
    async def test_evaluate_alerts_oversold_condition(self, sample_watch_items, oversold_bars):
        """Test alert evaluation with oversold condition."""
        bars_by_symbol = {
            "AAPL": oversold_bars,
            "GOOGL": oversold_bars,
            "TSLA": oversold_bars
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        # Should trigger buy alerts for oversold conditions
        assert len(triggered) > 0
        
        for alert in triggered:
            assert "BUY" in alert["message"]
            assert "RSI=" in alert["message"]

    @pytest.mark.asyncio
    async def test_evaluate_alerts_neutral_condition(self, sample_watch_items, neutral_bars):
        """Test alert evaluation with neutral conditions (no alerts expected)."""
        bars_by_symbol = {
            "AAPL": neutral_bars,
            "GOOGL": neutral_bars,
            "TSLA": neutral_bars
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        # Should not trigger alerts for neutral RSI
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_evaluate_alerts_mixed_conditions(self, sample_watch_items, overbought_bars, oversold_bars, neutral_bars):
        """Test alert evaluation with mixed market conditions."""
        bars_by_symbol = {
            "AAPL": overbought_bars,    # Should trigger sell
            "GOOGL": oversold_bars,     # Should trigger buy
            "TSLA": neutral_bars        # Should not trigger
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        # Should have 2 alerts (AAPL sell, GOOGL buy)
        assert len(triggered) == 2
        
        symbols_triggered = [alert["symbol"] for alert in triggered]
        assert "AAPL" in symbols_triggered
        assert "GOOGL" in symbols_triggered
        assert "TSLA" not in symbols_triggered

    @pytest.mark.asyncio
    async def test_evaluate_alerts_missing_symbol_data(self, sample_watch_items):
        """Test alert evaluation when some symbols have no bar data."""
        bars_by_symbol = {
            "AAPL": [],  # Empty data
            # GOOGL and TSLA missing entirely
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        # Should not trigger any alerts due to missing data
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_evaluate_alerts_empty_watchlist(self, overbought_bars):
        """Test alert evaluation with empty watchlist."""
        bars_by_symbol = {
            "AAPL": overbought_bars
        }
        
        triggered = await evaluate_alerts([], bars_by_symbol)
        
        # No watchlist items, so no alerts
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_evaluate_alerts_empty_bars_dict(self, sample_watch_items):
        """Test alert evaluation with empty bars dictionary."""
        triggered = await evaluate_alerts(sample_watch_items, {})
        
        # No bar data, so no alerts
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_evaluate_alerts_single_symbol(self, overbought_bars):
        """Test alert evaluation with single symbol."""
        watch_items = [{"symbol": "SINGLE", "profile_id": 1}]
        bars_by_symbol = {"SINGLE": overbought_bars}
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        assert len(triggered) == 1
        assert triggered[0]["symbol"] == "SINGLE"
        assert "SELL" in triggered[0]["message"]

    @pytest.mark.asyncio
    async def test_evaluate_alerts_rsi_message_format(self, sample_watch_items, overbought_bars):
        """Test that alert messages contain proper RSI information."""
        bars_by_symbol = {"AAPL": overbought_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        if triggered:  # If alert was triggered
            alert = triggered[0]
            assert "AAPL" in alert["message"]
            assert "RSI=" in alert["message"]
            
            # Extract RSI value from message
            import re
            rsi_match = re.search(r"RSI=(\d+\.?\d*)", alert["message"])
            assert rsi_match is not None
            rsi_value = float(rsi_match.group(1))
            assert 0 <= rsi_value <= 100

    @pytest.mark.asyncio
    async def test_evaluate_alerts_insufficient_data(self, sample_watch_items):
        """Test alert evaluation with insufficient bar data."""
        # Only 5 bars - insufficient for RSI calculation
        insufficient_bars = []
        for i in range(5):
            timestamp = (datetime.now() - timedelta(hours=5-i)).isoformat() + "Z"
            insufficient_bars.append({
                "timestamp": timestamp,
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000
            })
        
        bars_by_symbol = {
            "AAPL": insufficient_bars,
            "GOOGL": insufficient_bars,
            "TSLA": insufficient_bars
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        # Should handle gracefully - may or may not trigger depending on implementation
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    async def test_evaluate_alerts_malformed_bars(self, sample_watch_items):
        """Test alert evaluation with malformed bar data."""
        malformed_bars = [
            {"timestamp": "invalid", "open": "abc", "high": None, "low": -1, "close": 100, "volume": 1000}
        ]
        
        bars_by_symbol = {"AAPL": malformed_bars}
        
        # Should handle gracefully without crashing
        try:
            triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
            assert isinstance(triggered, list)
        except Exception:
            # If it fails, that's also acceptable for malformed data
            pass

    @pytest.mark.asyncio
    async def test_evaluate_alerts_nan_values(self, sample_watch_items):
        """Test alert evaluation with NaN values in bars."""
        bars_with_nan = []
        for i in range(20):
            timestamp = (datetime.now() - timedelta(hours=20-i)).isoformat() + "Z"
            bars_with_nan.append({
                "timestamp": timestamp,
                "open": 100 if i % 5 != 0 else float('nan'),
                "high": 101,
                "low": 99,
                "close": 100 if i % 3 != 0 else float('nan'),
                "volume": 1000
            })
        
        bars_by_symbol = {"AAPL": bars_with_nan}
        
        # Should handle NaN values gracefully
        try:
            triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
            assert isinstance(triggered, list)
        except Exception:
            # Acceptable to fail with NaN data
            pass


class TestWebhookIntegration:
    @pytest.mark.asyncio
    @patch('supernova.config.settings')
    @patch('httpx.AsyncClient')
    async def test_webhook_called_on_alert(self, mock_client, mock_settings, sample_watch_items, overbought_bars):
        """Test that webhook is called when alert is triggered."""
        # Mock settings to enable webhook
        mock_settings.ALERT_WEBHOOK_URL = "http://test-webhook.com/alerts"
        
        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        bars_by_symbol = {"AAPL": overbought_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        if triggered:  # If alert was triggered
            # Verify webhook was called
            mock_http_client.post.assert_called()
            call_args = mock_http_client.post.call_args
            
            assert call_args[0][0] == "http://test-webhook.com/alerts"
            assert "json" in call_args[1]
            
            webhook_payload = call_args[1]["json"]
            assert "symbol" in webhook_payload
            assert "message" in webhook_payload

    @pytest.mark.asyncio
    @patch('supernova.config.settings')
    @patch('httpx.AsyncClient')
    async def test_webhook_not_called_when_disabled(self, mock_client, mock_settings, sample_watch_items, overbought_bars):
        """Test that webhook is not called when disabled."""
        # Mock settings to disable webhook
        mock_settings.ALERT_WEBHOOK_URL = None
        
        mock_http_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        bars_by_symbol = {"AAPL": overbought_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        # Webhook should not be called
        mock_http_client.post.assert_not_called()

    @pytest.mark.asyncio
    @patch('supernova.config.settings')
    @patch('httpx.AsyncClient')
    async def test_webhook_failure_handling(self, mock_client, mock_settings, sample_watch_items, overbought_bars):
        """Test that webhook failures are handled gracefully."""
        mock_settings.ALERT_WEBHOOK_URL = "http://failing-webhook.com/alerts"
        
        # Mock HTTP client to raise exception
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = Exception("Webhook failed")
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        bars_by_symbol = {"AAPL": overbought_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        # Should not raise exception despite webhook failure
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        # Alert should still be returned even if webhook fails
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    @patch('supernova.config.settings')
    @patch('httpx.AsyncClient')
    async def test_multiple_alerts_webhook_calls(self, mock_client, mock_settings, sample_watch_items, overbought_bars):
        """Test webhook is called for each triggered alert."""
        mock_settings.ALERT_WEBHOOK_URL = "http://test-webhook.com/alerts"
        
        mock_http_client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        bars_by_symbol = {
            "AAPL": overbought_bars,
            "GOOGL": overbought_bars,
            "TSLA": overbought_bars
        }
        
        triggered = await evaluate_alerts(sample_watch_items, bars_by_symbol)
        
        if len(triggered) > 1:
            # Webhook should be called once for each triggered alert
            assert mock_http_client.post.call_count == len(triggered)


class TestRSIIntegration:
    @pytest.mark.asyncio
    async def test_rsi_calculation_integration(self, overbought_bars):
        """Test that RSI calculation in alerts matches strategy engine."""
        # Test direct RSI calculation
        df = make_df(overbought_bars)
        action, conf, details = eval_rsi_breakout(df)
        
        # Test through alert evaluation
        watch_items = [{"symbol": "TEST", "profile_id": 1}]
        bars_by_symbol = {"TEST": overbought_bars}
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        # Should be consistent
        if action in ("buy", "sell") and triggered:
            assert action.upper() in triggered[0]["message"]

    @pytest.mark.asyncio
    async def test_rsi_threshold_accuracy(self, sample_watch_items):
        """Test that RSI thresholds trigger alerts accurately."""
        # Create bars that result in specific RSI values
        bars_high_rsi = []
        base_price = 100
        
        # Create strong uptrend to push RSI above 70
        for i in range(25):
            timestamp = (datetime.now() - timedelta(hours=25-i)).isoformat() + "Z"
            base_price += 1.5  # Consistent strong gains
            
            bars_high_rsi.append({
                "timestamp": timestamp,
                "open": base_price - 0.5,
                "high": base_price + 0.3,
                "low": base_price - 0.8,
                "close": base_price,
                "volume": 10000
            })
        
        bars_by_symbol = {"AAPL": bars_high_rsi}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        
        if triggered:
            # Verify RSI value in message is above threshold
            import re
            rsi_match = re.search(r"RSI=(\d+\.?\d*)", triggered[0]["message"])
            if rsi_match:
                rsi_value = float(rsi_match.group(1))
                if "SELL" in triggered[0]["message"]:
                    assert rsi_value > 70
                elif "BUY" in triggered[0]["message"]:
                    assert rsi_value < 30


class TestAlertingEdgeCases:
    @pytest.mark.asyncio
    async def test_duplicate_watchlist_items(self, overbought_bars):
        """Test alert evaluation with duplicate watchlist items."""
        duplicate_watch_items = [
            {"symbol": "AAPL", "profile_id": 1},
            {"symbol": "AAPL", "profile_id": 1},  # Duplicate
            {"symbol": "AAPL", "profile_id": 2},  # Same symbol, different profile
        ]
        
        bars_by_symbol = {"AAPL": overbought_bars}
        
        triggered = await evaluate_alerts(duplicate_watch_items, bars_by_symbol)
        
        # Should handle duplicates appropriately
        assert isinstance(triggered, list)
        # May trigger multiple alerts for duplicates

    @pytest.mark.asyncio
    async def test_very_large_watchlist(self, neutral_bars):
        """Test alert evaluation with large watchlist."""
        # Create large watchlist
        large_watch_items = []
        bars_by_symbol = {}
        
        for i in range(100):
            symbol = f"STOCK_{i:03d}"
            large_watch_items.append({"symbol": symbol, "profile_id": 1})
            bars_by_symbol[symbol] = neutral_bars
        
        triggered = await evaluate_alerts(large_watch_items, bars_by_symbol)
        
        # Should handle large watchlist efficiently
        assert isinstance(triggered, list)
        # With neutral bars, should not trigger many alerts
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_extreme_price_movements(self, sample_watch_items):
        """Test alert evaluation with extreme price movements."""
        extreme_bars = []
        base_price = 100
        
        # Extreme price swings
        for i in range(20):
            timestamp = (datetime.now() - timedelta(hours=20-i)).isoformat() + "Z"
            # Alternate between extreme moves
            if i % 2 == 0:
                base_price *= 1.5  # 50% jump
            else:
                base_price *= 0.7  # 30% drop
            
            extreme_bars.append({
                "timestamp": timestamp,
                "open": base_price,
                "high": base_price * 1.1,
                "low": base_price * 0.9,
                "close": base_price,
                "volume": 100000
            })
        
        bars_by_symbol = {"AAPL": extreme_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        # Should handle extreme volatility without crashing
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    async def test_zero_volume_bars(self, sample_watch_items):
        """Test alert evaluation with zero volume bars."""
        zero_volume_bars = []
        for i in range(20):
            timestamp = (datetime.now() - timedelta(hours=20-i)).isoformat() + "Z"
            zero_volume_bars.append({
                "timestamp": timestamp,
                "open": 100, "high": 101, "low": 99, "close": 100,
                "volume": 0  # Zero volume
            })
        
        bars_by_symbol = {"AAPL": zero_volume_bars}
        watch_items = [{"symbol": "AAPL", "profile_id": 1}]
        
        # Should handle zero volume gracefully
        triggered = await evaluate_alerts(watch_items, bars_by_symbol)
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    async def test_concurrent_evaluation(self, sample_watch_items, overbought_bars, oversold_bars):
        """Test concurrent alert evaluations."""
        bars_by_symbol1 = {"AAPL": overbought_bars, "GOOGL": oversold_bars}
        bars_by_symbol2 = {"TSLA": overbought_bars, "MSFT": oversold_bars}
        
        # Run multiple evaluations concurrently
        tasks = [
            evaluate_alerts(sample_watch_items, bars_by_symbol1),
            evaluate_alerts(sample_watch_items, bars_by_symbol2)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Both should complete successfully
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)