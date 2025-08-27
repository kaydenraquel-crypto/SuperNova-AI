"""NovaSignal Integration Examples

This module provides comprehensive examples of how to use the NovaSignal integration
for various use cases including real-time data streaming, alert management, and
profile synchronization.

Usage Examples:
1. Real-time OHLCV streaming
2. Historical data fetching  
3. Alert sending and management
4. Profile and watchlist synchronization
5. Health monitoring and diagnostics
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_real_time_streaming():
    """Example: Stream real-time OHLCV data from NovaSignal"""
    from .novasignal import get_connector, novasignal_connection
    from .websocket_client import create_ohlcv_stream
    
    logger.info("Starting real-time OHLCV streaming example...")
    
    def ohlcv_callback(data: Dict[str, Any]):
        """Custom callback for OHLCV data processing"""
        symbol = data.get("symbol")
        close_price = data.get("close")
        volume = data.get("volume")
        timestamp = data.get("timestamp")
        
        logger.info(f"[{timestamp}] {symbol}: Price=${close_price:.2f}, Volume={volume:,}")
        
        # Example: Detect significant price movements
        if "prev_price" in ohlcv_callback.__dict__:
            prev_price = ohlcv_callback.prev_price.get(symbol, close_price)
            change_percent = ((close_price - prev_price) / prev_price) * 100
            
            if abs(change_percent) > 2.0:  # 2% price movement
                logger.warning(f"Significant price movement: {symbol} {change_percent:+.2f}%")
        else:
            ohlcv_callback.prev_price = {}
        
        ohlcv_callback.prev_price[symbol] = close_price
    
    try:
        # Create WebSocket client with custom callback
        symbols = ["AAPL", "GOOGL", "TSLA", "BTC-USD", "EUR-USD"]
        client = await create_ohlcv_stream(
            symbols=symbols,
            timeframes=["1m", "5m"],
            callback=ohlcv_callback
        )
        
        logger.info(f"Streaming data for {len(symbols)} symbols...")
        
        # Let it run for demonstration (in production, this would run indefinitely)
        await asyncio.sleep(60)
        
    except Exception as e:
        logger.error(f"Streaming example failed: {e}")
    finally:
        logger.info("Streaming example completed")


async def example_historical_data_analysis():
    """Example: Fetch and analyze historical data"""
    from .novasignal import get_connector, fetch_historical_ohlcv
    
    logger.info("Starting historical data analysis example...")
    
    try:
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in symbols:
            # Fetch historical data
            bars = await fetch_historical_ohlcv(
                symbol=symbol,
                timeframe="1h",
                limit=100
            )
            
            if bars:
                # Simple analysis
                prices = [bar.close for bar in bars]
                high_price = max(prices)
                low_price = min(prices)
                avg_price = sum(prices) / len(prices)
                latest_price = prices[-1]
                
                # Calculate price volatility
                price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                avg_volatility = sum(price_changes) / len(price_changes)
                
                logger.info(f"{symbol} Analysis:")
                logger.info(f"  Latest Price: ${latest_price:.2f}")
                logger.info(f"  Range: ${low_price:.2f} - ${high_price:.2f}")
                logger.info(f"  Average: ${avg_price:.2f}")
                logger.info(f"  Avg Volatility: ${avg_volatility:.2f}")
                
                # Generate trading signal based on simple logic
                if latest_price > avg_price * 1.05:
                    logger.info(f"  Signal: SELL (Overbought)")
                elif latest_price < avg_price * 0.95:
                    logger.info(f"  Signal: BUY (Oversold)")
                else:
                    logger.info(f"  Signal: HOLD (Neutral)")
                
                logger.info("")
        
    except Exception as e:
        logger.error(f"Historical data example failed: {e}")


async def example_alert_management():
    """Example: Send different types of alerts to NovaSignal"""
    from .novasignal import get_connector, AlertPriority
    
    logger.info("Starting alert management example...")
    
    try:
        connector = await get_connector()
        
        # Example alerts with different priorities and data
        alerts = [
            {
                "symbol": "AAPL",
                "message": "Strong bullish breakout detected above resistance at $150",
                "priority": AlertPriority.HIGH,
                "confidence": 0.85,
                "action": "buy",
                "strategy": "breakout_strategy",
                "indicators": {
                    "rsi": 65.2,
                    "macd": 1.23,
                    "resistance_level": 150.0
                }
            },
            {
                "symbol": "TSLA",
                "message": "Bearish divergence forming - consider reducing position",
                "priority": AlertPriority.MEDIUM,
                "confidence": 0.72,
                "action": "reduce",
                "strategy": "divergence_strategy",
                "indicators": {
                    "rsi": 42.1,
                    "macd": -0.87,
                    "volume_ratio": 0.65
                }
            },
            {
                "symbol": "BTC-USD",
                "message": "Extreme volatility detected - risk management recommended",
                "priority": AlertPriority.CRITICAL,
                "confidence": 0.95,
                "action": "avoid",
                "strategy": "volatility_monitor",
                "indicators": {
                    "volatility": 45.2,
                    "vix": 28.7,
                    "volume_spike": 3.2\n                }\n            }\n        ]\n        \n        for alert in alerts:\n            success = await connector.push_alert_to_novasignal(**alert)\n            if success:\n                logger.info(f\"Alert sent successfully: {alert['symbol']} - {alert['priority'].value}\")\n            else:\n                logger.error(f\"Failed to send alert: {alert['symbol']}\")\n            \n            # Small delay between alerts\n            await asyncio.sleep(1)\n    \n    except Exception as e:\n        logger.error(f\"Alert management example failed: {e}\")\n\n\nasync def example_profile_synchronization():\n    \"\"\"Example: Synchronize user profiles and watchlists\"\"\"\n    from .novasignal import get_connector\n    \n    logger.info(\"Starting profile synchronization example...\")\n    \n    try:\n        connector = await get_connector()\n        \n        # Example profile data\n        profile_data = {\n            \"risk_tolerance\": \"moderate\",\n            \"investment_horizon\": \"long_term\",\n            \"preferred_asset_classes\": [\"stocks\", \"crypto\"],\n            \"maximum_position_size\": 0.05,  # 5% max per position\n            \"stop_loss_preference\": 0.08,   # 8% stop loss\n            \"notification_preferences\": {\n                \"email\": True,\n                \"push\": True,\n                \"sms\": False\n            }\n        }\n        \n        profile_id = 12345  # Example profile ID\n        \n        # Sync profile to NovaSignal\n        success = await connector.sync_profile(profile_id, profile_data)\n        if success:\n            logger.info(f\"Profile {profile_id} synchronized successfully\")\n        else:\n            logger.error(f\"Failed to sync profile {profile_id}\")\n        \n        # Fetch watchlist from NovaSignal\n        watchlist = await connector.get_watchlist(profile_id)\n        logger.info(f\"Retrieved watchlist with {len(watchlist)} items:\")\n        for item in watchlist:\n            symbol = item.get(\"symbol\")\n            notes = item.get(\"notes\", \"No notes\")\n            logger.info(f\"  {symbol}: {notes}\")\n    \n    except Exception as e:\n        logger.error(f\"Profile synchronization example failed: {e}\")\n\n\nasync def example_health_monitoring():\n    \"\"\"Example: Monitor NovaSignal integration health\"\"\"\n    from .novasignal import get_connector, health_check\n    \n    logger.info(\"Starting health monitoring example...\")\n    \n    try:\n        # Perform comprehensive health check\n        health_status = await health_check()\n        \n        logger.info(\"NovaSignal Integration Health Status:\")\n        logger.info(f\"  Overall Status: {health_status.get('status', 'unknown')}\")\n        \n        if \"connection_details\" in health_status:\n            details = health_status[\"connection_details\"]\n            logger.info(f\"  Connection State: {details.get('connection_state', 'unknown')}\")\n            logger.info(f\"  WebSocket Connected: {details.get('websocket_connected', False)}\")\n            logger.info(f\"  HTTP Session Active: {details.get('http_session_active', False)}\")\n            logger.info(f\"  Redis Available: {details.get('redis_available', False)}\")\n            logger.info(f\"  Active Subscriptions: {details.get('active_subscriptions', 0)}\")\n            logger.info(f\"  Circuit Breaker State: {details.get('circuit_breaker_state', 'unknown')}\")\n            logger.info(f\"  Buffer Size: {details.get('buffer_size', 0)}\")\n        \n        # Get detailed connection info\n        connector = await get_connector()\n        status = await connector.get_connection_status()\n        \n        # Monitor connection quality over time\n        logger.info(\"\\nMonitoring connection for 30 seconds...\")\n        for i in range(6):  # 6 checks over 30 seconds\n            await asyncio.sleep(5)\n            current_status = await connector.get_connection_status()\n            \n            if current_status[\"connection_state\"] != \"connected\":\n                logger.warning(f\"Connection issue detected: {current_status['connection_state']}\")\n            else:\n                logger.info(f\"Check {i+1}: Connection healthy\")\n    \n    except Exception as e:\n        logger.error(f\"Health monitoring example failed: {e}\")\n\n\nasync def example_advanced_streaming_with_analysis():\n    \"\"\"Example: Advanced streaming with real-time analysis\"\"\"\n    from .websocket_client import create_multi_stream\n    from .novasignal import get_connector, AlertPriority\n    \n    logger.info(\"Starting advanced streaming analysis example...\")\n    \n    # Data storage for analysis\n    price_history = {}\n    trade_counts = {}\n    \n    async def ohlcv_handler(data: Dict[str, Any]):\n        \"\"\"Handle OHLCV data with trend analysis\"\"\"\n        symbol = data.get(\"symbol\")\n        close = data.get(\"close\")\n        \n        if symbol not in price_history:\n            price_history[symbol] = []\n        \n        price_history[symbol].append(close)\n        \n        # Keep only last 20 prices for moving average\n        if len(price_history[symbol]) > 20:\n            price_history[symbol].pop(0)\n        \n        # Calculate simple moving average\n        if len(price_history[symbol]) >= 10:\n            sma = sum(price_history[symbol][-10:]) / 10\n            \n            # Detect trend changes\n            if close > sma * 1.02:  # 2% above MA\n                logger.info(f\"{symbol}: Strong uptrend - Price: ${close:.2f}, SMA: ${sma:.2f}\")\n            elif close < sma * 0.98:  # 2% below MA\n                logger.info(f\"{symbol}: Strong downtrend - Price: ${close:.2f}, SMA: ${sma:.2f}\")\n    \n    async def trade_handler(data: Dict[str, Any]):\n        \"\"\"Handle trade data with volume analysis\"\"\"\n        symbol = data.get(\"symbol\")\n        volume = data.get(\"volume\", 0)\n        \n        if symbol not in trade_counts:\n            trade_counts[symbol] = 0\n        \n        trade_counts[symbol] += 1\n        \n        # Log high-volume trades\n        if volume > 10000:  # Threshold for \"large\" trade\n            logger.info(f\"{symbol}: Large trade detected - Volume: {volume:,}\")\n    \n    async def news_handler(data: Dict[str, Any]):\n        \"\"\"Handle news with sentiment impact analysis\"\"\"\n        headline = data.get(\"headline\", \"\")\n        symbols = data.get(\"symbols\", [])\n        sentiment = data.get(\"sentiment\", 0)  # Assume sentiment score provided\n        \n        # Alert on significant news\n        if abs(sentiment) > 0.7:  # Strong positive or negative sentiment\n            sentiment_label = \"POSITIVE\" if sentiment > 0 else \"NEGATIVE\"\n            logger.warning(f\"HIGH IMPACT NEWS ({sentiment_label}): {headline}\")\n            \n            # Send alert for affected symbols\n            connector = await get_connector()\n            for symbol in symbols:\n                await connector.push_alert_to_novasignal(\n                    symbol=symbol,\n                    message=f\"High-impact {sentiment_label.lower()} news: {headline[:100]}...\",\n                    priority=AlertPriority.HIGH,\n                    confidence=abs(sentiment),\n                    indicators={\"news_sentiment\": sentiment}\n                )\n    \n    try:\n        # Configure multi-stream client\n        stream_config = {\n            \"ohlcv\": {\n                \"symbols\": [\"AAPL\", \"GOOGL\", \"TSLA\"],\n                \"timeframes\": [\"1m\"],\n                \"callback\": ohlcv_handler\n            },\n            \"trades\": {\n                \"symbols\": [\"AAPL\", \"GOOGL\", \"TSLA\"],\n                \"callback\": trade_handler\n            },\n            \"news\": {\n                \"symbols\": [\"AAPL\", \"GOOGL\", \"TSLA\"],\n                \"categories\": [\"earnings\", \"analyst_ratings\"],\n                \"callback\": news_handler\n            }\n        }\n        \n        client = await create_multi_stream(stream_config)\n        \n        # Run analysis for demonstration period\n        logger.info(\"Advanced streaming analysis running...\")\n        await asyncio.sleep(120)  # Run for 2 minutes\n        \n        # Print summary\n        logger.info(\"\\nAnalysis Summary:\")\n        for symbol, prices in price_history.items():\n            if prices:\n                logger.info(f\"{symbol}: {len(prices)} price updates, Current: ${prices[-1]:.2f}\")\n        \n        for symbol, count in trade_counts.items():\n            logger.info(f\"{symbol}: {count} trades processed\")\n    \n    except Exception as e:\n        logger.error(f\"Advanced streaming example failed: {e}\")\n    finally:\n        if 'client' in locals():\n            await client.disconnect()\n\n\nasync def run_all_examples():\n    \"\"\"Run all NovaSignal integration examples\"\"\"\n    examples = [\n        (\"Historical Data Analysis\", example_historical_data_analysis),\n        (\"Alert Management\", example_alert_management),\n        (\"Profile Synchronization\", example_profile_synchronization),\n        (\"Health Monitoring\", example_health_monitoring),\n        # Note: Streaming examples commented out for batch execution\n        # (\"Real-time Streaming\", example_real_time_streaming),\n        # (\"Advanced Streaming Analysis\", example_advanced_streaming_with_analysis),\n    ]\n    \n    logger.info(\"Running all NovaSignal integration examples...\")\n    \n    for name, example_func in examples:\n        logger.info(f\"\\n{'='*50}\")\n        logger.info(f\"Running: {name}\")\n        logger.info(f\"{'='*50}\")\n        \n        try:\n            await example_func()\n            logger.info(f\"✓ {name} completed successfully\")\n        except Exception as e:\n            logger.error(f\"✗ {name} failed: {e}\")\n        \n        # Pause between examples\n        await asyncio.sleep(2)\n    \n    logger.info(\"\\nAll examples completed!\")\n\n\nif __name__ == \"__main__\":\n    # Run examples\n    asyncio.run(run_all_examples())