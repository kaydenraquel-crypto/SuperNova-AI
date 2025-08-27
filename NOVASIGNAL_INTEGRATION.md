# NovaSignal Integration Documentation

## Overview

The NovaSignal integration provides comprehensive bidirectional connectivity between SuperNova financial advisor platform and NovaSignal's market data and trading platform. This integration enables real-time market data streaming, alert delivery, profile synchronization, and seamless user experience across both platforms.

## Features

### 🔄 **Bidirectional Data Flow**
- **SuperNova → NovaSignal**: Alerts, advice notifications, analysis results
- **NovaSignal → SuperNova**: Market data, user requests, profile updates

### 📊 **Real-time Market Data Streaming**
- WebSocket-based OHLCV data streaming
- Support for multiple asset classes (stocks, crypto, forex, futures, options)
- Configurable timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Trade data and order book streaming
- News and sentiment data integration

### 🚨 **Advanced Alert System**
- Priority-based alert routing (Low, Medium, High, Critical)
- Rich alert metadata with confidence scores and indicators
- Webhook delivery with HMAC signature verification
- Alert acknowledgment and feedback loops

### 👤 **Profile & Watchlist Synchronization**
- User profile data synchronization
- Real-time watchlist updates
- Portfolio state management
- Trading history integration

### ⚡ **Performance & Reliability**
- Circuit breaker patterns for fault tolerance
- Automatic reconnection with exponential backoff
- Redis-based caching and buffering
- Connection pooling for high-frequency data
- Comprehensive health monitoring

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NovaSignal    │◄──►│    SuperNova     │◄──►│   SuperNova     │
│    Platform     │    │   Connectors     │    │   Core API      │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Market Data   │    │ • WebSocket      │    │ • Advice Engine │
│ • User Requests │    │ • REST API       │    │ • Alert System  │
│ • Profile Mgmt  │    │ • Circuit Breakers│    │ • Backtesting   │
│ • Dashboard UI  │    │ • Health Monitor │    │ • Strategy Eng  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create or update your `.env` file with NovaSignal settings:

```env
# NovaSignal API Configuration
NOVASIGNAL_API_URL=https://api.novasignal.io
NOVASIGNAL_WS_URL=wss://ws.novasignal.io
NOVASIGNAL_API_KEY=your_api_key_here
NOVASIGNAL_SECRET=your_secret_here
NOVASIGNAL_WEBHOOK_SECRET=webhook_secret_here

# NovaSignal Endpoints
NOVASIGNAL_ALERT_ENDPOINT=/webhooks/supernova/alerts
NOVASIGNAL_DASHBOARD_URL=https://dashboard.novasignal.io

# Performance Settings
NOVASIGNAL_CONNECTION_TIMEOUT=30
NOVASIGNAL_MAX_RETRIES=3
NOVASIGNAL_BUFFER_SIZE=1000

# Redis Configuration (optional but recommended)
REDIS_URL=redis://localhost:6379/0
```

### 3. Initialize Connection

```python
from connectors.novasignal import get_connector, novasignal_connection

# Get global connector instance
connector = await get_connector()

# Or use context manager
async with novasignal_connection() as connector:
    # Use connector here
    pass
```

## Usage Examples

### Real-time Market Data Streaming

```python
from connectors.novasignal import stream_ohlcv_data

async def market_data_callback(data):
    symbol = data['symbol']
    price = data['close']
    print(f\"{symbol}: ${price:.2f}\")

# Start streaming
await stream_ohlcv_data(
    symbols=[\"AAPL\", \"GOOGL\", \"TSLA\"],
    callback=market_data_callback
)
```

### Historical Data Fetching

```python
from connectors.novasignal import fetch_historical_ohlcv

# Fetch 100 hourly bars for AAPL
bars = await fetch_historical_ohlcv(
    symbol=\"AAPL\",
    timeframe=\"1h\",
    limit=100
)

for bar in bars[-5:]:  # Last 5 bars
    print(f\"{bar.timestamp}: O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}\")
```

### Sending Alerts

```python
from connectors.novasignal import get_connector, AlertPriority

connector = await get_connector()

await connector.push_alert_to_novasignal(
    symbol=\"AAPL\",
    message=\"Strong bullish breakout above $150 resistance\",
    priority=AlertPriority.HIGH,
    confidence=0.85,
    action=\"buy\",
    strategy=\"breakout_strategy\",
    indicators={\n        \"rsi\": 65.2,\n        \"macd\": 1.23,\n        \"resistance_level\": 150.0\n    }\n)\n```\n\n### Profile Synchronization\n\n```python\nconnector = await get_connector()\n\n# Sync profile to NovaSignal\nprofile_data = {\n    \"risk_tolerance\": \"moderate\",\n    \"investment_horizon\": \"long_term\",\n    \"preferred_assets\": [\"stocks\", \"crypto\"]\n}\n\nawait connector.sync_profile(profile_id=12345, profile_data=profile_data)\n\n# Get watchlist from NovaSignal\nwatchlist = await connector.get_watchlist(profile_id=12345)\nprint(f\"Watchlist has {len(watchlist)} items\")\n```\n\n### Health Monitoring\n\n```python\nfrom connectors.novasignal import health_check\n\n# Comprehensive health check\nhealth_status = await health_check()\nprint(f\"Integration status: {health_status['status']}\")\n\n# Detailed connection info\nconnector = await get_connector()\nstatus = await connector.get_connection_status()\nprint(f\"WebSocket connected: {status['websocket_connected']}\")\n```\n\n## API Endpoints\n\nThe integration adds several REST API endpoints to the SuperNova API:\n\n### Alert Management\n- `POST /novasignal/alerts/send` - Send custom alert to NovaSignal\n- `GET /novasignal/status` - Get integration connection status\n\n### Data Access\n- `GET /novasignal/historical/{symbol}` - Fetch historical data from NovaSignal\n- `GET /novasignal/watchlist/{profile_id}` - Sync watchlist from NovaSignal\n\n### Profile Management\n- `POST /novasignal/profile/{profile_id}/sync` - Sync profile data to NovaSignal\n- `POST /novasignal/advice-explanation/{profile_id}` - Send advice explanation to NovaSignal UI\n\n### Health & Monitoring\n- `GET /novasignal/health` - Comprehensive health check\n\n### Example API Usage\n\n```bash\n# Send alert\ncurl -X POST \"http://localhost:8000/novasignal/alerts/send\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\n    \"symbol\": \"AAPL\",\n    \"message\": \"Price target reached\",\n    \"priority\": \"high\",\n    \"confidence\": 0.9\n  }'\n\n# Get integration status\ncurl \"http://localhost:8000/novasignal/status\"\n\n# Fetch historical data\ncurl \"http://localhost:8000/novasignal/historical/AAPL?timeframe=1h&limit=50\"\n```\n\n## WebSocket Client Usage\n\nFor advanced streaming use cases, use the specialized WebSocket client:\n\n```python\nfrom connectors.websocket_client import NovaSignalWebSocketClient\n\nclient = NovaSignalWebSocketClient()\n\nif await client.connect():\n    # Subscribe to different data streams\n    await client.subscribe_ohlcv([\"AAPL\", \"GOOGL\"], [\"1m\", \"5m\"])\n    await client.subscribe_trades([\"BTC-USD\", \"ETH-USD\"])\n    await client.subscribe_news(categories=[\"earnings\", \"analyst_ratings\"])\n    \n    # Start listening\n    await client.listen()\nelse:\n    print(\"Failed to connect\")\n```\n\n## Error Handling & Circuit Breakers\n\nThe integration includes robust error handling:\n\n```python\nfrom connectors.novasignal import get_connector\n\ntry:\n    connector = await get_connector()\n    data = await connector.get_historical_data(\"AAPL\")\nexcept Exception as e:\n    # Circuit breaker will handle retries automatically\n    print(f\"Request failed: {e}\")\n    \n    # Check circuit breaker state\n    status = await connector.get_connection_status()\n    print(f\"Circuit breaker: {status['circuit_breaker_state']}\")\n```\n\n## Configuration Options\n\n### Connection Settings\n- `NOVASIGNAL_CONNECTION_TIMEOUT`: HTTP request timeout (default: 30s)\n- `NOVASIGNAL_MAX_RETRIES`: Max retry attempts (default: 3)\n- `NOVASIGNAL_BUFFER_SIZE`: WebSocket buffer size (default: 1000)\n\n### Caching & Performance\n- `REDIS_URL`: Redis connection for caching (optional)\n- Historical data cached for 5 minutes\n- Real-time data buffered for processing\n\n### Security\n- `NOVASIGNAL_WEBHOOK_SECRET`: HMAC signature verification\n- API key authentication for all requests\n- TLS encryption for WebSocket connections\n\n## Monitoring & Diagnostics\n\n### Health Check Metrics\n- Connection state (connected/disconnected/error)\n- WebSocket connection status\n- HTTP session health\n- Redis availability\n- Active subscription count\n- Circuit breaker state\n- Message buffer utilization\n\n### Logging\n\nThe integration provides comprehensive logging:\n\n```python\nimport logging\n\n# Configure logging level\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger('connectors.novasignal')\n\n# Key events logged:\n# - Connection establishment/loss\n# - Data streaming activity  \n# - Alert delivery status\n# - Error conditions and retries\n# - Performance metrics\n```\n\n## Troubleshooting\n\n### Common Issues\n\n1. **Connection Failures**\n   - Verify API credentials are correct\n   - Check network connectivity to NovaSignal endpoints\n   - Ensure firewall allows WebSocket connections\n\n2. **Data Streaming Issues**\n   - Check subscription parameters (symbols, timeframes)\n   - Verify WebSocket connection is established\n   - Monitor buffer usage for overflow\n\n3. **Alert Delivery Problems**\n   - Confirm webhook endpoint is accessible\n   - Verify HMAC signature configuration\n   - Check alert priority and formatting\n\n### Debug Mode\n\n```python\nimport logging\nlogging.getLogger('connectors.novasignal').setLevel(logging.DEBUG)\n\n# This will show detailed connection logs, message contents, etc.\n```\n\n### Health Check Endpoint\n\n```bash\n# Monitor integration health\ncurl \"http://localhost:8000/novasignal/health\" | jq\n```\n\n## Performance Considerations\n\n### Optimization Tips\n\n1. **Use Redis for caching** - Significantly improves response times\n2. **Batch alert sending** - Use `batch_send_alerts()` for multiple alerts\n3. **Configure appropriate buffer sizes** - Match your data volume\n4. **Monitor circuit breaker state** - Indicates integration health\n5. **Use connection pooling** - Reuse connections when possible\n\n### Scaling Guidelines\n\n- **Small Scale**: Single connection, <100 symbols\n- **Medium Scale**: Connection pooling, <1000 symbols, Redis recommended\n- **Large Scale**: Multiple instances, load balancing, dedicated Redis cluster\n\n## Security Best Practices\n\n1. **Secure API Keys**: Use environment variables, never commit to code\n2. **Webhook Signatures**: Always verify HMAC signatures\n3. **TLS Encryption**: Use HTTPS/WSS endpoints only\n4. **Rate Limiting**: Respect NovaSignal's rate limits\n5. **Access Controls**: Limit integration access to authorized users\n\n## Support & Resources\n\n### Integration Examples\nSee `connectors/novasignal_examples.py` for comprehensive usage examples.\n\n### API Documentation\nRefer to NovaSignal's official API documentation for endpoint details.\n\n### Error Codes\n- `CONNECTION_ERROR`: Network connectivity issues\n- `AUTHENTICATION_ERROR`: Invalid API credentials  \n- `RATE_LIMIT_ERROR`: API rate limit exceeded\n- `DATA_ERROR`: Invalid or malformed data\n- `CIRCUIT_BREAKER_OPEN`: Automatic failure protection active\n\n---\n\n## ROADMAP Phase 6 Completion Status\n\n✅ **Real-time OHLCV data streaming** from NovaSignal  \n✅ **WebSocket implementation** for live market data  \n✅ **REST API endpoints** for historical data requests  \n✅ **Multiple asset class support** (stocks, crypto, FX, futures, options)  \n✅ **Webhook system** to send SuperNova alerts to NovaSignal  \n✅ **Alert formatting** for NovaSignal dashboard display  \n✅ **Priority-based alert routing**  \n✅ **Alert acknowledgment feedback loop**  \n✅ **Bidirectional API framework** with authentication  \n✅ **Error handling and retry logic**  \n✅ **Circuit breaker patterns** for fault tolerance  \n✅ **User profile synchronization** between platforms  \n✅ **Watchlist synchronization**  \n✅ **Performance optimization** with caching and connection pooling  \n✅ **Health monitoring and diagnostics**  \n✅ **Comprehensive documentation and examples**  \n\n**Integration Status: ✅ COMPLETE**\n\nThe NovaSignal integration is now fully operational and ready for production use, providing seamless bidirectional connectivity between SuperNova and NovaSignal platforms.