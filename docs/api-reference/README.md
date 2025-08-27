# SuperNova AI API Reference

Complete API documentation for the SuperNova AI financial advisory platform. This reference covers all endpoints, authentication, request/response schemas, and integration examples.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
4. [Chat and AI Advisory](#chat-and-ai-advisory)
5. [Portfolio Management](#portfolio-management)
6. [Market Data and Sentiment](#market-data-and-sentiment)
7. [Backtesting](#backtesting)
8. [Alerts and Notifications](#alerts-and-notifications)
9. [WebSocket API](#websocket-api)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [SDKs and Examples](#sdks-and-examples)

## API Overview

### Base Information

- **Base URL**: `https://api.supernova-ai.com` (Production)
- **Base URL**: `http://localhost:8081` (Development)
- **API Version**: 1.0.0
- **Protocol**: HTTPS (Production), HTTP (Development)
- **Content Type**: `application/json`
- **Documentation**: Available at `/docs` (OpenAPI/Swagger)

### API Architecture

SuperNova AI uses a RESTful API architecture with the following characteristics:

- **RESTful Design**: Standard HTTP methods (GET, POST, PUT, DELETE)
- **JSON Format**: All requests and responses use JSON
- **Stateless**: Each request contains all necessary information
- **Idempotent**: Safe methods can be repeated without side effects
- **Versioned**: API versions maintained for backward compatibility

### Key Features

- **35+ Endpoints**: Comprehensive API coverage
- **Real-time Updates**: WebSocket support for live data
- **High Performance**: Optimized for low-latency responses
- **Scalable**: Supports high concurrent user loads
- **Secure**: JWT authentication and rate limiting

## Authentication

### JWT Token Authentication

SuperNova AI uses JSON Web Tokens (JWT) for secure API access.

#### Obtaining a Token

```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Using Tokens

Include the JWT token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Token Refresh

```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### API Key Authentication (Alternative)

For server-to-server integrations, API keys can be used:

```http
X-API-Key: your_api_key_here
```

## Core Endpoints

### User Onboarding and Profiles

#### Create User Profile

Creates a new user profile with risk assessment.

```http
POST /intake
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "income": 75000,
  "expenses": 45000,
  "assets": 150000,
  "debts": 50000,
  "time_horizon_yrs": 10,
  "objectives": "growth with income",
  "constraints": "ESG investing preferred",
  "risk_questions": [3, 2, 4, 3, 2]
}
```

**Response:**
```json
{
  "profile_id": 123,
  "risk_score": 6
}
```

**Risk Score Scale:**
- 1-2: Conservative (Capital preservation)
- 3-4: Moderate Conservative (Income focus)
- 5-6: Moderate (Balanced)
- 7-8: Moderate Aggressive (Growth focus)
- 9-10: Aggressive (Maximum growth)

#### Get Investment Advice

Requests personalized investment advice for a specific security.

```http
POST /advice
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "profile_id": 123,
  "symbol": "AAPL",
  "asset_class": "stock",
  "timeframe": "1h",
  "bars": [
    {
      "timestamp": "2025-08-26T09:30:00Z",
      "open": 150.50,
      "high": 152.75,
      "low": 149.25,
      "close": 151.80,
      "volume": 1250000
    }
  ],
  "sentiment_hint": 0.3,
  "strategy_template": "balanced",
  "params": {
    "risk_adjustment": 0.8
  }
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1h",
  "action": "buy",
  "confidence": 0.75,
  "rationale": "Strong technical indicators and positive sentiment suggest upward momentum. RSI shows healthy levels and moving averages are trending positively.",
  "key_indicators": {
    "rsi": 65.2,
    "sma_20": 149.85,
    "ema_12": 150.92,
    "macd": 0.45,
    "bollinger_upper": 153.20,
    "bollinger_lower": 147.40
  },
  "risk_notes": "Consider position sizing based on portfolio allocation. Monitor earnings announcement scheduled next week."
}
```

**Action Types:**
- `buy`: Recommended purchase
- `sell`: Recommended sale
- `hold`: Maintain current position
- `reduce`: Decrease position size
- `avoid`: Do not invest

### Watchlist Management

#### Add to Watchlist

```http
POST /watchlist/add
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "profile_id": 123,
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "asset_class": "stock"
}
```

**Response:**
```json
{
  "added_ids": [1, 2, 3]
}
```

#### Get Watchlist

```http
GET /watchlist/{profile_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "watchlist_items": [
    {
      "id": 1,
      "symbol": "AAPL",
      "asset_class": "stock",
      "added_date": "2025-08-26T10:30:00Z",
      "notes": "High-growth potential",
      "active": true
    }
  ]
}
```

## Chat and AI Advisory

### Chat Interface

#### Send Chat Message

```http
POST /chat
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "message": "What's your outlook on the technology sector?",
  "profile_id": 123,
  "advisor_type": "growth",
  "session_id": "sess_abc123",
  "context": {
    "portfolio_value": 250000,
    "current_allocation": {
      "stocks": 0.7,
      "bonds": 0.2,
      "cash": 0.1
    }
  }
}
```

**Response:**
```json
{
  "message_id": "msg_xyz789",
  "response": "The technology sector shows strong fundamentals with continued AI adoption driving growth. However, valuations are elevated in some segments. Consider diversified tech ETFs rather than individual stock picking for balanced exposure.",
  "advisor_type": "growth",
  "confidence": 0.85,
  "suggestions": [
    "Consider VGT (Vanguard Information Technology ETF)",
    "Review current tech allocation (currently 25% of portfolio)",
    "Monitor earnings season for guidance updates"
  ],
  "timestamp": "2025-08-26T14:30:00Z",
  "session_id": "sess_abc123"
}
```

#### Get Chat History

```http
GET /chat/history/{session_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "messages": [
    {
      "message_id": "msg_001",
      "user_message": "What's your outlook on tech?",
      "ai_response": "Technology sector shows strong fundamentals...",
      "timestamp": "2025-08-26T14:30:00Z",
      "advisor_type": "growth"
    }
  ],
  "total_messages": 1,
  "session_start": "2025-08-26T14:30:00Z"
}
```

### AI Advisor Types

#### Available Advisor Personalities

1. **Conservative Advisor** (`conservative`)
   - Focus: Capital preservation and stability
   - Risk tolerance: Low
   - Suitable for: Retirement planning, risk-averse investors

2. **Balanced Advisor** (`balanced`)
   - Focus: Balanced growth and income
   - Risk tolerance: Moderate
   - Suitable for: Most individual investors

3. **Growth Advisor** (`growth`)
   - Focus: Capital appreciation
   - Risk tolerance: Moderate-High
   - Suitable for: Long-term growth objectives

4. **Aggressive Advisor** (`aggressive`)
   - Focus: Maximum growth potential
   - Risk tolerance: High
   - Suitable for: Young investors, high risk tolerance

5. **Income Advisor** (`income`)
   - Focus: Regular income generation
   - Risk tolerance: Low-Moderate
   - Suitable for: Retirees, income-focused investors

## Portfolio Management

### Portfolio Operations

#### Get Portfolio Summary

```http
GET /portfolio/{profile_id}/summary
Authorization: Bearer <token>
```

**Response:**
```json
{
  "profile_id": 123,
  "total_value": 250000.00,
  "daily_pnl": 1250.50,
  "daily_pnl_percent": 0.50,
  "total_return": 0.15,
  "annualized_return": 0.12,
  "holdings_count": 15,
  "last_updated": "2025-08-26T16:00:00Z",
  "allocation": {
    "stocks": 0.70,
    "bonds": 0.20,
    "cash": 0.08,
    "alternatives": 0.02
  }
}
```

#### Add Portfolio Position

```http
POST /portfolio/{profile_id}/positions
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "shares": 100,
  "purchase_price": 150.00,
  "purchase_date": "2025-08-26",
  "account_type": "taxable"
}
```

**Response:**
```json
{
  "position_id": 456,
  "symbol": "AAPL",
  "shares": 100,
  "current_price": 151.80,
  "market_value": 15180.00,
  "cost_basis": 15000.00,
  "unrealized_pnl": 180.00,
  "unrealized_pnl_percent": 0.012
}
```

### Performance Analytics

#### Get Performance Metrics

```http
GET /portfolio/{profile_id}/performance
Authorization: Bearer <token>
Query Parameters:
  - start_date: 2025-01-01
  - end_date: 2025-08-26
  - benchmark: SPY
```

**Response:**
```json
{
  "period": {
    "start_date": "2025-01-01",
    "end_date": "2025-08-26"
  },
  "returns": {
    "total_return": 0.12,
    "annualized_return": 0.15,
    "volatility": 0.18,
    "sharpe_ratio": 0.83,
    "max_drawdown": -0.08
  },
  "benchmark_comparison": {
    "benchmark": "SPY",
    "portfolio_return": 0.12,
    "benchmark_return": 0.10,
    "alpha": 0.02,
    "beta": 0.95,
    "tracking_error": 0.05
  },
  "risk_metrics": {
    "var_95": -0.025,
    "cvar_95": -0.035,
    "sortino_ratio": 1.12
  }
}
```

## Market Data and Sentiment

### Sentiment Analysis

#### Get Historical Sentiment

```http
GET /sentiment/historical/{symbol}
Authorization: Bearer <token>
Query Parameters:
  - start_date: 2025-08-01
  - end_date: 2025-08-26
  - interval: daily
```

**Response:**
```json
{
  "symbol": "AAPL",
  "data": [
    {
      "date": "2025-08-26",
      "overall_score": 0.35,
      "confidence": 0.82,
      "news_sentiment": 0.42,
      "social_sentiment": 0.28,
      "sources": {
        "twitter": 150,
        "reddit": 45,
        "news": 23
      }
    }
  ],
  "summary": {
    "average_sentiment": 0.31,
    "trend": "positive",
    "volatility": 0.12
  }
}
```

#### Bulk Sentiment Data

```http
POST /sentiment/historical
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "start_date": "2025-08-01T00:00:00Z",
  "end_date": "2025-08-26T23:59:59Z",
  "interval": "daily",
  "include_intraday": false
}
```

**Response:**
```json
{
  "symbols_found": ["AAPL", "MSFT", "GOOGL"],
  "symbols_not_found": [],
  "data": {
    "AAPL": [
      {
        "timestamp": "2025-08-26T00:00:00Z",
        "overall_score": 0.35,
        "confidence": 0.82
      }
    ]
  },
  "summary": {
    "total_data_points": 75,
    "date_range": {
      "start": "2025-08-01T00:00:00Z",
      "end": "2025-08-26T23:59:59Z"
    }
  }
}
```

### Market Data Integration

#### NovaSignal Integration

```http
GET /novasignal/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "connected",
  "last_update": "2025-08-26T16:00:00Z",
  "data_feeds": {
    "market_data": "active",
    "sentiment": "active",
    "news": "active"
  },
  "latency_ms": 45
}
```

## Backtesting

### Strategy Backtesting

#### Run Backtest

```http
POST /backtest
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "strategy_template": "sma_crossover",
  "params": {
    "fast_period": 10,
    "slow_period": 30
  },
  "symbol": "AAPL",
  "timeframe": "1d",
  "bars": [
    {
      "timestamp": "2025-08-01T00:00:00Z",
      "open": 145.50,
      "high": 147.25,
      "low": 144.75,
      "close": 146.80,
      "volume": 2500000
    }
  ],
  "use_vectorbt": true,
  "start_cash": 10000.0,
  "fees": 0.001,
  "slippage": 0.001
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "engine": "VectorBT",
  "metrics": {
    "total_return": 0.234,
    "annualized_return": 0.185,
    "volatility": 0.21,
    "sharpe_ratio": 0.88,
    "max_drawdown": -0.12,
    "win_rate": 0.64,
    "profit_factor": 1.45,
    "total_trades": 23,
    "avg_trade_duration": 5.2,
    "final_value": 12340.50
  },
  "notes": "Strong performance with consistent returns. Consider reducing position size during high volatility periods."
}
```

#### Available Strategy Templates

1. **sma_crossover**: Simple Moving Average Crossover
2. **ema_momentum**: Exponential Moving Average Momentum
3. **rsi_meanrev**: RSI Mean Reversion
4. **macd_trend**: MACD Trend Following
5. **bollinger_bands**: Bollinger Bands Strategy
6. **buy_and_hold**: Buy and Hold Benchmark

### VectorBT High-Performance Backtesting

#### Advanced Backtest with VectorBT

```http
POST /backtest/vectorbt
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "strategy_config": {
    "name": "custom_momentum",
    "parameters": {
      "lookback_period": 20,
      "momentum_threshold": 0.02,
      "stop_loss": 0.05,
      "take_profit": 0.10
    }
  },
  "data_config": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2023-01-01",
    "end_date": "2025-08-26",
    "timeframe": "1d"
  },
  "execution_config": {
    "initial_cash": 100000,
    "fees": 0.001,
    "slippage": 0.0005,
    "sizing": "equal_weight"
  }
}
```

**Response:**
```json
{
  "backtest_id": "bt_abc123",
  "status": "completed",
  "execution_time_ms": 245,
  "results": {
    "portfolio_metrics": {
      "total_return": 0.187,
      "annual_return": 0.156,
      "volatility": 0.19,
      "sharpe_ratio": 0.82,
      "sortino_ratio": 1.15,
      "max_drawdown": -0.098,
      "calmar_ratio": 1.59
    },
    "individual_securities": {
      "AAPL": {
        "return": 0.215,
        "trades": 12,
        "win_rate": 0.67
      },
      "MSFT": {
        "return": 0.198,
        "trades": 14,
        "win_rate": 0.71
      },
      "GOOGL": {
        "return": 0.145,
        "trades": 10,
        "win_rate": 0.60
      }
    }
  }
}
```

## Alerts and Notifications

### Alert Management

#### Create Price Alert

```http
POST /alerts/create
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "profile_id": 123,
  "symbol": "AAPL",
  "alert_type": "price",
  "condition": "above",
  "threshold": 155.00,
  "notification_methods": ["email", "push"],
  "expiry_date": "2025-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "alert_id": 789,
  "status": "active",
  "created_at": "2025-08-26T16:00:00Z"
}
```

#### Evaluate Alerts

```http
POST /alerts/evaluate
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "watch": [
    {
      "symbol": "AAPL",
      "alert_type": "rsi",
      "threshold": 70
    }
  ],
  "bars": {
    "AAPL": [
      {
        "timestamp": "2025-08-26T16:00:00Z",
        "close": 151.80,
        "volume": 1250000
      }
    ]
  }
}
```

**Response:**
```json
[
  {
    "id": 1,
    "symbol": "AAPL",
    "message": "RSI overbought condition detected (72.5)",
    "triggered_at": "2025-08-26T16:00:00Z"
  }
]
```

### Alert Types

1. **Price Alerts**
   - Above/below price thresholds
   - Percentage change alerts
   - Support/resistance breaks

2. **Technical Alerts**
   - RSI overbought/oversold
   - MACD signal changes
   - Moving average crossovers
   - Bollinger Band squeezes

3. **Volume Alerts**
   - Volume spike detection
   - Unusual volume patterns
   - Volume-price divergences

4. **Sentiment Alerts**
   - Sentiment score changes
   - Social media buzz spikes
   - News sentiment shifts

## WebSocket API

### Real-time Data Streaming

#### Connection

```javascript
const ws = new WebSocket('wss://api.supernova-ai.com/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your_jwt_token'
}));
```

#### Subscribe to Market Data

```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market_data',
  symbols: ['AAPL', 'MSFT', 'GOOGL']
}));
```

#### Market Data Message

```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "price": 151.80,
  "change": 1.25,
  "change_percent": 0.83,
  "volume": 1250000,
  "timestamp": "2025-08-26T16:00:00Z"
}
```

#### Chat Messages

```javascript
// Send chat message via WebSocket
ws.send(JSON.stringify({
  type: 'chat',
  message: 'What do you think about AAPL?',
  profile_id: 123,
  advisor_type: 'growth'
}));
```

**Response:**
```json
{
  "type": "chat_response",
  "message_id": "msg_xyz789",
  "response": "AAPL shows strong momentum with positive technical indicators...",
  "advisor_type": "growth",
  "timestamp": "2025-08-26T16:00:00Z"
}
```

## Error Handling

### Standard HTTP Status Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "symbol",
      "issue": "Symbol 'INVALID' not found"
    },
    "timestamp": "2025-08-26T16:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_REQUIRED`: Missing or invalid token
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `VALIDATION_ERROR`: Request validation failed
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `EXTERNAL_SERVICE_ERROR`: Third-party service issue
- `INSUFFICIENT_DATA`: Not enough data for analysis

## Rate Limiting

### Rate Limit Headers

All API responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1693065600
X-RateLimit-Window: 3600
```

### Rate Limits by Tier

#### Free Tier
- **Rate Limit**: 100 requests/hour
- **Burst**: 20 requests/minute
- **WebSocket**: 5 concurrent connections

#### Professional Tier
- **Rate Limit**: 1,000 requests/hour
- **Burst**: 100 requests/minute
- **WebSocket**: 20 concurrent connections

#### Enterprise Tier
- **Rate Limit**: 10,000 requests/hour
- **Burst**: 500 requests/minute
- **WebSocket**: 100 concurrent connections

### Rate Limit Best Practices

1. **Implement Exponential Backoff**: Retry with increasing delays
2. **Cache Responses**: Reduce redundant API calls
3. **Batch Requests**: Use bulk endpoints when available
4. **Monitor Headers**: Check rate limit headers
5. **Use WebSockets**: For real-time data needs

## SDKs and Examples

### Python SDK

#### Installation

```bash
pip install supernova-ai-sdk
```

#### Basic Usage

```python
from supernova_ai import SuperNovaClient

client = SuperNovaClient(
    api_key="your_api_key",
    base_url="https://api.supernova-ai.com"
)

# Create profile
profile = client.create_profile(
    name="John Doe",
    email="john@example.com",
    risk_questions=[3, 2, 4, 3, 2]
)

# Get investment advice
advice = client.get_advice(
    profile_id=profile.id,
    symbol="AAPL",
    bars=historical_data
)

print(f"Recommendation: {advice.action}")
print(f"Confidence: {advice.confidence}")
print(f"Rationale: {advice.rationale}")
```

### JavaScript SDK

#### Installation

```bash
npm install supernova-ai-sdk
```

#### Basic Usage

```javascript
import { SuperNovaClient } from 'supernova-ai-sdk';

const client = new SuperNovaClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.supernova-ai.com'
});

// Create profile
const profile = await client.createProfile({
  name: 'John Doe',
  email: 'john@example.com',
  riskQuestions: [3, 2, 4, 3, 2]
});

// Get investment advice
const advice = await client.getAdvice({
  profileId: profile.id,
  symbol: 'AAPL',
  bars: historicalData
});

console.log(`Recommendation: ${advice.action}`);
console.log(`Confidence: ${advice.confidence}`);
```

### cURL Examples

#### Create Profile

```bash
curl -X POST "https://api.supernova-ai.com/intake" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "time_horizon_yrs": 10,
    "objectives": "growth",
    "risk_questions": [3, 2, 4, 3, 2]
  }'
```

#### Get Investment Advice

```bash
curl -X POST "https://api.supernova-ai.com/advice" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": 123,
    "symbol": "AAPL",
    "timeframe": "1h",
    "bars": [
      {
        "timestamp": "2025-08-26T09:30:00Z",
        "open": 150.50,
        "high": 152.75,
        "low": 149.25,
        "close": 151.80,
        "volume": 1250000
      }
    ]
  }'
```

---

## Additional Resources

- **Interactive API Explorer**: [/docs](https://api.supernova-ai.com/docs)
- **Postman Collection**: [Download Collection](https://api.supernova-ai.com/postman)
- **SDK Documentation**: [GitHub Repository](https://github.com/supernova-ai/sdks)
- **Support**: [support@supernova-ai.com](mailto:support@supernova-ai.com)

Last Updated: August 26, 2025  
Version: 1.0.0