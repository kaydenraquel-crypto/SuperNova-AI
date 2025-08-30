"""
History API Router
Historical market data endpoints for SuperNova AI
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["market-data"])

# ====================================
# HELPER FUNCTIONS
# ====================================

def validate_symbol(symbol: str, market: str) -> bool:
    """Validate symbol format based on market type"""
    if not symbol or len(symbol.strip()) == 0:
        return False
        
    symbol = symbol.upper().strip()
    
    if market == "crypto":
        # Accept formats like BTC, BTC/USD, BTCUSD
        return len(symbol) >= 3
    elif market == "stocks":
        # Accept formats like AAPL, TSLA
        return len(symbol) >= 1 and symbol.isalpha()
    else:
        # Default validation
        return len(symbol) >= 1

def generate_sample_historical_data(
    symbol: str, 
    market: str, 
    interval: int, 
    days: int,
    provider: str
) -> List[Dict[str, Any]]:
    """Generate sample historical OHLCV data"""
    try:
        # Calculate number of data points
        minutes_per_day = 24 * 60
        total_minutes = days * minutes_per_day
        data_points = min(total_minutes // interval, 1000)  # Cap at 1000 points
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        # Set base price based on market
        if market == "crypto":
            base_price = 50000.0 if "BTC" in symbol.upper() else 2000.0
        else:  # stocks
            base_price = 150.0
        
        # Generate price series with realistic volatility
        returns = np.random.normal(0.0005, 0.02, data_points)  # 0.05% return, 2% volatility
        prices = [base_price]
        
        for ret in returns:
            next_price = prices[-1] * (1 + ret)
            prices.append(max(next_price, 0.01))  # Ensure positive prices
        
        # Generate OHLCV data
        historical_data = []
        start_time = datetime.now() - timedelta(minutes=data_points * interval)
        
        for i in range(data_points):
            timestamp = start_time + timedelta(minutes=i * interval)
            
            open_price = prices[i]
            close_price = prices[i + 1]
            
            # Generate realistic high/low based on volatility
            volatility = abs(np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + volatility)
            low_price = min(open_price, close_price) * (1 - volatility)
            
            # Generate volume (higher volume for crypto)
            if market == "crypto":
                volume = np.random.randint(100, 50000)
            else:
                volume = np.random.randint(1000, 1000000)
            
            historical_data.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            })
        
        return historical_data[-data_points:] if historical_data else []
        
    except Exception as e:
        logger.error(f"Error generating sample data for {symbol}: {e}")
        return []

def get_supported_providers() -> List[str]:
    """Get list of supported data providers"""
    return ["auto", "sample", "binance", "alphavantage", "yahoo", "polygon"]

def get_supported_markets() -> List[str]:
    """Get list of supported markets"""
    return ["crypto", "stocks", "forex", "commodities"]

# ====================================
# ENDPOINTS
# ====================================

@router.get("/history")
async def api_history(
    symbol: str = Query(..., description="Trading symbol (e.g., BTC, AAPL)"),
    market: str = Query("crypto", description="Market type"),
    interval: int = Query(1, ge=1, le=1440, description="Time interval in minutes"),
    days: int = Query(30, ge=1, le=365, description="Number of days of historical data"),
    provider: str = Query("auto", description="Data provider")
):
    """
    Enhanced history endpoint supporting all US stocks and major cryptocurrencies
    
    Parameters:
    - symbol: Trading symbol (BTC, AAPL, etc.)
    - market: Market type (crypto, stocks, forex, commodities)
    - interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440)
    - days: Number of days of historical data (1-365)
    - provider: Data provider (auto, binance, alphavantage, yahoo, polygon)
    
    Returns:
    - Historical OHLCV data with metadata
    """
    try:
        # Validate inputs
        if not validate_symbol(symbol, market):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid symbol '{symbol}' for market '{market}'"
            )
        
        if market not in get_supported_markets():
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported market '{market}'. Supported: {get_supported_markets()}"
            )
        
        if provider not in get_supported_providers():
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported provider '{provider}'. Supported: {get_supported_providers()}"
            )
        
        # Common interval validation
        valid_intervals = [1, 5, 15, 30, 60, 240, 1440]  # 1m, 5m, 15m, 30m, 1h, 4h, 1d
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported interval '{interval}'. Supported: {valid_intervals}"
            )
        
        logger.info(f"Fetching historical data for {symbol} ({market}) - {days} days, {interval}min intervals")
        
        # Generate sample data (in production, this would fetch from real sources)
        historical_data = generate_sample_historical_data(symbol, market, interval, days, provider)
        
        if not historical_data:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for symbol '{symbol}'"
            )
        
        # Prepare response
        response = {
            "symbol": symbol.upper(),
            "market": market,
            "interval": interval,
            "days": days,
            "provider": provider,
            "data": historical_data,
            "metadata": {
                "total_points": len(historical_data),
                "start_time": historical_data[0]["timestamp"] if historical_data else None,
                "end_time": historical_data[-1]["timestamp"] if historical_data else None,
                "generated_at": datetime.now().isoformat(),
                "data_source": "sample_generator",  # In production: actual provider
                "interval_minutes": interval,
                "market_type": market
            },
            "status": "success"
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch historical data: {str(e)}"
        )

@router.get("/history/markets")
async def get_supported_markets_info():
    """Get information about supported markets"""
    return {
        "supported_markets": get_supported_markets(),
        "market_details": {
            "crypto": {
                "description": "Cryptocurrency markets",
                "examples": ["BTC", "ETH", "ADA", "DOT"],
                "intervals": [1, 5, 15, 30, 60, 240, 1440]
            },
            "stocks": {
                "description": "US Stock markets",
                "examples": ["AAPL", "TSLA", "GOOGL", "MSFT"],
                "intervals": [1, 5, 15, 30, 60, 240, 1440]
            },
            "forex": {
                "description": "Foreign exchange markets",
                "examples": ["EURUSD", "GBPUSD", "USDJPY"],
                "intervals": [1, 5, 15, 30, 60, 240, 1440]
            },
            "commodities": {
                "description": "Commodity markets",
                "examples": ["GOLD", "OIL", "SILVER"],
                "intervals": [5, 15, 30, 60, 240, 1440]
            }
        }
    }

@router.get("/history/providers")
async def get_supported_providers_info():
    """Get information about supported data providers"""
    return {
        "supported_providers": get_supported_providers(),
        "provider_details": {
            "auto": {
                "description": "Automatically select best provider",
                "markets": ["crypto", "stocks", "forex", "commodities"],
                "rate_limit": "High"
            },
            "sample": {
                "description": "Sample data generator for testing",
                "markets": ["crypto", "stocks", "forex", "commodities"],
                "rate_limit": "Unlimited"
            },
            "binance": {
                "description": "Binance cryptocurrency exchange",
                "markets": ["crypto"],
                "rate_limit": "1200 requests/minute"
            },
            "alphavantage": {
                "description": "Alpha Vantage financial data",
                "markets": ["stocks", "forex", "commodities"],
                "rate_limit": "5 requests/minute (free tier)"
            },
            "yahoo": {
                "description": "Yahoo Finance",
                "markets": ["stocks", "forex"],
                "rate_limit": "2000 requests/hour"
            },
            "polygon": {
                "description": "Polygon.io market data",
                "markets": ["stocks", "forex", "commodities"],
                "rate_limit": "5 requests/minute (free tier)"
            }
        }
    }

@router.get("/history/health")
async def history_health_check():
    """Health check for the history service"""
    try:
        # Test sample data generation
        test_data = generate_sample_historical_data("TEST", "crypto", 1, 1, "sample")
        
        return {
            "status": "healthy",
            "service": "history-api",
            "supported_markets": len(get_supported_markets()),
            "supported_providers": len(get_supported_providers()),
            "test_data_generation": "passed" if test_data else "failed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"History health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "history-api",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }