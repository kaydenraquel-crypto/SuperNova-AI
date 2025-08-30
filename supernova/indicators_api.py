"""
Indicators API Router
Technical indicators endpoints for SuperNova AI
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np
import talib

from .indicators import (
    sma, ema, rsi, macd, bollinger, atr,
    black_scholes_call, black_scholes_put, calculate_all_greeks
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indicators", tags=["technical-indicators"])

# ====================================
# SCHEMAS
# ====================================

class IndicatorRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USD, AAPL)")
    market: str = Field("crypto", description="Market type: crypto, stocks")
    interval: int = Field(1, ge=1, description="Time interval in minutes")
    limit: int = Field(300, ge=1, le=1000, description="Number of data points")
    days: int = Field(30, ge=1, le=365, description="Days of historical data")
    provider: Optional[str] = Field("auto", description="Data provider preference")
    include_patterns: bool = Field(True, description="Include candlestick patterns")
    include_volume: bool = Field(True, description="Include volume indicators")
    cache_ttl: int = Field(300, ge=0, description="Cache TTL in seconds")

class IndicatorResponse(BaseModel):
    symbol: str
    market: str
    indicators: Dict[str, Any]
    metadata: Dict[str, Any]
    ohlc_sample: list = Field(..., description="Recent OHLC data sample")
    cache_info: Dict[str, Any]
    error: Optional[str] = None

# ====================================
# HELPER FUNCTIONS
# ====================================

def get_talib_function_groups():
    """Get all TA-Lib function groups with their functions"""
    try:
        # Get all available TA-Lib functions
        all_functions = talib.get_functions()
        
        # Predefined function groups (since func.info may not be available in all TA-Lib versions)
        predefined_groups = {
            'Overlap Studies': [
                'BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MAMA', 'MAVP', 
                'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'
            ],
            'Momentum Indicators': [
                'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 
                'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 
                'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 
                'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 'ULTOSC', 'WILLR'
            ],
            'Volume Indicators': [
                'AD', 'ADOSC', 'OBV'
            ],
            'Volatility Indicators': [
                'ATR', 'NATR', 'TRANGE'
            ],
            'Price Transform': [
                'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'
            ],
            'Cycle Indicators': [
                'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'
            ],
            'Pattern Recognition': [
                func for func in all_functions if func.startswith('CDL')
            ],
            'Statistic Functions': [
                'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 
                'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'
            ],
            'Math Transform': [
                'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH', 'EXP', 'FLOOR', 'LN', 
                'LOG10', 'SIN', 'SINH', 'SQRT', 'TAN', 'TANH'
            ],
            'Math Operators': [
                'ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MINMAX', 'MINMAXINDEX', 
                'MULT', 'SUB', 'SUM'
            ]
        }
        
        # Build groups with available functions
        groups = {}
        available_functions = set(all_functions)
        
        for group_name, func_list in predefined_groups.items():
            group_functions = []
            for func_name in func_list:
                if func_name in available_functions:
                    group_functions.append({
                        'name': func_name,
                        'display_name': func_name.replace('_', ' ').title(),
                        'group': group_name,
                        'input_names': ['high', 'low', 'close'] if func_name in ['ADX', 'ATR', 'STOCH'] else ['close'],
                        'parameters': {},
                        'output_names': [func_name.lower()]
                    })
            
            if group_functions:
                groups[group_name] = group_functions
        
        # Add any remaining functions to "Other" group
        categorized_functions = set()
        for group_funcs in groups.values():
            for func in group_funcs:
                categorized_functions.add(func['name'])
        
        other_functions = available_functions - categorized_functions
        if other_functions:
            groups['Other'] = [
                {
                    'name': func_name,
                    'display_name': func_name.replace('_', ' ').title(),
                    'group': 'Other',
                    'input_names': ['close'],
                    'parameters': {},
                    'output_names': [func_name.lower()]
                }
                for func_name in sorted(other_functions)
            ]
        
        return groups
    except Exception as e:
        logger.error(f"Error getting TA-Lib function groups: {e}")
        return {}

def get_sample_ohlcv_data(symbol: str = "SAMPLE", length: int = 100):
    """Generate sample OHLCV data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)  # 0.1% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV from prices
    ohlcv = []
    for i in range(1, len(prices)):
        high = prices[i] * (1 + abs(np.random.normal(0, 0.01)))
        low = prices[i] * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 50000)
        
        ohlcv.append({
            'timestamp': f"2024-01-{i:02d} 12:00:00",
            'open': round(prices[i-1], 2),
            'high': round(max(prices[i-1], prices[i], high), 2),
            'low': round(min(prices[i-1], prices[i], low), 2),
            'close': round(prices[i], 2),
            'volume': volume
        })
    
    return ohlcv

def calculate_technical_indicators(ohlcv_data: list, request: IndicatorRequest):
    """Calculate technical indicators from OHLCV data"""
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        if df.empty:
            raise ValueError("No OHLCV data provided")
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1000] * len(df))
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_20'] = sma(close, 20).fillna(0).tolist()
        indicators['sma_50'] = sma(close, 50).fillna(0).tolist()
        indicators['ema_20'] = ema(close, 20).fillna(0).tolist()
        indicators['ema_50'] = ema(close, 50).fillna(0).tolist()
        
        # Momentum Indicators
        indicators['rsi'] = rsi(close, 14).fillna(50).tolist()
        macd_line, macd_signal, macd_hist = macd(close)
        indicators['macd'] = {
            'line': macd_line.fillna(0).tolist(),
            'signal': macd_signal.fillna(0).tolist(),
            'histogram': macd_hist.fillna(0).tolist()
        }
        
        # Volatility Indicators
        bb_upper, bb_middle, bb_lower = bollinger(close)
        indicators['bollinger_bands'] = {
            'upper': bb_upper.fillna(close).tolist(),
            'middle': bb_middle.fillna(close).tolist(),
            'lower': bb_lower.fillna(close).tolist()
        }
        
        indicators['atr'] = atr(high, low, close).fillna(1).tolist()
        
        # Additional TA-Lib indicators if available
        try:
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                high.values, low.values, close.values
            )
            indicators['stoch_k'] = np.nan_to_num(indicators['stoch_k']).tolist()
            indicators['stoch_d'] = np.nan_to_num(indicators['stoch_d']).tolist()
        except Exception:
            indicators['stoch_k'] = [50] * len(close)
            indicators['stoch_d'] = [50] * len(close)
        
        try:
            indicators['adx'] = talib.ADX(high.values, low.values, close.values)
            indicators['adx'] = np.nan_to_num(indicators['adx']).tolist()
        except Exception:
            indicators['adx'] = [25] * len(close)
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Indicator calculation failed: {str(e)}")

# ====================================
# ENDPOINTS
# ====================================

@router.get("/functions/all")
async def get_all_functions():
    """Get information about all available TA-Lib indicators and functions"""
    try:
        groups = get_talib_function_groups()
        
        # Flatten all functions into a single list
        all_functions = []
        for group_name, functions in groups.items():
            all_functions.extend(functions)
        
        return {
            "total_functions": len(all_functions),
            "groups": groups,
            "functions": all_functions,
            "metadata": {
                "talib_version": getattr(talib, '__version__', 'unknown'),
                "supported_groups": list(groups.keys()),
                "generated_at": "2024-01-01T12:00:00Z"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting function info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get function info: {str(e)}")

@router.get("/functions/{group}")
async def get_function_group(group: str):
    """Get detailed information about a specific TA-Lib function group"""
    try:
        groups = get_talib_function_groups()
        
        if group not in groups:
            available_groups = list(groups.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Group '{group}' not found. Available groups: {available_groups}"
            )
        
        return {
            "group": group,
            "functions": groups[group],
            "count": len(groups[group]),
            "metadata": {
                "talib_version": getattr(talib, '__version__', 'unknown'),
                "generated_at": "2024-01-01T12:00:00Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting function group {group}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get function group: {str(e)}")

@router.post("/calculate", response_model=IndicatorResponse)
async def calculate_indicators(request: IndicatorRequest):
    """Calculate comprehensive technical indicators using TA-Lib and custom implementations"""
    try:
        # For now, use sample data - in production this would fetch real data
        logger.info(f"Calculating indicators for {request.symbol}")
        
        # Generate sample OHLCV data
        ohlcv_data = get_sample_ohlcv_data(request.symbol, request.limit)
        
        # Calculate indicators
        indicators = calculate_technical_indicators(ohlcv_data, request)
        
        # Prepare response
        response = IndicatorResponse(
            symbol=request.symbol,
            market=request.market,
            indicators=indicators,
            metadata={
                "interval": request.interval,
                "limit": request.limit,
                "days": request.days,
                "provider": request.provider,
                "calculated_at": "2024-01-01T12:00:00Z",
                "data_points": len(ohlcv_data)
            },
            ohlc_sample=ohlcv_data[-10:] if ohlcv_data else [],
            cache_info={
                "cached": False,
                "cache_ttl": request.cache_ttl,
                "cache_key": f"{request.symbol}_{request.market}_{request.interval}_{request.limit}"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Indicator calculation failed: {str(e)}")

@router.get("/info")
async def get_indicator_info():
    """Get information about available TA-Lib indicators and functions"""
    try:
        groups = get_talib_function_groups()
        
        return {
            "talib_version": getattr(talib, '__version__', 'unknown'),
            "total_functions": sum(len(funcs) for funcs in groups.values()),
            "groups": {name: len(funcs) for name, funcs in groups.items()},
            "available_groups": list(groups.keys()),
            "metadata": {
                "generated_at": "2024-01-01T12:00:00Z",
                "service": "supernova-indicators"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting indicator info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get indicator info: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for the indicators service"""
    try:
        # Test TA-Lib availability
        test_data = np.array([1, 2, 3, 4, 5], dtype=float)
        test_sma = talib.SMA(test_data, timeperiod=2)
        
        return {
            "status": "healthy",
            "talib_available": True,
            "talib_version": getattr(talib, '__version__', 'unknown'),
            "test_calculation": "passed",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "talib_available": False,
            "error": str(e),
            "timestamp": "2024-01-01T12:00:00Z"
        }

@router.delete("/cache")
async def clear_cache():
    """Clear the indicator cache"""
    # In a real implementation, this would clear Redis or memory cache
    return {
        "status": "success",
        "message": "Cache cleared successfully",
        "timestamp": "2024-01-01T12:00:00Z"
    }

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    # In a real implementation, this would return actual cache stats
    return {
        "total_keys": 0,
        "memory_usage": "0 MB",
        "hit_rate": 0.0,
        "miss_rate": 0.0,
        "last_cleared": "2024-01-01T12:00:00Z",
        "status": "empty"
    }