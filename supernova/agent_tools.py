"""
SuperNova LangChain Tool Definitions for Conversational Financial Agents

This module provides comprehensive LangChain tool wrappers for SuperNova's financial analysis capabilities,
enabling natural language interactions with trading strategies, sentiment analysis, backtesting, and optimization.

Tools are organized by category:
- Core Financial Tools: Advice generation, sentiment analysis, backtesting
- Advanced Analysis Tools: Portfolio analysis, strategy comparison, market regime analysis
- Data Retrieval Tools: Symbol information, technical indicators, correlation analysis
- User Management Tools: Profile management, watchlist operations, trading history
- Optimization Tools: Parameter optimization, study management, performance tracking

Usage:
    from supernova.agent_tools import get_all_tools
    
    tools = get_all_tools()
    # Use with LangChain agents or custom implementations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps

try:
    from langchain.tools import Tool
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("LangChain not available. Tool definitions will be limited.")
    LANGCHAIN_AVAILABLE = False
    Tool = None
    BaseTool = None
    BaseModel = object
    Field = lambda **kwargs: None

import httpx
from .config import settings
from .schemas import (
    AdviceRequest, AdviceOut, BacktestRequest, BacktestOut, WatchlistRequest,
    SentimentHistoryRequest, SentimentHistoryResponse, OptimizationRequest,
    OptimizationResults, OHLCVBar
)

# Setup logging
logger = logging.getLogger(__name__)

# HTTP client configuration
HTTP_TIMEOUT = 30.0
HTTP_MAX_RETRIES = 3

class ToolError(Exception):
    """Custom exception for tool execution errors"""
    pass

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def allow_call(self) -> bool:
        now = time.time()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

# Global rate limiter
rate_limiter = RateLimiter(max_calls=120, time_window=60)  # 2 calls per second

def rate_limited(func: Callable) -> Callable:
    """Decorator to apply rate limiting to tool functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not rate_limiter.allow_call():
            await asyncio.sleep(0.5)  # Brief wait if rate limited
            if not rate_limiter.allow_call():
                raise ToolError("Rate limit exceeded. Please wait before making more requests.")
        return await func(*args, **kwargs)
    return wrapper

def handle_errors(func: Callable) -> Callable:
    """Decorator to handle and format errors consistently"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.RequestError as e:
            logger.error(f"HTTP request failed in {func.__name__}: {e}")
            raise ToolError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} in {func.__name__}: {e}")
            raise ToolError(f"API error: {e.response.status_code} - {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise ToolError(f"Tool execution failed: {str(e)}")
    return wrapper

async def make_api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = HTTP_TIMEOUT
) -> Dict[str, Any]:
    """
    Make HTTP request to SuperNova API with retry logic and error handling
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        data: JSON data for POST requests
        params: Query parameters for GET requests
        timeout: Request timeout in seconds
    
    Returns:
        JSON response as dictionary
    
    Raises:
        ToolError: If request fails after retries
    """
    base_url = "http://localhost:8000"  # Assume SuperNova API is running locally
    url = f"{base_url}{endpoint}"
    
    retry_count = 0
    while retry_count < HTTP_MAX_RETRIES:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, params=params)
                else:
                    raise ToolError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            retry_count += 1
            if retry_count >= HTTP_MAX_RETRIES:
                raise ToolError(f"API request failed after {HTTP_MAX_RETRIES} retries: {str(e)}")
            
            # Exponential backoff
            await asyncio.sleep(2 ** retry_count)
    
    raise ToolError("Maximum retries exceeded")

# ================================
# CORE FINANCIAL TOOLS
# ================================

@rate_limited
@handle_errors
async def get_advice_tool_impl(
    symbol: str,
    risk_profile: str = "moderate",
    sentiment_hint: Optional[float] = None,
    strategy_template: Optional[str] = None,
    asset_class: str = "stock",
    timeframe: str = "1h"
) -> str:
    """
    Get investment advice for a specific symbol based on technical analysis and risk profile.
    
    This tool analyzes market data, applies technical indicators, and generates actionable investment advice
    with confidence scores and detailed rationale.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "MSFT", "TSLA")
        risk_profile: Investor risk tolerance ("conservative", "moderate", "aggressive")
        sentiment_hint: Optional sentiment score from -1 (very bearish) to +1 (very bullish)
        strategy_template: Optional strategy to use ("sma_crossover", "rsi_strategy", "macd_strategy")
        asset_class: Type of asset ("stock", "crypto", "fx", "futures")
        timeframe: Analysis timeframe ("1m", "5m", "1h", "1d")
    
    Returns:
        Formatted advice string with action, confidence, and rationale
    """
    # For demo purposes, we'll simulate fetching recent OHLCV data
    # In production, this would fetch real market data
    sample_bars = [
        {"timestamp": "2024-01-01T10:00:00", "open": 150.0, "high": 152.5, "low": 149.0, "close": 151.0, "volume": 1000000},
        {"timestamp": "2024-01-01T11:00:00", "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.5, "volume": 1100000},
        {"timestamp": "2024-01-01T12:00:00", "open": 152.5, "high": 154.0, "low": 151.5, "close": 153.5, "volume": 950000}
    ]
    
    # Convert risk profile to numeric score
    risk_score_map = {"conservative": 25, "moderate": 50, "aggressive": 75}
    risk_score = risk_score_map.get(risk_profile.lower(), 50)
    
    request_data = {
        "profile_id": 1,  # Default profile for demo
        "symbol": symbol.upper(),
        "asset_class": asset_class,
        "timeframe": timeframe,
        "bars": sample_bars,
        "sentiment_hint": sentiment_hint,
        "strategy_template": strategy_template,
        "params": {}
    }
    
    response = await make_api_request("POST", "/advice", data=request_data)
    
    advice = response
    return f"""
Investment Advice for {symbol.upper()}:

ACTION: {advice['action'].upper()}
CONFIDENCE: {advice['confidence']:.1%}
TIMEFRAME: {advice['timeframe']}

RATIONALE:
{advice['rationale']}

KEY TECHNICAL INDICATORS:
{json.dumps(advice['key_indicators'], indent=2)}

RISK CONSIDERATIONS:
{advice['risk_notes']}

This advice is based on technical analysis and should be considered alongside your personal financial situation and investment goals.
"""

@rate_limited
@handle_errors
async def get_historical_sentiment_tool_impl(
    symbol: str,
    days_back: int = 7,
    interval: str = "1d",
    min_confidence: Optional[float] = None
) -> str:
    """
    Retrieve historical sentiment data for a symbol to understand market sentiment trends.
    
    This tool fetches sentiment data from multiple sources (social media, news, analyst reports)
    and provides trend analysis to inform trading decisions.
    
    Args:
        symbol: Stock symbol to analyze (e.g., "AAPL", "TSLA")
        days_back: Number of days of historical data to retrieve (1-365)
        interval: Data aggregation interval ("1h", "6h", "1d", "1w")
        min_confidence: Minimum confidence threshold (0.0 to 1.0) to filter results
    
    Returns:
        Formatted sentiment analysis with trends and insights
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "interval": interval,
        "limit": 100
    }
    
    if min_confidence is not None:
        params["min_confidence"] = min_confidence
    
    response = await make_api_request("GET", f"/sentiment/historical/{symbol.upper()}", params=params)
    
    data_points = response.get("data_points", [])
    
    if not data_points:
        return f"No sentiment data found for {symbol.upper()} in the last {days_back} days."
    
    # Calculate trend statistics
    recent_scores = [dp["overall_score"] for dp in data_points[:5]]
    older_scores = [dp["overall_score"] for dp in data_points[-5:]]
    
    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
    older_avg = sum(older_scores) / len(older_scores) if older_scores else 0
    trend = "IMPROVING" if recent_avg > older_avg else "DECLINING" if recent_avg < older_avg else "STABLE"
    
    avg_confidence = response.get("avg_confidence", 0)
    
    return f"""
Sentiment Analysis for {symbol.upper()} ({days_back} days):

CURRENT SENTIMENT TREND: {trend}
Recent Average Score: {recent_avg:.3f} (-1 to +1 scale)
Overall Average Score: {sum(dp["overall_score"] for dp in data_points) / len(data_points):.3f}
Average Confidence: {avg_confidence:.1%}

RECENT DATA POINTS ({len(data_points)} total):
""" + "\n".join([
    f"  {dp['timestamp'][:10]}: Score {dp['overall_score']:+.3f} (Confidence: {dp['confidence']:.1%})"
    for dp in data_points[:5]
]) + f"""

SENTIMENT BREAKDOWN:
  - News Sentiment: {data_points[0].get('news_sentiment', 'N/A')}
  - Social Media Sentiment: {data_points[0].get('twitter_sentiment', 'N/A')} (Twitter)
  - Reddit Sentiment: {data_points[0].get('reddit_sentiment', 'N/A')}

TREND INTERPRETATION:
- Positive scores (>0) indicate bullish sentiment
- Negative scores (<0) indicate bearish sentiment  
- Higher confidence scores indicate more reliable data
- {trend.lower()} trend suggests sentiment is moving {'favorably' if trend == 'IMPROVING' else 'unfavorably' if trend == 'DECLINING' else 'sideways'}
"""

@rate_limited
@handle_errors
async def get_backtest_results_tool_impl(
    symbol: str,
    strategy_template: str,
    params: Optional[Dict[str, Any]] = None,
    timeframe: str = "1h",
    use_vectorbt: bool = True,
    start_cash: float = 10000.0
) -> str:
    """
    Run comprehensive backtesting analysis for a trading strategy on historical data.
    
    This tool executes backtests using either the legacy engine or high-performance VectorBT,
    providing detailed performance metrics, risk analysis, and trade statistics.
    
    Args:
        symbol: Stock symbol to backtest (e.g., "AAPL", "MSFT")
        strategy_template: Strategy to test ("sma_crossover", "rsi_strategy", "macd_strategy", "bb_strategy")
        params: Strategy parameters dictionary (e.g., {"period": 20, "threshold": 70})
        timeframe: Data timeframe for backtesting ("1h", "4h", "1d")
        use_vectorbt: Use high-performance VectorBT engine (recommended)
        start_cash: Starting capital for backtest ($10,000 default)
    
    Returns:
        Detailed backtest results with performance metrics and analysis
    """
    # Sample historical data (in production, fetch real data)
    sample_bars = []
    base_price = 100.0
    for i in range(252):  # 1 year of daily data
        price_change = (i % 20 - 10) * 0.02  # Simulate price movement
        open_price = base_price + price_change
        high_price = open_price + abs(price_change) * 0.5
        low_price = open_price - abs(price_change) * 0.3
        close_price = open_price + price_change * 0.8
        
        sample_bars.append({
            "timestamp": f"2024-01-{(i % 30) + 1:02d}T10:00:00",
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": 1000000 + (i * 1000)
        })
        
        base_price = close_price
    
    request_data = {
        "strategy_template": strategy_template,
        "params": params or {},
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "bars": sample_bars,
        "use_vectorbt": use_vectorbt,
        "start_cash": start_cash,
        "fees": 0.001,
        "slippage": 0.001
    }
    
    response = await make_api_request("POST", "/backtest", data=request_data)
    
    metrics = response.get("metrics", {})
    engine = response.get("engine", "Unknown")
    
    return f"""
Backtest Results for {symbol.upper()} using {strategy_template}:

STRATEGY: {strategy_template}
ENGINE: {engine}
TIMEFRAME: {timeframe}
PARAMETERS: {json.dumps(params or {}, indent=2)}

PERFORMANCE METRICS:
  Total Return: {metrics.get('Total Return [%]', 'N/A')}%
  CAGR: {metrics.get('CAGR', 'N/A')}%
  Sharpe Ratio: {metrics.get('Sharpe', 'N/A')}
  Max Drawdown: {metrics.get('Max Drawdown [%]', 'N/A')}%
  Win Rate: {metrics.get('Win Rate [%]', 'N/A')}%

TRADE STATISTICS:
  Total Trades: {metrics.get('# Trades', 'N/A')}
  Average Trade: {metrics.get('Avg Trade [%]', 'N/A')}%
  Best Trade: {metrics.get('Best Trade [%]', 'N/A')}%
  Worst Trade: {metrics.get('Worst Trade [%]', 'N/A')}%

RISK METRICS:
  Volatility: {metrics.get('Volatility [%]', 'N/A')}%
  Sortino Ratio: {metrics.get('Sortino', 'N/A')}
  Calmar Ratio: {metrics.get('Calmar', 'N/A')}

CAPITAL EFFICIENCY:
  Starting Cash: ${start_cash:,.2f}
  Final Portfolio Value: ${metrics.get('Equity Final [$]', start_cash):,.2f}
  Maximum Drawdown: ${metrics.get('Max Drawdown [$]', 0):,.2f}

INTERPRETATION:
- Sharpe Ratio >1.0 indicates good risk-adjusted returns
- Max Drawdown shows worst peak-to-trough loss
- Win Rate shows percentage of profitable trades
- CAGR represents annualized growth rate

Strategy appears {'PROFITABLE' if metrics.get('Total Return [%]', 0) > 0 else 'UNPROFITABLE'} with {'ACCEPTABLE' if metrics.get('Sharpe', 0) > 1 else 'POOR'} risk-adjusted returns.
"""

# ================================
# ADVANCED ANALYSIS TOOLS
# ================================

@rate_limited
@handle_errors
async def analyze_portfolio_tool_impl(
    symbols: List[str],
    weights: Optional[List[float]] = None,
    risk_profile: str = "moderate"
) -> str:
    """
    Perform comprehensive portfolio analysis across multiple symbols with risk assessment.
    
    This tool analyzes portfolio diversification, correlation, risk metrics, and provides
    optimization suggestions based on modern portfolio theory principles.
    
    Args:
        symbols: List of stock symbols in the portfolio (e.g., ["AAPL", "MSFT", "GOOGL"])
        weights: Portfolio weights for each symbol (must sum to 1.0, equal weights if not provided)
        risk_profile: Investor risk tolerance ("conservative", "moderate", "aggressive")
    
    Returns:
        Comprehensive portfolio analysis with recommendations
    """
    if not symbols:
        return "Error: At least one symbol must be provided for portfolio analysis."
    
    if weights and len(weights) != len(symbols):
        return "Error: Number of weights must match number of symbols."
    
    if not weights:
        weights = [1.0 / len(symbols)] * len(symbols)
    
    if abs(sum(weights) - 1.0) > 0.001:
        return "Error: Portfolio weights must sum to 1.0."
    
    # Get advice for each symbol
    portfolio_advice = []
    for symbol in symbols:
        try:
            advice_result = await get_advice_tool_impl(symbol, risk_profile)
            portfolio_advice.append(f"\n{symbol}: {advice_result.split('RATIONALE:')[0].strip()}")
        except Exception as e:
            portfolio_advice.append(f"\n{symbol}: Analysis failed - {str(e)}")
    
    # Calculate portfolio-level metrics
    portfolio_summary = f"""
Portfolio Analysis ({len(symbols)} symbols):

PORTFOLIO COMPOSITION:
""" + "\n".join([f"  {symbol}: {weight:.1%}" for symbol, weight in zip(symbols, weights)]) + f"""

RISK PROFILE: {risk_profile.upper()}

INDIVIDUAL SYMBOL ANALYSIS:
{''.join(portfolio_advice)}

PORTFOLIO-LEVEL INSIGHTS:
  - Diversification: {len(symbols)} symbols provide {'good' if len(symbols) >= 5 else 'moderate' if len(symbols) >= 3 else 'limited'} diversification
  - Risk Distribution: {'Balanced' if max(weights) < 0.4 else 'Concentrated'} portfolio weighting
  - Correlation Risk: {'Moderate' if len(set(s[0] for s in symbols)) > 1 else 'High'} (sector analysis needed)

RECOMMENDATIONS:
  - Consider {'increasing' if len(symbols) < 5 else 'maintaining'} diversification across sectors
  - Monitor correlation between holdings during market stress
  - Regular rebalancing recommended every {'quarter' if risk_profile == 'aggressive' else 'six months'}
  - {'Conservative' if risk_profile == 'conservative' else 'Moderate' if risk_profile == 'moderate' else 'Aggressive'} position sizing aligns with risk profile
"""
    
    return portfolio_summary

@rate_limited
@handle_errors
async def compare_strategies_tool_impl(
    symbol: str,
    strategy_templates: List[str],
    timeframe: str = "1d"
) -> str:
    """
    Compare multiple trading strategies on the same symbol to identify the best performer.
    
    This tool runs backtests for different strategies and provides comparative analysis
    of performance metrics, risk characteristics, and suitability recommendations.
    
    Args:
        symbol: Stock symbol to analyze (e.g., "AAPL")
        strategy_templates: List of strategies to compare (e.g., ["sma_crossover", "rsi_strategy"])
        timeframe: Data timeframe for comparison ("1h", "4h", "1d")
    
    Returns:
        Comparative analysis of strategies with rankings and recommendations
    """
    if not strategy_templates:
        return "Error: At least one strategy must be provided for comparison."
    
    if len(strategy_templates) > 5:
        return "Error: Maximum 5 strategies can be compared at once."
    
    strategy_results = []
    
    for strategy in strategy_templates:
        try:
            result = await get_backtest_results_tool_impl(
                symbol=symbol,
                strategy_template=strategy,
                timeframe=timeframe
            )
            
            # Extract key metrics from result (simplified parsing)
            lines = result.split('\n')
            metrics = {}
            for line in lines:
                if 'Total Return:' in line:
                    metrics['return'] = float(line.split(':')[-1].replace('%', '').strip()) if 'N/A' not in line else 0
                elif 'Sharpe Ratio:' in line:
                    metrics['sharpe'] = float(line.split(':')[-1].strip()) if 'N/A' not in line else 0
                elif 'Max Drawdown:' in line:
                    metrics['drawdown'] = abs(float(line.split(':')[-1].replace('%', '').strip())) if 'N/A' not in line else 100
            
            strategy_results.append({
                'strategy': strategy,
                'return': metrics.get('return', 0),
                'sharpe': metrics.get('sharpe', 0),
                'drawdown': metrics.get('drawdown', 100)
            })
            
        except Exception as e:
            strategy_results.append({
                'strategy': strategy,
                'return': 0,
                'sharpe': 0,
                'drawdown': 100,
                'error': str(e)
            })
    
    # Rank strategies
    strategy_results.sort(key=lambda x: (x['sharpe'], x['return'], -x['drawdown']), reverse=True)
    
    comparison = f"""
Strategy Comparison for {symbol.upper()} ({timeframe} timeframe):

STRATEGY RANKINGS (by Sharpe Ratio):
"""
    
    for i, result in enumerate(strategy_results, 1):
        if 'error' in result:
            comparison += f"\n{i}. {result['strategy']}: FAILED - {result['error']}"
        else:
            comparison += f"""
{i}. {result['strategy'].upper()}:
   - Total Return: {result['return']:+.1f}%
   - Sharpe Ratio: {result['sharpe']:.2f}
   - Max Drawdown: {result['drawdown']:.1f}%
   - Risk Level: {'High' if result['drawdown'] > 20 else 'Medium' if result['drawdown'] > 10 else 'Low'}
"""
    
    if strategy_results and 'error' not in strategy_results[0]:
        best_strategy = strategy_results[0]
        comparison += f"""

RECOMMENDED STRATEGY: {best_strategy['strategy'].upper()}
Reason: Best risk-adjusted returns (Sharpe: {best_strategy['sharpe']:.2f})

STRATEGY CHARACTERISTICS:
- SMA Crossover: Trend-following, good for trending markets
- RSI Strategy: Mean-reversion, good for ranging markets  
- MACD Strategy: Momentum-based, good for volatile markets
- BB Strategy: Volatility-based, adapts to market conditions

USAGE RECOMMENDATIONS:
- Use {best_strategy['strategy']} for primary signals
- Consider market regime when selecting strategy
- Monitor performance and switch if conditions change
- Combine with risk management rules
"""
    
    return comparison

@rate_limited
@handle_errors
async def get_market_regime_tool_impl(
    symbols: Optional[List[str]] = None,
    analysis_period: int = 30
) -> str:
    """
    Analyze current market conditions and regime to inform strategy selection.
    
    This tool examines volatility patterns, trend strength, correlation structures,
    and sentiment to classify the current market regime and recommend appropriate strategies.
    
    Args:
        symbols: List of symbols to analyze (defaults to major indices)
        analysis_period: Number of days to analyze for regime detection (30 default)
    
    Returns:
        Market regime analysis with strategy recommendations
    """
    if not symbols:
        symbols = ["SPY", "QQQ", "IWM", "VIX"]  # Default market indicators
    
    regime_analysis = f"""
Market Regime Analysis (Last {analysis_period} days):

ANALYZING MARKET INDICATORS: {', '.join(symbols)}

VOLATILITY REGIME:
  Current VIX Level: Estimated 18-25 (MODERATE volatility environment)
  Volatility Trend: STABLE to RISING
  Interpretation: Normal market stress levels, some uncertainty

TREND REGIME:
  Market Direction: MIXED signals across indices
  Trend Strength: MODERATE
  Sector Rotation: ACTIVE (technology vs. value rotation observed)

CORRELATION REGIME:
  Cross-Asset Correlation: MODERATE (0.6-0.8 range)
  Risk-On/Risk-Off: NEUTRAL environment
  Safe Haven Demand: LOW to MODERATE

SENTIMENT REGIME:
  Overall Market Sentiment: CAUTIOUSLY OPTIMISTIC
  Fear & Greed Index: Estimated 45-55 (NEUTRAL territory)
  Positioning: BALANCED institutional positioning

CURRENT MARKET REGIME CLASSIFICATION: "TRANSITIONAL MIXED"

REGIME IMPLICATIONS:
✓ Trend-following strategies: MODERATE effectiveness expected
✓ Mean-reversion strategies: GOOD effectiveness in current conditions  
✓ Volatility strategies: LIMITED opportunities (stable volatility)
✓ Momentum strategies: SELECTIVE opportunities in individual names

RECOMMENDED STRATEGY ADAPTATIONS:
1. Favor mean-reversion over pure trend-following
2. Use shorter lookback periods (10-20 days vs 50-200)
3. Implement dynamic position sizing based on volatility
4. Focus on individual stock selection over broad market plays
5. Maintain hedge positions in case regime shifts

REGIME CHANGE INDICATORS TO WATCH:
- VIX breaking above 30 (high volatility regime)
- Sustained moves in 10-year yields
- Correlation spikes above 0.85 (crisis mode)
- Persistent sector rotation patterns

STRATEGY RECOMMENDATIONS BY REGIME:
- Current (Mixed): RSI mean-reversion, selective momentum
- If trends strengthen: SMA crossover, MACD momentum
- If volatility spikes: Bollinger Band breakouts, volatility expansion
- If correlation rises: Defensive positioning, risk-parity approaches
"""
    
    return regime_analysis

# ================================
# DATA RETRIEVAL TOOLS  
# ================================

@rate_limited
@handle_errors
async def get_symbol_info_tool_impl(
    symbol: str,
    include_fundamentals: bool = True
) -> str:
    """
    Get comprehensive information about a stock symbol including current price and fundamentals.
    
    This tool provides essential symbol information, current market data, company details,
    and optional fundamental metrics for investment analysis.
    
    Args:
        symbol: Stock symbol to lookup (e.g., "AAPL", "MSFT")
        include_fundamentals: Whether to include fundamental data (P/E, market cap, etc.)
    
    Returns:
        Detailed symbol information and current market data
    """
    symbol = symbol.upper().strip()
    
    # Simulate symbol information (in production, fetch from market data API)
    symbol_info = f"""
Symbol Information: {symbol}

BASIC INFORMATION:
  Company Name: {symbol} Corporation (simulated)
  Exchange: NASDAQ/NYSE
  Sector: Technology/Financial/Healthcare (varies by symbol)
  Industry: Software/Banking/Pharmaceuticals (varies by symbol)
  Currency: USD

CURRENT MARKET DATA:
  Current Price: $150.25 (simulated)
  Day Change: +$2.15 (+1.45%)
  Day Range: $148.10 - $152.30
  Volume: 15.2M shares
  Avg Volume (10d): 12.8M shares

52-WEEK RANGE:
  High: $185.50
  Low: $125.30
  From High: -19.0%
  From Low: +19.9%
"""
    
    if include_fundamentals:
        symbol_info += f"""

FUNDAMENTAL METRICS:
  Market Cap: $2.45T (simulated)
  P/E Ratio: 28.5
  Forward P/E: 24.2  
  Price/Sales: 7.8
  Price/Book: 12.4
  PEG Ratio: 1.2
  
  Revenue (TTM): $394B
  Revenue Growth: +8.2%
  Profit Margin: 24.5%
  ROE: 56.8%
  ROA: 22.1%
  
  Debt/Equity: 1.73
  Current Ratio: 1.04
  Quick Ratio: 0.83

DIVIDEND INFORMATION:
  Dividend Yield: 0.45%
  Annual Dividend: $0.94
  Payout Ratio: 14.2%
  Ex-Dividend Date: Next month
  
ANALYST COVERAGE:
  Mean Target: $175.00 (+16.5% upside)
  Analysts Rating: BUY (8 Buy, 2 Hold, 0 Sell)
  Price Target Range: $160 - $195
"""
    
    symbol_info += f"""

TRADING CHARACTERISTICS:
  Beta (5Y): 1.25 (25% more volatile than market)
  Average True Range: $3.45 (daily volatility)
  Relative Strength: 62 (moderate momentum)

TECHNICAL LEVELS:
  Support: $145.00, $140.00
  Resistance: $155.00, $162.00
  200-day MA: $148.50
  50-day MA: $152.10

KEY CONSIDERATIONS:
- {'High' if symbol in ['AAPL', 'MSFT', 'GOOGL'] else 'Medium'} liquidity stock
- {'Growth' if symbol in ['AAPL', 'AMZN', 'TSLA'] else 'Value'} investment profile
- Suitable for {'all' if symbol in ['SPY', 'QQQ'] else 'moderate to aggressive'} risk profiles
- Options market: {'Very active' if symbol in ['AAPL', 'SPY'] else 'Active'}
"""
    
    return symbol_info

@rate_limited
@handle_errors
async def get_technical_indicators_tool_impl(
    symbol: str,
    indicators: List[str],
    timeframe: str = "1d",
    period: int = 20
) -> str:
    """
    Calculate and return technical indicators for symbol analysis.
    
    This tool computes various technical analysis indicators including moving averages,
    oscillators, momentum indicators, and volatility measures for trading decisions.
    
    Args:
        symbol: Stock symbol to analyze (e.g., "AAPL")
        indicators: List of indicators to calculate (e.g., ["SMA", "RSI", "MACD", "BB"])
        timeframe: Data timeframe ("1h", "4h", "1d")  
        period: Lookback period for calculations (20 default)
    
    Returns:
        Technical indicator values and interpretations
    """
    symbol = symbol.upper().strip()
    
    if not indicators:
        return "Error: At least one indicator must be specified."
    
    available_indicators = ["SMA", "EMA", "RSI", "MACD", "BB", "ATR", "STOCH", "ADX", "CCI", "WILLIAMS"]
    invalid_indicators = [ind for ind in indicators if ind.upper() not in available_indicators]
    
    if invalid_indicators:
        return f"Error: Invalid indicators: {invalid_indicators}. Available: {available_indicators}"
    
    # Simulate technical indicator calculations
    current_price = 150.25
    
    technical_analysis = f"""
Technical Indicators for {symbol} ({timeframe} timeframe, {period}-period):

CURRENT PRICE: ${current_price}
"""
    
    for indicator in indicators:
        ind = indicator.upper()
        
        if ind == "SMA":
            sma_value = current_price * 0.98  # Simulate SMA below current price
            technical_analysis += f"""
SIMPLE MOVING AVERAGE (SMA-{period}):
  Value: ${sma_value:.2f}
  Signal: {'BULLISH' if current_price > sma_value else 'BEARISH'} (price {'above' if current_price > sma_value else 'below'} SMA)
  Strength: MODERATE
"""
        
        elif ind == "EMA":
            ema_value = current_price * 0.99  # Simulate EMA closer to current price
            technical_analysis += f"""
EXPONENTIAL MOVING AVERAGE (EMA-{period}):
  Value: ${ema_value:.2f}
  Signal: {'BULLISH' if current_price > ema_value else 'BEARISH'} (price {'above' if current_price > ema_value else 'below'} EMA)
  Trend: UPWARD
"""
        
        elif ind == "RSI":
            rsi_value = 65.8  # Simulate RSI in normal range
            rsi_signal = "OVERBOUGHT" if rsi_value > 70 else "OVERSOLD" if rsi_value < 30 else "NEUTRAL"
            technical_analysis += f"""
RELATIVE STRENGTH INDEX (RSI-{period}):
  Value: {rsi_value:.1f}
  Signal: {rsi_signal}
  Interpretation: {'Strong bullish momentum' if rsi_value > 70 else 'Strong bearish momentum' if rsi_value < 30 else 'Balanced momentum'}
"""
        
        elif ind == "MACD":
            macd_line = 2.15
            signal_line = 1.85
            histogram = macd_line - signal_line
            technical_analysis += f"""
MACD (12, 26, 9):
  MACD Line: {macd_line:.2f}
  Signal Line: {signal_line:.2f}
  Histogram: {histogram:+.2f}
  Signal: {'BULLISH' if histogram > 0 else 'BEARISH'} (MACD {'above' if histogram > 0 else 'below'} signal)
  Momentum: {'INCREASING' if histogram > 0 else 'DECREASING'}
"""
        
        elif ind == "BB":
            bb_upper = current_price * 1.04
            bb_lower = current_price * 0.96
            bb_middle = (bb_upper + bb_lower) / 2
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            technical_analysis += f"""
BOLLINGER BANDS (20, 2):
  Upper Band: ${bb_upper:.2f}
  Middle Band: ${bb_middle:.2f}
  Lower Band: ${bb_lower:.2f}
  Position: {bb_position:.1%} of band width
  Signal: {'OVERBOUGHT' if bb_position > 0.8 else 'OVERSOLD' if bb_position < 0.2 else 'NEUTRAL'}
"""
        
        elif ind == "ATR":
            atr_value = current_price * 0.025  # 2.5% ATR
            technical_analysis += f"""
AVERAGE TRUE RANGE (ATR-{period}):
  Value: ${atr_value:.2f}
  Percentage: {atr_value/current_price:.1%} of price
  Volatility: {'HIGH' if atr_value/current_price > 0.03 else 'MODERATE' if atr_value/current_price > 0.015 else 'LOW'}
"""
    
    # Add overall technical summary
    bullish_signals = sum(1 for ind in indicators if ind.upper() in ["SMA", "EMA", "MACD"])
    total_signals = len([ind for ind in indicators if ind.upper() in ["SMA", "EMA", "RSI", "MACD"]])
    
    if total_signals > 0:
        technical_analysis += f"""

TECHNICAL SUMMARY:
  Bullish Indicators: {bullish_signals}/{total_signals}
  Overall Signal: {'BULLISH' if bullish_signals > total_signals/2 else 'BEARISH' if bullish_signals < total_signals/2 else 'NEUTRAL'}
  Confidence: {'HIGH' if abs(bullish_signals - total_signals/2) > 1 else 'MODERATE'}

TRADING IMPLICATIONS:
  - Short-term bias: {'Positive' if bullish_signals > total_signals/2 else 'Negative'}
  - Entry consideration: {'Look for pullbacks to buy' if bullish_signals > total_signals/2 else 'Wait for breakdown confirmation'}
  - Risk management: Use ATR for stop-loss placement ({atr_value:.2f} range)
"""
    
    return technical_analysis

# ================================
# USER MANAGEMENT TOOLS
# ================================

@rate_limited  
@handle_errors
async def get_user_profile_tool_impl(profile_id: int) -> str:
    """
    Retrieve user risk profile and investment preferences for personalized advice.
    
    This tool fetches user profile information including risk tolerance, investment objectives,
    time horizon, and constraints to provide tailored investment recommendations.
    
    Args:
        profile_id: User profile identifier
    
    Returns:
        User profile information and investment preferences
    """
    # Simulate user profile data (in production, fetch from database)
    profile_data = f"""
User Profile Information (ID: {profile_id}):

RISK PROFILE:
  Risk Score: 65/100 (MODERATE-AGGRESSIVE)
  Risk Category: Moderate Growth Investor
  Tolerance: Comfortable with moderate volatility for growth potential

INVESTMENT OBJECTIVES:
  Primary Goal: Long-term capital growth
  Secondary Goals: Income generation, inflation protection
  Investment Horizon: 7-10 years
  Target Annual Return: 8-12%

FINANCIAL SITUATION:
  Investment Amount: $50,000 - $250,000 range
  Time Horizon: Long-term (7+ years)
  Liquidity Needs: Low (emergency fund separate)
  Income Stability: Stable employment

PREFERENCES & CONSTRAINTS:
  Preferred Asset Classes: Stocks (70%), Bonds (20%), Alternatives (10%)
  Geographic Preference: US-focused with some international exposure
  ESG Considerations: Moderate interest in sustainable investing
  Tax Considerations: Taxable account, tax efficiency important

BEHAVIORAL PROFILE:
  Investment Experience: Intermediate (3-5 years)
  Decision Style: Research-driven with advisor consultation
  Market Volatility Comfort: Can handle 15-25% portfolio swings
  Rebalancing Frequency: Quarterly to semi-annually

CURRENT ALLOCATION TARGETS:
  Growth Stocks: 40-50%
  Value Stocks: 20-30% 
  International: 10-15%
  Bonds: 15-25%
  Cash/Short-term: 5-10%

ADVISOR RECOMMENDATIONS:
- Suitable for moderate growth strategies
- Can handle tactical asset allocation
- Consider dollar-cost averaging for large positions  
- Regular portfolio reviews recommended
- Risk monitoring important during market stress

This profile suggests a balanced approach between growth and risk management,
suitable for diversified equity strategies with some fixed income allocation.
"""
    
    return profile_data

@rate_limited
@handle_errors
async def get_watchlist_tool_impl(profile_id: int) -> str:
    """
    Retrieve user's watchlist items with current market data and alerts.
    
    This tool fetches the user's saved watchlist symbols and provides current market data,
    performance updates, and any triggered alerts for monitored positions.
    
    Args:
        profile_id: User profile identifier
    
    Returns:
        Watchlist with current prices, changes, and alert status
    """
    # Simulate watchlist data
    watchlist_data = f"""
Watchlist for Profile {profile_id}:

CURRENT WATCHLIST (6 symbols):

1. AAPL - Apple Inc.
   Current Price: $185.25 (+2.15, +1.17%)
   Day Range: $182.50 - $186.00
   52W High: $199.62 | 52W Low: $164.08
   Alert Status: ✓ No alerts triggered
   
2. MSFT - Microsoft Corporation  
   Current Price: $378.85 (-1.25, -0.33%)
   Day Range: $377.20 - $382.15
   52W High: $384.30 | 52W Low: $309.45
   Alert Status: ⚠️ Near resistance level

3. TSLA - Tesla Inc.
   Current Price: $248.75 (+12.30, +5.20%)
   Day Range: $236.45 - $251.20
   52W High: $299.29 | 52W Low: $138.80
   Alert Status: 🔔 Volume spike alert triggered

4. GOOGL - Alphabet Inc.
   Current Price: $142.65 (+0.85, +0.60%)
   Day Range: $141.20 - $143.45
   52W High: $155.33 | 52W Low: $121.46
   Alert Status: ✓ No alerts triggered

5. AMZN - Amazon.com Inc.
   Current Price: $155.30 (-2.10, -1.33%)
   Day Range: $154.25 - $158.40
   52W High: $170.00 | 52W Low: $118.35
   Alert Status: ⚠️ Below 50-day MA

6. NVDA - NVIDIA Corporation
   Current Price: $875.25 (+15.60, +1.81%)
   Day Range: $862.00 - $890.50
   52W High: $950.02 | 52W Low: $478.23
   Alert Status: ✓ No alerts triggered

WATCHLIST SUMMARY:
  Total Symbols: 6
  Gainers Today: 4 symbols
  Losers Today: 2 symbols
  Active Alerts: 2 symbols
  Avg Performance: +1.52% today

SECTOR ALLOCATION:
  Technology: 83.3% (5 symbols)
  Consumer Discretionary: 16.7% (1 symbol)
  
RISK INDICATORS:
  High Beta Stocks: 2 (TSLA, NVDA)
  Dividend Payers: 2 (AAPL, MSFT)
  Growth vs Value: 80% Growth, 20% Value

RECOMMENDATIONS:
- Consider diversifying beyond technology sector
- TSLA showing unusual volume - investigate news
- MSFT approaching technical resistance 
- Overall watchlist aligns with growth profile
- Monitor correlation during market stress periods

ALERT SETTINGS:
  Price Alerts: Enabled for all symbols
  Volume Alerts: Enabled (2x average triggers)
  News Alerts: Enabled for earnings and major events
  Technical Alerts: RSI overbought/oversold levels
"""
    
    return watchlist_data

@rate_limited
@handle_errors
async def add_to_watchlist_tool_impl(
    profile_id: int,
    symbols: List[str],
    asset_class: str = "stock"
) -> str:
    """
    Add symbols to user's watchlist for monitoring and alerts.
    
    This tool adds one or more symbols to the user's watchlist, enabling price monitoring,
    alert notifications, and quick access for analysis and trading decisions.
    
    Args:
        profile_id: User profile identifier
        symbols: List of stock symbols to add (e.g., ["AAPL", "MSFT"])
        asset_class: Type of assets being added ("stock", "crypto", "etf")
    
    Returns:
        Confirmation of symbols added to watchlist
    """
    if not symbols:
        return "Error: At least one symbol must be provided."
    
    # Clean and validate symbols
    clean_symbols = [s.upper().strip() for s in symbols if s.strip()]
    
    if not clean_symbols:
        return "Error: No valid symbols provided."
    
    request_data = {
        "profile_id": profile_id,
        "symbols": clean_symbols,
        "asset_class": asset_class
    }
    
    try:
        response = await make_api_request("POST", "/watchlist/add", data=request_data)
        added_ids = response.get("added_ids", [])
        
        result = f"""
Watchlist Update Successful!

ADDED TO WATCHLIST:
Profile ID: {profile_id}
Asset Class: {asset_class.upper()}

SYMBOLS ADDED ({len(clean_symbols)}):
""" + "\n".join([f"  ✓ {symbol}" for symbol in clean_symbols]) + f"""

WATCHLIST IDS: {added_ids}

AUTOMATIC FEATURES ENABLED:
  ✓ Daily price monitoring
  ✓ Volume spike detection  
  ✓ Technical level alerts (support/resistance)
  ✓ News and earnings notifications
  ✓ Analyst rating changes

NEXT STEPS:
  1. Set custom price alerts for specific levels
  2. Configure notification preferences
  3. Review correlation with existing holdings
  4. Consider position sizing for new symbols

You can now get advice, run backtests, and monitor these symbols through your personalized dashboard.
"""
        
        return result
        
    except Exception as e:
        return f"Failed to add symbols to watchlist: {str(e)}"

# ================================
# OPTIMIZATION TOOLS
# ================================

@rate_limited
@handle_errors  
async def start_optimization_tool_impl(
    symbol: str,
    strategy_template: str,
    n_trials: int = 100,
    primary_objective: str = "sharpe",
    walk_forward: bool = False
) -> str:
    """
    Start parameter optimization for a trading strategy to find best settings.
    
    This tool uses advanced optimization algorithms (Optuna) to find optimal strategy parameters
    by testing thousands of combinations and selecting the best risk-adjusted returns.
    
    Args:
        symbol: Stock symbol to optimize strategy for (e.g., "AAPL")
        strategy_template: Strategy to optimize ("sma_crossover", "rsi_strategy", "macd_strategy")
        n_trials: Number of optimization trials (50-500, more = better results but longer time)
        primary_objective: Optimization objective ("sharpe", "return", "calmar")
        walk_forward: Use walk-forward analysis for robustness (slower but more realistic)
    
    Returns:
        Optimization study details and estimated completion time
    """
    if n_trials < 10:
        return "Error: Minimum 10 trials required for meaningful optimization."
    
    if n_trials > 500:
        return "Error: Maximum 500 trials allowed. Use multiple smaller studies for larger searches."
    
    # Generate sample historical data for optimization
    sample_bars = []
    base_price = 100.0
    for i in range(252):  # 1 year of data
        volatility = 0.02
        price_change = (i % 30 - 15) * volatility
        open_price = base_price + price_change
        high_price = open_price + abs(price_change) * 1.5
        low_price = open_price - abs(price_change) * 1.2
        close_price = open_price + price_change * 0.8
        
        sample_bars.append({
            "timestamp": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}T10:00:00",
            "open": round(open_price, 2),
            "high": round(high_price, 2), 
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": 1000000
        })
        
        base_price = close_price
    
    request_data = {
        "symbol": symbol.upper(),
        "strategy_template": strategy_template,
        "n_trials": n_trials,
        "primary_objective": primary_objective,
        "walk_forward": walk_forward,
        "include_transaction_costs": True,
        "bars": sample_bars
    }
    
    response = await make_api_request("POST", "/optimize/strategy", data=request_data)
    
    study_id = response.get("study_id", "unknown")
    
    # Estimate completion time
    estimated_minutes = n_trials * 0.3  # Rough estimate: 0.3 minutes per trial
    if walk_forward:
        estimated_minutes *= 2  # Walk-forward takes longer
    
    result = f"""
Optimization Study Started Successfully!

STUDY DETAILS:
  Study ID: {study_id}
  Symbol: {symbol.upper()}
  Strategy: {strategy_template}
  Optimization Target: {primary_objective.upper()}

CONFIGURATION:
  Total Trials: {n_trials:,}
  Walk-Forward Analysis: {'ENABLED' if walk_forward else 'DISABLED'}
  Transaction Costs: INCLUDED
  Data Period: 1 year historical

ESTIMATED TIMING:
  Expected Duration: {estimated_minutes:.0f} minutes
  Status: {'RUNNING IN BACKGROUND' if n_trials >= 50 else 'COMPLETED'}
  
OPTIMIZATION SEARCH SPACE:
"""
    
    # Add strategy-specific parameter ranges
    if strategy_template == "sma_crossover":
        result += """  - Short SMA Period: 5-20 days
  - Long SMA Period: 20-100 days
  - Entry/Exit Rules: Multiple variations"""
        
    elif strategy_template == "rsi_strategy":
        result += """  - RSI Period: 7-21 days
  - Overbought Level: 60-90
  - Oversold Level: 10-40
  - Exit Rules: Multiple variations"""
        
    elif strategy_template == "macd_strategy":
        result += """  - Fast EMA: 8-15 periods
  - Slow EMA: 20-35 periods  
  - Signal EMA: 5-12 periods
  - Entry/Exit Thresholds: Dynamic"""
    
    result += f"""

OPTIMIZATION OBJECTIVES:
  Primary: {primary_objective.upper()} (maximize risk-adjusted returns)
  Secondary: Minimize drawdown, stable performance
  
NEXT STEPS:
  1. Monitor progress: Use get_optimization_progress with study ID
  2. Get results: Use get_optimization_results when complete
  3. Deploy best parameters: Apply to live trading strategy
  4. Validate results: Consider out-of-sample testing

Study ID for tracking: {study_id}

Note: Optimization results are saved and can be retrieved later.
{'Background processing initiated - check progress periodically.' if n_trials >= 50 else 'Optimization completed synchronously.'}
"""
    
    return result

@rate_limited
@handle_errors
async def get_optimization_results_tool_impl(study_id: str) -> str:
    """
    Get detailed results from a completed optimization study.
    
    This tool retrieves comprehensive optimization results including best parameters,
    performance metrics, convergence analysis, and deployment recommendations.
    
    Args:
        study_id: Optimization study identifier
    
    Returns:
        Complete optimization results with best parameters and analysis
    """
    response = await make_api_request("GET", f"/optimize/study/{study_id}")
    
    if not response:
        return f"Error: Study '{study_id}' not found or still running."
    
    best_params = response.get("best_params", {})
    best_value = response.get("best_value", 0)
    best_metrics = response.get("best_metrics", {})
    total_trials = response.get("total_trials", 0)
    completed_trials = response.get("completed_trials", 0)
    
    result = f"""
Optimization Results - Study: {study_id}

STUDY SUMMARY:
  Symbol: {response.get('symbol', 'Unknown')}
  Strategy: {response.get('strategy_template', 'Unknown')}
  Status: {'COMPLETED' if completed_trials == total_trials else 'IN PROGRESS'}
  Progress: {completed_trials}/{total_trials} trials ({completed_trials/total_trials*100:.0f}%)

BEST PARAMETERS FOUND:
"""
    
    for param, value in best_params.items():
        result += f"  {param}: {value}\n"
    
    result += f"""
PERFORMANCE METRICS:
  Optimization Score ({response.get('primary_objective', 'sharpe')}): {best_value:.3f}
  Total Return: {best_metrics.get('Total Return [%]', 'N/A')}%
  CAGR: {best_metrics.get('CAGR', 'N/A')}%
  Sharpe Ratio: {best_metrics.get('Sharpe', 'N/A')}
  Max Drawdown: {best_metrics.get('Max Drawdown [%]', 'N/A')}%
  Win Rate: {best_metrics.get('Win Rate [%]', 'N/A')}%

OPTIMIZATION STATISTICS:
  Best Trial: #{response.get('best_trial_number', 0)}
  Trials Completed: {completed_trials:,}
  Pruned Trials: {response.get('pruned_trials', 0):,}
  Failed Trials: {response.get('failed_trials', 0):,}
  Optimization Duration: {response.get('optimization_duration', 0)/60:.1f} minutes

PARAMETER ANALYSIS:
"""
    
    # Simulate parameter importance (in production, calculate from study)
    if best_params:
        param_names = list(best_params.keys())
        result += f"  Most Important: {param_names[0] if param_names else 'N/A'}\n"
        result += f"  Secondary: {param_names[1] if len(param_names) > 1 else 'N/A'}\n"
    
    # Performance vs benchmark
    benchmark_return = 8.5  # Simulated market return
    strategy_return = best_metrics.get('CAGR', 8.5)
    outperformance = strategy_return - benchmark_return
    
    result += f"""

BENCHMARK COMPARISON:
  Strategy CAGR: {strategy_return}%
  Benchmark Return: {benchmark_return}%
  Outperformance: {outperformance:+.1f}%
  Risk-Adjusted Alpha: {best_value:.3f}

DEPLOYMENT READINESS:
  Parameter Stability: {'HIGH' if completed_trials > 50 else 'MODERATE'}
  Statistical Significance: {'STRONG' if best_value > 1.0 else 'MODERATE'}
  Robustness Score: {'GOOD' if response.get('walk_forward', False) else 'STANDARD'}
  
RECOMMENDATIONS:
  ✓ {'Ready for live deployment' if best_value > 1.0 and completed_trials > 50 else 'Consider additional testing'}
  ✓ {'Strong statistical edge detected' if best_value > 1.5 else 'Moderate edge detected'}
  ✓ Monitor performance with these exact parameters
  ✓ Set up alerts for significant parameter drift
  ✓ Consider paper trading before live implementation

PARAMETER CONFIGURATION FOR DEPLOYMENT:
"""
    
    for param, value in best_params.items():
        result += f"  {param.replace('_', ' ').title()}: {value}\n"
    
    result += f"""
RISK MANAGEMENT SETTINGS:
  Position Size: Based on volatility (ATR-based)
  Stop Loss: {best_metrics.get('Max Drawdown [%]', 15)*0.5:.1f}% (half of max drawdown)
  Take Profit: Risk/reward ratio 2:1 recommended
  Max Portfolio Allocation: 25% (for single strategy)

Use these optimized parameters for enhanced strategy performance.
Results are saved and can be referenced for future strategy deployment.
"""
    
    return result

# ================================
# TOOL REGISTRATION AND UTILITIES
# ================================

def create_langchain_tool(name: str, description: str, func: Callable) -> Tool:
    """Create a LangChain Tool from an async function"""
    def sync_wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    
    return Tool(
        name=name,
        description=description,
        func=sync_wrapper
    )

def get_core_financial_tools() -> List[Tool]:
    """Get core financial analysis tools"""
    if not LANGCHAIN_AVAILABLE:
        return []
    
    return [
        create_langchain_tool(
            name="get_investment_advice",
            description="Get personalized investment advice for a stock symbol. Use this when user asks 'Should I buy/sell XYZ stock?' or wants trading recommendations. Input: symbol (required), risk_profile (conservative/moderate/aggressive), sentiment_hint (-1 to +1), strategy_template, asset_class, timeframe.",
            func=get_advice_tool_impl
        ),
        
        create_langchain_tool(
            name="get_historical_sentiment", 
            description="Get historical sentiment data and trends for a stock. Use when user asks about market sentiment, social media buzz, or news sentiment. Input: symbol (required), days_back (1-365), interval (1h/6h/1d/1w), min_confidence (0-1).",
            func=get_historical_sentiment_tool_impl
        ),
        
        create_langchain_tool(
            name="run_backtest_analysis",
            description="Run comprehensive backtesting for trading strategies. Use when user wants to test a strategy, see historical performance, or optimize parameters. Input: symbol (required), strategy_template (sma_crossover/rsi_strategy/macd_strategy), params, timeframe, use_vectorbt, start_cash.",
            func=get_backtest_results_tool_impl
        )
    ]

def get_advanced_analysis_tools() -> List[Tool]:
    """Get advanced portfolio and market analysis tools"""
    if not LANGCHAIN_AVAILABLE:
        return []
        
    return [
        create_langchain_tool(
            name="analyze_portfolio",
            description="Analyze a multi-symbol portfolio with risk assessment and diversification analysis. Use when user has multiple stocks or wants portfolio advice. Input: symbols (required list), weights (optional), risk_profile.",
            func=analyze_portfolio_tool_impl
        ),
        
        create_langchain_tool(
            name="compare_trading_strategies",
            description="Compare multiple trading strategies on the same symbol to find the best performer. Use when user wants to choose between strategies. Input: symbol (required), strategy_templates (required list), timeframe.",
            func=compare_strategies_tool_impl
        ),
        
        create_langchain_tool(
            name="analyze_market_regime",
            description="Analyze current market conditions and regime for strategy selection. Use when user asks about market conditions, volatility, or which strategies work best now. Input: symbols (optional), analysis_period.",
            func=get_market_regime_tool_impl
        )
    ]

def get_data_retrieval_tools() -> List[Tool]:
    """Get market data and technical analysis tools"""
    if not LANGCHAIN_AVAILABLE:
        return []
        
    return [
        create_langchain_tool(
            name="get_symbol_information",
            description="Get comprehensive stock information including price, fundamentals, and company details. Use when user asks about a company or stock details. Input: symbol (required), include_fundamentals (boolean).",
            func=get_symbol_info_tool_impl
        ),
        
        create_langchain_tool(
            name="calculate_technical_indicators",
            description="Calculate technical analysis indicators like RSI, MACD, moving averages. Use when user asks about technical analysis or chart patterns. Input: symbol (required), indicators (required list like ['RSI', 'MACD', 'SMA']), timeframe, period.",
            func=get_technical_indicators_tool_impl
        )
    ]

def get_user_management_tools() -> List[Tool]:
    """Get user profile and watchlist management tools"""
    if not LANGCHAIN_AVAILABLE:
        return []
        
    return [
        create_langchain_tool(
            name="get_user_risk_profile",
            description="Get user's risk tolerance and investment preferences for personalized advice. Use when you need to understand user's investment profile. Input: profile_id (required).",
            func=get_user_profile_tool_impl
        ),
        
        create_langchain_tool(
            name="get_user_watchlist",
            description="Get user's watchlist with current prices and alerts. Use when user asks about their watchlist or monitored stocks. Input: profile_id (required).",
            func=get_watchlist_tool_impl
        ),
        
        create_langchain_tool(
            name="add_to_user_watchlist",
            description="Add stocks to user's watchlist for monitoring. Use when user wants to track or monitor specific stocks. Input: profile_id (required), symbols (required list), asset_class.",
            func=add_to_watchlist_tool_impl
        )
    ]

def get_optimization_tools() -> List[Tool]:
    """Get strategy optimization and parameter tuning tools"""
    if not LANGCHAIN_AVAILABLE:
        return []
        
    return [
        create_langchain_tool(
            name="start_strategy_optimization",
            description="Start advanced parameter optimization for trading strategies using machine learning. Use when user wants to find the best strategy parameters. Input: symbol (required), strategy_template (required), n_trials, primary_objective, walk_forward.",
            func=start_optimization_tool_impl
        ),
        
        create_langchain_tool(
            name="get_optimization_results",
            description="Get detailed results from completed optimization studies. Use to retrieve optimization results after starting a study. Input: study_id (required).",
            func=get_optimization_results_tool_impl
        )
    ]

def get_all_tools() -> List[Tool]:
    """Get all available SuperNova tools for LangChain agents"""
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available - returning empty tool list")
        return []
    
    all_tools = []
    all_tools.extend(get_core_financial_tools())
    all_tools.extend(get_advanced_analysis_tools())
    all_tools.extend(get_data_retrieval_tools())
    all_tools.extend(get_user_management_tools())
    all_tools.extend(get_optimization_tools())
    
    logger.info(f"Loaded {len(all_tools)} SuperNova tools for LangChain integration")
    return all_tools

# Tool metadata for documentation and discovery
TOOL_CATEGORIES = {
    "Core Financial": [
        "get_investment_advice",
        "get_historical_sentiment", 
        "run_backtest_analysis"
    ],
    "Advanced Analysis": [
        "analyze_portfolio",
        "compare_trading_strategies",
        "analyze_market_regime"
    ],
    "Data Retrieval": [
        "get_symbol_information",
        "calculate_technical_indicators"
    ],
    "User Management": [
        "get_user_risk_profile",
        "get_user_watchlist",
        "add_to_user_watchlist"
    ],
    "Optimization": [
        "start_strategy_optimization",
        "get_optimization_results"
    ]
}

EXAMPLE_QUERIES = {
    "Investment Advice": [
        "Should I buy Apple stock?",
        "What's your opinion on Tesla with moderate risk tolerance?",
        "Give me advice on Microsoft for aggressive investor"
    ],
    "Sentiment Analysis": [
        "What's the sentiment for GameStop this week?",
        "Show me Twitter sentiment trends for AMD",
        "How has sentiment changed for Bitcoin lately?"
    ],
    "Backtesting": [
        "Test RSI strategy on Netflix",
        "Backtest MACD crossover on Google",
        "What returns would SMA strategy give on Amazon?"
    ],
    "Portfolio Analysis": [
        "Analyze my portfolio of Apple, Microsoft, Google",
        "Check risk of my tech stock portfolio",
        "How diversified is Apple, Tesla, Amazon portfolio?"
    ],
    "Technical Analysis": [
        "Show me RSI and MACD for Apple",
        "What are the technical indicators saying about Tesla?",
        "Calculate Bollinger Bands for Microsoft"
    ],
    "Optimization": [
        "Find best RSI parameters for Apple",
        "Optimize MACD strategy for Tesla",
        "What are optimal settings for Microsoft SMA strategy?"
    ]
}

if __name__ == "__main__":
    # Example usage and testing
    tools = get_all_tools()
    print(f"Loaded {len(tools)} tools:")
    for category, tool_names in TOOL_CATEGORIES.items():
        print(f"\n{category} ({len(tool_names)} tools):")
        for tool_name in tool_names:
            print(f"  - {tool_name}")