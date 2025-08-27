#!/usr/bin/env python3
"""
SuperNova Prefect Integration Demo

This script demonstrates how to use the new Prefect-based sentiment analysis
workflows alongside the traditional asyncio implementation.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the supernova package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock environment for demo
class MockSettings:
    ENABLE_PREFECT = True  # Enable Prefect for demo
    X_BEARER_TOKEN = None
    REDDIT_CLIENT_ID = None
    REDDIT_CLIENT_SECRET = None
    REDDIT_USER_AGENT = "SuperNova-Demo/1.0"
    NEWSAPI_KEY = None

# Mock the dotenv module to avoid dependency
sys.modules['dotenv'] = type(sys)('dotenv')
sys.modules['dotenv'].load_dotenv = lambda: None

# Replace the settings module temporarily
import supernova.config
supernova.config.settings = MockSettings()

from supernova.sentiment import (
    generate_sentiment_signal, 
    generate_batch_sentiment_signals,
    MarketRegime, 
    score_text
)
from supernova.workflows import is_prefect_available, PREFECT_AVAILABLE

async def demo_basic_sentiment():
    """Demo basic sentiment analysis without external APIs"""
    print("=== Basic Sentiment Analysis Demo ===")
    
    # Test individual text scoring
    texts = [
        "AAPL stock is performing excellently with strong earnings beat!",
        "Tesla disappointed investors with poor quarterly results",
        "The market is showing neutral sentiment towards MSFT",
        "Breaking: NVDA announces groundbreaking AI chip, stock surges!"
    ]
    
    for text in texts:
        result = score_text(text)
        print(f"Text: {text[:50]}...")
        print(f"  Score: {result.score:.3f}, Confidence: {result.confidence:.3f}")
        print(f"  Figures mentioned: {list(result.figures.keys()) if result.figures else 'None'}")
        print()

async def demo_prefect_vs_asyncio():
    """Demo comparison between Prefect and asyncio implementations"""
    print("=== Prefect vs Asyncio Comparison Demo ===")
    print(f"Prefect Available: {PREFECT_AVAILABLE}")
    print(f"Prefect Enabled in Config: {is_prefect_available()}")
    print()
    
    # Since we don't have real API keys, we'll simulate the behavior
    # In a real environment, this would actually fetch data from Twitter, Reddit, and news sources
    
    symbol = "AAPL"
    
    print(f"Analyzing sentiment for {symbol}...")
    print("Note: This demo runs without real API keys, so results will be neutral")
    print()
    
    # Test with asyncio (traditional method)
    print("1. Using traditional asyncio.gather method:")
    start_time = datetime.now()
    
    try:
        signal_asyncio = await generate_sentiment_signal(
            symbol=symbol,
            use_prefect=False  # Force asyncio
        )
        duration_asyncio = (datetime.now() - start_time).total_seconds()
        
        print(f"   Overall Score: {signal_asyncio.overall_score:.3f}")
        print(f"   Confidence: {signal_asyncio.confidence:.3f}")
        print(f"   Duration: {duration_asyncio:.2f} seconds")
        print(f"   Sources: {list(signal_asyncio.source_breakdown.keys())}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test with Prefect (if available)
    print("2. Using Prefect workflow method:")
    start_time = datetime.now()
    
    try:
        signal_prefect = await generate_sentiment_signal(
            symbol=symbol,
            use_prefect=True  # Force Prefect
        )
        duration_prefect = (datetime.now() - start_time).total_seconds()
        
        print(f"   Overall Score: {signal_prefect.overall_score:.3f}")
        print(f"   Confidence: {signal_prefect.confidence:.3f}")
        print(f"   Duration: {duration_prefect:.2f} seconds")
        print(f"   Sources: {list(signal_prefect.source_breakdown.keys())}")
        
        if PREFECT_AVAILABLE:
            print("   * Used Prefect workflows with retry logic and monitoring")
        else:
            print("   > Fell back to asyncio (Prefect not available)")
        
    except Exception as e:
        print(f"   Error: {e}")

async def demo_batch_processing():
    """Demo batch sentiment analysis"""
    print("\n=== Batch Sentiment Analysis Demo ===")
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    print(f"Analyzing sentiment for {len(symbols)} symbols: {symbols}")
    print("Note: This demo runs without real API keys, so results will be neutral")
    print()
    
    start_time = datetime.now()
    
    try:
        batch_results = await generate_batch_sentiment_signals(
            symbols=symbols,
            use_prefect=True  # Try Prefect batch processing
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Batch processing completed in {duration:.2f} seconds")
        print()
        
        for symbol, signal in batch_results.items():
            print(f"{symbol}:")
            print(f"  Score: {signal.overall_score:.3f}")
            print(f"  Confidence: {signal.confidence:.3f}")
            print(f"  Timestamp: {signal.timestamp.strftime('%H:%M:%S')}")
        
        if PREFECT_AVAILABLE:
            print("\n* Used Prefect batch workflow with concurrent task execution")
        else:
            print("\n> Used individual asyncio calls (Prefect not available)")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")

async def demo_market_regime_effects():
    """Demo how market regime affects sentiment scoring"""
    print("\n=== Market Regime Effects Demo ===")
    
    from supernova.sentiment import adjust_for_market_regime
    
    base_sentiment = 0.6  # Positive sentiment
    
    regimes = [
        MarketRegime.BULL,
        MarketRegime.BEAR,
        MarketRegime.SIDEWAYS,
        MarketRegime.VOLATILE
    ]
    
    print(f"Base sentiment score: {base_sentiment}")
    print("Regime adjustments:")
    
    for regime in regimes:
        adjusted = adjust_for_market_regime(base_sentiment, regime)
        print(f"  {regime.value:>8}: {adjusted:.3f} (change: {adjusted-base_sentiment:+.3f})")

def print_prefect_features():
    """Print information about Prefect features"""
    print("\n=== Prefect Integration Features ===")
    
    features = [
        "* Automatic retries on API failures (configurable per data source)",
        "* Timeout handling for long-running tasks",
        "* Concurrent execution of data fetching tasks",
        "* Structured logging and monitoring",
        "* Observable workflow execution with task status tracking",
        "* Graceful error handling with fallback to neutral signals",
        "* Configurable retry delays and maximum attempts",
        "* Task result caching capabilities",
        "* Ready for scheduled execution and deployment",
        "* Backward compatibility with existing asyncio implementation"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n=== Configuration Options ===")
    config_options = [
        "ENABLE_PREFECT: Enable/disable Prefect workflows",
        "PREFECT_TWITTER_RETRIES: Retry attempts for Twitter API",
        "PREFECT_REDDIT_RETRIES: Retry attempts for Reddit API", 
        "PREFECT_NEWS_RETRIES: Retry attempts for News API",
        "PREFECT_RETRY_DELAY: Delay between retry attempts (seconds)",
        "PREFECT_TASK_TIMEOUT: Task timeout (seconds)",
        "PREFECT_MAX_CONCURRENT_TASKS: Maximum concurrent tasks",
    ]
    
    for option in config_options:
        print(f"  • {option}")

async def main():
    """Main demo function"""
    print("SuperNova Prefect Integration Demo")
    print("=" * 50)
    print()
    
    # Run all demos
    await demo_basic_sentiment()
    await demo_prefect_vs_asyncio()
    await demo_batch_processing()
    await demo_market_regime_effects()
    print_prefect_features()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print()
    print("Next steps to use Prefect in production:")
    print("1. Install Prefect: pip install prefect>=2.14.0")
    print("2. Set ENABLE_PREFECT=true in your environment")
    print("3. Configure API keys for Twitter, Reddit, and News sources")
    print("4. Optionally set up Prefect Server/Cloud for advanced monitoring")
    print("5. Deploy workflows using 'prefect deploy' command")

if __name__ == "__main__":
    asyncio.run(main())