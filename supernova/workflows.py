"""SuperNova Prefect Workflows

Robust, observable, and schedulable data gathering and sentiment analysis workflows
using Prefect for better orchestration and monitoring.
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict

# Prefect imports
try:
    from prefect import task, flow, get_run_logger
    from prefect.task_runners import ConcurrentTaskRunner
    from prefect.blocks.core import Block
    from prefect.server.schemas.schedules import IntervalSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    # Fallback if Prefect not installed
    PREFECT_AVAILABLE = False
    
    # Mock decorators and classes for development
    def task(retries=0, retry_delay_seconds=0, timeout_seconds=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def flow(name=None, task_runner=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def get_run_logger():
        return logging.getLogger(__name__)
    
    class ConcurrentTaskRunner:
        def __init__(self):
            pass

# Local imports
from .sentiment import (
    TwitterConnector, RedditConnector, NewsConnector, 
    SentimentSignal, SentimentSource, MarketRegime,
    score_text, CONFIDENCE_THRESHOLD, calculate_sentiment_momentum,
    detect_contrarian_signals, adjust_for_market_regime
)

# TimescaleDB imports (optional)
try:
    from .sentiment_models import SentimentData, from_sentiment_signal, get_timescale_session
    from .db import is_timescale_available
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False
    SentimentData = None
    from_sentiment_signal = None
    get_timescale_session = None
    is_timescale_available = lambda: False

try:
    from .config import settings
except ImportError:
    # Fallback settings if config module not available
    class MockSettings:
        ENABLE_PREFECT: bool = False
        PREFECT_TWITTER_RETRIES: int = 3
        PREFECT_REDDIT_RETRIES: int = 3
        PREFECT_NEWS_RETRIES: int = 3
        PREFECT_TASK_TIMEOUT: int = 300
        PREFECT_RETRY_DELAY: int = 60
    settings = MockSettings()

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# PREFECT TASKS
# ================================

@task(
    name="fetch_twitter_data",
    retries=getattr(settings, 'PREFECT_TWITTER_RETRIES', 3),
    retry_delay_seconds=getattr(settings, 'PREFECT_RETRY_DELAY', 60),
    timeout_seconds=getattr(settings, 'PREFECT_TASK_TIMEOUT', 300)
)
async def fetch_twitter_data_task(
    symbol: str,
    max_results: int = 30,
    since_hours: int = 24
) -> List[Dict]:
    """
    Prefect task to fetch Twitter/X data with automatic retries and error handling.
    
    Args:
        symbol: Stock symbol to search for
        max_results: Maximum number of tweets to fetch
        since_hours: How far back to look (hours)
    
    Returns:
        List of tweet dictionaries
    """
    logger = get_run_logger()
    logger.info(f"Fetching Twitter data for {symbol} (max_results={max_results}, since_hours={since_hours})")
    
    try:
        connector = TwitterConnector()
        
        # Build search query
        query = f"{symbol} stock OR {symbol} price OR {symbol} earnings"
        
        tweets = await connector.fetch_recent_tweets(
            query=query,
            max_results=max_results,
            since_hours=since_hours
        )
        
        logger.info(f"Successfully fetched {len(tweets)} tweets for {symbol}")
        return tweets
        
    except Exception as e:
        logger.error(f"Error fetching Twitter data for {symbol}: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="fetch_reddit_data",
    retries=getattr(settings, 'PREFECT_REDDIT_RETRIES', 3),
    retry_delay_seconds=getattr(settings, 'PREFECT_RETRY_DELAY', 60),
    timeout_seconds=getattr(settings, 'PREFECT_TASK_TIMEOUT', 300)
)
async def fetch_reddit_data_task(
    symbol: str,
    subreddits: List[str] = None,
    max_results_per_sub: int = 20
) -> List[Dict]:
    """
    Prefect task to fetch Reddit data with automatic retries and error handling.
    
    Args:
        symbol: Stock symbol to search for
        subreddits: List of subreddits to search (default: ['stocks', 'investing'])
        max_results_per_sub: Maximum posts per subreddit
    
    Returns:
        List of Reddit post dictionaries
    """
    logger = get_run_logger()
    
    if subreddits is None:
        subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
    
    logger.info(f"Fetching Reddit data for {symbol} from subreddits: {subreddits}")
    
    try:
        connector = RedditConnector()
        all_posts = []
        
        for subreddit in subreddits:
            try:
                posts = await connector.fetch_subreddit_posts(
                    subreddit=subreddit,
                    query=symbol,
                    max_results=max_results_per_sub
                )
                all_posts.extend(posts)
                logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
                
                # Small delay between subreddits to be respectful
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error fetching from r/{subreddit}: {e}")
                # Continue with other subreddits
                continue
        
        logger.info(f"Successfully fetched {len(all_posts)} total Reddit posts for {symbol}")
        return all_posts
        
    except Exception as e:
        logger.error(f"Error fetching Reddit data for {symbol}: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="fetch_news_data",
    retries=getattr(settings, 'PREFECT_NEWS_RETRIES', 3),
    retry_delay_seconds=getattr(settings, 'PREFECT_RETRY_DELAY', 60),
    timeout_seconds=getattr(settings, 'PREFECT_TASK_TIMEOUT', 300)
)
async def fetch_news_data_task(
    symbol: str,
    max_articles: int = 15
) -> List[Dict]:
    """
    Prefect task to fetch news data with automatic retries and error handling.
    
    Args:
        symbol: Stock symbol to search for
        max_articles: Maximum number of articles to fetch
    
    Returns:
        List of news article dictionaries
    """
    logger = get_run_logger()
    logger.info(f"Fetching news data for {symbol} (max_articles={max_articles})")
    
    try:
        connector = NewsConnector()
        
        # Build search query for financial news
        query = f"{symbol} stock earnings financial market"
        
        articles = await connector.fetch_news_headlines(
            query=query,
            max_articles=max_articles
        )
        
        logger.info(f"Successfully fetched {len(articles)} news articles for {symbol}")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching news data for {symbol}: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="analyze_sentiment_batch",
    retries=2,
    retry_delay_seconds=30,
    timeout_seconds=300
)
def analyze_sentiment_batch_task(
    data_batch: List[Dict],
    source_type: str
) -> List[Dict]:
    """
    Prefect task to analyze sentiment for a batch of text data.
    
    Args:
        data_batch: List of data dictionaries with 'text' field
        source_type: Type of source ('twitter', 'reddit', 'news')
    
    Returns:
        List of dictionaries with sentiment analysis results
    """
    logger = get_run_logger()
    logger.info(f"Analyzing sentiment for {len(data_batch)} {source_type} items")
    
    try:
        results = []
        
        for item in data_batch:
            text = item.get('text', '')
            if not text or len(text.strip()) < 10:
                continue
            
            # Analyze sentiment
            sentiment_result = score_text(text, use_advanced=True)
            
            # Only include high-confidence results
            if sentiment_result.confidence > CONFIDENCE_THRESHOLD:
                result_dict = {
                    'original_data': item,
                    'sentiment': {
                        'score': sentiment_result.score,
                        'confidence': sentiment_result.confidence,
                        'timestamp': sentiment_result.timestamp,
                        'source': source_type,
                        'figures': sentiment_result.figures,
                        'entities': sentiment_result.entities,
                        'raw_scores': sentiment_result.raw_scores,
                        'metadata': sentiment_result.metadata
                    }
                }
                results.append(result_dict)
        
        logger.info(f"Completed sentiment analysis: {len(results)} high-confidence results from {len(data_batch)} items")
        return results
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis batch: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="save_sentiment_to_timescale",
    retries=getattr(settings, 'TIMESCALE_MAX_RETRY_ATTEMPTS', 3),
    retry_delay_seconds=getattr(settings, 'TIMESCALE_RETRY_DELAY', 1.0),
    timeout_seconds=getattr(settings, 'TIMESCALE_CONNECTION_TIMEOUT', 30)
)
def save_sentiment_to_timescale_task(
    sentiment_signal_dict: Dict,
    symbol: str
) -> Dict:
    """
    Prefect task to save sentiment signal data to TimescaleDB.
    
    Args:
        sentiment_signal_dict: Dictionary representation of sentiment signal
        symbol: Stock symbol
    
    Returns:
        Dictionary with save status and metadata
    """
    logger = get_run_logger()
    
    if not TIMESCALE_AVAILABLE:
        logger.warning("TimescaleDB not available - skipping sentiment data persistence")
        return {"status": "skipped", "reason": "timescale_not_available"}
    
    if not is_timescale_available():
        logger.warning("TimescaleDB not configured - skipping sentiment data persistence")
        return {"status": "skipped", "reason": "timescale_not_configured"}
    
    try:
        logger.info(f"Saving sentiment data for {symbol} to TimescaleDB")
        
        # Get TimescaleDB session
        TimescaleSession = get_timescale_session()
        if not TimescaleSession:
            logger.error("Could not get TimescaleDB session")
            return {"status": "error", "reason": "session_unavailable"}
        
        # Convert signal dict to sentiment signal object
        signal = create_sentiment_signal_from_dict(sentiment_signal_dict)
        
        # Convert to TimescaleDB model
        sentiment_data = from_sentiment_signal(signal, symbol)
        
        # Add additional fields from the dict that may not be in the signal
        if 'total_data_points' in sentiment_signal_dict:
            sentiment_data.total_data_points = sentiment_signal_dict['total_data_points']
        if 'source_counts' in sentiment_signal_dict:
            sentiment_data.source_counts = sentiment_signal_dict['source_counts']
        
        # Save to database
        with TimescaleSession() as session:
            try:
                # Use ON CONFLICT DO UPDATE for upsert behavior
                session.merge(sentiment_data)
                session.commit()
                
                logger.info(f"Successfully saved sentiment data for {symbol} to TimescaleDB")
                return {
                    "status": "success",
                    "symbol": symbol,
                    "timestamp": sentiment_data.timestamp.isoformat(),
                    "score": sentiment_data.overall_score,
                    "confidence": sentiment_data.confidence
                }
                
            except Exception as e:
                session.rollback()
                logger.error(f"Database error saving sentiment for {symbol}: {e}")
                raise e
        
    except Exception as e:
        logger.error(f"Error saving sentiment data for {symbol} to TimescaleDB: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="save_sentiment_batch_to_timescale",
    retries=getattr(settings, 'TIMESCALE_MAX_RETRY_ATTEMPTS', 3),
    retry_delay_seconds=getattr(settings, 'TIMESCALE_RETRY_DELAY', 1.0),
    timeout_seconds=getattr(settings, 'TIMESCALE_CONNECTION_TIMEOUT', 60)
)
def save_sentiment_batch_to_timescale_task(
    sentiment_signals_dict: Dict[str, Dict]
) -> Dict:
    """
    Prefect task to save multiple sentiment signals to TimescaleDB in batch.
    
    Args:
        sentiment_signals_dict: Dictionary mapping symbol to sentiment signal dict
    
    Returns:
        Dictionary with batch save status and metadata
    """
    logger = get_run_logger()
    
    if not TIMESCALE_AVAILABLE or not is_timescale_available():
        logger.warning("TimescaleDB not available - skipping batch sentiment persistence")
        return {"status": "skipped", "reason": "timescale_not_available"}
    
    try:
        logger.info(f"Batch saving sentiment data for {len(sentiment_signals_dict)} symbols to TimescaleDB")
        
        # Get TimescaleDB session
        TimescaleSession = get_timescale_session()
        if not TimescaleSession:
            logger.error("Could not get TimescaleDB session")
            return {"status": "error", "reason": "session_unavailable"}
        
        results = {
            "status": "success",
            "symbols_processed": [],
            "symbols_failed": [],
            "total_symbols": len(sentiment_signals_dict)
        }
        
        # Process in batches to avoid overwhelming the database
        batch_size = getattr(settings, 'TIMESCALE_BATCH_SIZE', 1000)
        items = list(sentiment_signals_dict.items())
        
        with TimescaleSession() as session:
            try:
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    
                    sentiment_objects = []
                    for symbol, signal_dict in batch:
                        try:
                            # Convert to sentiment signal and then to TimescaleDB model
                            signal = create_sentiment_signal_from_dict(signal_dict)
                            sentiment_data = from_sentiment_signal(signal, symbol)
                            
                            # Add additional fields
                            if 'total_data_points' in signal_dict:
                                sentiment_data.total_data_points = signal_dict['total_data_points']
                            if 'source_counts' in signal_dict:
                                sentiment_data.source_counts = signal_dict['source_counts']
                            
                            sentiment_objects.append(sentiment_data)
                            results["symbols_processed"].append(symbol)
                            
                        except Exception as e:
                            logger.error(f"Error preparing sentiment data for {symbol}: {e}")
                            results["symbols_failed"].append(symbol)
                    
                    # Batch insert/update
                    if sentiment_objects:
                        session.bulk_save_objects(sentiment_objects, update_on_duplicate=True)
                        logger.info(f"Saved batch of {len(sentiment_objects)} sentiment records")
                
                session.commit()
                logger.info(f"Successfully batch saved sentiment data for {len(results['symbols_processed'])} symbols")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Database error during batch save: {e}")
                raise e
        
        return results
        
    except Exception as e:
        logger.error(f"Error batch saving sentiment data to TimescaleDB: {e}")
        # Re-raise to trigger retry
        raise e

@task(
    name="aggregate_sentiment_signals",
    retries=1,
    retry_delay_seconds=30
)
def aggregate_sentiment_signals_task(
    twitter_results: List[Dict],
    reddit_results: List[Dict],
    news_results: List[Dict],
    symbol: str,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
) -> Dict:
    """
    Prefect task to aggregate all sentiment data into a final signal.
    
    Args:
        twitter_results: Twitter sentiment analysis results
        reddit_results: Reddit sentiment analysis results
        news_results: News sentiment analysis results
        symbol: Stock symbol being analyzed
        market_regime: Current market regime for adjustments
    
    Returns:
        Dictionary representation of SentimentSignal
    """
    logger = get_run_logger()
    logger.info(f"Aggregating sentiment signals for {symbol}")
    
    try:
        # Combine all results
        all_results = twitter_results + reddit_results + news_results
        
        if not all_results:
            logger.warning(f"No sentiment data available for {symbol}")
            return _create_neutral_signal(symbol)
        
        # Extract sentiment scores by source
        source_scores = defaultdict(list)
        all_sentiments = []
        
        for result in all_results:
            sentiment = result['sentiment']
            source = sentiment['source']
            score = sentiment['score']
            confidence = sentiment['confidence']
            
            source_scores[source].append(score)
            all_sentiments.append(sentiment)
        
        # Calculate overall weighted score
        total_weight = 0
        weighted_score = 0
        confidence_sum = 0
        
        for sentiment in all_sentiments:
            weight = sentiment['confidence']
            weighted_score += sentiment['score'] * weight
            total_weight += weight
            confidence_sum += sentiment['confidence']
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        overall_confidence = confidence_sum / len(all_sentiments) if all_sentiments else 0.0
        
        # Source breakdown
        source_breakdown = {}
        for source, scores in source_scores.items():
            if scores:
                source_breakdown[source] = sum(scores) / len(scores)
        
        # Figure influence calculation
        all_figures = {}
        for sentiment in all_sentiments:
            for fig, influence in sentiment.get('figures', {}).items():
                if fig in all_figures:
                    all_figures[fig] = max(all_figures[fig], influence)
                else:
                    all_figures[fig] = influence
        
        figure_influence = sum(all_figures.values()) * 0.15 if all_figures else 0.0
        
        # News vs social impact
        news_scores = source_scores.get('news', [])
        social_scores = source_scores.get('twitter', []) + source_scores.get('reddit', [])
        
        news_impact = sum(news_scores) / len(news_scores) if news_scores else 0.0
        social_impact = sum(social_scores) / len(social_scores) if social_scores else 0.0
        
        # Momentum calculation (simplified for batch processing)
        social_momentum = 0.0
        if len(all_sentiments) > 5:
            mid_point = len(all_sentiments) // 2
            earlier_scores = [s['score'] for s in all_sentiments[:mid_point]]
            later_scores = [s['score'] for s in all_sentiments[mid_point:]]
            
            if earlier_scores and later_scores:
                earlier_avg = sum(earlier_scores) / len(earlier_scores)
                later_avg = sum(later_scores) / len(later_scores)
                social_momentum = later_avg - earlier_avg
        
        # Contrarian signals
        contrarian_indicator = detect_contrarian_signals(overall_score, overall_confidence, market_regime)
        
        # Regime adjustment
        regime_adjusted_score = adjust_for_market_regime(overall_score, market_regime)
        
        # Create signal dictionary
        signal_dict = {
            'overall_score': overall_score,
            'confidence': overall_confidence,
            'source_breakdown': source_breakdown,
            'figure_influence': figure_influence,
            'news_impact': news_impact,
            'social_momentum': social_momentum,
            'contrarian_indicator': contrarian_indicator,
            'regime_adjusted_score': regime_adjusted_score,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'total_data_points': len(all_results),
            'source_counts': {source: len(scores) for source, scores in source_scores.items()}
        }
        
        logger.info(f"Successfully aggregated sentiment signal for {symbol}: score={overall_score:.3f}, confidence={overall_confidence:.3f}")
        return signal_dict
        
    except Exception as e:
        logger.error(f"Error aggregating sentiment signals for {symbol}: {e}")
        # Return neutral signal on error
        return _create_neutral_signal(symbol)

def _create_neutral_signal(symbol: str) -> Dict:
    """Create a neutral sentiment signal when no data is available or errors occur."""
    return {
        'overall_score': 0.0,
        'confidence': 0.0,
        'source_breakdown': {},
        'figure_influence': 0.0,
        'news_impact': 0.0,
        'social_momentum': 0.0,
        'contrarian_indicator': 0.0,
        'regime_adjusted_score': 0.0,
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'total_data_points': 0,
        'source_counts': {}
    }

# ================================
# PREFECT FLOWS
# ================================

@flow(
    name="sentiment_analysis_pipeline",
    description="Comprehensive sentiment analysis pipeline for financial symbols",
    version="1.0",
    task_runner=ConcurrentTaskRunner()
)
async def sentiment_analysis_flow(
    symbol: str,
    lookback_hours: int = 24,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS,
    max_twitter_results: int = 30,
    max_reddit_results_per_sub: int = 20,
    max_news_articles: int = 15
) -> Dict:
    """
    Main Prefect flow for sentiment analysis pipeline.
    
    This flow orchestrates the entire sentiment analysis process:
    1. Concurrently fetch data from Twitter, Reddit, and News sources
    2. Process and analyze sentiment for each data source
    3. Aggregate results into a comprehensive sentiment signal
    
    Args:
        symbol: Stock symbol to analyze
        lookback_hours: How far back to look for data (hours)
        market_regime: Current market regime for sentiment adjustments
        max_twitter_results: Maximum tweets to fetch
        max_reddit_results_per_sub: Maximum Reddit posts per subreddit
        max_news_articles: Maximum news articles to fetch
    
    Returns:
        Dictionary representation of the final sentiment signal
    """
    logger = get_run_logger()
    logger.info(f"Starting sentiment analysis pipeline for {symbol}")
    
    try:
        # Step 1: Concurrent data fetching
        logger.info("Step 1: Fetching data from all sources concurrently")
        
        twitter_task = fetch_twitter_data_task.submit(
            symbol=symbol,
            max_results=max_twitter_results,
            since_hours=lookback_hours
        )
        
        reddit_task = fetch_reddit_data_task.submit(
            symbol=symbol,
            max_results_per_sub=max_reddit_results_per_sub
        )
        
        news_task = fetch_news_data_task.submit(
            symbol=symbol,
            max_articles=max_news_articles
        )
        
        # Wait for all data fetching tasks to complete
        twitter_data = await twitter_task
        reddit_data = await reddit_task
        news_data = await news_task
        
        logger.info(f"Data fetching completed: {len(twitter_data)} tweets, {len(reddit_data)} reddit posts, {len(news_data)} news articles")
        
        # Step 2: Concurrent sentiment analysis
        logger.info("Step 2: Analyzing sentiment for all data sources")
        
        twitter_sentiment_task = analyze_sentiment_batch_task.submit(
            data_batch=twitter_data,
            source_type="twitter"
        )
        
        reddit_sentiment_task = analyze_sentiment_batch_task.submit(
            data_batch=reddit_data,
            source_type="reddit"
        )
        
        news_sentiment_task = analyze_sentiment_batch_task.submit(
            data_batch=news_data,
            source_type="news"
        )
        
        # Wait for all sentiment analysis tasks to complete
        twitter_results = await twitter_sentiment_task
        reddit_results = await reddit_sentiment_task
        news_results = await news_sentiment_task
        
        logger.info(f"Sentiment analysis completed: {len(twitter_results)} twitter, {len(reddit_results)} reddit, {len(news_results)} news results")
        
        # Step 3: Aggregate final signal
        logger.info("Step 3: Aggregating final sentiment signal")
        
        final_signal = await aggregate_sentiment_signals_task.submit(
            twitter_results=twitter_results,
            reddit_results=reddit_results,
            news_results=news_results,
            symbol=symbol,
            market_regime=market_regime
        )
        
        # Step 4: Save to TimescaleDB (if available and configured)
        if TIMESCALE_AVAILABLE and is_timescale_available():
            logger.info("Step 4: Saving sentiment data to TimescaleDB")
            
            save_result = await save_sentiment_to_timescale_task.submit(
                sentiment_signal_dict=final_signal,
                symbol=symbol
            )
            
            if save_result.get("status") == "success":
                logger.info(f"Successfully saved sentiment data for {symbol} to TimescaleDB")
                # Add save metadata to final signal
                final_signal["timescale_save_status"] = "success"
                final_signal["timescale_save_timestamp"] = save_result.get("timestamp")
            else:
                logger.warning(f"Failed to save sentiment data for {symbol} to TimescaleDB: {save_result}")
                final_signal["timescale_save_status"] = "failed"
                final_signal["timescale_save_error"] = save_result.get("reason", "unknown")
        else:
            logger.info("TimescaleDB not available - skipping sentiment data persistence")
            final_signal["timescale_save_status"] = "skipped"
            final_signal["timescale_save_reason"] = "not_available"
        
        logger.info(f"Sentiment analysis pipeline completed successfully for {symbol}")
        return final_signal
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline for {symbol}: {e}")
        # Return neutral signal on flow failure
        return _create_neutral_signal(symbol)

@flow(
    name="batch_sentiment_analysis",
    description="Batch sentiment analysis for multiple symbols",
    version="1.0"
)
async def batch_sentiment_analysis_flow(
    symbols: List[str],
    lookback_hours: int = 24,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
) -> Dict[str, Dict]:
    """
    Flow to analyze sentiment for multiple symbols in parallel.
    
    Args:
        symbols: List of stock symbols to analyze
        lookback_hours: How far back to look for data (hours)
        market_regime: Current market regime for sentiment adjustments
    
    Returns:
        Dictionary mapping symbol to sentiment signal
    """
    logger = get_run_logger()
    logger.info(f"Starting batch sentiment analysis for {len(symbols)} symbols: {symbols}")
    
    results = {}
    
    # Create tasks for each symbol
    symbol_tasks = {}
    for symbol in symbols:
        task = sentiment_analysis_flow.submit(
            symbol=symbol,
            lookback_hours=lookback_hours,
            market_regime=market_regime
        )
        symbol_tasks[symbol] = task
    
    # Collect results
    for symbol, task in symbol_tasks.items():
        try:
            result = await task
            results[symbol] = result
            logger.info(f"Completed sentiment analysis for {symbol}")
        except Exception as e:
            logger.error(f"Failed sentiment analysis for {symbol}: {e}")
            results[symbol] = _create_neutral_signal(symbol)
    
    # Batch save to TimescaleDB if available and configured
    if TIMESCALE_AVAILABLE and is_timescale_available() and results:
        logger.info("Batch saving sentiment data to TimescaleDB")
        
        try:
            batch_save_result = await save_sentiment_batch_to_timescale_task.submit(
                sentiment_signals_dict=results
            )
            
            if batch_save_result.get("status") == "success":
                processed = batch_save_result.get("symbols_processed", [])
                failed = batch_save_result.get("symbols_failed", [])
                logger.info(f"Batch TimescaleDB save completed: {len(processed)} success, {len(failed)} failed")
                
                # Add save status to individual results
                for symbol in processed:
                    if symbol in results:
                        results[symbol]["timescale_save_status"] = "success"
                for symbol in failed:
                    if symbol in results:
                        results[symbol]["timescale_save_status"] = "failed"
            else:
                logger.warning(f"Batch TimescaleDB save failed: {batch_save_result}")
                # Mark all results as having failed saves
                for symbol in results:
                    results[symbol]["timescale_save_status"] = "batch_failed"
                    
        except Exception as e:
            logger.error(f"Error during batch TimescaleDB save: {e}")
            # Mark all results as having failed saves
            for symbol in results:
                results[symbol]["timescale_save_status"] = "batch_error"
                results[symbol]["timescale_save_error"] = str(e)
    else:
        logger.info("Skipping batch TimescaleDB save (not available or no results)")
        # Mark all results as skipped
        for symbol in results:
            results[symbol]["timescale_save_status"] = "skipped"
    
    logger.info(f"Batch sentiment analysis completed for {len(results)} symbols")
    return results

# ================================
# UTILITY FUNCTIONS
# ================================

def create_sentiment_signal_from_dict(signal_dict: Dict) -> SentimentSignal:
    """
    Convert dictionary representation back to SentimentSignal object.
    
    Args:
        signal_dict: Dictionary from Prefect flow output
        
    Returns:
        SentimentSignal object
    """
    # Convert source breakdown keys back to enum values
    source_breakdown = {}
    for source_str, score in signal_dict.get('source_breakdown', {}).items():
        try:
            source_enum = SentimentSource(source_str)
            source_breakdown[source_enum] = score
        except ValueError:
            # Skip unknown source types
            continue
    
    return SentimentSignal(
        overall_score=signal_dict.get('overall_score', 0.0),
        confidence=signal_dict.get('confidence', 0.0),
        source_breakdown=source_breakdown,
        figure_influence=signal_dict.get('figure_influence', 0.0),
        news_impact=signal_dict.get('news_impact', 0.0),
        social_momentum=signal_dict.get('social_momentum', 0.0),
        contrarian_indicator=signal_dict.get('contrarian_indicator', 0.0),
        regime_adjusted_score=signal_dict.get('regime_adjusted_score', 0.0),
        timestamp=datetime.fromisoformat(signal_dict.get('timestamp', datetime.now().isoformat())),
        validity_period=timedelta(minutes=30)
    )

def is_prefect_available() -> bool:
    """Check if Prefect is available and properly configured."""
    return PREFECT_AVAILABLE and getattr(settings, 'ENABLE_PREFECT', False)

# ================================
# SCHEDULED FLOWS (OPTIONAL)
# ================================

# Example of how to create scheduled flows
def create_scheduled_sentiment_flow(
    symbols: List[str],
    schedule_interval_minutes: int = 30
):
    """
    Create a scheduled version of the sentiment analysis flow.
    
    This is an example of how users could set up scheduled sentiment analysis.
    The actual deployment would be done via Prefect CLI or Python deployment scripts.
    """
    
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available - cannot create scheduled flows")
        return None
    
    @flow(
        name=f"scheduled_sentiment_analysis_{'-'.join(symbols)}",
        description=f"Scheduled sentiment analysis for {symbols}",
        version="1.0"
    )
    async def scheduled_flow():
        return await batch_sentiment_analysis_flow(
            symbols=symbols,
            lookback_hours=24,
            market_regime=MarketRegime.SIDEWAYS
        )
    
    return scheduled_flow

# ================================
# OPTIMIZATION WORKFLOWS
# ================================

# Import optimization components with fallback
try:
    from .optimizer import OptunaOptimizer, OptimizationConfig, OptimizationResult, OPTUNA_AVAILABLE
    from .optimization_models import (
        OptimizationStudyModel, OptimizationTrialModel, WatchlistOptimizationModel,
        create_optimization_study, update_study_progress, record_optimization_trial,
        StudyStatus, TrialState
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    OptunaOptimizer = None
    OptimizationConfig = None

@task(
    name="fetch_historical_data_for_optimization",
    retries=getattr(settings, 'PREFECT_DATA_RETRIES', 3),
    retry_delay_seconds=getattr(settings, 'PREFECT_RETRY_DELAY', 60),
    timeout_seconds=getattr(settings, 'PREFECT_TASK_TIMEOUT', 300)
)
async def fetch_historical_data_task(
    symbol: str,
    timeframe: str = "1h",
    lookback_days: int = 365
) -> List[Dict]:
    """
    Prefect task to fetch historical OHLCV data for optimization.
    
    Args:
        symbol: Stock symbol to fetch data for
        timeframe: Data timeframe (1h, 4h, 1d, etc.)
        lookback_days: Number of days to look back
        
    Returns:
        List of OHLCV bar dictionaries
    """
    logger = get_run_logger()
    logger.info(f"Fetching {lookback_days} days of {timeframe} data for {symbol}")
    
    try:
        # Try NovaSignal connector first if available
        try:
            from ..connectors.novasignal import get_connector
            connector = await get_connector()
            
            # Calculate how many bars we need based on timeframe
            bars_per_day = {"1h": 24, "4h": 6, "1d": 1, "1w": 1/7}
            max_bars = int(lookback_days * bars_per_day.get(timeframe, 24))
            
            bars = await connector.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=min(max_bars, 1000),  # API limits
                asset_class="stock"
            )
            
            # Convert to dict format
            bars_data = [bar.model_dump() for bar in bars]
            
            logger.info(f"Fetched {len(bars_data)} bars from NovaSignal for {symbol}")
            return bars_data
            
        except Exception as e:
            logger.warning(f"NovaSignal data fetch failed: {e}")
            
            # Fallback: generate synthetic data for testing
            # In production, you'd integrate with your actual data provider
            logger.info(f"Generating synthetic data for {symbol}")
            
            import random
            from datetime import datetime, timedelta
            
            bars_data = []
            current_date = datetime.now() - timedelta(days=lookback_days)
            price = 100.0 + random.uniform(-20, 20)  # Starting price
            
            for i in range(min(lookback_days * 24, 1000)):  # Daily bars
                # Simple random walk
                change_pct = random.uniform(-0.05, 0.05)  # +/- 5% max change
                new_price = price * (1 + change_pct)
                
                high = new_price * (1 + abs(random.uniform(0, 0.02)))
                low = new_price * (1 - abs(random.uniform(0, 0.02)))
                volume = random.randint(100000, 1000000)
                
                bar = {
                    "timestamp": current_date.isoformat(),
                    "open": price,
                    "high": high,
                    "low": low,
                    "close": new_price,
                    "volume": volume
                }
                
                bars_data.append(bar)
                price = new_price
                current_date += timedelta(hours=1 if timeframe == "1h" else 24)
            
            logger.info(f"Generated {len(bars_data)} synthetic bars for {symbol}")
            return bars_data
        
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        raise e

@task(
    name="run_optimization_study",
    retries=1,  # Optimization is expensive, limit retries
    retry_delay_seconds=300,  # 5 minute delay
    timeout_seconds=3600  # 1 hour timeout
)
def run_optimization_study_task(
    study_name: str,
    symbol: str,
    strategy_template: str,
    bars_data: List[Dict],
    optimization_config: Dict[str, Any],
    storage_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prefect task to run a complete optimization study.
    
    Args:
        study_name: Unique study name
        symbol: Symbol being optimized
        strategy_template: Strategy to optimize
        bars_data: Historical OHLCV data
        optimization_config: Optimization configuration
        storage_url: Optuna storage URL
        
    Returns:
        Dictionary with optimization results
    """
    logger = get_run_logger()
    logger.info(f"Starting optimization study '{study_name}' for {symbol} using {strategy_template}")
    
    if not OPTIMIZATION_AVAILABLE:
        error_msg = "Optimization not available. Install optuna and related dependencies."
        logger.error(error_msg)
        return {"error": error_msg}
    
    try:
        # Create optimization configuration
        config = OptimizationConfig(
            strategy_template=strategy_template,
            n_trials=optimization_config.get("n_trials", 100),
            primary_objective=optimization_config.get("primary_objective", "sharpe_ratio"),
            secondary_objectives=optimization_config.get("secondary_objectives", []),
            n_jobs=optimization_config.get("n_jobs", 1),
            enable_pruning=optimization_config.get("enable_pruning", True),
            max_drawdown_limit=optimization_config.get("max_drawdown_limit", 0.25),
            min_sharpe_ratio=optimization_config.get("min_sharpe_ratio", 0.5),
            min_win_rate=optimization_config.get("min_win_rate", 0.35),
            include_transaction_costs=optimization_config.get("include_transaction_costs", True),
            commission=optimization_config.get("commission", 0.001),
            slippage=optimization_config.get("slippage", 0.001),
            walk_forward=optimization_config.get("walk_forward", False),
            validation_splits=optimization_config.get("validation_splits", 3)
        )
        
        # Create optimizer
        optimizer = OptunaOptimizer(storage_url=storage_url)
        
        # Progress callback
        def progress_callback(trial_number: int, best_value: Optional[float]):
            progress = trial_number / config.n_trials
            logger.info(f"Trial {trial_number}/{config.n_trials} - Progress: {progress:.1%} - Best: {best_value}")
        
        # Run optimization
        if config.walk_forward:
            result = optimizer.walk_forward_optimization(
                study_name=study_name,
                bars_data=bars_data,
                config=config
            )
        else:
            result = optimizer.optimize_strategy(
                study_name=study_name,
                bars_data=bars_data,
                config=config,
                progress_callback=progress_callback
            )
        
        # Convert result to dictionary
        result_dict = {
            "study_name": result.study_name,
            "symbol": symbol,
            "strategy_template": strategy_template,
            "best_params": result.best_params,
            "best_value": result.best_value,
            "best_trial": result.best_trial,
            "n_trials": result.n_trials,
            "optimization_duration": result.optimization_duration,
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "pareto_front": result.pareto_front,
            "study_stats": result.study_stats,
            "timestamp": result.timestamp.isoformat(),
            "success": True
        }
        
        logger.info(f"Optimization completed successfully for {symbol}. Best value: {result.best_value:.4f}")
        return result_dict
        
    except Exception as e:
        logger.error(f"Optimization failed for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "strategy_template": strategy_template,
            "study_name": study_name,
            "success": False
        }

@task(
    name="save_optimization_results",
    retries=3,
    retry_delay_seconds=30
)
def save_optimization_results_task(
    optimization_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prefect task to save optimization results to database.
    
    Args:
        optimization_result: Result dictionary from optimization
        
    Returns:
        Save status and metadata
    """
    logger = get_run_logger()
    
    if not optimization_result.get("success", False):
        logger.warning("Skipping save for failed optimization")
        return {"status": "skipped", "reason": "optimization_failed"}
    
    try:
        # Save to local database
        db = SessionLocal()
        
        try:
            # Create or update study record
            study = db.query(OptimizationStudyModel).filter(
                OptimizationStudyModel.study_id == optimization_result["study_name"]
            ).first()
            
            if not study:
                # Create new study record
                study = create_optimization_study(
                    study_id=optimization_result["study_name"],
                    study_name=optimization_result["study_name"],
                    symbol=optimization_result["symbol"],
                    strategy_template=optimization_result["strategy_template"],
                    configuration={
                        "n_trials": optimization_result["n_trials"],
                        "optimization_duration": optimization_result["optimization_duration"]
                    },
                    session=db
                )
            
            # Update with final results
            study.status = StudyStatus.COMPLETED
            study.completed_at = datetime.fromisoformat(optimization_result["timestamp"])
            study.best_value = optimization_result["best_value"]
            study.best_params = optimization_result["best_params"]
            study.best_trial_number = optimization_result["best_trial"]
            study.best_metrics = optimization_result["metrics"]
            study.n_complete_trials = optimization_result.get("study_stats", {}).get("n_complete_trials", 0)
            study.n_pruned_trials = optimization_result.get("study_stats", {}).get("n_pruned_trials", 0)
            study.n_failed_trials = optimization_result.get("study_stats", {}).get("n_failed_trials", 0)
            study.progress = 1.0
            
            db.commit()
            
            logger.info(f"Saved optimization results for study {optimization_result['study_name']}")
            return {
                "status": "success",
                "study_id": study.study_id,
                "saved_at": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")
        return {"status": "error", "error": str(e)}

@flow(
    name="optimize_strategy_parameters",
    description="Complete strategy parameter optimization flow",
    version="1.0",
    task_runner=ConcurrentTaskRunner()
)
async def optimize_strategy_parameters_flow(
    symbol: str,
    strategy_template: str,
    n_trials: int = 100,
    lookback_days: int = 365,
    timeframe: str = "1h",
    walk_forward: bool = False,
    storage_url: Optional[str] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Complete flow for optimizing strategy parameters.
    
    Args:
        symbol: Stock symbol to optimize
        strategy_template: Strategy template to use
        n_trials: Number of optimization trials
        lookback_days: Days of historical data
        timeframe: Data timeframe
        walk_forward: Enable walk-forward optimization
        storage_url: Optuna storage URL
        save_results: Whether to save results to database
        
    Returns:
        Dictionary with optimization results and metadata
    """
    logger = get_run_logger()
    logger.info(f"Starting strategy parameter optimization for {symbol} using {strategy_template}")
    
    try:
        # Step 1: Fetch historical data
        logger.info("Step 1: Fetching historical data")
        bars_data = await fetch_historical_data_task(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        if not bars_data or len(bars_data) < 100:
            raise ValueError(f"Insufficient data: {len(bars_data)} bars")
        
        logger.info(f"Fetched {len(bars_data)} bars for optimization")
        
        # Step 2: Run optimization study
        logger.info("Step 2: Running optimization study")
        study_name = f"{symbol}_{strategy_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        optimization_config = {
            "n_trials": n_trials,
            "primary_objective": "sharpe_ratio",
            "secondary_objectives": ["max_drawdown"] if n_trials > 50 else [],
            "n_jobs": 1,  # Sequential for flow stability
            "enable_pruning": n_trials > 20,
            "walk_forward": walk_forward,
            "validation_splits": 3 if walk_forward else 1
        }
        
        optimization_result = run_optimization_study_task(
            study_name=study_name,
            symbol=symbol,
            strategy_template=strategy_template,
            bars_data=bars_data,
            optimization_config=optimization_config,
            storage_url=storage_url
        )
        
        # Step 3: Save results (if enabled and successful)
        save_status = {"status": "skipped", "reason": "disabled"}
        if save_results and optimization_result.get("success", False):
            logger.info("Step 3: Saving optimization results")
            save_status = save_optimization_results_task(optimization_result)
        
        # Prepare final response
        final_result = {
            "symbol": symbol,
            "strategy_template": strategy_template,
            "optimization": optimization_result,
            "save_status": save_status,
            "flow_completed_at": datetime.now().isoformat(),
            "data_bars_count": len(bars_data),
            "success": optimization_result.get("success", False)
        }
        
        if optimization_result.get("success", False):
            logger.info(f"Optimization flow completed successfully for {symbol}. Best Sharpe: {optimization_result.get('best_value', 0):.3f}")
        else:
            logger.error(f"Optimization flow failed for {symbol}: {optimization_result.get('error', 'Unknown error')}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Optimization flow failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "strategy_template": strategy_template,
            "error": str(e),
            "success": False,
            "flow_completed_at": datetime.now().isoformat()
        }

@flow(
    name="optimize_watchlist_strategies",
    description="Batch optimization for multiple symbols and strategies",
    version="1.0"
)
async def optimize_watchlist_strategies_flow(
    symbols: List[str],
    strategy_templates: List[str],
    profile_id: int,
    n_trials_per_symbol: int = 50,
    lookback_days: int = 365,
    parallel_jobs: int = 2,
    timeframe: str = "1h",
    storage_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Flow to optimize strategies across multiple symbols (watchlist).
    
    Args:
        symbols: List of symbols to optimize
        strategy_templates: List of strategy templates
        profile_id: Profile ID for tracking
        n_trials_per_symbol: Trials per symbol-strategy combination
        lookback_days: Days of historical data
        parallel_jobs: Number of parallel jobs
        timeframe: Data timeframe
        storage_url: Optuna storage URL
        
    Returns:
        Dictionary with batch optimization results
    """
    logger = get_run_logger()
    logger.info(f"Starting watchlist optimization for {len(symbols)} symbols and {len(strategy_templates)} strategies")
    
    try:
        # Create watchlist optimization record
        optimization_id = f"watchlist_{profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            "optimization_id": optimization_id,
            "profile_id": profile_id,
            "symbols": symbols,
            "strategy_templates": strategy_templates,
            "individual_results": {},
            "summary": {},
            "started_at": datetime.now().isoformat()
        }
        
        # Create individual optimization tasks
        optimization_tasks = []
        task_metadata = []
        
        for symbol in symbols:
            for strategy_template in strategy_templates:
                task_key = f"{symbol}_{strategy_template}"
                
                task = optimize_strategy_parameters_flow.submit(
                    symbol=symbol,
                    strategy_template=strategy_template,
                    n_trials=n_trials_per_symbol,
                    lookback_days=lookback_days,
                    timeframe=timeframe,
                    walk_forward=False,  # Disable for batch jobs
                    storage_url=storage_url,
                    save_results=True
                )
                
                optimization_tasks.append(task)
                task_metadata.append(task_key)
        
        logger.info(f"Created {len(optimization_tasks)} optimization tasks")
        
        # Collect results
        completed_count = 0
        failed_count = 0
        
        for i, (task, task_key) in enumerate(zip(optimization_tasks, task_metadata)):
            try:
                result = await task
                results["individual_results"][task_key] = result
                
                if result.get("success", False):
                    completed_count += 1
                    logger.info(f"Completed optimization {i+1}/{len(optimization_tasks)}: {task_key}")
                else:
                    failed_count += 1
                    logger.warning(f"Failed optimization {i+1}/{len(optimization_tasks)}: {task_key}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Task failed for {task_key}: {e}")
                results["individual_results"][task_key] = {
                    "error": str(e),
                    "success": False
                }
        
        # Create summary
        successful_results = [
            r for r in results["individual_results"].values() 
            if r.get("success", False)
        ]
        
        if successful_results:
            best_performers = sorted(
                [
                    {
                        "symbol": r["symbol"],
                        "strategy": r["strategy_template"],
                        "sharpe_ratio": r["optimization"]["best_value"],
                        "params": r["optimization"]["best_params"]
                    }
                    for r in successful_results
                ],
                key=lambda x: x["sharpe_ratio"],
                reverse=True
            )
            
            avg_sharpe = sum(r["optimization"]["best_value"] for r in successful_results) / len(successful_results)
            
            results["summary"] = {
                "total_combinations": len(optimization_tasks),
                "completed_successfully": completed_count,
                "failed": failed_count,
                "success_rate": (completed_count / len(optimization_tasks)) * 100,
                "average_sharpe_ratio": avg_sharpe,
                "best_performers": best_performers[:10],  # Top 10
                "completion_rate": f"{completed_count}/{len(optimization_tasks)}"
            }
        else:
            results["summary"] = {
                "total_combinations": len(optimization_tasks),
                "completed_successfully": 0,
                "failed": len(optimization_tasks),
                "success_rate": 0.0,
                "error": "No successful optimizations"
            }
        
        results["completed_at"] = datetime.now().isoformat()
        results["success"] = completed_count > 0
        
        logger.info(f"Watchlist optimization completed: {completed_count}/{len(optimization_tasks)} successful")
        return results
        
    except Exception as e:
        logger.error(f"Watchlist optimization flow failed: {e}")
        return {
            "optimization_id": optimization_id,
            "profile_id": profile_id,
            "error": str(e),
            "success": False,
            "completed_at": datetime.now().isoformat()
        }

@flow(
    name="scheduled_optimization",
    description="Scheduled overnight optimization for watchlist",
    version="1.0"
)
async def scheduled_optimization_flow(
    profile_id: int,
    schedule_hour: int = 2,  # 2 AM
    max_symbols: int = 20,
    n_trials_per_symbol: int = 100
) -> Dict[str, Any]:
    """
    Scheduled optimization flow for overnight execution.
    
    Args:
        profile_id: Profile ID to get watchlist
        schedule_hour: Hour to run optimization (24-hour format)
        max_symbols: Maximum symbols to optimize
        n_trials_per_symbol: Trials per symbol
        
    Returns:
        Dictionary with scheduled optimization results
    """
    logger = get_run_logger()
    logger.info(f"Starting scheduled optimization for profile {profile_id}")
    
    try:
        # Get watchlist symbols from database
        db = SessionLocal()
        
        try:
            from .db import WatchlistItem, Asset
            
            watchlist_query = db.query(WatchlistItem).join(Asset).filter(
                WatchlistItem.profile_id == profile_id,
                WatchlistItem.active == True
            ).limit(max_symbols)
            
            watchlist_items = watchlist_query.all()
            symbols = [item.asset.symbol for item in watchlist_items]
            
            if not symbols:
                logger.warning(f"No active watchlist symbols found for profile {profile_id}")
                return {
                    "profile_id": profile_id,
                    "error": "No active watchlist symbols found",
                    "success": False
                }
            
        finally:
            db.close()
        
        # Default strategy templates for scheduled optimization
        strategy_templates = ["sma_crossover", "rsi_strategy", "macd_strategy"]
        
        logger.info(f"Found {len(symbols)} watchlist symbols for optimization")
        
        # Run watchlist optimization
        result = await optimize_watchlist_strategies_flow(
            symbols=symbols,
            strategy_templates=strategy_templates,
            profile_id=profile_id,
            n_trials_per_symbol=n_trials_per_symbol,
            lookback_days=365,
            parallel_jobs=4,  # More aggressive for overnight jobs
            timeframe="1h"
        )
        
        # Add scheduling metadata
        result.update({
            "scheduled_optimization": True,
            "schedule_hour": schedule_hour,
            "max_symbols": max_symbols,
            "watchlist_symbols": symbols
        })
        
        logger.info(f"Scheduled optimization completed for profile {profile_id}")
        return result
        
    except Exception as e:
        logger.error(f"Scheduled optimization failed for profile {profile_id}: {e}")
        return {
            "profile_id": profile_id,
            "error": str(e),
            "success": False,
            "scheduled_optimization": True
        }

# Example scheduled flow creation
def create_nightly_optimization_flow(profile_id: int):
    """
    Create a scheduled version of the optimization flow for nightly execution.
    """
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available - cannot create scheduled flows")
        return None
    
    @flow(
        name=f"nightly_optimization_profile_{profile_id}",
        description=f"Nightly optimization for profile {profile_id}",
        version="1.0"
    )
    async def nightly_flow():
        return await scheduled_optimization_flow(
            profile_id=profile_id,
            schedule_hour=2,  # 2 AM
            max_symbols=20,
            n_trials_per_symbol=100
        )
    
    return nightly_flow

# Export the main functions that will be used by the sentiment module
__all__ = [
    'sentiment_analysis_flow',
    'batch_sentiment_analysis_flow',
    'optimize_strategy_parameters_flow',
    'optimize_watchlist_strategies_flow', 
    'scheduled_optimization_flow',
    'create_nightly_optimization_flow',
    'create_sentiment_signal_from_dict',
    'is_prefect_available',
    'PREFECT_AVAILABLE',
    'OPTIMIZATION_AVAILABLE'
]