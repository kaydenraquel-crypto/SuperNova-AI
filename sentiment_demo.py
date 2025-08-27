#!/usr/bin/env python3
"""SuperNova Sentiment Analysis Demonstration

This script demonstrates the comprehensive sentiment analysis capabilities
of the enhanced SuperNova framework, including:

1. Basic lexicon-based sentiment scoring
2. Advanced NLP models (when available)
3. Social media integration (when configured)
4. News sentiment analysis (when configured)
5. Signal blending and aggregation
6. Financial entity recognition
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from supernova.sentiment import (
    score_text,
    score_text_basic,
    detect_tickers,
    SentimentSource,
    MarketRegime,
    TwitterConnector,
    RedditConnector,
    NewsConnector,
    generate_sentiment_signal,
    FIN_LEX,
    PUBLIC_FIGS
)

def demo_basic_sentiment():
    """Demonstrate basic sentiment analysis functionality"""
    print("=" * 60)
    print("BASIC SENTIMENT ANALYSIS DEMO")
    print("=" * 60)
    
    test_texts = [
        "Apple beats Q3 earnings expectations with record revenue growth!",
        "Tesla stock plunges after disappointing delivery numbers and downgrade",
        "Warren Buffett comments on the bullish market outlook for 2024",
        "Fed investigation into major bank causes regulatory concerns",
        "Microsoft announces breakthrough AI partnership with strong profit margins",
        "Market volatility increases amid recession fears and inflation worries"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        result = score_text_basic(text)
        
        print(f"   Score: {result.score:+.3f} (confidence: {result.confidence:.3f})")
        print(f"   Figures: {list(result.figures.keys())}")
        
        # Detect tickers
        tickers = detect_tickers(text)
        if tickers:
            print(f"   Tickers: {tickers}")
        
        sentiment_label = "POSITIVE" if result.score > 0.1 else "NEGATIVE" if result.score < -0.1 else "NEUTRAL"
        print(f"   Overall: {sentiment_label}")

def demo_advanced_sentiment():
    """Demonstrate advanced sentiment analysis with multiple models"""
    print("\n" + "=" * 60)
    print("ADVANCED SENTIMENT ANALYSIS DEMO")
    print("=" * 60)
    
    test_texts = [
        "NVIDIA surges on exceptional AI chip demand and bullish guidance",
        "Banking sector faces headwinds from rising defaults and investigations",
        "Jerome Powell signals dovish stance amid economic recovery signs"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        result = score_text(text, use_advanced=True)
        
        print(f"   Score: {result.score:+.3f} (confidence: {result.confidence:.3f})")
        print(f"   Source: {result.source}")
        print(f"   Figures: {list(result.figures.keys())}")
        print(f"   Raw scores: {result.raw_scores}")
        print(f"   Entities: {result.entities}")
        
        if result.metadata:
            print(f"   Metadata: {result.metadata}")

def demo_lexicon_stats():
    """Show statistics about the financial lexicon"""
    print("\n" + "=" * 60)
    print("FINANCIAL LEXICON STATISTICS")
    print("=" * 60)
    
    positive_words = [word for word, score in FIN_LEX.items() if score > 0]
    negative_words = [word for word, score in FIN_LEX.items() if score < 0]
    neutral_words = [word for word, score in FIN_LEX.items() if score == 0]
    
    print(f"Total terms: {len(FIN_LEX)}")
    print(f"Positive terms: {len(positive_words)} ({len(positive_words)/len(FIN_LEX)*100:.1f}%)")
    print(f"Negative terms: {len(negative_words)} ({len(negative_words)/len(FIN_LEX)*100:.1f}%)")
    print(f"Neutral terms: {len(neutral_words)}")
    
    print(f"\nMost positive terms:")
    sorted_positive = sorted([(word, score) for word, score in FIN_LEX.items() if score > 0], 
                           key=lambda x: x[1], reverse=True)[:10]
    for word, score in sorted_positive:
        print(f"  {word}: +{score:.1f}")
    
    print(f"\nMost negative terms:")
    sorted_negative = sorted([(word, score) for word, score in FIN_LEX.items() if score < 0], 
                           key=lambda x: x[1])[:10]
    for word, score in sorted_negative:
        print(f"  {word}: {score:.1f}")
    
    print(f"\nPublic figures tracked: {len(PUBLIC_FIGS)}")
    sorted_figures = sorted(PUBLIC_FIGS.items(), key=lambda x: x[1], reverse=True)
    for name, influence in sorted_figures[:10]:
        print(f"  {name}: {influence:.1f}")

async def demo_social_media_integration():
    """Demonstrate social media integration (if configured)"""
    print("\n" + "=" * 60)
    print("SOCIAL MEDIA INTEGRATION DEMO")
    print("=" * 60)
    
    # Test Twitter connector
    twitter = TwitterConnector()
    print("Testing Twitter connector...")
    if twitter.bearer_token:
        print("✓ Twitter API configured")
        try:
            tweets = await twitter.fetch_recent_tweets("AAPL", max_results=5, since_hours=24)
            print(f"  Fetched {len(tweets)} tweets")
            if tweets:
                print("  Sample tweet:", tweets[0]['text'][:100] + "...")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("✗ Twitter API not configured (set X_BEARER_TOKEN)")
    
    # Test Reddit connector
    reddit = RedditConnector()
    print("\nTesting Reddit connector...")
    if reddit.client_id and reddit.client_secret:
        print("✓ Reddit API configured")
        try:
            posts = await reddit.fetch_subreddit_posts("stocks", "AAPL", max_results=5)
            print(f"  Fetched {len(posts)} posts")
            if posts:
                print("  Sample post:", posts[0]['title'])
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("✗ Reddit API not configured (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)")
    
    # Test News connector
    news = NewsConnector()
    print("\nTesting News connector...")
    try:
        articles = await news.fetch_news_headlines("Apple stock", max_articles=5)
        print(f"  Fetched {len(articles)} articles")
        if articles:
            print("  Sample article:", articles[0].get('title', 'No title'))
    except Exception as e:
        print(f"  Error: {e}")

async def demo_comprehensive_signal():
    """Demonstrate comprehensive sentiment signal generation"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SENTIMENT SIGNAL DEMO")
    print("=" * 60)
    
    symbol = "AAPL"
    print(f"Generating sentiment signal for {symbol}...")
    
    try:
        signal = await generate_sentiment_signal(
            symbol=symbol,
            lookback_hours=24,
            market_regime=MarketRegime.SIDEWAYS
        )
        
        print(f"\n📊 SENTIMENT SIGNAL FOR {symbol}")
        print(f"   Overall Score: {signal.overall_score:+.3f}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Figure Influence: {signal.figure_influence:+.3f}")
        print(f"   News Impact: {signal.news_impact:+.3f}")
        print(f"   Social Momentum: {signal.social_momentum:+.3f}")
        print(f"   Contrarian Indicator: {signal.contrarian_indicator:+.3f}")
        print(f"   Regime Adjusted: {signal.regime_adjusted_score:+.3f}")
        print(f"   Timestamp: {signal.timestamp}")
        print(f"   Valid Until: {signal.timestamp + signal.validity_period}")
        
        if signal.source_breakdown:
            print("\n📈 Source Breakdown:")
            for source, score in signal.source_breakdown.items():
                print(f"   {source.value}: {score:+.3f}")
        
    except Exception as e:
        print(f"Error generating signal: {e}")

def demo_integration_with_advisor():
    """Demonstrate how sentiment integrates with the advisor system"""
    print("\n" + "=" * 60)
    print("ADVISOR INTEGRATION DEMO")
    print("=" * 60)
    
    # This would show how sentiment hints are used in the advisor
    print("Example of sentiment integration with trading advice:")
    
    scenarios = [
        ("Positive sentiment + bullish technical", 0.7, "BUY", "Strong positive sentiment reinforces technical signals"),
        ("Negative sentiment + bearish technical", -0.8, "SELL", "Negative sentiment confirms technical weakness"),
        ("Mixed sentiment + neutral technical", 0.1, "HOLD", "Conflicting signals suggest caution"),
        ("Extreme positive sentiment", 0.9, "SELL", "Contrarian signal - extreme optimism may signal top")
    ]
    
    for scenario, sentiment_score, expected_action, reason in scenarios:
        print(f"\n• {scenario}:")
        print(f"  Sentiment Score: {sentiment_score:+.2f}")
        print(f"  Suggested Action: {expected_action}")
        print(f"  Reasoning: {reason}")

async def main():
    """Run all demonstrations"""
    print("🚀 SuperNova Sentiment Analysis System Demo")
    print("=" * 60)
    
    # Basic demonstrations (always work)
    demo_basic_sentiment()
    demo_advanced_sentiment()
    demo_lexicon_stats()
    
    # API-dependent demonstrations (may not work without configuration)
    await demo_social_media_integration()
    await demo_comprehensive_signal()
    
    # Integration demonstration
    demo_integration_with_advisor()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("=" * 60)
    print("\nTo enable full functionality:")
    print("1. Copy .env.example to .env")
    print("2. Configure API keys for Twitter, Reddit, and News APIs")
    print("3. Install additional dependencies:")
    print("   pip install tweepy praw spacy textblob vaderSentiment")
    print("   python -m spacy download en_core_web_sm")
    print("4. For FinBERT: pip install transformers torch")

if __name__ == "__main__":
    asyncio.run(main())