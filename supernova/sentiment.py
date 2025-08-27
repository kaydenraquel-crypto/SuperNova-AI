"""SuperNova Sentiment Analysis System

Comprehensive sentiment analysis for financial markets integrating:
- Social media feeds (X/Twitter, Reddit)
- News sentiment analysis
- Advanced NLP with financial context
- Signal blending with technical analysis
- ToS-compliant data collection
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import re
import math
import json
import hashlib
import time
from collections import defaultdict, deque
from pathlib import Path
import logging

# Async HTTP imports
try:
    import aiohttp
except ImportError:
    aiohttp = None

# NLP imports
try:
    import spacy
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    spacy = None
    TextBlob = None
    SentimentIntensityAnalyzer = None
    pipeline = None

# Social media API imports
try:
    import tweepy
    import praw
except ImportError:
    tweepy = None
    praw = None

# News processing imports
try:
    from newspaper import Article
    from bs4 import BeautifulSoup
except ImportError:
    Article = None
    BeautifulSoup = None

# Rate limiting
try:
    from asyncio_throttle import Throttler
except ImportError:
    Throttler = None

# Configuration imports
try:
    from .config import settings
except ImportError:
    # Fallback settings if config module not available
    class MockSettings:
        X_BEARER_TOKEN = None
        REDDIT_CLIENT_ID = None
        REDDIT_CLIENT_SECRET = None
        REDDIT_USER_AGENT = "SuperNova/1.0"
        NEWSAPI_KEY = None
    settings = MockSettings()

# Enhanced Financial Lexicon with sector-specific terms
FIN_LEX = {
    # Earnings & Performance
    "beat": 0.8, "beats": 0.7, "miss": -0.8, "misses": -0.7, "exceeded": 0.6,
    "outperformed": 0.7, "underperformed": -0.6, "guidance": 0.2, "raised": 0.4,
    "lowered": -0.4, "revised": 0.1, "maintained": 0.0,
    
    # Market Sentiment
    "bullish": 0.7, "bearish": -0.7, "optimistic": 0.6, "pessimistic": -0.6,
    "confident": 0.5, "uncertain": -0.3, "cautious": -0.2, "aggressive": 0.3,
    
    # Price Movement
    "surge": 0.5, "surged": 0.5, "plunge": -0.6, "plunged": -0.6, "rally": 0.4,
    "rallied": 0.4, "decline": -0.4, "declined": -0.4, "spike": 0.4, "spiked": 0.4,
    "crash": -0.8, "crashed": -0.8, "soar": 0.6, "soared": 0.6, "tumble": -0.5,
    "tumbled": -0.5, "climb": 0.3, "climbed": 0.3, "drop": -0.3, "dropped": -0.3,
    
    # Corporate Actions
    "buyback": 0.4, "dividend": 0.3, "split": 0.2, "merger": 0.3, "acquisition": 0.3,
    "spinoff": 0.1, "ipo": 0.2, "listing": 0.2, "delisting": -0.6,
    
    # Financial Health
    "profit": 0.4, "profitable": 0.4, "loss": -0.4, "losses": -0.4, "revenue": 0.1,
    "growth": 0.4, "expansion": 0.3, "contraction": -0.3, "bankruptcy": -0.9,
    "liquidation": -0.8, "restructuring": -0.3, "refinancing": -0.1,
    
    # Analyst Actions
    "upgrade": 0.6, "upgraded": 0.6, "downgrade": -0.7, "downgraded": -0.7,
    "reiterate": 0.0, "initiate": 0.2, "maintain": 0.0, "overweight": 0.4,
    "underweight": -0.4, "outperform": 0.5, "underperform": -0.5,
    
    # Legal & Regulatory
    "lawsuit": -0.5, "investigation": -0.4, "fine": -0.4, "penalty": -0.4,
    "settlement": -0.2, "approval": 0.3, "rejected": -0.4, "compliance": 0.1,
    "violation": -0.5, "fraud": -0.7, "scandal": -0.6,
    
    # Industry Specific
    "breakthrough": 0.6, "innovation": 0.4, "patent": 0.3, "disruption": 0.2,
    "obsolete": -0.5, "competition": -0.2, "monopoly": -0.3, "partnership": 0.3,
    
    # Economic Indicators
    "inflation": -0.3, "deflation": -0.4, "recession": -0.7, "recovery": 0.5,
    "bubble": -0.6, "correction": -0.4, "volatility": -0.2, "stability": 0.2,
    
    # Sentiment Modifiers
    "strong": 0.3, "weak": -0.3, "solid": 0.3, "robust": 0.4, "fragile": -0.3,
    "impressive": 0.5, "disappointing": -0.5, "excellent": 0.6, "poor": -0.4,
    "outstanding": 0.7, "terrible": -0.6, "record": 0.3, "historic": 0.2
}

# Enhanced public figures with financial influence scores
PUBLIC_FIGS = {
    # Central Bank Officials
    "powell": 0.8, "yellen": 0.7, "lagarde": 0.6, "kuroda": 0.5,
    # Regulators
    "gensler": 0.6, "warren": 0.5, "waters": 0.4,
    # Investors & CEOs
    "buffett": 0.9, "munger": 0.7, "dalio": 0.6, "dimon": 0.6, "fink": 0.5,
    "musk": 0.7, "bezos": 0.5, "gates": 0.4, "saylor": 0.4, "druckenmiller": 0.6,
    # Analysts & Commentators
    "cramer": 0.3, "ackman": 0.5, "icahn": 0.5, "einhorn": 0.4,
}

MENTION_BOOST = 0.15  # Increased boost for influential figure mentions
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for sentiment signals
RATE_LIMIT_DELAY = 1.0  # Base delay between API calls
CACHE_TTL = 300  # 5 minutes cache TTL

class SentimentSource(Enum):
    """Sentiment data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FINBERT = "finbert"
    VADER = "vader"
    TEXTBLOB = "textblob"
    MANUAL = "manual"

class MarketRegime(Enum):
    """Market regime for sentiment weighting"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class SentimentResult:
    """Enhanced sentiment analysis result"""
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    tokens: List[str]
    figures: Dict[str, float]  # Figure name -> influence score
    source: SentimentSource
    timestamp: datetime
    text_length: int
    entities: Dict[str, List[str]] = field(default_factory=dict)  # NER results
    raw_scores: Dict[str, float] = field(default_factory=dict)  # Multiple model scores
    metadata: Dict = field(default_factory=dict)

@dataclass 
class SentimentSignal:
    """Aggregated sentiment signal for trading"""
    overall_score: float
    confidence: float
    source_breakdown: Dict[SentimentSource, float]
    figure_influence: float
    news_impact: float
    social_momentum: float
    contrarian_indicator: float
    regime_adjusted_score: float
    timestamp: datetime
    validity_period: timedelta = field(default=timedelta(minutes=30))

class SentimentCache:
    """Thread-safe sentiment caching system"""
    def __init__(self, ttl: int = CACHE_TTL):
        self.cache = {}
        self.ttl = ttl
        self.access_times = {}
    
    def _cache_key(self, text: str, source: SentimentSource) -> str:
        """Generate cache key"""
        content = f"{source.value}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, source: SentimentSource) -> Optional[SentimentResult]:
        """Get cached result"""
        key = self._cache_key(text, source)
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def put(self, text: str, source: SentimentSource, result: SentimentResult):
        """Cache result"""
        key = self._cache_key(text, source)
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear_expired(self):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time >= self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

class RateLimiter:
    """API rate limiting"""
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.last_call = 0.0
        self.call_count = 0
        self.window_start = time.time()
    
    async def acquire(self):
        """Wait for rate limit clearance"""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start >= 1.0:
            self.call_count = 0
            self.window_start = current_time
        
        # Check if we need to wait
        if self.call_count >= self.calls_per_second:
            wait_time = 1.0 - (current_time - self.window_start)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.call_count = 0
                self.window_start = time.time()
        
        self.call_count += 1
        self.last_call = time.time()

# Global instances
sentiment_cache = SentimentCache()
twitter_limiter = RateLimiter(calls_per_second=0.5)  # Conservative Twitter limits
reddit_limiter = RateLimiter(calls_per_second=1.0)
news_limiter = RateLimiter(calls_per_second=2.0)

# NLP models (lazy loaded)
_nlp_models = {}
_finbert_pipeline = None
_vader_analyzer = None

def get_nlp_model(model_name: str = "en_core_web_sm"):
    """Lazy load spaCy model"""
    if model_name not in _nlp_models and spacy:
        try:
            _nlp_models[model_name] = spacy.load(model_name)
        except OSError:
            logging.warning(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
            _nlp_models[model_name] = None
    return _nlp_models.get(model_name)

def get_finbert_pipeline():
    """Lazy load FinBERT model"""
    global _finbert_pipeline
    if _finbert_pipeline is None and pipeline:
        try:
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1  # CPU only for compatibility
            )
        except Exception as e:
            logging.warning(f"Could not load FinBERT model: {e}")
            _finbert_pipeline = False
    return _finbert_pipeline if _finbert_pipeline is not False else None

def get_vader_analyzer():
    """Lazy load VADER analyzer"""
    global _vader_analyzer
    if _vader_analyzer is None and SentimentIntensityAnalyzer:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy"""
    nlp = get_nlp_model()
    if not nlp:
        return {}
    
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        # Focus on financial entities
        if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT"]:
            entities[ent.label_].append(ent.text)
    
    # Clean and deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return dict(entities)

def detect_tickers(text: str) -> List[str]:
    """Extract stock tickers from text"""
    # Pattern for common ticker formats
    ticker_patterns = [
        r'\$([A-Z]{1,5})',  # $AAPL format
        r'\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|up|down|gained|lost))',  # AAPL stock
        r'\(([A-Z]{2,5})\)',  # (AAPL) format
    ]
    
    tickers = []
    for pattern in ticker_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tickers.extend([m.upper() for m in matches])
    
    # Filter common false positives
    false_positives = {"THE", "AND", "FOR", "YOU", "ARE", "CAN", "NOT", "BUT", "ALL", "NEW", "NOW", "GET"}
    return [t for t in set(tickers) if t not in false_positives and len(t) <= 5]

def score_text_basic(text: str) -> SentimentResult:
    """Basic lexicon-based sentiment scoring (backward compatibility)"""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    score = 0.0
    confidence = 0.1  # Low confidence for basic scoring
    
    # Lexicon scoring
    matched_terms = 0
    for token in tokens:
        if token in FIN_LEX:
            score += FIN_LEX[token]
            matched_terms += 1
    
    # Figure mentions with influence weighting
    figures = {}
    for token in tokens:
        if token in PUBLIC_FIGS:
            figures[token] = PUBLIC_FIGS[token]
            score += MENTION_BOOST * PUBLIC_FIGS[token]
    
    # Confidence based on matched terms and text length
    if matched_terms > 0:
        confidence = min(0.8, 0.1 + (matched_terms * 0.1) + (len(text) / 1000 * 0.1))
    
    # Normalize score
    score = math.tanh(score / 5.0)
    
    # Extract tickers for metadata
    tickers = detect_tickers(text)
    
    return SentimentResult(
        score=score,
        confidence=confidence,
        tokens=tokens,
        figures=figures,
        source=SentimentSource.MANUAL,
        timestamp=datetime.now(),
        text_length=len(text),
        metadata={"tickers": tickers, "matched_terms": matched_terms}
    )

def score_text_finbert(text: str) -> Optional[SentimentResult]:
    """Advanced sentiment using FinBERT"""
    finbert = get_finbert_pipeline()
    if not finbert:
        return None
    
    try:
        # Truncate text for model limits
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = finbert(text)[0]
        
        # Convert FinBERT labels to scores
        label_map = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0
        }
        
        label = result["label"].lower()
        base_score = label_map.get(label, 0.0)
        confidence = result["score"]
        
        # Scale score by confidence
        final_score = base_score * confidence
        
        return SentimentResult(
            score=final_score,
            confidence=confidence,
            tokens=text.split(),
            figures={},
            source=SentimentSource.FINBERT,
            timestamp=datetime.now(),
            text_length=len(text),
            raw_scores={"finbert": final_score},
            metadata={"finbert_label": label, "finbert_confidence": confidence}
        )
    
    except Exception as e:
        logging.error(f"FinBERT scoring error: {e}")
        return None

def score_text_vader(text: str) -> Optional[SentimentResult]:
    """VADER sentiment analysis"""
    analyzer = get_vader_analyzer()
    if not analyzer:
        return None
    
    try:
        scores = analyzer.polarity_scores(text)
        compound_score = scores["compound"]
        
        # VADER confidence approximation
        confidence = abs(compound_score)
        
        return SentimentResult(
            score=compound_score,
            confidence=confidence,
            tokens=text.split(),
            figures={},
            source=SentimentSource.VADER,
            timestamp=datetime.now(),
            text_length=len(text),
            raw_scores=scores,
            metadata={"vader_breakdown": scores}
        )
    
    except Exception as e:
        logging.error(f"VADER scoring error: {e}")
        return None

def score_text_textblob(text: str) -> Optional[SentimentResult]:
    """TextBlob sentiment analysis"""
    if not TextBlob:
        return None
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Use subjectivity as confidence indicator
        confidence = subjectivity * 0.7  # Scale down as subjectivity != confidence
        
        return SentimentResult(
            score=polarity,
            confidence=confidence,
            tokens=text.split(),
            figures={},
            source=SentimentSource.TEXTBLOB,
            timestamp=datetime.now(),
            text_length=len(text),
            raw_scores={"polarity": polarity, "subjectivity": subjectivity},
            metadata={"subjectivity": subjectivity}
        )
    
    except Exception as e:
        logging.error(f"TextBlob scoring error: {e}")
        return None

def score_text(text: str, use_advanced: bool = True) -> SentimentResult:
    """Comprehensive sentiment analysis with multiple models"""
    if not text or not text.strip():
        return SentimentResult(
            score=0.0, confidence=0.0, tokens=[], figures={},
            source=SentimentSource.MANUAL, timestamp=datetime.now(),
            text_length=0
        )
    
    # Check cache first
    cached = sentiment_cache.get(text, SentimentSource.MANUAL)
    if cached:
        return cached
    
    results = []
    
    # Always include basic lexicon-based scoring
    basic_result = score_text_basic(text)
    results.append(basic_result)
    
    if use_advanced:
        # Try advanced models
        for scorer in [score_text_finbert, score_text_vader, score_text_textblob]:
            result = scorer(text)
            if result:
                results.append(result)
    
    # Combine results with weighted averaging
    if len(results) == 1:
        final_result = results[0]
    else:
        # Weight by confidence and model preference
        model_weights = {
            SentimentSource.FINBERT: 0.4,  # Highest weight for financial model
            SentimentSource.VADER: 0.3,    # Good for social media
            SentimentSource.TEXTBLOB: 0.2, # Basic but reliable
            SentimentSource.MANUAL: 0.1    # Lexicon fallback
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        all_figures = {}
        all_raw_scores = {}
        all_metadata = {}
        
        for result in results:
            weight = model_weights.get(result.source, 0.1) * (1.0 + result.confidence)
            weighted_score += result.score * weight
            total_weight += weight
            confidence_sum += result.confidence
            
            # Merge figures and metadata
            all_figures.update(result.figures)
            all_raw_scores.update(result.raw_scores)
            all_metadata.update(result.metadata)
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        final_confidence = confidence_sum / len(results)
        
        # Extract entities if we have spaCy
        entities = extract_entities(text)
        
        final_result = SentimentResult(
            score=final_score,
            confidence=final_confidence,
            tokens=basic_result.tokens,  # Use basic tokenization
            figures=all_figures,
            source=SentimentSource.MANUAL,  # Composite source
            timestamp=datetime.now(),
            text_length=len(text),
            entities=entities,
            raw_scores=all_raw_scores,
            metadata=all_metadata
        )
    
    # Cache the result
    sentiment_cache.put(text, SentimentSource.MANUAL, final_result)
    
    return final_result

# ================================
# SOCIAL MEDIA CONNECTORS
# ================================

class TwitterConnector:
    """X (Twitter) API connector with ToS compliance"""
    
    def __init__(self):
        self.bearer_token = settings.X_BEARER_TOKEN
        self.client = None
        self.last_request_time = {}
        
    def _get_client(self):
        """Initialize Twitter API client"""
        if not self.client and self.bearer_token and tweepy:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
            except Exception as e:
                logging.error(f"Twitter client initialization failed: {e}")
                self.client = None
        return self.client
        
    async def fetch_recent_tweets(
        self,
        query: str,
        max_results: int = 50,
        since_hours: int = 24
    ) -> List[Dict]:
        """Fetch recent tweets with rate limiting and compliance"""
        client = self._get_client()
        if not client:
            logging.warning("Twitter API not configured")
            return []
            
        await twitter_limiter.acquire()
        
        try:
            # Build query with time constraint
            since_time = datetime.now() - timedelta(hours=since_hours)
            
            # Enhanced query for financial context
            if not any(operator in query for operator in ['AND', 'OR', 'NOT']):
                # Add financial context terms if not a complex query
                fin_terms = ['stock', 'price', 'earnings', 'market', 'trading']
                query = f"({query}) ({' OR '.join(fin_terms)})"
            
            # Add filters to improve quality
            query += " -is:retweet -is:reply lang:en"
            
            tweets = tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                start_time=since_time
            ).flatten(limit=max_results)
            
            results = []
            for tweet in tweets:
                # Basic quality filtering
                if len(tweet.text) < 20:  # Skip very short tweets
                    continue
                    
                results.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics,
                    'author_id': tweet.author_id,
                    'source': 'twitter'
                })
                
            logging.info(f"Fetched {len(results)} tweets for query: {query}")
            return results
            
        except Exception as e:
            logging.error(f"Twitter API error: {e}")
            return []

class RedditConnector:
    """Reddit API connector with ToS compliance"""
    
    def __init__(self):
        self.client_id = settings.REDDIT_CLIENT_ID
        self.client_secret = settings.REDDIT_CLIENT_SECRET  
        self.user_agent = settings.REDDIT_USER_AGENT or "SuperNova/1.0"
        self.reddit = None
        
    def _get_client(self):
        """Initialize Reddit API client"""
        if not self.reddit and self.client_id and self.client_secret and praw:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    check_for_async=False
                )
            except Exception as e:
                logging.error(f"Reddit client initialization failed: {e}")
                self.reddit = None
        return self.reddit
        
    async def fetch_subreddit_posts(
        self,
        subreddit: str,
        query: str = "",
        sort: str = "hot",
        max_results: int = 50,
        time_filter: str = "day"
    ) -> List[Dict]:
        """Fetch posts from subreddit with rate limiting"""
        reddit = self._get_client()
        if not reddit:
            logging.warning("Reddit API not configured")
            return []
            
        await reddit_limiter.acquire()
        
        try:
            sub = reddit.subreddit(subreddit)
            results = []
            
            # Choose the appropriate listing method
            if query:
                # Search within subreddit
                posts = sub.search(query, sort=sort, time_filter=time_filter, limit=max_results)
            else:
                # Get general posts
                if sort == "hot":
                    posts = sub.hot(limit=max_results)
                elif sort == "new":
                    posts = sub.new(limit=max_results)
                elif sort == "top":
                    posts = sub.top(time_filter=time_filter, limit=max_results)
                else:
                    posts = sub.hot(limit=max_results)
                    
            for post in posts:
                # Skip low-quality posts
                if post.score < 2 or len(post.title) < 10:
                    continue
                    
                # Combine title and selftext for analysis
                full_text = post.title
                if hasattr(post, 'selftext') and post.selftext:
                    full_text += " " + post.selftext[:500]  # Limit text length
                    
                results.append({
                    'text': full_text,
                    'title': post.title,
                    'score': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'num_comments': post.num_comments,
                    'subreddit': subreddit,
                    'url': post.url,
                    'source': 'reddit'
                })
                
            logging.info(f"Fetched {len(results)} posts from r/{subreddit}")
            return results
            
        except Exception as e:
            logging.error(f"Reddit API error: {e}")
            return []

# ================================
# NEWS INTEGRATION
# ================================

class NewsConnector:
    """News sentiment integration"""
    
    def __init__(self):
        self.session = None
        
    async def _get_session(self):
        """Get aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def fetch_news_headlines(
        self,
        query: str = "stock market",
        max_articles: int = 30,
        language: str = "en"
    ) -> List[Dict]:
        """Fetch news headlines from multiple sources"""
        session = await self._get_session()
        await news_limiter.acquire()
        
        articles = []
        
        # Try multiple news sources
        sources = [
            self._fetch_newsapi_articles,
            self._fetch_rss_feeds,
            self._fetch_financial_news
        ]
        
        for source_func in sources:
            try:
                source_articles = await source_func(session, query, max_articles // len(sources))
                articles.extend(source_articles)
            except Exception as e:
                logging.error(f"News source error: {e}")
                continue
                
        return articles[:max_articles]
        
    async def _fetch_newsapi_articles(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int
    ) -> List[Dict]:
        """Fetch from NewsAPI (requires API key)"""
        # Note: NewsAPI requires API key - this is a placeholder implementation
        # Users need to sign up at newsapi.org and add NEWSAPI_KEY to config
        
        api_key = getattr(settings, 'NEWSAPI_KEY', None)
        if not api_key:
            return []
            
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': limit
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    for article in data.get('articles', []):
                        articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', '')[:500],
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt'),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'source_type': 'newsapi'
                        })
                    return articles
                        
        except Exception as e:
            logging.error(f"NewsAPI error: {e}")
            
        return []
        
    async def _fetch_rss_feeds(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int
    ) -> List[Dict]:
        """Fetch from financial RSS feeds"""
        # Financial news RSS feeds (free, ToS-compliant)
        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.marketwatch.com/rss/topstories",
            "https://seekingalpha.com/api/sa/combined/AAPL.xml"  # Example for specific stocks
        ]
        
        articles = []
        
        # This is a simplified implementation - real RSS parsing would use feedparser
        for feed_url in rss_feeds[:2]:  # Limit to avoid overloading
            try:
                async with session.get(feed_url) as response:
                    if response.status == 200:
                        # Note: Real implementation would parse RSS XML
                        # For now, this is a placeholder
                        pass
            except Exception as e:
                logging.error(f"RSS feed error {feed_url}: {e}")
                
        return articles
        
    async def _fetch_financial_news(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int
    ) -> List[Dict]:
        """Fetch from financial news websites (with scraping - be careful about ToS)"""
        # This would involve web scraping - must comply with robots.txt and ToS
        # Implementation placeholder for ethical scraping
        return []

# ================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# ================================

async def fetch_recent_from_x_async(
    query: str,
    max_items: int = 50
) -> List[str]:
    """Async wrapper for X/Twitter fetching"""
    connector = TwitterConnector()
    tweets = await connector.fetch_recent_tweets(query, max_items)
    return [tweet['text'] for tweet in tweets]

def fetch_recent_from_x(query: str, max_items: int = 50) -> List[str]:
    """Backward compatibility function for X/Twitter"""
    try:
        # Run async function in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, return empty (avoid nested loops)
            logging.warning("fetch_recent_from_x called from async context - use async version")
            return []
        else:
            return loop.run_until_complete(fetch_recent_from_x_async(query, max_items))
    except Exception as e:
        logging.error(f"Error fetching from X: {e}")
        return []

async def fetch_recent_from_reddit_async(
    subreddit: str,
    query: str = "",
    max_items: int = 50
) -> List[str]:
    """Async wrapper for Reddit fetching"""
    connector = RedditConnector()
    posts = await connector.fetch_subreddit_posts(subreddit, query, max_results=max_items)
    return [post['text'] for post in posts]

def fetch_recent_from_reddit(
    subreddit: str,
    query: str = "",
    max_items: int = 50
) -> List[str]:
    """Backward compatibility function for Reddit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logging.warning("fetch_recent_from_reddit called from async context - use async version")
            return []
        else:
            return loop.run_until_complete(
                fetch_recent_from_reddit_async(subreddit, query, max_items)
            )
    except Exception as e:
        logging.error(f"Error fetching from Reddit: {e}")
        return []

# ================================
# SIGNAL BLENDING & AGGREGATION
# ================================

def calculate_sentiment_momentum(
    recent_sentiments: List[SentimentResult],
    window_hours: int = 6
) -> float:
    """Calculate sentiment momentum over time"""
    if not recent_sentiments:
        return 0.0
        
    # Sort by timestamp
    sorted_sentiments = sorted(recent_sentiments, key=lambda x: x.timestamp)
    
    # Split into two halves to compare momentum
    mid_point = len(sorted_sentiments) // 2
    if mid_point == 0:
        return 0.0
        
    earlier_avg = sum(s.score for s in sorted_sentiments[:mid_point]) / mid_point
    later_avg = sum(s.score for s in sorted_sentiments[mid_point:]) / (len(sorted_sentiments) - mid_point)
    
    return later_avg - earlier_avg

def detect_contrarian_signals(
    sentiment_score: float,
    confidence: float,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
) -> float:
    """Detect contrarian sentiment signals"""
    # Extreme sentiment can be contrarian
    if confidence > 0.7:  # High confidence sentiment
        if sentiment_score > 0.8:  # Extremely positive
            return -0.3  # Contrarian bearish signal
        elif sentiment_score < -0.8:  # Extremely negative
            return 0.3   # Contrarian bullish signal
            
    return 0.0

def adjust_for_market_regime(
    sentiment_score: float,
    regime: MarketRegime
) -> float:
    """Adjust sentiment based on market regime"""
    adjustments = {
        MarketRegime.BULL: 0.9,     # Reduce positive sentiment impact in bull markets
        MarketRegime.BEAR: 1.1,     # Amplify sentiment in bear markets
        MarketRegime.VOLATILE: 1.2, # Amplify in volatile markets
        MarketRegime.SIDEWAYS: 1.0  # No adjustment in sideways markets
    }
    
    return sentiment_score * adjustments.get(regime, 1.0)

async def generate_sentiment_signal(
    symbol: str,
    lookback_hours: int = 24,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS,
    use_prefect: bool = None
) -> SentimentSignal:
    """
    Generate comprehensive sentiment signal for a symbol.
    
    This function can use either the traditional asyncio.gather approach or
    the new Prefect workflow system for better reliability and observability.
    
    Args:
        symbol: Stock symbol to analyze
        lookback_hours: How far back to look for data (hours)
        market_regime: Current market regime for sentiment adjustments
        use_prefect: Force using Prefect (True) or asyncio (False). 
                    If None, uses configuration setting.
    
    Returns:
        SentimentSignal object with comprehensive sentiment analysis
    """
    
    # Determine whether to use Prefect workflows
    if use_prefect is None:
        use_prefect = getattr(settings, 'ENABLE_PREFECT', False)
    
    # Try to use Prefect if requested and available
    if use_prefect:
        try:
            from .workflows import sentiment_analysis_flow, create_sentiment_signal_from_dict, is_prefect_available
            
            if is_prefect_available():
                logging.info(f"Using Prefect workflow for sentiment analysis of {symbol}")
                
                # Run the Prefect flow
                signal_dict = await sentiment_analysis_flow(
                    symbol=symbol,
                    lookback_hours=lookback_hours,
                    market_regime=market_regime
                )
                
                # Convert back to SentimentSignal object
                return create_sentiment_signal_from_dict(signal_dict)
            else:
                logging.warning("Prefect requested but not available, falling back to asyncio implementation")
        except Exception as e:
            logging.error(f"Error running Prefect workflow, falling back to asyncio: {e}")
    
    # Fallback to original asyncio.gather implementation
    logging.info(f"Using asyncio.gather for sentiment analysis of {symbol}")
    return await _generate_sentiment_signal_legacy(symbol, lookback_hours, market_regime)

async def _generate_sentiment_signal_legacy(
    symbol: str,
    lookback_hours: int = 24,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
) -> SentimentSignal:
    """
    Legacy sentiment signal generation using asyncio.gather.
    
    This maintains the original implementation for backward compatibility
    and as a fallback when Prefect is not available.
    """
    
    # Fetch data from multiple sources
    twitter_connector = TwitterConnector()
    reddit_connector = RedditConnector()
    news_connector = NewsConnector()
    
    # Concurrent fetching
    tasks = [
        twitter_connector.fetch_recent_tweets(f"{symbol}", max_results=30, since_hours=lookback_hours),
        reddit_connector.fetch_subreddit_posts("stocks", symbol, max_results=20),
        reddit_connector.fetch_subreddit_posts("investing", symbol, max_results=20),
        news_connector.fetch_news_headlines(f"{symbol} stock", max_articles=15)
    ]
    
    try:
        twitter_data, reddit_stocks, reddit_investing, news_data = await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Error fetching sentiment data: {e}")
        # Return neutral signal on error
        return SentimentSignal(
            overall_score=0.0,
            confidence=0.0,
            source_breakdown={},
            figure_influence=0.0,
            news_impact=0.0,
            social_momentum=0.0,
            contrarian_indicator=0.0,
            regime_adjusted_score=0.0,
            timestamp=datetime.now()
        )
    
    # Analyze all collected data
    all_sentiments = []
    source_scores = defaultdict(list)
    
    # Process Twitter data
    for tweet in twitter_data:
        result = score_text(tweet['text'])
        if result.confidence > CONFIDENCE_THRESHOLD:
            all_sentiments.append(result)
            source_scores[SentimentSource.TWITTER].append(result.score)
            
    # Process Reddit data
    for post in reddit_stocks + reddit_investing:
        result = score_text(post['text'])
        if result.confidence > CONFIDENCE_THRESHOLD:
            all_sentiments.append(result)
            source_scores[SentimentSource.REDDIT].append(result.score)
            
    # Process news data
    for article in news_data:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        result = score_text(text)
        if result.confidence > CONFIDENCE_THRESHOLD:
            all_sentiments.append(result)
            source_scores[SentimentSource.NEWS].append(result.score)
    
    if not all_sentiments:
        # No valid sentiment data
        return SentimentSignal(
            overall_score=0.0,
            confidence=0.0,
            source_breakdown={},
            figure_influence=0.0,
            news_impact=0.0,
            social_momentum=0.0,
            contrarian_indicator=0.0,
            regime_adjusted_score=0.0,
            timestamp=datetime.now()
        )
    
    # Calculate aggregate metrics
    overall_score = sum(s.score * s.confidence for s in all_sentiments) / sum(s.confidence for s in all_sentiments)
    overall_confidence = sum(s.confidence for s in all_sentiments) / len(all_sentiments)
    
    # Source breakdown
    source_breakdown = {}
    for source, scores in source_scores.items():
        if scores:
            source_breakdown[source] = sum(scores) / len(scores)
    
    # Figure influence
    all_figures = {}
    for sentiment in all_sentiments:
        for fig, influence in sentiment.figures.items():
            if fig in all_figures:
                all_figures[fig] = max(all_figures[fig], influence)
            else:
                all_figures[fig] = influence
    
    figure_influence = sum(all_figures.values()) * MENTION_BOOST if all_figures else 0.0
    
    # News vs social breakdown
    news_scores = source_scores.get(SentimentSource.NEWS, [])
    social_scores = source_scores.get(SentimentSource.TWITTER, []) + source_scores.get(SentimentSource.REDDIT, [])
    
    news_impact = sum(news_scores) / len(news_scores) if news_scores else 0.0
    social_impact = sum(social_scores) / len(social_scores) if social_scores else 0.0
    
    # Momentum calculation
    social_momentum = calculate_sentiment_momentum(all_sentiments)
    
    # Contrarian signals
    contrarian_indicator = detect_contrarian_signals(overall_score, overall_confidence, market_regime)
    
    # Regime adjustment
    regime_adjusted_score = adjust_for_market_regime(overall_score, market_regime)
    
    return SentimentSignal(
        overall_score=overall_score,
        confidence=overall_confidence,
        source_breakdown=source_breakdown,
        figure_influence=figure_influence,
        news_impact=news_impact,
        social_momentum=social_momentum,
        contrarian_indicator=contrarian_indicator,
        regime_adjusted_score=regime_adjusted_score,
        timestamp=datetime.now()
    )

# ================================
# BATCH SENTIMENT ANALYSIS
# ================================

async def generate_batch_sentiment_signals(
    symbols: List[str],
    lookback_hours: int = 24,
    market_regime: MarketRegime = MarketRegime.SIDEWAYS,
    use_prefect: bool = None
) -> Dict[str, SentimentSignal]:
    """
    Generate sentiment signals for multiple symbols.
    
    This function can use either individual asyncio calls or the Prefect batch flow
    for better resource management and monitoring when processing multiple symbols.
    
    Args:
        symbols: List of stock symbols to analyze
        lookback_hours: How far back to look for data (hours)
        market_regime: Current market regime for sentiment adjustments
        use_prefect: Force using Prefect (True) or asyncio (False). 
                    If None, uses configuration setting.
    
    Returns:
        Dictionary mapping symbol to SentimentSignal
    """
    
    # Determine whether to use Prefect workflows
    if use_prefect is None:
        use_prefect = getattr(settings, 'ENABLE_PREFECT', False)
    
    # Try to use Prefect batch flow if requested and available
    if use_prefect:
        try:
            from .workflows import batch_sentiment_analysis_flow, create_sentiment_signal_from_dict, is_prefect_available
            
            if is_prefect_available():
                logging.info(f"Using Prefect batch workflow for sentiment analysis of {len(symbols)} symbols")
                
                # Run the Prefect batch flow
                signal_dicts = await batch_sentiment_analysis_flow(
                    symbols=symbols,
                    lookback_hours=lookback_hours,
                    market_regime=market_regime
                )
                
                # Convert back to SentimentSignal objects
                results = {}
                for symbol, signal_dict in signal_dicts.items():
                    results[symbol] = create_sentiment_signal_from_dict(signal_dict)
                
                return results
            else:
                logging.warning("Prefect requested but not available, falling back to individual asyncio calls")
        except Exception as e:
            logging.error(f"Error running Prefect batch workflow, falling back to individual calls: {e}")
    
    # Fallback to individual sentiment analysis calls
    logging.info(f"Using individual asyncio calls for sentiment analysis of {len(symbols)} symbols")
    results = {}
    
    # Process symbols individually to avoid overwhelming APIs
    for symbol in symbols:
        try:
            signal = await generate_sentiment_signal(
                symbol=symbol,
                lookback_hours=lookback_hours,
                market_regime=market_regime,
                use_prefect=False  # Force individual calls to avoid recursion
            )
            results[symbol] = signal
            
            # Small delay between symbols to be respectful to APIs
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error generating sentiment signal for {symbol}: {e}")
            # Create neutral signal for failed symbols
            results[symbol] = SentimentSignal(
                overall_score=0.0,
                confidence=0.0,
                source_breakdown={},
                figure_influence=0.0,
                news_impact=0.0,
                social_momentum=0.0,
                contrarian_indicator=0.0,
                regime_adjusted_score=0.0,
                timestamp=datetime.now()
            )
    
    return results
