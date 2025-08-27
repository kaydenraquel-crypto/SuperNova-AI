from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Core application settings
    SUPERNOVA_ENV: str = os.getenv("SUPERNOVA_ENV", "dev")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./supernova.db")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    
    # Social Media API Keys
    X_BEARER_TOKEN: str | None = os.getenv("X_BEARER_TOKEN")
    REDDIT_CLIENT_ID: str | None = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str | None = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str | None = os.getenv("REDDIT_USER_AGENT")
    
    # News API Keys
    NEWSAPI_KEY: str | None = os.getenv("NEWSAPI_KEY")
    ALPHA_VANTAGE_KEY: str | None = os.getenv("ALPHA_VANTAGE_KEY")
    FINANCIAL_MODELING_PREP_KEY: str | None = os.getenv("FINANCIAL_MODELING_PREP_KEY")
    
    # NLP Model Configuration
    FINBERT_MODEL_PATH: str = os.getenv("FINBERT_MODEL_PATH", "ProsusAI/finbert")
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    
    # Sentiment Analysis Settings
    SENTIMENT_CACHE_TTL: int = int(os.getenv("SENTIMENT_CACHE_TTL", "300"))  # 5 minutes
    MAX_SENTIMENT_BATCH_SIZE: int = int(os.getenv("MAX_SENTIMENT_BATCH_SIZE", "100"))
    SENTIMENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.3"))
    
    # Rate Limiting (requests per second)
    TWITTER_RATE_LIMIT: float = float(os.getenv("TWITTER_RATE_LIMIT", "0.5"))
    REDDIT_RATE_LIMIT: float = float(os.getenv("REDDIT_RATE_LIMIT", "1.0"))
    NEWS_RATE_LIMIT: float = float(os.getenv("NEWS_RATE_LIMIT", "2.0"))
    
    # Data Source Preferences (enable/disable sources)
    ENABLE_TWITTER: bool = os.getenv("ENABLE_TWITTER", "true").lower() == "true"
    ENABLE_REDDIT: bool = os.getenv("ENABLE_REDDIT", "true").lower() == "true"
    ENABLE_NEWS: bool = os.getenv("ENABLE_NEWS", "true").lower() == "true"
    ENABLE_FINBERT: bool = os.getenv("ENABLE_FINBERT", "true").lower() == "true"
    ENABLE_VADER: bool = os.getenv("ENABLE_VADER", "true").lower() == "true"
    
    # Signal Blending Weights
    SENTIMENT_WEIGHT_TWITTER: float = float(os.getenv("SENTIMENT_WEIGHT_TWITTER", "0.3"))
    SENTIMENT_WEIGHT_REDDIT: float = float(os.getenv("SENTIMENT_WEIGHT_REDDIT", "0.2"))
    SENTIMENT_WEIGHT_NEWS: float = float(os.getenv("SENTIMENT_WEIGHT_NEWS", "0.4"))
    SENTIMENT_WEIGHT_FINBERT: float = float(os.getenv("SENTIMENT_WEIGHT_FINBERT", "0.4"))
    
    # Redis Cache (optional, for production)
    REDIS_URL: str | None = os.getenv("REDIS_URL")
    USE_REDIS_CACHE: bool = os.getenv("USE_REDIS_CACHE", "false").lower() == "true"
    
    # LLM Integration Settings
    # Core LLM Configuration
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, ollama, huggingface
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # OpenAI Configuration
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG_ID: str | None = os.getenv("OPENAI_ORG_ID")
    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    
    # Ollama Configuration (for local models)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Hugging Face Configuration
    HUGGINGFACE_API_TOKEN: str | None = os.getenv("HUGGINGFACE_API_TOKEN")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-large")
    
    # LLM Performance and Reliability
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
    
    # Response Caching
    LLM_CACHE_ENABLED: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    LLM_CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour
    
    # Cost Management
    LLM_DAILY_COST_LIMIT: float = float(os.getenv("LLM_DAILY_COST_LIMIT", "10.0"))  # $10 per day
    LLM_COST_TRACKING: bool = os.getenv("LLM_COST_TRACKING", "true").lower() == "true"
    
    # Fallback Configuration
    LLM_FALLBACK_ENABLED: bool = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    LLM_FALLBACK_SIMPLE_RATIONALE: bool = os.getenv("LLM_FALLBACK_SIMPLE_RATIONALE", "true").lower() == "true"
    
    # Advanced Features
    LLM_STREAMING: bool = os.getenv("LLM_STREAMING", "false").lower() == "true"
    LLM_CONTEXT_ENHANCEMENT: bool = os.getenv("LLM_CONTEXT_ENHANCEMENT", "true").lower() == "true"
    
    # Prefect Workflow Orchestration Settings
    ENABLE_PREFECT: bool = os.getenv("ENABLE_PREFECT", "true").lower() == "true"
    PREFECT_API_URL: str | None = os.getenv("PREFECT_API_URL")  # For Prefect Cloud/Server
    PREFECT_API_KEY: str | None = os.getenv("PREFECT_API_KEY")  # For Prefect Cloud
    
    # Prefect Task Configuration
    PREFECT_TWITTER_RETRIES: int = int(os.getenv("PREFECT_TWITTER_RETRIES", "3"))
    PREFECT_REDDIT_RETRIES: int = int(os.getenv("PREFECT_REDDIT_RETRIES", "3"))
    PREFECT_NEWS_RETRIES: int = int(os.getenv("PREFECT_NEWS_RETRIES", "3"))
    PREFECT_RETRY_DELAY: int = int(os.getenv("PREFECT_RETRY_DELAY", "60"))  # seconds
    PREFECT_TASK_TIMEOUT: int = int(os.getenv("PREFECT_TASK_TIMEOUT", "300"))  # seconds
    
    # Prefect Flow Configuration
    PREFECT_SENTIMENT_FLOW_NAME: str = os.getenv("PREFECT_SENTIMENT_FLOW_NAME", "sentiment_analysis_pipeline")
    PREFECT_BATCH_FLOW_NAME: str = os.getenv("PREFECT_BATCH_FLOW_NAME", "batch_sentiment_analysis")
    PREFECT_MAX_CONCURRENT_TASKS: int = int(os.getenv("PREFECT_MAX_CONCURRENT_TASKS", "10"))
    
    # Prefect Deployment Settings
    PREFECT_WORK_QUEUE: str = os.getenv("PREFECT_WORK_QUEUE", "default")
    PREFECT_DEPLOYMENT_TAGS: str = os.getenv("PREFECT_DEPLOYMENT_TAGS", "sentiment,supernova,production")
    
    # Prefect Logging and Monitoring
    PREFECT_LOG_LEVEL: str = os.getenv("PREFECT_LOG_LEVEL", "INFO")
    PREFECT_ENABLE_METRICS: bool = os.getenv("PREFECT_ENABLE_METRICS", "true").lower() == "true"
    
    # VectorBT Backtesting Configuration
    VECTORBT_ENABLED: bool = os.getenv("VECTORBT_ENABLED", "true").lower() == "true"
    VECTORBT_DEFAULT_ENGINE: bool = os.getenv("VECTORBT_DEFAULT_ENGINE", "true").lower() == "true"
    VECTORBT_DEFAULT_FEES: float = float(os.getenv("VECTORBT_DEFAULT_FEES", "0.001"))  # 0.1%
    VECTORBT_DEFAULT_SLIPPAGE: float = float(os.getenv("VECTORBT_DEFAULT_SLIPPAGE", "0.001"))  # 0.1%
    VECTORBT_PERFORMANCE_MODE: bool = os.getenv("VECTORBT_PERFORMANCE_MODE", "true").lower() == "true"
    VECTORBT_CACHE_SIGNALS: bool = os.getenv("VECTORBT_CACHE_SIGNALS", "true").lower() == "true"
    
    # Backtesting Performance Settings
    BACKTEST_MAX_BARS: int = int(os.getenv("BACKTEST_MAX_BARS", "10000"))  # Max bars for single backtest
    BACKTEST_MIN_BARS: int = int(os.getenv("BACKTEST_MIN_BARS", "50"))     # Min bars required
    BACKTEST_CACHE_TTL: int = int(os.getenv("BACKTEST_CACHE_TTL", "3600")) # 1 hour cache
    
    # Strategy Template Preferences
    ENABLE_LEGACY_STRATEGIES: bool = os.getenv("ENABLE_LEGACY_STRATEGIES", "true").lower() == "true"
    ENABLE_VBT_STRATEGIES: bool = os.getenv("ENABLE_VBT_STRATEGIES", "true").lower() == "true"
    DEFAULT_STRATEGY_ENGINE: str = os.getenv("DEFAULT_STRATEGY_ENGINE", "vectorbt")  # vectorbt or legacy

    # TimescaleDB Configuration for Time-Series Sentiment Data
    # Connection Settings
    TIMESCALE_HOST: str | None = os.getenv("TIMESCALE_HOST")  # e.g., "localhost" or "timescaledb.example.com"
    TIMESCALE_PORT: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    TIMESCALE_DB: str | None = os.getenv("TIMESCALE_DB")  # e.g., "supernova_timeseries"
    TIMESCALE_USER: str | None = os.getenv("TIMESCALE_USER")  # e.g., "supernova_user"
    TIMESCALE_PASSWORD: str | None = os.getenv("TIMESCALE_PASSWORD")
    
    # Connection Pool Settings
    TIMESCALE_POOL_SIZE: int = int(os.getenv("TIMESCALE_POOL_SIZE", "5"))
    TIMESCALE_MAX_OVERFLOW: int = int(os.getenv("TIMESCALE_MAX_OVERFLOW", "10"))
    TIMESCALE_POOL_TIMEOUT: int = int(os.getenv("TIMESCALE_POOL_TIMEOUT", "30"))
    TIMESCALE_POOL_RECYCLE: int = int(os.getenv("TIMESCALE_POOL_RECYCLE", "3600"))  # 1 hour
    
    # TimescaleDB Feature Configuration
    TIMESCALE_ENABLED: bool = os.getenv("TIMESCALE_ENABLED", "false").lower() == "true"
    TIMESCALE_AUTO_SETUP: bool = os.getenv("TIMESCALE_AUTO_SETUP", "true").lower() == "true"
    
    # Hypertable Configuration
    TIMESCALE_CHUNK_INTERVAL: str = os.getenv("TIMESCALE_CHUNK_INTERVAL", "1 day")  # TimescaleDB chunk size
    TIMESCALE_COMPRESSION_AFTER: str = os.getenv("TIMESCALE_COMPRESSION_AFTER", "7 days")  # When to compress data
    TIMESCALE_RETENTION_POLICY: str = os.getenv("TIMESCALE_RETENTION_POLICY", "1 year")  # When to drop data
    
    # Continuous Aggregates Configuration
    TIMESCALE_ENABLE_AGGREGATES: bool = os.getenv("TIMESCALE_ENABLE_AGGREGATES", "true").lower() == "true"
    TIMESCALE_HOURLY_AGGREGATES: bool = os.getenv("TIMESCALE_HOURLY_AGGREGATES", "true").lower() == "true"
    TIMESCALE_DAILY_AGGREGATES: bool = os.getenv("TIMESCALE_DAILY_AGGREGATES", "true").lower() == "true"
    TIMESCALE_WEEKLY_AGGREGATES: bool = os.getenv("TIMESCALE_WEEKLY_AGGREGATES", "true").lower() == "true"
    
    # Data Quality and Processing
    TIMESCALE_BATCH_SIZE: int = int(os.getenv("TIMESCALE_BATCH_SIZE", "1000"))  # Batch size for bulk inserts
    TIMESCALE_MAX_RETRY_ATTEMPTS: int = int(os.getenv("TIMESCALE_MAX_RETRY_ATTEMPTS", "3"))
    TIMESCALE_RETRY_DELAY: float = float(os.getenv("TIMESCALE_RETRY_DELAY", "1.0"))  # seconds
    TIMESCALE_CONNECTION_TIMEOUT: int = int(os.getenv("TIMESCALE_CONNECTION_TIMEOUT", "30"))  # seconds
    
    # Performance and Monitoring
    TIMESCALE_SLOW_QUERY_THRESHOLD: float = float(os.getenv("TIMESCALE_SLOW_QUERY_THRESHOLD", "5.0"))  # seconds
    TIMESCALE_LOG_QUERIES: bool = os.getenv("TIMESCALE_LOG_QUERIES", "false").lower() == "true"
    TIMESCALE_ENABLE_QUERY_STATS: bool = os.getenv("TIMESCALE_ENABLE_QUERY_STATS", "true").lower() == "true"
    
    # Backup and Maintenance
    TIMESCALE_ENABLE_BACKUP: bool = os.getenv("TIMESCALE_ENABLE_BACKUP", "false").lower() == "true"
    TIMESCALE_BACKUP_RETENTION: str = os.getenv("TIMESCALE_BACKUP_RETENTION", "30 days")
    TIMESCALE_MAINTENANCE_WINDOW: str = os.getenv("TIMESCALE_MAINTENANCE_WINDOW", "02:00-04:00")  # UTC time window
    
    # SSL and Security
    TIMESCALE_SSL_MODE: str = os.getenv("TIMESCALE_SSL_MODE", "prefer")  # disable, allow, prefer, require
    TIMESCALE_SSL_CERT: str | None = os.getenv("TIMESCALE_SSL_CERT")
    TIMESCALE_SSL_KEY: str | None = os.getenv("TIMESCALE_SSL_KEY")
    TIMESCALE_SSL_ROOT_CERT: str | None = os.getenv("TIMESCALE_SSL_ROOT_CERT")

    # NovaSignal Integration Settings
    NOVASIGNAL_API_URL: str = os.getenv("NOVASIGNAL_API_URL", "https://api.novasignal.io")
    NOVASIGNAL_WS_URL: str = os.getenv("NOVASIGNAL_WS_URL", "wss://ws.novasignal.io")
    NOVASIGNAL_API_KEY: str | None = os.getenv("NOVASIGNAL_API_KEY")
    NOVASIGNAL_SECRET: str | None = os.getenv("NOVASIGNAL_SECRET")
    NOVASIGNAL_WEBHOOK_SECRET: str | None = os.getenv("NOVASIGNAL_WEBHOOK_SECRET")
    NOVASIGNAL_ALERT_ENDPOINT: str = os.getenv("NOVASIGNAL_ALERT_ENDPOINT", "/webhooks/supernova/alerts")
    NOVASIGNAL_DASHBOARD_URL: str = os.getenv("NOVASIGNAL_DASHBOARD_URL", "https://dashboard.novasignal.io")
    
    # Connection and Performance Settings
    NOVASIGNAL_CONNECTION_TIMEOUT: int = int(os.getenv("NOVASIGNAL_CONNECTION_TIMEOUT", "30"))
    NOVASIGNAL_MAX_RETRIES: int = int(os.getenv("NOVASIGNAL_MAX_RETRIES", "3"))
    NOVASIGNAL_BUFFER_SIZE: int = int(os.getenv("NOVASIGNAL_BUFFER_SIZE", "1000"))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

settings = Settings()
