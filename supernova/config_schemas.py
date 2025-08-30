"""
SuperNova AI - Configuration Schema Definitions

This module defines Pydantic schemas for environment-specific configurations
with comprehensive validation and type checking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field, validator, root_validator
from datetime import datetime
import os
import re

from .config_management import Environment, ConfigurationLevel


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class DatabaseEngine(str, Enum):
    """Supported database engines."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    TIMESCALEDB = "timescaledb"


class CacheBackend(str, Enum):
    """Supported cache backends."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    # Environment
    supernova_env: Environment = Field(default=Environment.DEVELOPMENT, env="SUPERNOVA_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # Application
    app_name: str = Field(default="SuperNova AI", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(default="Advanced AI-Powered Trading System", env="APP_DESCRIPTION")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if v and len(v) < 32:
            raise ValueError('Encryption key must be at least 32 characters long')
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True
        extra = "allow"


class DatabaseConfig(BaseSettings):
    """Database configuration schema."""
    
    # Primary Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_engine: DatabaseEngine = Field(default=DatabaseEngine.SQLITE, env="DATABASE_ENGINE")
    
    # Connection Settings
    db_pool_size: int = Field(default=5, env="DB_POOL_SIZE", ge=1, le=50)
    db_max_overflow: int = Field(default=10, env="DB_MAX_OVERFLOW", ge=0, le=100)
    db_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT", ge=1, le=300)
    db_pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE", ge=60, le=86400)
    
    # Performance
    db_echo: bool = Field(default=False, env="DB_ECHO")
    db_echo_pool: bool = Field(default=False, env="DB_ECHO_POOL")
    
    # Security
    db_ssl_required: bool = Field(default=False, env="DB_SSL_REQUIRED")
    db_ssl_ca: Optional[str] = Field(default=None, env="DB_SSL_CA")
    db_ssl_cert: Optional[str] = Field(default=None, env="DB_SSL_CERT")
    db_ssl_key: Optional[str] = Field(default=None, env="DB_SSL_KEY")
    
    # Backup
    backup_enabled: bool = Field(default=True, env="BACKUP_ENABLED")
    backup_interval_hours: int = Field(default=6, env="BACKUP_INTERVAL_HOURS", ge=1, le=168)
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS", ge=1, le=365)
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('sqlite://', 'postgresql://', 'mysql://', 'timescaledb://')):
            raise ValueError('Invalid database URL format')
        return v

    class Config:
        env_prefix = "DB_"


class TimescaleConfig(BaseSettings):
    """TimescaleDB-specific configuration."""
    
    # Connection
    timescale_enabled: bool = Field(default=False, env="TIMESCALE_ENABLED")
    timescale_host: Optional[str] = Field(default=None, env="TIMESCALE_HOST")
    timescale_port: int = Field(default=5432, env="TIMESCALE_PORT", ge=1, le=65535)
    timescale_database: Optional[str] = Field(default=None, env="TIMESCALE_DATABASE")
    timescale_username: Optional[str] = Field(default=None, env="TIMESCALE_USERNAME")
    timescale_password: Optional[str] = Field(default=None, env="TIMESCALE_PASSWORD")
    
    # Hypertable Settings
    chunk_time_interval: str = Field(default="1 day", env="TIMESCALE_CHUNK_INTERVAL")
    compression_enabled: bool = Field(default=True, env="TIMESCALE_COMPRESSION_ENABLED")
    compression_after: str = Field(default="7 days", env="TIMESCALE_COMPRESSION_AFTER")
    
    # Data Retention
    retention_policy: str = Field(default="1 year", env="TIMESCALE_RETENTION_POLICY")
    drop_chunks_older_than: str = Field(default="1 year", env="TIMESCALE_DROP_CHUNKS_AFTER")
    
    # Continuous Aggregates
    enable_continuous_aggregates: bool = Field(default=True, env="TIMESCALE_ENABLE_AGGREGATES")
    hourly_aggregates: bool = Field(default=True, env="TIMESCALE_HOURLY_AGGREGATES")
    daily_aggregates: bool = Field(default=True, env="TIMESCALE_DAILY_AGGREGATES")
    
    @root_validator
    def validate_timescale_config(cls, values):
        enabled = values.get('timescale_enabled')
        if enabled:
            required_fields = ['timescale_host', 'timescale_database', 'timescale_username', 'timescale_password']
            for field in required_fields:
                if not values.get(field):
                    raise ValueError(f'{field} is required when TimescaleDB is enabled')
        return values


class LLMConfig(BaseSettings):
    """LLM configuration schema."""
    
    # General LLM Settings
    llm_enabled: bool = Field(default=True, env="LLM_ENABLED")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, env="LLM_TEMPERATURE", ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS", ge=1, le=32000)
    llm_timeout: int = Field(default=60, env="LLM_TIMEOUT", ge=1, le=600)
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES", ge=0, le=10)
    llm_retry_delay: float = Field(default=1.0, env="LLM_RETRY_DELAY", ge=0.1, le=10.0)
    
    # Features
    llm_streaming: bool = Field(default=False, env="LLM_STREAMING")
    llm_context_enhancement: bool = Field(default=True, env="LLM_CONTEXT_ENHANCEMENT")
    llm_cache_enabled: bool = Field(default=True, env="LLM_CACHE_ENABLED")
    llm_cache_ttl: int = Field(default=3600, env="LLM_CACHE_TTL", ge=60, le=86400)
    
    # Cost Management
    llm_cost_tracking: bool = Field(default=True, env="LLM_COST_TRACKING")
    llm_daily_cost_limit: float = Field(default=50.0, env="LLM_DAILY_COST_LIMIT", ge=0.01)
    llm_monthly_cost_limit: float = Field(default=1000.0, env="LLM_MONTHLY_COST_LIMIT", ge=0.01)
    llm_cost_alert_threshold: float = Field(default=0.8, env="LLM_COST_ALERT_THRESHOLD", ge=0.1, le=1.0)
    
    # Fallback
    llm_fallback_enabled: bool = Field(default=True, env="LLM_FALLBACK_ENABLED")
    llm_fallback_provider: Optional[LLMProvider] = Field(default=None, env="LLM_FALLBACK_PROVIDER")


class OpenAIConfig(BaseSettings):
    """OpenAI-specific configuration."""
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, env="OPENAI_ORG_ID")
    openai_project_id: Optional[str] = Field(default=None, env="OPENAI_PROJECT_ID")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    
    # Model Selection
    openai_default_model: str = Field(default="gpt-4", env="OPENAI_DEFAULT_MODEL")
    openai_fallback_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_FALLBACK_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    
    # Rate Limits
    openai_max_requests_per_minute: int = Field(default=3000, env="OPENAI_MAX_REQUESTS_PER_MINUTE", ge=1)
    openai_max_tokens_per_minute: int = Field(default=150000, env="OPENAI_MAX_TOKENS_PER_MINUTE", ge=1)
    
    # Cost per 1K tokens
    openai_gpt4_input_cost: float = Field(default=0.03, env="OPENAI_GPT4_INPUT_COST", ge=0.0)
    openai_gpt4_output_cost: float = Field(default=0.06, env="OPENAI_GPT4_OUTPUT_COST", ge=0.0)
    openai_gpt35_input_cost: float = Field(default=0.0015, env="OPENAI_GPT35_INPUT_COST", ge=0.0)
    openai_gpt35_output_cost: float = Field(default=0.002, env="OPENAI_GPT35_OUTPUT_COST", ge=0.0)
    
    @validator('openai_base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('OpenAI base URL must start with http:// or https://')
        return v


class AnthropicConfig(BaseSettings):
    """Anthropic-specific configuration."""
    
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", env="ANTHROPIC_BASE_URL")
    
    # Model Selection
    anthropic_default_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_DEFAULT_MODEL")
    anthropic_fallback_model: str = Field(default="claude-3-haiku-20240307", env="ANTHROPIC_FALLBACK_MODEL")
    
    # Rate Limits
    anthropic_max_requests_per_minute: int = Field(default=1000, env="ANTHROPIC_MAX_REQUESTS_PER_MINUTE", ge=1)
    anthropic_max_tokens_per_minute: int = Field(default=100000, env="ANTHROPIC_MAX_TOKENS_PER_MINUTE", ge=1)
    
    # Cost per 1K tokens
    anthropic_opus_input_cost: float = Field(default=0.015, env="ANTHROPIC_OPUS_INPUT_COST", ge=0.0)
    anthropic_opus_output_cost: float = Field(default=0.075, env="ANTHROPIC_OPUS_OUTPUT_COST", ge=0.0)
    anthropic_sonnet_input_cost: float = Field(default=0.003, env="ANTHROPIC_SONNET_INPUT_COST", ge=0.0)
    anthropic_sonnet_output_cost: float = Field(default=0.015, env="ANTHROPIC_SONNET_OUTPUT_COST", ge=0.0)
    
    # Safety
    anthropic_safety_level: str = Field(default="standard", env="ANTHROPIC_SAFETY_LEVEL")
    anthropic_content_filter: bool = Field(default=True, env="ANTHROPIC_CONTENT_FILTER")


class OllamaConfig(BaseSettings):
    """Ollama-specific configuration."""
    
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT", ge=1, le=600)
    
    # Model Configuration
    ollama_default_model: str = Field(default="llama2", env="OLLAMA_DEFAULT_MODEL")
    ollama_chat_model: str = Field(default="llama2:13b", env="OLLAMA_CHAT_MODEL")
    ollama_analysis_model: str = Field(default="codellama:13b", env="OLLAMA_ANALYSIS_MODEL")
    
    # Performance Settings
    ollama_context_window: int = Field(default=4096, env="OLLAMA_CONTEXT_WINDOW", ge=512, le=32768)
    ollama_num_gpu: int = Field(default=1, env="OLLAMA_NUM_GPU", ge=0, le=8)
    ollama_num_thread: int = Field(default=8, env="OLLAMA_NUM_THREAD", ge=1, le=64)
    
    # Resource Management
    ollama_max_concurrent: int = Field(default=2, env="OLLAMA_MAX_CONCURRENT", ge=1, le=10)
    ollama_memory_limit: str = Field(default="8GB", env="OLLAMA_MEMORY_LIMIT")
    ollama_auto_download: bool = Field(default=False, env="OLLAMA_AUTO_DOWNLOAD")
    
    @validator('ollama_base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Ollama base URL must start with http:// or https://')
        return v


class SecurityConfig(BaseSettings):
    """Security configuration schema."""
    
    # Authentication
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=15, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES", ge=1, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS", ge=1, le=30)
    
    # API Security
    api_key_encryption_enabled: bool = Field(default=True, env="API_KEY_ENCRYPTION_ENABLED")
    api_key_rotation_enabled: bool = Field(default=False, env="API_KEY_ROTATION_ENABLED")
    api_key_rotation_days: int = Field(default=90, env="API_KEY_ROTATION_DAYS", ge=1, le=365)
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE", ge=1, le=10000)
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR", ge=1, le=100000)
    
    # SSL/TLS
    ssl_required: bool = Field(default=False, env="SSL_REQUIRED")
    tls_version_min: str = Field(default="1.2", env="TLS_VERSION_MIN")
    
    # CORS
    cors_allowed_origins: List[str] = Field(default=["*"], env="CORS_ALLOWED_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters long')
        return v
    
    @root_validator
    def validate_production_security(cls, values):
        env = values.get('supernova_env')
        if env == Environment.PRODUCTION:
            if not values.get('ssl_required'):
                raise ValueError('SSL is required in production environment')
            if "*" in values.get('cors_allowed_origins', []):
                raise ValueError('Wildcard CORS origins not allowed in production')
        return values


class CacheConfig(BaseSettings):
    """Caching configuration schema."""
    
    cache_backend: CacheBackend = Field(default=CacheBackend.MEMORY, env="CACHE_BACKEND")
    cache_default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL", ge=60, le=86400)
    
    # Redis Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS", ge=1, le=100)
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT", ge=1, le=60)
    
    # Memory Cache Settings
    memory_cache_size: int = Field(default=1000, env="MEMORY_CACHE_SIZE", ge=10, le=100000)
    
    @root_validator
    def validate_cache_config(cls, values):
        backend = values.get('cache_backend')
        if backend == CacheBackend.REDIS:
            if not values.get('redis_url'):
                raise ValueError('Redis URL is required when using Redis cache backend')
        return values


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    log_file_path: str = Field(default="logs/supernova.log", env="LOG_FILE_PATH")
    log_max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT", ge=1, le=50)
    
    # Metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT", ge=1024, le=65535)
    
    # Health Checks
    health_check_enabled: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(default=300, env="HEALTH_CHECK_INTERVAL", ge=60, le=3600)
    
    # Alerting
    alerting_enabled: bool = Field(default=True, env="ALERTING_ENABLED")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    alert_email_enabled: bool = Field(default=False, env="ALERT_EMAIL_ENABLED")
    alert_email_recipients: List[str] = Field(default=[], env="ALERT_EMAIL_RECIPIENTS")
    
    @validator('alert_webhook_url')
    def validate_webhook_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Alert webhook URL must start with http:// or https://')
        return v


class DevelopmentConfig(BaseConfig, DatabaseConfig, LLMConfig, OpenAIConfig, AnthropicConfig, 
                       OllamaConfig, SecurityConfig, CacheConfig, MonitoringConfig):
    """Development environment configuration."""
    
    # Development-specific overrides
    debug: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.DEBUG)
    ssl_required: bool = Field(default=False)
    rate_limit_enabled: bool = Field(default=False)
    
    # Relaxed security for development
    cors_allowed_origins: List[str] = Field(default=["*"])
    db_echo: bool = Field(default=True)
    
    class Config:
        env_file = ".env.development"


class TestingConfig(BaseConfig, DatabaseConfig, LLMConfig, SecurityConfig, CacheConfig, MonitoringConfig):
    """Testing environment configuration."""
    
    # Testing-specific overrides
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.WARNING)
    
    # Use in-memory/fast backends for testing
    database_url: str = Field(default="sqlite:///:memory:")
    cache_backend: CacheBackend = Field(default=CacheBackend.MEMORY)
    
    # Minimal resource usage
    db_pool_size: int = Field(default=2)
    memory_cache_size: int = Field(default=100)
    
    # Disable external dependencies
    metrics_enabled: bool = Field(default=False)
    alerting_enabled: bool = Field(default=False)
    
    class Config:
        env_file = ".env.testing"


class StagingConfig(BaseConfig, DatabaseConfig, TimescaleConfig, LLMConfig, OpenAIConfig, 
                   AnthropicConfig, SecurityConfig, CacheConfig, MonitoringConfig):
    """Staging environment configuration."""
    
    # Production-like settings with some debugging
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    ssl_required: bool = Field(default=True)
    rate_limit_enabled: bool = Field(default=True)
    
    # Enhanced monitoring for staging
    metrics_enabled: bool = Field(default=True)
    health_check_enabled: bool = Field(default=True)
    
    # Moderate resource allocation
    db_pool_size: int = Field(default=10)
    
    class Config:
        env_file = ".env.staging"


class ProductionConfig(BaseConfig, DatabaseConfig, TimescaleConfig, LLMConfig, OpenAIConfig, 
                      AnthropicConfig, SecurityConfig, CacheConfig, MonitoringConfig):
    """Production environment configuration."""
    
    # Production-hardened settings
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    ssl_required: bool = Field(default=True)
    rate_limit_enabled: bool = Field(default=True)
    
    # Enhanced security
    api_key_encryption_enabled: bool = Field(default=True)
    api_key_rotation_enabled: bool = Field(default=True)
    
    # No wildcard CORS in production
    cors_allowed_origins: List[str] = Field(default=["https://dashboard.supernova-ai.com"])
    
    # Production resource allocation
    db_pool_size: int = Field(default=20)
    db_max_overflow: int = Field(default=30)
    
    # Full monitoring
    metrics_enabled: bool = Field(default=True)
    alerting_enabled: bool = Field(default=True)
    health_check_enabled: bool = Field(default=True)
    
    class Config:
        env_file = ".env.production"


# Configuration factory
def get_config_schema(environment: Environment) -> BaseConfig:
    """Factory function to get appropriate configuration schema based on environment."""
    
    config_map = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.TESTING: TestingConfig,
        Environment.STAGING: StagingConfig,
        Environment.PRODUCTION: ProductionConfig
    }
    
    config_class = config_map.get(environment, DevelopmentConfig)
    return config_class()


# Environment-specific validation rules
ENVIRONMENT_VALIDATION_RULES = {
    Environment.PRODUCTION: {
        'required_secrets': [
            'secret_key', 'jwt_secret_key', 'encryption_key',
            'openai_api_key', 'anthropic_api_key'
        ],
        'forbidden_settings': {
            'debug': True,
            'db_echo': True,
            'cors_allowed_origins': ['*']
        },
        'minimum_values': {
            'jwt_access_token_expire_minutes': 15,
            'rate_limit_per_minute': 60,
            'db_pool_size': 10
        }
    },
    Environment.STAGING: {
        'required_secrets': [
            'secret_key', 'jwt_secret_key'
        ],
        'forbidden_settings': {
            'debug': True
        },
        'minimum_values': {
            'db_pool_size': 5
        }
    },
    Environment.DEVELOPMENT: {
        'optional_secrets': [
            'openai_api_key', 'anthropic_api_key'
        ]
    },
    Environment.TESTING: {
        'recommended_settings': {
            'cache_backend': CacheBackend.MEMORY,
            'database_url': 'sqlite:///:memory:'
        }
    }
}


def validate_environment_config(config: BaseConfig) -> List[str]:
    """Validate configuration against environment-specific rules."""
    
    errors = []
    rules = ENVIRONMENT_VALIDATION_RULES.get(config.supernova_env, {})
    
    # Check required secrets
    for secret in rules.get('required_secrets', []):
        if not getattr(config, secret, None):
            errors.append(f"Required secret '{secret}' is not configured for {config.supernova_env}")
    
    # Check forbidden settings
    for setting, forbidden_value in rules.get('forbidden_settings', {}).items():
        if getattr(config, setting, None) == forbidden_value:
            errors.append(f"Setting '{setting}' cannot be '{forbidden_value}' in {config.supernova_env}")
    
    # Check minimum values
    for setting, min_value in rules.get('minimum_values', {}).items():
        current_value = getattr(config, setting, None)
        if current_value is not None and current_value < min_value:
            errors.append(f"Setting '{setting}' must be at least {min_value} in {config.supernova_env}")
    
    return errors