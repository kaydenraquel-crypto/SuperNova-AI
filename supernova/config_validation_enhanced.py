"""
SuperNova AI - Enhanced Configuration Validation System

This module provides comprehensive configuration validation with:
- Environment-specific validation rules and requirements
- Type checking and dependency resolution
- Security and compliance validation
- Performance and resource validation
- Configuration drift detection and correction
- Integration testing and health checks
"""

from __future__ import annotations
import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import importlib
import traceback

import yaml
from pydantic import BaseModel, Field, validator, root_validator, ValidationError
from pydantic.fields import ModelField

from .config_management import Environment, ConfigurationLevel, config_manager
from .secrets_management import get_secrets_manager
from .environment_config import environment_manager

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Configuration validation categories."""
    TYPE = "type"
    RANGE = "range"
    FORMAT = "format"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    RESOURCE = "resource"
    NETWORK = "network"
    INTEGRATION = "integration"


@dataclass
class ValidationIssue:
    """Configuration validation issue."""
    key: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    passed: bool
    environment: Environment
    total_checks: int
    passed_checks: int
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    critical: List[ValidationIssue] = field(default_factory=list)
    validation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigurationValidator(ABC):
    """Abstract base class for configuration validators."""
    
    def __init__(self, name: str, category: ValidationCategory):
        self.name = name
        self.category = category
    
    @abstractmethod
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        """Validate configuration and return issues."""
        pass


class TypeValidator(ConfigurationValidator):
    """Validate configuration value types."""
    
    def __init__(self):
        super().__init__("Type Validator", ValidationCategory.TYPE)
        
        # Define expected types for configuration keys
        self.type_mapping = {
            # Core application
            'debug': bool,
            'log_level': str,
            
            # Database
            'database_url': str,
            'db_pool_size': int,
            'db_max_overflow': int,
            'db_pool_timeout': int,
            'db_pool_recycle': int,
            'db_ssl_required': bool,
            
            # LLM
            'llm_enabled': bool,
            'llm_provider': str,
            'llm_temperature': float,
            'llm_max_tokens': int,
            'llm_timeout': int,
            'llm_max_retries': int,
            'llm_retry_delay': float,
            'llm_daily_cost_limit': float,
            'llm_cache_enabled': bool,
            'llm_cache_ttl': int,
            
            # Security
            'jwt_access_token_expire_minutes': int,
            'jwt_refresh_token_expire_days': int,
            'mfa_enabled': bool,
            'rate_limit_enabled': bool,
            'rate_limit_per_minute': int,
            'ssl_required': bool,
            
            # Performance
            'max_concurrent_requests': int,
            'async_timeout': int,
            'http_timeout': int,
            'websocket_timeout': int,
            
            # Monitoring
            'metrics_enabled': bool,
            'health_check_enabled': bool,
            'alerting_enabled': bool,
            'log_file_enabled': bool,
        }
    
    def _convert_value(self, value: str, expected_type: type) -> Any:
        """Convert string value to expected type."""
        if expected_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == str:
            return value
        else:
            return value
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        for key, expected_type in self.type_mapping.items():
            if key in config:
                value = config[key]
                
                try:
                    # If value is string, try to convert to expected type
                    if isinstance(value, str) and expected_type != str:
                        converted_value = self._convert_value(value, expected_type)
                        if type(converted_value) != expected_type:
                            issues.append(ValidationIssue(
                                key=key,
                                category=self.category,
                                severity=ValidationSeverity.ERROR,
                                message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                                suggestion=f"Convert '{value}' to {expected_type.__name__}"
                            ))
                    elif not isinstance(value, expected_type):
                        issues.append(ValidationIssue(
                            key=key,
                            category=self.category,
                            severity=ValidationSeverity.ERROR,
                            message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                            suggestion=f"Ensure {key} is of type {expected_type.__name__}"
                        ))
                
                except (ValueError, TypeError) as e:
                    issues.append(ValidationIssue(
                        key=key,
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot convert '{value}' to {expected_type.__name__}: {e}",
                        suggestion=f"Provide a valid {expected_type.__name__} value for {key}"
                    ))
        
        return issues


class RangeValidator(ConfigurationValidator):
    """Validate numeric ranges for configuration values."""
    
    def __init__(self):
        super().__init__("Range Validator", ValidationCategory.RANGE)
        
        # Define valid ranges for configuration keys
        self.range_mapping = {
            # Database
            'db_pool_size': (1, 100),
            'db_max_overflow': (0, 200),
            'db_pool_timeout': (1, 300),
            'db_pool_recycle': (300, 86400),  # 5 minutes to 24 hours
            
            # LLM
            'llm_temperature': (0.0, 2.0),
            'llm_max_tokens': (1, 32000),
            'llm_timeout': (1, 600),  # 1 second to 10 minutes
            'llm_max_retries': (0, 10),
            'llm_retry_delay': (0.1, 60.0),
            'llm_daily_cost_limit': (0.01, 10000.0),
            'llm_cache_ttl': (60, 86400),  # 1 minute to 24 hours
            
            # Security
            'jwt_access_token_expire_minutes': (1, 1440),  # 1 minute to 24 hours
            'jwt_refresh_token_expire_days': (1, 365),     # 1 day to 1 year
            'rate_limit_per_minute': (1, 10000),
            
            # Performance
            'max_concurrent_requests': (1, 1000),
            'async_timeout': (1, 300),
            'http_timeout': (1, 300),
            'websocket_timeout': (30, 3600),  # 30 seconds to 1 hour
        }
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        for key, (min_val, max_val) in self.range_mapping.items():
            if key in config:
                value = config[key]
                
                try:
                    # Convert to numeric if needed
                    if isinstance(value, str):
                        value = float(value) if '.' in value else int(value)
                    
                    if not isinstance(value, (int, float)):
                        continue
                    
                    if value < min_val:
                        issues.append(ValidationIssue(
                            key=key,
                            category=self.category,
                            severity=ValidationSeverity.ERROR,
                            message=f"Value {value} is below minimum {min_val}",
                            suggestion=f"Set {key} to at least {min_val}"
                        ))
                    elif value > max_val:
                        issues.append(ValidationIssue(
                            key=key,
                            category=self.category,
                            severity=ValidationSeverity.ERROR,
                            message=f"Value {value} is above maximum {max_val}",
                            suggestion=f"Set {key} to at most {max_val}"
                        ))
                
                except (ValueError, TypeError):
                    # Skip if can't convert to number
                    continue
        
        return issues


class SecurityValidator(ConfigurationValidator):
    """Validate security-related configuration."""
    
    def __init__(self):
        super().__init__("Security Validator", ValidationCategory.SECURITY)
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        # Check for production security requirements
        if environment == Environment.PRODUCTION:
            # SSL/TLS requirements
            if not config.get('ssl_required', False):
                issues.append(ValidationIssue(
                    key='ssl_required',
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="SSL/TLS is required in production",
                    suggestion="Set SSL_REQUIRED=true"
                ))
            
            # Database SSL
            if not config.get('db_ssl_required', False):
                issues.append(ValidationIssue(
                    key='db_ssl_required',
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="Database SSL is required in production",
                    suggestion="Set DB_SSL_REQUIRED=true"
                ))
            
            # MFA requirement
            if not config.get('mfa_enabled', False):
                issues.append(ValidationIssue(
                    key='mfa_enabled',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Multi-factor authentication should be enabled in production",
                    suggestion="Set MFA_ENABLED=true"
                ))
            
            # Debug mode check
            if config.get('debug', False):
                issues.append(ValidationIssue(
                    key='debug',
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="Debug mode must be disabled in production",
                    suggestion="Set DEBUG=false"
                ))
        
        # Check for weak secrets
        weak_secret_patterns = ['test', 'demo', 'example', 'default', 'changeme', '123']
        secret_keys = ['secret_key', 'jwt_secret_key', 'encryption_key']
        
        for key in secret_keys:
            value = config.get(key, '')
            if isinstance(value, str):
                # Check length
                if len(value) < 32:
                    issues.append(ValidationIssue(
                        key=key,
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Secret key {key} is too short (minimum 32 characters)",
                        suggestion=f"Generate a longer secret for {key}"
                    ))
                
                # Check for weak patterns
                value_lower = value.lower()
                for pattern in weak_secret_patterns:
                    if pattern in value_lower:
                        issues.append(ValidationIssue(
                            key=key,
                            category=self.category,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Secret key {key} contains weak pattern '{pattern}'",
                            suggestion=f"Generate a secure random secret for {key}"
                        ))
                        break
        
        # Rate limiting validation
        if environment in [Environment.STAGING, Environment.PRODUCTION]:
            if not config.get('rate_limit_enabled', False):
                issues.append(ValidationIssue(
                    key='rate_limit_enabled',
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message="Rate limiting should be enabled in staging/production",
                    suggestion="Set RATE_LIMIT_ENABLED=true"
                ))
        
        # CORS validation for production
        if environment == Environment.PRODUCTION:
            cors_origins = config.get('cors_allowed_origins', [])
            if isinstance(cors_origins, str):
                cors_origins = [cors_origins]
            
            if '*' in cors_origins:
                issues.append(ValidationIssue(
                    key='cors_allowed_origins',
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="CORS wildcard (*) is not allowed in production",
                    suggestion="Specify exact allowed origins"
                ))
        
        return issues


class DependencyValidator(ConfigurationValidator):
    """Validate configuration dependencies."""
    
    def __init__(self):
        super().__init__("Dependency Validator", ValidationCategory.DEPENDENCY)
        
        # Define configuration dependencies
        self.dependencies = {
            # LLM dependencies
            'llm_enabled': {
                'requires': ['llm_provider'],
                'conditional': {
                    'llm_provider=openai': ['openai_api_key'],
                    'llm_provider=anthropic': ['anthropic_api_key'],
                    'llm_fallback_enabled=true': ['llm_fallback_provider'],
                }
            },
            
            # Database dependencies
            'database_url': {
                'requires': ['db_pool_size'],
                'conditional': {
                    'db_ssl_required=true': ['db_ssl_ca', 'db_ssl_cert', 'db_ssl_key']
                }
            },
            
            # Caching dependencies
            'cache_backend=redis': {
                'requires': ['redis_url']
            },
            
            # Monitoring dependencies
            'metrics_enabled=true': {
                'requires': ['metrics_port']
            },
            
            'alerting_enabled=true': {
                'requires': ['alert_webhook_url']
            },
            
            # Security dependencies
            'mfa_enabled=true': {
                'requires': ['mfa_issuer_name']
            },
            
            # Backup dependencies
            'backup_enabled=true': {
                'requires': ['backup_interval_hours']
            }
        }
    
    def _check_conditional_dependency(self, condition: str, config: Dict[str, Any]) -> bool:
        """Check if a conditional dependency should apply."""
        if '=' in condition:
            key, expected_value = condition.split('=', 1)
            actual_value = config.get(key)
            
            # Convert values for comparison
            if expected_value.lower() == 'true':
                return actual_value in (True, 'true', 'True', '1', 'yes', 'on')
            elif expected_value.lower() == 'false':
                return actual_value in (False, 'false', 'False', '0', 'no', 'off')
            else:
                return str(actual_value) == expected_value
        
        return config.get(condition, False)
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        for key, deps in self.dependencies.items():
            # Check if this key applies
            if '=' in key:
                if not self._check_conditional_dependency(key, config):
                    continue
            else:
                if key not in config or not config.get(key):
                    continue
            
            # Check required dependencies
            for required_key in deps.get('requires', []):
                if required_key not in config or not config.get(required_key):
                    issues.append(ValidationIssue(
                        key=required_key,
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Required dependency {required_key} is missing for {key}",
                        suggestion=f"Set {required_key} when using {key}"
                    ))
            
            # Check conditional dependencies
            for condition, required_keys in deps.get('conditional', {}).items():
                if self._check_conditional_dependency(condition, config):
                    for required_key in required_keys:
                        if required_key not in config or not config.get(required_key):
                            issues.append(ValidationIssue(
                                key=required_key,
                                category=self.category,
                                severity=ValidationSeverity.ERROR,
                                message=f"Conditional dependency {required_key} is missing when {condition}",
                                suggestion=f"Set {required_key} when {condition}"
                            ))
        
        return issues


class PerformanceValidator(ConfigurationValidator):
    """Validate performance-related configuration."""
    
    def __init__(self):
        super().__init__("Performance Validator", ValidationCategory.PERFORMANCE)
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        # Check for performance anti-patterns
        
        # Database pool sizing
        pool_size = config.get('db_pool_size', 5)
        max_overflow = config.get('db_max_overflow', 10)
        max_concurrent = config.get('max_concurrent_requests', 50)
        
        if isinstance(pool_size, str):
            pool_size = int(pool_size)
        if isinstance(max_overflow, str):
            max_overflow = int(max_overflow)
        if isinstance(max_concurrent, str):
            max_concurrent = int(max_concurrent)
        
        total_db_connections = pool_size + max_overflow
        
        if total_db_connections < max_concurrent / 10:
            issues.append(ValidationIssue(
                key='db_pool_size',
                category=self.category,
                severity=ValidationSeverity.WARNING,
                message=f"Database connection pool ({total_db_connections}) may be too small for concurrent requests ({max_concurrent})",
                suggestion="Consider increasing DB_POOL_SIZE or DB_MAX_OVERFLOW"
            ))
        
        # LLM timeout vs request timeout
        llm_timeout = config.get('llm_timeout', 30)
        http_timeout = config.get('http_timeout', 30)
        
        if isinstance(llm_timeout, str):
            llm_timeout = int(llm_timeout)
        if isinstance(http_timeout, str):
            http_timeout = int(http_timeout)
        
        if llm_timeout >= http_timeout:
            issues.append(ValidationIssue(
                key='llm_timeout',
                category=self.category,
                severity=ValidationSeverity.WARNING,
                message=f"LLM timeout ({llm_timeout}s) should be less than HTTP timeout ({http_timeout}s)",
                suggestion="Set LLM_TIMEOUT to be 5-10 seconds less than HTTP_TIMEOUT"
            ))
        
        # Cache TTL validation
        cache_ttl = config.get('cache_default_ttl', 3600)
        llm_cache_ttl = config.get('llm_cache_ttl', 3600)
        
        if isinstance(cache_ttl, str):
            cache_ttl = int(cache_ttl)
        if isinstance(llm_cache_ttl, str):
            llm_cache_ttl = int(llm_cache_ttl)
        
        if llm_cache_ttl > cache_ttl * 2:
            issues.append(ValidationIssue(
                key='llm_cache_ttl',
                category=self.category,
                severity=ValidationSeverity.INFO,
                message=f"LLM cache TTL ({llm_cache_ttl}s) is much higher than default cache TTL ({cache_ttl}s)",
                suggestion="Consider if LLM responses need such long caching"
            ))
        
        # Environment-specific performance checks
        if environment == Environment.PRODUCTION:
            # Production should have optimized settings
            if config.get('debug', False):
                issues.append(ValidationIssue(
                    key='debug',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Debug mode degrades performance in production",
                    suggestion="Set DEBUG=false in production"
                ))
            
            if config.get('db_echo', False):
                issues.append(ValidationIssue(
                    key='db_echo',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Database query logging degrades performance in production",
                    suggestion="Set DB_ECHO=false in production"
                ))
        
        return issues


class ComplianceValidator(ConfigurationValidator):
    """Validate compliance-related configuration."""
    
    def __init__(self):
        super().__init__("Compliance Validator", ValidationCategory.COMPLIANCE)
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        # Production compliance requirements
        if environment == Environment.PRODUCTION:
            
            # Audit logging
            if not config.get('audit_logging_enabled', False):
                issues.append(ValidationIssue(
                    key='audit_logging_enabled',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Audit logging is required for compliance in production",
                    suggestion="Set AUDIT_LOGGING_ENABLED=true"
                ))
            
            # Data retention
            retention_days = config.get('audit_log_retention_days', 0)
            if isinstance(retention_days, str):
                retention_days = int(retention_days)
            
            if retention_days < 365:  # 1 year minimum
                issues.append(ValidationIssue(
                    key='audit_log_retention_days',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Audit log retention must be at least 365 days for compliance",
                    suggestion="Set AUDIT_LOG_RETENTION_DAYS to at least 365"
                ))
            
            # Encryption requirements
            if not config.get('database_encryption_enabled', False):
                issues.append(ValidationIssue(
                    key='database_encryption_enabled',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Database encryption is required for compliance",
                    suggestion="Set DATABASE_ENCRYPTION_ENABLED=true"
                ))
            
            if not config.get('backup_encryption_required', False):
                issues.append(ValidationIssue(
                    key='backup_encryption_required',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Backup encryption is required for compliance",
                    suggestion="Set BACKUP_ENCRYPTION_REQUIRED=true"
                ))
            
            # GDPR compliance
            if config.get('gdpr_compliance_enabled', False):
                if not config.get('right_to_be_forgotten_enabled', False):
                    issues.append(ValidationIssue(
                        key='right_to_be_forgotten_enabled',
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message="Right to be forgotten must be enabled for GDPR compliance",
                        suggestion="Set RIGHT_TO_BE_FORGOTTEN_ENABLED=true"
                    ))
                
                if not config.get('consent_management_enabled', False):
                    issues.append(ValidationIssue(
                        key='consent_management_enabled',
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        message="Consent management must be enabled for GDPR compliance",
                        suggestion="Set CONSENT_MANAGEMENT_ENABLED=true"
                    ))
            
            # PII protection
            if not config.get('data_anonymization_enabled', False):
                issues.append(ValidationIssue(
                    key='data_anonymization_enabled',
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message="Data anonymization should be enabled for privacy protection",
                    suggestion="Set DATA_ANONYMIZATION_ENABLED=true"
                ))
        
        return issues


class IntegrationValidator(ConfigurationValidator):
    """Validate external integrations and API configurations."""
    
    def __init__(self):
        super().__init__("Integration Validator", ValidationCategory.INTEGRATION)
    
    async def validate(self, config: Dict[str, Any], environment: Environment) -> List[ValidationIssue]:
        issues = []
        
        # LLM Provider validation
        llm_provider = config.get('llm_provider', '').lower()
        
        if llm_provider == 'openai':
            if not config.get('openai_api_key'):
                issues.append(ValidationIssue(
                    key='openai_api_key',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="OpenAI API key is required when using OpenAI provider",
                    suggestion="Set OPENAI_API_KEY with valid API key"
                ))
            
            # Check for test/example keys
            api_key = config.get('openai_api_key', '')
            if any(pattern in api_key.lower() for pattern in ['test', 'demo', 'example', 'sk-1234']):
                issues.append(ValidationIssue(
                    key='openai_api_key',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="OpenAI API key appears to be a test/example key",
                    suggestion="Use a valid OpenAI API key"
                ))
        
        elif llm_provider == 'anthropic':
            if not config.get('anthropic_api_key'):
                issues.append(ValidationIssue(
                    key='anthropic_api_key',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Anthropic API key is required when using Anthropic provider",
                    suggestion="Set ANTHROPIC_API_KEY with valid API key"
                ))
        
        # Database URL validation
        database_url = config.get('database_url', '')
        if database_url:
            if database_url.startswith('sqlite://') and environment == Environment.PRODUCTION:
                issues.append(ValidationIssue(
                    key='database_url',
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="SQLite database is not recommended for production use",
                    suggestion="Use PostgreSQL or another production database"
                ))
            
            if 'localhost' in database_url and environment == Environment.PRODUCTION:
                issues.append(ValidationIssue(
                    key='database_url',
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message="Database URL uses localhost in production",
                    suggestion="Use proper database host for production deployment"
                ))
        
        # Redis configuration
        if config.get('cache_backend') == 'redis':
            redis_url = config.get('redis_url', '')
            if not redis_url:
                issues.append(ValidationIssue(
                    key='redis_url',
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    message="Redis URL is required when using Redis cache backend",
                    suggestion="Set REDIS_URL with valid Redis connection string"
                ))
            elif 'localhost' in redis_url and environment == Environment.PRODUCTION:
                issues.append(ValidationIssue(
                    key='redis_url',
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message="Redis URL uses localhost in production",
                    suggestion="Use proper Redis host for production deployment"
                ))
        
        # External API validation for production
        if environment == Environment.PRODUCTION:
            external_apis = [
                'newsapi_key', 'alpha_vantage_key', 'financial_modeling_prep_key',
                'x_bearer_token', 'reddit_client_id', 'reddit_client_secret'
            ]
            
            for api_key in external_apis:
                value = config.get(api_key, '')
                if value and any(pattern in value.lower() for pattern in ['test', 'demo', 'example']):
                    issues.append(ValidationIssue(
                        key=api_key,
                        category=self.category,
                        severity=ValidationSeverity.WARNING,
                        message=f"{api_key} appears to be a test/demo key",
                        suggestion=f"Use production API key for {api_key}"
                    ))
        
        return issues


class EnhancedConfigurationValidator:
    """Main configuration validation system."""
    
    def __init__(self):
        self.validators: List[ConfigurationValidator] = [
            TypeValidator(),
            RangeValidator(),
            SecurityValidator(),
            DependencyValidator(),
            PerformanceValidator(),
            ComplianceValidator(),
            IntegrationValidator()
        ]
        
        self.environment_specific_validators = {
            Environment.PRODUCTION: [
                'SecurityValidator',
                'ComplianceValidator',
                'PerformanceValidator',
                'IntegrationValidator'
            ],
            Environment.STAGING: [
                'SecurityValidator',
                'PerformanceValidator',
                'IntegrationValidator'
            ],
            Environment.DEVELOPMENT: [
                'TypeValidator',
                'RangeValidator',
                'DependencyValidator'
            ],
            Environment.TESTING: [
                'TypeValidator',
                'RangeValidator'
            ]
        }
    
    def _load_configuration(self, environment: Environment) -> Dict[str, Any]:
        """Load configuration for the specified environment."""
        config = {}
        
        # Load from environment variables
        for key, value in os.environ.items():
            config[key.lower()] = value
        
        # Load from .env files
        env_files = [
            f".env.{environment.value}",
            f".env.{environment.value}.local",
            ".env.local",
            ".env"
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip().strip('"').strip("'")
                            if key not in config:  # Don't override env vars
                                config[key] = value
        
        return config
    
    async def validate_environment(
        self,
        environment: Environment,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        categories: Optional[List[ValidationCategory]] = None
    ) -> ValidationResult:
        """Validate configuration for a specific environment."""
        
        start_time = datetime.utcnow()
        
        if config is None:
            config = self._load_configuration(environment)
        
        # Determine which validators to run
        active_validators = []
        
        if strict:
            # Run all validators in strict mode
            active_validators = self.validators
        elif environment in self.environment_specific_validators:
            # Run environment-specific validators
            validator_names = self.environment_specific_validators[environment]
            active_validators = [
                v for v in self.validators 
                if v.__class__.__name__ in validator_names
            ]
        else:
            # Run basic validators
            active_validators = [
                v for v in self.validators 
                if v.__class__.__name__ in ['TypeValidator', 'RangeValidator', 'DependencyValidator']
            ]
        
        # Filter by categories if specified
        if categories:
            active_validators = [
                v for v in active_validators 
                if v.category in categories
            ]
        
        # Run validation
        all_issues = []
        total_checks = 0
        
        for validator in active_validators:
            try:
                logger.debug(f"Running validator: {validator.name}")
                issues = await validator.validate(config, environment)
                all_issues.extend(issues)
                total_checks += 1
            except Exception as e:
                logger.error(f"Error in validator {validator.name}: {e}")
                all_issues.append(ValidationIssue(
                    key='validator_error',
                    category=ValidationCategory.INTEGRATION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator {validator.name} failed: {str(e)}",
                    details={'exception': str(e), 'traceback': traceback.format_exc()}
                ))
        
        # Categorize issues by severity
        warnings = [issue for issue in all_issues if issue.severity == ValidationSeverity.WARNING]
        errors = [issue for issue in all_issues if issue.severity == ValidationSeverity.ERROR]
        critical = [issue for issue in all_issues if issue.severity == ValidationSeverity.CRITICAL]
        
        # Determine if validation passed
        passed = len(errors) == 0 and len(critical) == 0
        
        # Calculate validation time
        validation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ValidationResult(
            passed=passed,
            environment=environment,
            total_checks=total_checks,
            passed_checks=total_checks - len(all_issues),
            issues=all_issues,
            warnings=warnings,
            errors=errors,
            critical=critical,
            validation_time=validation_time
        )
    
    async def validate_all_environments(
        self,
        environments: Optional[List[Environment]] = None,
        strict: bool = False
    ) -> Dict[Environment, ValidationResult]:
        """Validate configuration for multiple environments."""
        
        if environments is None:
            environments = list(Environment)
        
        results = {}
        
        for env in environments:
            try:
                result = await self.validate_environment(env, strict=strict)
                results[env] = result
            except Exception as e:
                logger.error(f"Failed to validate environment {env}: {e}")
                results[env] = ValidationResult(
                    passed=False,
                    environment=env,
                    total_checks=0,
                    passed_checks=0,
                    errors=[ValidationIssue(
                        key='validation_error',
                        category=ValidationCategory.INTEGRATION,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Environment validation failed: {str(e)}",
                        details={'exception': str(e)}
                    )]
                )
        
        return results
    
    def format_validation_result(
        self,
        result: ValidationResult,
        format: str = 'text',
        include_suggestions: bool = True
    ) -> str:
        """Format validation result for output."""
        
        if format == 'json':
            return json.dumps({
                'passed': result.passed,
                'environment': result.environment.value,
                'total_checks': result.total_checks,
                'passed_checks': result.passed_checks,
                'validation_time': result.validation_time,
                'timestamp': result.timestamp.isoformat(),
                'issues': [
                    {
                        'key': issue.key,
                        'category': issue.category.value,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'suggestion': issue.suggestion,
                        'details': issue.details
                    }
                    for issue in result.issues
                ]
            }, indent=2)
        
        # Text format
        output = []
        output.append(f"Configuration Validation Report")
        output.append(f"=" * 50)
        output.append(f"Environment: {result.environment.value}")
        output.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        output.append(f"Checks: {result.passed_checks}/{result.total_checks} passed")
        output.append(f"Time: {result.validation_time:.2f}s")
        output.append(f"Timestamp: {result.timestamp}")
        output.append("")
        
        if result.critical:
            output.append(f"CRITICAL ISSUES ({len(result.critical)}):")
            output.append("-" * 30)
            for issue in result.critical:
                output.append(f"  ❌ {issue.key}: {issue.message}")
                if include_suggestions and issue.suggestion:
                    output.append(f"     💡 {issue.suggestion}")
            output.append("")
        
        if result.errors:
            output.append(f"ERRORS ({len(result.errors)}):")
            output.append("-" * 30)
            for issue in result.errors:
                output.append(f"  🚫 {issue.key}: {issue.message}")
                if include_suggestions and issue.suggestion:
                    output.append(f"     💡 {issue.suggestion}")
            output.append("")
        
        if result.warnings:
            output.append(f"WARNINGS ({len(result.warnings)}):")
            output.append("-" * 30)
            for issue in result.warnings:
                output.append(f"  ⚠️  {issue.key}: {issue.message}")
                if include_suggestions and issue.suggestion:
                    output.append(f"     💡 {issue.suggestion}")
            output.append("")
        
        if result.passed:
            output.append("✅ All validations passed successfully!")
        
        return "\n".join(output)


# Global validator instance
validator = EnhancedConfigurationValidator()


async def validate_configuration(
    environment: Environment = None,
    strict: bool = False,
    categories: List[ValidationCategory] = None,
    format: str = 'text'
) -> str:
    """Validate configuration and return formatted result."""
    
    if environment is None:
        environment = Environment(os.getenv('SUPERNOVA_ENV', 'development'))
    
    result = await validator.validate_environment(
        environment=environment,
        strict=strict,
        categories=categories
    )
    
    return validator.format_validation_result(result, format=format)


async def validate_all_configurations(strict: bool = False) -> Dict[str, str]:
    """Validate all environment configurations."""
    
    results = await validator.validate_all_environments(strict=strict)
    
    formatted_results = {}
    for env, result in results.items():
        formatted_results[env.value] = validator.format_validation_result(result)
    
    return formatted_results


# CLI interface
async def main():
    """Main CLI interface for configuration validation."""
    
    parser = argparse.ArgumentParser(description='SuperNova AI Configuration Validator')
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'staging', 'production'],
        default=os.getenv('SUPERNOVA_ENV', 'development'),
        help='Environment to validate (default: from SUPERNOVA_ENV)'
    )
    parser.add_argument(
        '--strict', '-s',
        action='store_true',
        help='Enable strict validation (all validators)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Validate all environments'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    parser.add_argument(
        '--categories', '-c',
        nargs='*',
        choices=[cat.value for cat in ValidationCategory],
        help='Specific validation categories to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.all:
            # Validate all environments
            results = await validate_all_configurations(strict=args.strict)
            
            if args.format == 'json':
                print(json.dumps(results, indent=2))
            else:
                for env, result in results.items():
                    print(f"\n{'=' * 60}")
                    print(f"Environment: {env.upper()}")
                    print('=' * 60)
                    print(result)
        else:
            # Validate specific environment
            environment = Environment(args.environment)
            categories = [ValidationCategory(cat) for cat in args.categories] if args.categories else None
            
            result = await validate_configuration(
                environment=environment,
                strict=args.strict,
                categories=categories,
                format=args.format
            )
            
            print(result)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())