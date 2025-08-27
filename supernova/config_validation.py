"""
SuperNova AI - Configuration Validation Framework

This module provides comprehensive configuration validation with:
- Type checking and constraint validation
- Dependency validation and circular dependency detection
- Cross-environment validation rules
- Security compliance validation
- Performance impact analysis
- Automated configuration testing
"""

from __future__ import annotations
import os
import json
import logging
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import inspect

from pydantic import BaseModel, Field, ValidationError, validator
import yaml

from .config_management import Environment, ConfigurationLevel, ConfigurationSource
from .config_schemas import get_config_schema, validate_environment_config
from .secrets_management import SecretValidator

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(str, Enum):
    """Types of validation checks."""
    TYPE_CHECK = "type_check"
    CONSTRAINT = "constraint"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    ENVIRONMENT = "environment"
    INTEGRATION = "integration"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_id: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    affected_keys: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_checks: int
    passed: int
    failed: int
    warnings: int
    errors: int
    critical: int
    execution_time: float
    results: List[ValidationResult]
    environment: Environment
    config_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigurationValidator(ABC):
    """Abstract base class for configuration validators."""
    
    def __init__(self, validator_id: str, description: str):
        self.validator_id = validator_id
        self.description = description
    
    @abstractmethod
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate configuration and return results."""
        pass
    
    def create_result(
        self,
        validation_type: ValidationType,
        severity: ValidationSeverity,
        passed: bool,
        message: str,
        **kwargs
    ) -> ValidationResult:
        """Helper to create validation result."""
        return ValidationResult(
            check_id=self.validator_id,
            validation_type=validation_type,
            severity=severity,
            passed=passed,
            message=message,
            **kwargs
        )


class TypeValidator(ConfigurationValidator):
    """Validates configuration types and constraints."""
    
    def __init__(self):
        super().__init__("type_validator", "Type and constraint validation")
        self.type_checks = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict
        }
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        
        # Get expected schema for environment
        try:
            schema = get_config_schema(environment)
            
            # Validate against Pydantic schema
            try:
                validated_config = schema.__class__(**config)
                results.append(
                    self.create_result(
                        ValidationType.TYPE_CHECK,
                        ValidationSeverity.INFO,
                        True,
                        "Configuration passed type validation"
                    )
                )
            except ValidationError as e:
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    results.append(
                        self.create_result(
                            ValidationType.TYPE_CHECK,
                            ValidationSeverity.ERROR,
                            False,
                            f"Type validation failed for {field}: {error['msg']}",
                            affected_keys=[field],
                            details={'error_type': error['type'], 'input_value': error.get('input')}
                        )
                    )
            
        except Exception as e:
            results.append(
                self.create_result(
                    ValidationType.TYPE_CHECK,
                    ValidationSeverity.CRITICAL,
                    False,
                    f"Schema validation error: {str(e)}"
                )
            )
        
        return results


class DependencyValidator(ConfigurationValidator):
    """Validates configuration dependencies and detects circular dependencies."""
    
    def __init__(self):
        super().__init__("dependency_validator", "Configuration dependency validation")
        
        # Define dependency relationships
        self.dependencies = {
            'llm_enabled': ['llm_provider'],
            'llm_provider': {
                'openai': ['openai_api_key'],
                'anthropic': ['anthropic_api_key'],
                'ollama': ['ollama_base_url'],
                'huggingface': ['huggingface_api_token']
            },
            'timescale_enabled': ['timescale_host', 'timescale_database', 'timescale_username', 'timescale_password'],
            'redis_cache_enabled': ['redis_url'],
            'ssl_required': ['tls_version_min'],
            'backup_enabled': ['backup_retention_days'],
            'rate_limit_enabled': ['rate_limit_per_minute'],
            'jwt_authentication': ['jwt_secret_key', 'jwt_algorithm'],
            'api_key_encryption_enabled': ['encryption_key'],
            'metrics_enabled': ['metrics_port'],
            'alerting_enabled': ['alert_webhook_url']
        }
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        
        # Check required dependencies
        for key, deps in self.dependencies.items():
            if config.get(key):
                if isinstance(deps, dict):
                    # Conditional dependencies based on value
                    provider_value = config.get('llm_provider') if key == 'llm_provider' else None
                    if provider_value and provider_value in deps:
                        required_deps = deps[provider_value]
                        missing_deps = [dep for dep in required_deps if not config.get(dep)]
                        
                        if missing_deps:
                            results.append(
                                self.create_result(
                                    ValidationType.DEPENDENCY,
                                    ValidationSeverity.ERROR,
                                    False,
                                    f"Missing dependencies for {key}={provider_value}: {missing_deps}",
                                    affected_keys=[key] + missing_deps,
                                    suggestions=[f"Set {dep} to enable {key}" for dep in missing_deps]
                                )
                            )
                        else:
                            results.append(
                                self.create_result(
                                    ValidationType.DEPENDENCY,
                                    ValidationSeverity.INFO,
                                    True,
                                    f"All dependencies satisfied for {key}={provider_value}"
                                )
                            )
                elif isinstance(deps, list):
                    # Simple list dependencies
                    missing_deps = [dep for dep in deps if not config.get(dep)]
                    
                    if missing_deps:
                        results.append(
                            self.create_result(
                                ValidationType.DEPENDENCY,
                                ValidationSeverity.ERROR,
                                False,
                                f"Missing dependencies for {key}: {missing_deps}",
                                affected_keys=[key] + missing_deps,
                                suggestions=[f"Set {dep} to enable {key}" for dep in missing_deps]
                            )
                        )
                    else:
                        results.append(
                            self.create_result(
                                ValidationType.DEPENDENCY,
                                ValidationSeverity.INFO,
                                True,
                                f"All dependencies satisfied for {key}"
                            )
                        )
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(config)
        if circular_deps:
            results.append(
                self.create_result(
                    ValidationType.DEPENDENCY,
                    ValidationSeverity.CRITICAL,
                    False,
                    f"Circular dependencies detected: {' -> '.join(circular_deps)}",
                    affected_keys=circular_deps,
                    suggestions=["Review configuration dependencies to break cycles"]
                )
            )
        
        # Check for conflicting configurations
        conflicts = self._check_configuration_conflicts(config)
        for conflict in conflicts:
            results.append(
                self.create_result(
                    ValidationType.DEPENDENCY,
                    ValidationSeverity.WARNING,
                    False,
                    conflict['message'],
                    affected_keys=conflict['keys'],
                    suggestions=conflict['suggestions']
                )
            )
        
        return results
    
    def _detect_circular_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """Detect circular dependencies in configuration."""
        # Simple implementation - can be enhanced
        visited = set()
        path = []
        
        def dfs(key: str) -> Optional[List[str]]:
            if key in path:
                # Found cycle
                cycle_start = path.index(key)
                return path[cycle_start:] + [key]
            
            if key in visited:
                return None
            
            visited.add(key)
            path.append(key)
            
            # Get dependencies for this key
            deps = self.dependencies.get(key, [])
            if isinstance(deps, dict):
                provider_value = config.get('llm_provider') if key == 'llm_provider' else None
                if provider_value and provider_value in deps:
                    deps = deps[provider_value]
                else:
                    deps = []
            
            if isinstance(deps, list):
                for dep in deps:
                    if config.get(dep):  # Only check active dependencies
                        cycle = dfs(dep)
                        if cycle:
                            return cycle
            
            path.pop()
            return None
        
        for key in config.keys():
            if key not in visited:
                cycle = dfs(key)
                if cycle:
                    return cycle
        
        return []
    
    def _check_configuration_conflicts(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for conflicting configuration settings."""
        conflicts = []
        
        # Debug mode in production
        if config.get('debug') and config.get('supernova_env') == Environment.PRODUCTION.value:
            conflicts.append({
                'message': "Debug mode should not be enabled in production environment",
                'keys': ['debug', 'supernova_env'],
                'suggestions': ["Set debug=False in production"]
            })
        
        # SQLite in production with high load
        if (config.get('supernova_env') == Environment.PRODUCTION.value and 
            'sqlite' in config.get('database_url', '').lower() and
            config.get('db_pool_size', 1) > 5):
            conflicts.append({
                'message': "SQLite with high connection pool size in production is not recommended",
                'keys': ['database_url', 'db_pool_size'],
                'suggestions': ["Use PostgreSQL for production", "Reduce connection pool size for SQLite"]
            })
        
        # Memory cache with Redis URL
        if (config.get('cache_backend') == 'memory' and config.get('redis_url')):
            conflicts.append({
                'message': "Redis URL configured but memory cache backend selected",
                'keys': ['cache_backend', 'redis_url'],
                'suggestions': ["Set cache_backend=redis to use Redis", "Remove redis_url if using memory cache"]
            })
        
        # SSL disabled with sensitive data
        if (not config.get('ssl_required') and 
            config.get('supernova_env') in [Environment.STAGING.value, Environment.PRODUCTION.value]):
            conflicts.append({
                'message': "SSL should be required in staging and production environments",
                'keys': ['ssl_required', 'supernova_env'],
                'suggestions': ["Enable SSL for secure environments"]
            })
        
        return conflicts


class SecurityValidator(ConfigurationValidator):
    """Validates security-related configuration settings."""
    
    def __init__(self):
        super().__init__("security_validator", "Security configuration validation")
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        
        # Secret key strength validation
        for key in ['secret_key', 'jwt_secret_key', 'encryption_key']:
            value = config.get(key)
            if value:
                strength_result = self._validate_key_strength(key, value)
                results.append(strength_result)
            elif environment in [Environment.PRODUCTION, Environment.STAGING]:
                results.append(
                    self.create_result(
                        ValidationType.SECURITY,
                        ValidationSeverity.CRITICAL,
                        False,
                        f"Required security key '{key}' is missing in {environment.value}",
                        affected_keys=[key],
                        suggestions=[f"Set a strong {key} for {environment.value} environment"]
                    )
                )
        
        # API key validation
        for provider in ['openai', 'anthropic', 'huggingface']:
            api_key = config.get(f'{provider}_api_key')
            if api_key:
                if self._is_placeholder_key(api_key):
                    results.append(
                        self.create_result(
                            ValidationType.SECURITY,
                            ValidationSeverity.ERROR,
                            False,
                            f"{provider.title()} API key appears to be a placeholder",
                            affected_keys=[f'{provider}_api_key'],
                            suggestions=[f"Replace with actual {provider.title()} API key"]
                        )
                    )
                elif len(api_key) < 20:
                    results.append(
                        self.create_result(
                            ValidationType.SECURITY,
                            ValidationSeverity.WARNING,
                            False,
                            f"{provider.title()} API key is unusually short",
                            affected_keys=[f'{provider}_api_key']
                        )
                    )
        
        # Password validation for database URLs
        database_url = config.get('database_url', '')
        if '://' in database_url and '@' in database_url:
            password_result = self._validate_database_password(database_url)
            if password_result:
                results.append(password_result)
        
        # CORS validation for production
        cors_origins = config.get('cors_allowed_origins', [])
        if isinstance(cors_origins, list) and '*' in cors_origins:
            if environment == Environment.PRODUCTION:
                results.append(
                    self.create_result(
                        ValidationType.SECURITY,
                        ValidationSeverity.CRITICAL,
                        False,
                        "Wildcard CORS origins are not allowed in production",
                        affected_keys=['cors_allowed_origins'],
                        suggestions=["Specify exact allowed origins for production"]
                    )
                )
            else:
                results.append(
                    self.create_result(
                        ValidationType.SECURITY,
                        ValidationSeverity.WARNING,
                        False,
                        f"Wildcard CORS origins detected in {environment.value}",
                        affected_keys=['cors_allowed_origins'],
                        suggestions=["Consider restricting CORS origins"]
                    )
                )
        
        # Rate limiting validation
        if not config.get('rate_limit_enabled'):
            severity = ValidationSeverity.CRITICAL if environment == Environment.PRODUCTION else ValidationSeverity.WARNING
            results.append(
                self.create_result(
                    ValidationType.SECURITY,
                    severity,
                    False,
                    f"Rate limiting is disabled in {environment.value}",
                    affected_keys=['rate_limit_enabled'],
                    suggestions=["Enable rate limiting for security"]
                )
            )
        
        # JWT configuration validation
        jwt_expire = config.get('jwt_access_token_expire_minutes', 15)
        if jwt_expire > 60 and environment == Environment.PRODUCTION:
            results.append(
                self.create_result(
                    ValidationType.SECURITY,
                    ValidationSeverity.WARNING,
                    False,
                    f"JWT access token expiry is long ({jwt_expire} minutes) for production",
                    affected_keys=['jwt_access_token_expire_minutes'],
                    suggestions=["Consider shorter token expiry for production (15-30 minutes)"]
                )
            )
        
        return results
    
    def _validate_key_strength(self, key_name: str, key_value: str) -> ValidationResult:
        """Validate cryptographic key strength."""
        
        if len(key_value) < 32:
            return self.create_result(
                ValidationType.SECURITY,
                ValidationSeverity.CRITICAL,
                False,
                f"{key_name} is too short (minimum 32 characters)",
                affected_keys=[key_name],
                suggestions=[f"Generate a stronger {key_name} with at least 32 characters"]
            )
        
        # Check for common weak patterns
        weak_patterns = ['123', 'abc', 'password', 'secret', 'key', 'test', 'demo']
        if any(pattern in key_value.lower() for pattern in weak_patterns):
            return self.create_result(
                ValidationType.SECURITY,
                ValidationSeverity.ERROR,
                False,
                f"{key_name} contains weak patterns",
                affected_keys=[key_name],
                suggestions=[f"Generate a random {key_name} without common words"]
            )
        
        # Check entropy (simplified)
        unique_chars = len(set(key_value))
        if unique_chars < len(key_value) * 0.6:
            return self.create_result(
                ValidationType.SECURITY,
                ValidationSeverity.WARNING,
                False,
                f"{key_name} has low entropy (many repeated characters)",
                affected_keys=[key_name],
                suggestions=[f"Generate a more random {key_name}"]
            )
        
        return self.create_result(
            ValidationType.SECURITY,
            ValidationSeverity.INFO,
            True,
            f"{key_name} passes security validation",
            affected_keys=[key_name]
        )
    
    def _is_placeholder_key(self, api_key: str) -> bool:
        """Check if API key is a placeholder."""
        placeholders = [
            'your-api-key-here',
            'your-key-here', 
            'replace-with-actual-key',
            'sk-placeholder',
            'dummy-key',
            'test-key',
            'example-key'
        ]
        return any(placeholder in api_key.lower() for placeholder in placeholders)
    
    def _validate_database_password(self, database_url: str) -> Optional[ValidationResult]:
        """Validate database password strength from URL."""
        try:
            # Extract password from URL
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            if parsed.password:
                validation = SecretValidator.validate_password_strength(parsed.password)
                if not validation['valid']:
                    return self.create_result(
                        ValidationType.SECURITY,
                        ValidationSeverity.ERROR,
                        False,
                        f"Database password is weak: {', '.join(validation['issues'])}",
                        affected_keys=['database_url'],
                        suggestions=validation['suggestions']
                    )
        except Exception:
            pass  # Ignore parsing errors
        
        return None


class PerformanceValidator(ConfigurationValidator):
    """Validates performance-related configuration settings."""
    
    def __init__(self):
        super().__init__("performance_validator", "Performance configuration validation")
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        
        # Database connection pool validation
        pool_size = config.get('db_pool_size', 5)
        max_overflow = config.get('db_max_overflow', 10)
        
        if environment == Environment.PRODUCTION:
            if pool_size < 10:
                results.append(
                    self.create_result(
                        ValidationType.PERFORMANCE,
                        ValidationSeverity.WARNING,
                        False,
                        f"Database pool size ({pool_size}) may be too small for production",
                        affected_keys=['db_pool_size'],
                        suggestions=["Consider increasing pool size to 15-20 for production"]
                    )
                )
            elif pool_size > 50:
                results.append(
                    self.create_result(
                        ValidationType.PERFORMANCE,
                        ValidationSeverity.WARNING,
                        False,
                        f"Database pool size ({pool_size}) may be too large",
                        affected_keys=['db_pool_size'],
                        suggestions=["Consider reducing pool size to avoid resource waste"]
                    )
                )
        
        # Cache configuration validation
        cache_backend = config.get('cache_backend', 'memory')
        cache_size = config.get('memory_cache_size', 1000)
        
        if cache_backend == 'memory' and cache_size > 10000:
            results.append(
                self.create_result(
                    ValidationType.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    False,
                    f"Memory cache size ({cache_size}) is very large",
                    affected_keys=['memory_cache_size'],
                    suggestions=["Consider using Redis for large cache sizes"]
                )
            )
        
        # LLM timeout validation
        llm_timeout = config.get('llm_timeout', 60)
        if llm_timeout > 300:
            results.append(
                self.create_result(
                    ValidationType.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    False,
                    f"LLM timeout ({llm_timeout}s) is very high",
                    affected_keys=['llm_timeout'],
                    suggestions=["Consider reducing timeout to improve responsiveness"]
                )
            )
        elif llm_timeout < 10:
            results.append(
                self.create_result(
                    ValidationType.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    False,
                    f"LLM timeout ({llm_timeout}s) may be too low",
                    affected_keys=['llm_timeout'],
                    suggestions=["Consider increasing timeout to avoid premature failures"]
                )
            )
        
        # Log level impact
        log_level = config.get('log_level', 'INFO')
        if log_level == 'DEBUG' and environment == Environment.PRODUCTION:
            results.append(
                self.create_result(
                    ValidationType.PERFORMANCE,
                    ValidationSeverity.ERROR,
                    False,
                    "DEBUG logging enabled in production will impact performance",
                    affected_keys=['log_level'],
                    suggestions=["Use INFO or WARNING level for production"]
                )
            )
        
        return results


class ComplianceValidator(ConfigurationValidator):
    """Validates compliance-related configuration settings."""
    
    def __init__(self):
        super().__init__("compliance_validator", "Compliance configuration validation")
        
        # Compliance requirements by environment
        self.requirements = {
            Environment.PRODUCTION: {
                'required_encryption': True,
                'audit_logging': True,
                'data_retention_days': 2555,  # 7 years
                'ssl_required': True,
                'backup_required': True
            },
            Environment.STAGING: {
                'audit_logging': True,
                'ssl_required': True
            }
        }
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        requirements = self.requirements.get(environment, {})
        
        # Check encryption requirements
        if requirements.get('required_encryption'):
            if not config.get('api_key_encryption_enabled'):
                results.append(
                    self.create_result(
                        ValidationType.COMPLIANCE,
                        ValidationSeverity.CRITICAL,
                        False,
                        f"API key encryption is required for {environment.value}",
                        affected_keys=['api_key_encryption_enabled'],
                        suggestions=["Enable API key encryption for compliance"]
                    )
                )
        
        # Check audit logging
        if requirements.get('audit_logging'):
            if not config.get('security_logging_enabled'):
                results.append(
                    self.create_result(
                        ValidationType.COMPLIANCE,
                        ValidationSeverity.ERROR,
                        False,
                        f"Audit logging is required for {environment.value}",
                        affected_keys=['security_logging_enabled'],
                        suggestions=["Enable security/audit logging for compliance"]
                    )
                )
        
        # Check data retention
        if requirements.get('data_retention_days'):
            retention_days = config.get('audit_log_retention_days', 0)
            if retention_days < requirements['data_retention_days']:
                results.append(
                    self.create_result(
                        ValidationType.COMPLIANCE,
                        ValidationSeverity.ERROR,
                        False,
                        f"Audit log retention ({retention_days} days) below compliance requirement ({requirements['data_retention_days']} days)",
                        affected_keys=['audit_log_retention_days'],
                        suggestions=[f"Set retention to at least {requirements['data_retention_days']} days"]
                    )
                )
        
        # Check SSL requirements
        if requirements.get('ssl_required'):
            if not config.get('ssl_required'):
                results.append(
                    self.create_result(
                        ValidationType.COMPLIANCE,
                        ValidationSeverity.CRITICAL,
                        False,
                        f"SSL is required for {environment.value}",
                        affected_keys=['ssl_required'],
                        suggestions=["Enable SSL for compliance"]
                    )
                )
        
        # Check backup requirements
        if requirements.get('backup_required'):
            if not config.get('backup_enabled'):
                results.append(
                    self.create_result(
                        ValidationType.COMPLIANCE,
                        ValidationSeverity.ERROR,
                        False,
                        f"Data backup is required for {environment.value}",
                        affected_keys=['backup_enabled'],
                        suggestions=["Enable data backup for compliance"]
                    )
                )
        
        return results


class IntegrationValidator(ConfigurationValidator):
    """Validates integration and connectivity settings."""
    
    def __init__(self):
        super().__init__("integration_validator", "Integration configuration validation")
    
    async def validate(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        
        results = []
        
        # Test database connectivity
        database_url = config.get('database_url')
        if database_url:
            connectivity_result = await self._test_database_connectivity(database_url)
            results.append(connectivity_result)
        
        # Test Redis connectivity if enabled
        if config.get('cache_backend') == 'redis':
            redis_url = config.get('redis_url')
            if redis_url:
                redis_result = await self._test_redis_connectivity(redis_url)
                results.append(redis_result)
        
        # Test TimescaleDB connectivity if enabled
        if config.get('timescale_enabled'):
            timescale_result = await self._test_timescale_connectivity(config)
            results.append(timescale_result)
        
        # Validate webhook URLs
        webhook_url = config.get('alert_webhook_url')
        if webhook_url:
            webhook_result = self._validate_webhook_url(webhook_url)
            results.append(webhook_result)
        
        return results
    
    async def _test_database_connectivity(self, database_url: str) -> ValidationResult:
        """Test database connectivity."""
        try:
            from sqlalchemy import create_engine, text
            
            # Create engine with minimal settings for testing
            engine = create_engine(database_url, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Simple connectivity test
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            engine.dispose()
            
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.INFO,
                True,
                "Database connectivity test passed",
                affected_keys=['database_url']
            )
        
        except Exception as e:
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.ERROR,
                False,
                f"Database connectivity test failed: {str(e)}",
                affected_keys=['database_url'],
                suggestions=["Check database URL and ensure database is running"]
            )
    
    async def _test_redis_connectivity(self, redis_url: str) -> ValidationResult:
        """Test Redis connectivity."""
        try:
            import redis
            
            client = redis.from_url(redis_url)
            client.ping()
            client.close()
            
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.INFO,
                True,
                "Redis connectivity test passed",
                affected_keys=['redis_url']
            )
        
        except Exception as e:
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.ERROR,
                False,
                f"Redis connectivity test failed: {str(e)}",
                affected_keys=['redis_url'],
                suggestions=["Check Redis URL and ensure Redis server is running"]
            )
    
    async def _test_timescale_connectivity(self, config: Dict[str, Any]) -> ValidationResult:
        """Test TimescaleDB connectivity."""
        try:
            from sqlalchemy import create_engine, text
            
            # Build TimescaleDB URL
            host = config.get('timescale_host')
            port = config.get('timescale_port', 5432)
            database = config.get('timescale_database')
            username = config.get('timescale_username')
            password = config.get('timescale_password')
            
            url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            engine = create_engine(url, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Test TimescaleDB extension
                result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"))
                if not result.fetchone():
                    raise Exception("TimescaleDB extension not found")
            
            engine.dispose()
            
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.INFO,
                True,
                "TimescaleDB connectivity test passed",
                affected_keys=['timescale_host', 'timescale_database']
            )
        
        except Exception as e:
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.ERROR,
                False,
                f"TimescaleDB connectivity test failed: {str(e)}",
                affected_keys=['timescale_host', 'timescale_database'],
                suggestions=["Check TimescaleDB configuration and ensure server is running"]
            )
    
    def _validate_webhook_url(self, webhook_url: str) -> ValidationResult:
        """Validate webhook URL format."""
        if not webhook_url.startswith(('http://', 'https://')):
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.ERROR,
                False,
                "Webhook URL must start with http:// or https://",
                affected_keys=['alert_webhook_url'],
                suggestions=["Use proper URL format for webhook"]
            )
        
        # Check for localhost in production
        if 'localhost' in webhook_url or '127.0.0.1' in webhook_url:
            return self.create_result(
                ValidationType.INTEGRATION,
                ValidationSeverity.WARNING,
                False,
                "Webhook URL points to localhost",
                affected_keys=['alert_webhook_url'],
                suggestions=["Use external URL for webhook in production"]
            )
        
        return self.create_result(
            ValidationType.INTEGRATION,
            ValidationSeverity.INFO,
            True,
            "Webhook URL format is valid",
            affected_keys=['alert_webhook_url']
        )


class ConfigurationValidationFramework:
    """Main configuration validation framework."""
    
    def __init__(self):
        self.validators: List[ConfigurationValidator] = [
            TypeValidator(),
            DependencyValidator(),
            SecurityValidator(),
            PerformanceValidator(),
            ComplianceValidator(),
            IntegrationValidator()
        ]
        
        self.validation_history: List[ValidationSummary] = []
    
    def add_validator(self, validator: ConfigurationValidator):
        """Add a custom validator."""
        self.validators.append(validator)
    
    def remove_validator(self, validator_id: str):
        """Remove a validator by ID."""
        self.validators = [v for v in self.validators if v.validator_id != validator_id]
    
    async def validate_configuration(
        self,
        config: Dict[str, Any],
        environment: Environment,
        context: Optional[Dict[str, Any]] = None,
        validator_ids: Optional[List[str]] = None
    ) -> ValidationSummary:
        """Validate configuration with all or specified validators."""
        
        start_time = datetime.utcnow()
        all_results = []
        
        # Filter validators if specific IDs requested
        validators_to_run = self.validators
        if validator_ids:
            validators_to_run = [v for v in self.validators if v.validator_id in validator_ids]
        
        # Run validators concurrently
        tasks = []
        for validator in validators_to_run:
            task = asyncio.create_task(validator.validate(config, environment, context))
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                # Handle validator exceptions
                all_results.append(
                    ValidationResult(
                        check_id=validators_to_run[i].validator_id,
                        validation_type=ValidationType.TYPE_CHECK,
                        severity=ValidationSeverity.CRITICAL,
                        passed=False,
                        message=f"Validator {validators_to_run[i].validator_id} failed: {str(results)}"
                    )
                )
            else:
                all_results.extend(results)
        
        # Calculate summary statistics
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        total_checks = len(all_results)
        passed = sum(1 for r in all_results if r.passed)
        failed = total_checks - passed
        
        # Count by severity
        warnings = sum(1 for r in all_results if r.severity == ValidationSeverity.WARNING)
        errors = sum(1 for r in all_results if r.severity == ValidationSeverity.ERROR)
        critical = sum(1 for r in all_results if r.severity == ValidationSeverity.CRITICAL)
        
        # Create config hash for change detection
        import hashlib
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Create summary
        summary = ValidationSummary(
            total_checks=total_checks,
            passed=passed,
            failed=failed,
            warnings=warnings,
            errors=errors,
            critical=critical,
            execution_time=execution_time,
            results=all_results,
            environment=environment,
            config_hash=config_hash
        )
        
        # Add to history
        self.validation_history.append(summary)
        
        # Keep only last 50 validation runs
        if len(self.validation_history) > 50:
            self.validation_history.pop(0)
        
        return summary
    
    def get_validation_report(
        self,
        summary: ValidationSummary,
        format: str = 'text'
    ) -> str:
        """Generate a validation report in specified format."""
        
        if format == 'json':
            return json.dumps({
                'summary': {
                    'total_checks': summary.total_checks,
                    'passed': summary.passed,
                    'failed': summary.failed,
                    'warnings': summary.warnings,
                    'errors': summary.errors,
                    'critical': summary.critical,
                    'execution_time': summary.execution_time,
                    'environment': summary.environment.value,
                    'timestamp': summary.timestamp.isoformat()
                },
                'results': [
                    {
                        'check_id': r.check_id,
                        'validation_type': r.validation_type.value,
                        'severity': r.severity.value,
                        'passed': r.passed,
                        'message': r.message,
                        'affected_keys': r.affected_keys,
                        'suggestions': r.suggestions
                    }
                    for r in summary.results
                ]
            }, indent=2)
        
        elif format == 'text':
            lines = []
            lines.append("=" * 60)
            lines.append("SUPERNOVA AI CONFIGURATION VALIDATION REPORT")
            lines.append("=" * 60)
            lines.append(f"Environment: {summary.environment.value}")
            lines.append(f"Timestamp: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append(f"Execution Time: {summary.execution_time:.2f}s")
            lines.append("")
            
            # Summary
            lines.append("SUMMARY")
            lines.append("-" * 20)
            lines.append(f"Total Checks: {summary.total_checks}")
            lines.append(f"Passed: {summary.passed}")
            lines.append(f"Failed: {summary.failed}")
            lines.append(f"Warnings: {summary.warnings}")
            lines.append(f"Errors: {summary.errors}")
            lines.append(f"Critical: {summary.critical}")
            lines.append("")
            
            # Group results by severity
            for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                           ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                severity_results = [r for r in summary.results if r.severity == severity]
                if severity_results:
                    lines.append(f"{severity.value.upper()} ISSUES")
                    lines.append("-" * 30)
                    
                    for result in severity_results:
                        status = "✓" if result.passed else "✗"
                        lines.append(f"{status} [{result.check_id}] {result.message}")
                        
                        if result.affected_keys:
                            lines.append(f"   Affected: {', '.join(result.affected_keys)}")
                        
                        if result.suggestions:
                            lines.append("   Suggestions:")
                            for suggestion in result.suggestions:
                                lines.append(f"   - {suggestion}")
                        
                        lines.append("")
                    
                    lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def compare_validations(
        self,
        summary1: ValidationSummary,
        summary2: ValidationSummary
    ) -> Dict[str, Any]:
        """Compare two validation summaries."""
        
        comparison = {
            'timespan': {
                'from': summary1.timestamp.isoformat(),
                'to': summary2.timestamp.isoformat()
            },
            'summary_changes': {
                'total_checks': summary2.total_checks - summary1.total_checks,
                'passed': summary2.passed - summary1.passed,
                'failed': summary2.failed - summary1.failed,
                'warnings': summary2.warnings - summary1.warnings,
                'errors': summary2.errors - summary1.errors,
                'critical': summary2.critical - summary1.critical
            },
            'new_issues': [],
            'resolved_issues': [],
            'persisting_issues': []
        }
        
        # Find new, resolved, and persisting issues
        summary1_issues = {(r.check_id, r.message): r for r in summary1.results if not r.passed}
        summary2_issues = {(r.check_id, r.message): r for r in summary2.results if not r.passed}
        
        for key, result in summary2_issues.items():
            if key not in summary1_issues:
                comparison['new_issues'].append({
                    'check_id': result.check_id,
                    'message': result.message,
                    'severity': result.severity.value
                })
            else:
                comparison['persisting_issues'].append({
                    'check_id': result.check_id,
                    'message': result.message,
                    'severity': result.severity.value
                })
        
        for key, result in summary1_issues.items():
            if key not in summary2_issues:
                comparison['resolved_issues'].append({
                    'check_id': result.check_id,
                    'message': result.message,
                    'severity': result.severity.value
                })
        
        return comparison
    
    def get_validation_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get validation trends over specified days."""
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_validations = [v for v in self.validation_history if v.timestamp > cutoff]
        
        if not recent_validations:
            return {'error': 'No validation data available for the specified period'}
        
        trends = {
            'period_days': days,
            'total_validations': len(recent_validations),
            'trends': {
                'errors': [v.errors for v in recent_validations],
                'warnings': [v.warnings for v in recent_validations],
                'passed_ratio': [v.passed / v.total_checks for v in recent_validations],
                'execution_time': [v.execution_time for v in recent_validations]
            },
            'averages': {
                'errors_per_run': sum(v.errors for v in recent_validations) / len(recent_validations),
                'warnings_per_run': sum(v.warnings for v in recent_validations) / len(recent_validations),
                'pass_rate': sum(v.passed / v.total_checks for v in recent_validations) / len(recent_validations),
                'avg_execution_time': sum(v.execution_time for v in recent_validations) / len(recent_validations)
            }
        }
        
        return trends


# Global validation framework
validation_framework = ConfigurationValidationFramework()


# Convenience functions
async def validate_config(
    config: Dict[str, Any],
    environment: Environment = Environment.DEVELOPMENT
) -> ValidationSummary:
    """Validate configuration using the global framework."""
    return await validation_framework.validate_configuration(config, environment)


async def quick_validate(config: Dict[str, Any], environment: Environment) -> bool:
    """Quick validation returning only pass/fail."""
    summary = await validation_framework.validate_configuration(config, environment)
    return summary.errors == 0 and summary.critical == 0