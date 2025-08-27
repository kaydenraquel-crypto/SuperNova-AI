"""
SuperNova AI Security Configuration
Comprehensive security settings and constants for production deployment
"""

import os
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import timedelta
from pydantic import BaseSettings, Field, validator
from cryptography.fernet import Fernet
import secrets


class SecurityLevel(str, Enum):
    """Security configuration levels"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class SecuritySettings(BaseSettings):
    """
    Comprehensive security configuration for SuperNova AI
    Covers authentication, authorization, encryption, and compliance
    """
    
    # =====================================
    # CORE SECURITY SETTINGS
    # =====================================
    
    SECURITY_LEVEL: SecurityLevel = SecurityLevel.PRODUCTION
    DEBUG_MODE: bool = Field(default=False, env="SUPERNOVA_DEBUG")
    
    # Security Keys and Secrets
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env="SUPERNOVA_ENCRYPTION_KEY")
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    CSRF_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # =====================================
    # JWT AUTHENTICATION
    # =====================================
    
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_ISSUER: str = "supernova-ai"
    JWT_AUDIENCE: str = "supernova-clients"
    JWT_REQUIRE_CLAIMS: List[str] = ["sub", "exp", "iat", "aud", "iss"]
    JWT_BLACKLIST_ENABLED: bool = True
    JWT_BLACKLIST_TOKEN_CHECKS: List[str] = ["access", "refresh"]
    
    # Multi-Factor Authentication
    MFA_ENABLED: bool = True
    MFA_ISSUER_NAME: str = "SuperNova AI"
    MFA_BACKUP_CODES_COUNT: int = 10
    TOTP_DRIFT_SECONDS: int = 30
    
    # =====================================
    # PASSWORD SECURITY
    # =====================================
    
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGITS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    PASSWORD_SPECIAL_CHARS: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    PASSWORD_HASH_ROUNDS: int = 12
    PASSWORD_HISTORY_COUNT: int = 10
    PASSWORD_MAX_AGE_DAYS: int = 90
    
    # Account Security
    LOGIN_MAX_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30
    LOCKOUT_ESCALATION_ENABLED: bool = True
    SESSION_TIMEOUT_MINUTES: int = 30
    CONCURRENT_SESSIONS_LIMIT: int = 3
    
    # =====================================
    # API SECURITY
    # =====================================
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    RATE_LIMIT_BURST: int = 20
    
    # API Keys
    API_KEY_LENGTH: int = 32
    API_KEY_PREFIX: str = "sn_"
    API_KEY_EXPIRY_DAYS: int = 365
    API_KEY_ROTATION_ENABLED: bool = True
    
    # Request Security
    MAX_REQUEST_SIZE_MB: int = 10
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_CONTENT_TYPES: List[str] = [
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
        "text/csv",
        "application/pdf"
    ]
    
    # =====================================
    # DATA ENCRYPTION
    # =====================================
    
    # Encryption Settings
    ENCRYPTION_ALGORITHM: str = "AES-256-GCM"
    FIELD_ENCRYPTION_ENABLED: bool = True
    DATABASE_ENCRYPTION_ENABLED: bool = True
    BACKUP_ENCRYPTION_ENABLED: bool = True
    
    # Key Management
    KEY_ROTATION_ENABLED: bool = True
    KEY_ROTATION_DAYS: int = 30
    KEY_DERIVATION_ITERATIONS: int = 100000
    
    # Sensitive Data Fields
    ENCRYPTED_FIELDS: List[str] = [
        "ssn", "tax_id", "bank_account", "credit_card",
        "api_keys", "passwords", "personal_notes"
    ]
    
    # =====================================
    # NETWORK SECURITY
    # =====================================
    
    # SSL/TLS
    SSL_REQUIRED: bool = True
    TLS_VERSION_MIN: str = "1.2"
    TLS_VERSION_PREFERRED: str = "1.3"
    HSTS_MAX_AGE: int = 31536000  # 1 year
    HSTS_INCLUDE_SUBDOMAINS: bool = True
    
    # CORS Configuration
    CORS_ALLOWED_ORIGINS: List[str] = ["https://dashboard.supernova-ai.com"]
    CORS_ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOWED_HEADERS: List[str] = [
        "Authorization", "Content-Type", "X-API-Key", "X-Request-ID",
        "X-CSRF-Token", "X-Client-Version"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_MAX_AGE: int = 86400
    
    # IP Security
    IP_WHITELIST_ENABLED: bool = False
    IP_WHITELIST: List[str] = []
    IP_BLACKLIST_ENABLED: bool = True
    IP_BLACKLIST: List[str] = []
    GEO_BLOCKING_ENABLED: bool = False
    ALLOWED_COUNTRIES: List[str] = ["US", "CA", "GB", "EU"]
    
    # =====================================
    # WEBSOCKET SECURITY
    # =====================================
    
    WS_AUTHENTICATION_REQUIRED: bool = True
    WS_MAX_CONNECTIONS_PER_USER: int = 5
    WS_MESSAGE_SIZE_LIMIT: int = 64 * 1024  # 64KB
    WS_RATE_LIMIT_MESSAGES_PER_SECOND: int = 10
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_CONNECTION_TIMEOUT: int = 300
    
    # =====================================
    # DATABASE SECURITY
    # =====================================
    
    DB_SSL_REQUIRED: bool = True
    DB_CONNECTION_ENCRYPTION: bool = True
    DB_QUERY_TIMEOUT: int = 30
    DB_CONNECTION_POOL_SIZE: int = 10
    DB_CONNECTION_POOL_MAX: int = 20
    DB_AUDIT_ENABLED: bool = True
    
    # Query Protection
    SQL_INJECTION_PROTECTION: bool = True
    NOSQL_INJECTION_PROTECTION: bool = True
    QUERY_COMPLEXITY_LIMIT: int = 1000
    RESULT_SIZE_LIMIT: int = 10000
    
    # =====================================
    # LOGGING AND MONITORING
    # =====================================
    
    SECURITY_LOGGING_ENABLED: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    SECURITY_LOG_LEVEL: str = "INFO"
    LOG_SENSITIVE_DATA: bool = False
    LOG_ENCRYPTION_ENABLED: bool = True
    
    # SIEM Integration
    SIEM_ENABLED: bool = False
    SIEM_ENDPOINT: Optional[str] = None
    SIEM_API_KEY: Optional[str] = None
    
    # Alerting
    SECURITY_ALERTS_ENABLED: bool = True
    ALERT_WEBHOOK_URL: Optional[str] = None
    ALERT_EMAIL_ENABLED: bool = True
    ALERT_EMAIL_RECIPIENTS: List[str] = []
    
    # =====================================
    # COMPLIANCE SETTINGS
    # =====================================
    
    # Regulatory Compliance
    GDPR_COMPLIANCE_ENABLED: bool = True
    CCPA_COMPLIANCE_ENABLED: bool = True
    SOC2_COMPLIANCE_ENABLED: bool = True
    PCI_DSS_COMPLIANCE_ENABLED: bool = True
    FINRA_COMPLIANCE_ENABLED: bool = True
    
    # Data Retention
    USER_DATA_RETENTION_DAYS: int = 2555  # 7 years
    TRANSACTION_DATA_RETENTION_DAYS: int = 2555  # 7 years
    LOG_DATA_RETENTION_DAYS: int = 2555  # 7 years
    BACKUP_RETENTION_DAYS: int = 2555  # 7 years
    
    # Privacy Settings
    DATA_ANONYMIZATION_ENABLED: bool = True
    RIGHT_TO_BE_FORGOTTEN_ENABLED: bool = True
    CONSENT_MANAGEMENT_ENABLED: bool = True
    COOKIE_CONSENT_REQUIRED: bool = True
    
    # =====================================
    # INCIDENT RESPONSE
    # =====================================
    
    INCIDENT_RESPONSE_ENABLED: bool = True
    AUTO_INCIDENT_CREATION: bool = True
    INCIDENT_SEVERITY_LEVELS: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    INCIDENT_NOTIFICATION_THRESHOLD: str = "MEDIUM"
    
    # Breach Notification
    BREACH_NOTIFICATION_ENABLED: bool = True
    BREACH_NOTIFICATION_HOURS: int = 72
    BREACH_ASSESSMENT_REQUIRED: bool = True
    
    # =====================================
    # BUSINESS CONTINUITY
    # =====================================
    
    BACKUP_ENABLED: bool = True
    BACKUP_FREQUENCY_HOURS: int = 6
    BACKUP_ENCRYPTION_REQUIRED: bool = True
    DISASTER_RECOVERY_ENABLED: bool = True
    RTO_TARGET_HOURS: int = 4  # Recovery Time Objective
    RPO_TARGET_HOURS: int = 1  # Recovery Point Objective
    
    # =====================================
    # VALIDATION AND INITIALIZATION
    # =====================================
    
    @validator('ENCRYPTION_KEY', pre=True)
    def validate_encryption_key(cls, v):
        if v is None:
            # Generate a new encryption key
            return Fernet.generate_key().decode()
        return v
    
    @validator('CORS_ALLOWED_ORIGINS')
    def validate_cors_origins(cls, v, values):
        security_level = values.get('SECURITY_LEVEL')
        if security_level == SecurityLevel.PRODUCTION and "*" in v:
            raise ValueError("Wildcard CORS origins not allowed in production")
        return v
    
    @validator('DEBUG_MODE')
    def validate_debug_mode(cls, v, values):
        security_level = values.get('SECURITY_LEVEL')
        if security_level == SecurityLevel.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    def get_encryption_key(self) -> bytes:
        """Get the encryption key as bytes"""
        return self.ENCRYPTION_KEY.encode() if isinstance(self.ENCRYPTION_KEY, str) else self.ENCRYPTION_KEY
    
    def get_fernet_cipher(self) -> Fernet:
        """Get a Fernet cipher instance"""
        return Fernet(self.get_encryption_key())
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.SECURITY_LEVEL in [SecurityLevel.PRODUCTION, SecurityLevel.ENTERPRISE]
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration as dictionary"""
        return {
            "secret_key": self.JWT_SECRET_KEY,
            "algorithm": self.JWT_ALGORITHM,
            "access_token_expire": timedelta(minutes=self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "refresh_token_expire": timedelta(days=self.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            "issuer": self.JWT_ISSUER,
            "audience": self.JWT_AUDIENCE,
        }
    
    def get_rate_limit_config(self) -> Dict[str, int]:
        """Get rate limiting configuration"""
        return {
            "per_minute": self.RATE_LIMIT_PER_MINUTE,
            "per_hour": self.RATE_LIMIT_PER_HOUR,
            "per_day": self.RATE_LIMIT_PER_DAY,
            "burst": self.RATE_LIMIT_BURST,
        }
    
    class Config:
        env_prefix = "SUPERNOVA_SECURITY_"
        env_file = ".env"
        case_sensitive = True


# Global security settings instance
security_settings = SecuritySettings()


# Security constants
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": f"max-age={security_settings.HSTS_MAX_AGE}; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' wss: https:; "
        "font-src 'self' https:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    ),
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# Sensitive fields that should be encrypted
SENSITIVE_FIELD_PATTERNS = [
    r"password", r"secret", r"key", r"token", r"api_key",
    r"ssn", r"tax_id", r"bank_account", r"credit_card",
    r"private_key", r"auth", r"credential"
]

# Security event types for logging
SECURITY_EVENT_TYPES = {
    "AUTHENTICATION_SUCCESS": "auth_success",
    "AUTHENTICATION_FAILURE": "auth_failure",
    "AUTHORIZATION_DENIED": "authz_denied",
    "PASSWORD_CHANGE": "password_change",
    "ACCOUNT_LOCKED": "account_locked",
    "SUSPICIOUS_ACTIVITY": "suspicious_activity",
    "DATA_ACCESS": "data_access",
    "ADMIN_ACTION": "admin_action",
    "SECURITY_VIOLATION": "security_violation",
    "ENCRYPTION_ERROR": "encryption_error",
}

# Risk levels for security events
RISK_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def get_security_level_config(level: SecurityLevel) -> Dict[str, Any]:
    """Get security configuration based on security level"""
    configs = {
        SecurityLevel.DEVELOPMENT: {
            "rate_limit_enabled": False,
            "ssl_required": False,
            "mfa_enabled": False,
            "audit_enabled": False,
        },
        SecurityLevel.STAGING: {
            "rate_limit_enabled": True,
            "ssl_required": True,
            "mfa_enabled": False,
            "audit_enabled": True,
        },
        SecurityLevel.PRODUCTION: {
            "rate_limit_enabled": True,
            "ssl_required": True,
            "mfa_enabled": True,
            "audit_enabled": True,
        },
        SecurityLevel.ENTERPRISE: {
            "rate_limit_enabled": True,
            "ssl_required": True,
            "mfa_enabled": True,
            "audit_enabled": True,
            "advanced_threat_protection": True,
            "zero_trust_enabled": True,
        },
    }
    return configs.get(level, configs[SecurityLevel.PRODUCTION])