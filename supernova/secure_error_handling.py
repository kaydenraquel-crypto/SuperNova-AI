"""
SuperNova AI Secure Error Handling and Comprehensive Security Logging
Production-grade error handling with security-aware logging and incident management
"""

import os
import sys
import traceback
import json
import hashlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import logging
import asyncio
from functools import wraps
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
import uuid

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk
from sentry_sdk import capture_exception, capture_message, set_tag, set_context

from .security_config import security_settings
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SECURITY = "security"
    SYSTEM = "system"
    NETWORK = "network"
    PERFORMANCE = "performance"


class SecurityRisk(str, Enum):
    """Security risk levels for errors"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SecurityError:
    """Security-aware error information"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    security_risk: SecurityRisk
    context: ErrorContext
    stack_trace: Optional[str] = None
    sensitive_data_exposed: bool = False
    public_message: str = "An error occurred"
    remediation_steps: Optional[List[str]] = None
    related_errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.context.timestamp.isoformat()
        return result


class SecureErrorHandler:
    """
    Secure error handling with comprehensive logging and incident management
    """
    
    def __init__(self):
        self.error_cache: Dict[str, SecurityError] = {}
        self.error_patterns: Dict[str, Dict] = {}
        self.error_frequencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.incident_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.LOW: 100
        }
        
        # Initialize Sentry if configured
        self._init_sentry()
        
        # Error classification patterns
        self.error_classifiers = {
            # Authentication errors
            (ErrorCategory.AUTHENTICATION, SecurityRisk.HIGH): [
                "invalid.*password", "authentication.*failed", "invalid.*credentials",
                "token.*expired", "unauthorized.*access", "invalid.*session"
            ],
            
            # Authorization errors
            (ErrorCategory.AUTHORIZATION, SecurityRisk.MEDIUM): [
                "permission.*denied", "access.*forbidden", "insufficient.*privileges",
                "role.*required", "unauthorized.*operation"
            ],
            
            # Validation errors
            (ErrorCategory.VALIDATION, SecurityRisk.LOW): [
                "validation.*failed", "invalid.*input", "missing.*required",
                "format.*invalid", "constraint.*violation"
            ],
            
            # Security errors
            (ErrorCategory.SECURITY, SecurityRisk.CRITICAL): [
                "injection.*detected", "attack.*detected", "malicious.*input",
                "security.*violation", "suspicious.*activity", "rate.*limit.*exceeded"
            ],
            
            # Database errors
            (ErrorCategory.DATABASE, SecurityRisk.MEDIUM): [
                "database.*connection", "sql.*error", "query.*timeout",
                "deadlock.*detected", "connection.*pool.*exhausted"
            ],
            
            # System errors
            (ErrorCategory.SYSTEM, SecurityRisk.HIGH): [
                "out.*of.*memory", "disk.*space", "service.*unavailable",
                "timeout", "connection.*refused", "system.*overload"
            ]
        }
        
        # Sensitive data patterns to redact
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',
            r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',
            r'token["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',
            r'(4[0-9]{12}(?:[0-9]{3})?)',  # Credit card
            r'([0-9]{3}-[0-9]{2}-[0-9]{4})',  # SSN
        ]
        
        # Public error messages
        self.public_error_messages = {
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.VALIDATION: "Invalid input provided. Please check your data.",
            ErrorCategory.BUSINESS_LOGIC: "Unable to process request due to business rules.",
            ErrorCategory.DATABASE: "Database service temporarily unavailable.",
            ErrorCategory.EXTERNAL_API: "External service temporarily unavailable.",
            ErrorCategory.SECURITY: "Request blocked for security reasons.",
            ErrorCategory.SYSTEM: "System temporarily unavailable. Please try again later.",
            ErrorCategory.NETWORK: "Network connectivity issue. Please try again.",
            ErrorCategory.PERFORMANCE: "Request timed out. Please try again."
        }
        
        # Start background tasks
        asyncio.create_task(self._error_analysis_task())
        asyncio.create_task(self._incident_detection_task())
    
    def _init_sentry(self):
        """Initialize Sentry error tracking if configured"""
        sentry_dsn = os.getenv('SENTRY_DSN')
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                traces_sample_rate=0.1,
                environment=security_settings.SECURITY_LEVEL.value,
                before_send=self._filter_sentry_event
            )
            logger.info("Sentry error tracking initialized")
    
    def _filter_sentry_event(self, event, hint):
        """Filter Sentry events to prevent sensitive data exposure"""
        
        # Remove sensitive data from event
        if 'extra' in event:
            event['extra'] = self._redact_sensitive_data(event['extra'])
        
        if 'contexts' in event:
            for context_name, context_data in event['contexts'].items():
                if isinstance(context_data, dict):
                    event['contexts'][context_name] = self._redact_sensitive_data(context_data)
        
        return event
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        public_message: Optional[str] = None
    ) -> SecurityError:
        """
        Centralized error handling with security awareness
        """
        
        error_id = str(uuid.uuid4())
        
        # Create context if not provided
        if context is None:
            context = ErrorContext(error_id=error_id, timestamp=datetime.now())
        else:\n            context.error_id = error_id\n        \n        # Classify error if not specified\n        if category is None or severity is None:\n            classified_category, classified_severity, security_risk = self._classify_error(error)\n            category = category or classified_category\n            severity = severity or classified_severity\n        else:\n            security_risk = self._assess_security_risk(error, category)\n        \n        # Get stack trace\n        stack_trace = self._get_safe_stack_trace(error)\n        \n        # Determine public message\n        if public_message is None:\n            public_message = self.public_error_messages.get(\n                category, \"An unexpected error occurred\"\n            )\n        \n        # Check for sensitive data exposure\n        sensitive_exposed = self._check_sensitive_data_exposure(error, stack_trace)\n        \n        # Create security error\n        security_error = SecurityError(\n            error_id=error_id,\n            error_type=type(error).__name__,\n            error_message=str(error),\n            severity=severity,\n            category=category,\n            security_risk=security_risk,\n            context=context,\n            stack_trace=stack_trace,\n            sensitive_data_exposed=sensitive_exposed,\n            public_message=public_message,\n            remediation_steps=self._get_remediation_steps(category, error)\n        )\n        \n        # Store error\n        self.error_cache[error_id] = security_error\n        \n        # Update error patterns\n        self._update_error_patterns(security_error)\n        \n        # Log error\n        self._log_security_error(security_error)\n        \n        # Send to external monitoring if critical\n        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:\n            self._send_to_external_monitoring(security_error)\n        \n        # Update error frequencies for pattern detection\n        error_key = f\"{category.value}:{type(error).__name__}\"\n        self.error_frequencies[error_key].append(datetime.now())\n        \n        return security_error\n    \n    def _classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity, SecurityRisk]:\n        \"\"\"Classify error based on type and message\"\"\"\n        \n        error_msg = str(error).lower()\n        error_type = type(error).__name__.lower()\n        \n        # Check against classification patterns\n        for (category, risk), patterns in self.error_classifiers.items():\n            for pattern in patterns:\n                if (re.search(pattern, error_msg) or \n                    re.search(pattern, error_type)):\n                    \n                    # Determine severity based on security risk\n                    if risk == SecurityRisk.CRITICAL:\n                        severity = ErrorSeverity.CRITICAL\n                    elif risk == SecurityRisk.HIGH:\n                        severity = ErrorSeverity.HIGH\n                    elif risk == SecurityRisk.MEDIUM:\n                        severity = ErrorSeverity.MEDIUM\n                    else:\n                        severity = ErrorSeverity.LOW\n                    \n                    return category, severity, risk\n        \n        # Default classification\n        if isinstance(error, HTTPException):\n            if error.status_code >= 500:\n                return ErrorCategory.SYSTEM, ErrorSeverity.HIGH, SecurityRisk.MEDIUM\n            elif error.status_code >= 400:\n                return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, SecurityRisk.LOW\n        \n        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM, SecurityRisk.LOW\n    \n    def _assess_security_risk(self, error: Exception, category: ErrorCategory) -> SecurityRisk:\n        \"\"\"Assess security risk level for error\"\"\"\n        \n        if category == ErrorCategory.SECURITY:\n            return SecurityRisk.CRITICAL\n        elif category == ErrorCategory.AUTHENTICATION:\n            return SecurityRisk.HIGH\n        elif category == ErrorCategory.AUTHORIZATION:\n            return SecurityRisk.MEDIUM\n        elif category in [ErrorCategory.DATABASE, ErrorCategory.SYSTEM]:\n            return SecurityRisk.MEDIUM\n        else:\n            return SecurityRisk.LOW\n    \n    def _get_safe_stack_trace(self, error: Exception) -> str:\n        \"\"\"Get stack trace with sensitive data redacted\"\"\"\n        \n        try:\n            # Get full stack trace\n            tb_lines = traceback.format_exception(type(error), error, error.__traceback__)\n            stack_trace = ''.join(tb_lines)\n            \n            # Redact sensitive data\n            redacted_trace = self._redact_sensitive_data({'stack_trace': stack_trace})\n            \n            return redacted_trace['stack_trace']\n            \n        except Exception:\n            return f\"Stack trace unavailable for {type(error).__name__}\"\n    \n    def _redact_sensitive_data(self, data: Any) -> Any:\n        \"\"\"Redact sensitive data from error information\"\"\"\n        \n        if isinstance(data, dict):\n            redacted = {}\n            for key, value in data.items():\n                if self._is_sensitive_key(key):\n                    redacted[key] = \"[REDACTED]\"\n                else:\n                    redacted[key] = self._redact_sensitive_data(value)\n            return redacted\n        \n        elif isinstance(data, list):\n            return [self._redact_sensitive_data(item) for item in data]\n        \n        elif isinstance(data, str):\n            # Apply regex patterns to redact sensitive data\n            redacted_str = data\n            for pattern in self.sensitive_patterns:\n                redacted_str = re.sub(pattern, r'\\1[REDACTED]', redacted_str, flags=re.IGNORECASE)\n            return redacted_str\n        \n        else:\n            return data\n    \n    def _is_sensitive_key(self, key: str) -> bool:\n        \"\"\"Check if a key name indicates sensitive data\"\"\"\n        sensitive_keys = {\n            'password', 'secret', 'key', 'token', 'auth', 'credential',\n            'ssn', 'social_security', 'credit_card', 'bank_account',\n            'api_key', 'private_key', 'session_id'\n        }\n        return key.lower() in sensitive_keys\n    \n    def _check_sensitive_data_exposure(self, error: Exception, stack_trace: str) -> bool:\n        \"\"\"Check if error might expose sensitive data\"\"\"\n        \n        error_str = str(error).lower()\n        \n        # Check for common patterns that might expose sensitive data\n        exposure_patterns = [\n            r'password.*=.*\\w+',\n            r'key.*=.*\\w+',\n            r'token.*=.*\\w+',\n            r'\\d{4}-\\d{4}-\\d{4}-\\d{4}',  # Credit card pattern\n            r'\\d{3}-\\d{2}-\\d{4}',       # SSN pattern\n        ]\n        \n        for pattern in exposure_patterns:\n            if re.search(pattern, error_str, re.IGNORECASE):\n                return True\n        \n        return False\n    \n    def _get_remediation_steps(self, category: ErrorCategory, error: Exception) -> List[str]:\n        \"\"\"Get remediation steps for error category\"\"\"\n        \n        remediation_map = {\n            ErrorCategory.AUTHENTICATION: [\n                \"Verify authentication credentials\",\n                \"Check token expiration\",\n                \"Ensure proper authentication flow\"\n            ],\n            ErrorCategory.AUTHORIZATION: [\n                \"Verify user permissions\",\n                \"Check role assignments\",\n                \"Review access control policies\"\n            ],\n            ErrorCategory.VALIDATION: [\n                \"Validate input format\",\n                \"Check required fields\",\n                \"Review data constraints\"\n            ],\n            ErrorCategory.SECURITY: [\n                \"Review security logs\",\n                \"Check for attack patterns\",\n                \"Implement additional security measures\",\n                \"Consider IP blocking if malicious\"\n            ],\n            ErrorCategory.DATABASE: [\n                \"Check database connectivity\",\n                \"Review query performance\",\n                \"Monitor connection pool\"\n            ],\n            ErrorCategory.SYSTEM: [\n                \"Check system resources\",\n                \"Review system logs\",\n                \"Monitor service health\"\n            ]\n        }\n        \n        return remediation_map.get(category, [\"Contact system administrator\"])\n    \n    def _update_error_patterns(self, security_error: SecurityError):\n        \"\"\"Update error patterns for trend analysis\"\"\"\n        \n        pattern_key = f\"{security_error.category.value}:{security_error.error_type}\"\n        \n        if pattern_key not in self.error_patterns:\n            self.error_patterns[pattern_key] = {\n                'count': 0,\n                'first_seen': security_error.context.timestamp,\n                'last_seen': security_error.context.timestamp,\n                'severity_distribution': defaultdict(int),\n                'endpoints': defaultdict(int),\n                'users': set(),\n                'ips': set()\n            }\n        \n        pattern = self.error_patterns[pattern_key]\n        pattern['count'] += 1\n        pattern['last_seen'] = security_error.context.timestamp\n        pattern['severity_distribution'][security_error.severity.value] += 1\n        \n        if security_error.context.endpoint:\n            pattern['endpoints'][security_error.context.endpoint] += 1\n        \n        if security_error.context.user_id:\n            pattern['users'].add(security_error.context.user_id)\n        \n        if security_error.context.client_ip:\n            pattern['ips'].add(security_error.context.client_ip)\n    \n    def _log_security_error(self, security_error: SecurityError):\n        \"\"\"Log security error with appropriate level and detail\"\"\"\n        \n        # Determine log level\n        if security_error.severity == ErrorSeverity.CRITICAL:\n            log_level = SecurityEventLevel.CRITICAL\n        elif security_error.severity == ErrorSeverity.HIGH:\n            log_level = SecurityEventLevel.ERROR\n        elif security_error.severity == ErrorSeverity.MEDIUM:\n            log_level = SecurityEventLevel.WARNING\n        else:\n            log_level = SecurityEventLevel.INFO\n        \n        # Log to security logger\n        security_logger.log_security_event(\n            event_type=SECURITY_EVENT_TYPES.get(\"SECURITY_VIOLATION\", \"ERROR_OCCURRED\"),\n            level=log_level,\n            client_ip=security_error.context.client_ip,\n            user_id=security_error.context.user_id,\n            endpoint=security_error.context.endpoint,\n            details={\n                \"error_id\": security_error.error_id,\n                \"error_type\": security_error.error_type,\n                \"category\": security_error.category.value,\n                \"severity\": security_error.severity.value,\n                \"security_risk\": security_error.security_risk.value,\n                \"sensitive_data_exposed\": security_error.sensitive_data_exposed,\n                \"redacted_message\": self._redact_sensitive_data(security_error.error_message)[:500]\n            }\n        )\n    \n    def _send_to_external_monitoring(self, security_error: SecurityError):\n        \"\"\"Send critical errors to external monitoring systems\"\"\"\n        \n        try:\n            # Send to Sentry\n            if sentry_sdk.Hub.current.client:\n                with sentry_sdk.configure_scope() as scope:\n                    scope.set_tag(\"error_category\", security_error.category.value)\n                    scope.set_tag(\"security_risk\", security_error.security_risk.value)\n                    scope.set_tag(\"error_id\", security_error.error_id)\n                    \n                    scope.set_context(\"error_context\", {\n                        \"user_id\": security_error.context.user_id,\n                        \"client_ip\": security_error.context.client_ip,\n                        \"endpoint\": security_error.context.endpoint,\n                        \"method\": security_error.context.method\n                    })\n                    \n                    if security_error.severity == ErrorSeverity.CRITICAL:\n                        sentry_sdk.capture_message(\n                            f\"Critical Error: {security_error.error_type}\",\n                            level=\"error\"\n                        )\n                    else:\n                        sentry_sdk.capture_message(\n                            f\"High Severity Error: {security_error.error_type}\",\n                            level=\"warning\"\n                        )\n        \n        except Exception as e:\n            logger.error(f\"Failed to send error to external monitoring: {e}\")\n    \n    async def _error_analysis_task(self):\n        \"\"\"Background task to analyze error patterns\"\"\"\n        \n        while True:\n            try:\n                current_time = datetime.now()\n                \n                # Analyze error patterns for anomalies\n                for pattern_key, pattern_data in self.error_patterns.items():\n                    # Check for error spikes\n                    if pattern_data['count'] > 50:  # Threshold for investigation\n                        recent_count = sum(1 for ts in self.error_frequencies[pattern_key]\n                                         if (current_time - ts).seconds < 300)  # Last 5 minutes\n                        \n                        if recent_count > 10:  # Spike detected\n                            security_logger.log_security_event(\n                                event_type=\"ERROR_SPIKE_DETECTED\",\n                                level=SecurityEventLevel.WARNING,\n                                details={\n                                    \"pattern\": pattern_key,\n                                    \"recent_count\": recent_count,\n                                    \"total_count\": pattern_data['count'],\n                                    \"affected_ips\": list(pattern_data['ips'])[:10],\n                                    \"affected_endpoints\": dict(list(pattern_data['endpoints'].items())[:5])\n                                }\n                            )\n                \n                await asyncio.sleep(300)  # Run every 5 minutes\n                \n            except Exception as e:\n                logger.error(f\"Error in error analysis task: {e}\")\n                await asyncio.sleep(60)\n    \n    async def _incident_detection_task(self):\n        \"\"\"Background task for incident detection and escalation\"\"\"\n        \n        while True:\n            try:\n                current_time = datetime.now()\n                time_window = current_time - timedelta(minutes=15)\n                \n                # Count recent errors by severity\n                recent_errors = {\n                    severity: 0 for severity in ErrorSeverity\n                }\n                \n                for error in self.error_cache.values():\n                    if error.context.timestamp > time_window:\n                        recent_errors[error.severity] += 1\n                \n                # Check thresholds for incident creation\n                for severity, count in recent_errors.items():\n                    threshold = self.incident_thresholds[severity]\n                    if count >= threshold:\n                        await self._create_incident(severity, count, time_window)\n                \n                await asyncio.sleep(300)  # Run every 5 minutes\n                \n            except Exception as e:\n                logger.error(f\"Error in incident detection task: {e}\")\n                await asyncio.sleep(60)\n    \n    async def _create_incident(self, severity: ErrorSeverity, error_count: int, time_window: datetime):\n        \"\"\"Create incident for error threshold breach\"\"\"\n        \n        incident_id = str(uuid.uuid4())\n        \n        incident_data = {\n            \"incident_id\": incident_id,\n            \"severity\": severity.value,\n            \"error_count\": error_count,\n            \"time_window_start\": time_window.isoformat(),\n            \"created_at\": datetime.now().isoformat(),\n            \"status\": \"open\",\n            \"auto_created\": True\n        }\n        \n        security_logger.log_security_event(\n            event_type=\"INCIDENT_CREATED\",\n            level=SecurityEventLevel.ERROR if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else SecurityEventLevel.WARNING,\n            details=incident_data\n        )\n        \n        logger.error(f\"Incident {incident_id} created: {error_count} {severity.value} errors in 15 minutes\")\n    \n    def get_error_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive error statistics\"\"\"\n        \n        current_time = datetime.now()\n        \n        # Basic counts\n        total_errors = len(self.error_cache)\n        \n        # Severity distribution\n        severity_counts = defaultdict(int)\n        category_counts = defaultdict(int)\n        risk_counts = defaultdict(int)\n        \n        for error in self.error_cache.values():\n            severity_counts[error.severity.value] += 1\n            category_counts[error.category.value] += 1\n            risk_counts[error.security_risk.value] += 1\n        \n        # Recent errors (last hour)\n        recent_cutoff = current_time - timedelta(hours=1)\n        recent_errors = sum(1 for error in self.error_cache.values()\n                          if error.context.timestamp > recent_cutoff)\n        \n        # Top error patterns\n        top_patterns = sorted(\n            [(k, v['count']) for k, v in self.error_patterns.items()],\n            key=lambda x: x[1],\n            reverse=True\n        )[:10]\n        \n        return {\n            \"total_errors\": total_errors,\n            \"recent_errors_1h\": recent_errors,\n            \"severity_distribution\": dict(severity_counts),\n            \"category_distribution\": dict(category_counts),\n            \"security_risk_distribution\": dict(risk_counts),\n            \"top_error_patterns\": top_patterns,\n            \"error_rate_per_minute\": recent_errors / 60,\n            \"patterns_tracked\": len(self.error_patterns)\n        }\n    \n    def create_error_response(\n        self,\n        security_error: SecurityError,\n        include_debug_info: bool = False\n    ) -> JSONResponse:\n        \"\"\"Create secure error response for API\"\"\"\n        \n        # Base response\n        response_data = {\n            \"error\": True,\n            \"error_id\": security_error.error_id,\n            \"message\": security_error.public_message,\n            \"timestamp\": security_error.context.timestamp.isoformat()\n        }\n        \n        # Add debug info in development mode\n        if include_debug_info and not security_settings.is_production():\n            response_data[\"debug\"] = {\n                \"error_type\": security_error.error_type,\n                \"category\": security_error.category.value,\n                \"severity\": security_error.severity.value\n            }\n        \n        # Add remediation steps for certain errors\n        if security_error.category in [ErrorCategory.VALIDATION, ErrorCategory.AUTHENTICATION]:\n            response_data[\"remediation\"] = security_error.remediation_steps\n        \n        # Determine HTTP status code\n        status_code_map = {\n            ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,\n            ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,\n            ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,\n            ErrorCategory.SECURITY: status.HTTP_403_FORBIDDEN,\n            ErrorCategory.DATABASE: status.HTTP_503_SERVICE_UNAVAILABLE,\n            ErrorCategory.EXTERNAL_API: status.HTTP_503_SERVICE_UNAVAILABLE,\n            ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,\n            ErrorCategory.NETWORK: status.HTTP_503_SERVICE_UNAVAILABLE,\n            ErrorCategory.PERFORMANCE: status.HTTP_408_REQUEST_TIMEOUT\n        }\n        \n        status_code = status_code_map.get(security_error.category, status.HTTP_500_INTERNAL_SERVER_ERROR)\n        \n        return JSONResponse(\n            content=response_data,\n            status_code=status_code,\n            headers={\n                \"X-Error-ID\": security_error.error_id,\n                \"X-Error-Category\": security_error.category.value\n            }\n        )\n\n\nclass SecureErrorMiddleware(BaseHTTPMiddleware):\n    \"\"\"Middleware for secure error handling\"\"\"\n    \n    def __init__(self, app, error_handler: SecureErrorHandler = None):\n        super().__init__(app)\n        self.error_handler = error_handler or SecureErrorHandler()\n    \n    async def dispatch(self, request: Request, call_next):\n        try:\n            return await call_next(request)\n        except Exception as error:\n            # Create error context\n            context = ErrorContext(\n                error_id=str(uuid.uuid4()),\n                timestamp=datetime.now(),\n                client_ip=self._get_client_ip(request),\n                user_agent=request.headers.get(\"user-agent\"),\n                endpoint=str(request.url.path),\n                method=request.method,\n                headers=dict(request.headers)\n            )\n            \n            # Handle error\n            security_error = self.error_handler.handle_error(error, context)\n            \n            # Return secure response\n            return self.error_handler.create_error_response(security_error)\n    \n    def _get_client_ip(self, request: Request) -> str:\n        \"\"\"Extract client IP address\"\"\"\n        forwarded_for = request.headers.get(\"x-forwarded-for\")\n        if forwarded_for:\n            return forwarded_for.split(\",\")[0].strip()\n        \n        real_ip = request.headers.get(\"x-real-ip\")\n        if real_ip:\n            return real_ip\n        \n        return request.client.host if request.client else \"unknown\"\n\n\n# Decorator for secure error handling\ndef secure_error_handler(category: ErrorCategory = None, public_message: str = None):\n    \"\"\"Decorator for automatic secure error handling\"\"\"\n    \n    def decorator(func: Callable) -> Callable:\n        @wraps(func)\n        async def async_wrapper(*args, **kwargs):\n            try:\n                return await func(*args, **kwargs)\n            except Exception as error:\n                error_handler = SecureErrorHandler()\n                context = ErrorContext(\n                    error_id=str(uuid.uuid4()),\n                    timestamp=datetime.now()\n                )\n                \n                security_error = error_handler.handle_error(\n                    error, context, category=category, public_message=public_message\n                )\n                \n                # Re-raise as HTTPException for FastAPI\n                response = error_handler.create_error_response(security_error)\n                raise HTTPException(\n                    status_code=response.status_code,\n                    detail=response.body.decode()\n                )\n        \n        @wraps(func)\n        def sync_wrapper(*args, **kwargs):\n            try:\n                return func(*args, **kwargs)\n            except Exception as error:\n                error_handler = SecureErrorHandler()\n                context = ErrorContext(\n                    error_id=str(uuid.uuid4()),\n                    timestamp=datetime.now()\n                )\n                \n                security_error = error_handler.handle_error(\n                    error, context, category=category, public_message=public_message\n                )\n                \n                # Log and re-raise\n                logger.error(f\"Error {security_error.error_id}: {security_error.public_message}\")\n                raise\n        \n        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper\n    \n    return decorator\n\n\n# Global instance\nsecure_error_handler = SecureErrorHandler()"}}]