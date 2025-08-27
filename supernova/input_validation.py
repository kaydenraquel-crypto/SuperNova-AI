"""
SuperNova AI Input Validation and Sanitization System
Comprehensive protection against injection attacks and malicious input
"""

import re
import html
import json
import base64
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
import bleach
from pydantic import BaseModel, validator, ValidationError
import sqlparse
from fastapi import HTTPException, status
import logging

from .security_config import security_settings
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class SanitizationError(Exception):
    """Custom sanitization error"""
    pass


class InputValidator:
    """
    Comprehensive input validation and sanitization system
    Protects against SQL injection, XSS, NoSQL injection, and other attacks
    """
    
    def __init__(self):
        # Regex patterns for common injection attempts
        self.sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
            r"\b(insert|update|delete|drop|create|alter)\b",
            r"(\bor\b.*=.*\bor\b)|(\band\b.*=.*\band\b)",
            r"['\";].*(--)|(\/\*)",
            r"\b(exec|execute|sp_|xp_)\b",
            r"(\bcast\b|\bconvert\b|\bchar\b|\bnchar\b)",
            r"\b(waitfor|delay)\b",
            r"@@\w+|@@version",
            r"\b(information_schema|sys\.|sysobjects)\b"
        ]
        
        self.nosql_injection_patterns = [
            r"\$where\s*:",
            r"\$regex\s*:",
            r"\$ne\s*:",
            r"\$gt\s*:",
            r"\$lt\s*:",
            r"\$in\s*:",
            r"\$nin\s*:",
            r"javascript:",
            r"eval\s*\(",
            r"function\s*\(",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$(){}]",
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",
            r"\b(rm|mv|cp|chmod|chown|kill|killall)\b",
            r"\b(wget|curl|nc|ncat|telnet|ssh)\b",
            r"\.\.\/",
            r"\/etc\/",
            r"\/proc\/",
        ]
        
        # Allowed HTML tags and attributes for sanitization
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
            'a', 'img', 'table', 'thead', 'tbody', 'tr', 'td', 'th'
        ]
        
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height'],
            'table': ['class'],
            'td': ['colspan', 'rowspan'],
            'th': ['colspan', 'rowspan']
        }
        
        # Financial data validation patterns
        self.financial_patterns = {
            'symbol': r'^[A-Z]{1,5}$',
            'currency': r'^[A-Z]{3}$',
            'amount': r'^\d+(\.\d{1,2})?$',
            'percentage': r'^-?\d+(\.\d{1,4})?$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'datetime': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
        }
    
    def validate_input(
        self,
        data: Any,
        validation_type: str = "general",
        max_length: Optional[int] = None,
        required: bool = True,
        custom_validator: Optional[Callable] = None
    ) -> Any:
        """
        Comprehensive input validation
        
        Args:
            data: Input data to validate
            validation_type: Type of validation (general, sql, nosql, xss, financial)
            max_length: Maximum length for string inputs
            required: Whether the field is required
            custom_validator: Custom validation function
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if required
            if required and (data is None or data == ""):
                raise ValidationError("Required field cannot be empty")
            
            if data is None or data == "":
                return data
            
            # Type-specific validation
            if validation_type == "sql":
                return self._validate_sql_input(data)
            elif validation_type == "nosql":
                return self._validate_nosql_input(data)
            elif validation_type == "xss":
                return self._validate_xss_input(data)
            elif validation_type == "command":
                return self._validate_command_input(data)
            elif validation_type == "financial":
                return self._validate_financial_input(data)
            elif validation_type == "json":
                return self._validate_json_input(data)
            elif validation_type == "url":
                return self._validate_url_input(data)
            elif validation_type == "email":
                return self._validate_email_input(data)
            else:
                return self._validate_general_input(data, max_length)
        
        except ValidationError:
            raise
        except Exception as e:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.ERROR,
                details={
                    "validation_type": validation_type,
                    "error": str(e),
                    "input_type": type(data).__name__
                }
            )
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def _validate_general_input(self, data: Any, max_length: Optional[int] = None) -> Any:
        """General input validation"""
        if isinstance(data, str):
            # Check length
            if max_length and len(data) > max_length:
                raise ValidationError(f"Input exceeds maximum length of {max_length}")
            
            # Check for null bytes
            if '\x00' in data:
                raise ValidationError("Null bytes not allowed")
            
            # Basic sanitization
            data = html.escape(data)
            
        return data
    
    def _validate_sql_input(self, data: Any) -> Any:
        """Validate input against SQL injection attempts"""
        if not isinstance(data, str):
            return data
        
        data_lower = data.lower()
        
        # Check against SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, data_lower, re.IGNORECASE):
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                    level=SecurityEventLevel.WARNING,
                    details={
                        "attack_type": "sql_injection_attempt",
                        "pattern": pattern,
                        "input_length": len(data)
                    }
                )
                raise ValidationError("Potentially malicious SQL input detected")
        
        # Additional SQL parsing validation
        try:
            # Parse to detect SQL structure
            parsed = sqlparse.parse(data)
            if parsed and len(parsed[0].tokens) > 1:
                # Complex SQL structure detected
                for token in parsed[0].tokens:
                    if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML):
                        if str(token).upper() in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']:
                            raise ValidationError("SQL keywords not allowed in input")
        except:
            pass  # Parsing failed, continue with other checks
        
        return html.escape(data)
    
    def _validate_nosql_input(self, data: Any) -> Any:
        """Validate input against NoSQL injection attempts"""
        if isinstance(data, str):
            data_lower = data.lower()
            
            # Check against NoSQL injection patterns
            for pattern in self.nosql_injection_patterns:
                if re.search(pattern, data_lower, re.IGNORECASE):
                    security_logger.log_security_event(
                        event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                        level=SecurityEventLevel.WARNING,
                        details={
                            "attack_type": "nosql_injection_attempt",
                            "pattern": pattern
                        }
                    )
                    raise ValidationError("Potentially malicious NoSQL input detected")
            
            return html.escape(data)
        
        elif isinstance(data, dict):
            # Validate dictionary keys and values
            validated_data = {}
            for key, value in data.items():
                validated_key = self._validate_nosql_input(key)
                validated_value = self._validate_nosql_input(value)
                validated_data[validated_key] = validated_value
            return validated_data
        
        elif isinstance(data, list):
            return [self._validate_nosql_input(item) for item in data]
        
        return data
    
    def _validate_xss_input(self, data: Any) -> Any:
        """Validate and sanitize input against XSS attacks"""
        if not isinstance(data, str):
            return data
        
        # Check against XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, data, re.IGNORECASE | re.DOTALL):
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                    level=SecurityEventLevel.WARNING,
                    details={
                        "attack_type": "xss_attempt",
                        "pattern": pattern
                    }
                )
                raise ValidationError("Potentially malicious XSS input detected")
        
        # Use bleach to sanitize HTML
        sanitized = bleach.clean(
            data,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        return sanitized
    
    def _validate_command_input(self, data: Any) -> Any:
        """Validate input against command injection attempts"""
        if not isinstance(data, str):
            return data
        
        # Check against command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                    level=SecurityEventLevel.WARNING,
                    details={
                        "attack_type": "command_injection_attempt",
                        "pattern": pattern
                    }
                )
                raise ValidationError("Potentially malicious command input detected")
        
        # Remove or escape dangerous characters
        sanitized = re.sub(r'[;&|`$(){}]', '', data)
        return html.escape(sanitized)
    
    def _validate_financial_input(self, data: Any) -> Any:
        """Validate financial data input"""
        if isinstance(data, dict):
            validated_data = {}
            for key, value in data.items():
                if key in self.financial_patterns:
                    if not re.match(self.financial_patterns[key], str(value)):
                        raise ValidationError(f"Invalid format for financial field '{key}'")
                validated_data[key] = self._validate_general_input(value)
            return validated_data
        
        return self._validate_general_input(data)
    
    def _validate_json_input(self, data: Any) -> Any:
        """Validate JSON input"""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                # Recursively validate parsed JSON
                return self._validate_nosql_input(parsed)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format")
        
        return data
    
    def _validate_url_input(self, data: Any) -> str:
        """Validate URL input"""
        if not isinstance(data, str):
            raise ValidationError("URL must be a string")
        
        # Basic URL validation
        if not re.match(r'^https?://', data):
            raise ValidationError("URL must start with http:// or https://")
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'vbscript:', 'data:', 'file:']
        data_lower = data.lower()
        
        for protocol in dangerous_protocols:
            if protocol in data_lower:
                raise ValidationError(f"Dangerous protocol '{protocol}' not allowed")
        
        # URL encode to prevent injection
        return urllib.parse.quote(data, safe=':/?#[]@!$&\'()*+,;=')
    
    def _validate_email_input(self, data: Any) -> str:
        """Validate email input"""
        if not isinstance(data, str):
            raise ValidationError("Email must be a string")
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data):
            raise ValidationError("Invalid email format")
        
        return html.escape(data.lower())
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove path traversal attempts
        sanitized = sanitized.replace('..', '_')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized
    
    def validate_file_upload(
        self,
        filename: str,
        content_type: str,
        file_size: int,
        allowed_extensions: Optional[List[str]] = None,
        max_size_mb: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate file upload"""
        validation_result = {
            "valid": True,
            "errors": [],
            "sanitized_filename": None
        }
        
        try:
            # Sanitize filename
            sanitized_filename = self.sanitize_filename(filename)
            validation_result["sanitized_filename"] = sanitized_filename
            
            # Check file extension
            if allowed_extensions:
                file_ext = os.path.splitext(sanitized_filename)[1].lower()
                if file_ext not in [f".{ext.lower()}" for ext in allowed_extensions]:
                    validation_result["errors"].append("File type not allowed")
                    validation_result["valid"] = False
            
            # Check content type
            if content_type not in security_settings.ALLOWED_CONTENT_TYPES:
                validation_result["errors"].append("Content type not allowed")
                validation_result["valid"] = False
            
            # Check file size
            max_size = (max_size_mb or security_settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
            if file_size > max_size:
                validation_result["errors"].append(f"File size exceeds {max_size_mb}MB limit")
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def validate_financial_symbol(self, symbol: str) -> str:
        """Validate financial symbol format"""
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")
        
        symbol = symbol.upper().strip()
        
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            raise ValidationError("Invalid symbol format (1-5 uppercase letters)")
        
        return symbol
    
    def validate_decimal_amount(self, amount: Union[str, int, float, Decimal], max_value: Optional[Decimal] = None) -> Decimal:
        """Validate decimal amount for financial calculations"""
        try:
            decimal_amount = Decimal(str(amount))
            
            if decimal_amount < 0:
                raise ValidationError("Amount cannot be negative")
            
            if max_value and decimal_amount > max_value:
                raise ValidationError(f"Amount exceeds maximum value of {max_value}")
            
            # Check precision (max 2 decimal places for currency)
            if decimal_amount.as_tuple().exponent < -2:
                raise ValidationError("Amount cannot have more than 2 decimal places")
            
            return decimal_amount
        
        except (InvalidOperation, ValueError):
            raise ValidationError("Invalid amount format")
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple:
        """Validate date range"""
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if start_dt >= end_dt:
                raise ValidationError("Start date must be before end date")
            
            # Check if date range is reasonable (not too far in the future)
            if end_dt > datetime.now().replace(year=datetime.now().year + 1):
                raise ValidationError("End date cannot be more than 1 year in the future")
            
            return (start_dt, end_dt)
        
        except ValueError:
            raise ValidationError("Invalid date format (use ISO format: YYYY-MM-DD)")


class RequestValidator:
    """Validate incoming HTTP requests"""
    
    def __init__(self):
        self.input_validator = InputValidator()
    
    def validate_request_size(self, content_length: int):
        """Validate request size"""
        max_size = security_settings.MAX_REQUEST_SIZE_MB * 1024 * 1024
        
        if content_length > max_size:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                details={
                    "attack_type": "oversized_request",
                    "content_length": content_length,
                    "max_allowed": max_size
                }
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request size exceeds limit"
            )
    
    def validate_content_type(self, content_type: str):
        """Validate request content type"""
        if content_type not in security_settings.ALLOWED_CONTENT_TYPES:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                details={
                    "attack_type": "invalid_content_type",
                    "content_type": content_type
                }
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported content type"
            )
    
    def validate_user_agent(self, user_agent: str):
        """Validate user agent string"""
        if not user_agent or len(user_agent) > 500:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
                level=SecurityEventLevel.INFO,
                details={"invalid_user_agent": user_agent[:100] if user_agent else "empty"}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user agent"
            )


# Global validators
input_validator = InputValidator()
request_validator = RequestValidator()


# =====================================
# PYDANTIC VALIDATORS
# =====================================

def validate_sql_safe(value: str) -> str:
    """Pydantic validator for SQL-safe input"""
    return input_validator.validate_input(value, "sql")

def validate_xss_safe(value: str) -> str:
    """Pydantic validator for XSS-safe input"""
    return input_validator.validate_input(value, "xss")

def validate_financial_symbol(value: str) -> str:
    """Pydantic validator for financial symbol"""
    return input_validator.validate_financial_symbol(value)

def validate_decimal_amount(value: Union[str, int, float]) -> Decimal:
    """Pydantic validator for decimal amount"""
    return input_validator.validate_decimal_amount(value)


# =====================================
# FASTAPI DEPENDENCIES
# =====================================

def validate_request_security(request):
    """FastAPI dependency for request security validation"""
    # Validate content length
    content_length = int(request.headers.get("content-length", 0))
    if content_length > 0:
        request_validator.validate_request_size(content_length)
    
    # Validate content type
    content_type = request.headers.get("content-type", "")
    if content_type:
        request_validator.validate_content_type(content_type.split(";")[0])
    
    # Validate user agent
    user_agent = request.headers.get("user-agent", "")
    request_validator.validate_user_agent(user_agent)
    
    return True