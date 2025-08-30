"""
SuperNova AI Security Validation Schemas
Comprehensive security-focused Pydantic models for input validation and compliance
"""

from pydantic import BaseModel, Field, validator, root_validator, EmailStr
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import re
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, AddressValueError

# Import security validators
from .input_validation import (
    validate_sql_safe, validate_xss_safe, validate_financial_symbol,
    validate_decimal_amount, input_validator
)


# ================================
# AUTHENTICATION & AUTHORIZATION
# ================================

class LoginRequest(BaseModel):
    """Secure login request with comprehensive validation"""
    username: str = Field(..., min_length=3, max_length=50, description="Username or email")
    password: str = Field(..., min_length=1, max_length=128, description="User password")
    remember_me: bool = Field(False, description="Remember login session")
    device_fingerprint: Optional[str] = Field(None, max_length=256, description="Device fingerprint")
    captcha_response: Optional[str] = Field(None, max_length=1000, description="CAPTCHA response")
    
    @validator('username')
    def validate_username(cls, v):
        # Check if it's an email or username
        if '@' in v:
            # Validate as email
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
                raise ValueError('Invalid email format')
        else:
            # Validate as username
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        
        return validate_xss_safe(v.lower())
    
    @validator('device_fingerprint')
    def validate_device_fingerprint(cls, v):
        if v is not None:
            # Ensure device fingerprint doesn't contain malicious content
            return validate_xss_safe(v)
        return v

class MFAVerificationRequest(BaseModel):
    """Multi-factor authentication verification"""
    user_id: str = Field(..., description="User identifier")
    mfa_code: str = Field(..., min_length=4, max_length=10, description="MFA verification code")
    method: Literal["totp", "sms", "email", "backup_code"] = Field(..., description="MFA method used")
    device_trust: bool = Field(False, description="Trust this device")
    
    @validator('mfa_code')
    def validate_mfa_code(cls, v):
        # Ensure MFA code is numeric or alphanumeric
        if not re.match(r'^[a-zA-Z0-9]+$', v):
            raise ValueError('Invalid MFA code format')
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        return validate_xss_safe(v)

class PasswordResetRequest(BaseModel):
    """Password reset request with validation"""
    email: EmailStr = Field(..., description="User email address")
    captcha_response: Optional[str] = Field(None, max_length=1000, description="CAPTCHA response")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    
    @validator('client_ip')
    def validate_ip_address(cls, v):
        if v is not None:
            try:
                IPv4Address(v)
            except AddressValueError:
                try:
                    IPv6Address(v)
                except AddressValueError:
                    raise ValueError('Invalid IP address format')
        return v

class TokenRefreshRequest(BaseModel):
    """JWT token refresh request"""
    refresh_token: str = Field(..., min_length=10, max_length=500, description="Refresh token")
    device_id: Optional[str] = Field(None, max_length=100, description="Device identifier")
    
    @validator('refresh_token')
    def validate_refresh_token(cls, v):
        # Basic token format validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Invalid token format')
        return v
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v


# ================================
# API SECURITY SCHEMAS
# ================================

class APIKeyCreationRequest(BaseModel):
    """Secure API key creation request"""
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(..., min_items=1, max_items=20, description="API key permissions")
    expiry_days: Optional[int] = Field(365, ge=1, le=3650, description="Days until expiry")
    ip_whitelist: Optional[List[str]] = Field(None, max_items=50, description="Allowed IP addresses")
    rate_limit_override: Optional[int] = Field(None, ge=1, le=10000, description="Custom rate limit")
    environment: Literal["development", "staging", "production"] = Field("production", description="Target environment")
    
    @validator('name')
    def validate_name(cls, v):
        return validate_xss_safe(v)
    
    @validator('permissions')
    def validate_permissions(cls, v):
        allowed_permissions = {
            'read:portfolio', 'write:portfolio', 'read:market_data', 'write:market_data',
            'read:analytics', 'write:analytics', 'read:user_data', 'write:user_data',
            'read:admin', 'write:admin', 'read:compliance', 'write:compliance'
        }
        
        for permission in v:
            if permission not in allowed_permissions:
                raise ValueError(f'Invalid permission: {permission}')
            
        # Check for conflicting permissions
        if 'write:admin' in v and len(v) > 1:
            raise ValueError('Admin write permission cannot be combined with others')
        
        return v
    
    @validator('ip_whitelist')
    def validate_ip_whitelist(cls, v):
        if v is not None:
            for ip in v:
                try:
                    IPv4Address(ip)
                except AddressValueError:
                    try:
                        IPv6Address(ip)
                    except AddressValueError:
                        raise ValueError(f'Invalid IP address: {ip}')
        return v

class RateLimitConfiguration(BaseModel):
    """Rate limiting configuration"""
    endpoint_pattern: str = Field(..., max_length=200, description="Endpoint pattern")
    requests_per_minute: int = Field(..., ge=1, le=10000, description="Requests per minute")
    requests_per_hour: int = Field(..., ge=1, le=100000, description="Requests per hour")
    requests_per_day: int = Field(..., ge=1, le=1000000, description="Requests per day")
    burst_limit: int = Field(..., ge=1, le=1000, description="Burst request limit")
    whitelist_ips: Optional[List[str]] = Field(None, description="IP addresses exempt from limits")
    
    @validator('endpoint_pattern')
    def validate_endpoint_pattern(cls, v):
        # Validate regex pattern for endpoints
        try:
            re.compile(v)
        except re.error:
            raise ValueError('Invalid regex pattern for endpoint')
        return validate_xss_safe(v)
    
    @root_validator
    def validate_rate_limits(cls, values):
        per_minute = values.get('requests_per_minute', 0)
        per_hour = values.get('requests_per_hour', 0)
        per_day = values.get('requests_per_day', 0)
        
        # Ensure logical rate limit progression
        if per_hour < per_minute * 60:
            raise ValueError('Hourly limit must be at least 60x the per-minute limit')
        
        if per_day < per_hour * 24:
            raise ValueError('Daily limit must be at least 24x the hourly limit')
        
        return values


# ================================
# INPUT VALIDATION SCHEMAS
# ================================

class FileUploadRequest(BaseModel):
    """Secure file upload request"""
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., gt=0, le=100*1024*1024, description="File size in bytes")  # 100MB limit
    checksum: str = Field(..., min_length=32, max_length=128, description="File checksum")
    upload_purpose: Literal["avatar", "document", "data_import", "report"] = Field(..., description="Upload purpose")
    
    @validator('filename')
    def validate_filename(cls, v):
        # Sanitize filename
        sanitized = input_validator.sanitize_filename(v)
        
        # Check for dangerous file extensions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js', '.jar', '.app', '.com']
        file_extension = sanitized.lower().split('.')[-1] if '.' in sanitized else ''
        
        if f'.{file_extension}' in dangerous_extensions:
            raise ValueError(f'File type .{file_extension} not allowed')
        
        return sanitized
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'application/pdf', 'text/csv', 'application/json',
            'text/plain', 'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        if v not in allowed_types:
            raise ValueError(f'Content type {v} not allowed')
        
        return v
    
    @validator('checksum')
    def validate_checksum(cls, v):
        # Validate checksum format (SHA-256 or MD5)
        if not re.match(r'^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{64}$', v):
            raise ValueError('Invalid checksum format')
        return v.lower()

class DatabaseQueryRequest(BaseModel):
    """Secure database query request"""
    query_type: Literal["select", "insert", "update", "delete"] = Field(..., description="Type of database operation")
    table_name: str = Field(..., min_length=1, max_length=100, description="Target table name")
    filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
    limit: Optional[int] = Field(100, ge=1, le=10000, description="Result limit")
    offset: Optional[int] = Field(0, ge=0, description="Result offset")
    order_by: Optional[str] = Field(None, max_length=100, description="Order by field")
    
    @validator('table_name')
    def validate_table_name(cls, v):
        # Validate table name format
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError('Invalid table name format')
        
        # Check against allowed tables
        allowed_tables = {
            'users', 'profiles', 'portfolios', 'transactions', 'market_data',
            'sentiment_data', 'strategies', 'backtests', 'alerts', 'audit_logs'
        }
        
        if v not in allowed_tables:
            raise ValueError(f'Access to table {v} not allowed')
        
        return v
    
    @validator('filters')
    def validate_filters(cls, v):
        if v is not None:
            # Validate filter values for SQL injection
            for key, value in v.items():
                if isinstance(value, str):
                    try:
                        input_validator.validate_input(value, "sql")
                    except Exception:
                        raise ValueError(f'Invalid filter value for {key}')
        return v
    
    @validator('order_by')
    def validate_order_by(cls, v):
        if v is not None:
            # Validate order by field
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*(\s+(ASC|DESC))?$', v.upper()):
                raise ValueError('Invalid order by format')
        return v


# ================================
# FINANCIAL DATA VALIDATION
# ================================

class PortfolioUpdateRequest(BaseModel):
    """Secure portfolio update with financial validation"""
    user_id: int = Field(..., gt=0, description="User identifier")
    positions: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000, description="Portfolio positions")
    total_value: Decimal = Field(..., gt=0, description="Total portfolio value")
    currency: str = Field("USD", pattern=r'^[A-Z]{3}$', description="Portfolio currency")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @validator('positions')
    def validate_positions(cls, v):
        validated_positions = []
        total_weight = Decimal('0')
        
        for position in v:
            # Validate required fields
            required_fields = ['symbol', 'quantity', 'price', 'weight']
            for field in required_fields:
                if field not in position:
                    raise ValueError(f'Missing required field: {field}')
            
            # Validate symbol
            symbol = validate_financial_symbol(position['symbol'])
            
            # Validate numeric fields
            quantity = validate_decimal_amount(position['quantity'])
            price = validate_decimal_amount(position['price'])
            weight = validate_decimal_amount(position['weight'])
            
            # Check weight is reasonable
            if weight > Decimal('0.5'):  # No single position > 50%
                raise ValueError(f'Position weight too high for {symbol}')
            
            total_weight += weight
            
            validated_positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'weight': weight,
                'value': quantity * price
            })
        
        # Check total weight doesn't exceed 100%
        if total_weight > Decimal('1.01'):  # Allow for small rounding differences
            raise ValueError('Total position weights exceed 100%')
        
        return validated_positions
    
    @validator('total_value')
    def validate_total_value(cls, v):
        return validate_decimal_amount(v, max_value=Decimal('999999999.99'))

class TransactionRequest(BaseModel):
    """Secure financial transaction request"""
    user_id: int = Field(..., gt=0, description="User identifier")
    transaction_type: Literal["buy", "sell", "deposit", "withdrawal", "dividend", "fee"] = Field(..., description="Transaction type")
    symbol: Optional[str] = Field(None, description="Financial symbol (for trades)")
    quantity: Optional[Decimal] = Field(None, gt=0, description="Transaction quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Transaction price")
    amount: Decimal = Field(..., description="Transaction amount")
    currency: str = Field("USD", pattern=r'^[A-Z]{3}$', description="Transaction currency")
    fees: Optional[Decimal] = Field(None, ge=0, description="Transaction fees")
    notes: Optional[str] = Field(None, max_length=500, description="Transaction notes")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if v is not None:
            return validate_financial_symbol(v)
        return v
    
    @validator('quantity', 'price', 'amount', 'fees')
    def validate_financial_amounts(cls, v):
        if v is not None:
            return validate_decimal_amount(v, max_value=Decimal('999999999.99'))
        return v
    
    @validator('notes')
    def validate_notes(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v
    
    @root_validator
    def validate_transaction_consistency(cls, values):
        transaction_type = values.get('transaction_type')
        symbol = values.get('symbol')
        quantity = values.get('quantity')
        price = values.get('price')
        
        # Validate required fields based on transaction type
        if transaction_type in ['buy', 'sell']:
            if not symbol:
                raise ValueError('Symbol required for buy/sell transactions')
            if not quantity:
                raise ValueError('Quantity required for buy/sell transactions')
            if not price:
                raise ValueError('Price required for buy/sell transactions')
        
        return values


# ================================
# COMPLIANCE & AUDIT SCHEMAS
# ================================

class AuditLogEntry(BaseModel):
    """Comprehensive audit log entry"""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: Literal[
        "authentication", "authorization", "data_access", "data_modification",
        "system_configuration", "security_event", "compliance_action",
        "financial_transaction", "user_action", "admin_action"
    ] = Field(..., description="Type of audit event")
    user_id: Optional[str] = Field(None, description="User who triggered the event")
    session_id: Optional[str] = Field(None, description="Session identifier")
    resource_type: Optional[str] = Field(None, description="Type of resource involved")
    resource_id: Optional[str] = Field(None, description="Identifier of resource")
    action: str = Field(..., max_length=200, description="Action performed")
    outcome: Literal["success", "failure", "partial", "pending"] = Field(..., description="Action outcome")
    risk_level: Literal["low", "medium", "high", "critical"] = Field("medium", description="Risk assessment")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, max_length=500, description="User agent string")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    retention_until: datetime = Field(..., description="Data retention expiry")
    
    @validator('event_id')
    def validate_event_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid event ID format')
        return v
    
    @validator('action')
    def validate_action(cls, v):
        return validate_xss_safe(v)
    
    @validator('client_ip')
    def validate_client_ip(cls, v):
        if v is not None:
            try:
                IPv4Address(v)
            except AddressValueError:
                try:
                    IPv6Address(v)
                except AddressValueError:
                    raise ValueError('Invalid IP address format')
        return v
    
    @validator('user_agent')
    def validate_user_agent(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v

class ComplianceExportRequest(BaseModel):
    """Request for compliance data export"""
    export_type: Literal[
        "user_data", "transaction_history", "audit_logs", "communication_logs",
        "system_logs", "security_events", "compliance_reports"
    ] = Field(..., description="Type of data to export")
    user_id: Optional[str] = Field(None, description="Specific user data to export")
    date_range: Dict[str, datetime] = Field(..., description="Export date range")
    format: Literal["json", "csv", "xml", "encrypted_zip"] = Field("json", description="Export format")
    include_pii: bool = Field(False, description="Include personally identifiable information")
    legal_basis: str = Field(..., min_length=1, max_length=500, description="Legal basis for export")
    requesting_authority: Optional[str] = Field(None, max_length=200, description="Authority requesting data")
    encryption_required: bool = Field(True, description="Require encryption for sensitive data")
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if 'start' not in v or 'end' not in v:
            raise ValueError('Date range must include start and end dates')
        
        start_date = v['start']
        end_date = v['end']
        
        if start_date >= end_date:
            raise ValueError('Start date must be before end date')
        
        # Limit export range to prevent system overload
        max_range = timedelta(days=2555)  # 7 years
        if (end_date - start_date) > max_range:
            raise ValueError('Export range cannot exceed 7 years')
        
        return v
    
    @validator('legal_basis', 'requesting_authority')
    def validate_text_fields(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v

class GDPRDataRequest(BaseModel):
    """GDPR data subject rights request"""
    request_type: Literal[
        "access", "rectification", "erasure", "portability", 
        "restrict_processing", "object_processing", "automated_decision_info"
    ] = Field(..., description="Type of GDPR request")
    user_id: str = Field(..., description="Data subject identifier")
    verification_method: Literal["email", "phone", "document", "in_person"] = Field(..., description="Identity verification method")
    verification_token: str = Field(..., min_length=16, max_length=256, description="Verification token")
    contact_email: EmailStr = Field(..., description="Contact email for response")
    specific_data_categories: Optional[List[str]] = Field(None, description="Specific data categories requested")
    reason_for_request: Optional[str] = Field(None, max_length=1000, description="Reason for the request")
    urgency_level: Literal["normal", "urgent", "legal_deadline"] = Field("normal", description="Request urgency")
    preferred_response_format: Literal["email", "encrypted_file", "postal_mail", "secure_portal"] = Field("email", description="Preferred response method")
    
    @validator('verification_token')
    def validate_verification_token(cls, v):
        # Ensure token format is secure
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid verification token format')
        return v
    
    @validator('reason_for_request')
    def validate_reason(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v

class SecurityIncidentReport(BaseModel):
    """Security incident reporting"""
    incident_id: str = Field(..., description="Unique incident identifier")
    incident_type: Literal[
        "data_breach", "unauthorized_access", "malware", "phishing",
        "ddos_attack", "insider_threat", "system_compromise", "privacy_violation"
    ] = Field(..., description="Type of security incident")
    severity_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Incident severity")
    discovered_at: datetime = Field(..., description="When incident was discovered")
    occurred_at: Optional[datetime] = Field(None, description="When incident actually occurred")
    affected_systems: List[str] = Field(..., min_items=1, description="Systems affected by incident")
    affected_users: Optional[List[str]] = Field(None, description="Users potentially affected")
    data_types_involved: Optional[List[str]] = Field(None, description="Types of data involved")
    immediate_actions_taken: List[str] = Field(default_factory=list, description="Immediate response actions")
    containment_status: Literal["not_contained", "partially_contained", "fully_contained"] = Field("not_contained", description="Containment status")
    root_cause_analysis: Optional[str] = Field(None, max_length=2000, description="Root cause analysis")
    lessons_learned: Optional[str] = Field(None, max_length=2000, description="Lessons learned")
    reporting_required: bool = Field(False, description="Whether regulatory reporting is required")
    regulatory_deadline: Optional[datetime] = Field(None, description="Deadline for regulatory reporting")
    
    @validator('incident_id')
    def validate_incident_id(cls, v):
        if not re.match(r'^INC-\d{4}-\d{6}$', v):
            raise ValueError('Incident ID must follow format INC-YYYY-NNNNNN')
        return v
    
    @validator('root_cause_analysis', 'lessons_learned')
    def validate_text_fields(cls, v):
        if v is not None:
            return validate_xss_safe(v)
        return v
    
    @root_validator
    def validate_incident_timeline(cls, values):
        discovered_at = values.get('discovered_at')
        occurred_at = values.get('occurred_at')
        
        if discovered_at and occurred_at:
            if occurred_at > discovered_at:
                raise ValueError('Incident occurrence date cannot be after discovery date')
        
        return values


# ================================
# WEBSOCKET SECURITY SCHEMAS
# ================================

class WebSocketAuthRequest(BaseModel):
    """WebSocket authentication request"""
    token: str = Field(..., min_length=10, description="Authentication token")
    protocol_version: str = Field("1.0", pattern=r'^\d+\.\d+$', description="Protocol version")
    client_info: Optional[Dict[str, str]] = Field(None, description="Client information")
    
    @validator('token')
    def validate_token(cls, v):
        # Basic token format validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Invalid token format')
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        if v is not None:
            for key, value in v.items():
                if len(key) > 50 or len(value) > 200:
                    raise ValueError('Client info values too long')
                v[key] = validate_xss_safe(value)
        return v

class WebSocketMessageValidation(BaseModel):
    """WebSocket message validation"""
    message_type: Literal[
        "chat", "typing", "presence", "heartbeat", "system",
        "market_data", "notification", "command"
    ] = Field(..., description="Message type")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    message_id: Optional[str] = Field(None, description="Message identifier")
    
    @validator('payload')
    def validate_payload(cls, v):
        # Validate payload size
        payload_str = str(v)
        if len(payload_str) > 64 * 1024:  # 64KB limit
            raise ValueError('Message payload too large')
        
        # Recursively validate string values in payload
        def validate_dict_values(d):
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = validate_xss_safe(value)
                elif isinstance(value, dict):
                    validate_dict_values(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            value[i] = validate_xss_safe(item)
                        elif isinstance(item, dict):
                            validate_dict_values(item)
        
        validate_dict_values(v)
        return v
    
    @validator('message_id')
    def validate_message_id(cls, v):
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('Invalid message ID format')
        return v


# ================================
# SYSTEM CONFIGURATION SCHEMAS
# ================================

class SecurityPolicyUpdate(BaseModel):
    """Security policy configuration update"""
    policy_name: str = Field(..., min_length=1, max_length=100, description="Policy name")
    policy_version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Policy version")
    effective_date: datetime = Field(..., description="When policy becomes effective")
    configuration: Dict[str, Any] = Field(..., description="Policy configuration")
    approval_required: bool = Field(True, description="Whether approval is required")
    impact_assessment: str = Field(..., max_length=1000, description="Impact assessment")
    rollback_plan: str = Field(..., max_length=1000, description="Rollback procedure")
    
    @validator('policy_name')
    def validate_policy_name(cls, v):
        return validate_xss_safe(v)
    
    @validator('impact_assessment', 'rollback_plan')
    def validate_text_fields(cls, v):
        return validate_xss_safe(v)
    
    @validator('configuration')
    def validate_configuration(cls, v):
        # Recursively validate configuration values
        def validate_config_values(d):
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = validate_xss_safe(value)
                elif isinstance(value, dict):
                    validate_config_values(value)
        
        validate_config_values(v)
        return v