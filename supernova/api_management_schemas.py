"""
SuperNova AI API Management Schemas
Pydantic models for API management requests and responses
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from .api_management_agent import APIKeyTier, QuotaType, UsageMetricType


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key"""
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    tier: APIKeyTier = Field(default=APIKeyTier.FREE, description="API key tier")
    scopes: Optional[List[str]] = Field(default=[], description="API key scopes/permissions")
    expires_at: Optional[datetime] = Field(None, description="Expiration date (optional)")
    allowed_ips: Optional[List[str]] = Field(default=[], description="Allowed IP addresses")
    allowed_origins: Optional[List[str]] = Field(default=[], description="Allowed origins for CORS")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for events")
    
    @validator('expires_at')
    def validate_expiration(cls, v):
        if v and v <= datetime.utcnow():
            raise ValueError('Expiration date must be in the future')
        return v
    
    @validator('allowed_ips')
    def validate_ips(cls, v):
        if v:
            import ipaddress
            for ip in v:
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    raise ValueError(f'Invalid IP address: {ip}')
        return v


class APIKeyResponse(BaseModel):
    """Response when creating or retrieving an API key"""
    key_id: str
    name: str
    tier: APIKeyTier
    scopes: List[str]
    prefix: str  # First few characters for display
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool
    rate_limits: Dict[str, int]
    quota_limits: Dict[str, float]
    usage_stats: Optional[Dict[str, Any]] = None


class APIKeyCreateResponse(APIKeyResponse):
    """Response when creating a new API key (includes actual key)"""
    api_key: str  # Only returned on creation


class APIKeyListResponse(BaseModel):
    """Response for listing API keys"""
    keys: List[APIKeyResponse]
    total: int
    page: int
    per_page: int


class APIKeyUpdateRequest(BaseModel):
    """Request to update an API key"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    scopes: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None
    allowed_ips: Optional[List[str]] = None
    allowed_origins: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    
    @validator('expires_at')
    def validate_expiration(cls, v):
        if v and v <= datetime.utcnow():
            raise ValueError('Expiration date must be in the future')
        return v


class APIKeyRotateResponse(BaseModel):
    """Response when rotating an API key"""
    key_id: str
    new_api_key: str
    rotated_at: datetime
    message: str = "API key rotated successfully. Update your applications immediately."


class QuotaConfigRequest(BaseModel):
    """Request to configure quotas"""
    quota_type: QuotaType
    limit: float = Field(..., ge=0, description="Quota limit")
    reset_period_days: int = Field(30, ge=1, le=365, description="Reset period in days")


class QuotaConfigResponse(BaseModel):
    """Response for quota configuration"""
    quota_type: QuotaType
    limit: float
    used: float
    remaining: float
    reset_time: datetime
    reset_period_days: int
    overage_count: int


class QuotaListResponse(BaseModel):
    """Response for listing all quotas"""
    quotas: List[QuotaConfigResponse]
    key_id: str
    tier: APIKeyTier


class UsageAnalyticsRequest(BaseModel):
    """Request for usage analytics"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    key_id: Optional[str] = None
    group_by: str = Field("hour", regex="^(hour|day|week|month)$")
    metrics: Optional[List[UsageMetricType]] = None
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        if v and v > datetime.utcnow():
            raise ValueError('Date cannot be in the future')
        return v
    
    @validator('end_date')
    def validate_end_after_start(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v


class UsageAnalyticsResponse(BaseModel):
    """Response for usage analytics"""
    period: Dict[str, str]  # start and end dates
    summary: Dict[str, Union[int, float]]
    endpoints: Dict[str, Dict[str, Union[int, float]]]
    time_series: Dict[str, Dict[str, Union[int, float]]]
    errors: Dict[str, int]
    geographic: Dict[str, int]
    key_performance: Optional[Dict[str, Any]] = None


class RateLimitConfigRequest(BaseModel):
    """Request to configure rate limits"""
    context: str = Field("default", description="Rate limit context (api, auth, upload, default)")
    requests_per_minute: int = Field(..., ge=1, le=100000)
    requests_per_hour: int = Field(..., ge=1, le=10000000)
    requests_per_day: int = Field(..., ge=1, le=100000000)
    burst_capacity: int = Field(..., ge=1, le=10000)
    algorithm: str = Field("adaptive", regex="^(token_bucket|sliding_window|fixed_window|adaptive)$")
    
    @validator('requests_per_hour')
    def validate_hour_limit(cls, v, values):
        rpm = values.get('requests_per_minute', 0)
        if v < rpm:
            raise ValueError('Hourly limit must be at least equal to per-minute limit')
        return v
    
    @validator('requests_per_day')
    def validate_day_limit(cls, v, values):
        rph = values.get('requests_per_hour', 0)
        if v < rph:
            raise ValueError('Daily limit must be at least equal to hourly limit')
        return v


class RateLimitConfigResponse(BaseModel):
    """Response for rate limit configuration"""
    context: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_capacity: int
    algorithm: str
    current_usage: Dict[str, int]
    next_reset: datetime


class SecurityEventRequest(BaseModel):
    """Request to report a security event"""
    event_type: str
    severity: str = Field("low", regex="^(low|medium|high|critical)$")
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class SecurityEventResponse(BaseModel):
    """Response for security events"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: str
    description: str
    status: str
    actions_taken: List[str]


class IPBlockRequest(BaseModel):
    """Request to block an IP address"""
    ip_address: str = Field(..., description="IP address to block")
    reason: str = Field(..., min_length=10, max_length=500)
    duration_minutes: int = Field(60, ge=1, le=43200, description="Block duration in minutes")
    notify_webhooks: bool = Field(True, description="Send webhook notifications")
    
    @validator('ip_address')
    def validate_ip(cls, v):
        import ipaddress
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError('Invalid IP address format')
        return v


class IPBlockResponse(BaseModel):
    """Response for IP blocking"""
    ip_address: str
    blocked_at: datetime
    blocked_until: datetime
    reason: str
    block_count: int


class SystemHealthResponse(BaseModel):
    """Response for system health check"""
    timestamp: datetime
    status: str  # healthy, degraded, critical
    health_score: float  # 0-100
    metrics: Dict[str, Union[int, float, str]]
    components: Dict[str, str]  # component: status
    alerts: List[Dict[str, Any]]
    uptime_seconds: int


class RealTimeMetricsResponse(BaseModel):
    """Response for real-time metrics"""
    timestamp: datetime
    current_rps: float
    peak_rps: float
    total_requests: int
    error_count: int
    avg_response_time: float
    active_api_keys: int
    unique_users_today: int
    unique_ips_today: int
    cache_hit_rate: float
    cache_size: int
    system_health_score: float
    threat_level: bool


class DashboardMetricsResponse(BaseModel):
    """Response for dashboard metrics"""
    overview: Dict[str, Union[int, float]]
    traffic_chart: List[Dict[str, Any]]
    top_endpoints: List[Dict[str, Any]]
    recent_errors: List[Dict[str, Any]]
    geographical_data: Dict[str, int]
    performance_trends: List[Dict[str, Any]]
    security_alerts: List[Dict[str, Any]]


class WebhookConfigRequest(BaseModel):
    """Request to configure webhooks"""
    url: str = Field(..., regex=r'^https?://.+')
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = Field(None, min_length=10)
    is_active: bool = Field(True)
    retry_config: Optional[Dict[str, int]] = None
    
    @validator('events')
    def validate_events(cls, v):
        valid_events = [
            'key_created', 'key_revoked', 'key_rotated', 'quota_exceeded',
            'rate_limit_exceeded', 'security_incident', 'system_alert'
        ]
        for event in v:
            if event not in valid_events:
                raise ValueError(f'Invalid event type: {event}')
        return v


class WebhookConfigResponse(BaseModel):
    """Response for webhook configuration"""
    webhook_id: str
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime]
    success_count: int
    failure_count: int


class BulkOperationRequest(BaseModel):
    """Request for bulk operations"""
    operation: str = Field(..., regex="^(revoke|activate|deactivate|update_tier)$")
    key_ids: List[str] = Field(..., min_items=1, max_items=100)
    parameters: Optional[Dict[str, Any]] = None


class BulkOperationResponse(BaseModel):
    """Response for bulk operations"""
    operation: str
    requested_count: int
    successful_count: int
    failed_count: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]]


class APIManagementConfigRequest(BaseModel):
    """Request to update global API management configuration"""
    global_rate_limits: Optional[Dict[str, int]] = None
    security_settings: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, Any]] = None
    cache_settings: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None


class APIManagementConfigResponse(BaseModel):
    """Response for API management configuration"""
    global_rate_limits: Dict[str, int]
    security_settings: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    cache_settings: Dict[str, Any]
    notification_settings: Dict[str, Any]
    last_updated: datetime
    updated_by: str


class ExportRequest(BaseModel):
    """Request to export data"""
    export_type: str = Field(..., regex="^(usage|keys|security|analytics)$")
    format: str = Field("json", regex="^(json|csv|xlsx)$")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filters: Optional[Dict[str, Any]] = None


class ExportResponse(BaseModel):
    """Response for data export"""
    export_id: str
    export_type: str
    format: str
    status: str  # queued, processing, completed, failed
    created_at: datetime
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    file_size: Optional[int] = None