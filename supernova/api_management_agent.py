"""
SuperNova AI API Management Agent
Comprehensive API traffic control, monitoring, and analytics system
"""

import time
import json
import hashlib
import secrets
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import statistics

import redis
import os
from fastapi import Request, HTTPException, status, Depends
from sqlalchemy import select, text, func, and_, or_, desc
from sqlalchemy.orm import Session

from .db import SessionLocal, User, APIKey, get_timescale_session
from .auth import get_current_user, require_permission
from .security_logger import security_logger, SecurityEventLevel
from .config import settings
from .advanced_rate_limiting import AdvancedRateLimiter, RateLimitConfig, ThreatLevel

logger = logging.getLogger(__name__)


class APIKeyTier(str, Enum):
    """API key tiers with different capabilities"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class QuotaType(str, Enum):
    """Types of quotas that can be managed"""
    REQUESTS = "requests"
    COMPUTE = "compute"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"


class UsageMetricType(str, Enum):
    """Types of usage metrics to track"""
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_COUNT = "error_count"
    BANDWIDTH_IN = "bandwidth_in"
    BANDWIDTH_OUT = "bandwidth_out"
    COMPUTE_TIME = "compute_time"


@dataclass
class APIKeyConfig:
    """Configuration for API keys"""
    tier: APIKeyTier = APIKeyTier.FREE
    scopes: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    quota_limits: Dict[QuotaType, int] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    webhook_url: Optional[str] = None
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class UsageRecord:
    """Record of API usage"""
    timestamp: datetime
    api_key_id: Optional[str]
    user_id: Optional[int]
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size: int
    response_size: int
    ip_address: str
    user_agent: Optional[str]
    error_message: Optional[str] = None


@dataclass
class QuotaUsage:
    """Quota usage tracking"""
    quota_type: QuotaType
    used: float
    limit: float
    reset_time: datetime
    overage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class APIManagementAgent:
    """
    Comprehensive API Management Agent with advanced features
    """
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        self.rate_limiter = AdvancedRateLimiter()
        
        # API key management
        self.api_keys: Dict[str, APIKeyConfig] = {}
        self.api_key_usage: Dict[str, List[UsageRecord]] = defaultdict(list)
        
        # Quota management
        self.quota_usage: Dict[str, Dict[QuotaType, QuotaUsage]] = defaultdict(dict)
        
        # Monitoring and analytics
        self.usage_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "unique_users": set(),
            "unique_ips": set(),
            "error_count": 0,
            "avg_response_time": 0.0,
            "peak_rps": 0.0,
            "current_rps": 0.0
        }
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.error_tracking: Dict[str, List[Dict]] = defaultdict(list)
        
        # Security monitoring
        self.suspicious_activities: deque = deque(maxlen=10000)
        self.blocked_ips: Dict[str, Dict] = {}
        
        # Caching system
        self.response_cache: Dict[str, Dict] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "size": 0}
        
        # Tier configurations
        self.tier_configs = {
            APIKeyTier.FREE: {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 5000,
                "max_request_size": 1024 * 1024,  # 1MB
                "max_response_size": 5 * 1024 * 1024,  # 5MB
                "allowed_endpoints": ["GET /api/market-data", "GET /api/basic-analysis"],
                "features": ["basic_analytics", "standard_support"]
            },
            APIKeyTier.PRO: {
                "requests_per_minute": 1000,
                "requests_per_hour": 20000,
                "requests_per_day": 100000,
                "max_request_size": 10 * 1024 * 1024,  # 10MB
                "max_response_size": 50 * 1024 * 1024,  # 50MB
                "allowed_endpoints": ["*"],
                "features": ["advanced_analytics", "priority_support", "webhooks"]
            },
            APIKeyTier.ENTERPRISE: {
                "requests_per_minute": 5000,
                "requests_per_hour": 100000,
                "requests_per_day": 1000000,
                "max_request_size": 100 * 1024 * 1024,  # 100MB
                "max_response_size": 500 * 1024 * 1024,  # 500MB
                "allowed_endpoints": ["*"],
                "features": ["all", "dedicated_support", "custom_integrations"]
            }
        }
        
        # Background tasks
        self._start_background_tasks()
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis for caching and distributed state"""
        try:
            import redis
            return redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_DB_API_MANAGEMENT', 3),
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    def _start_background_tasks(self):
        """Start background maintenance and monitoring tasks"""
        asyncio.create_task(self._update_performance_metrics())
        asyncio.create_task(self._cleanup_expired_data())
        asyncio.create_task(self._detect_anomalies())
        asyncio.create_task(self._update_quota_usage())
    
    # ========================================
    # API KEY MANAGEMENT
    # ========================================
    
    async def create_api_key(
        self,
        user_id: int,
        name: str,
        tier: APIKeyTier = APIKeyTier.FREE,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        db: Session = None
    ) -> Tuple[str, str]:
        """
        Create a new API key
        
        Returns:
            (api_key_id, actual_key) tuple
        """
        if db is None:
            db = SessionLocal()
        
        try:
            # Generate API key components
            key_id = f"sk_{secrets.token_urlsafe(16)}"
            actual_key = f"{key_id}.{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(actual_key.encode()).hexdigest()
            prefix = actual_key[:12]
            
            # Create database record
            db_api_key = APIKey(
                id=key_id,
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                prefix=prefix,
                scopes=json.dumps(scopes or []),
                rate_limit_tier=tier.value,
                expires_at=expires_at
            )
            
            db.add(db_api_key)
            db.commit()
            
            # Create configuration
            config = APIKeyConfig(
                tier=tier,
                scopes=scopes or [],
                rate_limits=self.tier_configs[tier].copy(),
                expires_at=expires_at
            )
            
            self.api_keys[key_id] = config
            
            # Initialize usage tracking
            self.quota_usage[key_id] = {}
            for quota_type in QuotaType:
                self.quota_usage[key_id][quota_type] = QuotaUsage(
                    quota_type=quota_type,
                    used=0.0,
                    limit=float(self.tier_configs[tier].get(f"{quota_type.value}_limit", 1000000)),
                    reset_time=datetime.utcnow() + timedelta(days=30)
                )
            
            # Log creation
            security_logger.log_security_event(
                event_type="API_KEY_CREATED",
                level=SecurityEventLevel.INFO,
                user_id=user_id,
                details={
                    "key_id": key_id,
                    "name": name,
                    "tier": tier.value,
                    "scopes": scopes
                }
            )
            
            return key_id, actual_key
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating API key: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create API key"
            )
        finally:
            if db:
                db.close()
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info"""
        try:
            # Extract key ID from the key
            if '.' not in api_key:
                return None
            
            key_id = api_key.split('.')[0]
            
            # Check in-memory cache first
            if key_id in self.api_keys:
                config = self.api_keys[key_id]
                
                # Check expiration
                if config.expires_at and datetime.utcnow() > config.expires_at:
                    return None
                
                # Verify actual key
                db = SessionLocal()
                try:
                    db_key = db.get(APIKey, key_id)
                    if db_key and db_key.is_active:
                        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                        if key_hash == db_key.key_hash:
                            # Update last used
                            db_key.last_used = datetime.utcnow()
                            db.commit()
                            
                            return {
                                "key_id": key_id,
                                "user_id": db_key.user_id,
                                "config": config,
                                "scopes": json.loads(db_key.scopes or "[]"),
                                "tier": config.tier
                            }
                finally:
                    db.close()
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    async def revoke_api_key(self, key_id: str, user_id: int) -> bool:
        """Revoke an API key"""
        db = SessionLocal()
        try:
            db_key = db.get(APIKey, key_id)
            if db_key and db_key.user_id == user_id:
                db_key.is_active = False
                db.commit()
                
                # Remove from memory
                if key_id in self.api_keys:
                    del self.api_keys[key_id]
                
                # Log revocation
                security_logger.log_security_event(
                    event_type="API_KEY_REVOKED",
                    level=SecurityEventLevel.INFO,
                    user_id=user_id,
                    details={"key_id": key_id}
                )
                
                return True
            
            return False
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error revoking API key: {e}")
            return False
        finally:
            db.close()
    
    async def rotate_api_key(self, key_id: str, user_id: int) -> Optional[str]:
        """Rotate an API key (generate new key, keep same ID)"""
        db = SessionLocal()
        try:
            db_key = db.get(APIKey, key_id)
            if db_key and db_key.user_id == user_id and db_key.is_active:
                # Generate new key
                new_actual_key = f"{key_id}.{secrets.token_urlsafe(32)}"
                new_key_hash = hashlib.sha256(new_actual_key.encode()).hexdigest()
                new_prefix = new_actual_key[:12]
                
                # Update database
                db_key.key_hash = new_key_hash
                db_key.prefix = new_prefix
                db.commit()
                
                # Log rotation
                security_logger.log_security_event(
                    event_type="API_KEY_ROTATED",
                    level=SecurityEventLevel.INFO,
                    user_id=user_id,
                    details={"key_id": key_id}
                )
                
                return new_actual_key
            
            return None
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error rotating API key: {e}")
            return None
        finally:
            db.close()
    
    # ========================================
    # QUOTA AND USAGE MANAGEMENT
    # ========================================
    
    async def check_quota(
        self,
        key_id: str,
        quota_type: QuotaType,
        usage_amount: float = 1.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if quota allows usage"""
        if key_id not in self.quota_usage:
            return False, {"error": "Invalid API key"}
        
        if quota_type not in self.quota_usage[key_id]:
            return True, {"unlimited": True}  # No limit set
        
        quota = self.quota_usage[key_id][quota_type]
        
        # Check if quota reset time has passed
        if datetime.utcnow() > quota.reset_time:
            quota.used = 0.0
            quota.reset_time = datetime.utcnow() + timedelta(days=30)
            quota.overage_count = 0
        
        # Check if usage would exceed limit
        if quota.used + usage_amount > quota.limit:
            quota.overage_count += 1
            return False, {
                "quota_exceeded": True,
                "used": quota.used,
                "limit": quota.limit,
                "reset_time": quota.reset_time.isoformat(),
                "overage_count": quota.overage_count
            }
        
        return True, {
            "used": quota.used,
            "limit": quota.limit,
            "remaining": quota.limit - quota.used,
            "reset_time": quota.reset_time.isoformat()
        }
    
    async def record_usage(
        self,
        key_id: Optional[str],
        user_id: Optional[int],
        request: Request,
        response_time_ms: float,
        status_code: int,
        request_size: int,
        response_size: int,
        error_message: Optional[str] = None
    ):
        """Record API usage for monitoring and billing"""
        
        # Create usage record
        usage_record = UsageRecord(
            timestamp=datetime.utcnow(),
            api_key_id=key_id,
            user_id=user_id,
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size=request_size,
            response_size=response_size,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent"),
            error_message=error_message
        )
        
        # Store in memory (recent usage)
        if key_id:
            self.api_key_usage[key_id].append(usage_record)
            
            # Update quotas
            if key_id in self.quota_usage:
                # Update request quota
                if QuotaType.REQUESTS in self.quota_usage[key_id]:
                    self.quota_usage[key_id][QuotaType.REQUESTS].used += 1.0
                
                # Update bandwidth quota
                if QuotaType.BANDWIDTH in self.quota_usage[key_id]:
                    self.quota_usage[key_id][QuotaType.BANDWIDTH].used += (request_size + response_size)
                
                # Update compute quota (based on response time)
                if QuotaType.COMPUTE in self.quota_usage[key_id]:
                    self.quota_usage[key_id][QuotaType.COMPUTE].used += response_time_ms / 1000.0
        
        # Update global metrics
        self.usage_metrics["total_requests"] += 1
        if user_id:
            self.usage_metrics["unique_users"].add(user_id)
        self.usage_metrics["unique_ips"].add(usage_record.ip_address)
        
        if status_code >= 400:
            self.usage_metrics["error_count"] += 1
        
        # Update average response time
        current_avg = self.usage_metrics["avg_response_time"]
        total_requests = self.usage_metrics["total_requests"]
        self.usage_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time_ms) / total_requests
        )
        
        # Store in Redis for persistence (if available)
        if self.redis_client:
            try:
                usage_key = f"usage:{key_id or 'anonymous'}:{datetime.utcnow().strftime('%Y%m%d%H')}"
                self.redis_client.lpush(usage_key, json.dumps(asdict(usage_record), default=str))
                self.redis_client.expire(usage_key, 86400 * 7)  # Keep for 7 days
            except Exception as e:
                logger.error(f"Error storing usage in Redis: {e}")
    
    # ========================================
    # REQUEST/RESPONSE MANAGEMENT
    # ========================================
    
    async def process_request(
        self,
        request: Request,
        api_key_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process incoming request with validation and rate limiting"""
        
        start_time = time.time()
        
        # Extract request information
        endpoint = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        content_length = int(request.headers.get("content-length", 0))
        
        # Validate API key if present
        if api_key_info:
            key_id = api_key_info["key_id"]
            config = api_key_info["config"]
            user_id = api_key_info["user_id"]
            
            # Check tier-specific endpoint access
            tier_config = self.tier_configs[config.tier]
            allowed_endpoints = tier_config["allowed_endpoints"]
            
            if allowed_endpoints != ["*"]:
                endpoint_allowed = any(
                    endpoint.startswith(allowed.replace("*", ""))
                    for allowed in allowed_endpoints
                )
                if not endpoint_allowed:
                    return False, {
                        "error": "endpoint_not_allowed",
                        "message": f"Endpoint {endpoint} not allowed for {config.tier.value} tier"
                    }
            
            # Check request size limits
            max_request_size = tier_config["max_request_size"]
            if content_length > max_request_size:
                return False, {
                    "error": "request_too_large",
                    "message": f"Request size {content_length} exceeds limit {max_request_size}",
                    "max_size": max_request_size
                }
            
            # Check quotas
            for quota_type in [QuotaType.REQUESTS, QuotaType.BANDWIDTH]:
                usage_amount = 1.0 if quota_type == QuotaType.REQUESTS else content_length
                allowed, quota_info = await self.check_quota(key_id, quota_type, usage_amount)
                
                if not allowed:
                    return False, {
                        "error": "quota_exceeded",
                        "quota_type": quota_type.value,
                        **quota_info
                    }
        else:
            key_id = None
            user_id = None
            config = None
        
        # Rate limiting check
        context = self._determine_rate_limit_context(endpoint)
        allowed, rate_info = await self.rate_limiter.check_rate_limit(
            request, context, user_id, key_id
        )
        
        if not allowed:
            return False, {
                "error": "rate_limit_exceeded",
                **rate_info
            }
        
        # Check IP restrictions
        if config and config.allowed_ips:
            if client_ip not in config.allowed_ips:
                return False, {
                    "error": "ip_not_allowed",
                    "message": f"IP {client_ip} not in allowed list"
                }
        
        # Check for blocked IPs
        if client_ip in self.blocked_ips:
            block_info = self.blocked_ips[client_ip]
            if datetime.utcnow() < block_info["blocked_until"]:
                return False, {
                    "error": "ip_blocked",
                    "message": "IP temporarily blocked",
                    "blocked_until": block_info["blocked_until"].isoformat()
                }
        
        return True, {
            "allowed": True,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
    
    def _determine_rate_limit_context(self, endpoint: str) -> str:
        """Determine rate limiting context based on endpoint"""
        if "/auth/" in endpoint:
            return "auth"
        elif "/upload" in endpoint:
            return "upload"
        elif "/api/" in endpoint:
            return "api"
        else:
            return "default"
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and valid"""
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if datetime.utcnow() < cached["expires_at"]:
                self.cache_stats["hits"] += 1
                return cached
            else:
                # Remove expired cache
                del self.response_cache[cache_key]
                self.cache_stats["size"] -= 1
        
        self.cache_stats["misses"] += 1
        return None
    
    async def cache_response(
        self,
        cache_key: str,
        response_data: Dict[str, Any],
        ttl_seconds: int = 300
    ):
        """Cache response for future use"""
        if len(self.response_cache) > 10000:  # Prevent memory overflow
            # Remove oldest entries
            sorted_items = sorted(
                self.response_cache.items(),
                key=lambda x: x[1]["created_at"]
            )
            for key, _ in sorted_items[:1000]:
                del self.response_cache[key]
            self.cache_stats["size"] -= 1000
        
        self.response_cache[cache_key] = {
            "data": response_data,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds)
        }
        self.cache_stats["size"] += 1
    
    # ========================================
    # MONITORING AND ANALYTICS
    # ========================================
    
    async def get_usage_analytics(
        self,
        key_id: Optional[str] = None,
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive usage analytics"""
        
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        analytics = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0.0,
                "total_bandwidth": 0,
                "unique_ips": 0,
                "peak_rps": 0.0
            },
            "endpoints": {},
            "time_series": {},
            "errors": {},
            "geographic": {}
        }
        
        # Filter usage records
        usage_records = []
        
        if key_id and key_id in self.api_key_usage:
            usage_records = [
                record for record in self.api_key_usage[key_id]
                if start_date <= record.timestamp <= end_date
            ]
        elif user_id:
            # Collect from all user's API keys
            db = SessionLocal()
            try:
                user_keys = db.query(APIKey).filter(
                    APIKey.user_id == user_id,
                    APIKey.is_active == True
                ).all()
                
                for db_key in user_keys:
                    if db_key.id in self.api_key_usage:
                        user_records = [
                            record for record in self.api_key_usage[db_key.id]
                            if start_date <= record.timestamp <= end_date
                        ]
                        usage_records.extend(user_records)
            finally:
                db.close()
        else:
            # Global analytics
            for key_usage in self.api_key_usage.values():
                period_records = [
                    record for record in key_usage
                    if start_date <= record.timestamp <= end_date
                ]
                usage_records.extend(period_records)
        
        if not usage_records:
            return analytics
        
        # Calculate summary statistics
        analytics["summary"]["total_requests"] = len(usage_records)
        analytics["summary"]["successful_requests"] = len([
            r for r in usage_records if r.status_code < 400
        ])
        analytics["summary"]["failed_requests"] = len([
            r for r in usage_records if r.status_code >= 400
        ])
        
        if usage_records:
            analytics["summary"]["avg_response_time"] = statistics.mean([
                r.response_time_ms for r in usage_records
            ])
            analytics["summary"]["total_bandwidth"] = sum([
                r.request_size + r.response_size for r in usage_records
            ])
            analytics["summary"]["unique_ips"] = len(set([
                r.ip_address for r in usage_records
            ]))
        
        # Endpoint statistics
        endpoint_stats = defaultdict(lambda: {
            "count": 0, "avg_response_time": 0.0, "error_count": 0
        })
        
        for record in usage_records:
            endpoint = f"{record.method} {record.endpoint}"
            endpoint_stats[endpoint]["count"] += 1
            if record.status_code >= 400:
                endpoint_stats[endpoint]["error_count"] += 1
        
        # Calculate average response times
        for endpoint in endpoint_stats:
            endpoint_records = [r for r in usage_records if f"{r.method} {r.endpoint}" == endpoint]
            if endpoint_records:
                endpoint_stats[endpoint]["avg_response_time"] = statistics.mean([
                    r.response_time_ms for r in endpoint_records
                ])
        
        analytics["endpoints"] = dict(endpoint_stats)
        
        # Time series data (hourly buckets)
        time_series = defaultdict(lambda: {"requests": 0, "errors": 0, "avg_response_time": 0.0})
        
        for record in usage_records:
            hour_bucket = record.timestamp.replace(minute=0, second=0, microsecond=0)
            hour_key = hour_bucket.isoformat()
            time_series[hour_key]["requests"] += 1
            if record.status_code >= 400:
                time_series[hour_key]["errors"] += 1
        
        analytics["time_series"] = dict(time_series)
        
        # Error analysis
        error_stats = defaultdict(int)
        for record in usage_records:
            if record.status_code >= 400:
                error_key = f"{record.status_code}"
                if record.error_message:
                    error_key += f" - {record.error_message[:100]}"
                error_stats[error_key] += 1
        
        analytics["errors"] = dict(error_stats)
        
        return analytics
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time API metrics"""
        current_time = datetime.utcnow()
        
        # Calculate current RPS (last minute)
        current_requests = 0
        minute_ago = current_time - timedelta(minutes=1)
        
        for key_usage in self.api_key_usage.values():
            current_requests += len([
                record for record in key_usage
                if record.timestamp > minute_ago
            ])
        
        self.usage_metrics["current_rps"] = current_requests / 60.0
        self.usage_metrics["peak_rps"] = max(
            self.usage_metrics["peak_rps"],
            self.usage_metrics["current_rps"]
        )
        
        # System health indicators
        health_score = 100.0
        
        # Reduce score based on error rate
        if self.usage_metrics["total_requests"] > 0:
            error_rate = self.usage_metrics["error_count"] / self.usage_metrics["total_requests"]
            health_score -= (error_rate * 50)  # Max 50 point reduction
        
        # Reduce score based on response time
        if self.usage_metrics["avg_response_time"] > 1000:  # > 1 second
            health_score -= 30
        elif self.usage_metrics["avg_response_time"] > 500:  # > 500ms
            health_score -= 15
        
        return {
            "timestamp": current_time.isoformat(),
            "current_rps": self.usage_metrics["current_rps"],
            "peak_rps": self.usage_metrics["peak_rps"],
            "total_requests": self.usage_metrics["total_requests"],
            "error_count": self.usage_metrics["error_count"],
            "avg_response_time": self.usage_metrics["avg_response_time"],
            "active_api_keys": len(self.api_keys),
            "unique_users_today": len(self.usage_metrics["unique_users"]),
            "unique_ips_today": len(self.usage_metrics["unique_ips"]),
            "cache_hit_rate": (
                self.cache_stats["hits"] / 
                max(1, self.cache_stats["hits"] + self.cache_stats["misses"])
            ) * 100,
            "cache_size": self.cache_stats["size"],
            "system_health_score": max(0, health_score),
            "threat_level": self.rate_limiter.global_stats().get("under_attack", False)
        }
    
    # ========================================
    # SECURITY AND COMPLIANCE
    # ========================================
    
    async def detect_suspicious_activity(
        self,
        usage_record: UsageRecord
    ) -> Optional[Dict[str, Any]]:
        """Detect and flag suspicious activity"""
        
        suspicious_indicators = []
        severity = "low"
        
        # Check for rapid-fire requests
        if usage_record.api_key_id:
            recent_requests = [
                r for r in self.api_key_usage[usage_record.api_key_id]
                if (usage_record.timestamp - r.timestamp).total_seconds() < 10
            ]
            
            if len(recent_requests) > 50:  # 50 requests in 10 seconds
                suspicious_indicators.append("rapid_fire_requests")
                severity = "high"
        
        # Check for unusual endpoint patterns
        if usage_record.endpoint in ["/admin", "/config", "/.env", "/backup"]:
            suspicious_indicators.append("probing_sensitive_endpoints")
            severity = "medium"
        
        # Check for error patterns
        if usage_record.status_code in [401, 403]:
            # Count recent auth failures from this IP
            ip_errors = [
                r for r in self.api_key_usage.get(usage_record.api_key_id, [])
                if (r.ip_address == usage_record.ip_address and
                    r.status_code in [401, 403] and
                    (usage_record.timestamp - r.timestamp).total_seconds() < 300)
            ]
            
            if len(ip_errors) > 10:  # 10 auth failures in 5 minutes
                suspicious_indicators.append("brute_force_attempt")
                severity = "critical"
        
        # Check user agent patterns
        if usage_record.user_agent:
            suspicious_agents = ["bot", "scraper", "scanner", "curl", "python-requests"]
            if any(agent in usage_record.user_agent.lower() for agent in suspicious_agents):
                suspicious_indicators.append("automated_client")
                severity = "low"
        
        if suspicious_indicators:
            activity = {
                "timestamp": usage_record.timestamp,
                "ip_address": usage_record.ip_address,
                "api_key_id": usage_record.api_key_id,
                "user_id": usage_record.user_id,
                "indicators": suspicious_indicators,
                "severity": severity,
                "endpoint": usage_record.endpoint,
                "status_code": usage_record.status_code
            }
            
            self.suspicious_activities.append(activity)
            
            # Auto-block for critical severity
            if severity == "critical":
                await self.block_ip(
                    usage_record.ip_address,
                    reason="Automated blocking due to critical suspicious activity",
                    duration_minutes=60
                )
            
            return activity
        
        return None
    
    async def block_ip(
        self,
        ip_address: str,
        reason: str,
        duration_minutes: int = 60
    ):
        """Block an IP address"""
        blocked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        self.blocked_ips[ip_address] = {
            "blocked_at": datetime.utcnow(),
            "blocked_until": blocked_until,
            "reason": reason,
            "block_count": self.blocked_ips.get(ip_address, {}).get("block_count", 0) + 1
        }
        
        security_logger.log_security_event(
            event_type="IP_BLOCKED",
            level=SecurityEventLevel.WARNING,
            client_ip=ip_address,
            details={
                "reason": reason,
                "duration_minutes": duration_minutes,
                "blocked_until": blocked_until.isoformat()
            }
        )
    
    async def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            del self.blocked_ips[ip_address]
            
            security_logger.log_security_event(
                event_type="IP_UNBLOCKED",
                level=SecurityEventLevel.INFO,
                client_ip=ip_address,
                details={"action": "manual_unblock"}
            )
    
    # ========================================
    # BACKGROUND TASKS
    # ========================================
    
    async def _update_performance_metrics(self):
        """Update performance metrics periodically"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Calculate metrics for last minute
                minute_metrics = {
                    "timestamp": current_time,
                    "requests": 0,
                    "errors": 0,
                    "avg_response_time": 0.0,
                    "unique_ips": set()
                }
                
                minute_ago = current_time - timedelta(minutes=1)
                
                # Aggregate from all API keys
                all_records = []
                for key_usage in self.api_key_usage.values():
                    recent_records = [
                        r for r in key_usage if r.timestamp > minute_ago
                    ]
                    all_records.extend(recent_records)
                
                if all_records:
                    minute_metrics["requests"] = len(all_records)
                    minute_metrics["errors"] = len([
                        r for r in all_records if r.status_code >= 400
                    ])
                    minute_metrics["avg_response_time"] = statistics.mean([
                        r.response_time_ms for r in all_records
                    ])
                    minute_metrics["unique_ips"] = len(set([
                        r.ip_address for r in all_records
                    ]))
                
                # Convert set to count for storage
                minute_metrics["unique_ips"] = minute_metrics["unique_ips"]
                
                self.performance_history.append(minute_metrics)
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self):
        """Clean up expired data periodically"""
        while True:
            try:
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=24)  # Keep 24 hours
                
                # Clean up usage records
                for key_id in list(self.api_key_usage.keys()):
                    self.api_key_usage[key_id] = deque([
                        record for record in self.api_key_usage[key_id]
                        if record.timestamp > cutoff_time
                    ], maxlen=10000)
                    
                    # Remove empty entries
                    if not self.api_key_usage[key_id]:
                        del self.api_key_usage[key_id]
                
                # Clean up expired cached responses
                expired_keys = [
                    key for key, cached in self.response_cache.items()
                    if current_time > cached["expires_at"]
                ]
                
                for key in expired_keys:
                    del self.response_cache[key]
                    self.cache_stats["size"] -= 1
                
                # Clean up expired IP blocks
                expired_blocks = [
                    ip for ip, block_info in self.blocked_ips.items()
                    if current_time > block_info["blocked_until"]
                ]
                
                for ip in expired_blocks:
                    del self.blocked_ips[ip]
                
                logger.info(f"Cleanup completed: removed {len(expired_keys)} cached responses, "
                           f"{len(expired_blocks)} IP blocks")
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    async def _detect_anomalies(self):
        """Detect anomalies in API usage patterns"""
        while True:
            try:
                # Get recent performance data
                if len(self.performance_history) < 10:
                    await asyncio.sleep(300)  # Wait for more data
                    continue
                
                recent_data = list(self.performance_history)[-10:]  # Last 10 minutes
                
                # Calculate baselines
                baseline_rps = statistics.mean([m["requests"] for m in recent_data])
                baseline_response_time = statistics.mean([m["avg_response_time"] for m in recent_data])
                baseline_error_rate = statistics.mean([m["errors"] / max(1, m["requests"]) for m in recent_data])
                
                current_minute = recent_data[-1] if recent_data else None
                if not current_minute:
                    await asyncio.sleep(300)
                    continue
                
                anomalies = []
                
                # Check for traffic spikes
                if current_minute["requests"] > baseline_rps * 3:
                    anomalies.append({
                        "type": "traffic_spike",
                        "severity": "high",
                        "current": current_minute["requests"],
                        "baseline": baseline_rps
                    })
                
                # Check for response time increases
                if current_minute["avg_response_time"] > baseline_response_time * 2:
                    anomalies.append({
                        "type": "performance_degradation",
                        "severity": "medium",
                        "current": current_minute["avg_response_time"],
                        "baseline": baseline_response_time
                    })
                
                # Check for error rate increases
                current_error_rate = current_minute["errors"] / max(1, current_minute["requests"])
                if current_error_rate > baseline_error_rate * 2 and current_error_rate > 0.1:
                    anomalies.append({
                        "type": "error_rate_spike",
                        "severity": "high",
                        "current": current_error_rate,
                        "baseline": baseline_error_rate
                    })
                
                # Log anomalies
                for anomaly in anomalies:
                    security_logger.log_security_event(
                        event_type="ANOMALY_DETECTED",
                        level=SecurityEventLevel.WARNING if anomaly["severity"] == "high" else SecurityEventLevel.INFO,
                        details=anomaly
                    )
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(300)
    
    async def _update_quota_usage(self):
        """Update quota usage from Redis/database"""
        while True:
            try:
                # Sync quota usage from persistent storage
                if self.redis_client:
                    for key_id in self.api_keys.keys():
                        try:
                            quota_key = f"quota:{key_id}"
                            stored_quotas = self.redis_client.hgetall(quota_key)
                            
                            for quota_type_str, usage_json in stored_quotas.items():
                                try:
                                    quota_type = QuotaType(quota_type_str)
                                    usage_data = json.loads(usage_json)
                                    
                                    if key_id not in self.quota_usage:
                                        self.quota_usage[key_id] = {}
                                    
                                    self.quota_usage[key_id][quota_type] = QuotaUsage(
                                        quota_type=quota_type,
                                        used=usage_data["used"],
                                        limit=usage_data["limit"],
                                        reset_time=datetime.fromisoformat(usage_data["reset_time"]),
                                        overage_count=usage_data.get("overage_count", 0)
                                    )
                                except (ValueError, KeyError, json.JSONDecodeError) as e:
                                    logger.error(f"Error parsing quota data for {key_id}: {e}")
                        except Exception as e:
                            logger.error(f"Error syncing quotas for {key_id}: {e}")
                
                # Persist current quota usage to Redis
                if self.redis_client:
                    for key_id, quotas in self.quota_usage.items():
                        quota_key = f"quota:{key_id}"
                        quota_data = {}
                        
                        for quota_type, usage in quotas.items():
                            quota_data[quota_type.value] = json.dumps({
                                "used": usage.used,
                                "limit": usage.limit,
                                "reset_time": usage.reset_time.isoformat(),
                                "overage_count": usage.overage_count,
                                "last_updated": usage.last_updated.isoformat()
                            })
                        
                        if quota_data:
                            self.redis_client.hmset(quota_key, quota_data)
                            self.redis_client.expire(quota_key, 86400 * 31)  # 31 days
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating quota usage: {e}")
                await asyncio.sleep(300)


# Global instance
api_management_agent = APIManagementAgent()