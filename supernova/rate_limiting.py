"""
SuperNova AI Rate Limiting and DDoS Protection System
Advanced traffic control and attack mitigation
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
import asyncio
import logging

import redis
from fastapi import Request, HTTPException, status, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .security_config import security_settings
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Rate limit types"""
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


class ThrottleLevel(str, Enum):
    """Traffic throttle levels"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_capacity: int
    window_size: int = 60  # seconds
    
    def __post_init__(self):
        # Calculate derived limits
        self.requests_per_second = self.requests_per_minute / 60


@dataclass
class TrafficMetrics:
    """Traffic analysis metrics"""
    request_count: int = 0
    unique_ips: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    suspicious_activity_score: int = 0
    last_updated: datetime = None


class SlidingWindowCounter:
    """Sliding window rate limiting counter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def add_request(self, timestamp: float = None) -> bool:
        """Add request and return True if within limit"""
        if timestamp is None:
            timestamp = time.time()
        
        # Remove old requests outside window
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        # Check if at capacity
        if len(self.requests) >= self.max_requests:
            return False
        
        # Add new request
        self.requests.append(timestamp)
        return True
    
    def get_count(self, timestamp: float = None) -> int:
        """Get current request count in window"""
        if timestamp is None:
            timestamp = time.time()
        
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        return len(self.requests)
    
    def time_until_reset(self, timestamp: float = None) -> float:
        """Time until oldest request expires"""
        if timestamp is None:
            timestamp = time.time()
        
        if not self.requests:
            return 0
        
        return max(0, self.requests[0] + self.window_size - timestamp)


class TokenBucket:
    """Token bucket rate limiting algorithm"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens and return True if successful"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_available_tokens(self) -> int:
        """Get number of available tokens"""
        self._refill()
        return int(self.tokens)


class RateLimiter:
    """
    Comprehensive rate limiting system with multiple algorithms
    and adaptive throttling
    """
    
    def __init__(self):
        # Redis client for distributed rate limiting
        self.redis_client = self._setup_redis()
        
        # In-memory counters for local rate limiting
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        
        # Rate limit configurations
        self.rate_limits = self._load_rate_limits()
        
        # Traffic analysis
        self.traffic_metrics = TrafficMetrics()
        self.suspicious_ips: Dict[str, Dict] = {}
        
        # Adaptive throttling
        self.current_throttle_level = ThrottleLevel.NORMAL
        self.throttle_history = deque(maxlen=100)
        
        # Request patterns for anomaly detection
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis client for distributed rate limiting"""
        try:
            if hasattr(security_settings, 'REDIS_URL') and security_settings.REDIS_URL:
                return redis.from_url(
                    security_settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory rate limiting: {e}")
        
        return None
    
    def _load_rate_limits(self) -> Dict[RateLimitType, RateLimit]:
        """Load rate limit configurations"""
        return {
            RateLimitType.PER_IP: RateLimit(
                requests_per_minute=security_settings.RATE_LIMIT_PER_MINUTE,
                requests_per_hour=security_settings.RATE_LIMIT_PER_HOUR,
                requests_per_day=security_settings.RATE_LIMIT_PER_DAY,
                burst_capacity=security_settings.RATE_LIMIT_BURST
            ),
            RateLimitType.PER_USER: RateLimit(
                requests_per_minute=security_settings.RATE_LIMIT_PER_MINUTE * 2,
                requests_per_hour=security_settings.RATE_LIMIT_PER_HOUR * 2,
                requests_per_day=security_settings.RATE_LIMIT_PER_DAY * 2,
                burst_capacity=security_settings.RATE_LIMIT_BURST * 2
            ),
            RateLimitType.PER_API_KEY: RateLimit(
                requests_per_minute=security_settings.RATE_LIMIT_PER_MINUTE * 5,
                requests_per_hour=security_settings.RATE_LIMIT_PER_HOUR * 5,
                requests_per_day=security_settings.RATE_LIMIT_PER_DAY * 5,
                burst_capacity=security_settings.RATE_LIMIT_BURST * 3
            ),
            RateLimitType.GLOBAL: RateLimit(
                requests_per_minute=security_settings.RATE_LIMIT_PER_MINUTE * 100,
                requests_per_hour=security_settings.RATE_LIMIT_PER_HOUR * 100,
                requests_per_day=security_settings.RATE_LIMIT_PER_DAY * 100,
                burst_capacity=security_settings.RATE_LIMIT_BURST * 50
            )
        }
    
    async def check_rate_limit(
        self,
        request: Request,
        limit_type: RateLimitType = RateLimitType.PER_IP,
        identifier: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits
        
        Returns:
            (allowed, info) where info contains limit details
        """
        if not security_settings.RATE_LIMIT_ENABLED:
            return True, {"limit_type": "disabled"}
        
        # Get identifier for rate limiting
        if identifier is None:
            identifier = self._get_identifier(request, limit_type)
        
        # Get rate limit configuration
        rate_limit = self._get_adaptive_rate_limit(limit_type, identifier)
        
        # Check limits using different algorithms
        if self.redis_client:
            allowed, info = await self._check_redis_rate_limit(identifier, rate_limit, limit_type)
        else:
            allowed, info = await self._check_memory_rate_limit(identifier, rate_limit, limit_type)
        
        # Update traffic metrics
        await self._update_traffic_metrics(request, allowed, info)
        
        # Analyze request patterns
        await self._analyze_request_patterns(identifier, request, allowed)
        
        # Log rate limit events
        if not allowed:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
                level=SecurityEventLevel.WARNING,
                client_ip=request.client.host,
                endpoint=str(request.url.path),
                details={
                    "rate_limit_exceeded": True,
                    "limit_type": limit_type.value,
                    "identifier": identifier,
                    "throttle_level": self.current_throttle_level.value,
                    **info
                }
            )
        
        return allowed, info
    
    def _get_identifier(self, request: Request, limit_type: RateLimitType) -> str:
        """Get identifier for rate limiting based on type"""
        if limit_type == RateLimitType.PER_IP:
            # Use X-Forwarded-For if behind proxy, otherwise client IP
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            return request.client.host
        
        elif limit_type == RateLimitType.PER_USER:
            # Extract user ID from auth token (simplified)
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return hashlib.sha256(auth_header.encode()).hexdigest()[:16]
            return request.client.host
        
        elif limit_type == RateLimitType.PER_API_KEY:
            api_key = request.headers.get("X-API-Key", "")
            if api_key:
                return hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return request.client.host
        
        elif limit_type == RateLimitType.PER_ENDPOINT:
            return f"{request.method}:{request.url.path}"
        
        else:  # GLOBAL
            return "global"
    
    def _get_adaptive_rate_limit(self, limit_type: RateLimitType, identifier: str) -> RateLimit:
        """Get adaptive rate limit based on current conditions"""
        base_limit = self.rate_limits[limit_type]
        
        # Adjust based on throttle level
        throttle_multipliers = {
            ThrottleLevel.NORMAL: 1.0,
            ThrottleLevel.ELEVATED: 0.8,
            ThrottleLevel.HIGH: 0.5,
            ThrottleLevel.CRITICAL: 0.2
        }
        
        multiplier = throttle_multipliers[self.current_throttle_level]
        
        # Check if IP is suspicious
        if identifier in self.suspicious_ips:
            suspicious_data = self.suspicious_ips[identifier]
            severity = suspicious_data.get("severity", 1)
            multiplier *= max(0.1, 1.0 / severity)
        
        return RateLimit(
            requests_per_minute=int(base_limit.requests_per_minute * multiplier),
            requests_per_hour=int(base_limit.requests_per_hour * multiplier),
            requests_per_day=int(base_limit.requests_per_day * multiplier),
            burst_capacity=int(base_limit.burst_capacity * multiplier),
            window_size=base_limit.window_size
        )
    
    async def _check_redis_rate_limit(
        self,
        identifier: str,
        rate_limit: RateLimit,
        limit_type: RateLimitType
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis"""
        now = time.time()
        
        try:
            # Use Redis sliding window log
            key_minute = f"rate_limit:{limit_type.value}:{identifier}:minute"
            key_hour = f"rate_limit:{limit_type.value}:{identifier}:hour"
            key_day = f"rate_limit:{limit_type.value}:{identifier}:day"
            
            pipe = self.redis_client.pipeline()
            
            # Check minute window
            pipe.zremrangebyscore(key_minute, 0, now - 60)
            pipe.zcard(key_minute)
            pipe.zadd(key_minute, {str(now): now})
            pipe.expire(key_minute, 120)  # 2 minutes TTL
            
            # Check hour window
            pipe.zremrangebyscore(key_hour, 0, now - 3600)
            pipe.zcard(key_hour)
            pipe.zadd(key_hour, {str(now): now})
            pipe.expire(key_hour, 7200)  # 2 hours TTL
            
            # Check day window
            pipe.zremrangebyscore(key_day, 0, now - 86400)
            pipe.zcard(key_day)
            pipe.zadd(key_day, {str(now): now})
            pipe.expire(key_day, 172800)  # 2 days TTL
            
            results = pipe.execute()
            
            minute_count = results[1]
            hour_count = results[5]
            day_count = results[9]
            
            # Check limits
            if minute_count > rate_limit.requests_per_minute:
                return False, {
                    "limit": "per_minute",
                    "current": minute_count,
                    "max": rate_limit.requests_per_minute,
                    "reset_time": 60 - (now % 60)
                }
            
            if hour_count > rate_limit.requests_per_hour:
                return False, {
                    "limit": "per_hour",
                    "current": hour_count,
                    "max": rate_limit.requests_per_hour,
                    "reset_time": 3600 - (now % 3600)
                }
            
            if day_count > rate_limit.requests_per_day:
                return False, {
                    "limit": "per_day",
                    "current": day_count,
                    "max": rate_limit.requests_per_day,
                    "reset_time": 86400 - (now % 86400)
                }
            
            return True, {
                "minute_count": minute_count,
                "hour_count": hour_count,
                "day_count": day_count,
                "limits": {
                    "per_minute": rate_limit.requests_per_minute,
                    "per_hour": rate_limit.requests_per_hour,
                    "per_day": rate_limit.requests_per_day
                }
            }
        
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to memory-based rate limiting
            return await self._check_memory_rate_limit(identifier, rate_limit, limit_type)
    
    async def _check_memory_rate_limit(
        self,
        identifier: str,
        rate_limit: RateLimit,
        limit_type: RateLimitType
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using in-memory structures"""
        
        # Create key for this limit type and identifier
        key = f"{limit_type.value}:{identifier}"
        
        # Check sliding window (per minute)
        if key not in self.sliding_windows:
            self.sliding_windows[key] = SlidingWindowCounter(60, rate_limit.requests_per_minute)
        
        window = self.sliding_windows[key]
        
        if not window.add_request():
            return False, {
                "limit": "per_minute",
                "current": window.get_count(),
                "max": rate_limit.requests_per_minute,
                "reset_time": window.time_until_reset()
            }
        
        # Check token bucket for burst protection
        bucket_key = f"bucket:{key}"
        if bucket_key not in self.token_buckets:
            self.token_buckets[bucket_key] = TokenBucket(
                rate_limit.burst_capacity,
                rate_limit.requests_per_second
            )
        
        bucket = self.token_buckets[bucket_key]
        
        if not bucket.consume():
            return False, {
                "limit": "burst",
                "current": rate_limit.burst_capacity - bucket.get_available_tokens(),
                "max": rate_limit.burst_capacity,
                "reset_time": 1.0  # Tokens refill continuously
            }
        
        return True, {
            "current_window": window.get_count(),
            "available_tokens": bucket.get_available_tokens(),
            "limits": {
                "per_minute": rate_limit.requests_per_minute,
                "burst_capacity": rate_limit.burst_capacity
            }
        }
    
    async def _update_traffic_metrics(self, request: Request, allowed: bool, info: Dict[str, Any]):
        """Update traffic analysis metrics"""
        self.traffic_metrics.request_count += 1
        self.traffic_metrics.last_updated = datetime.utcnow()
        
        # Track unique IPs
        # This is simplified - in production, use more sophisticated tracking
        
        # Update error rate
        if not allowed:
            # Increment error count (simplified)
            pass
        
        # Update suspicious activity score based on patterns
        if not allowed or self._detect_suspicious_pattern(request):
            self.traffic_metrics.suspicious_activity_score += 1
        
        # Adjust throttle level based on metrics
        await self._adjust_throttle_level()
    
    async def _analyze_request_patterns(self, identifier: str, request: Request, allowed: bool):
        """Analyze request patterns for anomaly detection"""
        now = time.time()
        
        # Store request timestamp
        self.request_patterns[identifier].append(now)
        
        # Keep only recent requests (last hour)
        cutoff = now - 3600
        self.request_patterns[identifier] = [
            t for t in self.request_patterns[identifier] if t > cutoff
        ]
        
        # Analyze patterns
        requests = self.request_patterns[identifier]
        
        if len(requests) > 10:  # Need sufficient data
            # Check for suspicious patterns
            suspicion_score = 0
            
            # Pattern 1: Too many requests in short time
            recent_requests = [t for t in requests if t > now - 60]
            if len(recent_requests) > 50:
                suspicion_score += 2
            
            # Pattern 2: Regular interval requests (bot-like)
            if len(requests) >= 20:
                intervals = [requests[i] - requests[i-1] for i in range(1, len(requests))]
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
                
                # Low variance indicates regular intervals
                if variance < 1.0 and avg_interval < 5.0:
                    suspicion_score += 1
            
            # Pattern 3: High error rate
            if not allowed:
                suspicion_score += 1
            
            # Update suspicious IPs tracking
            if suspicion_score > 2:
                if identifier not in self.suspicious_ips:
                    self.suspicious_ips[identifier] = {
                        "first_seen": datetime.utcnow(),
                        "severity": 1,
                        "patterns": []
                    }
                
                suspicious_data = self.suspicious_ips[identifier]
                suspicious_data["severity"] = min(5, suspicious_data["severity"] + 1)
                suspicious_data["last_seen"] = datetime.utcnow()
                
                # Log suspicious activity
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
                    level=SecurityEventLevel.WARNING,
                    client_ip=identifier,
                    details={
                        "suspicion_score": suspicion_score,
                        "severity": suspicious_data["severity"],
                        "recent_requests": len(recent_requests),
                        "pattern_analysis": {
                            "total_requests": len(requests),
                            "time_window": "1_hour",
                            "avg_interval": avg_interval if 'avg_interval' in locals() else None
                        }
                    }
                )
    
    def _detect_suspicious_pattern(self, request: Request) -> bool:
        """Detect suspicious request patterns"""
        # Check for suspicious user agents
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_agents = [
            "bot", "crawler", "scraper", "spider", "scan",
            "python-requests", "curl", "wget", "automation"
        ]
        
        if any(agent in user_agent for agent in suspicious_agents):
            return True
        
        # Check for suspicious request parameters
        query_params = str(request.query_params).lower()
        suspicious_params = [
            "union", "select", "drop", "insert", "delete",
            "<script>", "javascript:", "eval(", "alert("
        ]
        
        if any(param in query_params for param in suspicious_params):
            return True
        
        # Check for missing or suspicious headers
        required_headers = ["user-agent", "accept"]
        if not all(header in request.headers for header in required_headers):
            return True
        
        return False
    
    async def _adjust_throttle_level(self):
        """Adjust throttle level based on current metrics"""
        current_time = datetime.utcnow()
        
        # Calculate metrics for the last minute
        minute_ago = current_time - timedelta(minutes=1)
        
        # Simple throttle level adjustment based on suspicious activity
        if self.traffic_metrics.suspicious_activity_score > 100:
            new_level = ThrottleLevel.CRITICAL
        elif self.traffic_metrics.suspicious_activity_score > 50:
            new_level = ThrottleLevel.HIGH
        elif self.traffic_metrics.suspicious_activity_score > 20:
            new_level = ThrottleLevel.ELEVATED
        else:
            new_level = ThrottleLevel.NORMAL
        
        if new_level != self.current_throttle_level:
            old_level = self.current_throttle_level
            self.current_throttle_level = new_level
            
            self.throttle_history.append({
                "timestamp": current_time,
                "old_level": old_level.value,
                "new_level": new_level.value,
                "reason": "traffic_analysis"
            })
            
            security_logger.log_security_event(
                event_type="THROTTLE_LEVEL_CHANGED",
                level=SecurityEventLevel.INFO if new_level == ThrottleLevel.NORMAL else SecurityEventLevel.WARNING,
                details={
                    "old_level": old_level.value,
                    "new_level": new_level.value,
                    "suspicious_score": self.traffic_metrics.suspicious_activity_score,
                    "total_requests": self.traffic_metrics.request_count
                }
            )
    
    async def reset_suspicious_ip(self, ip: str):
        """Reset suspicious status for IP"""
        if ip in self.suspicious_ips:
            del self.suspicious_ips[ip]
            
            security_logger.log_security_event(
                event_type="SUSPICIOUS_IP_RESET",
                level=SecurityEventLevel.INFO,
                client_ip=ip,
                details={"action": "manual_reset"}
            )
    
    def get_rate_limit_status(self, identifier: str, limit_type: RateLimitType) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        key = f"{limit_type.value}:{identifier}"
        
        status = {
            "identifier": identifier,
            "limit_type": limit_type.value,
            "throttle_level": self.current_throttle_level.value,
            "is_suspicious": identifier in self.suspicious_ips
        }
        
        if key in self.sliding_windows:
            window = self.sliding_windows[key]
            status["current_requests"] = window.get_count()
            status["time_until_reset"] = window.time_until_reset()
        
        if f"bucket:{key}" in self.token_buckets:
            bucket = self.token_buckets[f"bucket:{key}"]
            status["available_tokens"] = bucket.get_available_tokens()
        
        return status


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/metrics", "/favicon.ico"]:
            return await call_next(request)
        
        # Check rate limits
        allowed, info = await self.rate_limiter.check_rate_limit(request)
        
        if not allowed:
            # Return rate limit exceeded response
            headers = {
                "X-RateLimit-Limit": str(info.get("max", "unknown")),
                "X-RateLimit-Remaining": str(max(0, info.get("max", 0) - info.get("current", 0))),
                "X-RateLimit-Reset": str(int(time.time() + info.get("reset_time", 60))),
                "Retry-After": str(int(info.get("reset_time", 60)))
            }
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=headers
            )
        
        # Add rate limit headers to successful responses
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(info.get("limits", {}).get("per_minute", "unknown"))
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, info.get("limits", {}).get("per_minute", 0) - info.get("minute_count", 0))
        )
        
        return response


# Global rate limiter instance
rate_limiter = RateLimiter()


# =====================================
# DDOS PROTECTION
# =====================================

class DDoSProtection:
    """Advanced DDoS protection system"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.attack_detection_enabled = True
        self.mitigation_strategies = {
            "rate_limiting": True,
            "ip_blocking": True,
            "challenge_response": True,
            "geo_blocking": False
        }
        
        # Attack detection thresholds
        self.attack_thresholds = {
            "requests_per_second": 100,
            "unique_ips_threshold": 1000,
            "error_rate_threshold": 0.5,
            "pattern_anomaly_score": 10
        }
        
        # Blocked IPs and networks
        self.blocked_ips: Dict[str, Dict] = {}
        self.blocked_networks: List[str] = []
        
    async def detect_attack(self, metrics: TrafficMetrics) -> Dict[str, Any]:
        """Detect potential DDoS attack"""
        attack_indicators = {
            "attack_detected": False,
            "attack_type": None,
            "severity": "low",
            "indicators": []
        }
        
        # Check request volume
        if metrics.request_count > self.attack_thresholds["requests_per_second"]:
            attack_indicators["indicators"].append("high_request_volume")
            attack_indicators["attack_detected"] = True
            attack_indicators["attack_type"] = "volume_based"
        
        # Check error rate
        if metrics.error_rate > self.attack_thresholds["error_rate_threshold"]:
            attack_indicators["indicators"].append("high_error_rate")
            attack_indicators["attack_detected"] = True
            attack_indicators["attack_type"] = "application_layer"
        
        # Check suspicious activity score
        if metrics.suspicious_activity_score > self.attack_thresholds["pattern_anomaly_score"]:
            attack_indicators["indicators"].append("anomalous_patterns")
            attack_indicators["attack_detected"] = True
            attack_indicators["attack_type"] = "pattern_based"
        
        # Determine severity
        if len(attack_indicators["indicators"]) >= 3:
            attack_indicators["severity"] = "critical"
        elif len(attack_indicators["indicators"]) >= 2:
            attack_indicators["severity"] = "high"
        elif len(attack_indicators["indicators"]) >= 1:
            attack_indicators["severity"] = "medium"
        
        return attack_indicators
    
    async def activate_mitigation(self, attack_info: Dict[str, Any]):
        """Activate DDoS mitigation strategies"""
        if attack_info["severity"] == "critical":
            # Activate all mitigation strategies
            self.rate_limiter.current_throttle_level = ThrottleLevel.CRITICAL
            
            security_logger.log_security_event(
                event_type="DDOS_MITIGATION_ACTIVATED",
                level=SecurityEventLevel.CRITICAL,
                details={
                    "attack_type": attack_info["attack_type"],
                    "severity": attack_info["severity"],
                    "indicators": attack_info["indicators"],
                    "mitigation_level": "maximum"
                }
            )
        
        elif attack_info["severity"] == "high":
            self.rate_limiter.current_throttle_level = ThrottleLevel.HIGH
            
        elif attack_info["severity"] == "medium":
            self.rate_limiter.current_throttle_level = ThrottleLevel.ELEVATED


# Global DDoS protection instance
ddos_protection = DDoSProtection(rate_limiter)