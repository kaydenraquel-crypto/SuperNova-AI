"""
Advanced Rate Limiting System for SuperNova AI
Minimal working version to resolve import issues
"""

from enum import Enum
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import time
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_capacity: int = 20
    burst_refill_rate: float = 1.0
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    adaptive_enabled: bool = True


@dataclass
class ClientProfile:
    """Client profile for rate limiting"""
    client_id: str
    ip_address: str
    user_agent: Optional[str] = None
    request_history: deque = None
    tokens: float = 0.0
    last_refill: float = 0.0
    reputation_score: int = 70
    threat_level: ThreatLevel = ThreatLevel.LOW
    bot_probability: float = 0.0
    window_start: int = 0
    current_window_count: int = 0
    endpoint_usage: defaultdict = None
    method_distribution: defaultdict = None
    avg_request_interval: float = 0.0
    consecutive_failures: int = 0
    circuit_open: bool = False
    last_failure_time: float = 0.0
    blacklist_status: bool = False
    legitimate_score: float = 1.0
    
    def __post_init__(self):
        if self.request_history is None:
            self.request_history = deque(maxlen=1000)
        if self.endpoint_usage is None:
            self.endpoint_usage = defaultdict(int)
        if self.method_distribution is None:
            self.method_distribution = defaultdict(int)


class AdvancedRateLimiter:
    """Advanced rate limiter with threat detection"""
    
    def __init__(self):
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.rate_configs = {
            "default": RateLimitConfig(),
            "api": RateLimitConfig(requests_per_minute=200, requests_per_hour=5000),
            "auth": RateLimitConfig(requests_per_minute=20, requests_per_hour=100),
            "upload": RateLimitConfig(requests_per_minute=10, requests_per_hour=50)
        }
        self.threat_level_multipliers = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 0.7,
            ThreatLevel.HIGH: 0.3,
            ThreatLevel.CRITICAL: 0.1
        }
        self.global_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "current_rps": 0.0,
            "peak_rps": 0.0,
            "active_clients": 0
        }
        self.suspicious_ips = set()
        self.known_good_ips = set()
        self.ip_blacklist = []
        self.blocked_countries = []
        self.allowed_countries = []

    async def check_rate_limit(self, request, context: str = "default") -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits"""
        try:
            # Extract client information
            client_id = self._generate_client_id(
                ip=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent", ""),
                user_id=getattr(request.state, 'user_id', None),
                api_key=request.headers.get("x-api-key")
            )
            
            # Get or create client profile
            profile = self._get_client_profile(
                client_id=client_id,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent")
            )
            
            # Get rate limit configuration
            config = self.rate_configs.get(context, self.rate_configs["default"])
            
            # Simple token bucket check
            current_time = time.time()
            
            # Initialize tokens if first request
            if profile.last_refill == 0:
                profile.tokens = config.burst_capacity
                profile.last_refill = current_time
            
            # Refill tokens based on time elapsed
            time_elapsed = current_time - profile.last_refill
            tokens_to_add = time_elapsed * config.burst_refill_rate
            profile.tokens = min(config.burst_capacity, profile.tokens + tokens_to_add)
            profile.last_refill = current_time
            
            # Check if tokens available
            if profile.tokens >= 1.0:
                profile.tokens -= 1.0
                allowed = True
                retry_after = 0
            else:
                allowed = False
                retry_after = int((1.0 - profile.tokens) / config.burst_refill_rate)
            
            # Update global metrics
            self.global_metrics["total_requests"] += 1
            if not allowed:
                self.global_metrics["blocked_requests"] += 1
            
            # Prepare response info
            response_info = {
                "allowed": allowed,
                "limit": config.requests_per_minute,
                "remaining": max(0, int(profile.tokens)),
                "reset_time": int(current_time + 60),
                "retry_after": retry_after,
                "threat_level": profile.threat_level.value,
                "client_reputation": profile.reputation_score,
                "rate_limit_algorithm": config.algorithm.value
            }
            
            return allowed, response_info
            
        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            # Fail open - allow request but log error
            return True, {"error": str(e), "allowed": True}

    def _get_client_profile(self, client_id: str, ip_address: str, user_agent: Optional[str]) -> ClientProfile:
        """Get or create client profile"""
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = ClientProfile(
                client_id=client_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
        return self.client_profiles[client_id]

    def _generate_client_id(self, ip: str, user_agent: str, user_id: Optional[str], api_key: Optional[str]) -> str:
        """Generate unique client identifier"""
        if user_id:
            return f"user:{user_id}"
        elif api_key:
            import hashlib
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        else:
            import hashlib
            ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
            return f"ip:{ip}:ua:{ua_hash}"

    def _get_client_ip(self, request) -> str:
        """Extract client IP address with proxy support"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


# Global instance
advanced_rate_limiter = AdvancedRateLimiter()