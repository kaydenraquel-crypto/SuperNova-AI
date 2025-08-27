"""
SuperNova AI Advanced Rate Limiting and DDoS Protection System
Comprehensive traffic control, attack mitigation, and adaptive security measures
"""

import time
import hashlib
import json
import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any, Set
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import statistics
from ipaddress import IPv4Network, IPv6Network, ip_network, ip_address

import redis
from fastapi import Request, HTTPException, status, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .security_config import security_settings
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class AttackPattern(str, Enum):
    """Attack pattern types"""
    VOLUMETRIC = "volumetric"
    PROTOCOL = "protocol"
    APPLICATION = "application"
    DISTRIBUTED = "distributed"
    SLOWLORIS = "slowloris"
    HTTP_FLOOD = "http_flood"
    API_ABUSE = "api_abuse"


@dataclass
class RateLimitConfig:
    """Advanced rate limit configuration"""
    # Basic limits
    requests_per_second: int = 10
    requests_per_minute: int = 600
    requests_per_hour: int = 10000
    requests_per_day: int = 100000
    
    # Burst handling
    burst_capacity: int = 50
    burst_refill_rate: float = 1.0
    
    # Algorithm settings
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.ADAPTIVE
    window_size: int = 60  # seconds
    
    # Adaptive settings
    adaptive_enabled: bool = True
    baseline_threshold: float = 0.8
    spike_threshold: float = 2.0
    cooldown_period: int = 300  # seconds
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 50
    recovery_timeout: int = 60


@dataclass
class ClientProfile:
    """Comprehensive client behavior profile"""
    client_id: str
    ip_address: str
    user_agent: Optional[str] = None
    
    # Request patterns
    request_history: deque = None
    endpoint_usage: Dict[str, int] = None
    method_distribution: Dict[str, int] = None
    
    # Timing patterns
    avg_request_interval: float = 0.0
    request_timing_variance: float = 0.0
    peak_request_times: List[int] = None
    
    # Behavioral indicators
    bot_probability: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW
    legitimate_score: float = 1.0
    
    # Rate limiting state
    tokens: float = 0.0
    last_refill: float = 0.0
    current_window_count: int = 0
    window_start: float = 0.0
    
    # Circuit breaker state
    consecutive_failures: int = 0
    circuit_open: bool = False
    last_failure_time: float = 0.0
    
    # Reputation
    reputation_score: float = 100.0
    trust_level: str = "unknown"
    whitelist_status: bool = False
    blacklist_status: bool = False
    
    def __post_init__(self):
        if self.request_history is None:
            self.request_history = deque(maxlen=1000)
        if self.endpoint_usage is None:
            self.endpoint_usage = defaultdict(int)
        if self.method_distribution is None:
            self.method_distribution = defaultdict(int)
        if self.peak_request_times is None:
            self.peak_request_times = []


class AdvancedRateLimiter:
    """
    Advanced rate limiting system with DDoS protection and behavioral analysis
    """
    
    def __init__(self):
        self.redis_client = self._get_redis_client()
        
        # Client tracking
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.global_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "blocked_requests": 0,
            "current_rps": 0.0,
            "peak_rps": 0.0,
            "active_clients": 0
        }
        
        # Threat detection
        self.attack_patterns: Dict[AttackPattern, Dict] = {}
        self.suspicious_ips: Set[str] = set()
        self.known_good_ips: Set[str] = set()
        
        # IP whitelist/blacklist with CIDR support
        self.ip_whitelist: List[str] = [
            "127.0.0.0/8",     # Localhost
            "10.0.0.0/8",      # Private networks
            "172.16.0.0/12",
            "192.168.0.0/16"
        ]
        self.ip_blacklist: List[str] = []
        
        # Geofencing
        self.allowed_countries: Set[str] = set()
        self.blocked_countries: Set[str] = set()
        
        # Rate limit configurations by context
        self.rate_configs = {
            "default": RateLimitConfig(),
            "api": RateLimitConfig(
                requests_per_second=50,
                requests_per_minute=1000,
                requests_per_hour=20000,
                burst_capacity=100
            ),
            "auth": RateLimitConfig(
                requests_per_second=2,
                requests_per_minute=20,
                requests_per_hour=100,
                burst_capacity=5
            ),
            "upload": RateLimitConfig(
                requests_per_second=1,
                requests_per_minute=10,
                requests_per_hour=100,
                burst_capacity=3
            )
        }
        
        # Adaptive thresholds
        self.adaptive_baselines: Dict[str, float] = {}
        self.threat_level_multipliers = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.2,
            ThreatLevel.CRITICAL: 0.1
        }
        
        # Background tasks
        self._start_background_tasks()
    
    def _get_redis_client(self):
        """Initialize Redis client for distributed rate limiting"""
        try:
            import redis
            return redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 2)),
                decode_responses=True
            )
        except ImportError:
            logger.warning("Redis not available, using in-memory rate limiting")
            return None
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        asyncio.create_task(self._cleanup_expired_data())
        asyncio.create_task(self._update_global_metrics())
        asyncio.create_task(self._analyze_attack_patterns())
        asyncio.create_task(self._update_adaptive_baselines())
    
    async def check_rate_limit(
        self,
        request: Request,
        context: str = "default",
        user_id: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive rate limit check with threat analysis
        """
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        endpoint = str(request.url.path)
        method = request.method
        
        # Generate client identifier
        client_id = self._generate_client_id(client_ip, user_agent, user_id, api_key)
        
        # Get or create client profile
        profile = self._get_client_profile(client_id, client_ip, user_agent)
        
        # Pre-checks
        pre_check_result = await self._pre_flight_checks(profile, request)
        if not pre_check_result["allowed"]:
            return False, pre_check_result
        
        # Update profile with request data
        self._update_client_profile(profile, endpoint, method)\n        \n        # Get rate limit configuration\n        config = self.rate_configs.get(context, self.rate_configs["default"])\n        \n        # Apply adaptive adjustments\n        if config.adaptive_enabled:\n            config = self._apply_adaptive_adjustments(config, profile, context)\n        \n        # Check rate limits using specified algorithm\n        rate_check_result = self._check_rate_limit_algorithm(\n            profile, config, endpoint, method\n        )\n        \n        # Update global metrics\n        self._update_request_metrics(rate_check_result["allowed"])\n        \n        # Behavioral analysis\n        behavioral_analysis = self._analyze_client_behavior(profile)\n        \n        # Threat assessment\n        threat_assessment = self._assess_threat_level(profile, behavioral_analysis)\n        \n        # Security logging\n        if not rate_check_result["allowed"] or threat_assessment["threat_level"] != ThreatLevel.LOW:\n            await self._log_rate_limit_event(profile, rate_check_result, threat_assessment)\n        \n        # Prepare response\n        response_info = {\n            "allowed": rate_check_result["allowed"],\n            "limit": config.requests_per_minute,\n            "remaining": max(0, config.requests_per_minute - profile.current_window_count),\n            "reset_time": int(profile.window_start + config.window_size),\n            "retry_after": rate_check_result.get("retry_after", 60),\n            "threat_level": threat_assessment["threat_level"].value,\n            "client_reputation": profile.reputation_score,\n            "rate_limit_algorithm": config.algorithm.value\n        }\n        \n        return rate_check_result["allowed"], response_info\n    \n    async def _pre_flight_checks(self, profile: ClientProfile, request: Request) -> Dict[str, Any]:\n        """Pre-flight security checks before rate limiting"""\n        \n        result = {"allowed": True, "reason": None, "block_duration": 0}\n        \n        # IP whitelist/blacklist check\n        if profile.blacklist_status:\n            result["allowed"] = False\n            result["reason"] = "IP blacklisted"\n            return result\n        \n        if self._is_ip_blocked(profile.ip_address):\n            result["allowed"] = False\n            result["reason"] = "IP in blocked range"\n            return result\n        \n        # Circuit breaker check\n        if profile.circuit_open:\n            current_time = time.time()\n            if current_time - profile.last_failure_time < 60:  # Circuit open for 1 minute\n                result["allowed"] = False\n                result["reason"] = "Circuit breaker open"\n                result["retry_after"] = 60 - int(current_time - profile.last_failure_time)\n                return result\n            else:\n                # Try to close circuit\n                profile.circuit_open = False\n                profile.consecutive_failures = 0\n        \n        # Geofencing check (if enabled)\n        if self.blocked_countries or self.allowed_countries:\n            country = self._get_country_from_ip(profile.ip_address)\n            if country:\n                if self.blocked_countries and country in self.blocked_countries:\n                    result["allowed"] = False\n                    result["reason"] = f"Country {country} is blocked"\n                    return result\n                \n                if self.allowed_countries and country not in self.allowed_countries:\n                    result["allowed"] = False\n                    result["reason"] = f"Country {country} not in allowed list"\n                    return result\n        \n        # Check for ongoing attacks\n        if self._is_under_attack() and profile.reputation_score < 50:\n            result["allowed"] = False\n            result["reason"] = "System under attack, low reputation clients blocked"\n            return result\n        \n        return result\n    \n    def _check_rate_limit_algorithm(\n        self,\n        profile: ClientProfile,\n        config: RateLimitConfig,\n        endpoint: str,\n        method: str\n    ) -> Dict[str, Any]:\n        """Check rate limits using specified algorithm"""\n        \n        current_time = time.time()\n        \n        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:\n            return self._token_bucket_check(profile, config, current_time)\n        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:\n            return self._sliding_window_check(profile, config, current_time)\n        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:\n            return self._fixed_window_check(profile, config, current_time)\n        elif config.algorithm == RateLimitAlgorithm.ADAPTIVE:\n            return self._adaptive_rate_limit_check(profile, config, current_time, endpoint)\n        else:\n            return self._token_bucket_check(profile, config, current_time)  # Default\n    \n    def _token_bucket_check(self, profile: ClientProfile, config: RateLimitConfig, current_time: float) -> Dict[str, Any]:\n        \"\"\"Token bucket algorithm implementation\"\"\"\n        \n        # Initialize tokens if first request\n        if profile.last_refill == 0:\n            profile.tokens = config.burst_capacity\n            profile.last_refill = current_time\n        \n        # Refill tokens based on time elapsed\n        time_elapsed = current_time - profile.last_refill\n        tokens_to_add = time_elapsed * config.burst_refill_rate\n        profile.tokens = min(config.burst_capacity, profile.tokens + tokens_to_add)\n        profile.last_refill = current_time\n        \n        # Check if tokens available\n        if profile.tokens >= 1.0:\n            profile.tokens -= 1.0\n            return {\"allowed\": True, \"tokens_remaining\": profile.tokens}\n        else:\n            retry_after = int((1.0 - profile.tokens) / config.burst_refill_rate)\n            return {\"allowed\": False, \"retry_after\": retry_after}\n    \n    def _sliding_window_check(self, profile: ClientProfile, config: RateLimitConfig, current_time: float) -> Dict[str, Any]:\n        \"\"\"Sliding window algorithm implementation\"\"\"\n        \n        window_size = 60.0  # 1 minute window\n        \n        # Remove old requests outside window\n        cutoff_time = current_time - window_size\n        profile.request_history = deque(\n            [t for t in profile.request_history if t > cutoff_time],\n            maxlen=1000\n        )\n        \n        # Check if under limit\n        if len(profile.request_history) < config.requests_per_minute:\n            profile.request_history.append(current_time)\n            return {\"allowed\": True, \"requests_in_window\": len(profile.request_history)}\n        else:\n            # Calculate retry after based on oldest request in window\n            if profile.request_history:\n                oldest_request = profile.request_history[0]\n                retry_after = int(oldest_request + window_size - current_time)\n            else:\n                retry_after = 60\n            return {\"allowed\": False, \"retry_after\": max(1, retry_after)}\n    \n    def _fixed_window_check(self, profile: ClientProfile, config: RateLimitConfig, current_time: float) -> Dict[str, Any]:\n        \"\"\"Fixed window algorithm implementation\"\"\"\n        \n        window_size = 60.0  # 1 minute window\n        current_window = int(current_time // window_size)\n        \n        # Reset counter if new window\n        if profile.window_start != current_window:\n            profile.window_start = current_window\n            profile.current_window_count = 0\n        \n        # Check if under limit\n        if profile.current_window_count < config.requests_per_minute:\n            profile.current_window_count += 1\n            return {\"allowed\": True, \"requests_in_window\": profile.current_window_count}\n        else:\n            # Calculate time until next window\n            next_window_start = (current_window + 1) * window_size\n            retry_after = int(next_window_start - current_time)\n            return {\"allowed\": False, \"retry_after\": max(1, retry_after)}\n    \n    def _adaptive_rate_limit_check(\n        self,\n        profile: ClientProfile,\n        config: RateLimitConfig,\n        current_time: float,\n        endpoint: str\n    ) -> Dict[str, Any]:\n        \"\"\"Adaptive rate limiting based on system load and client behavior\"\"\"\n        \n        # Base check using sliding window\n        base_result = self._sliding_window_check(profile, config, current_time)\n        \n        if not base_result[\"allowed\"]:\n            return base_result\n        \n        # Adaptive adjustments\n        system_load = self._get_system_load()\n        client_trust = profile.reputation_score / 100.0\n        \n        # Adjust limits based on system load\n        if system_load > 0.8:\n            # High load - be more restrictive\n            if profile.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:\n                return {\"allowed\": False, \"retry_after\": 60, \"reason\": \"High system load\"}\n            elif profile.threat_level == ThreatLevel.MEDIUM and system_load > 0.9:\n                return {\"allowed\": False, \"retry_after\": 30, \"reason\": \"High system load\"}\n        \n        # Apply trust-based adjustments\n        if client_trust < 0.5 and system_load > 0.6:\n            return {\"allowed\": False, \"retry_after\": 30, \"reason\": \"Low trust during load\"}\n        \n        return base_result\n    \n    def _get_client_profile(self, client_id: str, ip_address: str, user_agent: Optional[str]) -> ClientProfile:\n        \"\"\"Get or create client profile\"\"\"\n        \n        if client_id not in self.client_profiles:\n            self.client_profiles[client_id] = ClientProfile(\n                client_id=client_id,\n                ip_address=ip_address,\n                user_agent=user_agent\n            )\n        \n        return self.client_profiles[client_id]\n    \n    def _update_client_profile(self, profile: ClientProfile, endpoint: str, method: str):\n        \"\"\"Update client profile with request information\"\"\"\n        \n        current_time = time.time()\n        \n        # Update endpoint usage\n        profile.endpoint_usage[endpoint] += 1\n        profile.method_distribution[method] += 1\n        \n        # Update timing patterns\n        if len(profile.request_history) > 0:\n            last_request = profile.request_history[-1]\n            interval = current_time - last_request\n            \n            # Calculate moving average of request intervals\n            if profile.avg_request_interval == 0:\n                profile.avg_request_interval = interval\n            else:\n                profile.avg_request_interval = (profile.avg_request_interval * 0.9) + (interval * 0.1)\n        \n        # Add to request history (already done in algorithm checks)\n        # This is for additional analysis\n        \n    def _analyze_client_behavior(self, profile: ClientProfile) -> Dict[str, Any]:\n        \"\"\"Analyze client behavior for bot detection and threat assessment\"\"\"\n        \n        analysis = {\n            \"bot_indicators\": [],\n            \"suspicious_patterns\": [],\n            \"legitimacy_score\": 1.0,\n            \"behavioral_flags\": []\n        }\n        \n        if len(profile.request_history) < 10:\n            return analysis  # Not enough data\n        \n        # Analyze request timing patterns\n        recent_intervals = []\n        for i in range(1, min(50, len(profile.request_history))):\n            interval = profile.request_history[i] - profile.request_history[i-1]\n            recent_intervals.append(interval)\n        \n        if recent_intervals:\n            timing_variance = statistics.stdev(recent_intervals) if len(recent_intervals) > 1 else 0\n            avg_interval = statistics.mean(recent_intervals)\n            \n            # Bot indicators based on timing\n            if timing_variance < 0.1 and avg_interval < 1.0:  # Very consistent, fast requests\n                analysis[\"bot_indicators\"].append(\"consistent_timing\")\n                analysis[\"legitimacy_score\"] *= 0.7\n            \n            if avg_interval < 0.1:  # Extremely fast requests\n                analysis[\"bot_indicators\"].append(\"rapid_fire\")\n                analysis[\"legitimacy_score\"] *= 0.5\n        \n        # Analyze endpoint usage patterns\n        total_requests = sum(profile.endpoint_usage.values())\n        if total_requests > 100:\n            # Check for unusual endpoint concentration\n            max_endpoint_usage = max(profile.endpoint_usage.values())\n            if max_endpoint_usage / total_requests > 0.9:  # 90% to single endpoint\n                analysis[\"suspicious_patterns\"].append(\"endpoint_concentration\")\n                analysis[\"legitimacy_score\"] *= 0.8\n        \n        # User agent analysis\n        if profile.user_agent:\n            if len(profile.user_agent) < 10 or \"bot\" in profile.user_agent.lower():\n                analysis[\"bot_indicators\"].append(\"suspicious_user_agent\")\n                analysis[\"legitimacy_score\"] *= 0.6\n        \n        # Update profile\n        profile.bot_probability = 1.0 - analysis[\"legitimacy_score\"]\n        profile.legitimate_score = analysis[\"legitimacy_score\"]\n        \n        return analysis\n    \n    def _assess_threat_level(self, profile: ClientProfile, behavioral_analysis: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Assess overall threat level for client\"\"\"\n        \n        threat_score = 0\n        factors = []\n        \n        # Behavioral factors\n        if profile.bot_probability > 0.8:\n            threat_score += 40\n            factors.append(\"high_bot_probability\")\n        elif profile.bot_probability > 0.5:\n            threat_score += 20\n            factors.append(\"medium_bot_probability\")\n        \n        # Request volume factors\n        if len(profile.request_history) > 500:  # High volume client\n            threat_score += 15\n            factors.append(\"high_volume\")\n        \n        # Reputation factors\n        if profile.reputation_score < 30:\n            threat_score += 30\n            factors.append(\"low_reputation\")\n        elif profile.reputation_score < 60:\n            threat_score += 15\n            factors.append(\"medium_reputation\")\n        \n        # Failed attempts\n        if profile.consecutive_failures > 10:\n            threat_score += 25\n            factors.append(\"repeated_failures\")\n        \n        # Determine threat level\n        if threat_score >= 70:\n            level = ThreatLevel.CRITICAL\n        elif threat_score >= 50:\n            level = ThreatLevel.HIGH\n        elif threat_score >= 25:\n            level = ThreatLevel.MEDIUM\n        else:\n            level = ThreatLevel.LOW\n        \n        profile.threat_level = level\n        \n        return {\n            \"threat_level\": level,\n            \"threat_score\": threat_score,\n            \"factors\": factors\n        }\n    \n    def _apply_adaptive_adjustments(self, config: RateLimitConfig, profile: ClientProfile, context: str) -> RateLimitConfig:\n        \"\"\"Apply adaptive adjustments to rate limit configuration\"\"\"\n        \n        # Create a copy to avoid modifying the original\n        adjusted_config = RateLimitConfig(\n            requests_per_second=config.requests_per_second,\n            requests_per_minute=config.requests_per_minute,\n            requests_per_hour=config.requests_per_hour,\n            requests_per_day=config.requests_per_day,\n            burst_capacity=config.burst_capacity,\n            burst_refill_rate=config.burst_refill_rate,\n            algorithm=config.algorithm\n        )\n        \n        # Apply threat level multipliers\n        multiplier = self.threat_level_multipliers.get(profile.threat_level, 1.0)\n        \n        adjusted_config.requests_per_minute = int(adjusted_config.requests_per_minute * multiplier)\n        adjusted_config.requests_per_hour = int(adjusted_config.requests_per_hour * multiplier)\n        adjusted_config.burst_capacity = int(adjusted_config.burst_capacity * multiplier)\n        \n        # Apply reputation adjustments\n        reputation_multiplier = max(0.1, profile.reputation_score / 100.0)\n        if profile.reputation_score > 80:  # High reputation gets bonus\n            reputation_multiplier = min(2.0, reputation_multiplier * 1.5)\n        \n        adjusted_config.requests_per_minute = int(adjusted_config.requests_per_minute * reputation_multiplier)\n        \n        return adjusted_config\n    \n    def _generate_client_id(self, ip: str, user_agent: str, user_id: Optional[str], api_key: Optional[str]) -> str:\n        \"\"\"Generate unique client identifier\"\"\"\n        \n        if user_id:\n            return f\"user:{user_id}\"\n        elif api_key:\n            return f\"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}\"\n        else:\n            # Combine IP and user agent hash\n            ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]\n            return f\"ip:{ip}:ua:{ua_hash}\"\n    \n    def _get_client_ip(self, request: Request) -> str:\n        \"\"\"Extract client IP address with proxy support\"\"\"\n        \n        # Check for forwarded headers\n        forwarded_for = request.headers.get(\"x-forwarded-for\")\n        if forwarded_for:\n            return forwarded_for.split(\",\")[0].strip()\n        \n        real_ip = request.headers.get(\"x-real-ip\")\n        if real_ip:\n            return real_ip\n        \n        return request.client.host if request.client else \"unknown\"\n    \n    def _is_ip_blocked(self, ip_address: str) -> bool:\n        \"\"\"Check if IP is in blocked ranges\"\"\"\n        try:\n            ip = ip_address(ip_address)\n            for blocked_range in self.ip_blacklist:\n                if ip in ip_network(blocked_range):\n                    return True\n        except ValueError:\n            return False\n        return False\n    \n    def _get_country_from_ip(self, ip_address: str) -> Optional[str]:\n        \"\"\"Get country code from IP address (placeholder)\"\"\"\n        # This would integrate with a GeoIP service\n        return None\n    \n    def _is_under_attack(self) -> bool:\n        \"\"\"Determine if system is currently under attack\"\"\"\n        current_rps = self.global_metrics[\"current_rps\"]\n        blocked_percentage = (self.global_metrics[\"blocked_requests\"] / \n                             max(1, self.global_metrics[\"total_requests\"])) * 100\n        \n        return current_rps > 1000 or blocked_percentage > 20\n    \n    def _get_system_load(self) -> float:\n        \"\"\"Get current system load (0.0 to 1.0)\"\"\"\n        # This would integrate with system monitoring\n        # For now, use request rate as proxy\n        current_rps = self.global_metrics[\"current_rps\"]\n        max_capacity = 5000  # Adjust based on system capacity\n        return min(1.0, current_rps / max_capacity)\n    \n    def _update_request_metrics(self, allowed: bool):\n        \"\"\"Update global request metrics\"\"\"\n        self.global_metrics[\"total_requests\"] += 1\n        if not allowed:\n            self.global_metrics[\"blocked_requests\"] += 1\n    \n    async def _log_rate_limit_event(self, profile: ClientProfile, rate_result: Dict, threat_assessment: Dict):\n        \"\"\"Log rate limiting events for monitoring\"\"\"\n        \n        if not rate_result[\"allowed\"]:\n            event_type = SECURITY_EVENT_TYPES[\"RATE_LIMIT_EXCEEDED\"]\n            level = SecurityEventLevel.WARNING\n        else:\n            event_type = SECURITY_EVENT_TYPES[\"SUSPICIOUS_ACTIVITY\"]\n            level = SecurityEventLevel.INFO\n        \n        security_logger.log_security_event(\n            event_type=event_type,\n            level=level,\n            client_ip=profile.ip_address,\n            details={\n                \"client_id\": profile.client_id,\n                \"threat_level\": threat_assessment[\"threat_level\"],\n                \"threat_score\": threat_assessment[\"threat_score\"],\n                \"bot_probability\": profile.bot_probability,\n                \"reputation_score\": profile.reputation_score,\n                \"allowed\": rate_result[\"allowed\"],\n                \"retry_after\": rate_result.get(\"retry_after\")\n            }\n        )\n    \n    async def _cleanup_expired_data(self):\n        \"\"\"Background task to clean up expired client data\"\"\"\n        while True:\n            try:\n                current_time = time.time()\n                cutoff_time = current_time - 3600  # Keep data for 1 hour\n                \n                # Clean up inactive clients\n                inactive_clients = []\n                for client_id, profile in self.client_profiles.items():\n                    if (profile.request_history and \n                        profile.request_history[-1] < cutoff_time):\n                        inactive_clients.append(client_id)\n                \n                for client_id in inactive_clients[:100]:  # Limit cleanup batch\n                    del self.client_profiles[client_id]\n                \n                logger.info(f\"Cleaned up {len(inactive_clients)} inactive client profiles\")\n                \n                await asyncio.sleep(300)  # Run every 5 minutes\n            except Exception as e:\n                logger.error(f\"Error in cleanup task: {e}\")\n                await asyncio.sleep(60)\n    \n    async def _update_global_metrics(self):\n        \"\"\"Update global system metrics\"\"\"\n        request_times = deque(maxlen=60)  # Last 60 seconds\n        \n        while True:\n            try:\n                current_time = time.time()\n                \n                # Count recent requests\n                recent_requests = 0\n                for profile in self.client_profiles.values():\n                    recent_requests += len([\n                        t for t in profile.request_history\n                        if t > current_time - 60\n                    ])\n                \n                self.global_metrics[\"current_rps\"] = recent_requests / 60.0\n                self.global_metrics[\"peak_rps\"] = max(\n                    self.global_metrics[\"peak_rps\"],\n                    self.global_metrics[\"current_rps\"]\n                )\n                self.global_metrics[\"active_clients\"] = len(self.client_profiles)\n                \n                await asyncio.sleep(10)  # Update every 10 seconds\n            except Exception as e:\n                logger.error(f\"Error updating metrics: {e}\")\n                await asyncio.sleep(30)\n    \n    async def _analyze_attack_patterns(self):\n        \"\"\"Analyze patterns to detect coordinated attacks\"\"\"\n        while True:\n            try:\n                current_time = time.time()\n                \n                # Analyze for volumetric attacks\n                high_volume_ips = []\n                for profile in self.client_profiles.values():\n                    recent_requests = len([\n                        t for t in profile.request_history\n                        if t > current_time - 300  # Last 5 minutes\n                    ])\n                    if recent_requests > 1000:  # Threshold for concern\n                        high_volume_ips.append(profile.ip_address)\n                \n                if len(high_volume_ips) > 10:  # Multiple high-volume IPs\n                    logger.warning(f\"Potential distributed attack detected from {len(high_volume_ips)} IPs\")\n                    # Implement additional countermeasures\n                \n                await asyncio.sleep(60)  # Analyze every minute\n            except Exception as e:\n                logger.error(f\"Error in attack pattern analysis: {e}\")\n                await asyncio.sleep(60)\n    \n    async def _update_adaptive_baselines(self):\n        \"\"\"Update adaptive baselines for rate limiting\"\"\"\n        while True:\n            try:\n                # Update baselines based on historical data\n                # This would implement more sophisticated ML-based baseline updates\n                await asyncio.sleep(900)  # Update every 15 minutes\n            except Exception as e:\n                logger.error(f\"Error updating baselines: {e}\")\n                await asyncio.sleep(300)\n    \n    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Get comprehensive client information for debugging\"\"\"\n        profile = self.client_profiles.get(client_id)\n        if not profile:\n            return None\n        \n        return {\n            \"client_id\": profile.client_id,\n            \"ip_address\": profile.ip_address,\n            \"threat_level\": profile.threat_level.value,\n            \"reputation_score\": profile.reputation_score,\n            \"bot_probability\": profile.bot_probability,\n            \"total_requests\": len(profile.request_history),\n            \"recent_requests_1min\": len([\n                t for t in profile.request_history\n                if t > time.time() - 60\n            ]),\n            \"endpoint_usage\": dict(profile.endpoint_usage),\n            \"tokens_remaining\": profile.tokens,\n            \"circuit_open\": profile.circuit_open\n        }\n    \n    def get_global_stats(self) -> Dict[str, Any]:\n        \"\"\"Get global rate limiting statistics\"\"\"\n        return {\n            \"global_metrics\": self.global_metrics.copy(),\n            \"active_clients\": len(self.client_profiles),\n            \"suspicious_ips\": len(self.suspicious_ips),\n            \"known_good_ips\": len(self.known_good_ips),\n            \"system_load\": self._get_system_load(),\n            \"under_attack\": self._is_under_attack()\n        }\n\n\n# Global instance\nadvanced_rate_limiter = AdvancedRateLimiter()\n\n\nclass RateLimitMiddleware(BaseHTTPMiddleware):\n    \"\"\"FastAPI middleware for advanced rate limiting\"\"\"\n    \n    def __init__(self, app, rate_limiter: AdvancedRateLimiter = None):\n        super().__init__(app)\n        self.rate_limiter = rate_limiter or advanced_rate_limiter\n    \n    async def dispatch(self, request: Request, call_next):\n        # Determine context based on endpoint\n        context = \"default\"\n        if \"/api/\" in request.url.path:\n            context = \"api\"\n        elif \"/auth/\" in request.url.path:\n            context = \"auth\"\n        elif request.url.path.startswith(\"/upload\"):\n            context = \"upload\"\n        \n        # Check rate limits\n        allowed, info = await self.rate_limiter.check_rate_limit(request, context)\n        \n        if not allowed:\n            # Create rate limit response\n            response = Response(\n                content=json.dumps({\n                    \"error\": \"Rate limit exceeded\",\n                    \"message\": f\"Too many requests. Try again in {info.get('retry_after', 60)} seconds.\",\n                    \"details\": {\n                        \"limit\": info.get(\"limit\"),\n                        \"remaining\": 0,\n                        \"reset_time\": info.get(\"reset_time\"),\n                        \"threat_level\": info.get(\"threat_level\")\n                    }\n                }),\n                status_code=status.HTTP_429_TOO_MANY_REQUESTS,\n                media_type=\"application/json\",\n                headers={\n                    \"X-RateLimit-Limit\": str(info.get(\"limit\", \"unknown\")),\n                    \"X-RateLimit-Remaining\": \"0\",\n                    \"X-RateLimit-Reset\": str(info.get(\"reset_time\", 0)),\n                    \"Retry-After\": str(info.get(\"retry_after\", 60)),\n                    \"X-RateLimit-Algorithm\": info.get(\"rate_limit_algorithm\", \"adaptive\")\n                }\n            )\n            return response\n        \n        # Add rate limit headers to response\n        response = await call_next(request)\n        response.headers[\"X-RateLimit-Limit\"] = str(info.get(\"limit\", \"unknown\"))\n        response.headers[\"X-RateLimit-Remaining\"] = str(info.get(\"remaining\", \"unknown\"))\n        response.headers[\"X-RateLimit-Reset\"] = str(info.get(\"reset_time\", 0))\n        \n        return response"}}]