"""
SuperNova AI Advanced Authentication and Session Security
Enhanced security features including device fingerprinting, behavioral analysis,
session management, and advanced MFA
"""

import os
import secrets
import hashlib
import hmac
import base64
import json
import pyotp
import qrcode
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Set, Tuple
from enum import Enum
import jwt
from passlib.context import CryptContext
from passlib.hash import argon2, bcrypt
import redis
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer
import logging
from dataclasses import dataclass, asdict
from collections import deque
import time
import geoip2.database
import user_agents

from .security_config import security_settings, SECURITY_EVENT_TYPES
from .security_logger import security_logger, SecurityEventLevel
from .injection_prevention import injection_prevention

logger = logging.getLogger(__name__)


class AuthenticationMethod(str, Enum):
    """Authentication methods supported"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    OAUTH = "oauth"


class SessionStatus(str, Enum):
    """Session status values"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPICIOUS = "suspicious"
    LOCKED = "locked"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DeviceFingerprint:
    """Device fingerprint data structure"""
    user_agent: str
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    platform: Optional[str] = None
    plugins: Optional[List[str]] = None
    canvas_fingerprint: Optional[str] = None
    webgl_fingerprint: Optional[str] = None
    audio_fingerprint: Optional[str] = None
    
    def generate_hash(self) -> str:
        """Generate hash from fingerprint data"""
        fingerprint_data = asdict(self)
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()


@dataclass
class SessionInfo:
    """Comprehensive session information"""
    session_id: str
    user_id: str
    user_role: str
    ip_address: str
    device_fingerprint: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus
    location: Optional[Dict[str, str]] = None
    risk_level: RiskLevel = RiskLevel.LOW
    mfa_verified: bool = False
    trusted_device: bool = False
    concurrent_sessions: int = 0
    failed_attempts: int = 0
    
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE and not self.is_expired()
    
    def time_since_last_activity(self) -> timedelta:
        return datetime.now(timezone.utc) - self.last_activity


class AdvancedAuthenticationManager:
    """
    Advanced authentication manager with enhanced security features
    """
    
    def __init__(self):
        # Password context with multiple secure algorithms
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__rounds=12,
            bcrypt__rounds=12
        )
        
        # Redis client for session management
        self.redis_client = self._get_redis_client()
        
        # In-memory stores (for development - use Redis in production)
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.failed_login_attempts: Dict[str, deque] = {}
        self.suspicious_activities: Dict[str, List[Dict]] = {}
        self.trusted_devices: Dict[str, Set[str]] = {}  # user_id -> device_fingerprints
        self.rate_limit_cache: Dict[str, deque] = {}
        
        # Security policies
        self.max_failed_attempts = security_settings.LOGIN_MAX_ATTEMPTS
        self.lockout_duration = timedelta(minutes=security_settings.LOCKOUT_DURATION_MINUTES)
        self.session_timeout = timedelta(minutes=security_settings.SESSION_TIMEOUT_MINUTES)
        self.max_concurrent_sessions = security_settings.CONCURRENT_SESSIONS_LIMIT
        
        # Load GeoIP database if available
        self.geoip_reader = self._load_geoip_database()
        
        # Device trust settings
        self.device_trust_duration = timedelta(days=30)
        self.require_mfa_for_untrusted = True
        
    def _get_redis_client(self):
        """Initialize Redis client for session storage"""
        try:
            import redis
            return redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 1)),
                decode_responses=True
            )
        except ImportError:
            logger.warning("Redis not available, using in-memory storage")
            return None
    
    def _load_geoip_database(self):
        """Load GeoIP database for location tracking"""
        try:
            import geoip2.database
            geoip_path = os.getenv('GEOIP_DATABASE_PATH', 'GeoLite2-City.mmdb')
            if os.path.exists(geoip_path):
                return geoip2.database.Reader(geoip_path)
        except (ImportError, FileNotFoundError):
            logger.warning("GeoIP database not available")
        return None
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        request: Request,
        device_fingerprint: Optional[DeviceFingerprint] = None,
        mfa_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive user authentication with security analysis
        """
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Check for injection attempts in credentials
        credential_check = injection_prevention.comprehensive_injection_detection(
            f"{username}:{password}", 
            context={"field_type": "credentials"},
            client_ip=client_ip
        )
        
        if not credential_check['safe']:
            await self._log_security_event(
                "AUTHENTICATION_FAILURE",
                SecurityEventLevel.WARNING,
                client_ip,
                {"reason": "injection_attempt_in_credentials"}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authentication data"
            )
        
        # Rate limiting check
        if not self._check_rate_limit(client_ip, "login"):
            await self._log_security_event(
                "AUTHENTICATION_FAILURE",
                SecurityEventLevel.WARNING,
                client_ip,
                {"reason": "rate_limit_exceeded"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts"
            )
        
        # Check for account lockout
        if self._is_account_locked(username, client_ip):
            await self._log_security_event(
                "AUTHENTICATION_FAILURE",
                SecurityEventLevel.WARNING,
                client_ip,
                {"reason": "account_locked", "username": username}
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to failed attempts"
            )
        
        try:
            # Simulate user lookup (replace with actual database lookup)
            user = await self._get_user(username)
            if not user:
                self._record_failed_attempt(username, client_ip)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Verify password
            if not self.pwd_context.verify(password, user.get('password_hash', '')):
                self._record_failed_attempt(username, client_ip)
                await self._log_security_event(
                    "AUTHENTICATION_FAILURE",
                    SecurityEventLevel.WARNING,
                    client_ip,
                    {"reason": "invalid_password", "username": username}
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Generate device fingerprint hash
            device_hash = device_fingerprint.generate_hash() if device_fingerprint else ""
            
            # Risk assessment
            risk_assessment = await self._assess_authentication_risk(
                user, client_ip, user_agent, device_hash
            )
            
            # Check if MFA is required
            mfa_required = self._is_mfa_required(user, risk_assessment, device_hash)
            
            if mfa_required and not mfa_code:
                # Return MFA challenge
                challenge_token = self._generate_mfa_challenge_token(user['id'], client_ip)
                return {
                    "status": "mfa_required",
                    "challenge_token": challenge_token,
                    "mfa_methods": user.get('mfa_methods', ['totp']),
                    "risk_level": risk_assessment['level']
                }
            
            # Verify MFA if provided
            if mfa_code and not await self._verify_mfa(user, mfa_code, mfa_required):
                self._record_failed_attempt(username, client_ip)
                await self._log_security_event(
                    "AUTHENTICATION_FAILURE",
                    SecurityEventLevel.WARNING,
                    client_ip,
                    {"reason": "invalid_mfa", "username": username}
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid MFA code"
                )
            
            # Check concurrent session limits
            await self._enforce_session_limits(user['id'])
            
            # Create session
            session_info = await self._create_session(
                user, client_ip, user_agent, device_hash, risk_assessment
            )
            
            # Generate tokens
            access_token, refresh_token = self._generate_tokens(user, session_info)
            
            # Update device trust if applicable
            if risk_assessment['level'] in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                self._update_device_trust(user['id'], device_hash)
            
            # Clear failed attempts
            self._clear_failed_attempts(username, client_ip)
            
            # Log successful authentication
            await self._log_security_event(
                "AUTHENTICATION_SUCCESS",
                SecurityEventLevel.INFO,
                client_ip,
                {
                    "username": username,
                    "user_id": user['id'],
                    "risk_level": risk_assessment['level'],
                    "mfa_used": bool(mfa_code),
                    "trusted_device": session_info.trusted_device
                }
            )
            
            return {
                "status": "authenticated",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": security_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "session_id": session_info.session_id,
                "user": {
                    "id": user['id'],
                    "username": user['username'],
                    "role": user['role'],
                    "permissions": self._get_user_permissions(user['role'])
                },
                "security_info": {
                    "risk_level": risk_assessment['level'],
                    "trusted_device": session_info.trusted_device,
                    "location": session_info.location,
                    "requires_password_change": user.get('requires_password_change', False)
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self._log_security_event(
                "AUTHENTICATION_ERROR",
                SecurityEventLevel.ERROR,
                client_ip,
                {"error": str(e), "username": username}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    async def _assess_authentication_risk(
        self,
        user: Dict[str, Any],
        client_ip: str,
        user_agent: str,
        device_hash: str
    ) -> Dict[str, Any]:
        """Comprehensive risk assessment for authentication"""
        
        risk_factors = []
        risk_score = 0
        
        # Check IP reputation and geolocation
        location = self._get_location(client_ip)
        if location:
            # Check if login from unusual location
            user_locations = user.get('recent_locations', [])
            if location not in user_locations[-5:]:  # Last 5 locations
                risk_factors.append("unusual_location")
                risk_score += 25
        
        # Check device fingerprint
        user_devices = self.trusted_devices.get(user['id'], set())
        if device_hash not in user_devices:
            risk_factors.append("unknown_device")
            risk_score += 20
        
        # Check user agent analysis
        try:
            ua = user_agents.parse(user_agent)
            if ua.is_bot:
                risk_factors.append("bot_user_agent")
                risk_score += 40
        except:
            risk_factors.append("invalid_user_agent")
            risk_score += 15
        
        # Check time-based patterns
        current_hour = datetime.now().hour
        user_login_hours = user.get('typical_login_hours', [])\n        if user_login_hours and current_hour not in user_login_hours:\n            risk_factors.append("unusual_time")\n            risk_score += 10\n        \n        # Check for suspicious IP\n        if injection_prevention.is_suspicious_ip(client_ip):\n            risk_factors.append("suspicious_ip")\n            risk_score += 30\n        \n        # Check recent failed attempts\n        recent_failures = len([\n            attempt for attempt in self.failed_login_attempts.get(client_ip, [])\n            if time.time() - attempt < 3600  # Last hour\n        ])\n        if recent_failures > 3:\n            risk_factors.append("recent_failures")\n            risk_score += 20\n        \n        # Determine risk level\n        if risk_score >= 60:\n            level = RiskLevel.CRITICAL\n        elif risk_score >= 40:\n            level = RiskLevel.HIGH\n        elif risk_score >= 20:\n            level = RiskLevel.MEDIUM\n        else:\n            level = RiskLevel.LOW\n        \n        return {\n            'level': level,\n            'score': risk_score,\n            'factors': risk_factors,\n            'location': location\n        }\n    \n    def _is_mfa_required(\n        self,\n        user: Dict[str, Any],\n        risk_assessment: Dict[str, Any],\n        device_hash: str\n    ) -> bool:\n        \"\"\"Determine if MFA is required based on risk and policy\"\"\"\n        \n        # Always require MFA for admin users\n        if user.get('role') == 'admin':\n            return True\n        \n        # Require MFA for high/critical risk\n        if risk_assessment['level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:\n            return True\n        \n        # Require MFA for untrusted devices if policy enabled\n        if self.require_mfa_for_untrusted:\n            user_devices = self.trusted_devices.get(user['id'], set())\n            if device_hash not in user_devices:\n                return True\n        \n        # Check user MFA settings\n        return user.get('mfa_enabled', False)\n    \n    async def _verify_mfa(\n        self,\n        user: Dict[str, Any],\n        mfa_code: str,\n        required: bool = True\n    ) -> bool:\n        \"\"\"Verify MFA code\"\"\"\n        \n        if not required and not mfa_code:\n            return True\n        \n        # Get user's MFA secret (in production, this would be encrypted)\n        mfa_secret = user.get('mfa_secret')\n        if not mfa_secret:\n            return False\n        \n        # Verify TOTP code\n        totp = pyotp.TOTP(mfa_secret)\n        \n        # Allow for time drift\n        for drift in [-1, 0, 1]:\n            if totp.verify(mfa_code, valid_window=drift):\n                return True\n        \n        return False\n    \n    async def _create_session(\n        self,\n        user: Dict[str, Any],\n        client_ip: str,\n        user_agent: str,\n        device_hash: str,\n        risk_assessment: Dict[str, Any]\n    ) -> SessionInfo:\n        \"\"\"Create secure session with comprehensive tracking\"\"\"\n        \n        session_id = self._generate_secure_session_id()\n        current_time = datetime.now(timezone.utc)\n        \n        # Determine session expiry based on risk\n        if risk_assessment['level'] == RiskLevel.CRITICAL:\n            expires_at = current_time + timedelta(minutes=15)  # Short session for high risk\n        elif risk_assessment['level'] == RiskLevel.HIGH:\n            expires_at = current_time + timedelta(minutes=30)\n        else:\n            expires_at = current_time + self.session_timeout\n        \n        # Check if device is trusted\n        trusted_device = device_hash in self.trusted_devices.get(user['id'], set())\n        \n        # Count concurrent sessions\n        concurrent_sessions = len([\n            s for s in self.active_sessions.values()\n            if s.user_id == user['id'] and s.is_active()\n        ])\n        \n        session_info = SessionInfo(\n            session_id=session_id,\n            user_id=user['id'],\n            user_role=user['role'],\n            ip_address=client_ip,\n            device_fingerprint=device_hash,\n            user_agent=user_agent,\n            created_at=current_time,\n            last_activity=current_time,\n            expires_at=expires_at,\n            status=SessionStatus.ACTIVE,\n            location=risk_assessment.get('location'),\n            risk_level=risk_assessment['level'],\n            mfa_verified=True,  # MFA was verified to reach this point\n            trusted_device=trusted_device,\n            concurrent_sessions=concurrent_sessions\n        )\n        \n        # Store session\n        self.active_sessions[session_id] = session_info\n        \n        # Store in Redis if available\n        if self.redis_client:\n            session_data = asdict(session_info)\n            # Convert datetime objects to strings for JSON serialization\n            for key, value in session_data.items():\n                if isinstance(value, datetime):\n                    session_data[key] = value.isoformat()\n                elif isinstance(value, Enum):\n                    session_data[key] = value.value\n            \n            self.redis_client.setex(\n                f\"session:{session_id}\",\n                int(self.session_timeout.total_seconds()),\n                json.dumps(session_data)\n            )\n        \n        return session_info\n    \n    def _generate_tokens(self, user: Dict[str, Any], session_info: SessionInfo) -> Tuple[str, str]:\n        \"\"\"Generate JWT access and refresh tokens\"\"\"\n        \n        current_time = datetime.now(timezone.utc)\n        \n        # Access token payload\n        access_payload = {\n            \"sub\": user['id'],\n            \"username\": user['username'],\n            \"role\": user['role'],\n            \"session_id\": session_info.session_id,\n            \"iat\": current_time,\n            \"exp\": current_time + timedelta(minutes=security_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),\n            \"aud\": security_settings.JWT_AUDIENCE,\n            \"iss\": security_settings.JWT_ISSUER,\n            \"device_hash\": session_info.device_fingerprint[:16],  # Partial hash for tracking\n            \"risk_level\": session_info.risk_level.value,\n            \"mfa_verified\": session_info.mfa_verified\n        }\n        \n        # Refresh token payload (longer lived, less information)\n        refresh_payload = {\n            \"sub\": user['id'],\n            \"session_id\": session_info.session_id,\n            \"iat\": current_time,\n            \"exp\": current_time + timedelta(days=security_settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),\n            \"aud\": security_settings.JWT_AUDIENCE,\n            \"iss\": security_settings.JWT_ISSUER,\n            \"token_type\": \"refresh\"\n        }\n        \n        # Generate tokens\n        access_token = jwt.encode(\n            access_payload,\n            security_settings.JWT_SECRET_KEY,\n            algorithm=security_settings.JWT_ALGORITHM\n        )\n        \n        refresh_token = jwt.encode(\n            refresh_payload,\n            security_settings.JWT_SECRET_KEY,\n            algorithm=security_settings.JWT_ALGORITHM\n        )\n        \n        return access_token, refresh_token\n    \n    async def refresh_token(self, refresh_token: str, request: Request) -> Dict[str, Any]:\n        \"\"\"Refresh access token with security validation\"\"\"\n        \n        try:\n            # Decode refresh token\n            payload = jwt.decode(\n                refresh_token,\n                security_settings.JWT_SECRET_KEY,\n                algorithms=[security_settings.JWT_ALGORITHM],\n                audience=security_settings.JWT_AUDIENCE,\n                issuer=security_settings.JWT_ISSUER\n            )\n            \n            # Validate token type\n            if payload.get('token_type') != 'refresh':\n                raise HTTPException(\n                    status_code=status.HTTP_401_UNAUTHORIZED,\n                    detail=\"Invalid token type\"\n                )\n            \n            # Get session info\n            session_id = payload.get('session_id')\n            session_info = self.active_sessions.get(session_id)\n            \n            if not session_info or not session_info.is_active():\n                raise HTTPException(\n                    status_code=status.HTTP_401_UNAUTHORIZED,\n                    detail=\"Session expired or invalid\"\n                )\n            \n            # Security checks\n            client_ip = self._get_client_ip(request)\n            if client_ip != session_info.ip_address:\n                # IP changed - require re-authentication for high security\n                await self._log_security_event(\n                    \"SUSPICIOUS_ACTIVITY\",\n                    SecurityEventLevel.WARNING,\n                    client_ip,\n                    {\n                        \"reason\": \"ip_change_on_token_refresh\",\n                        \"original_ip\": session_info.ip_address,\n                        \"new_ip\": client_ip,\n                        \"user_id\": session_info.user_id\n                    }\n                )\n                \n                if session_info.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:\n                    await self.terminate_session(session_id)\n                    raise HTTPException(\n                        status_code=status.HTTP_401_UNAUTHORIZED,\n                        detail=\"Re-authentication required\"\n                    )\n            \n            # Update session activity\n            session_info.last_activity = datetime.now(timezone.utc)\n            \n            # Get user info\n            user = await self._get_user_by_id(session_info.user_id)\n            if not user:\n                raise HTTPException(\n                    status_code=status.HTTP_401_UNAUTHORIZED,\n                    detail=\"User not found\"\n                )\n            \n            # Generate new access token\n            new_access_token, _ = self._generate_tokens(user, session_info)\n            \n            return {\n                \"access_token\": new_access_token,\n                \"token_type\": \"bearer\",\n                \"expires_in\": security_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60\n            }\n            \n        except jwt.ExpiredSignatureError:\n            raise HTTPException(\n                status_code=status.HTTP_401_UNAUTHORIZED,\n                detail=\"Refresh token expired\"\n            )\n        except jwt.InvalidTokenError as e:\n            raise HTTPException(\n                status_code=status.HTTP_401_UNAUTHORIZED,\n                detail=\"Invalid refresh token\"\n            )\n    \n    async def terminate_session(self, session_id: str, reason: str = \"user_logout\") -> bool:\n        \"\"\"Terminate a session securely\"\"\"\n        \n        session_info = self.active_sessions.get(session_id)\n        if not session_info:\n            return False\n        \n        # Update session status\n        session_info.status = SessionStatus.TERMINATED\n        \n        # Remove from active sessions\n        del self.active_sessions[session_id]\n        \n        # Remove from Redis if available\n        if self.redis_client:\n            self.redis_client.delete(f\"session:{session_id}\")\n        \n        # Log session termination\n        await self._log_security_event(\n            \"SESSION_TERMINATED\",\n            SecurityEventLevel.INFO,\n            session_info.ip_address,\n            {\n                \"session_id\": session_id,\n                \"user_id\": session_info.user_id,\n                \"reason\": reason,\n                \"duration\": (datetime.now(timezone.utc) - session_info.created_at).total_seconds()\n            }\n        )\n        \n        return True\n    \n    def _generate_secure_session_id(self) -> str:\n        \"\"\"Generate cryptographically secure session ID\"\"\"\n        random_bytes = secrets.token_bytes(32)\n        timestamp = str(int(time.time()))\n        combined = random_bytes + timestamp.encode()\n        return hashlib.sha256(combined).hexdigest()\n    \n    def _get_client_ip(self, request: Request) -> str:\n        \"\"\"Extract client IP with proxy support\"\"\"\n        # Check for forwarded headers\n        forwarded_for = request.headers.get(\"x-forwarded-for\")\n        if forwarded_for:\n            return forwarded_for.split(\",\")[0].strip()\n        \n        real_ip = request.headers.get(\"x-real-ip\")\n        if real_ip:\n            return real_ip\n        \n        return request.client.host if request.client else \"unknown\"\n    \n    def _get_location(self, ip_address: str) -> Optional[Dict[str, str]]:\n        \"\"\"Get location from IP address\"\"\"\n        if not self.geoip_reader:\n            return None\n        \n        try:\n            response = self.geoip_reader.city(ip_address)\n            return {\n                \"country\": response.country.name,\n                \"city\": response.city.name,\n                \"region\": response.subdivisions.most_specific.name\n            }\n        except Exception:\n            return None\n    \n    def _check_rate_limit(self, identifier: str, action: str, limit: int = 10, window: int = 300) -> bool:\n        \"\"\"Check rate limiting for actions\"\"\"\n        key = f\"{identifier}:{action}\"\n        current_time = time.time()\n        \n        if key not in self.rate_limit_cache:\n            self.rate_limit_cache[key] = deque()\n        \n        # Clean old entries\n        while (self.rate_limit_cache[key] and \n               current_time - self.rate_limit_cache[key][0] > window):\n            self.rate_limit_cache[key].popleft()\n        \n        # Check limit\n        if len(self.rate_limit_cache[key]) >= limit:\n            return False\n        \n        # Add current request\n        self.rate_limit_cache[key].append(current_time)\n        return True\n    \n    def _is_account_locked(self, username: str, ip_address: str) -> bool:\n        \"\"\"Check if account is locked due to failed attempts\"\"\"\n        key = f\"{username}:{ip_address}\"\n        if key not in self.failed_login_attempts:\n            return False\n        \n        current_time = time.time()\n        recent_failures = [\n            attempt for attempt in self.failed_login_attempts[key]\n            if current_time - attempt < self.lockout_duration.total_seconds()\n        ]\n        \n        return len(recent_failures) >= self.max_failed_attempts\n    \n    def _record_failed_attempt(self, username: str, ip_address: str):\n        \"\"\"Record failed login attempt\"\"\"\n        key = f\"{username}:{ip_address}\"\n        if key not in self.failed_login_attempts:\n            self.failed_login_attempts[key] = deque(maxlen=20)\n        \n        self.failed_login_attempts[key].append(time.time())\n    \n    def _clear_failed_attempts(self, username: str, ip_address: str):\n        \"\"\"Clear failed login attempts after successful login\"\"\"\n        key = f\"{username}:{ip_address}\"\n        if key in self.failed_login_attempts:\n            del self.failed_login_attempts[key]\n    \n    def _update_device_trust(self, user_id: str, device_hash: str):\n        \"\"\"Update device trust for user\"\"\"\n        if user_id not in self.trusted_devices:\n            self.trusted_devices[user_id] = set()\n        \n        self.trusted_devices[user_id].add(device_hash)\n    \n    async def _log_security_event(self, event_type: str, level: SecurityEventLevel, client_ip: str, details: Dict):\n        \"\"\"Log security events\"\"\"\n        security_logger.log_security_event(\n            event_type=event_type,\n            level=level,\n            client_ip=client_ip,\n            details=details\n        )\n    \n    async def _get_user(self, username: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Get user by username (placeholder - implement with actual database)\"\"\"\n        # This is a placeholder - implement with actual database lookup\n        return {\n            \"id\": \"user123\",\n            \"username\": username,\n            \"password_hash\": self.pwd_context.hash(\"testpassword\"),\n            \"role\": \"user\",\n            \"mfa_enabled\": True,\n            \"mfa_secret\": pyotp.random_base32()\n        }\n    \n    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Get user by ID (placeholder - implement with actual database)\"\"\"\n        # Placeholder implementation\n        return await self._get_user(\"testuser\")\n    \n    def _get_user_permissions(self, role: str) -> List[str]:\n        \"\"\"Get permissions for user role\"\"\"\n        # Simplified permission mapping\n        role_permissions = {\n            \"admin\": [\"read\", \"write\", \"delete\", \"admin\"],\n            \"user\": [\"read\", \"write\"],\n            \"viewer\": [\"read\"]\n        }\n        return role_permissions.get(role, [])\n    \n    def _generate_mfa_challenge_token(self, user_id: str, client_ip: str) -> str:\n        \"\"\"Generate temporary MFA challenge token\"\"\"\n        payload = {\n            \"sub\": user_id,\n            \"purpose\": \"mfa_challenge\",\n            \"ip\": client_ip,\n            \"iat\": datetime.now(timezone.utc),\n            \"exp\": datetime.now(timezone.utc) + timedelta(minutes=5)\n        }\n        return jwt.encode(payload, security_settings.JWT_SECRET_KEY, algorithm=security_settings.JWT_ALGORITHM)\n    \n    async def _enforce_session_limits(self, user_id: str):\n        \"\"\"Enforce concurrent session limits\"\"\"\n        active_user_sessions = [\n            s for s in self.active_sessions.values()\n            if s.user_id == user_id and s.is_active()\n        ]\n        \n        if len(active_user_sessions) >= self.max_concurrent_sessions:\n            # Terminate oldest session\n            oldest_session = min(active_user_sessions, key=lambda s: s.created_at)\n            await self.terminate_session(oldest_session.session_id, \"concurrent_limit\")\n\n\n# Global instance\nadvanced_auth_manager = AdvancedAuthenticationManager()"}}]