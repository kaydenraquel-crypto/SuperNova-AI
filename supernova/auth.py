"""
SuperNova AI Authentication and Authorization System
Comprehensive JWT-based authentication with RBAC and MFA support
"""

import os
import secrets
import hashlib
import base64
import pyotp
import qrcode
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import redis
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from .security_config import security_settings, SECURITY_EVENT_TYPES
from .db import SessionLocal, User
from .security_logger import SecurityLogger

logger = logging.getLogger(__name__)
security_logger = SecurityLogger()

# Password context for hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token blacklist (in production, use Redis)
token_blacklist: Set[str] = set()

# Redis client for session management
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 1)),
    decode_responses=True
) if os.getenv('REDIS_URL') else None


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    USER = "user"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions"""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    
    # Profile management
    CREATE_PROFILE = "create_profile"
    READ_PROFILE = "read_profile"
    UPDATE_PROFILE = "update_profile"
    DELETE_PROFILE = "delete_profile"
    
    # Financial data
    READ_FINANCIAL_DATA = "read_financial_data"
    WRITE_FINANCIAL_DATA = "write_financial_data"
    EXPORT_FINANCIAL_DATA = "export_financial_data"
    
    # System administration
    ADMIN_DASHBOARD = "admin_dashboard"
    SYSTEM_CONFIG = "system_config"
    SECURITY_LOGS = "security_logs"
    USER_MANAGEMENT = "user_management"
    
    # API access
    API_READ = "api_read"
    API_WRITE = "api_write"
    API_ADMIN = "api_admin"
    
    # WebSocket access
    WEBSOCKET_CONNECT = "websocket_connect"
    WEBSOCKET_BROADCAST = "websocket_broadcast"
    
    # Collaboration permissions
    CREATE_TEAM = "create_team"
    MANAGE_TEAM = "manage_team"
    INVITE_MEMBERS = "invite_members"
    SHARE_PORTFOLIO = "share_portfolio"
    SHARE_STRATEGY = "share_strategy"
    ACCESS_SHARED_CONTENT = "access_shared_content"
    MANAGE_TEAM_SETTINGS = "manage_team_settings"
    MODERATE_TEAM_CHAT = "moderate_team_chat"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.CREATE_USER, Permission.READ_USER, Permission.UPDATE_USER, Permission.DELETE_USER,
        Permission.CREATE_PROFILE, Permission.READ_PROFILE, Permission.UPDATE_PROFILE, Permission.DELETE_PROFILE,
        Permission.READ_FINANCIAL_DATA, Permission.WRITE_FINANCIAL_DATA, Permission.EXPORT_FINANCIAL_DATA,
        Permission.ADMIN_DASHBOARD, Permission.SYSTEM_CONFIG, Permission.SECURITY_LOGS, Permission.USER_MANAGEMENT,
        Permission.API_READ, Permission.API_WRITE, Permission.API_ADMIN,
        Permission.WEBSOCKET_CONNECT, Permission.WEBSOCKET_BROADCAST,
        Permission.CREATE_TEAM, Permission.MANAGE_TEAM, Permission.INVITE_MEMBERS,
        Permission.SHARE_PORTFOLIO, Permission.SHARE_STRATEGY, Permission.ACCESS_SHARED_CONTENT,
        Permission.MANAGE_TEAM_SETTINGS, Permission.MODERATE_TEAM_CHAT
    ],
    UserRole.MANAGER: [
        Permission.READ_USER, Permission.UPDATE_USER,
        Permission.CREATE_PROFILE, Permission.READ_PROFILE, Permission.UPDATE_PROFILE,
        Permission.READ_FINANCIAL_DATA, Permission.WRITE_FINANCIAL_DATA, Permission.EXPORT_FINANCIAL_DATA,
        Permission.API_READ, Permission.API_WRITE,
        Permission.WEBSOCKET_CONNECT, Permission.WEBSOCKET_BROADCAST,
        Permission.CREATE_TEAM, Permission.MANAGE_TEAM, Permission.INVITE_MEMBERS,
        Permission.SHARE_PORTFOLIO, Permission.SHARE_STRATEGY, Permission.ACCESS_SHARED_CONTENT,
        Permission.MODERATE_TEAM_CHAT
    ],
    UserRole.ANALYST: [
        Permission.READ_USER,
        Permission.READ_PROFILE, Permission.UPDATE_PROFILE,
        Permission.READ_FINANCIAL_DATA, Permission.WRITE_FINANCIAL_DATA,
        Permission.API_READ, Permission.API_WRITE,
        Permission.WEBSOCKET_CONNECT,
        Permission.CREATE_TEAM, Permission.INVITE_MEMBERS,
        Permission.SHARE_PORTFOLIO, Permission.SHARE_STRATEGY, Permission.ACCESS_SHARED_CONTENT
    ],
    UserRole.USER: [
        Permission.READ_PROFILE, Permission.UPDATE_PROFILE,
        Permission.READ_FINANCIAL_DATA,
        Permission.API_READ,
        Permission.WEBSOCKET_CONNECT,
        Permission.CREATE_TEAM, Permission.INVITE_MEMBERS,
        Permission.SHARE_PORTFOLIO, Permission.SHARE_STRATEGY, Permission.ACCESS_SHARED_CONTENT
    ],
    UserRole.VIEWER: [
        Permission.READ_PROFILE,
        Permission.READ_FINANCIAL_DATA,
        Permission.API_READ
    ]
}


class AuthenticationError(Exception):
    """Custom authentication error"""
    pass


class AuthorizationError(Exception):
    """Custom authorization error"""
    pass


class MFARequiredError(Exception):
    """MFA verification required error"""
    pass


class TokenPayload:
    """JWT token payload structure"""
    def __init__(self, user_id: int, email: str, role: UserRole, 
                 permissions: List[str], session_id: str = None):
        self.user_id = user_id
        self.email = email
        self.role = role
        self.permissions = permissions
        self.session_id = session_id or secrets.token_urlsafe(16)


class AuthManager:
    """Comprehensive authentication and authorization manager"""
    
    def __init__(self):
        self.jwt_config = security_settings.get_jwt_config()
        self.failed_attempts: Dict[str, Dict] = {}
        self.active_sessions: Dict[str, Dict] = {}
    
    # =====================================
    # PASSWORD MANAGEMENT
    # =====================================
    
    def hash_password(self, password: str) -> str:
        """Hash a password with bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength according to security policy"""
        errors = []
        
        if len(password) < security_settings.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {security_settings.PASSWORD_MIN_LENGTH} characters long")
        
        if security_settings.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if security_settings.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if security_settings.PASSWORD_REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if security_settings.PASSWORD_REQUIRE_SPECIAL and not any(c in security_settings.PASSWORD_SPECIAL_CHARS for c in password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength_score": self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length bonus
        score += min(25, len(password) * 2)
        
        # Character variety bonus
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in security_settings.PASSWORD_SPECIAL_CHARS for c in password):
            score += 15
        
        # Complexity bonus
        unique_chars = len(set(password))
        score += min(15, unique_chars)
        
        return min(100, score)
    
    # =====================================
    # JWT TOKEN MANAGEMENT
    # =====================================
    
    def create_access_token(self, payload: TokenPayload, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = {
            "sub": str(payload.user_id),
            "email": payload.email,
            "role": payload.role.value,
            "permissions": payload.permissions,
            "session_id": payload.session_id,
            "token_type": "access",
            "iss": self.jwt_config["issuer"],
            "aud": self.jwt_config["audience"]
        }
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.jwt_config["access_token_expire"]
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.jwt_config["secret_key"],
            algorithm=self.jwt_config["algorithm"]
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, payload: TokenPayload) -> str:
        """Create a JWT refresh token"""
        to_encode = {
            "sub": str(payload.user_id),
            "session_id": payload.session_id,
            "token_type": "refresh",
            "iss": self.jwt_config["issuer"],
            "aud": self.jwt_config["audience"]
        }
        
        expire = datetime.utcnow() + self.jwt_config["refresh_token_expire"]
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.jwt_config["secret_key"],
            algorithm=self.jwt_config["algorithm"]
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            # Check if token is blacklisted
            if self.is_token_blacklisted(token):
                raise AuthenticationError("Token has been revoked")
            
            payload = jwt.decode(
                token,
                self.jwt_config["secret_key"],
                algorithms=[self.jwt_config["algorithm"]],
                audience=self.jwt_config["audience"],
                issuer=self.jwt_config["issuer"]
            )
            
            # Verify token type
            if payload.get("token_type") != token_type:
                raise AuthenticationError("Invalid token type")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_config["secret_key"],
                algorithms=[self.jwt_config["algorithm"]],
                options={"verify_exp": False}  # Don't verify expiration for blacklisting
            )
            
            token_id = hashlib.sha256(token.encode()).hexdigest()
            
            if redis_client:
                # Store in Redis with expiration based on token exp time
                exp_timestamp = payload.get("exp", 0)
                ttl = max(0, exp_timestamp - int(datetime.utcnow().timestamp()))
                redis_client.setex(f"blacklist:{token_id}", ttl, "1")
            else:
                # Fallback to in-memory set
                token_blacklist.add(token_id)
                
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["AUTHENTICATION_SUCCESS"],
                user_id=payload.get("sub"),
                details={"action": "token_blacklisted"}
            )
        
        except jwt.JWTError:
            pass  # Invalid token, ignore
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        token_id = hashlib.sha256(token.encode()).hexdigest()
        
        if redis_client:
            return redis_client.exists(f"blacklist:{token_id}")
        else:
            return token_id in token_blacklist
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token, "refresh")
        
        # Get user data
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(payload["sub"])).first()
            if not user:
                raise AuthenticationError("User not found")
            
            # Create new access token
            token_payload = TokenPayload(
                user_id=user.id,
                email=user.email,
                role=UserRole(user.role),
                permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(user.role)]],
                session_id=payload["session_id"]
            )
            
            return self.create_access_token(token_payload)
        
        finally:
            db.close()
    
    # =====================================
    # MULTI-FACTOR AUTHENTICATION
    # =====================================
    
    def generate_mfa_secret(self, user_email: str) -> str:
        """Generate MFA secret for user"""
        return pyotp.random_base32()
    
    def generate_mfa_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for MFA setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=security_settings.MFA_ISSUER_NAME
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify MFA TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=security_settings.TOTP_DRIFT_SECONDS // 30)
    
    def generate_backup_codes(self) -> List[str]:
        """Generate MFA backup codes"""
        return [secrets.token_hex(4).upper() for _ in range(security_settings.MFA_BACKUP_CODES_COUNT)]
    
    # =====================================
    # SESSION MANAGEMENT
    # =====================================
    
    def create_session(self, user_id: int, request: Request) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "user_id": user_id,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "active": True
        }
        
        if redis_client:
            redis_client.hset(f"session:{session_id}", mapping=session_data)
            redis_client.expire(f"session:{session_id}", security_settings.SESSION_TIMEOUT_MINUTES * 60)
        else:
            self.active_sessions[session_id] = session_data
        
        return session_id
    
    def validate_session(self, session_id: str, request: Request) -> bool:
        """Validate user session"""
        if redis_client:
            session_data = redis_client.hgetall(f"session:{session_id}")
            if not session_data:
                return False
        else:
            session_data = self.active_sessions.get(session_id)
            if not session_data:
                return False
        
        # Update last activity
        if redis_client:
            redis_client.hset(f"session:{session_id}", "last_activity", datetime.utcnow().isoformat())
            redis_client.expire(f"session:{session_id}", security_settings.SESSION_TIMEOUT_MINUTES * 60)
        else:
            session_data["last_activity"] = datetime.utcnow().isoformat()
        
        return True
    
    def revoke_session(self, session_id: str):
        """Revoke user session"""
        if redis_client:
            redis_client.delete(f"session:{session_id}")
        else:
            self.active_sessions.pop(session_id, None)
    
    def revoke_all_user_sessions(self, user_id: int):
        """Revoke all sessions for a user"""
        if redis_client:
            # This would require a more complex implementation in Redis
            # For now, we'll mark sessions as revoked
            pass
        else:
            sessions_to_remove = [
                session_id for session_id, session_data in self.active_sessions.items()
                if session_data["user_id"] == user_id
            ]
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
    
    # =====================================
    # ACCOUNT SECURITY
    # =====================================
    
    def record_failed_attempt(self, identifier: str, request: Request):
        """Record failed login attempt"""
        client_ip = request.client.host
        
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = {
                "count": 0,
                "last_attempt": None,
                "locked_until": None
            }
        
        attempt_data = self.failed_attempts[client_ip]
        attempt_data["count"] += 1
        attempt_data["last_attempt"] = datetime.utcnow()
        
        # Check if account should be locked
        if attempt_data["count"] >= security_settings.LOGIN_MAX_ATTEMPTS:
            lockout_duration = timedelta(minutes=security_settings.LOCKOUT_DURATION_MINUTES)
            
            # Escalate lockout duration if enabled
            if security_settings.LOCKOUT_ESCALATION_ENABLED:
                escalation_factor = min(attempt_data["count"] // security_settings.LOGIN_MAX_ATTEMPTS, 4)
                lockout_duration *= escalation_factor
            
            attempt_data["locked_until"] = datetime.utcnow() + lockout_duration
            
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["ACCOUNT_LOCKED"],
                client_ip=client_ip,
                details={
                    "identifier": identifier,
                    "failed_attempts": attempt_data["count"],
                    "locked_until": attempt_data["locked_until"].isoformat()
                }
            )
    
    def is_account_locked(self, identifier: str, request: Request) -> bool:
        """Check if account is locked due to failed attempts"""
        client_ip = request.client.host
        
        if client_ip not in self.failed_attempts:
            return False
        
        attempt_data = self.failed_attempts[client_ip]
        
        if attempt_data.get("locked_until"):
            if datetime.utcnow() < attempt_data["locked_until"]:
                return True
            else:
                # Lock expired, reset attempts
                self.clear_failed_attempts(client_ip)
                return False
        
        return False
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed login attempts"""
        self.failed_attempts.pop(identifier, None)
    
    # =====================================
    # AUTHORIZATION
    # =====================================
    
    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if user role has specific permission"""
        return permission in ROLE_PERMISSIONS.get(user_role, [])
    
    def require_permission(self, permission: Permission):
        """Decorator factory for requiring specific permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be implemented as a FastAPI dependency
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Global authentication manager instance
auth_manager = AuthManager()

# FastAPI security scheme
security_scheme = HTTPBearer()


# =====================================
# FASTAPI DEPENDENCIES
# =====================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    request: Request = None
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user"""
    try:
        token = credentials.credentials
        payload = auth_manager.verify_token(token, "access")
        
        # Validate session if present
        session_id = payload.get("session_id")
        if session_id and not auth_manager.validate_session(session_id, request):
            raise AuthenticationError("Invalid session")
        
        return payload
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_permission(permission: Permission):
    """FastAPI dependency to require specific permission"""
    def permission_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = UserRole(current_user["role"])
        
        if not auth_manager.has_permission(user_role, permission):
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["AUTHORIZATION_DENIED"],
                user_id=current_user["sub"],
                details={
                    "required_permission": permission.value,
                    "user_role": user_role.value
                }
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        
        return current_user
    
    return permission_checker


async def require_role(role: UserRole):
    """FastAPI dependency to require specific role"""
    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = UserRole(current_user["role"])
        
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required"
            )
        
        return current_user
    
    return role_checker


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency to require admin role"""
    return await require_role(UserRole.ADMIN)(current_user)


# =====================================
# UTILITY FUNCTIONS
# =====================================

def create_api_key(user_id: int, name: str) -> Dict[str, str]:
    """Create API key for user"""
    key = security_settings.API_KEY_PREFIX + secrets.token_urlsafe(security_settings.API_KEY_LENGTH)
    
    # In production, store this in database with expiration
    return {
        "key": key,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=security_settings.API_KEY_EXPIRY_DAYS)).isoformat()
    }


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key and return associated user info"""
    # In production, look up key in database
    # This is a placeholder implementation
    return None