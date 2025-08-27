"""
SuperNova AI API Security and WebSocket Protection
Advanced security middleware and protection mechanisms
"""

import os
import time
import json
import hmac
import hashlib
import base64
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import re
import logging

from fastapi import Request, Response, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import jwt

from .security_config import security_settings, SECURITY_HEADERS
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES
from .auth import auth_manager, get_current_user
from .input_validation import input_validator, request_validator
from .rate_limiting import rate_limiter, RateLimitType

logger = logging.getLogger(__name__)


class APISecurityLevel(str, Enum):
    """API security levels"""
    PUBLIC = "public"           # No authentication required
    AUTHENTICATED = "authenticated"  # Authentication required
    AUTHORIZED = "authorized"   # Authorization required
    ADMIN = "admin"            # Admin access required


class WebSocketSecurityLevel(str, Enum):
    """WebSocket security levels"""
    OPEN = "open"              # No authentication
    TOKEN = "token"            # Token-based auth
    SESSION = "session"        # Session-based auth
    CERTIFICATE = "certificate" # Certificate-based auth


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive API security middleware
    Handles authentication, authorization, input validation, and security headers
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_config = security_settings
        
        # Endpoint security configuration
        self.endpoint_security = self._load_endpoint_security_config()
        
        # CSRF protection
        self.csrf_exempt_paths = {"/health", "/metrics", "/docs", "/openapi.json"}
        
        # Request tracking
        self.active_requests: Dict[str, Dict] = {}
        
    def _load_endpoint_security_config(self) -> Dict[str, Dict]:
        """Load endpoint-specific security configuration"""
        return {
            # Public endpoints (no auth required)
            "GET:/health": {"security_level": APISecurityLevel.PUBLIC},
            "GET:/metrics": {"security_level": APISecurityLevel.PUBLIC},
            "GET:/docs": {"security_level": APISecurityLevel.PUBLIC},
            "GET:/openapi.json": {"security_level": APISecurityLevel.PUBLIC},
            
            # Authentication required
            "POST:/chat": {"security_level": APISecurityLevel.AUTHENTICATED},
            "GET:/chat/sessions": {"security_level": APISecurityLevel.AUTHENTICATED},
            "POST:/advice": {"security_level": APISecurityLevel.AUTHENTICATED},
            "POST:/backtest": {"security_level": APISecurityLevel.AUTHENTICATED},
            
            # Authorization required
            "POST:/intake": {"security_level": APISecurityLevel.AUTHORIZED},
            "POST:/watchlist/add": {"security_level": APISecurityLevel.AUTHORIZED},
            "POST:/optimize/strategy": {"security_level": APISecurityLevel.AUTHORIZED},
            
            # Admin endpoints
            "GET:/optimize/dashboard": {"security_level": APISecurityLevel.ADMIN},
            "DELETE:/optimize/study/*": {"security_level": APISecurityLevel.ADMIN},
        }
    
    async def dispatch(self, request: Request, call_next):
        """Main security middleware dispatch"""
        start_time = time.time()
        request_id = self._generate_request_id()
        
        try:
            # Add request to tracking
            self.active_requests[request_id] = {
                "start_time": start_time,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", "")
            }
            
            # Add request ID to request state
            request.state.request_id = request_id
            
            # Step 1: Basic request validation
            await self._validate_basic_request(request)
            
            # Step 2: Rate limiting
            await self._check_rate_limits(request)
            
            # Step 3: Input validation
            await self._validate_request_input(request)
            
            # Step 4: Authentication and authorization
            await self._check_authentication_authorization(request)
            
            # Step 5: CSRF protection
            await self._check_csrf_protection(request)
            
            # Step 6: Process request
            response = await call_next(request)
            
            # Step 7: Add security headers
            self._add_security_headers(response)
            
            # Step 8: Log successful request
            processing_time = time.time() - start_time
            await self._log_request_success(request, response, processing_time)
            
            return response
            
        except HTTPException as e:
            # Log security-related HTTP exceptions
            processing_time = time.time() - start_time
            await self._log_request_failure(request, e, processing_time)
            raise
            
        except Exception as e:
            # Log unexpected errors
            processing_time = time.time() - start_time
            await self._log_request_error(request, e, processing_time)
            
            # Don't expose internal errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
            
        finally:
            # Remove from active requests
            self.active_requests.pop(request_id, None)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def _validate_basic_request(self, request: Request):
        """Validate basic request properties"""
        # Validate content length
        content_length = int(request.headers.get("content-length", 0))
        if content_length > security_settings.MAX_REQUEST_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Validate content type for non-GET requests
        if request.method not in ["GET", "HEAD", "OPTIONS"]:
            content_type = request.headers.get("content-type", "")
            if content_type and not any(
                allowed in content_type 
                for allowed in security_settings.ALLOWED_CONTENT_TYPES
            ):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported content type"
                )
        
        # Validate user agent
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or len(user_agent) > 500:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
                level=SecurityEventLevel.INFO,
                client_ip=request.client.host,
                details={"invalid_user_agent": user_agent[:100]}
            )
        
        # Check for suspicious headers
        await self._check_suspicious_headers(request)
    
    async def _check_suspicious_headers(self, request: Request):
        """Check for suspicious request headers"""
        suspicious_patterns = [
            r"union.*select",
            r"<script",
            r"javascript:",
            r"eval\(",
            r"alert\(",
            r"\.\./",
            r"etc/passwd"
        ]
        
        for header_name, header_value in request.headers.items():
            header_value_lower = header_value.lower()
            
            for pattern in suspicious_patterns:
                if re.search(pattern, header_value_lower):
                    security_logger.log_security_event(
                        event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                        level=SecurityEventLevel.WARNING,
                        client_ip=request.client.host,
                        endpoint=request.url.path,
                        details={
                            "suspicious_header": header_name,
                            "pattern": pattern,
                            "value": header_value[:100]
                        }
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Malicious request detected"
                    )
    
    async def _check_rate_limits(self, request: Request):
        """Check rate limits for the request"""
        # Determine rate limit type based on authentication
        if "Authorization" in request.headers:
            limit_type = RateLimitType.PER_USER
        elif "X-API-Key" in request.headers:
            limit_type = RateLimitType.PER_API_KEY
        else:
            limit_type = RateLimitType.PER_IP
        
        allowed, info = await rate_limiter.check_rate_limit(request, limit_type)
        
        if not allowed:
            headers = {
                "X-RateLimit-Limit": str(info.get("max", "unknown")),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + info.get("reset_time", 60))),
                "Retry-After": str(int(info.get("reset_time", 60)))
            }
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=headers
            )
    
    async def _validate_request_input(self, request: Request):
        """Validate request input for security"""
        # Validate query parameters
        for param_name, param_value in request.query_params.items():
            try:
                input_validator.validate_input(param_value, "xss", max_length=1000)
            except Exception:
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                    level=SecurityEventLevel.WARNING,
                    client_ip=request.client.host,
                    endpoint=request.url.path,
                    details={
                        "invalid_query_param": param_name,
                        "value": param_value[:100]
                    }
                )
                
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid query parameter: {param_name}"
                )
        
        # Validate path parameters
        path = request.url.path
        if "../" in path or "\\" in path:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                client_ip=request.client.host,
                details={"path_traversal_attempt": path}
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path"
            )
    
    async def _check_authentication_authorization(self, request: Request):
        """Check authentication and authorization requirements"""
        endpoint_key = f"{request.method}:{request.url.path}"
        
        # Check for wildcard endpoint patterns
        endpoint_config = None
        for pattern, config in self.endpoint_security.items():
            if pattern.endswith("/*") and endpoint_key.startswith(pattern[:-2]):
                endpoint_config = config
                break
            elif pattern == endpoint_key:
                endpoint_config = config
                break
        
        if not endpoint_config:
            # Default to authenticated for unknown endpoints
            endpoint_config = {"security_level": APISecurityLevel.AUTHENTICATED}
        
        security_level = APISecurityLevel(endpoint_config["security_level"])
        
        if security_level == APISecurityLevel.PUBLIC:
            return  # No authentication required
        
        # Extract authentication credentials
        auth_header = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")
        
        if not auth_header and not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate JWT token
        user_data = None
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                user_data = auth_manager.verify_token(token)
                request.state.user = user_data
            except Exception as e:
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["AUTHENTICATION_FAILURE"],
                    level=SecurityEventLevel.WARNING,
                    client_ip=request.client.host,
                    details={"error": str(e)}
                )
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Validate API key
        elif api_key:
            # In production, validate API key against database
            # For now, just check format
            if not api_key.startswith(security_settings.API_KEY_PREFIX):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
        
        # Check authorization level
        if security_level == APISecurityLevel.ADMIN:
            if not user_data or user_data.get("role") != "admin":
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["AUTHORIZATION_DENIED"],
                    level=SecurityEventLevel.WARNING,
                    client_ip=request.client.host,
                    endpoint=request.url.path,
                    user_id=user_data.get("sub") if user_data else None,
                    details={"required_role": "admin"}
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
    
    async def _check_csrf_protection(self, request: Request):
        """Check CSRF protection for state-changing requests"""
        if (request.method in ["POST", "PUT", "DELETE", "PATCH"] and 
            request.url.path not in self.csrf_exempt_paths):
            
            csrf_token = request.headers.get("x-csrf-token")
            if not csrf_token:
                # For API endpoints, CSRF token might be optional if using API key
                api_key = request.headers.get("x-api-key")
                if not api_key:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="CSRF token required"
                    )
            else:
                # Validate CSRF token
                if not self._validate_csrf_token(csrf_token):
                    security_logger.log_security_event(
                        event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                        level=SecurityEventLevel.WARNING,
                        client_ip=request.client.host,
                        endpoint=request.url.path,
                        details={"invalid_csrf_token": csrf_token[:16]}
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid CSRF token"
                    )
    
    def _validate_csrf_token(self, token: str) -> bool:
        """Validate CSRF token"""
        try:
            # In production, implement proper CSRF token validation
            # This is a simplified version
            decoded = base64.b64decode(token)
            # Add proper validation logic here
            return len(decoded) == 32
        except:
            return False
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Add additional security headers
        response.headers["X-Request-ID"] = getattr(response, "request_id", "unknown")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
    
    async def _log_request_success(self, request: Request, response: Response, processing_time: float):
        """Log successful request"""
        security_logger.log_security_event(
            event_type="API_REQUEST_SUCCESS",
            level=SecurityEventLevel.INFO,
            client_ip=request.client.host,
            endpoint=request.url.path,
            method=request.method,
            user_id=getattr(request.state, "user", {}).get("sub") if hasattr(request.state, "user") else None,
            details={
                "status_code": response.status_code,
                "processing_time_ms": processing_time * 1000,
                "request_size": int(request.headers.get("content-length", 0))
            }
        )
    
    async def _log_request_failure(self, request: Request, error: HTTPException, processing_time: float):
        """Log failed request"""
        security_logger.log_security_event(
            event_type="API_REQUEST_FAILURE",
            level=SecurityEventLevel.WARNING,
            client_ip=request.client.host,
            endpoint=request.url.path,
            method=request.method,
            details={
                "status_code": error.status_code,
                "error_detail": error.detail,
                "processing_time_ms": processing_time * 1000
            }
        )
    
    async def _log_request_error(self, request: Request, error: Exception, processing_time: float):
        """Log request error"""
        security_logger.log_security_event(
            event_type="API_REQUEST_ERROR",
            level=SecurityEventLevel.ERROR,
            client_ip=request.client.host,
            endpoint=request.url.path,
            method=request.method,
            details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "processing_time_ms": processing_time * 1000
            }
        )


class WebSocketSecurity:
    """
    WebSocket security manager with authentication and message validation
    """
    
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        self.connection_limits: Dict[str, int] = {}
        self.message_counters: Dict[str, Dict] = {}
        
    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        token: Optional[str] = None,
        security_level: WebSocketSecurityLevel = WebSocketSecurityLevel.TOKEN
    ) -> Dict[str, Any]:
        """Authenticate WebSocket connection"""
        
        if security_level == WebSocketSecurityLevel.OPEN:
            return {"authenticated": True, "user": None}
        
        if not token:
            await websocket.close(code=1008, reason="Authentication required")
            raise WebSocketDisconnect(code=1008, reason="Authentication required")
        
        try:
            if security_level == WebSocketSecurityLevel.TOKEN:
                # Validate JWT token
                user_data = auth_manager.verify_token(token)
                
                # Check connection limits
                user_id = user_data.get("sub")
                if user_id:
                    current_connections = self.connection_limits.get(user_id, 0)
                    if current_connections >= security_settings.WS_MAX_CONNECTIONS_PER_USER:
                        await websocket.close(code=1008, reason="Connection limit exceeded")
                        raise WebSocketDisconnect(code=1008, reason="Connection limit exceeded")
                
                return {"authenticated": True, "user": user_data}
            
            elif security_level == WebSocketSecurityLevel.SESSION:
                # Validate session-based authentication
                # Implementation depends on session management
                pass
            
            elif security_level == WebSocketSecurityLevel.CERTIFICATE:
                # Validate certificate-based authentication
                # Implementation depends on certificate validation
                pass
        
        except Exception as e:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["AUTHENTICATION_FAILURE"],
                level=SecurityEventLevel.WARNING,
                client_ip=websocket.client.host if websocket.client else "unknown",
                details={
                    "websocket_auth_failure": str(e),
                    "security_level": security_level.value
                }
            )
            
            await websocket.close(code=1008, reason="Authentication failed")
            raise WebSocketDisconnect(code=1008, reason="Authentication failed")
    
    async def register_connection(
        self,
        connection_id: str,
        websocket: WebSocket,
        user_data: Optional[Dict] = None
    ):
        """Register new WebSocket connection"""
        
        # Store connection info
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "user": user_data,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "message_count": 0,
            "client_ip": websocket.client.host if websocket.client else "unknown"
        }
        
        # Update connection limits
        if user_data:
            user_id = user_data.get("sub")
            if user_id:
                self.connection_limits[user_id] = self.connection_limits.get(user_id, 0) + 1
        
        # Initialize message counter
        self.message_counters[connection_id] = {
            "messages": deque(maxlen=100),
            "last_reset": time.time()
        }
        
        security_logger.log_security_event(
            event_type="WEBSOCKET_CONNECTION_ESTABLISHED",
            level=SecurityEventLevel.INFO,
            client_ip=websocket.client.host if websocket.client else "unknown",
            user_id=user_data.get("sub") if user_data else None,
            details={"connection_id": connection_id}
        )
    
    async def unregister_connection(self, connection_id: str):
        """Unregister WebSocket connection"""
        if connection_id in self.active_connections:
            connection_info = self.active_connections[connection_id]
            user_data = connection_info.get("user")
            
            # Update connection limits
            if user_data:
                user_id = user_data.get("sub")
                if user_id and user_id in self.connection_limits:
                    self.connection_limits[user_id] = max(0, self.connection_limits[user_id] - 1)
                    if self.connection_limits[user_id] == 0:
                        del self.connection_limits[user_id]
            
            # Clean up
            del self.active_connections[connection_id]
            self.message_counters.pop(connection_id, None)
            
            security_logger.log_security_event(
                event_type="WEBSOCKET_CONNECTION_CLOSED",
                level=SecurityEventLevel.INFO,
                client_ip=connection_info.get("client_ip", "unknown"),
                user_id=user_data.get("sub") if user_data else None,
                details={
                    "connection_id": connection_id,
                    "duration_seconds": (datetime.utcnow() - connection_info["connected_at"]).total_seconds(),
                    "message_count": connection_info["message_count"]
                }
            )
    
    async def validate_message(
        self,
        connection_id: str,
        message: str,
        max_size: int = None
    ) -> bool:
        """Validate incoming WebSocket message"""
        
        if connection_id not in self.active_connections:
            return False
        
        # Check message size
        message_size = len(message.encode('utf-8'))
        max_allowed = max_size or security_settings.WS_MESSAGE_SIZE_LIMIT
        
        if message_size > max_allowed:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                client_ip=self.active_connections[connection_id].get("client_ip"),
                details={
                    "oversized_websocket_message": True,
                    "message_size": message_size,
                    "max_allowed": max_allowed,
                    "connection_id": connection_id
                }
            )
            return False
        
        # Check rate limiting
        if not self._check_message_rate_limit(connection_id):
            return False
        
        # Validate message content
        try:
            message_data = json.loads(message)
            
            # Validate message structure
            if not isinstance(message_data, dict):
                return False
            
            # Check for required fields
            if "type" not in message_data:
                return False
            
            # Validate message type
            allowed_types = ["chat", "typing", "presence", "heartbeat", "system"]
            if message_data["type"] not in allowed_types:
                return False
            
            # Validate message content for security
            if "data" in message_data:
                data = message_data["data"]
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            try:
                                input_validator.validate_input(value, "xss", max_length=10000)
                            except:
                                return False
            
            return True
            
        except json.JSONDecodeError:
            return False
        except Exception:
            return False
    
    def _check_message_rate_limit(self, connection_id: str) -> bool:
        """Check message rate limiting"""
        if connection_id not in self.message_counters:
            return False
        
        counter_info = self.message_counters[connection_id]
        current_time = time.time()
        
        # Add current message timestamp
        counter_info["messages"].append(current_time)
        
        # Count messages in the last second
        one_second_ago = current_time - 1.0
        recent_messages = [
            timestamp for timestamp in counter_info["messages"]
            if timestamp > one_second_ago
        ]
        
        # Check rate limit
        if len(recent_messages) > security_settings.WS_RATE_LIMIT_MESSAGES_PER_SECOND:
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
                level=SecurityEventLevel.WARNING,
                client_ip=self.active_connections[connection_id].get("client_ip"),
                details={
                    "websocket_rate_limit_exceeded": True,
                    "messages_per_second": len(recent_messages),
                    "connection_id": connection_id
                }
            )
            return False
        
        # Update connection activity
        if connection_id in self.active_connections:
            self.active_connections[connection_id]["last_activity"] = datetime.utcnow()
            self.active_connections[connection_id]["message_count"] += 1
        
        return True
    
    async def send_heartbeat(self, connection_id: str):
        """Send heartbeat to WebSocket connection"""
        if connection_id in self.active_connections:
            connection_info = self.active_connections[connection_id]
            websocket = connection_info["websocket"]
            
            try:
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "server_time": time.time()
                }
                
                await websocket.send_text(json.dumps(heartbeat_message))
                
            except Exception as e:
                # Connection might be dead, remove it
                await self.unregister_connection(connection_id)
    
    async def cleanup_inactive_connections(self):
        """Clean up inactive WebSocket connections"""
        current_time = datetime.utcnow()
        timeout_threshold = timedelta(seconds=security_settings.WS_CONNECTION_TIMEOUT)
        
        inactive_connections = []
        
        for connection_id, connection_info in self.active_connections.items():
            if current_time - connection_info["last_activity"] > timeout_threshold:
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            connection_info = self.active_connections[connection_id]
            websocket = connection_info["websocket"]
            
            try:
                await websocket.close(code=1000, reason="Connection timeout")
            except:
                pass  # Connection might already be closed
            
            await self.unregister_connection(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        total_connections = len(self.active_connections)
        user_connections = len(self.connection_limits)
        
        total_messages = sum(
            conn["message_count"] for conn in self.active_connections.values()
        )
        
        return {
            "total_connections": total_connections,
            "authenticated_users": user_connections,
            "total_messages_processed": total_messages,
            "average_messages_per_connection": total_messages / max(1, total_connections),
            "connection_limits": dict(self.connection_limits)
        }


# Global instances
websocket_security = WebSocketSecurity()


# =====================================
# UTILITY FUNCTIONS
# =====================================

def create_csrf_token() -> str:
    """Create CSRF token"""
    token_data = {
        "timestamp": time.time(),
        "random": base64.b64encode(os.urandom(16)).decode()
    }
    
    token_json = json.dumps(token_data)
    return base64.b64encode(token_json.encode()).decode()


def validate_api_signature(
    payload: str,
    signature: str,
    secret: str,
    timestamp: Optional[str] = None
) -> bool:
    """Validate API request signature"""
    if timestamp:
        # Check timestamp tolerance (5 minutes)
        try:
            request_time = float(timestamp)
            current_time = time.time()
            
            if abs(current_time - request_time) > 300:  # 5 minutes
                return False
            
            payload = f"{timestamp}.{payload}"
        except:
            return False
    
    # Calculate expected signature
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures (constant time comparison)
    return hmac.compare_digest(signature, expected_signature)


async def secure_websocket_endpoint(
    websocket: WebSocket,
    security_level: WebSocketSecurityLevel = WebSocketSecurityLevel.TOKEN
):
    """Decorator for securing WebSocket endpoints"""
    
    # Extract token from query parameters or headers
    token = websocket.query_params.get("token")
    if not token and "authorization" in websocket.headers:
        auth_header = websocket.headers["authorization"]
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    # Authenticate connection
    auth_result = await websocket_security.authenticate_websocket(
        websocket, token, security_level
    )
    
    # Accept connection
    await websocket.accept()
    
    # Generate connection ID
    import uuid
    connection_id = str(uuid.uuid4())
    
    # Register connection
    await websocket_security.register_connection(
        connection_id, websocket, auth_result.get("user")
    )
    
    return connection_id