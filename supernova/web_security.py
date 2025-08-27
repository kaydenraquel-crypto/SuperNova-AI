"""
SuperNova AI Web Security Protection
CORS, CSRF, XSS, and other web-specific security measures
"""

import os
import json
import hmac
import hashlib
import base64
import secrets
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from urllib.parse import urlparse
import logging

from fastapi import Request, Response, HTTPException, status, Cookie
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .security_config import security_settings
from .security_logger import security_logger, SecurityEventLevel, SECURITY_EVENT_TYPES

logger = logging.getLogger(__name__)


class CSRFProtection:
    """
    Comprehensive CSRF (Cross-Site Request Forgery) protection
    """
    
    def __init__(self):
        self.secret_key = security_settings.CSRF_SECRET_KEY
        self.exempt_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
        self.exempt_paths = {"/health", "/metrics", "/docs", "/openapi.json"}
        self.token_lifetime = 3600  # 1 hour
        
        # Double-submit cookie configuration
        self.cookie_name = "__Host-csrf-token"
        self.header_name = "X-CSRF-Token"
        self.form_field_name = "csrf_token"
        
    def generate_token(self, session_id: str = None) -> str:
        """Generate CSRF token"""
        if session_id is None:
            session_id = secrets.token_urlsafe(16)
        
        timestamp = int(datetime.utcnow().timestamp())
        random_value = secrets.token_urlsafe(16)
        
        # Create token payload
        payload = {
            "session_id": session_id,
            "timestamp": timestamp,
            "random": random_value
        }
        
        # Sign the payload
        payload_json = json.dumps(payload, sort_keys=True)
        signature = self._sign_payload(payload_json)
        
        # Combine payload and signature
        token_data = {
            "payload": base64.b64encode(payload_json.encode()).decode(),
            "signature": signature
        }
        
        return base64.b64encode(json.dumps(token_data).encode()).decode()
    
    def validate_token(self, token: str, session_id: str = None) -> bool:
        """Validate CSRF token"""
        try:
            # Decode token
            token_data = json.loads(base64.b64decode(token).decode())
            payload_b64 = token_data["payload"]
            signature = token_data["signature"]
            
            # Verify signature
            payload_json = base64.b64decode(payload_b64).decode()
            if not self._verify_signature(payload_json, signature):
                return False
            
            # Parse payload
            payload = json.loads(payload_json)
            
            # Check timestamp
            token_timestamp = payload["timestamp"]
            current_timestamp = int(datetime.utcnow().timestamp())
            
            if current_timestamp - token_timestamp > self.token_lifetime:
                return False
            
            # Check session ID if provided
            if session_id and payload.get("session_id") != session_id:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"CSRF token validation failed: {e}")
            return False
    
    def _sign_payload(self, payload: str) -> str:
        """Sign payload with HMAC"""
        return hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_signature(self, payload: str, signature: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = self._sign_payload(payload)
        return hmac.compare_digest(signature, expected_signature)
    
    def is_exempt_request(self, request: Request) -> bool:
        """Check if request is exempt from CSRF protection"""
        # Check method
        if request.method in self.exempt_methods:
            return True
        
        # Check path
        if request.url.path in self.exempt_paths:
            return True
        
        # Check if API request with API key
        if request.headers.get("X-API-Key"):
            return True
        
        # Check content type for API requests
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type and request.headers.get("Authorization"):
            return True
        
        return False
    
    async def validate_request(self, request: Request) -> bool:
        """Validate request for CSRF protection"""
        if self.is_exempt_request(request):
            return True
        
        # Get CSRF token from header or form
        csrf_token = request.headers.get(self.header_name)
        
        if not csrf_token:
            # Try to get from form data (for form submissions)
            if hasattr(request, "_form"):
                form_data = await request.form()
                csrf_token = form_data.get(self.form_field_name)
        
        if not csrf_token:
            return False
        
        # Get session ID from cookie or JWT
        session_id = self._extract_session_id(request)
        
        return self.validate_token(csrf_token, session_id)
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request"""
        # Try to get from session cookie
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            return session_cookie
        
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                import jwt
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token, 
                    options={"verify_signature": False}  # Just extracting session ID
                )
                return payload.get("session_id")
            except:
                pass
        
        return None
    
    def add_token_to_response(self, response: Response, session_id: str = None):
        """Add CSRF token to response"""
        token = self.generate_token(session_id)
        
        # Set as secure cookie
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            httponly=True,
            secure=security_settings.SSL_REQUIRED,
            samesite="strict",
            max_age=self.token_lifetime
        )
        
        # Also add to response headers for JavaScript access
        response.headers["X-CSRF-Token"] = token


class XSSProtection:
    """
    XSS (Cross-Site Scripting) protection system
    """
    
    def __init__(self):
        self.content_security_policy = self._build_csp()
        
        # XSS detection patterns
        self.xss_patterns = [
            # Script tags
            r'<script[^>]*>.*?</script>',
            r'<script[^>]*>',
            
            # Event handlers
            r'on\w+\s*=',
            
            # JavaScript URIs
            r'javascript\s*:',
            r'vbscript\s*:',
            r'data\s*:.*base64',
            
            # HTML entities that could be XSS
            r'&#x?\d+;',
            
            # Style expressions
            r'expression\s*\(',
            r'@import',
            
            # Object/embed tags
            r'<(object|embed|applet)[^>]*>',
            
            # Iframe tags
            r'<iframe[^>]*>',
            
            # Meta refresh
            r'<meta[^>]*refresh[^>]*>',
            
            # Form elements
            r'<form[^>]*action\s*=[^>]*>',
        ]
        
        # Compiled regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.xss_patterns]
    
    def _build_csp(self) -> str:
        """Build Content Security Policy"""
        if security_settings.is_production():
            # Strict CSP for production
            return (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' wss: https:; "
                "media-src 'none'; "
                "object-src 'none'; "
                "frame-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "upgrade-insecure-requests; "
                "block-all-mixed-content"
            )
        else:
            # More permissive CSP for development
            return (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "img-src 'self' data: https: http:; "
                "connect-src 'self' ws: wss: https: http:; "
                "frame-ancestors 'none'"
            )
    
    def detect_xss(self, content: str) -> List[Dict[str, Any]]:
        """Detect potential XSS in content"""
        detected_patterns = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.finditer(content)
            for match in matches:
                detected_patterns.append({
                    "pattern_index": i,
                    "pattern": self.xss_patterns[i],
                    "match": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return detected_patterns
    
    def sanitize_html(self, content: str, allow_safe_tags: bool = True) -> str:
        """Sanitize HTML content"""
        import bleach
        
        if allow_safe_tags:
            # Allow safe HTML tags
            allowed_tags = [
                'p', 'br', 'strong', 'em', 'u', 'i', 'b',
                'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'blockquote', 'code', 'pre'
            ]
            
            allowed_attributes = {
                '*': ['class', 'id'],
                'a': ['href', 'title'],
                'img': ['src', 'alt', 'width', 'height']
            }
        else:
            # Strip all HTML
            allowed_tags = []
            allowed_attributes = {}
        
        return bleach.clean(
            content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True,
            strip_comments=True
        )
    
    def validate_input(self, content: str, field_name: str = None) -> Tuple[bool, List[str]]:
        """Validate input for XSS"""
        issues = []
        
        # Detect XSS patterns
        xss_detections = self.detect_xss(content)
        
        if xss_detections:
            for detection in xss_detections:
                issues.append(f"Potential XSS detected: {detection['pattern']}")
            
            # Log XSS attempt
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                details={
                    "xss_attempt": True,
                    "field_name": field_name,
                    "patterns_detected": len(xss_detections),
                    "content_length": len(content),
                    "detections": xss_detections[:5]  # Limit logged detections
                }
            )
        
        return len(issues) == 0, issues
    
    def add_xss_headers(self, response: Response):
        """Add XSS protection headers"""
        response.headers["Content-Security-Policy"] = self.content_security_policy
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"


class CORSProtection:
    """
    Enhanced CORS (Cross-Origin Resource Sharing) protection
    """
    
    def __init__(self):
        self.allowed_origins = self._get_allowed_origins()
        self.allowed_methods = security_settings.CORS_ALLOWED_METHODS
        self.allowed_headers = security_settings.CORS_ALLOWED_HEADERS
        self.allow_credentials = security_settings.CORS_ALLOW_CREDENTIALS
        self.max_age = security_settings.CORS_MAX_AGE
        
        # Origin validation regex patterns
        self.origin_patterns = self._compile_origin_patterns()
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins with environment-specific defaults"""
        origins = security_settings.CORS_ALLOWED_ORIGINS.copy()
        
        # Add localhost for development
        if not security_settings.is_production():
            localhost_origins = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ]
            origins.extend(localhost_origins)
        
        return origins
    
    def _compile_origin_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for origin validation"""
        patterns = []
        
        for origin in self.allowed_origins:
            if "*" in origin:
                # Convert wildcard to regex
                pattern = origin.replace("*", ".*")
                patterns.append(re.compile(f"^{pattern}$"))
            else:
                # Exact match
                patterns.append(re.compile(f"^{re.escape(origin)}$"))
        
        return patterns
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not origin:
            return False
        
        # Check exact matches first
        if origin in self.allowed_origins:
            return True
        
        # Check pattern matches
        for pattern in self.origin_patterns:
            if pattern.match(origin):
                return True
        
        return False
    
    def validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate CORS request"""
        origin = request.headers.get("origin")
        method = request.method
        
        validation_result = {
            "allowed": False,
            "origin": origin,
            "preflight": method == "OPTIONS",
            "issues": []
        }
        
        # Check if origin is provided for cross-origin requests
        if not origin:
            if method != "GET":
                validation_result["issues"].append("Missing origin header for non-GET request")
                return validation_result
        else:
            # Validate origin
            if not self.is_origin_allowed(origin):
                validation_result["issues"].append(f"Origin not allowed: {origin}")
                security_logger.log_security_event(
                    event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                    level=SecurityEventLevel.WARNING,
                    client_ip=request.client.host,
                    details={
                        "cors_violation": True,
                        "disallowed_origin": origin,
                        "method": method
                    }
                )
                return validation_result
        
        # Check method
        if method not in self.allowed_methods:
            validation_result["issues"].append(f"Method not allowed: {method}")
            return validation_result
        
        # For preflight requests, check requested headers
        if method == "OPTIONS":
            requested_headers = request.headers.get("access-control-request-headers", "")
            if requested_headers:
                requested_header_list = [h.strip().lower() for h in requested_headers.split(",")]
                allowed_header_list = [h.lower() for h in self.allowed_headers]
                
                for header in requested_header_list:
                    if header not in allowed_header_list:
                        validation_result["issues"].append(f"Header not allowed: {header}")
                        return validation_result
        
        validation_result["allowed"] = True
        return validation_result
    
    def add_cors_headers(self, response: Response, request: Request):
        """Add CORS headers to response"""
        origin = request.headers.get("origin")
        
        if origin and self.is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif not origin:
            # For same-origin requests, don't add CORS headers
            return
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        # Add exposed headers for client access
        exposed_headers = [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
        response.headers["Access-Control-Expose-Headers"] = ", ".join(exposed_headers)


class WebSecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive web security middleware combining CORS, CSRF, and XSS protection
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.csrf_protection = CSRFProtection()
        self.xss_protection = XSSProtection()
        self.cors_protection = CORSProtection()
        
        # Security policy configuration
        self.enforce_https = security_settings.SSL_REQUIRED
        self.strict_transport_security = security_settings.is_production()
    
    async def dispatch(self, request: Request, call_next):
        """Main security middleware dispatch"""
        
        # Step 1: HTTPS enforcement
        if self.enforce_https and request.url.scheme != "https":
            return self._redirect_to_https(request)
        
        # Step 2: CORS validation
        cors_validation = self.cors_protection.validate_request(request)
        if not cors_validation["allowed"]:
            return self._cors_error_response(cors_validation)
        
        # Handle CORS preflight requests
        if cors_validation["preflight"]:
            response = Response(status_code=200)
            self.cors_protection.add_cors_headers(response, request)
            return response
        
        # Step 3: CSRF validation
        if not await self.csrf_protection.validate_request(request):
            security_logger.log_security_event(
                event_type=SECURITY_EVENT_TYPES["SECURITY_VIOLATION"],
                level=SecurityEventLevel.WARNING,
                client_ip=request.client.host,
                endpoint=request.url.path,
                method=request.method,
                details={"csrf_validation_failed": True}
            )
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "CSRF token validation failed"}
            )
        
        # Step 4: Request body XSS validation (for non-GET requests)
        if request.method not in ["GET", "HEAD", "OPTIONS"]:
            await self._validate_request_body_xss(request)
        
        # Process request
        response = await call_next(request)
        
        # Step 5: Add security headers
        self._add_security_headers(response, request)
        
        # Step 6: Add CORS headers
        self.cors_protection.add_cors_headers(response, request)
        
        # Step 7: Add XSS protection headers
        self.xss_protection.add_xss_headers(response)
        
        # Step 8: Add CSRF token for form-based requests
        if self._should_add_csrf_token(request, response):
            session_id = self.csrf_protection._extract_session_id(request)
            self.csrf_protection.add_token_to_response(response, session_id)
        
        return response
    
    def _redirect_to_https(self, request: Request) -> Response:
        """Redirect HTTP requests to HTTPS"""
        https_url = request.url.replace(scheme="https")
        return Response(
            status_code=status.HTTP_301_MOVED_PERMANENTLY,
            headers={"Location": str(https_url)}
        )
    
    def _cors_error_response(self, cors_validation: Dict[str, Any]) -> JSONResponse:
        """Return CORS error response"""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "CORS policy violation",
                "issues": cors_validation["issues"]
            }
        )
    
    async def _validate_request_body_xss(self, request: Request):
        """Validate request body for XSS"""
        try:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                # For JSON requests, parse and validate
                body = await request.body()
                if body:
                    try:
                        json_data = json.loads(body.decode())
                        self._validate_json_for_xss(json_data)
                    except json.JSONDecodeError:
                        pass  # Invalid JSON will be handled by the endpoint
            
            elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
                # For form requests, validate form fields
                form = await request.form()
                for field_name, field_value in form.items():
                    if isinstance(field_value, str):
                        is_valid, issues = self.xss_protection.validate_input(field_value, field_name)
                        if not is_valid:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"XSS detected in field {field_name}: {issues[0]}"
                            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Request body XSS validation error: {e}")
    
    def _validate_json_for_xss(self, data: Any, path: str = ""):
        """Recursively validate JSON data for XSS"""
        if isinstance(data, dict):
            for key, value in data.items():
                self._validate_json_for_xss(value, f"{path}.{key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_json_for_xss(item, f"{path}[{i}]")
        elif isinstance(data, str):
            is_valid, issues = self.xss_protection.validate_input(data, path)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"XSS detected in JSON field {path}: {issues[0]}"
                )
    
    def _add_security_headers(self, response: Response, request: Request):
        """Add general security headers"""
        # Strict Transport Security
        if self.strict_transport_security:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={security_settings.HSTS_MAX_AGE}; "
                f"includeSubDomains{'; preload' if security_settings.is_production() else ''}"
            )
        
        # Content Type Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Frame Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (formerly Feature Policy)
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "usb=(), magnetometer=(), accelerometer=(), gyroscope=()"
        )
        
        # Add request tracking header
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
    
    def _should_add_csrf_token(self, request: Request, response: Response) -> bool:
        """Determine if CSRF token should be added to response"""
        # Add token for HTML responses that might contain forms
        content_type = response.headers.get("content-type", "")
        
        if "text/html" in content_type:
            return True
        
        # Add token for API responses if requested
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return True
        
        return False


# =====================================
# CONTENT SECURITY POLICY BUILDER
# =====================================

class CSPBuilder:
    """Content Security Policy builder with templates"""
    
    def __init__(self):
        self.directives = {}
    
    def add_directive(self, directive: str, sources: List[str]):
        """Add CSP directive"""
        self.directives[directive] = sources
        return self
    
    def default_src(self, sources: List[str]):
        """Set default-src directive"""
        return self.add_directive("default-src", sources)
    
    def script_src(self, sources: List[str]):
        """Set script-src directive"""
        return self.add_directive("script-src", sources)
    
    def style_src(self, sources: List[str]):
        """Set style-src directive"""
        return self.add_directive("style-src", sources)
    
    def img_src(self, sources: List[str]):
        """Set img-src directive"""
        return self.add_directive("img-src", sources)
    
    def connect_src(self, sources: List[str]):
        """Set connect-src directive"""
        return self.add_directive("connect-src", sources)
    
    def font_src(self, sources: List[str]):
        """Set font-src directive"""
        return self.add_directive("font-src", sources)
    
    def object_src(self, sources: List[str]):
        """Set object-src directive"""
        return self.add_directive("object-src", sources)
    
    def media_src(self, sources: List[str]):
        """Set media-src directive"""
        return self.add_directive("media-src", sources)
    
    def frame_src(self, sources: List[str]):
        """Set frame-src directive"""
        return self.add_directive("frame-src", sources)
    
    def build(self) -> str:
        """Build CSP string"""
        policy_parts = []
        
        for directive, sources in self.directives.items():
            sources_str = " ".join(sources)
            policy_parts.append(f"{directive} {sources_str}")
        
        return "; ".join(policy_parts)
    
    @classmethod
    def strict_policy(cls) -> str:
        """Build strict CSP policy for production"""
        return (cls()
                .default_src(["'self'"])
                .script_src(["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"])
                .style_src(["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"])
                .img_src(["'self'", "data:", "https:"])
                .connect_src(["'self'", "wss:", "https:"])
                .font_src(["'self'", "https://fonts.gstatic.com"])
                .object_src(["'none'"])
                .media_src(["'none'"])
                .frame_src(["'none'"])
                .add_directive("frame-ancestors", ["'none'"])
                .add_directive("base-uri", ["'self'"])
                .add_directive("form-action", ["'self'"])
                .build())
    
    @classmethod
    def development_policy(cls) -> str:
        """Build permissive CSP policy for development"""
        return (cls()
                .default_src(["'self'", "'unsafe-inline'", "'unsafe-eval'"])
                .img_src(["'self'", "data:", "https:", "http:"])
                .connect_src(["'self'", "ws:", "wss:", "https:", "http:"])
                .frame_ancestors(["'none'"])
                .build())


# Global instances
csrf_protection = CSRFProtection()
xss_protection = XSSProtection()
cors_protection = CORSProtection()


# =====================================
# UTILITY FUNCTIONS
# =====================================

def sanitize_user_input(content: str, allow_html: bool = False) -> str:
    """Sanitize user input for safe storage and display"""
    return xss_protection.sanitize_html(content, allow_html)


def validate_cors_origin(origin: str) -> bool:
    """Validate CORS origin"""
    return cors_protection.is_origin_allowed(origin)


def create_csrf_token_for_session(session_id: str) -> str:
    """Create CSRF token for specific session"""
    return csrf_protection.generate_token(session_id)