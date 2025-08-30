"""
SuperNova AI API Management Middleware
Comprehensive request/response processing middleware
"""

import time
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from .api_management_agent import api_management_agent, APIKeyTier
from .security_logger import security_logger, SecurityEventLevel

logger = logging.getLogger(__name__)


class APIManagementMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive API Management Middleware that handles:
    - Request validation and preprocessing
    - API key authentication and authorization
    - Rate limiting and quota management
    - Response caching and optimization
    - Usage tracking and analytics
    - Security monitoring and compliance
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.excluded_paths = {
            "/health", "/metrics", "/favicon.ico", "/docs", "/redoc", "/openapi.json"
        }
        self.admin_paths = {"/api/management/", "/admin/"}
        
    async def dispatch(self, request: Request, call_next: Callable):
        """Main middleware processing logic"""
        start_time = time.time()
        
        # Skip middleware for excluded paths
        if self._should_skip_middleware(request):
            return await call_next(request)
        
        try:
            # Phase 1: Request preprocessing and validation
            preprocess_result = await self._preprocess_request(request)
            if not preprocess_result["allowed"]:
                return self._create_error_response(preprocess_result)
            
            # Phase 2: API key authentication and validation
            api_key_info = await self._authenticate_api_key(request)
            
            # Phase 3: Authorization and access control
            auth_result = await self._authorize_request(request, api_key_info)
            if not auth_result["allowed"]:
                return self._create_error_response(auth_result)
            
            # Phase 4: Rate limiting and quota checks
            rate_limit_result = await self._check_rate_limits(request, api_key_info)
            if not rate_limit_result["allowed"]:
                return self._create_rate_limit_response(rate_limit_result)
            
            # Phase 5: Check response cache
            cache_key = self._generate_cache_key(request, api_key_info)
            cached_response = await api_management_agent.get_cached_response(cache_key)
            if cached_response:
                response = self._create_cached_response(cached_response)
                await self._record_request_completion(
                    request, api_key_info, response, start_time, from_cache=True
                )
                return response
            
            # Phase 6: Process request through application
            try:
                response = await call_next(request)
            except Exception as e:
                logger.error(f"Application error: {e}")
                response = self._create_error_response({
                    "error": "internal_server_error",
                    "message": "An internal server error occurred"
                }, status_code=500)
            
            # Phase 7: Response post-processing
            response = await self._postprocess_response(
                request, response, api_key_info, cache_key
            )
            
            # Phase 8: Record usage and analytics
            await self._record_request_completion(
                request, api_key_info, response, start_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Middleware error: {e}")
            response = self._create_error_response({
                "error": "middleware_error",
                "message": "Request processing failed"
            }, status_code=500)
            
            # Still try to record the error
            try:
                await self._record_request_completion(
                    request, None, response, start_time, error=str(e)
                )
            except:
                pass  # Don't let recording errors break the response
            
            return response
    
    def _should_skip_middleware(self, request: Request) -> bool:
        """Check if middleware should be skipped for this request"""
        path = request.url.path
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    async def _preprocess_request(self, request: Request) -> Dict[str, Any]:
        """Preprocess and validate incoming request"""
        try:
            # Basic request validation
            if not request.method:
                return {
                    "allowed": False,
                    "error": "invalid_method",
                    "message": "Invalid HTTP method"
                }
            
            # Check request size limits
            content_length = int(request.headers.get("content-length", 0))
            max_request_size = 100 * 1024 * 1024  # 100MB default
            
            if content_length > max_request_size:
                return {
                    "allowed": False,
                    "error": "request_too_large",
                    "message": f"Request size {content_length} exceeds maximum {max_request_size}",
                    "max_size": max_request_size
                }
            
            # Check for required headers
            user_agent = request.headers.get("user-agent")
            if not user_agent:
                logger.warning(f"Request from {request.client.host} missing User-Agent header")
            
            # Validate content type for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                if content_length > 0 and not content_type:
                    return {
                        "allowed": False,
                        "error": "missing_content_type",
                        "message": "Content-Type header required for requests with body"
                    }
            
            # Check for suspicious request patterns
            if self._detect_malicious_request(request):
                await self._log_security_event(
                    "MALICIOUS_REQUEST_DETECTED",
                    request,
                    {"pattern": "request_preprocessing"}
                )
                
                return {
                    "allowed": False,
                    "error": "security_violation",
                    "message": "Request blocked due to security policy"
                }
            
            return {"allowed": True}
            
        except Exception as e:
            logger.error(f"Error in request preprocessing: {e}")
            return {
                "allowed": False,
                "error": "preprocessing_error",
                "message": "Request validation failed"
            }
    
    async def _authenticate_api_key(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate API key from request headers"""
        try:
            # Check for API key in headers
            api_key = None
            
            # Try X-API-Key header
            if "x-api-key" in request.headers:
                api_key = request.headers["x-api-key"]
            # Try Authorization header with Bearer token
            elif "authorization" in request.headers:
                auth_header = request.headers["authorization"]
                if auth_header.startswith("Bearer ") and "sk_" in auth_header:
                    api_key = auth_header[7:]  # Remove "Bearer "
            
            if not api_key:
                return None  # No API key provided
            
            # Validate API key
            api_key_info = await api_management_agent.validate_api_key(api_key)
            if not api_key_info:
                await self._log_security_event(
                    "INVALID_API_KEY",
                    request,
                    {"api_key_prefix": api_key[:12] if len(api_key) > 12 else api_key}
                )
                return None
            
            return api_key_info
            
        except Exception as e:
            logger.error(f"Error authenticating API key: {e}")
            return None
    
    async def _authorize_request(
        self,
        request: Request,
        api_key_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Authorize request based on API key and endpoint access"""
        try:
            path = request.url.path
            method = request.method
            
            # Check if path requires admin access
            if any(path.startswith(admin_path) for admin_path in self.admin_paths):
                if not api_key_info:
                    return {
                        "allowed": False,
                        "error": "authentication_required",
                        "message": "API key required for this endpoint"
                    }
                
                if api_key_info["config"].tier != APIKeyTier.ADMIN:
                    return {
                        "allowed": False,
                        "error": "insufficient_privileges",
                        "message": "Admin API key required for this endpoint"
                    }
            
            # Check scopes if API key is provided
            if api_key_info:
                scopes = api_key_info["scopes"]
                required_scope = self._determine_required_scope(path, method)
                
                if required_scope and required_scope not in scopes and "admin" not in scopes:
                    return {
                        "allowed": False,
                        "error": "insufficient_scope",
                        "message": f"Scope '{required_scope}' required for this endpoint"
                    }
                
                # Check tier-specific endpoint restrictions
                tier_config = api_management_agent.tier_configs[api_key_info["config"].tier]
                allowed_endpoints = tier_config["allowed_endpoints"]
                
                if allowed_endpoints != ["*"]:
                    endpoint_pattern = f"{method} {path}"
                    if not any(
                        endpoint_pattern.startswith(allowed.replace("*", ""))
                        for allowed in allowed_endpoints
                    ):
                        return {
                            "allowed": False,
                            "error": "endpoint_not_allowed",
                            "message": f"Endpoint not allowed for {api_key_info['config'].tier.value} tier"
                        }
            
            return {"allowed": True}
            
        except Exception as e:
            logger.error(f"Error in request authorization: {e}")
            return {
                "allowed": False,
                "error": "authorization_error",
                "message": "Authorization check failed"
            }
    
    async def _check_rate_limits(
        self,
        request: Request,
        api_key_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check rate limits and quotas"""
        try:
            # Use API management agent for comprehensive rate limiting
            allowed, info = await api_management_agent.process_request(request, api_key_info)
            
            if not allowed:
                await self._log_security_event(
                    "RATE_LIMIT_EXCEEDED",
                    request,
                    {
                        "api_key_id": api_key_info["key_id"] if api_key_info else None,
                        "limit_info": info
                    }
                )
            
            return {"allowed": allowed, **info}
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            # Allow request to proceed if rate limiting fails
            return {"allowed": True, "error": "rate_limit_check_failed"}
    
    def _generate_cache_key(
        self,
        request: Request,
        api_key_info: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for request"""
        try:
            # Include method, path, and query parameters
            key_components = [
                request.method,
                request.url.path,
                str(request.query_params)
            ]
            
            # Include user context if available
            if api_key_info:
                key_components.append(f"user:{api_key_info['user_id']}")
            
            # Create hash of components
            key_string = "|".join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"fallback_{int(time.time())}"
    
    async def _postprocess_response(
        self,
        request: Request,
        response: Response,
        api_key_info: Optional[Dict[str, Any]],
        cache_key: str
    ) -> Response:
        """Post-process response before returning to client"""
        try:
            # Add standard API management headers
            response.headers["X-API-Management"] = "SuperNova-AI"
            response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
            response.headers["X-Timestamp"] = datetime.utcnow().isoformat()
            
            # Add rate limit headers if API key is present
            if api_key_info:
                key_id = api_key_info["key_id"]
                config = api_key_info["config"]
                
                response.headers["X-RateLimit-Tier"] = config.tier.value
                
                # Get current usage from rate limiter
                if key_id in api_management_agent.quota_usage:
                    from .api_management_agent import QuotaType
                    request_quota = api_management_agent.quota_usage[key_id].get(QuotaType.REQUESTS)
                    if request_quota:
                        response.headers["X-RateLimit-Limit"] = str(int(request_quota.limit))
                        response.headers["X-RateLimit-Remaining"] = str(
                            int(max(0, request_quota.limit - request_quota.used))
                        )
                        response.headers["X-RateLimit-Reset"] = request_quota.reset_time.isoformat()
            
            # Cache successful responses if appropriate
            if (response.status_code == 200 and 
                request.method == "GET" and
                self._is_cacheable_endpoint(request.url.path)):
                
                try:
                    # Determine cache TTL based on endpoint
                    cache_ttl = self._get_cache_ttl(request.url.path)
                    if cache_ttl > 0:
                        # Extract response data for caching
                        response_data = {
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": None  # Would need to read response body
                        }
                        
                        await api_management_agent.cache_response(
                            cache_key, response_data, cache_ttl
                        )
                        
                        response.headers["X-Cache"] = "MISS"
                        response.headers["X-Cache-TTL"] = str(cache_ttl)
                except Exception as e:
                    logger.error(f"Error caching response: {e}")
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            # Add CORS headers if needed
            if api_key_info and api_key_info["config"].allowed_origins:
                origin = request.headers.get("origin")
                if origin in api_key_info["config"].allowed_origins:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in response post-processing: {e}")
            return response  # Return original response if post-processing fails
    
    def _create_cached_response(self, cached_data: Dict[str, Any]) -> Response:
        """Create response from cached data"""
        try:
            response = JSONResponse(
                content=cached_data["data"].get("content"),
                status_code=cached_data["data"].get("status_code", 200)
            )
            
            # Add original headers
            for key, value in cached_data["data"].get("headers", {}).items():
                response.headers[key] = value
            
            # Add cache headers
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Age"] = str(
                int((datetime.utcnow() - cached_data["created_at"]).total_seconds())
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating cached response: {e}")
            # Return error response if cache data is corrupted
            return self._create_error_response({
                "error": "cache_error",
                "message": "Cached response corrupted"
            }, status_code=500)
    
    async def _record_request_completion(
        self,
        request: Request,
        api_key_info: Optional[Dict[str, Any]],
        response: Response,
        start_time: float,
        from_cache: bool = False,
        error: Optional[str] = None
    ):
        """Record request completion for analytics and monitoring"""
        try:
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Calculate request/response sizes
            request_size = int(request.headers.get("content-length", 0))
            response_size = 0
            
            # Estimate response size (would be more accurate with actual content)
            if hasattr(response, "body"):
                response_size = len(response.body) if response.body else 0
            elif hasattr(response, "content"):
                response_size = len(str(response.content))
            
            # Record usage
            await api_management_agent.record_usage(
                key_id=api_key_info["key_id"] if api_key_info else None,
                user_id=api_key_info["user_id"] if api_key_info else None,
                request=request,
                response_time_ms=processing_time,
                status_code=response.status_code,
                request_size=request_size,
                response_size=response_size,
                error_message=error
            )
            
            # Check for suspicious activity
            from .api_management_agent import UsageRecord
            usage_record = UsageRecord(
                timestamp=datetime.utcnow(),
                api_key_id=api_key_info["key_id"] if api_key_info else None,
                user_id=api_key_info["user_id"] if api_key_info else None,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=processing_time,
                request_size=request_size,
                response_size=response_size,
                ip_address=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent"),
                error_message=error
            )
            
            suspicious_activity = await api_management_agent.detect_suspicious_activity(usage_record)
            if suspicious_activity:
                await self._log_security_event(
                    "SUSPICIOUS_ACTIVITY_DETECTED",
                    request,
                    suspicious_activity
                )
            
            # Log performance issues
            if processing_time > 5000:  # Requests taking longer than 5 seconds
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {processing_time:.0f}ms"
                )
            
        except Exception as e:
            logger.error(f"Error recording request completion: {e}")
    
    def _detect_malicious_request(self, request: Request) -> bool:
        """Detect potentially malicious request patterns"""
        try:
            path = request.url.path.lower()
            query = str(request.query_params).lower()
            
            # SQL injection patterns
            sql_patterns = [
                "union select", "drop table", "insert into", "delete from",
                "'; drop", "1' or '1'='1", "' or 1=1", "' union select"
            ]
            
            if any(pattern in path or pattern in query for pattern in sql_patterns):
                return True
            
            # XSS patterns
            xss_patterns = [
                "<script>", "javascript:", "onload=", "onerror=",
                "alert(", "eval(", "<iframe"
            ]
            
            if any(pattern in path or pattern in query for pattern in xss_patterns):
                return True
            
            # Path traversal patterns
            if "../" in path or "..%2f" in path or "..\\x2f" in path:
                return True
            
            # Suspicious file access patterns
            suspicious_files = [
                ".env", "config.php", "wp-config.php", ".htaccess",
                "backup.sql", "database.sql", "admin.php"
            ]
            
            if any(file in path for file in suspicious_files):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in malicious request detection: {e}")
            return False
    
    def _determine_required_scope(self, path: str, method: str) -> Optional[str]:
        """Determine required scope for endpoint"""
        # Map endpoints to required scopes
        scope_mapping = {
            "/api/advisor": "advisor:read",
            "/api/backtester": "backtesting:read",
            "/api/portfolio": "portfolio:read",
            "/api/market-data": "market:read",
            "/api/management": "management:admin"
        }
        
        for endpoint_prefix, scope in scope_mapping.items():
            if path.startswith(endpoint_prefix):
                if method in ["POST", "PUT", "DELETE", "PATCH"]:
                    return scope.replace(":read", ":write")
                return scope
        
        return None  # No specific scope required
    
    def _is_cacheable_endpoint(self, path: str) -> bool:
        """Check if endpoint responses can be cached"""
        cacheable_patterns = [
            "/api/market-data",
            "/api/indicators",
            "/api/sentiment/history",
            "/api/analytics/performance"
        ]
        
        return any(path.startswith(pattern) for pattern in cacheable_patterns)
    
    def _get_cache_ttl(self, path: str) -> int:
        """Get cache TTL in seconds for endpoint"""
        ttl_mapping = {
            "/api/market-data": 60,      # 1 minute
            "/api/indicators": 300,      # 5 minutes
            "/api/sentiment": 600,       # 10 minutes
            "/api/analytics": 1800       # 30 minutes
        }
        
        for pattern, ttl in ttl_mapping.items():
            if path.startswith(pattern):
                return ttl
        
        return 0  # No caching by default
    
    def _create_error_response(
        self,
        error_info: Dict[str, Any],
        status_code: int = 400
    ) -> JSONResponse:
        """Create standardized error response"""
        return JSONResponse(
            content={
                "error": error_info.get("error", "unknown_error"),
                "message": error_info.get("message", "An error occurred"),
                "timestamp": datetime.utcnow().isoformat(),
                "details": {k: v for k, v in error_info.items() 
                          if k not in ["error", "message", "allowed"]}
            },
            status_code=status_code,
            headers={
                "X-API-Management": "SuperNova-AI",
                "X-Error-Type": error_info.get("error", "unknown_error")
            }
        )
    
    def _create_rate_limit_response(self, limit_info: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response"""
        retry_after = limit_info.get("retry_after", 60)
        
        return JSONResponse(
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": retry_after,
                "limit_details": {k: v for k, v in limit_info.items() 
                                if k not in ["error", "allowed"]},
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "X-RateLimit-Limit": str(limit_info.get("limit", "unknown")),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                "Retry-After": str(retry_after),
                "X-API-Management": "SuperNova-AI"
            }
        )
    
    async def _log_security_event(
        self,
        event_type: str,
        request: Request,
        details: Dict[str, Any]
    ):
        """Log security event"""
        try:
            security_logger.log_security_event(
                event_type=event_type,
                level=SecurityEventLevel.WARNING,
                client_ip=request.client.host if request.client else "unknown",
                endpoint=request.url.path,
                user_agent=request.headers.get("user-agent"),
                details=details
            )
        except Exception as e:
            logger.error(f"Error logging security event: {e}")


# Authentication dependency for API key validation
async def get_api_key_info(request: Request) -> Optional[Dict[str, Any]]:
    """Dependency to extract API key information from request"""
    api_key = None
    
    # Try X-API-Key header
    if "x-api-key" in request.headers:
        api_key = request.headers["x-api-key"]
    # Try Authorization header with Bearer token
    elif "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.startswith("Bearer ") and "sk_" in auth_header:
            api_key = auth_header[7:]  # Remove "Bearer "
    
    if api_key:
        return await api_management_agent.validate_api_key(api_key)
    
    return None


def require_api_key(api_key_info: Optional[Dict[str, Any]] = Depends(get_api_key_info)):
    """Dependency that requires a valid API key"""
    if not api_key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key_info


def require_api_key_tier(required_tier: APIKeyTier):
    """Dependency factory that requires specific API key tier"""
    def check_tier(api_key_info: Dict[str, Any] = Depends(require_api_key)):
        if api_key_info["config"].tier.value < required_tier.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key tier '{required_tier.value}' or higher required"
            )
        return api_key_info
    
    return check_tier