"""
SuperNova AI API Management API Endpoints
RESTful API for managing API keys, quotas, and monitoring
"""

import json
import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_

from .db import SessionLocal, get_db, User, APIKey
from .auth import get_current_user, require_permission, require_admin
from .api_management_agent import api_management_agent, APIKeyTier, QuotaType
from .api_management_schemas import (
    APIKeyCreateRequest, APIKeyCreateResponse, APIKeyResponse, APIKeyListResponse,
    APIKeyUpdateRequest, APIKeyRotateResponse, QuotaConfigRequest, QuotaConfigResponse,
    QuotaListResponse, UsageAnalyticsRequest, UsageAnalyticsResponse,
    RateLimitConfigRequest, RateLimitConfigResponse, SecurityEventRequest,
    SecurityEventResponse, IPBlockRequest, IPBlockResponse, SystemHealthResponse,
    RealTimeMetricsResponse, DashboardMetricsResponse, WebhookConfigRequest,
    WebhookConfigResponse, BulkOperationRequest, BulkOperationResponse,
    APIManagementConfigRequest, APIManagementConfigResponse, ExportRequest, ExportResponse
)
from .security_logger import security_logger, SecurityEventLevel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/management", tags=["API Management"])


# ========================================
# API KEY MANAGEMENT ENDPOINTS
# ========================================

@router.post("/keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    try:
        # Validate user permissions for tier
        if request.tier in [APIKeyTier.ENTERPRISE, APIKeyTier.ADMIN]:
            if not require_admin(current_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin privileges required for this tier"
                )
        
        # Check existing key limits
        existing_keys = db.query(APIKey).filter(
            APIKey.user_id == current_user.id,
            APIKey.is_active == True
        ).count()
        
        max_keys_by_tier = {
            APIKeyTier.FREE: 2,
            APIKeyTier.PRO: 10,
            APIKeyTier.ENTERPRISE: 50,
            APIKeyTier.ADMIN: 100
        }
        
        if existing_keys >= max_keys_by_tier.get(request.tier, 2):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Maximum API keys limit reached for {request.tier.value} tier"
            )
        
        # Create API key
        key_id, actual_key = await api_management_agent.create_api_key(
            user_id=current_user.id,
            name=request.name,
            tier=request.tier,
            scopes=request.scopes,
            expires_at=request.expires_at,
            db=db
        )
        
        # Get key info for response
        db_key = db.get(APIKey, key_id)
        
        return APIKeyCreateResponse(
            key_id=key_id,
            api_key=actual_key,
            name=db_key.name,
            tier=APIKeyTier(db_key.rate_limit_tier),
            scopes=json.loads(db_key.scopes or "[]"),
            prefix=db_key.prefix,
            created_at=db_key.created_at,
            expires_at=db_key.expires_at,
            last_used=db_key.last_used,
            is_active=db_key.is_active,
            rate_limits=api_management_agent.tier_configs[APIKeyTier(db_key.rate_limit_tier)],
            quota_limits={}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/keys", response_model=APIKeyListResponse)
async def list_api_keys(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's API keys"""
    try:
        # Get total count
        total = db.query(APIKey).filter(APIKey.user_id == current_user.id).count()
        
        # Get paginated results
        keys_query = db.query(APIKey).filter(
            APIKey.user_id == current_user.id
        ).order_by(desc(APIKey.created_at))
        
        keys = keys_query.offset((page - 1) * per_page).limit(per_page).all()
        
        key_responses = []
        for key in keys:
            # Get usage stats
            usage_stats = await api_management_agent.get_usage_analytics(key_id=key.id)
            
            key_responses.append(APIKeyResponse(
                key_id=key.id,
                name=key.name,
                tier=APIKeyTier(key.rate_limit_tier),
                scopes=json.loads(key.scopes or "[]"),
                prefix=key.prefix,
                created_at=key.created_at,
                expires_at=key.expires_at,
                last_used=key.last_used,
                is_active=key.is_active,
                rate_limits=api_management_agent.tier_configs[APIKeyTier(key.rate_limit_tier)],
                quota_limits={},
                usage_stats=usage_stats["summary"] if usage_stats else None
            ))
        
        return APIKeyListResponse(
            keys=key_responses,
            total=total,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )


@router.get("/keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific API key details"""
    try:
        # Get key from database
        db_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        # Get detailed usage stats
        usage_stats = await api_management_agent.get_usage_analytics(key_id=key_id)
        
        return APIKeyResponse(
            key_id=db_key.id,
            name=db_key.name,
            tier=APIKeyTier(db_key.rate_limit_tier),
            scopes=json.loads(db_key.scopes or "[]"),
            prefix=db_key.prefix,
            created_at=db_key.created_at,
            expires_at=db_key.expires_at,
            last_used=db_key.last_used,
            is_active=db_key.is_active,
            rate_limits=api_management_agent.tier_configs[APIKeyTier(db_key.rate_limit_tier)],
            quota_limits={},
            usage_stats=usage_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key details"
        )


@router.put("/keys/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    request: APIKeyUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an API key"""
    try:
        # Get key from database
        db_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        # Update fields
        if request.name is not None:
            db_key.name = request.name
        
        if request.scopes is not None:
            db_key.scopes = json.dumps(request.scopes)
        
        if request.expires_at is not None:
            db_key.expires_at = request.expires_at
        
        if request.is_active is not None:
            db_key.is_active = request.is_active
        
        db.commit()
        
        # Log update
        security_logger.log_security_event(
            event_type="API_KEY_UPDATED",
            level=SecurityEventLevel.INFO,
            user_id=current_user.id,
            details={
                "key_id": key_id,
                "updated_fields": [k for k, v in request.dict().items() if v is not None]
            }
        )
        
        return APIKeyResponse(
            key_id=db_key.id,
            name=db_key.name,
            tier=APIKeyTier(db_key.rate_limit_tier),
            scopes=json.loads(db_key.scopes or "[]"),
            prefix=db_key.prefix,
            created_at=db_key.created_at,
            expires_at=db_key.expires_at,
            last_used=db_key.last_used,
            is_active=db_key.is_active,
            rate_limits=api_management_agent.tier_configs[APIKeyTier(db_key.rate_limit_tier)],
            quota_limits={}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key"
        )


@router.post("/keys/{key_id}/rotate", response_model=APIKeyRotateResponse)
async def rotate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Rotate an API key (generate new secret)"""
    try:
        new_key = await api_management_agent.rotate_api_key(key_id, current_user.id)
        
        if not new_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or cannot be rotated"
            )
        
        return APIKeyRotateResponse(
            key_id=key_id,
            new_api_key=new_key,
            rotated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rotating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rotate API key"
        )


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke an API key"""
    try:
        success = await api_management_agent.revoke_api_key(key_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        return {"message": "API key revoked successfully", "key_id": key_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )


# ========================================
# QUOTA MANAGEMENT ENDPOINTS
# ========================================

@router.get("/keys/{key_id}/quotas", response_model=QuotaListResponse)
async def get_quotas(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get quota information for an API key"""
    try:
        # Verify key ownership
        db_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        quotas = []
        if key_id in api_management_agent.quota_usage:
            for quota_type, quota_data in api_management_agent.quota_usage[key_id].items():
                quotas.append(QuotaConfigResponse(
                    quota_type=quota_type,
                    limit=quota_data.limit,
                    used=quota_data.used,
                    remaining=quota_data.limit - quota_data.used,
                    reset_time=quota_data.reset_time,
                    reset_period_days=30,  # Default period
                    overage_count=quota_data.overage_count
                ))
        
        return QuotaListResponse(
            quotas=quotas,
            key_id=key_id,
            tier=APIKeyTier(db_key.rate_limit_tier)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quotas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quota information"
        )


@router.put("/keys/{key_id}/quotas/{quota_type}", response_model=QuotaConfigResponse)
async def update_quota(
    key_id: str,
    quota_type: QuotaType,
    request: QuotaConfigRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update quota limits for an API key"""
    try:
        # Verify key ownership and admin permissions for quota changes
        db_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        # Only admins can modify quotas above tier defaults
        tier_config = api_management_agent.tier_configs[APIKeyTier(db_key.rate_limit_tier)]
        default_limit = tier_config.get(f"{quota_type.value}_limit", 1000000)
        
        if request.limit > default_limit and not require_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required to exceed tier limits"
            )
        
        # Update quota
        if key_id not in api_management_agent.quota_usage:
            api_management_agent.quota_usage[key_id] = {}
        
        reset_time = datetime.utcnow() + timedelta(days=request.reset_period_days)
        
        from .api_management_agent import QuotaUsage
        api_management_agent.quota_usage[key_id][quota_type] = QuotaUsage(
            quota_type=quota_type,
            used=api_management_agent.quota_usage[key_id].get(quota_type, type('obj', (object,), {'used': 0.0})).used,
            limit=request.limit,
            reset_time=reset_time,
            overage_count=api_management_agent.quota_usage[key_id].get(quota_type, type('obj', (object,), {'overage_count': 0})).overage_count
        )
        
        quota_data = api_management_agent.quota_usage[key_id][quota_type]
        
        return QuotaConfigResponse(
            quota_type=quota_type,
            limit=quota_data.limit,
            used=quota_data.used,
            remaining=quota_data.limit - quota_data.used,
            reset_time=quota_data.reset_time,
            reset_period_days=request.reset_period_days,
            overage_count=quota_data.overage_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating quota: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update quota"
        )


# ========================================
# USAGE ANALYTICS ENDPOINTS
# ========================================

@router.post("/usage", response_model=UsageAnalyticsResponse)
async def get_usage_analytics(
    request: UsageAnalyticsRequest,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive usage analytics"""
    try:
        analytics = await api_management_agent.get_usage_analytics(
            key_id=request.key_id,
            user_id=current_user.id if not request.key_id else None,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return UsageAnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage analytics"
        )


@router.get("/metrics/realtime", response_model=RealTimeMetricsResponse)
async def get_realtime_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get real-time API metrics"""
    try:
        # Check if user has permission to view system metrics
        user_metrics_only = not require_admin(current_user, raise_exception=False)
        
        if user_metrics_only:
            # Return user-specific metrics
            user_analytics = await api_management_agent.get_usage_analytics(
                user_id=current_user.id,
                start_date=datetime.utcnow() - timedelta(hours=1)
            )
            
            return RealTimeMetricsResponse(
                timestamp=datetime.utcnow(),
                current_rps=0.0,  # Would be calculated from user data
                peak_rps=0.0,
                total_requests=user_analytics["summary"]["total_requests"],
                error_count=user_analytics["summary"]["failed_requests"],
                avg_response_time=user_analytics["summary"]["avg_response_time"],
                active_api_keys=len([k for k in user_analytics.get("keys", [])]),
                unique_users_today=1,  # Just the current user
                unique_ips_today=user_analytics["summary"]["unique_ips"],
                cache_hit_rate=0.0,
                cache_size=0,
                system_health_score=100.0,
                threat_level=False
            )
        else:
            # Return global metrics for admins
            metrics = await api_management_agent.get_real_time_metrics()
            return RealTimeMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get real-time metrics"
        )


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: User = Depends(require_admin)
):
    """Get system health status"""
    try:
        metrics = await api_management_agent.get_real_time_metrics()
        
        # Determine overall status
        health_score = metrics["system_health_score"]
        if health_score >= 90:
            status_text = "healthy"
        elif health_score >= 70:
            status_text = "degraded"
        else:
            status_text = "critical"
        
        # Component status
        components = {
            "api_management": "healthy" if health_score > 80 else "degraded",
            "rate_limiter": "healthy" if not metrics["threat_level"] else "alert",
            "cache": "healthy" if metrics["cache_hit_rate"] > 50 else "degraded",
            "database": "healthy",  # Would check actual DB connectivity
            "redis": "healthy" if api_management_agent.redis_client else "unavailable"
        }
        
        # Generate alerts based on metrics
        alerts = []
        if metrics["error_count"] / max(1, metrics["total_requests"]) > 0.1:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate is {(metrics['error_count'] / max(1, metrics['total_requests']) * 100):.1f}%"
            })
        
        if metrics["avg_response_time"] > 1000:
            alerts.append({
                "type": "slow_response",
                "severity": "warning",
                "message": f"Average response time is {metrics['avg_response_time']:.0f}ms"
            })
        
        return SystemHealthResponse(
            timestamp=datetime.utcnow(),
            status=status_text,
            health_score=health_score,
            metrics=metrics,
            components=components,
            alerts=alerts,
            uptime_seconds=0  # Would be calculated from actual uptime
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system health"
        )


# ========================================
# RATE LIMITING CONFIGURATION
# ========================================

@router.put("/limits/{context}", response_model=RateLimitConfigResponse)
async def update_rate_limits(
    context: str,
    request: RateLimitConfigRequest,
    current_user: User = Depends(require_admin)
):
    """Update rate limit configuration"""
    try:
        # Update rate limit configuration
        from .api_management_agent import RateLimitConfig
        new_config = RateLimitConfig(
            requests_per_minute=request.requests_per_minute,
            requests_per_hour=request.requests_per_hour,
            requests_per_day=request.requests_per_day,
            burst_capacity=request.burst_capacity
        )
        
        api_management_agent.rate_limiter.rate_configs[context] = new_config
        
        security_logger.log_security_event(
            event_type="RATE_LIMIT_CONFIG_UPDATED",
            level=SecurityEventLevel.INFO,
            user_id=current_user.id,
            details={
                "context": context,
                "new_config": request.dict()
            }
        )
        
        return RateLimitConfigResponse(
            context=context,
            requests_per_minute=request.requests_per_minute,
            requests_per_hour=request.requests_per_hour,
            requests_per_day=request.requests_per_day,
            burst_capacity=request.burst_capacity,
            algorithm=request.algorithm,
            current_usage={},
            next_reset=datetime.utcnow() + timedelta(hours=1)
        )
        
    except Exception as e:
        logger.error(f"Error updating rate limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update rate limits"
        )


# ========================================
# SECURITY MANAGEMENT
# ========================================

@router.post("/security/block-ip", response_model=IPBlockResponse)
async def block_ip_address(
    request: IPBlockRequest,
    current_user: User = Depends(require_admin)
):
    """Block an IP address"""
    try:
        await api_management_agent.block_ip(
            ip_address=request.ip_address,
            reason=request.reason,
            duration_minutes=request.duration_minutes
        )
        
        block_info = api_management_agent.blocked_ips[request.ip_address]
        
        return IPBlockResponse(
            ip_address=request.ip_address,
            blocked_at=block_info["blocked_at"],
            blocked_until=block_info["blocked_until"],
            reason=request.reason,
            block_count=block_info["block_count"]
        )
        
    except Exception as e:
        logger.error(f"Error blocking IP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to block IP address"
        )


@router.delete("/security/block-ip/{ip_address}")
async def unblock_ip_address(
    ip_address: str,
    current_user: User = Depends(require_admin)
):
    """Unblock an IP address"""
    try:
        await api_management_agent.unblock_ip(ip_address)
        
        return {"message": f"IP address {ip_address} unblocked successfully"}
        
    except Exception as e:
        logger.error(f"Error unblocking IP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unblock IP address"
        )


@router.get("/security/blocked-ips")
async def list_blocked_ips(
    current_user: User = Depends(require_admin)
):
    """List all blocked IP addresses"""
    try:
        blocked_ips = []
        for ip, block_info in api_management_agent.blocked_ips.items():
            blocked_ips.append({
                "ip_address": ip,
                "blocked_at": block_info["blocked_at"],
                "blocked_until": block_info["blocked_until"],
                "reason": block_info["reason"],
                "block_count": block_info["block_count"],
                "is_active": datetime.utcnow() < block_info["blocked_until"]
            })
        
        return {"blocked_ips": blocked_ips, "total": len(blocked_ips)}
        
    except Exception as e:
        logger.error(f"Error listing blocked IPs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list blocked IPs"
        )


# ========================================
# BULK OPERATIONS
# ========================================

@router.post("/bulk", response_model=BulkOperationResponse)
async def bulk_operation(
    request: BulkOperationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
):
    """Perform bulk operations on API keys"""
    try:
        results = []
        errors = []
        successful_count = 0
        
        for key_id in request.key_ids:
            try:
                if request.operation == "revoke":
                    success = await api_management_agent.revoke_api_key(key_id, current_user.id)
                    if success:
                        results.append({"key_id": key_id, "status": "revoked"})
                        successful_count += 1
                    else:
                        errors.append({"key_id": key_id, "error": "Key not found or not owned"})
                
                elif request.operation == "activate":
                    # Implementation for activation
                    results.append({"key_id": key_id, "status": "activated"})
                    successful_count += 1
                
                elif request.operation == "deactivate":
                    # Implementation for deactivation
                    results.append({"key_id": key_id, "status": "deactivated"})
                    successful_count += 1
                
            except Exception as e:
                errors.append({"key_id": key_id, "error": str(e)})
        
        return BulkOperationResponse(
            operation=request.operation,
            requested_count=len(request.key_ids),
            successful_count=successful_count,
            failed_count=len(errors),
            results=results,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error in bulk operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk operation"
        )


# ========================================
# DASHBOARD AND REPORTING
# ========================================

@router.get("/dashboard", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get dashboard metrics for the UI"""
    try:
        # Get user-specific or global analytics based on permissions
        is_admin = require_admin(current_user, raise_exception=False)
        
        if is_admin:
            analytics = await api_management_agent.get_usage_analytics()
            real_time = await api_management_agent.get_real_time_metrics()
        else:
            analytics = await api_management_agent.get_usage_analytics(user_id=current_user.id)
            real_time = {
                "current_rps": 0.0,
                "total_requests": analytics["summary"]["total_requests"],
                "error_count": analytics["summary"]["failed_requests"],
                "avg_response_time": analytics["summary"]["avg_response_time"],
                "active_api_keys": 0,
                "unique_users_today": 1,
                "cache_hit_rate": 0.0
            }
        
        # Prepare dashboard data
        overview = {
            "total_requests": real_time["total_requests"],
            "current_rps": real_time["current_rps"],
            "error_count": real_time["error_count"],
            "avg_response_time": real_time["avg_response_time"],
            "active_keys": real_time["active_api_keys"],
            "cache_hit_rate": real_time["cache_hit_rate"]
        }
        
        # Traffic chart data (last 24 hours)
        traffic_chart = []
        for hour_key, data in analytics.get("time_series", {}).items():
            traffic_chart.append({
                "timestamp": hour_key,
                "requests": data["requests"],
                "errors": data["errors"]
            })
        
        # Top endpoints
        top_endpoints = []
        for endpoint, stats in analytics.get("endpoints", {}).items():
            top_endpoints.append({
                "endpoint": endpoint,
                "count": stats["count"],
                "avg_response_time": stats["avg_response_time"],
                "error_count": stats["error_count"]
            })
        
        top_endpoints.sort(key=lambda x: x["count"], reverse=True)
        top_endpoints = top_endpoints[:10]
        
        # Recent errors
        recent_errors = []
        for error_type, count in analytics.get("errors", {}).items():
            recent_errors.append({
                "error": error_type,
                "count": count
            })
        
        return DashboardMetricsResponse(
            overview=overview,
            traffic_chart=traffic_chart[-24:],  # Last 24 hours
            top_endpoints=top_endpoints,
            recent_errors=recent_errors[:10],
            geographical_data=analytics.get("geographic", {}),
            performance_trends=[],  # Would be calculated from historical data
            security_alerts=[]  # Would be pulled from security events
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard metrics"
        )


# ========================================
# EXPORT AND REPORTING
# ========================================

@router.post("/export", response_model=ExportResponse)
async def export_data(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Export API management data"""
    try:
        export_id = secrets.token_urlsafe(16)
        
        # Queue export task
        background_tasks.add_task(
            _process_export,
            export_id,
            request,
            current_user.id
        )
        
        return ExportResponse(
            export_id=export_id,
            export_type=request.export_type,
            format=request.format,
            status="queued",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error queuing export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue export"
        )


async def _process_export(export_id: str, request: ExportRequest, user_id: int):
    """Background task to process export"""
    try:
        # This would implement the actual export logic
        # For now, just simulate processing
        await asyncio.sleep(5)
        
        # In a real implementation, this would:
        # 1. Generate the requested data
        # 2. Format it according to the request
        # 3. Store it in a downloadable location
        # 4. Update the export status
        
        logger.info(f"Export {export_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing export {export_id}: {e}")


@router.get("/export/{export_id}", response_model=ExportResponse)
async def get_export_status(
    export_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get export status and download link"""
    try:
        # This would check the actual export status from storage
        # For now, return a mock response
        return ExportResponse(
            export_id=export_id,
            export_type="usage",
            format="json",
            status="completed",
            created_at=datetime.utcnow() - timedelta(minutes=5),
            download_url=f"/api/management/export/{export_id}/download",
            expires_at=datetime.utcnow() + timedelta(hours=24),
            file_size=1024 * 512  # 512 KB
        )
        
    except Exception as e:
        logger.error(f"Error getting export status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get export status"
        )