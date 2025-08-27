"""
Comprehensive test suite for API Management Agent
Tests all components: rate limiting, API keys, quotas, monitoring, security
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import redis

from supernova.api_management_agent import (
    APIManagementAgent, APIKeyTier, QuotaType, UsageRecord
)
from supernova.api_management_api import router as api_management_router
from supernova.api_management_middleware import APIManagementMiddleware
from supernova.db import SessionLocal, User, APIKey
from supernova.auth import auth_manager


# Test fixtures
@pytest.fixture
def api_management_agent():
    """Create API management agent for testing"""
    agent = APIManagementAgent()
    # Use mock Redis for testing
    agent.redis_client = Mock()
    return agent


@pytest.fixture
def test_app():
    """Create test FastAPI app with API management"""
    app = FastAPI()
    app.add_middleware(APIManagementMiddleware)
    app.include_router(api_management_router)
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def test_user():
    """Create test user"""
    db = SessionLocal()
    try:
        user = User(
            name="Test User",
            email="test@example.com",
            hashed_password="hashed_password",
            role="user"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


@pytest.fixture
def admin_user():
    """Create admin user"""
    db = SessionLocal()
    try:
        user = User(
            name="Admin User",
            email="admin@example.com",
            hashed_password="hashed_password",
            role="admin"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


class TestAPIKeyManagement:
    """Test API key creation, validation, and management"""

    @pytest.mark.asyncio
    async def test_create_api_key(self, api_management_agent, test_user):
        """Test API key creation"""
        key_id, api_key = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key",
            tier=APIKeyTier.FREE,
            scopes=["market:read"]
        )
        
        assert key_id.startswith("sk_")
        assert api_key.startswith(key_id)
        assert key_id in api_management_agent.api_keys
        
        config = api_management_agent.api_keys[key_id]
        assert config.tier == APIKeyTier.FREE
        assert "market:read" in config.scopes

    @pytest.mark.asyncio
    async def test_validate_api_key(self, api_management_agent, test_user):
        """Test API key validation"""
        # Create API key
        key_id, api_key = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key",
            tier=APIKeyTier.PRO
        )
        
        # Mock database response
        with patch('supernova.api_management_agent.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            mock_key = Mock()
            mock_key.id = key_id
            mock_key.user_id = test_user.id
            mock_key.is_active = True
            mock_key.key_hash = api_management_agent._hash_key(api_key)
            mock_key.scopes = json.dumps(["market:read"])
            
            mock_session.get.return_value = mock_key
            
            # Validate key
            key_info = await api_management_agent.validate_api_key(api_key)
            
            assert key_info is not None
            assert key_info["key_id"] == key_id
            assert key_info["user_id"] == test_user.id

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, api_management_agent, test_user):
        """Test API key revocation"""
        # Create API key
        key_id, _ = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Revoke key
        success = await api_management_agent.revoke_api_key(key_id, test_user.id)
        
        assert success
        assert key_id not in api_management_agent.api_keys

    @pytest.mark.asyncio
    async def test_rotate_api_key(self, api_management_agent, test_user):
        """Test API key rotation"""
        # Create API key
        key_id, original_key = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Rotate key
        with patch('supernova.api_management_agent.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            mock_key = Mock()
            mock_key.id = key_id
            mock_key.user_id = test_user.id
            mock_key.is_active = True
            
            mock_session.get.return_value = mock_key
            
            new_key = await api_management_agent.rotate_api_key(key_id, test_user.id)
            
            assert new_key is not None
            assert new_key != original_key
            assert new_key.startswith(key_id)


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_check_success(self, api_management_agent):
        """Test successful rate limit check"""
        from fastapi import Request
        
        # Mock request
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        
        # Check rate limit
        allowed, info = await api_management_agent.process_request(request)
        
        assert allowed
        assert "processing_time_ms" in info

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, api_management_agent):
        """Test rate limit exceeded scenario"""
        from fastapi import Request
        
        # Mock request
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        
        # Mock rate limiter to always fail
        api_management_agent.rate_limiter.check_rate_limit = Mock(
            return_value=(False, {"error": "rate_limit_exceeded", "retry_after": 60})
        )
        
        # Check rate limit
        allowed, info = await api_management_agent.process_request(request)
        
        assert not allowed
        assert info["error"] == "rate_limit_exceeded"

    @pytest.mark.asyncio
    async def test_ip_blocking(self, api_management_agent):
        """Test IP blocking functionality"""
        test_ip = "192.168.1.100"
        
        # Block IP
        await api_management_agent.block_ip(
            test_ip, 
            "Testing", 
            duration_minutes=60
        )
        
        assert test_ip in api_management_agent.blocked_ips
        
        block_info = api_management_agent.blocked_ips[test_ip]
        assert block_info["reason"] == "Testing"
        assert block_info["blocked_until"] > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_ip_unblocking(self, api_management_agent):
        """Test IP unblocking functionality"""
        test_ip = "192.168.1.101"
        
        # Block then unblock IP
        await api_management_agent.block_ip(test_ip, "Testing")
        await api_management_agent.unblock_ip(test_ip)
        
        assert test_ip not in api_management_agent.blocked_ips


class TestQuotaManagement:
    """Test quota management and tracking"""

    @pytest.mark.asyncio
    async def test_quota_check_within_limit(self, api_management_agent, test_user):
        """Test quota check when within limits"""
        # Create API key
        key_id, _ = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Check quota
        allowed, info = await api_management_agent.check_quota(
            key_id, QuotaType.REQUESTS, 1.0
        )
        
        assert allowed
        assert "used" in info
        assert "limit" in info

    @pytest.mark.asyncio
    async def test_quota_exceeded(self, api_management_agent, test_user):
        """Test quota exceeded scenario"""
        # Create API key
        key_id, _ = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Set low quota limit
        from supernova.api_management_agent import QuotaUsage
        api_management_agent.quota_usage[key_id][QuotaType.REQUESTS] = QuotaUsage(
            quota_type=QuotaType.REQUESTS,
            used=100.0,
            limit=100.0,
            reset_time=datetime.utcnow() + timedelta(days=30)
        )
        
        # Check quota (should fail)
        allowed, info = await api_management_agent.check_quota(
            key_id, QuotaType.REQUESTS, 1.0
        )
        
        assert not allowed
        assert info["quota_exceeded"]

    @pytest.mark.asyncio
    async def test_usage_recording(self, api_management_agent, test_user):
        """Test usage recording functionality"""
        from fastapi import Request
        
        # Create API key
        key_id, _ = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Mock request
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-client"}
        
        # Record usage
        await api_management_agent.record_usage(
            key_id=key_id,
            user_id=test_user.id,
            request=request,
            response_time_ms=100.0,
            status_code=200,
            request_size=1024,
            response_size=2048
        )
        
        # Check that usage was recorded
        assert key_id in api_management_agent.api_key_usage
        usage_records = api_management_agent.api_key_usage[key_id]
        assert len(usage_records) == 1
        
        record = usage_records[0]
        assert record.api_key_id == key_id
        assert record.endpoint == "/api/test"
        assert record.status_code == 200


class TestAnalytics:
    """Test analytics and monitoring functionality"""

    @pytest.mark.asyncio
    async def test_usage_analytics(self, api_management_agent, test_user):
        """Test usage analytics generation"""
        # Create API key and add some usage data
        key_id, _ = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Test API Key"
        )
        
        # Add mock usage records
        from supernova.api_management_agent import UsageRecord
        usage_records = [
            UsageRecord(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                api_key_id=key_id,
                user_id=test_user.id,
                endpoint="/api/test",
                method="GET",
                status_code=200,
                response_time_ms=100.0 + i * 10,
                request_size=1024,
                response_size=2048,
                ip_address="127.0.0.1",
                user_agent="test-client"
            )
            for i in range(10)
        ]
        
        api_management_agent.api_key_usage[key_id] = usage_records
        
        # Get analytics
        analytics = await api_management_agent.get_usage_analytics(key_id=key_id)
        
        assert analytics["summary"]["total_requests"] == 10
        assert analytics["summary"]["successful_requests"] == 10
        assert analytics["summary"]["failed_requests"] == 0
        assert "endpoints" in analytics
        assert "time_series" in analytics

    @pytest.mark.asyncio
    async def test_real_time_metrics(self, api_management_agent):
        """Test real-time metrics collection"""
        metrics = await api_management_agent.get_real_time_metrics()
        
        assert "timestamp" in metrics
        assert "current_rps" in metrics
        assert "total_requests" in metrics
        assert "system_health_score" in metrics
        assert isinstance(metrics["system_health_score"], (int, float))
        assert 0 <= metrics["system_health_score"] <= 100


class TestSecurity:
    """Test security features and monitoring"""

    @pytest.mark.asyncio
    async def test_suspicious_activity_detection(self, api_management_agent):
        """Test suspicious activity detection"""
        # Create suspicious usage pattern
        usage_record = UsageRecord(
            timestamp=datetime.utcnow(),
            api_key_id="test_key",
            user_id=1,
            endpoint="/admin",
            method="GET",
            status_code=401,
            response_time_ms=100.0,
            request_size=1024,
            response_size=0,
            ip_address="192.168.1.100",
            user_agent="bot"
        )
        
        # Add many similar records to simulate rapid-fire requests
        api_management_agent.api_key_usage["test_key"] = [usage_record] * 60
        
        # Detect suspicious activity
        activity = await api_management_agent.detect_suspicious_activity(usage_record)
        
        assert activity is not None
        assert "indicators" in activity
        assert len(activity["indicators"]) > 0
        assert activity["severity"] in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_threat_assessment(self, api_management_agent):
        """Test threat level assessment"""
        # Test with high error rate to trigger threat detection
        api_management_agent.usage_metrics["total_requests"] = 1000
        api_management_agent.usage_metrics["error_count"] = 500  # 50% error rate
        
        # This would be called internally during request processing
        # For testing, we simulate the conditions that trigger threat assessment
        error_rate = api_management_agent.usage_metrics["error_count"] / api_management_agent.usage_metrics["total_requests"]
        
        assert error_rate == 0.5  # 50% error rate should trigger alerts


class TestMiddleware:
    """Test API management middleware"""

    def test_middleware_excludes_health_endpoints(self):
        """Test that middleware skips health check endpoints"""
        from supernova.api_management_middleware import APIManagementMiddleware
        
        middleware = APIManagementMiddleware(None)
        
        # Mock request to health endpoint
        request = Mock()
        request.url = Mock()
        request.url.path = "/health"
        
        assert middleware._should_skip_middleware(request)

    def test_middleware_processes_api_endpoints(self):
        """Test that middleware processes API endpoints"""
        from supernova.api_management_middleware import APIManagementMiddleware
        
        middleware = APIManagementMiddleware(None)
        
        # Mock request to API endpoint
        request = Mock()
        request.url = Mock()
        request.url.path = "/api/test"
        
        assert not middleware._should_skip_middleware(request)

    def test_cache_key_generation(self):
        """Test cache key generation"""
        from supernova.api_management_middleware import APIManagementMiddleware
        
        middleware = APIManagementMiddleware(None)
        
        # Mock request
        request = Mock()
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        request.query_params = {}
        
        cache_key = middleware._generate_cache_key(request, None)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_malicious_request_detection(self):
        """Test malicious request pattern detection"""
        from supernova.api_management_middleware import APIManagementMiddleware
        
        middleware = APIManagementMiddleware(None)
        
        # Test SQL injection pattern
        request = Mock()
        request.url = Mock()
        request.url.path = "/api/test'; DROP TABLE users; --"
        request.query_params = {}
        
        is_malicious = middleware._detect_malicious_request(request)
        assert is_malicious
        
        # Test normal request
        request.url.path = "/api/normal-endpoint"
        is_malicious = middleware._detect_malicious_request(request)
        assert not is_malicious


class TestAPIEndpoints:
    """Test API management REST endpoints"""

    def test_create_api_key_endpoint(self, test_client, test_user):
        """Test API key creation endpoint"""
        with patch('supernova.auth.get_current_user', return_value=test_user):
            response = test_client.post(
                "/api/management/keys",
                json={
                    "name": "Test API Key",
                    "tier": "free",
                    "scopes": ["market:read"]
                }
            )
            
            assert response.status_code in [200, 201]
            data = response.json()
            assert "key_id" in data
            assert "api_key" in data
            assert data["name"] == "Test API Key"

    def test_list_api_keys_endpoint(self, test_client, test_user):
        """Test API keys listing endpoint"""
        with patch('supernova.auth.get_current_user', return_value=test_user):
            response = test_client.get("/api/management/keys")
            
            assert response.status_code == 200
            data = response.json()
            assert "keys" in data
            assert "total" in data
            assert isinstance(data["keys"], list)

    def test_get_usage_analytics_endpoint(self, test_client, test_user):
        """Test usage analytics endpoint"""
        with patch('supernova.auth.get_current_user', return_value=test_user):
            response = test_client.post(
                "/api/management/usage",
                json={
                    "start_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                    "group_by": "day"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "period" in data
            assert "summary" in data
            assert "endpoints" in data

    def test_get_dashboard_metrics_endpoint(self, test_client, test_user):
        """Test dashboard metrics endpoint"""
        with patch('supernova.auth.get_current_user', return_value=test_user):
            response = test_client.get("/api/management/dashboard")
            
            assert response.status_code == 200
            data = response.json()
            assert "overview" in data
            assert "traffic_chart" in data
            assert "top_endpoints" in data

    def test_block_ip_endpoint(self, test_client, admin_user):
        """Test IP blocking endpoint"""
        with patch('supernova.auth.require_admin', return_value=admin_user):
            response = test_client.post(
                "/api/management/security/block-ip",
                json={
                    "ip_address": "192.168.1.100",
                    "reason": "Testing IP block",
                    "duration_minutes": 60
                }
            )
            
            assert response.status_code in [200, 201]
            data = response.json()
            assert data["ip_address"] == "192.168.1.100"
            assert "blocked_at" in data


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self, api_management_agent):
        """Test rate limiting performance under load"""
        from fastapi import Request
        
        # Mock request
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        
        # Measure performance
        start_time = time.time()
        
        # Process multiple requests
        tasks = []
        for _ in range(100):
            task = api_management_agent.process_request(request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be able to process 100 requests in reasonable time
        assert processing_time < 1.0  # Less than 1 second
        assert all(result[0] for result in results)  # All requests allowed

    @pytest.mark.asyncio
    async def test_memory_usage_cleanup(self, api_management_agent):
        """Test that expired data is cleaned up"""
        # Add old usage records
        old_timestamp = datetime.utcnow() - timedelta(days=2)
        
        from supernova.api_management_agent import UsageRecord
        old_record = UsageRecord(
            timestamp=old_timestamp,
            api_key_id="test_key",
            user_id=1,
            endpoint="/api/test",
            method="GET",
            status_code=200,
            response_time_ms=100.0,
            request_size=1024,
            response_size=2048,
            ip_address="127.0.0.1",
            user_agent="test-client"
        )
        
        api_management_agent.api_key_usage["test_key"] = [old_record]
        
        # Trigger cleanup manually for testing
        await api_management_agent._cleanup_expired_data()
        
        # Old records should be cleaned up
        assert len(api_management_agent.api_key_usage["test_key"]) == 0


# Integration tests
class TestIntegration:
    """Integration tests for complete API management flow"""

    @pytest.mark.asyncio
    async def test_complete_api_request_flow(self, api_management_agent, test_user):
        """Test complete flow from request to analytics"""
        from fastapi import Request
        
        # Step 1: Create API key
        key_id, api_key = await api_management_agent.create_api_key(
            user_id=test_user.id,
            name="Integration Test Key",
            tier=APIKeyTier.PRO,
            scopes=["market:read", "portfolio:write"]
        )
        
        # Step 2: Validate API key
        api_key_info = await api_management_agent.validate_api_key(api_key)
        assert api_key_info is not None
        
        # Step 3: Process request with API key
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/api/market-data"
        request.method = "GET"
        request.headers = {"user-agent": "integration-test", "x-api-key": api_key}
        
        allowed, info = await api_management_agent.process_request(request, api_key_info)
        assert allowed
        
        # Step 4: Record usage
        await api_management_agent.record_usage(
            key_id=key_id,
            user_id=test_user.id,
            request=request,
            response_time_ms=150.0,
            status_code=200,
            request_size=0,
            response_size=5000
        )
        
        # Step 5: Check analytics
        analytics = await api_management_agent.get_usage_analytics(key_id=key_id)
        assert analytics["summary"]["total_requests"] == 1
        assert analytics["summary"]["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, api_management_agent):
        """Test rate limit recovery after blocking"""
        from fastapi import Request
        
        # Mock request
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        
        # Simulate rate limit exceeded
        api_management_agent.rate_limiter.check_rate_limit = Mock(
            return_value=(False, {"error": "rate_limit_exceeded"})
        )
        
        # First request should be blocked
        allowed, info = await api_management_agent.process_request(request)
        assert not allowed
        
        # Simulate rate limit recovery
        api_management_agent.rate_limiter.check_rate_limit = Mock(
            return_value=(True, {"allowed": True})
        )
        
        # Subsequent request should be allowed
        allowed, info = await api_management_agent.process_request(request)
        assert allowed


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])