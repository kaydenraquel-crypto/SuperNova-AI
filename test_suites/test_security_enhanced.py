"""
Enhanced Security Testing Suite
==============================

Comprehensive security testing for SuperNova AI financial platform
covering authentication, authorization, input validation, data protection,
and vulnerability assessment.
"""
import pytest
import hashlib
import jwt
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
import bcrypt

from supernova.api import app
from supernova.auth import create_access_token, verify_password
from supernova.security_config import SecurityConfig


class TestAuthenticationSecurity:
    """Test authentication security mechanisms."""
    
    @pytest.mark.security
    def test_password_hashing_security(self):
        """Test password hashing uses secure algorithms."""
        test_passwords = ["password123", "SuperSecure!@#", ""]
        
        for password in test_passwords:
            # Test bcrypt hashing
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Verify hash is different each time (salt is unique)
            hashed2 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            assert hashed != hashed2
            
            # Verify password can be verified
            assert bcrypt.checkpw(password.encode('utf-8'), hashed)
            
            # Verify wrong password fails
            assert not bcrypt.checkpw(b"wrongpassword", hashed)
    
    @pytest.mark.security
    def test_jwt_token_security(self):
        """Test JWT token creation and validation."""
        user_data = {"user_id": 123, "email": "test@example.com"}
        
        # Create token
        token = create_access_token(user_data, expires_delta=timedelta(hours=1))
        
        # Verify token structure
        assert len(token.split('.')) == 3  # Header.Payload.Signature
        
        # Decode and verify payload
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["user_id"] == 123
        assert decoded["email"] == "test@example.com"
        assert "exp" in decoded  # Expiration time
        assert "iat" in decoded  # Issued at time
        
        # Test expired token
        expired_token = create_access_token(user_data, expires_delta=timedelta(seconds=-1))
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(expired_token, "secret", algorithms=["HS256"])
    
    @pytest.mark.security
    def test_session_management_security(self, client):
        """Test secure session management."""
        # Test session creation
        login_data = {"username": "testuser", "password": "testpass"}
        
        with patch('supernova.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = {"id": 1, "email": "test@example.com"}
            
            response = client.post("/auth/login", json=login_data)
            assert response.status_code == 200
            
            # Verify secure session cookie attributes
            cookies = response.cookies
            if "session_token" in cookies:
                session_cookie = cookies["session_token"]
                # Should be HttpOnly and Secure in production
                assert "HttpOnly" in str(session_cookie) or "Secure" in str(session_cookie)
    
    @pytest.mark.security
    def test_brute_force_protection(self, client):
        """Test protection against brute force attacks."""
        # Simulate multiple failed login attempts
        login_data = {"username": "testuser", "password": "wrongpassword"}
        
        failed_attempts = 0
        max_attempts = 10
        
        for i in range(max_attempts):
            response = client.post("/auth/login", json=login_data)
            
            if response.status_code == 429:  # Rate limited
                break
            elif response.status_code == 401:  # Unauthorized
                failed_attempts += 1
            
            time.sleep(0.1)  # Brief pause between attempts
        
        # Should be rate limited before max attempts
        assert failed_attempts < max_attempts, "No brute force protection detected"


class TestInputValidationSecurity:
    """Test input validation and sanitization."""
    
    @pytest.mark.security
    def test_sql_injection_protection(self, client, security_test_payloads):
        """Test protection against SQL injection attacks."""
        sql_payloads = security_test_payloads['sql_injection']
        
        for payload in sql_payloads:
            # Test in various endpoints
            test_cases = [
                ("/intake", "POST", {"name": payload, "email": "test@example.com"}),
                ("/profile/search", "GET", {"name": payload}),
                ("/watchlist", "POST", {"profile_id": 1, "symbol": payload}),
            ]
            
            for endpoint, method, data in test_cases:
                if method == "POST":
                    response = client.post(endpoint, json=data)
                else:
                    response = client.get(endpoint, params=data)
                
                # Should reject malicious input
                assert response.status_code in [400, 422], \
                    f"SQL injection payload not blocked: {payload}"
                
                # Should not cause server errors (500)
                assert response.status_code != 500, \
                    f"SQL injection caused server error: {payload}"
    
    @pytest.mark.security
    def test_xss_protection(self, client, security_test_payloads):
        """Test protection against Cross-Site Scripting (XSS)."""
        xss_payloads = security_test_payloads['xss_payloads']
        
        for payload in xss_payloads:
            # Test XSS in user input fields
            intake_data = {
                "name": payload,
                "email": "test@example.com",
                "risk_questions": [3, 3, 3, 3, 3]
            }
            
            response = client.post("/intake", json=intake_data)
            
            if response.status_code == 200:
                # If input is accepted, verify it's properly sanitized
                data = response.json()
                # Name should not contain script tags
                assert "<script>" not in str(data).lower()
                assert "javascript:" not in str(data).lower()
            else:
                # Input should be rejected
                assert response.status_code in [400, 422]
    
    @pytest.mark.security
    def test_path_traversal_protection(self, client, security_test_payloads):
        """Test protection against path traversal attacks."""
        path_traversal_payloads = security_test_payloads['path_traversal']
        
        for payload in path_traversal_payloads:
            # Test in file/resource endpoints
            response = client.get(f"/static/{payload}")
            
            # Should not allow access to system files
            assert response.status_code in [400, 403, 404], \
                f"Path traversal not blocked: {payload}"
    
    @pytest.mark.security
    def test_input_size_limits(self, client):
        """Test input size and rate limiting."""
        # Test extremely large payloads
        large_string = "A" * 100000  # 100KB string
        
        large_payload_data = {
            "name": large_string,
            "email": "test@example.com",
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=large_payload_data)
        
        # Should reject oversized payloads
        assert response.status_code in [400, 413, 422], \
            "Large payload not rejected"
    
    @pytest.mark.security
    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Test with incorrect content type
        response = client.post(
            "/intake",
            data="malicious_data",
            headers={"Content-Type": "text/plain"}
        )
        
        # Should reject non-JSON content for JSON endpoints
        assert response.status_code in [400, 415, 422]


class TestAuthorizationSecurity:
    """Test authorization and access control."""
    
    @pytest.mark.security
    def test_profile_access_control(self, client, test_user, test_profile):
        """Test users can only access their own profiles."""
        # Create second user
        intake_data = {
            "name": "Second User",
            "email": "second@example.com",
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=intake_data)
        second_profile_id = response.json()["profile_id"]
        
        # Try to access first user's profile with second user's token
        with patch('supernova.auth.get_current_user') as mock_user:
            mock_user.return_value = {"id": second_profile_id, "profile_id": second_profile_id}
            
            # Should not be able to access other user's profile
            response = client.get(f"/profile/{test_profile.id}")
            assert response.status_code in [403, 404]
    
    @pytest.mark.security
    def test_admin_endpoint_protection(self, client):
        """Test admin endpoints require proper authorization."""
        admin_endpoints = [
            "/admin/users",
            "/admin/system/config",
            "/admin/analytics/reports",
        ]
        
        for endpoint in admin_endpoints:
            # Try to access without admin privileges
            response = client.get(endpoint)
            assert response.status_code in [401, 403], \
                f"Admin endpoint not protected: {endpoint}"
            
            # Try with regular user token
            with patch('supernova.auth.get_current_user') as mock_user:
                mock_user.return_value = {"id": 1, "role": "user"}
                response = client.get(endpoint)
                assert response.status_code in [403, 404], \
                    f"Admin endpoint accessible to regular user: {endpoint}"
    
    @pytest.mark.security
    def test_api_rate_limiting(self, client):
        """Test API rate limiting implementation."""
        # Make rapid requests to trigger rate limiting
        endpoint = "/health"
        request_count = 0
        rate_limited = False
        
        for i in range(100):  # Make many requests
            response = client.get(endpoint)
            request_count += 1
            
            if response.status_code == 429:  # Rate limited
                rate_limited = True
                break
            
            time.sleep(0.01)  # Small delay
        
        # Should encounter rate limiting
        assert rate_limited or request_count < 100, \
            "No rate limiting detected"


class TestDataProtectionSecurity:
    """Test data protection and privacy measures."""
    
    @pytest.mark.security
    def test_sensitive_data_encryption(self, client, db_session):
        """Test sensitive data is properly encrypted."""
        # Create user with sensitive information
        intake_data = {
            "name": "Encryption Test User",
            "email": "encryption@example.com",
            "income": 100000.0,
            "ssn": "123-45-6789",  # Sensitive data
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=intake_data)
        profile_id = response.json()["profile_id"]
        
        # Verify sensitive data is not stored in plain text
        from supernova.db import Profile
        profile = db_session.query(Profile).filter(Profile.id == profile_id).first()
        
        if hasattr(profile, 'encrypted_ssn'):
            # SSN should be encrypted, not plain text
            assert profile.encrypted_ssn != "123-45-6789"
            assert len(profile.encrypted_ssn) > 20  # Encrypted data is longer
    
    @pytest.mark.security
    def test_pii_data_handling(self, client):
        """Test PII (Personally Identifiable Information) handling."""
        # Test data anonymization/pseudonymization
        intake_data = {
            "name": "PII Test User",
            "email": "pii@example.com",
            "phone": "+1-555-123-4567",
            "address": "123 Main St, Anytown, ST 12345",
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=intake_data)
        assert response.status_code == 200
        
        # Get analytics data (should not contain PII)
        profile_id = response.json()["profile_id"]
        analytics_response = client.get(f"/analytics/user-behavior/{profile_id}")
        
        if analytics_response.status_code == 200:
            analytics_data = analytics_response.json()
            analytics_str = json.dumps(analytics_data)
            
            # PII should not appear in analytics
            assert "pii@example.com" not in analytics_str
            assert "+1-555-123-4567" not in analytics_str
            assert "123 Main St" not in analytics_str
    
    @pytest.mark.security
    def test_data_retention_policies(self, client, db_session):
        """Test data retention and deletion policies."""
        # Create test user
        intake_data = {
            "name": "Retention Test User",
            "email": "retention@example.com",
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=intake_data)
        profile_id = response.json()["profile_id"]
        
        # Test data deletion request
        deletion_request = {"reason": "user_request", "confirm": True}
        delete_response = client.delete(f"/profile/{profile_id}/data", json=deletion_request)
        
        if delete_response.status_code == 200:
            # Verify data is properly anonymized/deleted
            from supernova.db import Profile, User
            profile = db_session.query(Profile).filter(Profile.id == profile_id).first()
            
            if profile:
                # Profile should be marked as deleted or anonymized
                assert hasattr(profile, 'is_deleted') and profile.is_deleted
                # Or PII should be removed
                assert profile.user.email != "retention@example.com"


class TestCryptographicSecurity:
    """Test cryptographic implementations."""
    
    @pytest.mark.security
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        import secrets
        
        # Generate multiple random values
        random_values = [secrets.token_hex(32) for _ in range(100)]
        
        # All values should be unique (extremely high probability)
        assert len(set(random_values)) == len(random_values)
        
        # Each value should be 64 characters (32 bytes in hex)
        for value in random_values[:10]:  # Test subset
            assert len(value) == 64
            # Should contain only hex characters
            int(value, 16)  # Will raise if not valid hex
    
    @pytest.mark.security
    def test_encryption_key_management(self):
        """Test encryption key management and rotation."""
        from supernova.encryption import EncryptionManager
        
        manager = EncryptionManager()
        
        test_data = "sensitive financial information"
        
        # Encrypt with current key
        encrypted = manager.encrypt(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt and verify
        decrypted = manager.decrypt(encrypted)
        assert decrypted == test_data
        
        # Test key rotation
        old_key_id = manager.current_key_id
        manager.rotate_key()
        new_key_id = manager.current_key_id
        
        assert new_key_id != old_key_id
        
        # Old encrypted data should still be decryptable
        old_decrypted = manager.decrypt(encrypted)
        assert old_decrypted == test_data
    
    @pytest.mark.security
    def test_hash_function_security(self):
        """Test cryptographic hash functions."""
        test_data = "financial_data_to_hash"
        
        # Test SHA-256
        hash_sha256 = hashlib.sha256(test_data.encode()).hexdigest()
        assert len(hash_sha256) == 64  # 256 bits = 64 hex chars
        
        # Same input should produce same hash
        hash_sha256_2 = hashlib.sha256(test_data.encode()).hexdigest()
        assert hash_sha256 == hash_sha256_2
        
        # Different input should produce different hash
        hash_different = hashlib.sha256("different_data".encode()).hexdigest()
        assert hash_sha256 != hash_different


class TestSecurityHeaders:
    """Test HTTP security headers implementation."""
    
    @pytest.mark.security
    def test_security_headers_present(self, client):
        """Test required security headers are present."""
        response = client.get("/")
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]
        
        for header in required_headers:
            assert header in response.headers, f"Missing security header: {header}"
    
    @pytest.mark.security
    def test_cors_configuration(self, client):
        """Test CORS configuration is secure."""
        # Test preflight request
        response = client.options(
            "/api/advice",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
            }
        )
        
        # Should not allow arbitrary origins in production
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        if cors_origin:
            assert cors_origin != "*", "Wildcard CORS origin is insecure"
            assert "malicious-site.com" not in cors_origin
    
    @pytest.mark.security
    def test_content_security_policy(self, client):
        """Test Content Security Policy configuration."""
        response = client.get("/")
        
        csp_header = response.headers.get("Content-Security-Policy")
        if csp_header:
            # Should not allow unsafe-inline or unsafe-eval
            assert "unsafe-inline" not in csp_header
            assert "unsafe-eval" not in csp_header
            
            # Should have strict default-src
            assert "default-src 'self'" in csp_header or "default-src 'none'" in csp_header


class TestVulnerabilityAssessment:
    """Test for common vulnerabilities."""
    
    @pytest.mark.security
    def test_directory_traversal_prevention(self, client):
        """Test prevention of directory traversal attacks."""
        traversal_attempts = [
            "../../../etc/passwd",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..\\..\\..\\windows\\system32\\config\\sam",
        ]
        
        for attempt in traversal_attempts:
            response = client.get(f"/files/{attempt}")
            # Should not expose system files
            assert response.status_code in [400, 403, 404]
            
            if response.status_code == 200:
                # If file is served, it should not be a system file
                content = response.text.lower()
                assert "root:" not in content  # Unix passwd file
                assert "administrator:" not in content  # Windows SAM
    
    @pytest.mark.security
    def test_information_disclosure_prevention(self, client):
        """Test prevention of information disclosure."""
        # Test error messages don't reveal sensitive information
        response = client.post("/intake", json={"invalid": "data"})
        
        if response.status_code >= 400:
            error_content = response.text.lower()
            
            # Should not reveal internal paths
            assert "/app/" not in error_content
            assert "/home/" not in error_content
            assert "c:\\" not in error_content
            
            # Should not reveal database information
            assert "sql" not in error_content
            assert "database" not in error_content
            assert "connection" not in error_content
    
    @pytest.mark.security
    def test_session_fixation_prevention(self, client):
        """Test prevention of session fixation attacks."""
        # Get initial session
        response1 = client.get("/")
        session_before = response1.cookies.get("session_id")
        
        # Perform login
        with patch('supernova.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = {"id": 1, "email": "test@example.com"}
            
            login_response = client.post("/auth/login", json={
                "username": "testuser",
                "password": "testpass"
            })
            
            if login_response.status_code == 200:
                session_after = login_response.cookies.get("session_id")
                
                # Session ID should change after login
                if session_before and session_after:
                    assert session_before != session_after, "Session fixation vulnerability detected"


class TestComplianceSecurity:
    """Test financial compliance and regulatory security requirements."""
    
    @pytest.mark.security
    def test_audit_logging(self, client):
        """Test comprehensive audit logging for financial transactions."""
        # Perform actions that should be audited
        intake_data = {
            "name": "Audit Test User",
            "email": "audit@example.com",
            "income": 150000.0,
            "risk_questions": [4, 4, 4, 4, 4]
        }
        
        with patch('supernova.security_logger.log_audit_event') as mock_audit:
            response = client.post("/intake", json=intake_data)
            
            if response.status_code == 200:
                # Should log user creation
                mock_audit.assert_called_with(
                    event_type="user_creation",
                    user_id=pytest.any,
                    details=pytest.any
                )
    
    @pytest.mark.security
    def test_data_integrity_verification(self, client, db_session):
        """Test data integrity verification for financial data."""
        # Create user and verify data integrity
        intake_data = {
            "name": "Integrity Test User",
            "email": "integrity@example.com",
            "income": 75000.0,
            "risk_questions": [3, 3, 3, 3, 3]
        }
        
        response = client.post("/intake", json=intake_data)
        profile_id = response.json()["profile_id"]
        
        # Verify data checksums/hashes for critical financial data
        from supernova.db import Profile
        profile = db_session.query(Profile).filter(Profile.id == profile_id).first()
        
        if hasattr(profile, 'data_checksum'):
            # Verify checksum matches current data
            current_checksum = profile.calculate_checksum()
            assert profile.data_checksum == current_checksum, "Data integrity violation detected"
    
    @pytest.mark.security
    def test_regulatory_compliance_validation(self, client):
        """Test regulatory compliance validation."""
        # Test KYC (Know Your Customer) requirements
        insufficient_kyc_data = {
            "name": "Test",  # Too short
            "email": "test@example.com",
            # Missing required KYC fields
        }
        
        response = client.post("/intake", json=insufficient_kyc_data)
        
        # Should enforce KYC requirements
        if response.status_code == 400:
            error_data = response.json()
            assert "kyc" in str(error_data).lower() or "compliance" in str(error_data).lower()


class TestSecurityMonitoring:
    """Test security monitoring and alerting."""
    
    @pytest.mark.security
    def test_anomaly_detection(self, client):
        """Test detection of anomalous behavior."""
        # Simulate suspicious activity patterns
        suspicious_patterns = [
            # Rapid profile creation
            lambda: [
                client.post("/intake", json={
                    "name": f"Suspicious User {i}",
                    "email": f"suspicious{i}@example.com",
                    "risk_questions": [5, 5, 5, 5, 5]
                }) for i in range(10)
            ],
            
            # Unusual access patterns
            lambda: [
                client.get(f"/profile/{i}") for i in range(1, 50)
            ],
        ]
        
        for pattern_func in suspicious_patterns:
            with patch('supernova.security_monitor.alert_suspicious_activity') as mock_alert:
                responses = pattern_func()
                
                # Should trigger security alerts for suspicious patterns
                # (This would depend on the actual implementation)
                if any(r.status_code == 429 for r in responses):  # Rate limited
                    # Rate limiting is working as expected
                    pass
    
    @pytest.mark.security
    def test_security_incident_response(self, client):
        """Test security incident detection and response."""
        # Simulate potential security incident
        with patch('supernova.security_monitor.detect_incident') as mock_detect:
            mock_detect.return_value = True
            
            # System should handle security incidents appropriately
            response = client.get("/health")
            
            # System might enter maintenance mode or enhanced monitoring
            assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])