"""
Comprehensive Authentication System Tests
Tests for JWT authentication, RBAC, MFA, and security features
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the application and dependencies
from supernova.api import app
from supernova.db import Base, User, UserSession, APIKey, SecurityEvent
from supernova.auth import auth_manager, UserRole, Permission, ROLE_PERMISSIONS
from supernova.security_config import security_settings
from supernova.schemas import (
    LoginRequest, RegisterRequest, TokenResponse, PasswordChangeRequest,
    MFASetupRequest, PasswordResetRequest, PasswordResetConfirmRequest
)

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth.db"
test_engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Test client
client = TestClient(app)

# Test fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Set up test database"""
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    yield
    # Clean up
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def test_db_session():
    """Create a test database session"""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "confirm_password": "SecurePassword123!",
        "role": "user"
    }

@pytest.fixture
def test_admin_data():
    """Test admin user data"""
    return {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "AdminPassword123!",
        "confirm_password": "AdminPassword123!",
        "role": "admin"
    }

@pytest.fixture
def authenticated_user(test_db_session, test_user_data):
    """Create an authenticated test user"""
    # Create user
    hashed_password = auth_manager.hash_password(test_user_data["password"])
    user = User(
        name=test_user_data["name"],
        email=test_user_data["email"],
        hashed_password=hashed_password,
        role=test_user_data["role"],
        is_active=True,
        email_verified=True
    )
    test_db_session.add(user)
    test_db_session.commit()
    test_db_session.refresh(user)
    
    # Generate tokens
    from supernova.auth import TokenPayload
    token_payload = TokenPayload(
        user_id=user.id,
        email=user.email,
        role=UserRole(user.role),
        permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(user.role)]]
    )
    access_token = auth_manager.create_access_token(token_payload)
    
    return {
        "user": user,
        "access_token": access_token,
        "headers": {"Authorization": f"Bearer {access_token}"}
    }

# Authentication Tests
class TestAuthentication:
    """Test authentication functionality"""
    
    def test_user_registration_success(self, test_user_data):
        """Test successful user registration"""
        response = client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == test_user_data["email"]
        assert data["user"]["role"] == test_user_data["role"]
    
    def test_user_registration_duplicate_email(self, test_user_data):
        """Test registration with duplicate email"""
        # Register first user
        client.post("/auth/register", json=test_user_data)
        
        # Try to register again with same email
        response = client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]
    
    def test_user_registration_weak_password(self, test_user_data):
        """Test registration with weak password"""
        test_user_data["password"] = "weak"
        test_user_data["confirm_password"] = "weak"
        
        response = client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 400
        assert "Password validation failed" in response.json()["detail"]
    
    def test_user_registration_password_mismatch(self, test_user_data):
        """Test registration with password mismatch"""
        test_user_data["confirm_password"] = "DifferentPassword123!"
        
        response = client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_user_login_success(self, authenticated_user):
        """Test successful user login"""
        login_data = {
            "email": authenticated_user["user"].email,
            "password": "SecurePassword123!"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == login_data["email"]
    
    def test_user_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]
    
    def test_user_login_inactive_account(self, test_db_session, test_user_data):
        """Test login with inactive account"""
        # Create inactive user
        hashed_password = auth_manager.hash_password(test_user_data["password"])
        user = User(
            name=test_user_data["name"],
            email=test_user_data["email"],
            hashed_password=hashed_password,
            role=test_user_data["role"],
            is_active=False,
            email_verified=True
        )
        test_db_session.add(user)
        test_db_session.commit()
        
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 403
        assert "deactivated" in response.json()["detail"]
    
    def test_token_refresh_success(self, authenticated_user):
        """Test successful token refresh"""
        # First login to get tokens
        login_data = {
            "email": authenticated_user["user"].email,
            "password": "SecurePassword123!"
        }
        
        login_response = client.post("/auth/login", json=login_data)
        tokens = login_response.json()
        
        # Refresh token
        refresh_data = {
            "refresh_token": tokens["refresh_token"]
        }
        
        response = client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["access_token"] != tokens["access_token"]  # New token
    
    def test_token_refresh_invalid_token(self):
        """Test token refresh with invalid token"""
        refresh_data = {
            "refresh_token": "invalid_token"
        }
        
        response = client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
    
    def test_user_logout(self, authenticated_user):
        """Test user logout"""
        headers = authenticated_user["headers"]
        logout_data = {
            "all_devices": False
        }
        
        response = client.post("/auth/logout", json=logout_data, headers=headers)
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    def test_get_user_profile(self, authenticated_user):
        """Test getting user profile"""
        headers = authenticated_user["headers"]
        
        response = client.get("/auth/profile", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["email"] == authenticated_user["user"].email
        assert data["name"] == authenticated_user["user"].name
        assert data["role"] == authenticated_user["user"].role
    
    def test_unauthorized_access(self):
        """Test access without authentication"""
        response = client.get("/auth/profile")
        
        assert response.status_code == 401


class TestPasswordManagement:
    """Test password management functionality"""
    
    def test_password_change_success(self, authenticated_user):
        """Test successful password change"""
        headers = authenticated_user["headers"]
        password_data = {
            "current_password": "SecurePassword123!",
            "new_password": "NewSecurePassword123!",
            "confirm_password": "NewSecurePassword123!"
        }
        
        response = client.post("/auth/password/change", json=password_data, headers=headers)
        
        assert response.status_code == 200
        assert "Password changed successfully" in response.json()["message"]
    
    def test_password_change_wrong_current(self, authenticated_user):
        """Test password change with wrong current password"""
        headers = authenticated_user["headers"]
        password_data = {
            "current_password": "WrongPassword123!",
            "new_password": "NewSecurePassword123!",
            "confirm_password": "NewSecurePassword123!"
        }
        
        response = client.post("/auth/password/change", json=password_data, headers=headers)
        
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"]
    
    def test_password_change_weak_new_password(self, authenticated_user):
        """Test password change with weak new password"""
        headers = authenticated_user["headers"]
        password_data = {
            "current_password": "SecurePassword123!",
            "new_password": "weak",
            "confirm_password": "weak"
        }
        
        response = client.post("/auth/password/change", json=password_data, headers=headers)
        
        assert response.status_code == 400
        assert "Password validation failed" in response.json()["detail"]
    
    def test_password_reset_request(self, authenticated_user):
        """Test password reset request"""
        reset_data = {
            "email": authenticated_user["user"].email
        }
        
        response = client.post("/auth/password/reset", json=reset_data)
        
        assert response.status_code == 200
        assert "reset link has been sent" in response.json()["message"]
    
    def test_password_reset_nonexistent_email(self):
        """Test password reset request for nonexistent email"""
        reset_data = {
            "email": "nonexistent@example.com"
        }
        
        response = client.post("/auth/password/reset", json=reset_data)
        
        # Should still return success to not reveal email existence
        assert response.status_code == 200
        assert "reset link has been sent" in response.json()["message"]


class TestMFA:
    """Test Multi-Factor Authentication functionality"""
    
    def test_mfa_setup(self, authenticated_user):
        """Test MFA setup"""
        headers = authenticated_user["headers"]
        mfa_data = {
            "password": "SecurePassword123!"
        }
        
        response = client.post("/auth/mfa/setup", json=mfa_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "secret" in data
        assert "qr_code" in data
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10  # Default backup codes count


class TestRoleBasedAccess:
    """Test Role-Based Access Control"""
    
    def test_admin_access(self, test_db_session, test_admin_data):
        """Test admin user access to admin endpoints"""
        # Create admin user
        hashed_password = auth_manager.hash_password(test_admin_data["password"])
        admin_user = User(
            name=test_admin_data["name"],
            email=test_admin_data["email"],
            hashed_password=hashed_password,
            role="admin",
            is_active=True,
            email_verified=True
        )
        test_db_session.add(admin_user)
        test_db_session.commit()
        test_db_session.refresh(admin_user)
        
        # Generate admin token
        from supernova.auth import TokenPayload
        token_payload = TokenPayload(
            user_id=admin_user.id,
            email=admin_user.email,
            role=UserRole.ADMIN,
            permissions=[p.value for p in ROLE_PERMISSIONS[UserRole.ADMIN]]
        )
        access_token = auth_manager.create_access_token(token_payload)
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test admin endpoint access (if we had any)
        # For now, just verify the token contains admin permissions
        payload = auth_manager.verify_token(access_token)
        assert UserRole.ADMIN.value in payload["role"]
        assert Permission.ADMIN_DASHBOARD.value in payload["permissions"]
    
    def test_user_denied_admin_access(self, authenticated_user):
        """Test regular user denied access to admin functions"""
        # Verify user doesn't have admin permissions
        payload = auth_manager.verify_token(authenticated_user["access_token"])
        assert payload["role"] == UserRole.USER.value
        assert Permission.ADMIN_DASHBOARD.value not in payload["permissions"]


class TestSecurityFeatures:
    """Test security features and protections"""
    
    def test_password_strength_validation(self):
        """Test password strength validation"""
        weak_passwords = [
            "123456",
            "password",
            "abc",
            "PASSWORD123",  # Missing lowercase/special
            "password123",  # Missing uppercase/special
            "Password!",    # Missing digits
        ]
        
        for password in weak_passwords:
            validation = auth_manager.validate_password_strength(password)
            assert not validation["valid"]
            assert len(validation["errors"]) > 0
    
    def test_strong_password_validation(self):
        """Test strong password validation"""
        strong_passwords = [
            "SecurePassword123!",
            "MyStrongP@ssw0rd",
            "Complex$Password1",
        ]
        
        for password in strong_passwords:
            validation = auth_manager.validate_password_strength(password)
            assert validation["valid"]
            assert len(validation["errors"]) == 0
            assert validation["strength_score"] >= 80
    
    @patch('supernova.auth.auth_manager.record_failed_attempt')
    def test_failed_login_tracking(self, mock_record_attempt):
        """Test failed login attempt tracking"""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        mock_record_attempt.assert_called_once()
    
    def test_jwt_token_structure(self, authenticated_user):
        """Test JWT token contains required fields"""
        token = authenticated_user["access_token"]
        payload = auth_manager.verify_token(token)
        
        required_fields = ["sub", "email", "role", "permissions", "iss", "aud", "exp", "iat"]
        for field in required_fields:
            assert field in payload
        
        assert payload["iss"] == security_settings.JWT_ISSUER
        assert payload["aud"] == security_settings.JWT_AUDIENCE
    
    def test_token_expiration(self):
        """Test token expiration"""
        from supernova.auth import TokenPayload
        
        # Create token with very short expiration
        token_payload = TokenPayload(
            user_id=1,
            email="test@example.com",
            role=UserRole.USER,
            permissions=[]
        )
        
        short_expiry = timedelta(seconds=1)
        token = auth_manager.create_access_token(token_payload, expires_delta=short_expiry)
        
        # Wait for token to expire
        time.sleep(2)
        
        with pytest.raises(Exception):  # Should raise AuthenticationError
            auth_manager.verify_token(token)
    
    def test_token_blacklisting(self, authenticated_user):
        """Test token blacklisting"""
        token = authenticated_user["access_token"]
        
        # Verify token is valid
        payload = auth_manager.verify_token(token)
        assert payload is not None
        
        # Blacklist token
        auth_manager.blacklist_token(token)
        
        # Verify token is now invalid
        with pytest.raises(Exception):  # Should raise AuthenticationError
            auth_manager.verify_token(token)


class TestAuthenticationIntegration:
    """Test authentication integration with existing endpoints"""
    
    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication"""
        response = client.post("/advice", json={
            "profile_id": 1,
            "symbol": "AAPL",
            "bars": []
        })
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_auth(self, authenticated_user):
        """Test accessing protected endpoint with authentication"""
        headers = authenticated_user["headers"]
        
        # This might fail due to business logic, but should not fail due to auth
        response = client.post("/advice", json={
            "profile_id": 1,
            "symbol": "AAPL",
            "bars": []
        }, headers=headers)
        
        # Should not be 401 (unauthorized)
        assert response.status_code != 401


# Performance and Load Tests
class TestAuthenticationPerformance:
    """Test authentication system performance"""
    
    def test_password_hashing_performance(self):
        """Test password hashing performance"""
        password = "SecurePassword123!"
        
        start_time = time.time()
        hashed = auth_manager.hash_password(password)
        hash_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert hash_time < 1.0  # Less than 1 second
        assert auth_manager.verify_password(password, hashed)
    
    def test_token_creation_performance(self):
        """Test JWT token creation performance"""
        from supernova.auth import TokenPayload
        
        token_payload = TokenPayload(
            user_id=1,
            email="test@example.com",
            role=UserRole.USER,
            permissions=["read_profile"]
        )
        
        start_time = time.time()
        for _ in range(100):  # Create 100 tokens
            auth_manager.create_access_token(token_payload)
        creation_time = time.time() - start_time
        
        # Should be fast enough for production use
        assert creation_time < 1.0  # Less than 1 second for 100 tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])