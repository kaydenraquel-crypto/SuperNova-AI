"""
SuperNova AI Collaboration System Tests
Comprehensive test suite for team management, sharing, and real-time features
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Test imports
from supernova.api import app
from supernova.db import Base, User
from supernova.collaboration_models import (
    Team, TeamInvitation, SharedPortfolio, ShareLink,
    TeamChat, ChatMessage, Notification, TeamActivity
)
from supernova.collaboration_service import collaboration_service
from supernova.collaboration_websocket import collaboration_websocket_handler
from supernova.auth import auth_manager, UserRole


# Test Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_collaboration.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test"""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client():
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_user(db_session):
    """Create a sample user for testing"""
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=auth_manager.hash_password("testpassword123"),
        role=UserRole.USER.value,
        is_active=True,
        email_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_admin_user(db_session):
    """Create a sample admin user for testing"""
    user = User(
        name="Admin User",
        email="admin@example.com",
        hashed_password=auth_manager.hash_password("adminpassword123"),
        role=UserRole.ADMIN.value,
        is_active=True,
        email_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(sample_user):
    """Generate authentication headers for testing"""
    from supernova.auth import TokenPayload, Permission, ROLE_PERMISSIONS
    
    payload = TokenPayload(
        user_id=sample_user.id,
        email=sample_user.email,
        role=UserRole(sample_user.role),
        permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(sample_user.role)]]
    )
    
    token = auth_manager.create_access_token(payload)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(sample_admin_user):
    """Generate admin authentication headers for testing"""
    from supernova.auth import TokenPayload, Permission, ROLE_PERMISSIONS
    
    payload = TokenPayload(
        user_id=sample_admin_user.id,
        email=sample_admin_user.email,
        role=UserRole(sample_admin_user.role),
        permissions=[p.value for p in ROLE_PERMISSIONS[UserRole(sample_admin_user.role)]]
    )
    
    token = auth_manager.create_access_token(payload)
    return {"Authorization": f"Bearer {token}"}


# ================================
# TEAM MANAGEMENT TESTS
# ================================

class TestTeamManagement:
    """Test team creation, management, and member operations"""
    
    def test_create_team_success(self, client, auth_headers, db_session):
        """Test successful team creation"""
        team_data = {
            "name": "Test Team",
            "description": "A test team for collaboration",
            "is_private": True,
            "max_members": 10,
            "generate_invite_code": True
        }
        
        response = client.post(
            "/api/collaboration/teams",
            json=team_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == team_data["name"]
        assert data["description"] == team_data["description"]
        assert "invite_code" in data
        assert data["your_role"] == "owner"
    
    def test_create_team_invalid_data(self, client, auth_headers):
        """Test team creation with invalid data"""
        team_data = {
            "name": "",  # Empty name should fail
            "max_members": -1  # Invalid member count
        }
        
        response = client.post(
            "/api/collaboration/teams",
            json=team_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_get_user_teams(self, client, auth_headers, db_session, sample_user):
        """Test retrieving user's teams"""
        # Create test team
        team = Team(
            name="User's Team",
            description="Test team",
            created_by=sample_user.id,
            is_private=True
        )
        db_session.add(team)
        db_session.commit()
        
        response = client.get(
            "/api/collaboration/teams",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "teams" in data
        assert "total" in data
        assert data["total"] >= 1
    
    def test_get_team_details(self, client, auth_headers, db_session, sample_user):
        """Test retrieving detailed team information"""
        # Create test team
        team = Team(
            name="Detail Test Team",
            description="Team for detail testing",
            created_by=sample_user.id
        )
        db_session.add(team)
        db_session.commit()
        
        response = client.get(
            f"/api/collaboration/teams/{team.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "team" in data
        assert "members" in data
        assert "recent_activity" in data
        assert "statistics" in data
    
    def test_invite_team_members(self, client, auth_headers, db_session, sample_user):
        """Test inviting members to team"""
        # Create test team
        team = Team(
            name="Invitation Test Team",
            created_by=sample_user.id
        )
        db_session.add(team)
        db_session.commit()
        
        invitation_data = {
            "emails": ["newuser@example.com", "anotheruser@example.com"],
            "role": "member",
            "message": "Welcome to our team!"
        }
        
        response = client.post(
            f"/api/collaboration/teams/{team.id}/invite",
            json=invitation_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "invitations_sent" in data
        assert data["invitations_sent"] == 2
    
    def test_join_team_by_code(self, client, auth_headers, db_session):
        """Test joining team with invite code"""
        # Create team with invite code
        team = Team(
            name="Public Join Team",
            created_by=1,
            invite_code="TESTCODE123"
        )
        db_session.add(team)
        db_session.commit()
        
        join_data = {
            "invite_code": "TESTCODE123"
        }
        
        response = client.post(
            "/api/collaboration/teams/join",
            json=join_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "team_id" in data
        assert data["team_id"] == team.id
    
    def test_join_team_invalid_code(self, client, auth_headers):
        """Test joining team with invalid invite code"""
        join_data = {
            "invite_code": "INVALIDCODE"
        }
        
        response = client.post(
            "/api/collaboration/teams/join",
            json=join_data,
            headers=auth_headers
        )
        
        assert response.status_code == 404


# ================================
# PORTFOLIO SHARING TESTS
# ================================

class TestPortfolioSharing:
    """Test portfolio sharing functionality"""
    
    @pytest.fixture
    def sample_portfolio(self, db_session, sample_user):
        """Create a sample portfolio for testing"""
        from supernova.analytics_models import Portfolio
        portfolio = Portfolio(
            name="Test Portfolio",
            description="Portfolio for testing",
            user_id=sample_user.id,
            initial_value=10000.0
        )
        db_session.add(portfolio)
        db_session.commit()
        db_session.refresh(portfolio)
        return portfolio
    
    def test_share_portfolio_with_team(self, client, auth_headers, db_session, sample_user, sample_portfolio):
        """Test sharing portfolio with team"""
        # Create test team
        team = Team(name="Share Test Team", created_by=sample_user.id)
        db_session.add(team)
        db_session.commit()
        
        share_data = {
            "portfolio_id": sample_portfolio.id,
            "target_type": "team",
            "target_id": team.id,
            "permission_level": "view"
        }
        
        response = client.post(
            "/api/collaboration/share/portfolio",
            json=share_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "share_id" in data
        assert data["permission_level"] == "view"
    
    def test_share_portfolio_public(self, client, auth_headers, db_session, sample_portfolio):
        """Test creating public portfolio share"""
        share_data = {
            "portfolio_id": sample_portfolio.id,
            "target_type": "public",
            "permission_level": "view",
            "password_protected": False
        }
        
        response = client.post(
            "/api/collaboration/share/portfolio",
            json=share_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "share_token" in data
        assert "share_url" in data
    
    def test_create_share_link(self, client, auth_headers, sample_portfolio):
        """Test creating public share link"""
        share_link_data = {
            "resource_type": "portfolio",
            "resource_id": sample_portfolio.id,
            "permission_level": "view",
            "title": "My Portfolio Share",
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        
        response = client.post(
            "/api/collaboration/share-links",
            json=share_link_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["resource_type"] == "portfolio"
        assert data["title"] == "My Portfolio Share"


# ================================
# WEBSOCKET COLLABORATION TESTS
# ================================

class TestWebSocketCollaboration:
    """Test real-time collaboration WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_auth(self):
        """Test WebSocket authentication"""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.receive_text.return_value = json.dumps({
            "type": "auth",
            "token": "valid_token"
        })
        
        # Mock token verification
        with patch.object(auth_manager, 'verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "123", "session_id": "test_session"}
            
            # Test connection
            connection_id = await collaboration_websocket_handler.connect_user_collaboration(
                mock_websocket, 123, [1, 2]
            )
            
            assert connection_id is not None
            assert 123 in [session.user_id for session in collaboration_websocket_handler.collaboration_sessions.values()]
    
    @pytest.mark.asyncio
    async def test_team_message_broadcast(self):
        """Test team message broadcasting"""
        # Create mock connections
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        # Connect users to team
        conn1 = await collaboration_websocket_handler.connect_user_collaboration(mock_websocket1, 1, [100])
        conn2 = await collaboration_websocket_handler.connect_user_collaboration(mock_websocket2, 2, [100])
        
        # Send team message
        message = {
            "type": "team_message",
            "data": {
                "team_id": 100,
                "content": "Hello team!",
                "message_type": "text"
            }
        }
        
        await collaboration_websocket_handler.handle_collaboration_message(conn1, message)
        
        # Verify broadcast
        mock_websocket2.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_portfolio_collaboration(self):
        """Test portfolio collaboration events"""
        mock_websocket = AsyncMock()
        
        connection_id = await collaboration_websocket_handler.connect_user_collaboration(
            mock_websocket, 1, []
        )
        
        # Send portfolio collaboration event
        message = {
            "type": "portfolio_collaboration",
            "data": {
                "portfolio_id": 123,
                "action": "edit",
                "metadata": {"position": "AAPL", "quantity": 100}
            }
        }
        
        await collaboration_websocket_handler.handle_collaboration_message(connection_id, message)
        
        # Verify portfolio watchers updated
        assert 123 in collaboration_websocket_handler.portfolio_watchers
    
    @pytest.mark.asyncio
    async def test_presence_updates(self):
        """Test user presence status updates"""
        mock_websocket = AsyncMock()
        
        connection_id = await collaboration_websocket_handler.connect_user_collaboration(
            mock_websocket, 1, [100]
        )
        
        # Update presence
        await collaboration_websocket_handler._broadcast_presence_update(
            1, collaboration_websocket_handler.UserStatus.AWAY, [100]
        )
        
        # Verify presence broadcast
        assert collaboration_websocket_handler.user_status.get(1) == collaboration_websocket_handler.UserStatus.ONLINE


# ================================
# NOTIFICATION TESTS
# ================================

class TestNotifications:
    """Test notification system"""
    
    def test_get_notifications(self, client, auth_headers, db_session, sample_user):
        """Test retrieving user notifications"""
        # Create test notification
        notification = Notification(
            user_id=sample_user.id,
            type="team_invite",
            title="Team Invitation",
            message="You've been invited to join a team"
        )
        db_session.add(notification)
        db_session.commit()
        
        response = client.get(
            "/api/collaboration/notifications",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data
        assert "unread_count" in data
    
    def test_mark_notifications_read(self, client, auth_headers, db_session, sample_user):
        """Test marking notifications as read"""
        # Create test notifications
        notification1 = Notification(
            user_id=sample_user.id,
            type="team_invite",
            title="Team Invitation 1",
            message="First invitation"
        )
        notification2 = Notification(
            user_id=sample_user.id,
            type="portfolio_shared",
            title="Portfolio Shared",
            message="A portfolio was shared with you"
        )
        db_session.add_all([notification1, notification2])
        db_session.commit()
        
        mark_read_data = {
            "notification_ids": [notification1.id, notification2.id]
        }
        
        response = client.post(
            "/api/collaboration/notifications/mark-read",
            json=mark_read_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200


# ================================
# COLLABORATION SERVICE TESTS
# ================================

class TestCollaborationService:
    """Test collaboration service layer"""
    
    def test_create_team_service(self, db_session, sample_user):
        """Test team creation through service layer"""
        from supernova.collaboration_schemas import TeamCreateRequest
        
        request = TeamCreateRequest(
            name="Service Test Team",
            description="Testing service layer",
            is_private=False,
            max_members=25,
            generate_invite_code=True
        )
        
        result = collaboration_service.create_team(sample_user.id, request)
        
        assert result["name"] == "Service Test Team"
        assert result["your_role"] == "owner"
        assert "invite_code" in result
    
    def test_get_team_statistics(self, db_session, sample_user):
        """Test team statistics calculation"""
        # Create test team
        team = Team(
            name="Stats Test Team",
            created_by=sample_user.id
        )
        db_session.add(team)
        db_session.commit()
        
        stats = collaboration_service._get_team_statistics(db_session, team.id)
        
        assert "member_count" in stats
        assert "active_members_7d" in stats
        assert "shared_portfolios" in stats
        assert "collaboration_score" in stats


# ================================
# SECURITY TESTS
# ================================

class TestCollaborationSecurity:
    """Test security aspects of collaboration features"""
    
    def test_team_access_control(self, client, auth_headers, db_session):
        """Test team access control and permissions"""
        # Create team owned by different user
        team = Team(
            name="Other User's Team",
            created_by=999,  # Different user
            is_private=True
        )
        db_session.add(team)
        db_session.commit()
        
        # Try to access team details without permission
        response = client.get(
            f"/api/collaboration/teams/{team.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 403
    
    def test_portfolio_sharing_permissions(self, client, auth_headers, db_session, sample_user):
        """Test portfolio sharing permission validation"""
        from supernova.analytics_models import Portfolio
        
        # Create portfolio owned by different user
        other_portfolio = Portfolio(
            name="Other User's Portfolio",
            user_id=999,  # Different user
            initial_value=5000.0
        )
        db_session.add(other_portfolio)
        db_session.commit()
        
        share_data = {
            "portfolio_id": other_portfolio.id,
            "target_type": "public",
            "permission_level": "view"
        }
        
        response = client.post(
            "/api/collaboration/share/portfolio",
            json=share_data,
            headers=auth_headers
        )
        
        assert response.status_code == 404  # Portfolio not found (not owned)
    
    def test_invitation_security(self, client, auth_headers, db_session):
        """Test invitation system security"""
        # Create team owned by different user
        team = Team(
            name="Restricted Team",
            created_by=999,
            is_private=True
        )
        db_session.add(team)
        db_session.commit()
        
        invitation_data = {
            "emails": ["test@example.com"],
            "role": "member"
        }
        
        # Try to invite to team without permission
        response = client.post(
            f"/api/collaboration/teams/{team.id}/invite",
            json=invitation_data,
            headers=auth_headers
        )
        
        assert response.status_code == 403


# ================================
# INTEGRATION TESTS
# ================================

class TestCollaborationIntegration:
    """Test full integration scenarios"""
    
    def test_team_collaboration_workflow(self, client, auth_headers, admin_auth_headers, db_session, sample_user):
        """Test complete team collaboration workflow"""
        # 1. Create team
        team_data = {
            "name": "Integration Test Team",
            "description": "Full workflow test",
            "is_private": False,
            "max_members": 5,
            "generate_invite_code": True
        }
        
        response = client.post(
            "/api/collaboration/teams",
            json=team_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        team = response.json()
        
        # 2. Invite members
        invitation_data = {
            "emails": ["member1@example.com", "member2@example.com"],
            "role": "member"
        }
        
        response = client.post(
            f"/api/collaboration/teams/{team['id']}/invite",
            json=invitation_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # 3. Share portfolio with team
        from supernova.analytics_models import Portfolio
        portfolio = Portfolio(
            name="Shared Portfolio",
            user_id=sample_user.id,
            initial_value=10000.0
        )
        db_session.add(portfolio)
        db_session.commit()
        
        share_data = {
            "portfolio_id": portfolio.id,
            "target_type": "team",
            "target_id": team["id"],
            "permission_level": "comment"
        }
        
        response = client.post(
            "/api/collaboration/share/portfolio",
            json=share_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # 4. Verify team details include shared portfolio
        response = client.get(
            f"/api/collaboration/teams/{team['id']}",
            headers=auth_headers
        )
        assert response.status_code == 200
        team_details = response.json()
        assert team_details["statistics"]["shared_portfolios"] >= 1


# ================================
# PERFORMANCE TESTS
# ================================

class TestCollaborationPerformance:
    """Test performance aspects of collaboration features"""
    
    def test_large_team_handling(self, client, auth_headers, db_session, sample_user):
        """Test handling large teams efficiently"""
        # Create team
        team = Team(
            name="Large Team Test",
            created_by=sample_user.id,
            max_members=1000
        )
        db_session.add(team)
        db_session.commit()
        
        # Simulate many members (in real test, would create actual users)
        import time
        start_time = time.time()
        
        response = client.get(
            f"/api/collaboration/teams/{team.id}",
            headers=auth_headers
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self):
        """Test WebSocket broadcasting performance"""
        # Create multiple mock connections
        connections = []
        for i in range(100):
            mock_ws = AsyncMock()
            conn_id = await collaboration_websocket_handler.connect_user_collaboration(
                mock_ws, i, [1]
            )
            connections.append((conn_id, mock_ws))
        
        import time
        start_time = time.time()
        
        # Broadcast message to team
        await collaboration_websocket_handler._broadcast_to_team(1, {
            "type": "performance_test",
            "data": {"message": "Testing broadcast performance"}
        })
        
        end_time = time.time()
        broadcast_time = end_time - start_time
        
        assert broadcast_time < 0.1  # Should broadcast to 100 connections in < 100ms


# ================================
# TEST RUNNER
# ================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])