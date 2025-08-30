"""
SuperNova AI Collaboration API Endpoints
RESTful API for team management, sharing, and real-time collaboration
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime

# Database and models
from .db import SessionLocal, User
from .collaboration_models import (
    Team, TeamInvitation, SharedPortfolio, ShareLink, TeamChat, ChatMessage,
    DirectMessage, Notification, TeamRole, SharePermission, NotificationType
)

# Schemas
from .collaboration_schemas import (
    # Team management
    TeamCreateRequest, TeamUpdateRequest, TeamListResponse, TeamDetailResponse,
    TeamInviteRequest, TeamInviteResponse, TeamInviteByCodeRequest,
    UpdateMemberRoleRequest, RemoveMemberRequest, TransferOwnershipRequest,
    
    # Sharing
    SharePortfolioRequest, ShareStrategyRequest, CreateShareLinkRequest,
    ShareLinkInfo, ShareLinkListResponse, AccessShareLinkRequest,
    
    # Communication
    CreateChatChannelRequest, ChatChannelInfo, SendMessageRequest,
    ChatMessageInfo, ChatHistoryResponse, SendDirectMessageRequest, DirectMessageInfo,
    
    # Notifications
    NotificationInfo, NotificationListResponse, MarkNotificationsReadRequest,
    
    # Activity and analytics
    ActivityInfo, TeamActivityResponse, TeamAnalyticsResponse,
    
    # Comments
    AddCommentRequest, CommentInfo, CommentListResponse
)

# Services and auth
from .collaboration_service import collaboration_service, CollaborationError
from .auth import get_current_user, require_permission, Permission, UserRole
from .websocket_handler import websocket_handler
from .security_logger import SecurityLogger

import logging

logger = logging.getLogger(__name__)
security_logger = SecurityLogger()

# Create router
router = APIRouter(prefix="/api/collaboration", tags=["collaboration"])


# ================================
# TEAM MANAGEMENT ENDPOINTS
# ================================

@router.post("/teams", response_model=Dict[str, Any])
async def create_team(
    request: TeamCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new team"""
    try:
        result = collaboration_service.create_team(
            user_id=int(current_user["sub"]),
            request=request
        )
        
        security_logger.log_security_event(
            event_type="TEAM_CREATED",
            user_id=current_user["sub"],
            details={
                "team_name": request.name,
                "is_private": request.is_private,
                "max_members": request.max_members
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create team"
        )


@router.get("/teams", response_model=TeamListResponse)
async def get_user_teams(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: dict = Depends(get_current_user)
):
    """Get user's teams"""
    try:
        result = collaboration_service.get_user_teams(
            user_id=int(current_user["sub"]),
            page=page,
            per_page=per_page
        )
        return result
        
    except Exception as e:
        logger.error(f"Error getting user teams: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve teams"
        )


@router.get("/teams/{team_id}", response_model=TeamDetailResponse)
async def get_team_details(
    team_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed team information"""
    try:
        result = collaboration_service.get_team_details(
            team_id=team_id,
            user_id=int(current_user["sub"])
        )
        return result
        
    except Exception as e:
        logger.error(f"Error getting team details: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve team details"
        )


@router.put("/teams/{team_id}")
async def update_team(
    team_id: int,
    request: TeamUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update team information"""
    try:
        # Implementation would go here
        # collaboration_service.update_team(team_id, int(current_user["sub"]), request)
        
        return {"message": "Team updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update team"
        )


@router.delete("/teams/{team_id}")
async def delete_team(
    team_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a team (owner only)"""
    try:
        # Implementation would go here
        # collaboration_service.delete_team(team_id, int(current_user["sub"]))
        
        security_logger.log_security_event(
            event_type="TEAM_DELETED",
            user_id=current_user["sub"],
            details={"team_id": team_id}
        )
        
        return {"message": "Team deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete team"
        )


# ================================
# TEAM INVITATION ENDPOINTS
# ================================

@router.post("/teams/{team_id}/invite", response_model=TeamInviteResponse)
async def invite_team_members(
    team_id: int,
    request: TeamInviteRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Invite users to team"""
    try:
        result = collaboration_service.invite_team_members(
            team_id=team_id,
            user_id=int(current_user["sub"]),
            request=request
        )
        
        # Send email invitations in background
        # background_tasks.add_task(send_invitation_emails, team_id, request.emails)
        
        security_logger.log_security_event(
            event_type="TEAM_INVITATIONS_SENT",
            user_id=current_user["sub"],
            details={
                "team_id": team_id,
                "invitations_count": len(request.emails),
                "role": request.role.value
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error inviting team members: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send invitations"
        )


@router.post("/teams/join", response_model=Dict[str, Any])
async def join_team_by_code(
    request: TeamInviteByCodeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Join team using invite code"""
    try:
        result = collaboration_service.join_team_by_code(
            user_id=int(current_user["sub"]),
            invite_code=request.invite_code
        )
        
        security_logger.log_security_event(
            event_type="TEAM_JOINED_BY_CODE",
            user_id=current_user["sub"],
            details={"invite_code": request.invite_code}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error joining team by code: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join team"
        )


@router.get("/invitations/pending")
async def get_pending_invitations(
    current_user: dict = Depends(get_current_user)
):
    """Get user's pending team invitations"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_pending_invitations(int(current_user["sub"]))
        
        return {"invitations": [], "total": 0}
        
    except Exception as e:
        logger.error(f"Error getting pending invitations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve invitations"
        )


@router.post("/invitations/{invitation_token}/accept")
async def accept_team_invitation(
    invitation_token: str,
    current_user: dict = Depends(get_current_user)
):
    """Accept team invitation"""
    try:
        # Implementation would go here
        # result = collaboration_service.accept_invitation(invitation_token, int(current_user["sub"]))
        
        security_logger.log_security_event(
            event_type="TEAM_INVITATION_ACCEPTED",
            user_id=current_user["sub"],
            details={"invitation_token": invitation_token}
        )
        
        return {"message": "Invitation accepted successfully"}
        
    except Exception as e:
        logger.error(f"Error accepting invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept invitation"
        )


@router.post("/invitations/{invitation_token}/decline")
async def decline_team_invitation(
    invitation_token: str,
    current_user: dict = Depends(get_current_user)
):
    """Decline team invitation"""
    try:
        # Implementation would go here
        # collaboration_service.decline_invitation(invitation_token, int(current_user["sub"]))
        
        return {"message": "Invitation declined"}
        
    except Exception as e:
        logger.error(f"Error declining invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decline invitation"
        )


# ================================
# MEMBER MANAGEMENT ENDPOINTS
# ================================

@router.put("/teams/{team_id}/members/{user_id}/role")
async def update_member_role(
    team_id: int,
    user_id: int,
    request: UpdateMemberRoleRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update team member role"""
    try:
        # Implementation would go here
        # collaboration_service.update_member_role(team_id, int(current_user["sub"]), user_id, request.new_role)
        
        security_logger.log_security_event(
            event_type="TEAM_MEMBER_ROLE_UPDATED",
            user_id=current_user["sub"],
            details={
                "team_id": team_id,
                "target_user_id": user_id,
                "new_role": request.new_role.value
            }
        )
        
        return {"message": "Member role updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating member role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update member role"
        )


@router.delete("/teams/{team_id}/members/{user_id}")
async def remove_team_member(
    team_id: int,
    user_id: int,
    request: RemoveMemberRequest,
    current_user: dict = Depends(get_current_user)
):
    """Remove team member"""
    try:
        # Implementation would go here
        # collaboration_service.remove_member(team_id, int(current_user["sub"]), user_id, request.reason)
        
        security_logger.log_security_event(
            event_type="TEAM_MEMBER_REMOVED",
            user_id=current_user["sub"],
            details={
                "team_id": team_id,
                "removed_user_id": user_id,
                "reason": request.reason
            }
        )
        
        return {"message": "Member removed successfully"}
        
    except Exception as e:
        logger.error(f"Error removing team member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove member"
        )


@router.post("/teams/{team_id}/transfer-ownership")
async def transfer_team_ownership(
    team_id: int,
    request: TransferOwnershipRequest,
    current_user: dict = Depends(get_current_user)
):
    """Transfer team ownership"""
    try:
        # Implementation would go here
        # collaboration_service.transfer_ownership(team_id, int(current_user["sub"]), request.new_owner_id)
        
        security_logger.log_security_event(
            event_type="TEAM_OWNERSHIP_TRANSFERRED",
            user_id=current_user["sub"],
            details={
                "team_id": team_id,
                "new_owner_id": request.new_owner_id
            }
        )
        
        return {"message": "Ownership transferred successfully"}
        
    except Exception as e:
        logger.error(f"Error transferring ownership: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transfer ownership"
        )


# ================================
# PORTFOLIO SHARING ENDPOINTS
# ================================

@router.post("/share/portfolio", response_model=Dict[str, Any])
async def share_portfolio(
    request: SharePortfolioRequest,
    current_user: dict = Depends(get_current_user)
):
    """Share portfolio with team, user, or create public link"""
    try:
        result = collaboration_service.share_portfolio(
            user_id=int(current_user["sub"]),
            request=request
        )
        
        security_logger.log_security_event(
            event_type="PORTFOLIO_SHARED",
            user_id=current_user["sub"],
            details={
                "portfolio_id": request.portfolio_id,
                "target_type": request.target_type,
                "target_id": request.target_id,
                "permission_level": request.permission_level.value
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sharing portfolio: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share portfolio"
        )


@router.post("/share/strategy", response_model=Dict[str, Any])
async def share_strategy(
    request: ShareStrategyRequest,
    current_user: dict = Depends(get_current_user)
):
    """Share strategy with team, user, or create public link"""
    try:
        # Implementation would go here
        # result = collaboration_service.share_strategy(int(current_user["sub"]), request)
        
        return {"message": "Strategy shared successfully"}
        
    except Exception as e:
        logger.error(f"Error sharing strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share strategy"
        )


@router.post("/share-links", response_model=ShareLinkInfo)
async def create_share_link(
    request: CreateShareLinkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create public share link"""
    try:
        # Implementation would go here
        # result = collaboration_service.create_share_link(int(current_user["sub"]), request)
        
        return {
            "id": 1,
            "share_token": "sample_token",
            "resource_type": request.resource_type,
            "resource_id": request.resource_id,
            "title": request.title,
            "permission_level": request.permission_level,
            "current_views": 0,
            "max_views": request.max_views,
            "expires_at": request.expires_at,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
    except Exception as e:
        logger.error(f"Error creating share link: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create share link"
        )


@router.get("/share-links", response_model=ShareLinkListResponse)
async def get_user_share_links(
    current_user: dict = Depends(get_current_user)
):
    """Get user's share links"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_user_share_links(int(current_user["sub"]))
        
        return {"links": [], "total": 0}
        
    except Exception as e:
        logger.error(f"Error getting share links: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve share links"
        )


@router.post("/shared/{share_token}/access")
async def access_shared_resource(
    share_token: str,
    request: AccessShareLinkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Access shared resource via public link"""
    try:
        # Implementation would go here
        # result = collaboration_service.access_shared_resource(share_token, request.password, int(current_user["sub"]))
        
        return {"message": "Access granted", "resource_data": {}}
        
    except Exception as e:
        logger.error(f"Error accessing shared resource: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access shared resource"
        )


# ================================
# COMMUNICATION ENDPOINTS
# ================================

@router.post("/teams/{team_id}/channels", response_model=ChatChannelInfo)
async def create_chat_channel(
    team_id: int,
    request: CreateChatChannelRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create team chat channel"""
    try:
        # Implementation would go here
        # result = collaboration_service.create_chat_channel(team_id, int(current_user["sub"]), request)
        
        return {
            "id": 1,
            "name": request.name,
            "description": request.description,
            "is_private": request.is_private,
            "channel_type": request.channel_type,
            "message_count": 0,
            "last_message_at": None,
            "created_at": datetime.utcnow(),
            "unread_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error creating chat channel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat channel"
        )


@router.get("/teams/{team_id}/channels", response_model=List[ChatChannelInfo])
async def get_team_channels(
    team_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get team chat channels"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_team_channels(team_id, int(current_user["sub"]))
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting team channels: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve channels"
        )


@router.post("/channels/{channel_id}/messages", response_model=ChatMessageInfo)
async def send_message(
    channel_id: int,
    request: SendMessageRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send message to chat channel"""
    try:
        # Implementation would go here
        # result = collaboration_service.send_message(channel_id, int(current_user["sub"]), request)
        
        # Broadcast to WebSocket subscribers
        # await websocket_handler.broadcast_to_channel(channel_id, {
        #     "type": "new_message",
        #     "data": result
        # })
        
        return {
            "id": 1,
            "author_id": int(current_user["sub"]),
            "author_name": current_user.get("name", "Unknown"),
            "content": request.content,
            "message_type": request.message_type,
            "thread_id": request.thread_id,
            "mentions": request.mentions,
            "reactions": None,
            "created_at": datetime.utcnow(),
            "edited_at": None,
            "is_deleted": False
        }
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message"
        )


@router.get("/channels/{channel_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_history(
    channel_id: int,
    limit: int = Query(50, ge=1, le=100),
    before: Optional[int] = Query(None, description="Message ID to paginate before"),
    current_user: dict = Depends(get_current_user)
):
    """Get chat message history"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_chat_history(channel_id, int(current_user["sub"]), limit, before)
        
        return {
            "messages": [],
            "total": 0,
            "has_more": False,
            "cursor": None
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )


@router.post("/messages/direct", response_model=DirectMessageInfo)
async def send_direct_message(
    request: SendDirectMessageRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send direct message to user"""
    try:
        # Implementation would go here
        # result = collaboration_service.send_direct_message(int(current_user["sub"]), request)
        
        return {
            "id": 1,
            "sender_id": int(current_user["sub"]),
            "sender_name": current_user.get("name", "Unknown"),
            "recipient_id": request.recipient_id,
            "recipient_name": "Unknown",
            "content": request.content,
            "is_read": False,
            "created_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error sending direct message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send direct message"
        )


# ================================
# NOTIFICATION ENDPOINTS
# ================================

@router.get("/notifications", response_model=NotificationListResponse)
async def get_notifications(
    unread_only: bool = Query(False, description="Get only unread notifications"),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get user notifications"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_notifications(int(current_user["sub"]), unread_only, limit)
        
        return {
            "notifications": [],
            "unread_count": 0,
            "total": 0
        }
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notifications"
        )


@router.post("/notifications/mark-read")
async def mark_notifications_read(
    request: MarkNotificationsReadRequest,
    current_user: dict = Depends(get_current_user)
):
    """Mark notifications as read"""
    try:
        # Implementation would go here
        # collaboration_service.mark_notifications_read(int(current_user["sub"]), request)
        
        return {"message": "Notifications marked as read"}
        
    except Exception as e:
        logger.error(f"Error marking notifications read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notifications as read"
        )


# ================================
# ACTIVITY AND ANALYTICS ENDPOINTS
# ================================

@router.get("/teams/{team_id}/activity", response_model=TeamActivityResponse)
async def get_team_activity(
    team_id: int,
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get team activity feed"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_team_activity(team_id, int(current_user["sub"]), limit)
        
        return {
            "activities": [],
            "total": 0,
            "has_more": False
        }
        
    except Exception as e:
        logger.error(f"Error getting team activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve team activity"
        )


@router.get("/teams/{team_id}/analytics", response_model=TeamAnalyticsResponse)
async def get_team_analytics(
    team_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get team analytics and statistics"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_team_analytics(team_id, int(current_user["sub"]))
        
        return {
            "member_count": 0,
            "active_members_7d": 0,
            "messages_sent_7d": 0,
            "portfolios_shared": 0,
            "strategies_shared": 0,
            "collaboration_score": 0.0,
            "activity_trend": []
        }
        
    except Exception as e:
        logger.error(f"Error getting team analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve team analytics"
        )


# ================================
# COMMENT ENDPOINTS
# ================================

@router.post("/comments", response_model=CommentInfo)
async def add_comment(
    request: AddCommentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Add comment to portfolio or strategy"""
    try:
        # Implementation would go here
        # result = collaboration_service.add_comment(int(current_user["sub"]), request)
        
        return {
            "id": 1,
            "author_id": int(current_user["sub"]),
            "author_name": current_user.get("name", "Unknown"),
            "content": request.content,
            "parent_comment_id": request.parent_comment_id,
            "mentions": request.mentions,
            "is_resolved": False,
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "reply_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add comment"
        )


@router.get("/comments", response_model=CommentListResponse)
async def get_comments(
    resource_type: str = Query(..., pattern="^(portfolio|strategy|backtest)$"),
    resource_id: int = Query(...),
    current_user: dict = Depends(get_current_user)
):
    """Get comments for portfolio or strategy"""
    try:
        # Implementation would go here
        # result = collaboration_service.get_comments(resource_type, resource_id, int(current_user["sub"]))
        
        return {
            "comments": [],
            "total": 0
        }
        
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve comments"
        )


@router.put("/comments/{comment_id}/resolve")
async def resolve_comment(
    comment_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Mark comment as resolved"""
    try:
        # Implementation would go here
        # collaboration_service.resolve_comment(comment_id, int(current_user["sub"]))
        
        return {"message": "Comment resolved"}
        
    except Exception as e:
        logger.error(f"Error resolving comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve comment"
        )


# ================================
# HEALTH CHECK ENDPOINT
# ================================

@router.get("/health")
async def collaboration_health():
    """Health check for collaboration service"""
    return {
        "status": "healthy",
        "service": "collaboration",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }