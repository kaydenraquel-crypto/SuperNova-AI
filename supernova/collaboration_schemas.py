"""
SuperNova AI Collaboration Schemas
Pydantic models for team collaboration, sharing, and communication API validation
"""

from pydantic import BaseModel, Field, EmailStr, validator, model_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .input_validation import validate_sql_safe, validate_xss_safe


# ================================
# COLLABORATION ENUMS
# ================================

class TeamRole(str, Enum):
    """Team role hierarchy"""
    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    VIEWER = "viewer"


class SharePermission(str, Enum):
    """Sharing permission levels"""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


class NotificationType(str, Enum):
    """Notification types"""
    TEAM_INVITE = "team_invite"
    TEAM_UPDATE = "team_update"
    PORTFOLIO_SHARED = "portfolio_shared"
    STRATEGY_SHARED = "strategy_shared"
    COMMENT_MENTION = "comment_mention"
    DIRECT_MESSAGE = "direct_message"
    ACTIVITY_UPDATE = "activity_update"
    SYSTEM_ALERT = "system_alert"


class ActivityType(str, Enum):
    """Activity feed types"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ROLE_CHANGED = "role_changed"
    PORTFOLIO_CREATED = "portfolio_created"
    PORTFOLIO_SHARED = "portfolio_shared"
    STRATEGY_CREATED = "strategy_created"
    COMMENT_ADDED = "comment_added"
    DOCUMENT_UPLOADED = "document_uploaded"


# ================================
# TEAM MANAGEMENT SCHEMAS
# ================================

class TeamCreateRequest(BaseModel):
    """Team creation request"""
    name: str = Field(..., min_length=3, max_length=100, description="Team name")
    description: Optional[str] = Field(None, max_length=1000, description="Team description")
    is_private: bool = Field(default=True, description="Private team visibility")
    max_members: int = Field(default=50, ge=5, le=1000, description="Maximum team members")
    generate_invite_code: bool = Field(default=False, description="Generate team invite code")
    
    @validator('name')
    def validate_name(cls, v):
        return validate_xss_safe(v.strip())
    
    @validator('description')
    def validate_description(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v


class TeamUpdateRequest(BaseModel):
    """Team update request"""
    name: Optional[str] = Field(None, min_length=3, max_length=100, description="Team name")
    description: Optional[str] = Field(None, max_length=1000, description="Team description")
    is_private: Optional[bool] = Field(None, description="Private team visibility")
    max_members: Optional[int] = Field(None, ge=5, le=1000, description="Maximum team members")
    
    @validator('name')
    def validate_name(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v


class TeamMemberInfo(BaseModel):
    """Team member information"""
    user_id: int = Field(..., description="User ID")
    name: str = Field(..., description="User name")
    email: EmailStr = Field(..., description="User email")
    role: TeamRole = Field(..., description="Team role")
    joined_at: datetime = Field(..., description="Join timestamp")
    invited_by: Optional[int] = Field(None, description="Inviter user ID")
    is_active: bool = Field(..., description="Active status")
    last_activity: Optional[datetime] = Field(None, description="Last activity")


class TeamInfo(BaseModel):
    """Team information response"""
    id: int = Field(..., description="Team ID")
    name: str = Field(..., description="Team name")
    description: Optional[str] = Field(None, description="Team description")
    is_private: bool = Field(..., description="Private visibility")
    member_count: int = Field(..., description="Current member count")
    max_members: int = Field(..., description="Maximum members allowed")
    created_by: int = Field(..., description="Creator user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    invite_code: Optional[str] = Field(None, description="Team invite code")
    your_role: Optional[TeamRole] = Field(None, description="Your role in team")


class TeamListResponse(BaseModel):
    """Team list response"""
    teams: List[TeamInfo] = Field(..., description="List of teams")
    total: int = Field(..., description="Total team count")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Items per page")


class TeamDetailResponse(BaseModel):
    """Detailed team information"""
    team: TeamInfo = Field(..., description="Team information")
    members: List[TeamMemberInfo] = Field(..., description="Team members")
    recent_activity: List["ActivityInfo"] = Field(..., description="Recent activities")
    statistics: Dict[str, Any] = Field(..., description="Team statistics")


# ================================
# TEAM INVITATION SCHEMAS
# ================================

class TeamInviteRequest(BaseModel):
    """Team invitation request"""
    emails: List[EmailStr] = Field(..., min_items=1, max_items=20, description="Email addresses")
    role: TeamRole = Field(default=TeamRole.MEMBER, description="Assigned role")
    message: Optional[str] = Field(None, max_length=500, description="Personal message")
    
    @validator('message')
    def validate_message(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v


class TeamInviteByCodeRequest(BaseModel):
    """Team join by invite code request"""
    invite_code: str = Field(..., min_length=6, max_length=20, description="Team invite code")


class TeamInviteResponse(BaseModel):
    """Team invitation response"""
    invitations_sent: int = Field(..., description="Number of invitations sent")
    failed_invitations: List[str] = Field(..., description="Failed email addresses")
    expires_at: datetime = Field(..., description="Invitation expiration")


class TeamInviteInfo(BaseModel):
    """Team invitation information"""
    id: int = Field(..., description="Invitation ID")
    team_name: str = Field(..., description="Team name")
    inviter_name: str = Field(..., description="Inviter name")
    role: TeamRole = Field(..., description="Assigned role")
    expires_at: datetime = Field(..., description="Expiration time")
    message: Optional[str] = Field(None, description="Personal message")


class PendingInvitationsResponse(BaseModel):
    """Pending invitations response"""
    invitations: List[TeamInviteInfo] = Field(..., description="Pending invitations")
    total: int = Field(..., description="Total pending count")


# ================================
# MEMBER MANAGEMENT SCHEMAS
# ================================

class UpdateMemberRoleRequest(BaseModel):
    """Update team member role request"""
    user_id: int = Field(..., description="Target user ID")
    new_role: TeamRole = Field(..., description="New role")
    
    @validator('new_role')
    def validate_role_change(cls, v):
        if v == TeamRole.OWNER:
            raise ValueError("Cannot assign owner role through this endpoint")
        return v


class RemoveMemberRequest(BaseModel):
    """Remove team member request"""
    user_id: int = Field(..., description="User ID to remove")
    reason: Optional[str] = Field(None, max_length=200, description="Removal reason")
    
    @validator('reason')
    def validate_reason(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v


class TransferOwnershipRequest(BaseModel):
    """Transfer team ownership request"""
    new_owner_id: int = Field(..., description="New owner user ID")
    confirm_transfer: bool = Field(..., description="Confirmation flag")
    
    @validator('confirm_transfer')
    def validate_confirmation(cls, v):
        if not v:
            raise ValueError("Transfer confirmation required")
        return v


# ================================
# SHARING & PERMISSIONS SCHEMAS
# ================================

class SharePortfolioRequest(BaseModel):
    """Portfolio sharing request"""
    portfolio_id: int = Field(..., description="Portfolio ID to share")
    target_type: str = Field(..., pattern="^(team|user|public)$", description="Share target type")
    target_id: Optional[int] = Field(None, description="Team or user ID")
    permission_level: SharePermission = Field(default=SharePermission.VIEW, description="Permission level")
    expires_at: Optional[datetime] = Field(None, description="Share expiration")
    password_protected: bool = Field(default=False, description="Password protection")
    password: Optional[str] = Field(None, min_length=6, description="Access password")
    
    @model_validator(mode='before')
    def validate_target(cls, values):
        target_type = values.get('target_type')
        target_id = values.get('target_id')
        
        if target_type in ['team', 'user'] and not target_id:
            raise ValueError(f"target_id required for {target_type} sharing")
        
        if target_type == 'public' and target_id:
            raise ValueError("target_id not allowed for public sharing")
        
        return values
    
    @model_validator(mode='before')
    def validate_password(cls, values):
        password_protected = values.get('password_protected', False)
        password = values.get('password')
        
        if password_protected and not password:
            raise ValueError("Password required when password_protected is True")
        
        return values


class ShareStrategyRequest(BaseModel):
    """Strategy sharing request"""
    strategy_id: int = Field(..., description="Strategy ID to share")
    target_type: str = Field(..., pattern="^(team|user|public)$", description="Share target type")
    target_id: Optional[int] = Field(None, description="Team or user ID")
    permission_level: SharePermission = Field(default=SharePermission.VIEW, description="Permission level")
    expires_at: Optional[datetime] = Field(None, description="Share expiration")
    include_backtest_results: bool = Field(default=True, description="Include backtest data")
    
    @model_validator(mode='before')
    def validate_target(cls, values):
        target_type = values.get('target_type')
        target_id = values.get('target_id')
        
        if target_type in ['team', 'user'] and not target_id:
            raise ValueError(f"target_id required for {target_type} sharing")
        
        if target_type == 'public' and target_id:
            raise ValueError("target_id not allowed for public sharing")
        
        return values


class CreateShareLinkRequest(BaseModel):
    """Create public share link request"""
    resource_type: str = Field(..., pattern="^(portfolio|strategy|analysis)$", description="Resource type")
    resource_id: int = Field(..., description="Resource ID")
    permission_level: SharePermission = Field(default=SharePermission.VIEW, description="Permission level")
    title: Optional[str] = Field(None, max_length=255, description="Share link title")
    description: Optional[str] = Field(None, max_length=1000, description="Share description")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    max_views: Optional[int] = Field(None, ge=1, description="Maximum view count")
    password_protected: bool = Field(default=False, description="Password protection")
    password: Optional[str] = Field(None, min_length=6, description="Access password")
    
    @validator('title', 'description')
    def validate_text_fields(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v
    
    @model_validator(mode='before')
    def validate_password(cls, values):
        password_protected = values.get('password_protected', False)
        password = values.get('password')
        
        if password_protected and not password:
            raise ValueError("Password required when password_protected is True")
        
        return values


class ShareLinkInfo(BaseModel):
    """Share link information"""
    id: int = Field(..., description="Share link ID")
    share_token: str = Field(..., description="Public share token")
    resource_type: str = Field(..., description="Resource type")
    resource_id: int = Field(..., description="Resource ID")
    title: Optional[str] = Field(None, description="Link title")
    permission_level: SharePermission = Field(..., description="Permission level")
    current_views: int = Field(..., description="Current view count")
    max_views: Optional[int] = Field(None, description="Maximum views")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    created_at: datetime = Field(..., description="Creation time")
    is_active: bool = Field(..., description="Active status")


class ShareLinkListResponse(BaseModel):
    """Share links list response"""
    links: List[ShareLinkInfo] = Field(..., description="Share links")
    total: int = Field(..., description="Total count")


class AccessShareLinkRequest(BaseModel):
    """Access share link request"""
    password: Optional[str] = Field(None, description="Access password if required")


# ================================
# COMMUNICATION SCHEMAS
# ================================

class CreateChatChannelRequest(BaseModel):
    """Create team chat channel request"""
    name: str = Field(..., min_length=3, max_length=100, description="Channel name")
    description: Optional[str] = Field(None, max_length=500, description="Channel description")
    is_private: bool = Field(default=False, description="Private channel")
    channel_type: str = Field(default="general", description="Channel type")
    
    @validator('name')
    def validate_name(cls, v):
        # Remove special characters and validate
        cleaned = validate_xss_safe(v.strip())
        if not cleaned.replace('-', '').replace('_', '').replace(' ', '').isalnum():
            raise ValueError("Channel name must contain only letters, numbers, hyphens, and underscores")
        return cleaned
    
    @validator('description')
    def validate_description(cls, v):
        if v:
            return validate_xss_safe(v.strip())
        return v


class ChatChannelInfo(BaseModel):
    """Chat channel information"""
    id: int = Field(..., description="Channel ID")
    name: str = Field(..., description="Channel name")
    description: Optional[str] = Field(None, description="Channel description")
    is_private: bool = Field(..., description="Private status")
    channel_type: str = Field(..., description="Channel type")
    message_count: int = Field(..., description="Total messages")
    last_message_at: Optional[datetime] = Field(None, description="Last message time")
    created_at: datetime = Field(..., description="Creation time")
    unread_count: Optional[int] = Field(None, description="Unread message count for user")


class SendMessageRequest(BaseModel):
    """Send chat message request"""
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    message_type: str = Field(default="text", description="Message type")
    thread_id: Optional[int] = Field(None, description="Parent message ID for threading")
    mentions: Optional[List[int]] = Field(None, description="Mentioned user IDs")
    
    @validator('content')
    def validate_content(cls, v):
        return validate_xss_safe(v.strip())


class ChatMessageInfo(BaseModel):
    """Chat message information"""
    id: int = Field(..., description="Message ID")
    author_id: int = Field(..., description="Author user ID")
    author_name: str = Field(..., description="Author name")
    content: str = Field(..., description="Message content")
    message_type: str = Field(..., description="Message type")
    thread_id: Optional[int] = Field(None, description="Parent message ID")
    mentions: Optional[List[int]] = Field(None, description="Mentioned user IDs")
    reactions: Optional[Dict[str, List[int]]] = Field(None, description="Message reactions")
    created_at: datetime = Field(..., description="Creation time")
    edited_at: Optional[datetime] = Field(None, description="Last edit time")
    is_deleted: bool = Field(..., description="Deletion status")


class ChatHistoryResponse(BaseModel):
    """Chat history response"""
    messages: List[ChatMessageInfo] = Field(..., description="Chat messages")
    total: int = Field(..., description="Total message count")
    has_more: bool = Field(..., description="More messages available")
    cursor: Optional[str] = Field(None, description="Pagination cursor")


class SendDirectMessageRequest(BaseModel):
    """Send direct message request"""
    recipient_id: int = Field(..., description="Recipient user ID")
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    message_type: str = Field(default="text", description="Message type")
    
    @validator('content')
    def validate_content(cls, v):
        return validate_xss_safe(v.strip())


class DirectMessageInfo(BaseModel):
    """Direct message information"""
    id: int = Field(..., description="Message ID")
    sender_id: int = Field(..., description="Sender user ID")
    sender_name: str = Field(..., description="Sender name")
    recipient_id: int = Field(..., description="Recipient user ID")
    recipient_name: str = Field(..., description="Recipient name")
    content: str = Field(..., description="Message content")
    is_read: bool = Field(..., description="Read status")
    created_at: datetime = Field(..., description="Creation time")


# ================================
# NOTIFICATION SCHEMAS
# ================================

class NotificationInfo(BaseModel):
    """Notification information"""
    id: int = Field(..., description="Notification ID")
    type: NotificationType = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    action_url: Optional[str] = Field(None, description="Action URL")
    is_read: bool = Field(..., description="Read status")
    created_at: datetime = Field(..., description="Creation time")


class NotificationListResponse(BaseModel):
    """Notification list response"""
    notifications: List[NotificationInfo] = Field(..., description="Notifications")
    unread_count: int = Field(..., description="Unread notification count")
    total: int = Field(..., description="Total notifications")


class MarkNotificationsReadRequest(BaseModel):
    """Mark notifications as read request"""
    notification_ids: Optional[List[int]] = Field(None, description="Specific notification IDs")
    mark_all: bool = Field(default=False, description="Mark all as read")
    
    @model_validator(mode='before')
    def validate_request(cls, values):
        notification_ids = values.get('notification_ids')
        mark_all = values.get('mark_all', False)
        
        if not mark_all and not notification_ids:
            raise ValueError("Either notification_ids or mark_all must be provided")
        
        return values


# ================================
# ACTIVITY & ANALYTICS SCHEMAS
# ================================

class ActivityInfo(BaseModel):
    """Activity information"""
    id: int = Field(..., description="Activity ID")
    user_id: int = Field(..., description="User ID")
    user_name: str = Field(..., description="User name")
    activity_type: ActivityType = Field(..., description="Activity type")
    title: str = Field(..., description="Activity title")
    description: Optional[str] = Field(None, description="Activity description")
    resource_type: Optional[str] = Field(None, description="Related resource type")
    resource_id: Optional[int] = Field(None, description="Related resource ID")
    created_at: datetime = Field(..., description="Activity timestamp")


class TeamActivityResponse(BaseModel):
    """Team activity feed response"""
    activities: List[ActivityInfo] = Field(..., description="Activities")
    total: int = Field(..., description="Total activity count")
    has_more: bool = Field(..., description="More activities available")


class TeamAnalyticsResponse(BaseModel):
    """Team analytics response"""
    member_count: int = Field(..., description="Total members")
    active_members_7d: int = Field(..., description="Active members (7 days)")
    messages_sent_7d: int = Field(..., description="Messages sent (7 days)")
    portfolios_shared: int = Field(..., description="Shared portfolios count")
    strategies_shared: int = Field(..., description="Shared strategies count")
    collaboration_score: float = Field(..., description="Team collaboration score")
    activity_trend: List[Dict[str, Any]] = Field(..., description="Activity trend data")


# ================================
# COMMENT SCHEMAS
# ================================

class AddCommentRequest(BaseModel):
    """Add comment request"""
    resource_type: str = Field(..., pattern="^(portfolio|strategy|backtest)$", description="Resource type")
    resource_id: int = Field(..., description="Resource ID")
    content: str = Field(..., min_length=1, max_length=5000, description="Comment content")
    parent_comment_id: Optional[int] = Field(None, description="Parent comment for replies")
    mentions: Optional[List[int]] = Field(None, description="Mentioned user IDs")
    
    @validator('content')
    def validate_content(cls, v):
        return validate_xss_safe(v.strip())


class CommentInfo(BaseModel):
    """Comment information"""
    id: int = Field(..., description="Comment ID")
    author_id: int = Field(..., description="Author user ID")
    author_name: str = Field(..., description="Author name")
    content: str = Field(..., description="Comment content")
    parent_comment_id: Optional[int] = Field(None, description="Parent comment ID")
    mentions: Optional[List[int]] = Field(None, description="Mentioned user IDs")
    is_resolved: bool = Field(..., description="Resolution status")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    reply_count: int = Field(default=0, description="Number of replies")


class CommentListResponse(BaseModel):
    """Comment list response"""
    comments: List[CommentInfo] = Field(..., description="Comments")
    total: int = Field(..., description="Total comment count")


# Forward reference resolution
ActivityInfo.model_rebuild()
TeamDetailResponse.model_rebuild()