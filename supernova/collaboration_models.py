"""
SuperNova AI Multi-User Collaboration Models
Comprehensive team management, sharing, and collaboration database models
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean, Table, Enum as SqlEnum
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import Optional, List
from enum import Enum
import uuid

from .db import Base


# ================================
# COLLABORATION ENUMS
# ================================

class TeamRole(str, Enum):
    """Team role hierarchy"""
    OWNER = "owner"          # Full control, can delete team
    ADMIN = "admin"          # Can manage members and permissions
    MODERATOR = "moderator"  # Can manage content and some members
    MEMBER = "member"        # Standard member access
    VIEWER = "viewer"        # Read-only access


class SharePermission(str, Enum):
    """Sharing permission levels"""
    VIEW = "view"            # Read-only access
    COMMENT = "comment"      # Can view and comment
    EDIT = "edit"           # Can modify content
    ADMIN = "admin"         # Can manage sharing settings


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
# ASSOCIATION TABLES
# ================================

team_members = Table(
    'team_members',
    Base.metadata,
    Column('team_id', Integer, ForeignKey('teams.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role', String(20), default=TeamRole.MEMBER.value),
    Column('joined_at', DateTime(timezone=True), server_default=func.now()),
    Column('invited_by', Integer, ForeignKey('users.id')),
    Column('is_active', Boolean, default=True),
    Column('permissions', Text)  # JSON string for custom permissions
)

portfolio_shares = Table(
    'portfolio_shares',
    Base.metadata,
    Column('portfolio_id', Integer, ForeignKey('portfolios.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('permission_level', String(20), default=SharePermission.VIEW.value),
    Column('shared_at', DateTime(timezone=True), server_default=func.now()),
    Column('shared_by', Integer, ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True)
)

strategy_shares = Table(
    'strategy_shares',
    Base.metadata,
    Column('strategy_id', Integer, ForeignKey('strategies.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('permission_level', String(20), default=SharePermission.VIEW.value),
    Column('shared_at', DateTime(timezone=True), server_default=func.now()),
    Column('shared_by', Integer, ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True)
)


# ================================
# TEAM MANAGEMENT MODELS
# ================================

class Team(Base):
    """Team/organization model for collaboration"""
    __tablename__ = "teams"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Team configuration
    is_private: Mapped[bool] = mapped_column(Boolean, default=True)
    max_members: Mapped[int] = mapped_column(Integer, default=50)
    invite_code: Mapped[str] = mapped_column(String(20), unique=True, nullable=True)
    
    # Ownership and management
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    parent_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=True)
    
    # Settings (JSON)
    settings: Mapped[str] = mapped_column(Text, nullable=True)  # JSON configuration
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    creator: Mapped["User"] = relationship("User", foreign_keys=[created_by])
    parent_team: Mapped["Team"] = relationship("Team", remote_side=[id])
    sub_teams: Mapped[List["Team"]] = relationship("Team", back_populates="parent_team")
    
    # Many-to-many relationships
    members: Mapped[List["User"]] = relationship(
        "User", 
        secondary=team_members,
        back_populates="teams",
        primaryjoin="Team.id == team_members.c.team_id",
        secondaryjoin="User.id == team_members.c.user_id"
    )
    
    # Related entities
    shared_portfolios: Mapped[List["SharedPortfolio"]] = relationship(back_populates="team")
    team_chats: Mapped[List["TeamChat"]] = relationship(back_populates="team")
    team_activities: Mapped[List["TeamActivity"]] = relationship(back_populates="team")


class TeamInvitation(Base):
    """Team invitation management"""
    __tablename__ = "team_invitations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    
    # Invitation details
    email: Mapped[str] = mapped_column(String(255), index=True)
    invited_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String(20), default=TeamRole.MEMBER.value)
    
    # Token and expiration
    invitation_token: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    
    # Status
    is_accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    accepted_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Personal message
    message: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Relationships
    team: Mapped["Team"] = relationship()
    inviter: Mapped["User"] = relationship(foreign_keys=[invited_by])
    acceptor: Mapped["User"] = relationship(foreign_keys=[accepted_by])


# ================================
# SHARING & PERMISSIONS MODELS
# ================================

class SharedPortfolio(Base):
    """Portfolio sharing with teams and individuals"""
    __tablename__ = "shared_portfolios"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    
    # Sharing targets (either team or individual)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    
    # Sharing configuration
    permission_level: Mapped[str] = mapped_column(String(20), default=SharePermission.VIEW.value)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    public_share_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=True)
    
    # Access control
    password_protected: Mapped[bool] = mapped_column(Boolean, default=False)
    access_password: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Sharing metadata
    shared_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship()
    team: Mapped["Team"] = relationship(back_populates="shared_portfolios")
    shared_user: Mapped["User"] = relationship(foreign_keys=[user_id])
    sharer: Mapped["User"] = relationship(foreign_keys=[shared_by])


class ShareLink(Base):
    """Public share links with access controls"""
    __tablename__ = "share_links"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Link configuration
    share_token: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    resource_type: Mapped[str] = mapped_column(String(50))  # portfolio, strategy, analysis
    resource_id: Mapped[int] = mapped_column(Integer, index=True)
    
    # Access configuration
    permission_level: Mapped[str] = mapped_column(String(20), default=SharePermission.VIEW.value)
    password_protected: Mapped[bool] = mapped_column(Boolean, default=False)
    access_password: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Limitations
    max_views: Mapped[int] = mapped_column(Integer, nullable=True)
    current_views: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Analytics
    unique_visitors: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    creator: Mapped["User"] = relationship()


# ================================
# COMMUNICATION MODELS
# ================================

class TeamChat(Base):
    """Team chat/discussion channels"""
    __tablename__ = "team_chats"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    
    # Channel information
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, nullable=True)
    is_private: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Channel type
    channel_type: Mapped[str] = mapped_column(String(20), default="general")  # general, portfolio, strategy
    resource_id: Mapped[int] = mapped_column(Integer, nullable=True)  # linked portfolio/strategy
    
    # Management
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    last_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    team: Mapped["Team"] = relationship(back_populates="team_chats")
    creator: Mapped["User"] = relationship()
    messages: Mapped[List["ChatMessage"]] = relationship(back_populates="chat")


class ChatMessage(Base):
    """Chat messages within team channels"""
    __tablename__ = "chat_messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("team_chats.id"), index=True)
    
    # Message content
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    message_type: Mapped[str] = mapped_column(String(20), default="text")  # text, file, image, system
    
    # Message metadata
    thread_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id"), nullable=True)
    mentions: Mapped[str] = mapped_column(Text, nullable=True)  # JSON array of mentioned user IDs
    attachments: Mapped[str] = mapped_column(Text, nullable=True)  # JSON array of file info
    
    # Editing and reactions
    edited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    reactions: Mapped[str] = mapped_column(Text, nullable=True)  # JSON object
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Status
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    chat: Mapped["TeamChat"] = relationship(back_populates="messages")
    author: Mapped["User"] = relationship()
    parent_message: Mapped["ChatMessage"] = relationship(remote_side=[id])
    replies: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="parent_message")


class DirectMessage(Base):
    """Direct messages between users"""
    __tablename__ = "direct_messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Participants
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    recipient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Message content
    content: Mapped[str] = mapped_column(Text)
    message_type: Mapped[str] = mapped_column(String(20), default="text")
    
    # Message status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Status
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Relationships
    sender: Mapped["User"] = relationship(foreign_keys=[sender_id])
    recipient: Mapped["User"] = relationship(foreign_keys=[recipient_id])


# ================================
# NOTIFICATION MODELS
# ================================

class Notification(Base):
    """User notification system"""
    __tablename__ = "notifications"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Notification content
    type: Mapped[str] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    
    # Notification data
    data: Mapped[str] = mapped_column(Text, nullable=True)  # JSON with additional context
    action_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Related entities
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=True)
    related_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Delivery channels
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    push_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    team: Mapped["Team"] = relationship()
    related_user: Mapped["User"] = relationship(foreign_keys=[related_user_id])


# ================================
# ACTIVITY & AUDIT MODELS
# ================================

class TeamActivity(Base):
    """Team activity feed"""
    __tablename__ = "team_activities"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    
    # Activity details
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    activity_type: Mapped[str] = mapped_column(String(50), index=True)
    
    # Activity content
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, nullable=True)
    activity_metadata: Mapped[str] = mapped_column(Text, nullable=True)  # JSON with activity data
    
    # Related entities
    resource_type: Mapped[str] = mapped_column(String(50), nullable=True)  # portfolio, strategy, etc.
    resource_id: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Visibility
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    team: Mapped["Team"] = relationship(back_populates="team_activities")
    user: Mapped["User"] = relationship()


class CollaborationAudit(Base):
    """Audit trail for collaboration actions"""
    __tablename__ = "collaboration_audits"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Audit details
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    resource_type: Mapped[str] = mapped_column(String(50))
    resource_id: Mapped[int] = mapped_column(Integer)
    
    # Audit metadata
    details: Mapped[str] = mapped_column(Text, nullable=True)  # JSON with action details
    ip_address: Mapped[str] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Context
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user: Mapped["User"] = relationship()
    team: Mapped["Team"] = relationship()


# ================================
# PORTFOLIO COLLABORATION MODELS
# ================================

class PortfolioComment(Base):
    """Comments on portfolios and strategies"""
    __tablename__ = "portfolio_comments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Comment target
    resource_type: Mapped[str] = mapped_column(String(20))  # portfolio, strategy, backtest
    resource_id: Mapped[int] = mapped_column(Integer, index=True)
    
    # Comment content
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    
    # Threading
    parent_comment_id: Mapped[int] = mapped_column(ForeignKey("portfolio_comments.id"), nullable=True)
    
    # Metadata
    mentions: Mapped[str] = mapped_column(Text, nullable=True)  # JSON array of mentioned users
    attachments: Mapped[str] = mapped_column(Text, nullable=True)  # JSON array of attachments
    
    # Status
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    author: Mapped["User"] = relationship(foreign_keys=[user_id])
    parent_comment: Mapped["PortfolioComment"] = relationship(remote_side=[id])
    replies: Mapped[List["PortfolioComment"]] = relationship("PortfolioComment", back_populates="parent_comment")
    resolver: Mapped["User"] = relationship(foreign_keys=[resolved_by])


# ================================
# UPDATE USER MODEL WITH COLLABORATION
# ================================

# Add this to the User model in db.py via relationship back_populates
# This will be handled in the integration step

# Ensure all models are properly registered with the Base
__all__ = [
    'Team', 'TeamInvitation', 'SharedPortfolio', 'ShareLink',
    'TeamChat', 'ChatMessage', 'DirectMessage', 'Notification',
    'TeamActivity', 'CollaborationAudit', 'PortfolioComment',
    'TeamRole', 'SharePermission', 'NotificationType', 'ActivityType'
]