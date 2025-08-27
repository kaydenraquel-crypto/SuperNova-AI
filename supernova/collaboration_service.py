"""
SuperNova AI Collaboration Service Layer
Business logic for team management, sharing, and communication features
"""

import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, or_, func, desc, text
from fastapi import HTTPException, status

# Database imports
from .db import SessionLocal, User
from .collaboration_models import (
    Team, TeamInvitation, SharedPortfolio, ShareLink, TeamChat, ChatMessage,
    DirectMessage, Notification, TeamActivity, CollaborationAudit, PortfolioComment,
    TeamRole, SharePermission, NotificationType, ActivityType,
    team_members, portfolio_shares, strategy_shares
)
from .analytics_models import Portfolio
from .schemas import Strategy

# Collaboration imports
from .collaboration_schemas import (
    TeamCreateRequest, TeamUpdateRequest, TeamInviteRequest, SharePortfolioRequest,
    CreateShareLinkRequest, SendMessageRequest, AddCommentRequest
)

# Auth and security imports
from .auth import auth_manager, Permission, UserRole
from .security_logger import SecurityLogger

import logging

logger = logging.getLogger(__name__)
security_logger = SecurityLogger()


class CollaborationError(Exception):
    """Base collaboration error"""
    pass


class InsufficientPermissionsError(CollaborationError):
    """User lacks required permissions"""
    pass


class TeamNotFoundError(CollaborationError):
    """Team not found"""
    pass


class MembershipError(CollaborationError):
    """Team membership related error"""
    pass


class CollaborationService:
    """
    Comprehensive collaboration service for team management, sharing, and communication
    """
    
    def __init__(self):
        self.session_factory = SessionLocal
    
    # ================================
    # TEAM MANAGEMENT
    # ================================
    
    def create_team(self, user_id: int, request: TeamCreateRequest) -> Dict[str, Any]:
        """Create a new team"""
        db = self.session_factory()
        try:
            # Check user permissions
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Generate invite code if requested
            invite_code = None
            if request.generate_invite_code:
                invite_code = self._generate_invite_code()
            
            # Create team
            team = Team(
                name=request.name,
                description=request.description,
                is_private=request.is_private,
                max_members=request.max_members,
                invite_code=invite_code,
                created_by=user_id,
                settings=json.dumps({
                    "notifications_enabled": True,
                    "activity_visibility": "all",
                    "join_approval_required": request.is_private
                })
            )
            
            db.add(team)
            db.flush()  # Get the team ID
            
            # Add creator as owner
            team_member_insert = team_members.insert().values(
                team_id=team.id,
                user_id=user_id,
                role=TeamRole.OWNER.value,
                invited_by=user_id,
                permissions=json.dumps({
                    "manage_team": True,
                    "manage_members": True,
                    "manage_settings": True,
                    "delete_team": True
                })
            )
            db.execute(team_member_insert)
            
            # Log activity
            self._log_team_activity(
                db, team.id, user_id,
                ActivityType.USER_JOINED,
                f"Team '{team.name}' created",
                {"role": TeamRole.OWNER.value}
            )
            
            # Log audit
            self._log_collaboration_audit(
                db, user_id, "create_team", "team", team.id,
                {"team_name": team.name, "is_private": request.is_private}
            )
            
            db.commit()
            
            return {
                "id": team.id,
                "name": team.name,
                "description": team.description,
                "is_private": team.is_private,
                "invite_code": invite_code,
                "created_at": team.created_at,
                "your_role": TeamRole.OWNER.value
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating team: {e}")
            raise HTTPException(status_code=500, detail="Failed to create team")
        finally:
            db.close()
    
    def get_user_teams(self, user_id: int, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """Get teams for a user"""
        db = self.session_factory()
        try:
            # Query user's teams with role information
            query = db.query(
                Team,
                text("team_members.role").label("user_role"),
                text("team_members.joined_at").label("joined_at")
            ).join(
                team_members, Team.id == team_members.c.team_id
            ).filter(
                and_(
                    team_members.c.user_id == user_id,
                    team_members.c.is_active == True,
                    Team.is_active == True
                )
            ).order_by(desc(Team.created_at))
            
            # Pagination
            total = query.count()
            teams = query.offset((page - 1) * per_page).limit(per_page).all()
            
            # Format response
            team_list = []
            for team, role, joined_at in teams:
                # Get member count
                member_count = db.query(func.count(team_members.c.user_id)).filter(
                    and_(
                        team_members.c.team_id == team.id,
                        team_members.c.is_active == True
                    )
                ).scalar()
                
                team_list.append({
                    "id": team.id,
                    "name": team.name,
                    "description": team.description,
                    "is_private": team.is_private,
                    "member_count": member_count,
                    "max_members": team.max_members,
                    "created_by": team.created_by,
                    "created_at": team.created_at,
                    "your_role": role,
                    "joined_at": joined_at,
                    "invite_code": team.invite_code if role in [TeamRole.OWNER.value, TeamRole.ADMIN.value] else None
                })
            
            return {
                "teams": team_list,
                "total": total,
                "page": page,
                "per_page": per_page
            }
            
        except Exception as e:
            logger.error(f"Error getting user teams: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve teams")
        finally:
            db.close()
    
    def get_team_details(self, team_id: int, user_id: int) -> Dict[str, Any]:
        """Get detailed team information"""
        db = self.session_factory()
        try:
            # Check user membership
            user_role = self._get_user_team_role(db, team_id, user_id)
            if not user_role:
                raise HTTPException(status_code=403, detail="Not a team member")
            
            # Get team with member information
            team = db.query(Team).filter(Team.id == team_id, Team.is_active == True).first()
            if not team:
                raise TeamNotFoundError("Team not found")
            
            # Get members
            members_query = db.query(
                User.id, User.name, User.email,
                text("team_members.role").label("role"),
                text("team_members.joined_at").label("joined_at"),
                text("team_members.invited_by").label("invited_by"),
                text("team_members.is_active").label("is_active")
            ).join(
                team_members, User.id == team_members.c.user_id
            ).filter(team_members.c.team_id == team_id)
            
            members = []
            for user_data in members_query.all():
                members.append({
                    "user_id": user_data.id,
                    "name": user_data.name,
                    "email": user_data.email,
                    "role": user_data.role,
                    "joined_at": user_data.joined_at,
                    "invited_by": user_data.invited_by,
                    "is_active": user_data.is_active
                })
            
            # Get recent activities
            activities = self._get_team_activities(db, team_id, limit=10)
            
            # Get team statistics
            stats = self._get_team_statistics(db, team_id)
            
            return {
                "team": {
                    "id": team.id,
                    "name": team.name,
                    "description": team.description,
                    "is_private": team.is_private,
                    "member_count": len([m for m in members if m["is_active"]]),
                    "max_members": team.max_members,
                    "created_by": team.created_by,
                    "created_at": team.created_at,
                    "invite_code": team.invite_code if user_role in [TeamRole.OWNER.value, TeamRole.ADMIN.value] else None,
                    "your_role": user_role
                },
                "members": members,
                "recent_activity": activities,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error getting team details: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail="Failed to retrieve team details")
        finally:
            db.close()
    
    def invite_team_members(self, team_id: int, user_id: int, request: TeamInviteRequest) -> Dict[str, Any]:
        """Invite users to team"""
        db = self.session_factory()
        try:
            # Check permissions
            user_role = self._get_user_team_role(db, team_id, user_id)
            if user_role not in [TeamRole.OWNER.value, TeamRole.ADMIN.value, TeamRole.MODERATOR.value]:
                raise InsufficientPermissionsError("Insufficient permissions to invite members")
            
            team = db.query(Team).filter(Team.id == team_id).first()
            if not team:
                raise TeamNotFoundError("Team not found")
            
            successful_invites = 0
            failed_invites = []
            
            for email in request.emails:
                try:
                    # Check if user already exists
                    existing_user = db.query(User).filter(User.email == email).first()
                    
                    # Check if already a member
                    if existing_user:
                        existing_member = db.query(team_members).filter(
                            and_(
                                team_members.c.team_id == team_id,
                                team_members.c.user_id == existing_user.id,
                                team_members.c.is_active == True
                            )
                        ).first()
                        if existing_member:
                            failed_invites.append(f"{email} - Already a team member")
                            continue
                    
                    # Check for existing pending invitation
                    existing_invite = db.query(TeamInvitation).filter(
                        and_(
                            TeamInvitation.team_id == team_id,
                            TeamInvitation.email == email,
                            TeamInvitation.is_accepted == False,
                            TeamInvitation.expires_at > datetime.utcnow()
                        )
                    ).first()
                    
                    if existing_invite:
                        failed_invites.append(f"{email} - Invitation already pending")
                        continue
                    
                    # Create invitation
                    invitation = TeamInvitation(
                        team_id=team_id,
                        email=email,
                        invited_by=user_id,
                        role=request.role.value,
                        invitation_token=secrets.token_urlsafe(32),
                        expires_at=datetime.utcnow() + timedelta(days=7),
                        message=request.message
                    )
                    
                    db.add(invitation)
                    successful_invites += 1
                    
                    # Create notification if user exists
                    if existing_user:
                        self._create_notification(
                            db, existing_user.id,
                            NotificationType.TEAM_INVITE,
                            f"Invitation to join {team.name}",
                            f"You've been invited to join the team '{team.name}' as a {request.role.value}.",
                            {
                                "team_id": team_id,
                                "invitation_id": invitation.id,
                                "inviter_name": db.query(User.name).filter(User.id == user_id).scalar()
                            },
                            f"/teams/invitations/{invitation.invitation_token}"
                        )
                    
                    # TODO: Send email invitation (implement email service)
                    
                except Exception as e:
                    failed_invites.append(f"{email} - {str(e)}")
            
            # Log activity
            if successful_invites > 0:
                self._log_team_activity(
                    db, team_id, user_id,
                    ActivityType.USER_JOINED,
                    f"Invited {successful_invites} new members",
                    {"invited_emails": len(request.emails), "role": request.role.value}
                )
            
            db.commit()
            
            return {
                "invitations_sent": successful_invites,
                "failed_invitations": failed_invites,
                "expires_at": datetime.utcnow() + timedelta(days=7)
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error inviting team members: {e}")
            if isinstance(e, (InsufficientPermissionsError, TeamNotFoundError)):
                raise HTTPException(status_code=403 if isinstance(e, InsufficientPermissionsError) else 404, detail=str(e))
            raise HTTPException(status_code=500, detail="Failed to send invitations")
        finally:
            db.close()
    
    def join_team_by_code(self, user_id: int, invite_code: str) -> Dict[str, Any]:
        """Join team using invite code"""
        db = self.session_factory()
        try:
            # Find team by invite code
            team = db.query(Team).filter(
                and_(
                    Team.invite_code == invite_code,
                    Team.is_active == True
                )
            ).first()
            
            if not team:
                raise HTTPException(status_code=404, detail="Invalid invite code")
            
            # Check if already a member
            existing_member = db.query(team_members).filter(
                and_(
                    team_members.c.team_id == team.id,
                    team_members.c.user_id == user_id,
                    team_members.c.is_active == True
                )
            ).first()
            
            if existing_member:
                raise HTTPException(status_code=409, detail="Already a team member")
            
            # Check team capacity
            member_count = db.query(func.count(team_members.c.user_id)).filter(
                and_(
                    team_members.c.team_id == team.id,
                    team_members.c.is_active == True
                )
            ).scalar()
            
            if member_count >= team.max_members:
                raise HTTPException(status_code=409, detail="Team is at maximum capacity")
            
            # Add user to team
            team_member_insert = team_members.insert().values(
                team_id=team.id,
                user_id=user_id,
                role=TeamRole.MEMBER.value,
                invited_by=team.created_by  # Use team creator as inviter
            )
            db.execute(team_member_insert)
            
            # Log activity
            self._log_team_activity(
                db, team.id, user_id,
                ActivityType.USER_JOINED,
                f"Joined team via invite code",
                {"method": "invite_code", "role": TeamRole.MEMBER.value}
            )
            
            db.commit()
            
            return {
                "message": f"Successfully joined team '{team.name}'",
                "team_id": team.id,
                "team_name": team.name,
                "your_role": TeamRole.MEMBER.value
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error joining team by code: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail="Failed to join team")
        finally:
            db.close()
    
    # ================================
    # PORTFOLIO SHARING
    # ================================
    
    def share_portfolio(self, user_id: int, request: SharePortfolioRequest) -> Dict[str, Any]:
        """Share portfolio with team or user"""
        db = self.session_factory()
        try:
            # Verify portfolio ownership
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == request.portfolio_id,
                    Portfolio.user_id == user_id
                )
            ).first()
            
            if not portfolio:
                raise HTTPException(status_code=404, detail="Portfolio not found or not owned by user")
            
            # Handle different sharing types
            if request.target_type == "team":
                return self._share_portfolio_with_team(db, user_id, portfolio, request)
            elif request.target_type == "user":
                return self._share_portfolio_with_user(db, user_id, portfolio, request)
            elif request.target_type == "public":
                return self._create_public_portfolio_share(db, user_id, portfolio, request)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error sharing portfolio: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail="Failed to share portfolio")
        finally:
            db.close()
    
    # ================================
    # HELPER METHODS
    # ================================
    
    def _generate_invite_code(self) -> str:
        """Generate unique team invite code"""
        return secrets.token_hex(6).upper()
    
    def _get_user_team_role(self, db: Session, team_id: int, user_id: int) -> Optional[str]:
        """Get user's role in team"""
        result = db.query(text("role")).select_from(team_members).filter(
            and_(
                team_members.c.team_id == team_id,
                team_members.c.user_id == user_id,
                team_members.c.is_active == True
            )
        ).scalar()
        return result
    
    def _log_team_activity(self, db: Session, team_id: int, user_id: int, 
                          activity_type: ActivityType, title: str, metadata: Dict[str, Any]):
        """Log team activity"""
        activity = TeamActivity(
            team_id=team_id,
            user_id=user_id,
            activity_type=activity_type.value,
            title=title,
            metadata=json.dumps(metadata)
        )
        db.add(activity)
    
    def _log_collaboration_audit(self, db: Session, user_id: int, action: str, 
                               resource_type: str, resource_id: int, details: Dict[str, Any]):
        """Log collaboration audit trail"""
        audit = CollaborationAudit(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=json.dumps(details)
        )
        db.add(audit)
    
    def _create_notification(self, db: Session, user_id: int, notification_type: NotificationType,
                           title: str, message: str, data: Dict[str, Any], action_url: str = None):
        """Create user notification"""
        notification = Notification(
            user_id=user_id,
            type=notification_type.value,
            title=title,
            message=message,
            data=json.dumps(data),
            action_url=action_url
        )
        db.add(notification)
    
    def _get_team_activities(self, db: Session, team_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent team activities"""
        activities = db.query(TeamActivity, User.name).join(
            User, TeamActivity.user_id == User.id
        ).filter(TeamActivity.team_id == team_id).order_by(
            desc(TeamActivity.created_at)
        ).limit(limit).all()
        
        return [{
            "id": activity.id,
            "user_id": activity.user_id,
            "user_name": user_name,
            "activity_type": activity.activity_type,
            "title": activity.title,
            "description": activity.description,
            "created_at": activity.created_at
        } for activity, user_name in activities]
    
    def _get_team_statistics(self, db: Session, team_id: int) -> Dict[str, Any]:
        """Get team statistics"""
        # Member count
        member_count = db.query(func.count(team_members.c.user_id)).filter(
            and_(
                team_members.c.team_id == team_id,
                team_members.c.is_active == True
            )
        ).scalar()
        
        # Active members in last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        active_members = db.query(func.count(func.distinct(TeamActivity.user_id))).filter(
            and_(
                TeamActivity.team_id == team_id,
                TeamActivity.created_at >= seven_days_ago
            )
        ).scalar()
        
        # Shared portfolios count
        shared_portfolios = db.query(func.count(SharedPortfolio.id)).filter(
            SharedPortfolio.team_id == team_id
        ).scalar()
        
        return {
            "member_count": member_count or 0,
            "active_members_7d": active_members or 0,
            "shared_portfolios": shared_portfolios or 0,
            "collaboration_score": min(100, (active_members or 0) * 10 + (shared_portfolios or 0) * 5)
        }
    
    def _share_portfolio_with_team(self, db: Session, user_id: int, portfolio: Portfolio, 
                                 request: SharePortfolioRequest) -> Dict[str, Any]:
        """Share portfolio with specific team"""
        # Verify team membership
        user_role = self._get_user_team_role(db, request.target_id, user_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="Not a member of target team")
        
        # Check for existing share
        existing_share = db.query(SharedPortfolio).filter(
            and_(
                SharedPortfolio.portfolio_id == request.portfolio_id,
                SharedPortfolio.team_id == request.target_id,
                SharedPortfolio.is_active == True
            )
        ).first()
        
        if existing_share:
            raise HTTPException(status_code=409, detail="Portfolio already shared with this team")
        
        # Create share
        shared_portfolio = SharedPortfolio(
            portfolio_id=request.portfolio_id,
            team_id=request.target_id,
            permission_level=request.permission_level.value,
            shared_by=user_id,
            expires_at=request.expires_at,
            password_protected=request.password_protected,
            access_password=hashlib.sha256(request.password.encode()).hexdigest() if request.password else None
        )
        
        db.add(shared_portfolio)
        
        # Log activity
        team = db.query(Team).filter(Team.id == request.target_id).first()
        self._log_team_activity(
            db, request.target_id, user_id,
            ActivityType.PORTFOLIO_SHARED,
            f"Shared portfolio '{portfolio.name}' with team",
            {"portfolio_id": portfolio.id, "permission_level": request.permission_level.value}
        )
        
        db.commit()
        
        return {
            "message": f"Portfolio shared with team '{team.name}'",
            "share_id": shared_portfolio.id,
            "permission_level": request.permission_level.value,
            "expires_at": request.expires_at
        }
    
    def _share_portfolio_with_user(self, db: Session, user_id: int, portfolio: Portfolio, 
                                 request: SharePortfolioRequest) -> Dict[str, Any]:
        """Share portfolio with specific user"""
        # Verify target user exists
        target_user = db.query(User).filter(User.id == request.target_id).first()
        if not target_user:
            raise HTTPException(status_code=404, detail="Target user not found")
        
        # Check for existing share
        existing_share = db.query(SharedPortfolio).filter(
            and_(
                SharedPortfolio.portfolio_id == request.portfolio_id,
                SharedPortfolio.user_id == request.target_id,
                SharedPortfolio.is_active == True
            )
        ).first()
        
        if existing_share:
            raise HTTPException(status_code=409, detail="Portfolio already shared with this user")
        
        # Create share
        shared_portfolio = SharedPortfolio(
            portfolio_id=request.portfolio_id,
            user_id=request.target_id,
            permission_level=request.permission_level.value,
            shared_by=user_id,
            expires_at=request.expires_at,
            password_protected=request.password_protected,
            access_password=hashlib.sha256(request.password.encode()).hexdigest() if request.password else None
        )
        
        db.add(shared_portfolio)
        
        # Create notification
        self._create_notification(
            db, request.target_id,
            NotificationType.PORTFOLIO_SHARED,
            f"Portfolio '{portfolio.name}' shared with you",
            f"A portfolio has been shared with you by {db.query(User.name).filter(User.id == user_id).scalar()}.",
            {"portfolio_id": portfolio.id, "shared_by": user_id, "permission_level": request.permission_level.value},
            f"/portfolios/{portfolio.id}"
        )
        
        db.commit()
        
        return {
            "message": f"Portfolio shared with user '{target_user.name}'",
            "share_id": shared_portfolio.id,
            "permission_level": request.permission_level.value,
            "expires_at": request.expires_at
        }
    
    def _create_public_portfolio_share(self, db: Session, user_id: int, portfolio: Portfolio, 
                                     request: SharePortfolioRequest) -> Dict[str, Any]:
        """Create public portfolio share link"""
        share_token = secrets.token_urlsafe(16)
        
        # Create public share
        shared_portfolio = SharedPortfolio(
            portfolio_id=request.portfolio_id,
            permission_level=request.permission_level.value,
            is_public=True,
            public_share_token=share_token,
            shared_by=user_id,
            expires_at=request.expires_at,
            password_protected=request.password_protected,
            access_password=hashlib.sha256(request.password.encode()).hexdigest() if request.password else None
        )
        
        db.add(shared_portfolio)
        
        # Log audit
        self._log_collaboration_audit(
            db, user_id, "create_public_share", "portfolio", portfolio.id,
            {"permission_level": request.permission_level.value, "expires_at": request.expires_at.isoformat() if request.expires_at else None}
        )
        
        db.commit()
        
        return {
            "message": "Public share link created",
            "share_token": share_token,
            "share_url": f"/shared/portfolio/{share_token}",
            "permission_level": request.permission_level.value,
            "expires_at": request.expires_at
        }


# Global service instance
collaboration_service = CollaborationService()