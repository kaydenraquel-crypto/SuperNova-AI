"""
SuperNova AI Collaboration WebSocket Handler
Real-time communication and collaboration features via WebSocket
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

# Database imports
from .db import SessionLocal, User
from .collaboration_models import (
    Team, TeamChat, ChatMessage, DirectMessage, Notification,
    TeamRole, team_members
)
from .websocket_handler import WebSocketChatHandler, WebSocketConnection, MessageType, UserStatus

logger = logging.getLogger(__name__)


@dataclass
class CollaborationSession:
    """Extended session for collaboration features"""
    user_id: int
    team_ids: Set[int] = field(default_factory=set)
    channel_ids: Set[int] = field(default_factory=set)
    portfolio_ids: Set[int] = field(default_factory=set)
    permissions: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)


class CollaborationWebSocketHandler(WebSocketChatHandler):
    """
    Extended WebSocket handler with collaboration features
    Adds team chat, portfolio collaboration, and real-time notifications
    """
    
    def __init__(self):
        super().__init__()
        
        # Collaboration-specific connection tracking
        self.team_connections: Dict[int, Set[str]] = {}          # team_id -> connection_ids
        self.channel_connections: Dict[int, Set[str]] = {}       # channel_id -> connection_ids
        self.portfolio_watchers: Dict[int, Set[str]] = {}        # portfolio_id -> connection_ids
        
        # User presence tracking
        self.user_status: Dict[int, UserStatus] = {}             # user_id -> status
        self.user_team_presence: Dict[int, Dict[int, datetime]] = {}  # user_id -> {team_id: last_seen}
        
        # Collaboration sessions
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}  # connection_id -> session
    
    async def connect_user_collaboration(self, websocket: WebSocket, user_id: int, 
                                       team_ids: List[int] = None) -> str:
        """
        Connect user for collaboration with team context
        """
        try:
            connection_id = await self.connect(websocket, str(user_id))
            
            # Create collaboration session
            session = CollaborationSession(
                user_id=user_id,
                team_ids=set(team_ids or [])
            )
            
            # Load user's teams and permissions
            await self._load_user_teams(session, user_id)
            
            # Subscribe to team channels
            await self._subscribe_to_teams(connection_id, session.team_ids)
            
            # Update user presence
            self.user_status[user_id] = UserStatus.ONLINE
            
            # Store session
            self.collaboration_sessions[connection_id] = session
            
            # Notify teams of user coming online
            await self._broadcast_presence_update(user_id, UserStatus.ONLINE, list(session.team_ids))
            
            logger.info(f"User {user_id} connected for collaboration with teams: {session.team_ids}")
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting user for collaboration: {e}")
            raise HTTPException(status_code=500, detail="Failed to establish collaboration connection")
    
    async def disconnect_user_collaboration(self, connection_id: str):
        """
        Disconnect user from collaboration
        """
        try:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            user_id = int(connection.user_id)
            
            # Get collaboration session
            session = self.collaboration_sessions.get(connection_id)
            
            # Clean up subscriptions
            if session:
                await self._unsubscribe_from_teams(connection_id, session.team_ids)
                
                # Update presence
                self.user_status[user_id] = UserStatus.OFFLINE
                
                # Notify teams of user going offline
                await self._broadcast_presence_update(user_id, UserStatus.OFFLINE, list(session.team_ids))
                
                # Remove session
                del self.collaboration_sessions[connection_id]
            
            # Call parent disconnect
            await self.disconnect(connection_id)
            
            logger.info(f"User {user_id} disconnected from collaboration")
            
        except Exception as e:
            logger.error(f"Error disconnecting user from collaboration: {e}")
    
    async def handle_collaboration_message(self, connection_id: str, message: Dict[str, Any]):
        """
        Handle collaboration-specific messages
        """
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            if message_type == "join_team":
                await self._handle_join_team(connection_id, data)
            elif message_type == "leave_team":
                await self._handle_leave_team(connection_id, data)
            elif message_type == "join_channel":
                await self._handle_join_channel(connection_id, data)
            elif message_type == "leave_channel":
                await self._handle_leave_channel(connection_id, data)
            elif message_type == "team_message":
                await self._handle_team_message(connection_id, data)
            elif message_type == "direct_message":
                await self._handle_direct_message(connection_id, data)
            elif message_type == "typing_indicator":
                await self._handle_typing_indicator(connection_id, data)
            elif message_type == "portfolio_collaboration":
                await self._handle_portfolio_collaboration(connection_id, data)
            elif message_type == "presence_update":
                await self._handle_presence_update(connection_id, data)
            else:
                # Pass to parent handler for standard messages
                await super().handle_message(connection_id, message)
                
        except Exception as e:
            logger.error(f"Error handling collaboration message: {e}")
            await self._send_error(connection_id, f"Failed to process message: {str(e)}")
    
    # ================================
    # TEAM COLLABORATION HANDLERS
    # ================================
    
    async def _handle_join_team(self, connection_id: str, data: Dict[str, Any]):
        """Handle user joining team chat"""
        try:
            team_id = data.get("team_id")
            if not team_id:
                await self._send_error(connection_id, "Team ID required")
                return
            
            session = self.collaboration_sessions.get(connection_id)
            if not session:
                await self._send_error(connection_id, "No collaboration session")
                return
            
            # Verify team membership
            if not await self._verify_team_membership(session.user_id, team_id):
                await self._send_error(connection_id, "Not a team member")
                return
            
            # Add to team connections
            if team_id not in self.team_connections:
                self.team_connections[team_id] = set()
            self.team_connections[team_id].add(connection_id)
            
            # Add to session
            session.team_ids.add(team_id)
            session.last_activity = datetime.now()
            
            # Confirm joining
            await self._send_message(connection_id, {
                "type": "team_joined",
                "data": {
                    "team_id": team_id,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Notify other team members
            await self._broadcast_to_team(team_id, {
                "type": "user_joined_team",
                "data": {
                    "user_id": session.user_id,
                    "team_id": team_id,
                    "timestamp": datetime.now().isoformat()
                }
            }, exclude_connection=connection_id)
            
        except Exception as e:
            logger.error(f"Error handling join team: {e}")
            await self._send_error(connection_id, "Failed to join team")
    
    async def _handle_leave_team(self, connection_id: str, data: Dict[str, Any]):
        """Handle user leaving team chat"""
        try:
            team_id = data.get("team_id")
            if not team_id:
                return
            
            session = self.collaboration_sessions.get(connection_id)
            if not session:
                return
            
            # Remove from team connections
            if team_id in self.team_connections:
                self.team_connections[team_id].discard(connection_id)
                if not self.team_connections[team_id]:
                    del self.team_connections[team_id]
            
            # Remove from session
            session.team_ids.discard(team_id)
            
            # Confirm leaving
            await self._send_message(connection_id, {
                "type": "team_left",
                "data": {
                    "team_id": team_id,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Notify other team members
            await self._broadcast_to_team(team_id, {
                "type": "user_left_team",
                "data": {
                    "user_id": session.user_id,
                    "team_id": team_id,
                    "timestamp": datetime.now().isoformat()
                }
            }, exclude_connection=connection_id)
            
        except Exception as e:
            logger.error(f"Error handling leave team: {e}")
    
    async def _handle_team_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle team chat message"""
        try:
            team_id = data.get("team_id")
            channel_id = data.get("channel_id")
            content = data.get("content", "").strip()
            message_type = data.get("message_type", "text")
            
            if not team_id or not content:
                await self._send_error(connection_id, "Team ID and content required")
                return
            
            session = self.collaboration_sessions.get(connection_id)
            if not session or team_id not in session.team_ids:
                await self._send_error(connection_id, "Not connected to team")
                return
            
            # Save message to database
            message_id = await self._save_team_message(
                team_id, channel_id, session.user_id, content, message_type, data
            )
            
            # Create broadcast message
            broadcast_data = {
                "type": "team_message",
                "data": {
                    "message_id": message_id,
                    "team_id": team_id,
                    "channel_id": channel_id,
                    "user_id": session.user_id,
                    "content": content,
                    "message_type": message_type,
                    "timestamp": datetime.now().isoformat(),
                    "mentions": data.get("mentions", [])
                }
            }
            
            # Broadcast to team members
            if channel_id:
                await self._broadcast_to_channel(channel_id, broadcast_data)
            else:
                await self._broadcast_to_team(team_id, broadcast_data)
            
            # Handle mentions (send notifications)
            mentions = data.get("mentions", [])
            if mentions:
                await self._handle_message_mentions(team_id, session.user_id, mentions, content)
            
        except Exception as e:
            logger.error(f"Error handling team message: {e}")
            await self._send_error(connection_id, "Failed to send message")
    
    async def _handle_direct_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle direct message between users"""
        try:
            recipient_id = data.get("recipient_id")
            content = data.get("content", "").strip()
            message_type = data.get("message_type", "text")
            
            if not recipient_id or not content:
                await self._send_error(connection_id, "Recipient ID and content required")
                return
            
            session = self.collaboration_sessions.get(connection_id)
            if not session:
                await self._send_error(connection_id, "No collaboration session")
                return
            
            # Save direct message
            message_id = await self._save_direct_message(
                session.user_id, recipient_id, content, message_type
            )
            
            # Create message data
            message_data = {
                "type": "direct_message",
                "data": {
                    "message_id": message_id,
                    "sender_id": session.user_id,
                    "recipient_id": recipient_id,
                    "content": content,
                    "message_type": message_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Send to recipient if online
            recipient_connections = self._get_user_connections(recipient_id)
            for conn_id in recipient_connections:
                await self._send_message(conn_id, message_data)
            
            # Confirm to sender
            await self._send_message(connection_id, {
                "type": "direct_message_sent",
                "data": {
                    "message_id": message_id,
                    "recipient_id": recipient_id,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Create notification if recipient is offline
            if not recipient_connections:
                await self._create_direct_message_notification(
                    session.user_id, recipient_id, content
                )
            
        except Exception as e:
            logger.error(f"Error handling direct message: {e}")
            await self._send_error(connection_id, "Failed to send direct message")
    
    async def _handle_portfolio_collaboration(self, connection_id: str, data: Dict[str, Any]):
        """Handle portfolio collaboration events"""
        try:
            portfolio_id = data.get("portfolio_id")
            action = data.get("action")  # view, edit, comment, etc.
            
            if not portfolio_id or not action:
                await self._send_error(connection_id, "Portfolio ID and action required")
                return
            
            session = self.collaboration_sessions.get(connection_id)
            if not session:
                await self._send_error(connection_id, "No collaboration session")
                return
            
            # Add to portfolio watchers
            if portfolio_id not in self.portfolio_watchers:
                self.portfolio_watchers[portfolio_id] = set()
            self.portfolio_watchers[portfolio_id].add(connection_id)
            
            # Broadcast collaboration event
            await self._broadcast_to_portfolio_watchers(portfolio_id, {
                "type": "portfolio_collaboration",
                "data": {
                    "portfolio_id": portfolio_id,
                    "user_id": session.user_id,
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    **data.get("metadata", {})
                }
            }, exclude_connection=connection_id)
            
        except Exception as e:
            logger.error(f"Error handling portfolio collaboration: {e}")
    
    # ================================
    # BROADCAST METHODS
    # ================================
    
    async def _broadcast_to_team(self, team_id: int, message: Dict[str, Any], 
                               exclude_connection: str = None):
        """Broadcast message to all team members"""
        if team_id not in self.team_connections:
            return
        
        connections = self.team_connections[team_id].copy()
        if exclude_connection:
            connections.discard(exclude_connection)
        
        for connection_id in connections:
            try:
                await self._send_message(connection_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection {connection_id}: {e}")
                # Remove failed connection
                self.team_connections[team_id].discard(connection_id)
    
    async def _broadcast_to_channel(self, channel_id: int, message: Dict[str, Any], 
                                  exclude_connection: str = None):
        """Broadcast message to channel subscribers"""
        if channel_id not in self.channel_connections:
            return
        
        connections = self.channel_connections[channel_id].copy()
        if exclude_connection:
            connections.discard(exclude_connection)
        
        for connection_id in connections:
            try:
                await self._send_message(connection_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to channel connection {connection_id}: {e}")
                # Remove failed connection
                self.channel_connections[channel_id].discard(connection_id)
    
    async def _broadcast_to_portfolio_watchers(self, portfolio_id: int, message: Dict[str, Any], 
                                             exclude_connection: str = None):
        """Broadcast to portfolio collaboration watchers"""
        if portfolio_id not in self.portfolio_watchers:
            return
        
        connections = self.portfolio_watchers[portfolio_id].copy()
        if exclude_connection:
            connections.discard(exclude_connection)
        
        for connection_id in connections:
            try:
                await self._send_message(connection_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to portfolio watcher {connection_id}: {e}")
                # Remove failed connection
                self.portfolio_watchers[portfolio_id].discard(connection_id)
    
    async def _broadcast_presence_update(self, user_id: int, status: UserStatus, team_ids: List[int]):
        """Broadcast user presence update to teams"""
        presence_message = {
            "type": "presence_update",
            "data": {
                "user_id": user_id,
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for team_id in team_ids:
            await self._broadcast_to_team(team_id, presence_message)
    
    # ================================
    # DATABASE OPERATIONS
    # ================================
    
    async def _save_team_message(self, team_id: int, channel_id: Optional[int], 
                                user_id: int, content: str, message_type: str, 
                                data: Dict[str, Any]) -> int:
        """Save team message to database"""
        db = SessionLocal()
        try:
            message = ChatMessage(
                chat_id=channel_id or team_id,  # Use team_id if no specific channel
                user_id=user_id,
                content=content,
                message_type=message_type,
                thread_id=data.get("thread_id"),
                mentions=json.dumps(data.get("mentions", [])),
                attachments=json.dumps(data.get("attachments", []))
            )
            
            db.add(message)
            db.commit()
            db.refresh(message)
            
            return message.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving team message: {e}")
            raise
        finally:
            db.close()
    
    async def _save_direct_message(self, sender_id: int, recipient_id: int, 
                                 content: str, message_type: str) -> int:
        """Save direct message to database"""
        db = SessionLocal()
        try:
            message = DirectMessage(
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=content,
                message_type=message_type
            )
            
            db.add(message)
            db.commit()
            db.refresh(message)
            
            return message.id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving direct message: {e}")
            raise
        finally:
            db.close()
    
    # ================================
    # HELPER METHODS
    # ================================
    
    async def _load_user_teams(self, session: CollaborationSession, user_id: int):
        """Load user's teams and set permissions"""
        db = SessionLocal()
        try:
            # Get user's teams
            teams = db.query(Team.id, text("team_members.role")).select_from(
                Team
            ).join(
                team_members, Team.id == team_members.c.team_id
            ).filter(
                and_(
                    team_members.c.user_id == user_id,
                    team_members.c.is_active == True,
                    Team.is_active == True
                )
            ).all()
            
            for team_id, role in teams:
                session.team_ids.add(team_id)
                session.permissions[f"team_{team_id}"] = {
                    "role": role,
                    "can_moderate": role in [TeamRole.OWNER.value, TeamRole.ADMIN.value, TeamRole.MODERATOR.value],
                    "can_manage": role in [TeamRole.OWNER.value, TeamRole.ADMIN.value]
                }
            
        except Exception as e:
            logger.error(f"Error loading user teams: {e}")
        finally:
            db.close()
    
    async def _subscribe_to_teams(self, connection_id: str, team_ids: Set[int]):
        """Subscribe connection to team channels"""
        for team_id in team_ids:
            if team_id not in self.team_connections:
                self.team_connections[team_id] = set()
            self.team_connections[team_id].add(connection_id)
    
    async def _unsubscribe_from_teams(self, connection_id: str, team_ids: Set[int]):
        """Unsubscribe connection from team channels"""
        for team_id in team_ids:
            if team_id in self.team_connections:
                self.team_connections[team_id].discard(connection_id)
                if not self.team_connections[team_id]:
                    del self.team_connections[team_id]
    
    async def _verify_team_membership(self, user_id: int, team_id: int) -> bool:
        """Verify user is a team member"""
        db = SessionLocal()
        try:
            member = db.query(team_members).filter(
                and_(
                    team_members.c.team_id == team_id,
                    team_members.c.user_id == user_id,
                    team_members.c.is_active == True
                )
            ).first()
            
            return member is not None
            
        except Exception as e:
            logger.error(f"Error verifying team membership: {e}")
            return False
        finally:
            db.close()
    
    def _get_user_connections(self, user_id: int) -> List[str]:
        """Get all connections for a user"""
        connections = []
        for conn_id, session in self.collaboration_sessions.items():
            if session.user_id == user_id:
                connections.append(conn_id)
        return connections
    
    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            try:
                await connection.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        await self._send_message(connection_id, {
            "type": "error",
            "data": {
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        })


# Global collaboration WebSocket handler instance
collaboration_websocket_handler = CollaborationWebSocketHandler()