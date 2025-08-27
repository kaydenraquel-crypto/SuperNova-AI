"""
WebSocket Handler for Real-time Chat Communication
Provides comprehensive WebSocket management for SuperNova chat system
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    CHAT = "chat"
    TYPING = "typing"
    PRESENCE = "presence"
    SYSTEM = "system"
    NOTIFICATION = "notification"
    CHART_DATA = "chart_data"
    MARKET_UPDATE = "market_update"
    VOICE_MESSAGE = "voice_message"
    FILE_SHARE = "file_share"
    SCREEN_SHARE = "screen_share"

class UserStatus(str, Enum):
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection with metadata"""
    websocket: WebSocket
    user_id: str
    session_id: str
    profile_id: Optional[int]
    connected_at: datetime
    last_activity: datetime
    status: UserStatus = UserStatus.ONLINE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert connection to dictionary for serialization"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata
        }

class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None

class WebSocketChatHandler:
    """
    Comprehensive WebSocket handler for real-time chat functionality
    Supports multiple users, sessions, and advanced features
    """
    
    def __init__(self):
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}  # connection_id -> connection
        self.session_connections: Dict[str, Set[str]] = {}    # session_id -> set of connection_ids
        self.user_connections: Dict[str, Set[str]] = {}       # user_id -> set of connection_ids
        
        # Chat rooms and presence
        self.chat_rooms: Dict[str, Set[str]] = {}             # room_id -> set of user_ids
        self.user_presence: Dict[str, UserStatus] = {}        # user_id -> status
        
        # Market data subscriptions
        self.market_subscriptions: Dict[str, Set[str]] = {}   # symbol -> set of connection_ids
        
        # Message history and analytics
        self.message_history: Dict[str, List[Dict]] = {}      # session_id -> messages
        self.connection_stats: Dict[str, Dict] = {}           # connection_id -> stats
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._market_data_task: Optional[asyncio.Task] = None
        self._presence_update_task: Optional[asyncio.Task] = None
        
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        session_id: str,
        profile_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Handle new WebSocket connection
        Returns connection ID
        """
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"{user_id}_{session_id}_{uuid.uuid4().hex[:8]}"
        
        # Create connection object
        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            profile_id=profile_id,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Update session connections
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(connection_id)
        
        # Update user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Update user presence
        self.user_presence[user_id] = UserStatus.ONLINE
        
        # Initialize stats
        self.connection_stats[connection_id] = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connected_duration": 0
        }
        
        # Start background tasks if not running
        await self._start_background_tasks()
        
        # Notify other users in session about new connection
        await self._broadcast_presence_update(user_id, UserStatus.ONLINE, session_id)
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id}, session: {session_id})")
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user_id
        session_id = connection.session_id
        
        # Update connection stats
        if connection_id in self.connection_stats:
            self.connection_stats[connection_id]["connected_duration"] = (
                datetime.now() - connection.connected_at
            ).total_seconds()
        
        # Remove from connections
        del self.connections[connection_id]
        
        # Remove from session connections
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
                # Update user presence to offline if no connections
                self.user_presence[user_id] = UserStatus.OFFLINE
                await self._broadcast_presence_update(user_id, UserStatus.OFFLINE, session_id)
        
        # Remove from market subscriptions
        for symbol in list(self.market_subscriptions.keys()):
            if connection_id in self.market_subscriptions[symbol]:
                self.market_subscriptions[symbol].discard(connection_id)
                if not self.market_subscriptions[symbol]:
                    del self.market_subscriptions[symbol]
        
        logger.info(f"WebSocket disconnected: {connection_id} (user: {user_id})")
    
    async def handle_message(self, connection_id: str, message_data: str):
        """Handle incoming WebSocket message"""
        if connection_id not in self.connections:
            logger.warning(f"Message from unknown connection: {connection_id}")
            return
        
        connection = self.connections[connection_id]
        connection.update_activity()
        
        try:
            # Parse message
            raw_message = json.loads(message_data)
            message = WebSocketMessage(**raw_message)
            
            # Update stats
            self.connection_stats[connection_id]["messages_received"] += 1
            self.connection_stats[connection_id]["bytes_received"] += len(message_data)
            
            # Set message metadata
            message.user_id = connection.user_id
            message.session_id = connection.session_id
            message.timestamp = datetime.now()
            message.message_id = message.message_id or str(uuid.uuid4())
            
            # Route message based on type
            if message.type == MessageType.CHAT:
                await self._handle_chat_message(connection_id, message)
            elif message.type == MessageType.TYPING:
                await self._handle_typing_indicator(connection_id, message)
            elif message.type == MessageType.PRESENCE:
                await self._handle_presence_update(connection_id, message)
            elif message.type == MessageType.VOICE_MESSAGE:
                await self._handle_voice_message(connection_id, message)
            elif message.type == MessageType.FILE_SHARE:
                await self._handle_file_share(connection_id, message)
            elif message.type == MessageType.SCREEN_SHARE:
                await self._handle_screen_share(connection_id, message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error_message(connection_id, f"Error processing message: {str(e)}")
    
    async def _handle_chat_message(self, connection_id: str, message: WebSocketMessage):
        """Handle chat message"""
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        # Store message in history
        if session_id not in self.message_history:
            self.message_history[session_id] = []
        
        chat_message = {
            "id": message.message_id,
            "user_id": message.user_id,
            "content": message.data.get("content", ""),
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.data.get("metadata", {})
        }
        
        self.message_history[session_id].append(chat_message)
        
        # Broadcast to other connections in session
        await self._broadcast_to_session(session_id, {
            "type": MessageType.CHAT.value,
            "data": chat_message
        }, exclude_connection=connection_id)
        
        logger.debug(f"Chat message from {connection.user_id}: {chat_message['content'][:100]}")
    
    async def _handle_typing_indicator(self, connection_id: str, message: WebSocketMessage):
        """Handle typing indicator"""
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        # Broadcast typing indicator to other users in session
        await self._broadcast_to_session(session_id, {
            "type": MessageType.TYPING.value,
            "data": {
                "user_id": connection.user_id,
                "is_typing": message.data.get("is_typing", True)
            }
        }, exclude_connection=connection_id)
    
    async def _handle_presence_update(self, connection_id: str, message: WebSocketMessage):
        """Handle user presence update"""
        connection = self.connections[connection_id]
        user_id = connection.user_id
        session_id = connection.session_id
        
        new_status = UserStatus(message.data.get("status", UserStatus.ONLINE.value))
        
        # Update presence
        self.user_presence[user_id] = new_status
        connection.status = new_status
        
        # Broadcast presence update
        await self._broadcast_presence_update(user_id, new_status, session_id)
    
    async def _handle_voice_message(self, connection_id: str, message: WebSocketMessage):
        """Handle voice message"""
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        voice_data = {
            "id": message.message_id,
            "user_id": connection.user_id,
            "audio_url": message.data.get("audio_url"),
            "duration": message.data.get("duration", 0),
            "timestamp": message.timestamp.isoformat(),
            "transcript": message.data.get("transcript", "")
        }
        
        # Store in message history
        if session_id not in self.message_history:
            self.message_history[session_id] = []
        self.message_history[session_id].append({
            "id": message.message_id,
            "user_id": connection.user_id,
            "type": "voice",
            "content": voice_data,
            "timestamp": message.timestamp.isoformat()
        })
        
        # Broadcast to session
        await self._broadcast_to_session(session_id, {
            "type": MessageType.VOICE_MESSAGE.value,
            "data": voice_data
        }, exclude_connection=connection_id)
    
    async def _handle_file_share(self, connection_id: str, message: WebSocketMessage):
        """Handle file sharing"""
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        file_data = {
            "id": message.message_id,
            "user_id": connection.user_id,
            "file_name": message.data.get("file_name"),
            "file_size": message.data.get("file_size"),
            "file_url": message.data.get("file_url"),
            "file_type": message.data.get("file_type"),
            "timestamp": message.timestamp.isoformat()
        }
        
        # Store in message history
        if session_id not in self.message_history:
            self.message_history[session_id] = []
        self.message_history[session_id].append({
            "id": message.message_id,
            "user_id": connection.user_id,
            "type": "file",
            "content": file_data,
            "timestamp": message.timestamp.isoformat()
        })
        
        # Broadcast to session
        await self._broadcast_to_session(session_id, {
            "type": MessageType.FILE_SHARE.value,
            "data": file_data
        })
    
    async def _handle_screen_share(self, connection_id: str, message: WebSocketMessage):
        """Handle screen sharing"""
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        screen_data = {
            "user_id": connection.user_id,
            "screen_id": message.data.get("screen_id"),
            "sharing": message.data.get("sharing", False),
            "timestamp": message.timestamp.isoformat()
        }
        
        # Broadcast to session
        await self._broadcast_to_session(session_id, {
            "type": MessageType.SCREEN_SHARE.value,
            "data": screen_data
        })
    
    async def _broadcast_presence_update(self, user_id: str, status: UserStatus, session_id: str):
        """Broadcast presence update to session"""
        await self._broadcast_to_session(session_id, {
            "type": MessageType.PRESENCE.value,
            "data": {
                "user_id": user_id,
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _broadcast_to_session(self, session_id: str, message: Dict, exclude_connection: Optional[str] = None):
        """Broadcast message to all connections in a session"""
        if session_id not in self.session_connections:
            return
        
        message_str = json.dumps(message)
        message_bytes = len(message_str)
        
        # Send to all connections in session
        for conn_id in self.session_connections[session_id]:
            if conn_id == exclude_connection:
                continue
                
            if conn_id in self.connections:
                connection = self.connections[conn_id]
                try:
                    await connection.websocket.send_text(message_str)
                    
                    # Update stats
                    self.connection_stats[conn_id]["messages_sent"] += 1
                    self.connection_stats[conn_id]["bytes_sent"] += message_bytes
                    
                except Exception as e:
                    logger.error(f"Error sending message to {conn_id}: {e}")
                    # Connection might be dead, clean up later
    
    async def _broadcast_to_user(self, user_id: str, message: Dict):
        """Broadcast message to all connections of a user"""
        if user_id not in self.user_connections:
            return
        
        message_str = json.dumps(message)
        message_bytes = len(message_str)
        
        for conn_id in self.user_connections[user_id]:
            if conn_id in self.connections:
                connection = self.connections[conn_id]
                try:
                    await connection.websocket.send_text(message_str)
                    
                    # Update stats
                    self.connection_stats[conn_id]["messages_sent"] += 1
                    self.connection_stats[conn_id]["bytes_sent"] += message_bytes
                    
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
    
    async def _send_error_message(self, connection_id: str, error_message: str):
        """Send error message to specific connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        try:
            await connection.websocket.send_text(json.dumps({
                "type": MessageType.SYSTEM.value,
                "data": {
                    "level": "error",
                    "message": error_message,
                    "timestamp": datetime.now().isoformat()
                }
            }))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    async def broadcast_market_data(self, symbol: str, market_data: Dict):
        """Broadcast market data to subscribers"""
        if symbol not in self.market_subscriptions:
            return
        
        message = {
            "type": MessageType.MARKET_UPDATE.value,
            "data": {
                "symbol": symbol,
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        message_str = json.dumps(message)
        message_bytes = len(message_str)
        
        # Send to all subscribers
        for conn_id in list(self.market_subscriptions[symbol]):
            if conn_id in self.connections:
                connection = self.connections[conn_id]
                try:
                    await connection.websocket.send_text(message_str)
                    
                    # Update stats
                    self.connection_stats[conn_id]["messages_sent"] += 1
                    self.connection_stats[conn_id]["bytes_sent"] += message_bytes
                    
                except Exception as e:
                    logger.error(f"Error sending market data to {conn_id}: {e}")
                    # Remove dead connection
                    self.market_subscriptions[symbol].discard(conn_id)
    
    async def subscribe_to_market_data(self, connection_id: str, symbol: str):
        """Subscribe connection to market data for symbol"""
        if connection_id not in self.connections:
            return
        
        if symbol not in self.market_subscriptions:
            self.market_subscriptions[symbol] = set()
        
        self.market_subscriptions[symbol].add(connection_id)
        logger.info(f"Connection {connection_id} subscribed to {symbol} market data")
    
    async def unsubscribe_from_market_data(self, connection_id: str, symbol: str):
        """Unsubscribe connection from market data"""
        if symbol in self.market_subscriptions:
            self.market_subscriptions[symbol].discard(connection_id)
            if not self.market_subscriptions[symbol]:
                del self.market_subscriptions[symbol]
    
    def get_connection_stats(self) -> Dict:
        """Get comprehensive connection statistics"""
        return {
            "total_connections": len(self.connections),
            "active_sessions": len(self.session_connections),
            "online_users": len([u for u, s in self.user_presence.items() if s == UserStatus.ONLINE]),
            "market_subscriptions": {symbol: len(subs) for symbol, subs in self.market_subscriptions.items()},
            "connection_details": {
                conn_id: {
                    **connection.to_dict(),
                    **self.connection_stats.get(conn_id, {})
                }
                for conn_id, connection in self.connections.items()
            }
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session"""
        if session_id not in self.session_connections:
            return None
        
        connections = []
        for conn_id in self.session_connections[session_id]:
            if conn_id in self.connections:
                connection = self.connections[conn_id]
                connections.append(connection.to_dict())
        
        return {
            "session_id": session_id,
            "active_connections": len(connections),
            "connections": connections,
            "message_count": len(self.message_history.get(session_id, [])),
            "recent_messages": self.message_history.get(session_id, [])[-10:]  # Last 10 messages
        }
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_connections())
        
        if self._presence_update_task is None or self._presence_update_task.done():
            self._presence_update_task = asyncio.create_task(self._update_user_presence())
    
    async def _cleanup_connections(self):
        """Background task to clean up stale connections"""
        while True:
            try:
                current_time = datetime.now()
                stale_connections = []
                
                # Find stale connections (inactive for more than 5 minutes)
                for conn_id, connection in self.connections.items():
                    if (current_time - connection.last_activity).total_seconds() > 300:  # 5 minutes
                        stale_connections.append(conn_id)
                
                # Clean up stale connections
                for conn_id in stale_connections:
                    logger.info(f"Cleaning up stale connection: {conn_id}")
                    await self.disconnect(conn_id)
                
                # Sleep for 30 seconds before next cleanup
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(30)
    
    async def _update_user_presence(self):
        """Background task to update user presence based on activity"""
        while True:
            try:
                current_time = datetime.now()
                
                # Update user presence based on last activity
                for user_id, conn_ids in self.user_connections.items():
                    if not conn_ids:  # No active connections
                        self.user_presence[user_id] = UserStatus.OFFLINE
                        continue
                    
                    # Find most recent activity across all user connections
                    latest_activity = None
                    for conn_id in conn_ids:
                        if conn_id in self.connections:
                            connection = self.connections[conn_id]
                            if latest_activity is None or connection.last_activity > latest_activity:
                                latest_activity = connection.last_activity
                    
                    if latest_activity:
                        inactive_minutes = (current_time - latest_activity).total_seconds() / 60
                        
                        # Update status based on inactivity
                        if inactive_minutes > 30:  # 30 minutes
                            new_status = UserStatus.AWAY
                        elif inactive_minutes > 60:  # 1 hour
                            new_status = UserStatus.OFFLINE
                        else:
                            new_status = UserStatus.ONLINE
                        
                        # Update if status changed
                        if self.user_presence.get(user_id) != new_status:
                            self.user_presence[user_id] = new_status
                            
                            # Notify all user sessions about presence change
                            user_sessions = set()
                            for conn_id in conn_ids:
                                if conn_id in self.connections:
                                    user_sessions.add(self.connections[conn_id].session_id)
                            
                            for session_id in user_sessions:
                                await self._broadcast_presence_update(user_id, new_status, session_id)
                
                # Sleep for 1 minute before next update
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in presence update task: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Shutdown the WebSocket handler and clean up resources"""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._market_data_task:
            self._market_data_task.cancel()
        if self._presence_update_task:
            self._presence_update_task.cancel()
        
        # Close all connections
        for conn_id in list(self.connections.keys()):
            await self.disconnect(conn_id)
        
        logger.info("WebSocket handler shutdown complete")

# Global WebSocket handler instance
websocket_handler = WebSocketChatHandler()