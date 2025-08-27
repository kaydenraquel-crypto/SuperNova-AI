"""
SuperNova AI WebSocket API Endpoints
Real-time communication endpoints for chat and collaboration
"""

import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.routing import APIRouter
from datetime import datetime

# WebSocket handlers
from .websocket_handler import websocket_handler
from .collaboration_websocket import collaboration_websocket_handler

# Auth
from .auth import auth_manager, AuthenticationError

logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Standard chat WebSocket endpoint
    """
    await websocket.accept()
    connection_id = None
    
    try:
        # Wait for authentication message
        auth_data = await websocket.receive_text()
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "auth":
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Authentication required"
            }))
            await websocket.close()
            return
        
        # Verify token
        token = auth_message.get("token")
        if not token:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Token required"
            }))
            await websocket.close()
            return
        
        try:
            payload = auth_manager.verify_token(token, "access")
            user_id = payload["sub"]
            session_id = payload.get("session_id")
        except AuthenticationError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid token"
            }))
            await websocket.close()
            return
        
        # Connect to chat handler
        connection_id = await websocket_handler.connect(
            websocket, user_id, session_id
        )
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }))
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_handler.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message processing failed"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            await websocket_handler.disconnect(connection_id)


@router.websocket("/ws/collaboration")
async def websocket_collaboration_endpoint(websocket: WebSocket):
    """
    Collaboration WebSocket endpoint with team support
    """
    await websocket.accept()
    connection_id = None
    
    try:
        # Wait for authentication message
        auth_data = await websocket.receive_text()
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "auth":
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Authentication required"
            }))
            await websocket.close()
            return
        
        # Verify token
        token = auth_message.get("token")
        team_ids = auth_message.get("team_ids", [])
        
        if not token:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Token required"
            }))
            await websocket.close()
            return
        
        try:
            payload = auth_manager.verify_token(token, "access")
            user_id = int(payload["sub"])
        except AuthenticationError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid token"
            }))
            await websocket.close()
            return
        
        # Connect to collaboration handler
        connection_id = await collaboration_websocket_handler.connect_user_collaboration(
            websocket, user_id, team_ids
        )
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "user_id": user_id,
                "team_ids": team_ids,
                "timestamp": datetime.now().isoformat()
            }
        }))
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await collaboration_websocket_handler.handle_collaboration_message(
                    connection_id, message
                )
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling collaboration message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message processing failed"
                }))
                
    except WebSocketDisconnect:
        logger.info("Collaboration WebSocket disconnected")
    except Exception as e:
        logger.error(f"Collaboration WebSocket error: {e}")
    finally:
        if connection_id:
            await collaboration_websocket_handler.disconnect_user_collaboration(connection_id)


@router.websocket("/ws/portfolio/{portfolio_id}")
async def websocket_portfolio_collaboration(websocket: WebSocket, portfolio_id: int):
    """
    Portfolio-specific collaboration WebSocket
    """
    await websocket.accept()
    connection_id = None
    
    try:
        # Wait for authentication message
        auth_data = await websocket.receive_text()
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "auth":
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Authentication required"
            }))
            await websocket.close()
            return
        
        # Verify token and portfolio access
        token = auth_message.get("token")
        if not token:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Token required"
            }))
            await websocket.close()
            return
        
        try:
            payload = auth_manager.verify_token(token, "access")
            user_id = int(payload["sub"])
        except AuthenticationError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid token"
            }))
            await websocket.close()
            return
        
        # TODO: Verify portfolio access permissions
        # This would check if user has access to the portfolio via sharing
        
        # Connect to collaboration handler
        connection_id = await collaboration_websocket_handler.connect_user_collaboration(
            websocket, user_id, []
        )
        
        # Subscribe to portfolio collaboration
        await collaboration_websocket_handler.handle_collaboration_message(
            connection_id,
            {
                "type": "portfolio_collaboration",
                "data": {
                    "portfolio_id": portfolio_id,
                    "action": "join"
                }
            }
        )
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "timestamp": datetime.now().isoformat()
            }
        }))
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Add portfolio context to all messages
                if "data" not in message:
                    message["data"] = {}
                message["data"]["portfolio_id"] = portfolio_id
                
                await collaboration_websocket_handler.handle_collaboration_message(
                    connection_id, message
                )
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling portfolio collaboration message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Message processing failed"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Portfolio {portfolio_id} collaboration WebSocket disconnected")
    except Exception as e:
        logger.error(f"Portfolio {portfolio_id} collaboration WebSocket error: {e}")
    finally:
        if connection_id:
            await collaboration_websocket_handler.disconnect_user_collaboration(connection_id)


@router.get("/ws/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics
    """
    stats = {
        "standard_connections": len(websocket_handler.connections),
        "collaboration_connections": len(collaboration_websocket_handler.connections),
        "active_teams": len(collaboration_websocket_handler.team_connections),
        "active_channels": len(collaboration_websocket_handler.channel_connections),
        "portfolio_watchers": len(collaboration_websocket_handler.portfolio_watchers),
        "timestamp": datetime.now().isoformat()
    }
    
    return stats


@router.post("/ws/broadcast/team/{team_id}")
async def broadcast_to_team(
    team_id: int,
    message: Dict[str, Any],
    current_user: dict = Depends(lambda: {"sub": "1"})  # Placeholder for auth
):
    """
    Broadcast message to all team members
    """
    try:
        await collaboration_websocket_handler._broadcast_to_team(team_id, {
            "type": "team_broadcast",
            "data": {
                **message,
                "timestamp": datetime.now().isoformat(),
                "sender_id": current_user["sub"]
            }
        })
        
        return {
            "success": True,
            "message": f"Message broadcast to team {team_id}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error broadcasting to team {team_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast message"
        )


@router.post("/ws/broadcast/portfolio/{portfolio_id}")
async def broadcast_to_portfolio_watchers(
    portfolio_id: int,
    message: Dict[str, Any],
    current_user: dict = Depends(lambda: {"sub": "1"})  # Placeholder for auth
):
    """
    Broadcast message to portfolio collaboration watchers
    """
    try:
        await collaboration_websocket_handler._broadcast_to_portfolio_watchers(
            portfolio_id,
            {
                "type": "portfolio_broadcast",
                "data": {
                    **message,
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.now().isoformat(),
                    "sender_id": current_user["sub"]
                }
            }
        )
        
        return {
            "success": True,
            "message": f"Message broadcast to portfolio {portfolio_id} watchers",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error broadcasting to portfolio {portfolio_id} watchers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast message"
        )