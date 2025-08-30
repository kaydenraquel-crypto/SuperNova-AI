#!/usr/bin/env python3
"""
WebSocket Integration Test
Tests Socket.IO connectivity and chat functionality
"""

import asyncio
import json
from datetime import datetime
import socketio

async def test_websocket_connection():
    """Test Socket.IO connection and chat functionality"""
    print("=== Socket.IO Integration Test ===")
    
    # Create Socket.IO client
    sio = socketio.AsyncClient()
    
    # Test connection events
    @sio.event
    async def connect():
        print(f"[OK] Connected to Socket.IO server at {datetime.now().isoformat()}")
        
        # Test authentication
        await sio.emit('subscribe', {'channel': 'notifications'})
        print("[INFO] Subscribed to notifications channel")
        
        await sio.emit('subscribe', {'channel': 'market_data'})
        print("[INFO] Subscribed to market_data channel")
        
        # Test chat message
        test_message = {
            'sessionId': 'test_session_123',
            'message': 'Hello, what are the current market trends?',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[SEND] Sending test chat message: {test_message['message']}")
        await sio.emit('chat_message', test_message)
    
    @sio.event
    async def connected(data):
        print(f"[OK] Connection confirmed: {data}")
    
    @sio.event
    async def subscribed(data):
        print(f"[INFO] Subscribed to channel: {data['channel']}")
    
    @sio.event
    async def chat_message(data):
        print(f"[CHAT] Received chat response: {data}")
    
    @sio.event
    async def market_data(data):
        print(f"[MARKET] Received market data: {data}")
    
    @sio.event
    async def notification(data):
        print(f"[NOTIF] Received notification: {data}")
    
    @sio.event
    async def error(data):
        print(f"[ERROR] Error: {data}")
    
    @sio.event
    async def disconnect():
        print("[INFO] Disconnected from Socket.IO server")
    
    try:
        # Connect to server (without authentication for now to test basic connectivity)
        print("[INFO] Connecting to Socket.IO server at http://localhost:8000...")
        await sio.connect('http://localhost:8000')
        
        # Keep connection alive for testing
        await asyncio.sleep(10)
        
        print("[OK] Test completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
    finally:
        await sio.disconnect()

async def test_authenticated_connection():
    """Test Socket.IO connection with authentication"""
    print("\n=== Authenticated Socket.IO Test ===")
    
    # Create Socket.IO client with auth
    sio = socketio.AsyncClient()
    
    @sio.event
    async def connect():
        print("[OK] Connected - sending authentication...")
        # This would normally require a real token
        # For testing, we'll see what happens without one
        
    @sio.event
    async def error(data):
        print(f"[ERROR] Auth Error: {data}")
    
    try:
        # Attempt connection with fake auth token
        await sio.connect(
            'http://localhost:8000',
            auth={'token': 'fake_token_for_testing'}
        )
        
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"[ERROR] Authenticated connection failed (expected): {e}")
    finally:
        await sio.disconnect()

async def test_socket_io_endpoint():
    """Test if Socket.IO endpoint is accessible"""
    print("\n=== Socket.IO Endpoint Test ===")
    
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test Socket.IO endpoint
            async with session.get('http://localhost:8000/socket.io/?transport=polling') as response:
                if response.status == 200:
                    print("[OK] Socket.IO endpoint is accessible")
                    text = await response.text()
                    print(f"[INFO] Response preview: {text[:200]}...")
                else:
                    print(f"[ERROR] Socket.IO endpoint returned status: {response.status}")
                    
    except Exception as e:
        print(f"[ERROR] Socket.IO endpoint test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_socket_io_endpoint())
    asyncio.run(test_websocket_connection())
    asyncio.run(test_authenticated_connection())