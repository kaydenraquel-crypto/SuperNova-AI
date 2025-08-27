"""
SuperNova WebSocket Optimizer
High-performance WebSocket scaling and optimization for real-time communication
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import weakref
import uuid
from enum import Enum
import gzip
import zlib

# WebSocket and networking
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

# Message queuing and broadcasting
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .performance_monitor import performance_collector, performance_timer
from .cache_manager import cache_manager
from .async_processor import async_processor, TaskPriority

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe" 
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    ERROR = "error"
    SYSTEM = "system"
    BROADCAST = "broadcast"
    PRIVATE = "private"
    TYPING = "typing"
    PRESENCE = "presence"
    ACK = "ack"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Any = None
    channel: Optional[str] = None
    user_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    requires_ack: bool = False
    compression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type.value,
            'data': self.data,
            'channel': self.channel,
            'user_id': self.user_id,
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'requires_ack': self.requires_ack
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary"""
        return cls(
            type=MessageType(data['type']),
            data=data.get('data'),
            channel=data.get('channel'),
            user_id=data.get('user_id'),
            message_id=data.get('message_id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            requires_ack=data.get('requires_ack', False)
        )

@dataclass
class ConnectionMetrics:
    """WebSocket connection metrics"""
    connection_id: str
    user_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    subscriptions: Set[str] = field(default_factory=set)
    ping_latency_ms: Optional[float] = None
    is_authenticated: bool = False

class CompressionManager:
    """Handle message compression for large payloads"""
    
    @staticmethod
    def should_compress(message: str, threshold: int = 1024) -> bool:
        """Determine if message should be compressed"""
        return len(message.encode('utf-8')) > threshold
    
    @staticmethod
    def compress_message(message: str, method: str = 'gzip') -> bytes:
        """Compress message using specified method"""
        if method == 'gzip':
            return gzip.compress(message.encode('utf-8'))
        elif method == 'zlib':
            return zlib.compress(message.encode('utf-8'))
        else:
            return message.encode('utf-8')
    
    @staticmethod
    def decompress_message(data: bytes, method: str = 'gzip') -> str:
        """Decompress message"""
        if method == 'gzip':
            return gzip.decompress(data).decode('utf-8')
        elif method == 'zlib':
            return zlib.decompress(data).decode('utf-8')
        else:
            return data.decode('utf-8')

class MessageQueue:
    """High-performance message queue with batching"""
    
    def __init__(self, max_size: int = 10000, batch_size: int = 100):
        self.max_size = max_size
        self.batch_size = batch_size
        self.queue: asyncio.Queue = asyncio.Queue(max_size)
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.batch_processor_task = None
        
    async def start(self):
        """Start message queue processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("Message queue started")
    
    async def stop(self):
        """Stop message queue processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message queue stopped")
    
    async def put(self, message: WebSocketMessage):
        """Add message to queue"""
        try:
            await self.queue.put(message)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping message")
    
    async def get_batch(self) -> List[WebSocketMessage]:
        """Get batch of messages"""
        return await self.batch_queue.get()
    
    async def _batch_processor(self):
        """Process messages in batches for efficiency"""
        batch = []
        last_batch_time = time.time()
        batch_timeout = 0.05  # 50ms timeout
        
        while self.is_running:
            try:
                # Collect messages for batch
                while len(batch) < self.batch_size:
                    try:
                        message = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=batch_timeout
                        )
                        batch.append(message)
                    except asyncio.TimeoutError:
                        break
                    
                    # Check timeout
                    if time.time() - last_batch_time > batch_timeout:
                        break
                
                # Send batch if not empty
                if batch:
                    await self.batch_queue.put(batch.copy())
                    batch.clear()
                    last_batch_time = time.time()
                
                await asyncio.sleep(0.001)  # Brief pause
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                batch.clear()
                await asyncio.sleep(0.1)

class ConnectionPool:
    """Manage WebSocket connections with optimization"""
    
    def __init__(self, max_connections: int = 10000):
        self.max_connections = max_connections
        self.connections: Dict[str, WebSocket] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # channel -> connection_ids
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> connection_ids
        
        # Connection management
        self.connection_lock = asyncio.Lock()
        self.cleanup_task = None
        self.is_running = False
        
        # Performance tracking
        self.total_connections = 0
        self.peak_connections = 0
        self.connection_errors = 0
        
    async def start(self):
        """Start connection pool"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        logger.info("Connection pool started")
    
    async def stop(self):
        """Stop connection pool"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        await self.disconnect_all()
        
        logger.info("Connection pool stopped")
    
    async def add_connection(self, connection_id: str, websocket: WebSocket, user_id: Optional[str] = None) -> bool:
        """Add connection to pool"""
        async with self.connection_lock:
            if len(self.connections) >= self.max_connections:
                logger.warning("Connection pool full, rejecting connection")
                return False
            
            self.connections[connection_id] = websocket
            self.connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                user_id=user_id
            )
            
            if user_id:
                self.user_connections[user_id].add(connection_id)
            
            self.total_connections += 1
            self.peak_connections = max(self.peak_connections, len(self.connections))
            
            logger.debug(f"Connection added: {connection_id}, Total: {len(self.connections)}")
            return True
    
    async def remove_connection(self, connection_id: str):
        """Remove connection from pool"""
        async with self.connection_lock:
            if connection_id not in self.connections:
                return
            
            # Get metrics before removal
            metrics = self.connection_metrics.get(connection_id)
            
            # Remove from subscriptions
            for channel_connections in self.subscriptions.values():
                channel_connections.discard(connection_id)
            
            # Remove from user connections
            if metrics and metrics.user_id:
                self.user_connections[metrics.user_id].discard(connection_id)
                if not self.user_connections[metrics.user_id]:
                    del self.user_connections[metrics.user_id]
            
            # Remove connection
            del self.connections[connection_id]
            if connection_id in self.connection_metrics:
                del self.connection_metrics[connection_id]
            
            logger.debug(f"Connection removed: {connection_id}, Remaining: {len(self.connections)}")
    
    async def get_connection(self, connection_id: str) -> Optional[WebSocket]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    async def subscribe(self, connection_id: str, channel: str):
        """Subscribe connection to channel"""
        if connection_id in self.connections:
            self.subscriptions[channel].add(connection_id)
            
            if connection_id in self.connection_metrics:
                self.connection_metrics[connection_id].subscriptions.add(channel)
            
            logger.debug(f"Connection {connection_id} subscribed to {channel}")
    
    async def unsubscribe(self, connection_id: str, channel: str):
        """Unsubscribe connection from channel"""
        self.subscriptions[channel].discard(connection_id)
        
        if connection_id in self.connection_metrics:
            self.connection_metrics[connection_id].subscriptions.discard(channel)
        
        # Clean up empty channels
        if not self.subscriptions[channel]:
            del self.subscriptions[channel]
        
        logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
    
    async def get_channel_connections(self, channel: str) -> Set[str]:
        """Get connections subscribed to channel"""
        return self.subscriptions.get(channel, set()).copy()
    
    async def get_user_connections(self, user_id: str) -> Set[str]:
        """Get connections for specific user"""
        return self.user_connections.get(user_id, set()).copy()
    
    async def update_metrics(self, connection_id: str, sent_bytes: int = 0, received_bytes: int = 0):
        """Update connection metrics"""
        if connection_id in self.connection_metrics:
            metrics = self.connection_metrics[connection_id]
            metrics.last_activity = datetime.now()
            metrics.bytes_sent += sent_bytes
            metrics.bytes_received += received_bytes
            
            if sent_bytes > 0:
                metrics.messages_sent += 1
            if received_bytes > 0:
                metrics.messages_received += 1
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections periodically"""
        while self.is_running:
            try:
                current_time = datetime.now()
                stale_threshold = timedelta(minutes=5)  # 5 minutes without activity
                
                stale_connections = []
                
                async with self.connection_lock:
                    for connection_id, metrics in self.connection_metrics.items():
                        if current_time - metrics.last_activity > stale_threshold:
                            websocket = self.connections.get(connection_id)
                            
                            if websocket and websocket.client_state == WebSocketState.DISCONNECTED:
                                stale_connections.append(connection_id)
                
                # Remove stale connections
                for connection_id in stale_connections:
                    await self.remove_connection(connection_id)
                    logger.info(f"Removed stale connection: {connection_id}")
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
                await asyncio.sleep(30)
    
    async def disconnect_all(self):
        """Disconnect all connections"""
        connections = list(self.connections.items())
        
        for connection_id, websocket in connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1000, reason="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing connection {connection_id}: {e}")
            
            await self.remove_connection(connection_id)
        
        logger.info("All connections disconnected")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_connections': len(self.connections),
            'total_connections': self.total_connections,
            'peak_connections': self.peak_connections,
            'connection_errors': self.connection_errors,
            'channels': len(self.subscriptions),
            'subscriptions': sum(len(conns) for conns in self.subscriptions.values()),
            'users_connected': len(self.user_connections),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of connection pool"""
        # Rough estimation based on data structures
        base_memory = 0
        base_memory += len(self.connections) * 200  # WebSocket objects
        base_memory += len(self.connection_metrics) * 500  # Metrics objects
        base_memory += sum(len(conns) * 50 for conns in self.subscriptions.values())  # Subscriptions
        base_memory += sum(len(conns) * 50 for conns in self.user_connections.values())  # User connections
        
        return base_memory / 1024 / 1024  # Convert to MB

class WebSocketBroadcaster:
    """High-performance message broadcasting"""
    
    def __init__(self, connection_pool: ConnectionPool, message_queue: MessageQueue):
        self.connection_pool = connection_pool
        self.message_queue = message_queue
        self.compression_manager = CompressionManager()
        
        # Broadcasting statistics
        self.messages_broadcast = 0
        self.broadcast_errors = 0
        self.total_bytes_sent = 0
        
        # Redis for distributed broadcasting
        self.redis_client = None
        self.redis_enabled = False
    
    async def initialize_redis(self, redis_url: str):
        """Initialize Redis for distributed broadcasting"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, distributed broadcasting disabled")
            return
        
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            self.redis_enabled = True
            logger.info("Redis broadcasting initialized")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
    
    @performance_timer(category="websocket_broadcast")
    async def broadcast_to_channel(self, channel: str, message: WebSocketMessage) -> int:
        """Broadcast message to all connections in channel"""
        
        connections = await self.connection_pool.get_channel_connections(channel)
        if not connections:
            return 0
        
        # Prepare message
        message_json = json.dumps(message.to_dict())
        
        # Compress if needed
        compressed = False
        if self.compression_manager.should_compress(message_json):
            message_data = self.compression_manager.compress_message(message_json)
            compressed = True
        else:
            message_data = message_json
        
        # Broadcast to all connections concurrently
        broadcast_tasks = []
        
        for connection_id in connections:
            websocket = await self.connection_pool.get_connection(connection_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                task = asyncio.create_task(
                    self._send_message_safe(websocket, connection_id, message_data, compressed)
                )
                broadcast_tasks.append(task)
        
        # Wait for all broadcasts to complete
        results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        
        # Count successful broadcasts
        successful_broadcasts = sum(1 for r in results if r is True)
        self.messages_broadcast += successful_broadcasts
        self.total_bytes_sent += len(message_data) * successful_broadcasts
        
        # Publish to Redis for distributed broadcasting
        if self.redis_enabled and successful_broadcasts > 0:
            try:
                await self._publish_to_redis(channel, message)
            except Exception as e:
                logger.error(f"Redis publish error: {e}")
        
        return successful_broadcasts
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Broadcast message to all connections of a specific user"""
        
        connections = await self.connection_pool.get_user_connections(user_id)
        if not connections:
            return 0
        
        message_json = json.dumps(message.to_dict())
        
        # Send to all user connections
        broadcast_tasks = []
        
        for connection_id in connections:
            websocket = await self.connection_pool.get_connection(connection_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                task = asyncio.create_task(
                    self._send_message_safe(websocket, connection_id, message_json, False)
                )
                broadcast_tasks.append(task)
        
        results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        successful_broadcasts = sum(1 for r in results if r is True)
        
        self.messages_broadcast += successful_broadcasts
        return successful_broadcasts
    
    async def broadcast_to_all(self, message: WebSocketMessage, exclude_connections: Optional[Set[str]] = None) -> int:
        """Broadcast message to all active connections"""
        
        all_connections = set(self.connection_pool.connections.keys())
        
        if exclude_connections:
            all_connections -= exclude_connections
        
        if not all_connections:
            return 0
        
        message_json = json.dumps(message.to_dict())
        broadcast_tasks = []
        
        for connection_id in all_connections:
            websocket = await self.connection_pool.get_connection(connection_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                task = asyncio.create_task(
                    self._send_message_safe(websocket, connection_id, message_json, False)
                )
                broadcast_tasks.append(task)
        
        results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        successful_broadcasts = sum(1 for r in results if r is True)
        
        self.messages_broadcast += successful_broadcasts
        return successful_broadcasts
    
    async def _send_message_safe(self, websocket: WebSocket, connection_id: str, message_data: Union[str, bytes], compressed: bool) -> bool:
        """Safely send message to WebSocket connection"""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            if compressed:
                # Send binary data for compressed messages
                await websocket.send_bytes(message_data)
            else:
                # Send text data for uncompressed messages
                await websocket.send_text(message_data)
            
            # Update metrics
            data_size = len(message_data) if isinstance(message_data, (str, bytes)) else 0
            await self.connection_pool.update_metrics(connection_id, sent_bytes=data_size)
            
            return True
            
        except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedError):
            # Connection closed, remove from pool
            await self.connection_pool.remove_connection(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            self.broadcast_errors += 1
            return False
    
    async def _publish_to_redis(self, channel: str, message: WebSocketMessage):
        """Publish message to Redis for distributed broadcasting"""
        if not self.redis_client:
            return
        
        redis_message = {
            'source': 'supernova',
            'channel': channel,
            'message': message.to_dict()
        }
        
        await self.redis_client.publish(f"ws:{channel}", json.dumps(redis_message))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broadcasting statistics"""
        return {
            'messages_broadcast': self.messages_broadcast,
            'broadcast_errors': self.broadcast_errors,
            'total_bytes_sent': self.total_bytes_sent,
            'redis_enabled': self.redis_enabled,
            'error_rate': self.broadcast_errors / max(1, self.messages_broadcast)
        }

class WebSocketOptimizer:
    """Main WebSocket optimization coordinator"""
    
    def __init__(self, max_connections: int = 10000):
        self.connection_pool = ConnectionPool(max_connections)
        self.message_queue = MessageQueue()
        self.broadcaster = WebSocketBroadcaster(self.connection_pool, self.message_queue)
        
        # Components
        self.is_running = False
        self.message_processor_task = None
        
        # Heartbeat management
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task = None
    
    async def start(self):
        """Start WebSocket optimization system"""
        if self.is_running:
            return
        
        logger.info("Starting WebSocket optimizer")
        
        self.is_running = True
        
        # Start components
        await self.connection_pool.start()
        await self.message_queue.start()
        
        # Start message processor
        self.message_processor_task = asyncio.create_task(self._process_messages())
        
        # Start heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_manager())
        
        logger.info("WebSocket optimizer started")
    
    async def stop(self):
        """Stop WebSocket optimization system"""
        if not self.is_running:
            return
        
        logger.info("Stopping WebSocket optimizer")
        
        self.is_running = False
        
        # Cancel tasks
        if self.message_processor_task:
            self.message_processor_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Stop components
        await self.message_queue.stop()
        await self.connection_pool.stop()
        
        logger.info("WebSocket optimizer stopped")
    
    async def handle_connection(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Handle new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        success = await self.connection_pool.add_connection(connection_id, websocket, user_id)
        
        if not success:
            await websocket.close(code=1013, reason="Server overloaded")
            raise RuntimeError("Connection pool full")
        
        return connection_id
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming WebSocket message"""
        try:
            # Parse message
            message_data = json.loads(raw_message)
            message = WebSocketMessage.from_dict(message_data)
            
            # Update connection metrics
            await self.connection_pool.update_metrics(
                connection_id, 
                received_bytes=len(raw_message)
            )
            
            # Route message based on type
            if message.type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(connection_id, message)
            elif message.type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(connection_id, message)
            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(connection_id, message)
            elif message.type == MessageType.BROADCAST:
                await self._handle_broadcast(connection_id, message)
            else:
                # Queue message for processing
                await self.message_queue.put(message)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {connection_id}")
            await self._send_error(connection_id, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self._send_error(connection_id, "Message processing error")
    
    async def disconnect(self, connection_id: str):
        """Handle connection disconnect"""
        await self.connection_pool.remove_connection(connection_id)
        logger.debug(f"Connection disconnected: {connection_id}")
    
    async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle channel subscription"""
        if message.channel:
            await self.connection_pool.subscribe(connection_id, message.channel)
            
            # Send confirmation
            response = WebSocketMessage(
                type=MessageType.SYSTEM,
                data={'status': 'subscribed', 'channel': message.channel},
                message_id=message.message_id
            )
            await self._send_to_connection(connection_id, response)
    
    async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle channel unsubscription"""
        if message.channel:
            await self.connection_pool.unsubscribe(connection_id, message.channel)
            
            # Send confirmation
            response = WebSocketMessage(
                type=MessageType.SYSTEM,
                data={'status': 'unsubscribed', 'channel': message.channel},
                message_id=message.message_id
            )
            await self._send_to_connection(connection_id, response)
    
    async def _handle_heartbeat(self, connection_id: str, message: WebSocketMessage):
        """Handle heartbeat message"""
        response = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={'pong': True, 'server_time': datetime.now().isoformat()},
            message_id=message.message_id
        )
        await self._send_to_connection(connection_id, response)
    
    async def _handle_broadcast(self, connection_id: str, message: WebSocketMessage):
        """Handle broadcast message"""
        if message.channel:
            # Broadcast to channel
            count = await self.broadcaster.broadcast_to_channel(message.channel, message)
            logger.debug(f"Broadcast to {count} connections in channel {message.channel}")
        elif message.user_id:
            # Send to specific user
            count = await self.broadcaster.broadcast_to_user(message.user_id, message)
            logger.debug(f"Sent to {count} connections for user {message.user_id}")
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection"""
        websocket = await self.connection_pool.get_connection(connection_id)
        
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            message_json = json.dumps(message.to_dict())
            
            try:
                await websocket.send_text(message_json)
                await self.connection_pool.update_metrics(connection_id, sent_bytes=len(message_json))
            except Exception as e:
                logger.error(f"Error sending to connection {connection_id}: {e}")
                await self.connection_pool.remove_connection(connection_id)
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            data={'error': error_message}
        )
        await self._send_to_connection(connection_id, error_msg)
    
    async def _process_messages(self):
        """Process queued messages"""
        while self.is_running:
            try:
                batch = await self.message_queue.get_batch()
                
                # Process batch of messages
                for message in batch:
                    if message.channel:
                        await self.broadcaster.broadcast_to_channel(message.channel, message)
                    elif message.user_id:
                        await self.broadcaster.broadcast_to_user(message.user_id, message)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _heartbeat_manager(self):
        """Manage heartbeat for all connections"""
        while self.is_running:
            try:
                # Send heartbeat to all connections
                heartbeat_msg = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={'ping': True, 'server_time': datetime.now().isoformat()}
                )
                
                await self.broadcaster.broadcast_to_all(heartbeat_msg)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket statistics"""
        return {
            'connection_pool': self.connection_pool.get_stats(),
            'broadcaster': self.broadcaster.get_stats(),
            'message_queue': {
                'queue_size': self.message_queue.queue.qsize(),
                'batch_queue_size': self.message_queue.batch_queue.qsize(),
                'is_running': self.message_queue.is_running
            },
            'system': {
                'is_running': self.is_running,
                'heartbeat_interval': self.heartbeat_interval
            }
        }

# Global optimizer instance
websocket_optimizer = WebSocketOptimizer(max_connections=10000)

# Convenience functions
async def initialize_websocket_optimizer():
    """Initialize WebSocket optimizer"""
    await websocket_optimizer.start()
    logger.info("WebSocket optimizer initialized")

async def shutdown_websocket_optimizer():
    """Shutdown WebSocket optimizer"""
    await websocket_optimizer.stop()
    logger.info("WebSocket optimizer shutdown")

async def broadcast_to_channel(channel: str, data: Any, message_type: MessageType = MessageType.DATA) -> int:
    """Broadcast message to channel"""
    message = WebSocketMessage(type=message_type, data=data, channel=channel)
    return await websocket_optimizer.broadcaster.broadcast_to_channel(channel, message)

async def send_to_user(user_id: str, data: Any, message_type: MessageType = MessageType.PRIVATE) -> int:
    """Send message to specific user"""
    message = WebSocketMessage(type=message_type, data=data, user_id=user_id)
    return await websocket_optimizer.broadcaster.broadcast_to_user(user_id, message)

# Export key components
__all__ = [
    'WebSocketOptimizer', 'ConnectionPool', 'WebSocketBroadcaster', 'MessageQueue',
    'WebSocketMessage', 'MessageType', 'ConnectionMetrics',
    'initialize_websocket_optimizer', 'shutdown_websocket_optimizer',
    'broadcast_to_channel', 'send_to_user', 'websocket_optimizer'
]