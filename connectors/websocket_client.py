"""NovaSignal WebSocket Client Helper Module

Specialized WebSocket client for handling different types of market data streams
and providing additional utilities for connection management.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, WebSocketException

from ..supernova.config import settings

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams available from NovaSignal"""
    OHLCV = "ohlcv"
    TRADES = "trades"
    ORDER_BOOK = "orderbook"
    NEWS = "news"
    ALERTS = "alerts"
    SYSTEM = "system"


@dataclass
class StreamSubscription:
    """Stream subscription configuration"""
    stream_type: StreamType
    symbols: Set[str]
    parameters: Dict[str, Any]
    callback: Callable[[Dict[str, Any]], None]
    is_active: bool = True


class NovaSignalWebSocketClient:
    """Specialized WebSocket client for NovaSignal real-time data streams"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or settings.NOVASIGNAL_WS_URL
        self.api_key = api_key or settings.NOVASIGNAL_API_KEY
        self.connection: Optional[websockets.WebSocketServerProtocol] = None
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        self._message_handlers = {
            "subscription_confirmed": self._handle_subscription_confirmed,
            "subscription_error": self._handle_subscription_error,
            "heartbeat": self._handle_heartbeat,
            "error": self._handle_error,
            "data": self._handle_data
        }
    
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            # Build connection URL with authentication
            url = f"{self.base_url}/stream"
            if self.api_key:
                url += f"?api_key={self.api_key}"
            
            logger.info(f"Connecting to NovaSignal WebSocket: {url}")
            
            self.connection = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("WebSocket connection established successfully")
            
            # Send initial authentication if needed
            await self._authenticate()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def _authenticate(self):
        """Send authentication message"""
        if self.api_key:
            auth_message = {
                "type": "authenticate",
                "api_key": self.api_key,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self._send_message(auth_message)
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.connection:
            logger.info("Closing WebSocket connection")
            await self.connection.close()
            self.connection = None
        self.is_connected = False
    
    async def subscribe_ohlcv(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1m", "5m", "1h"],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """Subscribe to OHLCV data stream"""
        subscription_id = f"ohlcv_{len(self.subscriptions)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.OHLCV,
            symbols=set(symbols),
            parameters={"timeframes": timeframes},
            callback=callback or self._default_ohlcv_handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if self.is_connected:
            await self._send_subscription(subscription_id, subscription)
        
        logger.info(f"Subscribed to OHLCV data for {len(symbols)} symbols")
        return subscription_id
    
    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """Subscribe to trade data stream"""
        subscription_id = f"trades_{len(self.subscriptions)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.TRADES,
            symbols=set(symbols),
            parameters={},
            callback=callback or self._default_trades_handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if self.is_connected:
            await self._send_subscription(subscription_id, subscription)
        
        logger.info(f"Subscribed to trade data for {len(symbols)} symbols")
        return subscription_id
    
    async def subscribe_orderbook(
        self,
        symbols: List[str],
        depth: int = 20,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """Subscribe to order book data stream"""
        subscription_id = f"orderbook_{len(self.subscriptions)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.ORDER_BOOK,
            symbols=set(symbols),
            parameters={"depth": depth},
            callback=callback or self._default_orderbook_handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if self.is_connected:
            await self._send_subscription(subscription_id, subscription)
        
        logger.info(f"Subscribed to order book data for {len(symbols)} symbols")
        return subscription_id
    
    async def subscribe_news(
        self,
        symbols: List[str] = [],
        categories: List[str] = [],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """Subscribe to news stream"""
        subscription_id = f"news_{len(self.subscriptions)}"
        
        subscription = StreamSubscription(
            stream_type=StreamType.NEWS,
            symbols=set(symbols),
            parameters={"categories": categories},
            callback=callback or self._default_news_handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if self.is_connected:
            await self._send_subscription(subscription_id, subscription)
        
        logger.info("Subscribed to news stream")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a data stream"""
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription {subscription_id} not found")
            return False
        
        subscription = self.subscriptions[subscription_id]
        subscription.is_active = False
        
        if self.is_connected:
            unsubscribe_message = {
                "type": "unsubscribe",
                "subscription_id": subscription_id,
                "stream_type": subscription.stream_type.value
            }
            await self._send_message(unsubscribe_message)
        
        del self.subscriptions[subscription_id]
        logger.info(f"Unsubscribed from {subscription_id}")
        return True
    
    async def _send_subscription(self, subscription_id: str, subscription: StreamSubscription):
        """Send subscription message to WebSocket"""
        message = {
            "type": "subscribe",
            "subscription_id": subscription_id,
            "stream_type": subscription.stream_type.value,
            "symbols": list(subscription.symbols),
            "parameters": subscription.parameters
        }
        await self._send_message(message)
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message through WebSocket"""
        if not self.is_connected or not self.connection:
            raise RuntimeError("WebSocket not connected")
        
        try:
            await self.connection.send(json.dumps(message))
            logger.debug(f"Sent message: {message['type']}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def listen(self):
        """Listen for incoming WebSocket messages"""
        if not self.is_connected or not self.connection:
            raise RuntimeError("WebSocket not connected")
        
        try:
            async for message in self.connection:
                await self._process_message(message)
        
        except (ConnectionClosedError, ConnectionClosedOK):
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            await self._handle_disconnection()
        
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            await self._handle_disconnection()
        
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket listener: {e}")
            self.is_connected = False
            await self._handle_disconnection()
    
    async def _process_message(self, raw_message: str):
        """Process incoming WebSocket message"""
        try:
            message = json.loads(raw_message)
            message_type = message.get("type", "unknown")
            
            handler = self._message_handlers.get(message_type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"No handler for message type: {message_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_subscription_confirmed(self, message: Dict[str, Any]):
        """Handle subscription confirmation"""
        subscription_id = message.get("subscription_id")
        logger.info(f"Subscription confirmed: {subscription_id}")
    
    async def _handle_subscription_error(self, message: Dict[str, Any]):
        """Handle subscription error"""
        subscription_id = message.get("subscription_id")
        error = message.get("error", "Unknown error")
        logger.error(f"Subscription error for {subscription_id}: {error}")
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat message"""
        logger.debug("Received heartbeat")
        
        # Send heartbeat response
        response = {
            "type": "heartbeat_response",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await self._send_message(response)
    
    async def _handle_error(self, message: Dict[str, Any]):
        """Handle error message"""
        error_code = message.get("code", "UNKNOWN")
        error_message = message.get("message", "Unknown error")
        logger.error(f"Server error {error_code}: {error_message}")
    
    async def _handle_data(self, message: Dict[str, Any]):
        """Handle data message and route to appropriate subscription"""
        subscription_id = message.get("subscription_id")
        stream_type = message.get("stream_type")
        data = message.get("data")
        
        if not subscription_id or subscription_id not in self.subscriptions:
            logger.warning(f"Received data for unknown subscription: {subscription_id}")
            return
        
        subscription = self.subscriptions[subscription_id]
        if not subscription.is_active:
            return
        
        try:
            await subscription.callback(data)
        except Exception as e:
            logger.error(f"Error in subscription callback: {e}")
    
    async def _handle_disconnection(self):
        """Handle WebSocket disconnection"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))  # Exponential backoff
            
            logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay} seconds")
            await asyncio.sleep(delay)
            
            if await self.connect():
                # Re-establish all active subscriptions
                for subscription_id, subscription in self.subscriptions.items():
                    if subscription.is_active:
                        await self._send_subscription(subscription_id, subscription)
                
                # Start listening again
                asyncio.create_task(self.listen())
            else:
                await self._handle_disconnection()  # Try again
        else:
            logger.error("Max reconnection attempts reached. Giving up.")
    
    async def _default_ohlcv_handler(self, data: Dict[str, Any]):
        """Default handler for OHLCV data"""
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        close_price = data.get("close")
        logger.info(f"OHLCV: {symbol} ({timeframe}) - Close: {close_price}")
    
    async def _default_trades_handler(self, data: Dict[str, Any]):
        """Default handler for trade data"""
        symbol = data.get("symbol")
        price = data.get("price")
        volume = data.get("volume")
        logger.info(f"Trade: {symbol} - Price: {price}, Volume: {volume}")
    
    async def _default_orderbook_handler(self, data: Dict[str, Any]):
        """Default handler for order book data"""
        symbol = data.get("symbol")
        best_bid = data.get("bids", [{}])[0].get("price") if data.get("bids") else "N/A"
        best_ask = data.get("asks", [{}])[0].get("price") if data.get("asks") else "N/A"
        logger.info(f"Order Book: {symbol} - Best Bid: {best_bid}, Best Ask: {best_ask}")
    
    async def _default_news_handler(self, data: Dict[str, Any]):
        """Default handler for news data"""
        headline = data.get("headline", "No headline")
        symbols = data.get("symbols", [])
        logger.info(f"News: {headline} (Symbols: {', '.join(symbols)})")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            "is_connected": self.is_connected,
            "reconnect_attempts": self.reconnect_attempts,
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.is_active]),
            "total_subscriptions": len(self.subscriptions),
            "base_url": self.base_url
        }


# Utility functions for easy WebSocket management
async def create_ohlcv_stream(
    symbols: List[str],
    timeframes: List[str] = ["1m", "5m"],
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> NovaSignalWebSocketClient:
    """Create and configure OHLCV stream client"""
    client = NovaSignalWebSocketClient()
    
    if await client.connect():
        await client.subscribe_ohlcv(symbols, timeframes, callback)
        asyncio.create_task(client.listen())
        return client
    else:
        raise RuntimeError("Failed to establish WebSocket connection")


async def create_multi_stream(
    config: Dict[str, Any]
) -> NovaSignalWebSocketClient:
    """Create WebSocket client with multiple stream subscriptions"""
    client = NovaSignalWebSocketClient()
    
    if not await client.connect():
        raise RuntimeError("Failed to establish WebSocket connection")
    
    # Subscribe to different streams based on config
    if "ohlcv" in config:
        ohlcv_config = config["ohlcv"]
        await client.subscribe_ohlcv(
            symbols=ohlcv_config.get("symbols", []),
            timeframes=ohlcv_config.get("timeframes", ["1m"]),
            callback=ohlcv_config.get("callback")
        )
    
    if "trades" in config:
        trades_config = config["trades"]
        await client.subscribe_trades(
            symbols=trades_config.get("symbols", []),
            callback=trades_config.get("callback")
        )
    
    if "orderbook" in config:
        orderbook_config = config["orderbook"]
        await client.subscribe_orderbook(
            symbols=orderbook_config.get("symbols", []),
            depth=orderbook_config.get("depth", 20),
            callback=orderbook_config.get("callback")
        )
    
    if "news" in config:
        news_config = config["news"]
        await client.subscribe_news(
            symbols=news_config.get("symbols", []),
            categories=news_config.get("categories", []),
            callback=news_config.get("callback")
        )
    
    # Start listening
    asyncio.create_task(client.listen())
    return client