"""NovaSignal Integration Connector

Comprehensive bidirectional integration with NovaSignal platform providing:
- Real-time OHLCV data streaming via WebSocket
- REST API for historical data and user operations
- Alert delivery system with webhooks
- Profile and watchlist synchronization
- Circuit breaker patterns and error handling
"""

import asyncio
import json
import logging
import hashlib
import hmac
from datetime import datetime, timezone
from typing import List, Dict, Optional, AsyncGenerator, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..supernova.config import settings
from ..supernova.schemas import OHLCVBar

# Configure logger
logger = logging.getLogger(__name__)


class NovaSignalConnectionState(Enum):
    """Connection states for NovaSignal integration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class AlertPriority(Enum):
    """Alert priority levels for NovaSignal dashboard"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NovaSignalAlert:
    """Structured alert for NovaSignal platform"""
    symbol: str
    message: str
    priority: AlertPriority
    timestamp: str
    profile_id: Optional[int] = None
    confidence: Optional[float] = None
    action: Optional[str] = None
    strategy: Optional[str] = None
    indicators: Optional[Dict[str, float]] = None
    dashboard_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MarketDataSubscription:
    """Market data subscription configuration"""
    symbols: List[str]
    asset_classes: List[str]
    timeframes: List[str]
    callback: Callable[[Dict[str, Any]], None]


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and 
            (datetime.now().timestamp() - self.last_failure_time) > self.recovery_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class NovaSignalConnector:
    """Main NovaSignal integration connector with full bidirectional capabilities"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.redis_client: Optional[redis.Redis] = None
        self.connection_state = NovaSignalConnectionState.DISCONNECTED
        self.subscriptions: List[MarketDataSubscription] = []
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.NOVASIGNAL_MAX_RETRIES,
            recovery_timeout=60
        )
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._data_buffer = asyncio.Queue(maxsize=settings.NOVASIGNAL_BUFFER_SIZE)
    
    async def initialize(self):
        """Initialize all connections and components"""
        logger.info("Initializing NovaSignal connector...")
        
        # Initialize HTTP session with authentication
        timeout = aiohttp.ClientTimeout(total=settings.NOVASIGNAL_CONNECTION_TIMEOUT)
        headers = self._get_auth_headers()
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            base_url=settings.NOVASIGNAL_API_URL
        )
        
        # Initialize Redis for caching and buffering
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Proceeding without cache.")
            self.redis_client = None
        
        logger.info("NovaSignal connector initialized successfully")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers for API requests"""
        headers = {
            "User-Agent": "SuperNova-Connector/1.0",
            "Content-Type": "application/json"
        }
        
        if settings.NOVASIGNAL_API_KEY:
            headers["X-API-Key"] = settings.NOVASIGNAL_API_KEY
        
        return headers
    
    def _generate_webhook_signature(self, payload: str) -> str:
        """Generate HMAC signature for webhook verification"""
        if not settings.NOVASIGNAL_WEBHOOK_SECRET:
            return ""
        
        return hmac.new(
            settings.NOVASIGNAL_WEBHOOK_SECRET.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    async def _make_api_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated API request with retry logic"""
        if not self.session:
            raise RuntimeError("Connector not initialized")
        
        async with self.session.request(
            method,
            endpoint,
            json=data,
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def connect_websocket(self) -> None:
        """Establish WebSocket connection for real-time data"""
        if self.connection_state == NovaSignalConnectionState.CONNECTED:
            return
        
        self.connection_state = NovaSignalConnectionState.CONNECTING
        logger.info("Connecting to NovaSignal WebSocket...")
        
        try:
            # Add authentication to WebSocket connection
            auth_params = {}
            if settings.NOVASIGNAL_API_KEY:
                auth_params["api_key"] = settings.NOVASIGNAL_API_KEY
            
            ws_url = f"{settings.NOVASIGNAL_WS_URL}/stream"
            if auth_params:
                ws_url += "?" + "&".join([f"{k}={v}" for k, v in auth_params.items()])
            
            self.ws_connection = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.connection_state = NovaSignalConnectionState.CONNECTED
            logger.info("WebSocket connection established")
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
        except Exception as e:
            self.connection_state = NovaSignalConnectionState.ERROR
            logger.error(f"WebSocket connection failed: {e}")
            await self._schedule_reconnect()
            raise
    
    async def _heartbeat_loop(self):
        """Maintain WebSocket connection with heartbeat"""
        while self.ws_connection and not self.ws_connection.closed:
            try:
                await self.ws_connection.ping()
                await asyncio.sleep(30)
            except (ConnectionClosedError, ConnectionClosedOK):
                logger.warning("WebSocket connection lost during heartbeat")
                await self._handle_disconnect()
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnection and attempt reconnect"""
        self.connection_state = NovaSignalConnectionState.DISCONNECTED
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        await self._schedule_reconnect()
    
    async def _schedule_reconnect(self, delay: int = 5):
        """Schedule automatic reconnection"""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        
        self.connection_state = NovaSignalConnectionState.RECONNECTING
        logger.info(f"Scheduling reconnect in {delay} seconds...")
        
        async def reconnect():
            await asyncio.sleep(delay)
            try:
                await self.connect_websocket()
                # Resubscribe to all active subscriptions
                for subscription in self.subscriptions:
                    await self._send_subscription(subscription)
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await self._schedule_reconnect(delay * 2)  # Exponential backoff
        
        self._reconnect_task = asyncio.create_task(reconnect())
    
    async def _send_subscription(self, subscription: MarketDataSubscription):
        """Send subscription message via WebSocket"""
        if not self.ws_connection or self.connection_state != NovaSignalConnectionState.CONNECTED:
            raise RuntimeError("WebSocket not connected")
        
        message = {
            "type": "subscribe",
            "symbols": subscription.symbols,
            "asset_classes": subscription.asset_classes,
            "timeframes": subscription.timeframes
        }
        
        await self.ws_connection.send(json.dumps(message))
        logger.info(f"Subscribed to {len(subscription.symbols)} symbols")
    
    async def subscribe_market_data(
        self,
        symbols: List[str],
        asset_classes: List[str] = ["stock"],
        timeframes: List[str] = ["1m", "5m", "1h"],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """Subscribe to real-time market data"""
        subscription = MarketDataSubscription(
            symbols=symbols,
            asset_classes=asset_classes,
            timeframes=timeframes,
            callback=callback or self._default_market_data_handler
        )
        
        self.subscriptions.append(subscription)
        
        if self.connection_state == NovaSignalConnectionState.CONNECTED:
            await self._send_subscription(subscription)
    
    async def _default_market_data_handler(self, data: Dict[str, Any]):
        """Default handler for market data - stores to buffer and cache"""
        try:
            # Add to processing queue
            if not self._data_buffer.full():
                await self._data_buffer.put(data)
            
            # Cache to Redis if available
            if self.redis_client:
                cache_key = f"ohlcv:{data.get('symbol')}:{data.get('timeframe')}"
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute expiry
                    json.dumps(data)
                )
            
            logger.debug(f"Processed market data for {data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def stream_market_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time market data from WebSocket"""
        if not self.ws_connection or self.connection_state != NovaSignalConnectionState.CONNECTED:
            raise RuntimeError("WebSocket not connected")
        
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "ohlcv":
                        # Process and yield OHLCV data
                        processed_data = await self._process_ohlcv_data(data)
                        if processed_data:
                            yield processed_data
                    
                    elif data.get("type") == "error":
                        logger.error(f"WebSocket error: {data.get('message')}")
                    
                    elif data.get("type") == "heartbeat":
                        logger.debug("Received heartbeat")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                
        except (ConnectionClosedError, ConnectionClosedOK):
            logger.warning("WebSocket connection closed during streaming")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Error in market data stream: {e}")
            raise
    
    async def _process_ohlcv_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate OHLCV data from NovaSignal"""
        try:
            # Extract and validate OHLCV fields
            ohlcv_data = {
                "symbol": raw_data.get("symbol"),
                "timeframe": raw_data.get("timeframe"),
                "timestamp": raw_data.get("timestamp"),
                "open": float(raw_data.get("open", 0)),
                "high": float(raw_data.get("high", 0)),
                "low": float(raw_data.get("low", 0)),
                "close": float(raw_data.get("close", 0)),
                "volume": float(raw_data.get("volume", 0)),
                "asset_class": raw_data.get("asset_class", "stock")
            }
            
            # Validate required fields
            if not all([ohlcv_data["symbol"], ohlcv_data["timestamp"]]):
                logger.warning("Invalid OHLCV data: missing required fields")
                return None
            
            # Call subscription callbacks
            for subscription in self.subscriptions:
                if (
                    ohlcv_data["symbol"] in subscription.symbols and
                    ohlcv_data["timeframe"] in subscription.timeframes
                ):
                    try:
                        await subscription.callback(ohlcv_data)
                    except Exception as e:
                        logger.error(f"Error in subscription callback: {e}")
            
            return ohlcv_data
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing OHLCV data: {e}")
            return None
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        asset_class: str = "stock"
    ) -> List[OHLCVBar]:
        """Fetch historical OHLCV data via REST API"""
        cache_key = f"historical:{symbol}:{timeframe}:{limit}"
        
        # Try cache first
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Retrieved historical data from cache for {symbol}")
                data = json.loads(cached)
                return [OHLCVBar(**bar) for bar in data]
        
        # Fetch from API
        try:
            response = await self.circuit_breaker.call(
                self._make_api_request,
                "GET",
                "/market-data/historical",
                params={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                    "asset_class": asset_class
                }
            )
            
            bars = []
            for bar_data in response.get("data", []):
                try:
                    bar = OHLCVBar(
                        timestamp=bar_data["timestamp"],
                        open=float(bar_data["open"]),
                        high=float(bar_data["high"]),
                        low=float(bar_data["low"]),
                        close=float(bar_data["close"]),
                        volume=float(bar_data["volume"])
                    )
                    bars.append(bar)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid bar data: {e}")
                    continue
            
            # Cache the results
            if self.redis_client and bars:
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute cache
                    json.dumps([bar.model_dump() for bar in bars])
                )
            
            logger.info(f"Fetched {len(bars)} historical bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise
    
    async def push_alert_to_novasignal(
        self,
        symbol: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        profile_id: Optional[int] = None,
        confidence: Optional[float] = None,
        action: Optional[str] = None,
        strategy: Optional[str] = None,
        indicators: Optional[Dict[str, float]] = None
    ) -> bool:
        """Send structured alert to NovaSignal dashboard"""
        try:
            alert = NovaSignalAlert(
                symbol=symbol,
                message=message,
                priority=priority,
                timestamp=datetime.now(timezone.utc).isoformat(),
                profile_id=profile_id,
                confidence=confidence,
                action=action,
                strategy=strategy,
                indicators=indicators,
                dashboard_url=f"{settings.NOVASIGNAL_DASHBOARD_URL}/alerts"
            )
            
            # Prepare webhook payload
            payload = json.dumps(alert.to_dict())
            signature = self._generate_webhook_signature(payload)
            
            headers = {
                "X-SuperNova-Signature": f"sha256={signature}",
                "X-SuperNova-Timestamp": str(int(datetime.now().timestamp()))
            }
            
            await self.circuit_breaker.call(
                self._make_api_request,
                "POST",
                settings.NOVASIGNAL_ALERT_ENDPOINT,
                data=alert.to_dict()
            )
            
            logger.info(f"Alert sent to NovaSignal: {symbol} - {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert to NovaSignal: {e}")
            return False
    
    async def get_watchlist(self, profile_id: int) -> List[Dict[str, Any]]:
        """Fetch user watchlist from NovaSignal"""
        try:
            response = await self.circuit_breaker.call(
                self._make_api_request,
                "GET",
                f"/users/{profile_id}/watchlist"
            )
            
            watchlist = response.get("watchlist", [])
            logger.info(f"Retrieved watchlist with {len(watchlist)} items for profile {profile_id}")
            return watchlist
            
        except Exception as e:
            logger.error(f"Failed to fetch watchlist for profile {profile_id}: {e}")
            return []
    
    async def sync_profile(
        self,
        profile_id: int,
        profile_data: Dict[str, Any]
    ) -> bool:
        """Synchronize user profile with NovaSignal"""
        try:
            await self.circuit_breaker.call(
                self._make_api_request,
                "PUT",
                f"/users/{profile_id}/profile",
                data=profile_data
            )
            
            logger.info(f"Profile synchronized for user {profile_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync profile for user {profile_id}: {e}")
            return False
    
    async def send_advice_explanation(
        self,
        profile_id: int,
        advice_data: Dict[str, Any]
    ) -> bool:
        """Send advice explanation for NovaSignal UI rendering"""
        try:
            explanation_data = {
                "profile_id": profile_id,
                "advice": advice_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "supernova"
            }
            
            await self.circuit_breaker.call(
                self._make_api_request,
                "POST",
                f"/users/{profile_id}/advice-explanations",
                data=explanation_data
            )
            
            logger.info(f"Advice explanation sent for profile {profile_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send advice explanation: {e}")
            return False
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and diagnostics"""
        return {
            "connection_state": self.connection_state.value,
            "websocket_connected": (
                self.ws_connection is not None and
                not self.ws_connection.closed
            ),
            "http_session_active": self.session is not None,
            "redis_available": self.redis_client is not None,
            "active_subscriptions": len(self.subscriptions),
            "circuit_breaker_state": self.circuit_breaker.state,
            "buffer_size": self._data_buffer.qsize()
        }
    
    async def close(self):
        """Clean shutdown of all connections"""
        logger.info("Closing NovaSignal connector...")
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._reconnect_task:
            self._reconnect_task.cancel()
        
        # Close WebSocket
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        self.connection_state = NovaSignalConnectionState.DISCONNECTED
        logger.info("NovaSignal connector closed")


# Global connector instance
_connector: Optional[NovaSignalConnector] = None


async def get_connector() -> NovaSignalConnector:
    """Get or create global connector instance"""
    global _connector
    if _connector is None:
        _connector = NovaSignalConnector()
        await _connector.initialize()
    return _connector


@asynccontextmanager
async def novasignal_connection():
    """Context manager for NovaSignal connector lifecycle"""
    connector = await get_connector()
    try:
        await connector.connect_websocket()
        yield connector
    finally:
        # Keep connection alive for reuse
        pass


# Legacy compatibility functions
async def push_alert_to_novasignal(
    symbol: str, 
    message: str,
    priority: str = "medium",
    **kwargs
) -> None:
    """Legacy compatibility wrapper for alert pushing"""
    connector = await get_connector()
    priority_enum = AlertPriority(priority.lower())
    await connector.push_alert_to_novasignal(
        symbol=symbol,
        message=message,
        priority=priority_enum,
        **kwargs
    )


async def get_watchlist(profile_id: int) -> List[Dict]:
    """Legacy compatibility wrapper for watchlist fetching"""
    connector = await get_connector()
    return await connector.get_watchlist(profile_id)


# Utility functions for integration
async def stream_ohlcv_data(
    symbols: List[str],
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    """Start streaming OHLCV data for specified symbols"""
    connector = await get_connector()
    
    await connector.subscribe_market_data(
        symbols=symbols,
        callback=callback
    )
    
    # Stream data indefinitely
    async for data in connector.stream_market_data():
        logger.debug(f"Received OHLCV data: {data['symbol']}")


async def fetch_historical_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100
) -> List[OHLCVBar]:
    """Fetch historical OHLCV data"""
    connector = await get_connector()
    return await connector.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit
    )


async def health_check() -> Dict[str, Any]:
    """Perform health check on NovaSignal integration"""
    try:
        connector = await get_connector()
        return await connector.get_connection_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
