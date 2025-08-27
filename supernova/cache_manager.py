"""
SuperNova Cache Manager
High-performance caching system with Redis, in-memory, and database caching
"""

import asyncio
import json
import pickle
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from functools import wraps
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

# Redis imports
try:
    import redis.asyncio as redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Database imports
from sqlalchemy import text, select, delete, update, func
from sqlalchemy.orm import sessionmaker

from .config import settings
from .db import SessionLocal, get_timescale_session, is_timescale_available

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_requests: int = 0
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0

class CacheKeyGenerator:
    """Generate consistent cache keys"""
    
    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list, tuple)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:8])
        
        return ":".join(key_parts)
    
    @staticmethod
    def sentiment_key(symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate cache key for sentiment data"""
        return CacheKeyGenerator.generate_key("sentiment", symbol, timeframe, start_date, end_date)
    
    @staticmethod
    def backtest_key(symbol: str, strategy: str, params: dict) -> str:
        """Generate cache key for backtest results"""
        return CacheKeyGenerator.generate_key("backtest", symbol, strategy, params)
    
    @staticmethod
    def market_data_key(symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for market data"""
        return CacheKeyGenerator.generate_key("market_data", symbol, timeframe, limit)
    
    @staticmethod
    def optimization_key(study_id: str, trial_params: dict) -> str:
        """Generate cache key for optimization results"""
        return CacheKeyGenerator.generate_key("optimization", study_id, trial_params)

class InMemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.metrics = CacheMetrics()
        
    def _evict_lru(self):
        """Evict least recently used items"""
        while len(self.cache) >= self.max_size:
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    del self.cache[lru_key]
                    self.metrics.evictions += 1
            else:
                break
    
    def _update_access_order(self, key: str):
        """Update access order for LRU"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _is_expired(self, item: Dict) -> bool:
        """Check if cache item is expired"""
        if item.get('ttl', 0) == 0:
            return False
        return time.time() > item['timestamp'] + item['ttl']
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        start_time = time.time()
        
        with self.lock:
            self.metrics.total_requests += 1
            
            if key not in self.cache:
                self.metrics.misses += 1
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if self._is_expired(item):
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.metrics.misses += 1
                return None
            
            # Update access order
            self._update_access_order(key)
            
            self.metrics.hits += 1
            
            # Update average get time
            get_time = (time.time() - start_time) * 1000
            self.metrics.avg_get_time_ms = (
                (self.metrics.avg_get_time_ms * (self.metrics.hits - 1) + get_time) / self.metrics.hits
            )
            
            return item['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        start_time = time.time()
        
        with self.lock:
            # Evict if necessary
            if key not in self.cache:
                self._evict_lru()
            
            # Set item
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl
            }
            
            # Update access order
            self._update_access_order(key)
            
            self.metrics.sets += 1
            
            # Update average set time
            set_time = (time.time() - start_time) * 1000
            self.metrics.avg_set_time_ms = (
                (self.metrics.avg_set_time_ms * (self.metrics.sets - 1) + set_time) / self.metrics.sets
            )
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.metrics.deletes += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache items"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> CacheMetrics:
        """Get cache statistics"""
        with self.lock:
            self.metrics.hit_rate = (
                self.metrics.hits / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            )
            
            # Estimate memory usage (rough calculation)
            total_size = 0
            for item in self.cache.values():
                try:
                    total_size += len(pickle.dumps(item['value']))
                except:
                    total_size += 1024  # Rough estimate for unpickleable objects
            
            self.metrics.memory_usage_mb = total_size / 1024 / 1024
            
            return self.metrics

class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self.metrics = CacheMetrics()
        
    async def connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache initialization")
            return False
            
        try:
            # Parse Redis URL or use default settings
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            
            self.redis_client = redis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            logger.info("Redis cache connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache"""
        if not self.is_connected:
            return None
            
        start_time = time.time()
        
        try:
            self.metrics.total_requests += 1
            
            value = await self.redis_client.get(key)
            
            if value is None:
                self.metrics.misses += 1
                return None
            
            # Deserialize value
            try:
                deserialized = pickle.loads(value)
                self.metrics.hits += 1
                
                # Update metrics
                get_time = (time.time() - start_time) * 1000
                self.metrics.avg_get_time_ms = (
                    (self.metrics.avg_get_time_ms * (self.metrics.hits - 1) + get_time) / self.metrics.hits
                )
                
                return deserialized
                
            except (pickle.PickleError, EOFError):
                # Fallback to JSON if pickle fails
                try:
                    return json.loads(value.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.error(f"Failed to deserialize cached value for key: {key}")
                    self.metrics.misses += 1
                    return None
                    
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis cache"""
        if not self.is_connected:
            return False
            
        start_time = time.time()
        
        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except (pickle.PickleError, TypeError):
                # Fallback to JSON for simple types
                try:
                    serialized = json.dumps(value).encode('utf-8')
                except (TypeError, ValueError):
                    logger.error(f"Failed to serialize value for key: {key}")
                    return False
            
            # Set with TTL
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
            
            self.metrics.sets += 1
            
            # Update metrics
            set_time = (time.time() - start_time) * 1000
            self.metrics.avg_set_time_ms = (
                (self.metrics.avg_set_time_ms * (self.metrics.sets - 1) + set_time) / self.metrics.sets
            )
            
            return True
            
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        if not self.is_connected:
            return False
            
        try:
            result = await self.redis_client.delete(key)
            if result:
                self.metrics.deletes += 1
            return bool(result)
            
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self.is_connected:
            return 0
            
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.metrics.deletes += deleted
                return deleted
            return 0
            
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis clear pattern error for {pattern}: {e}")
            return 0
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        if not self.is_connected:
            return {}
            
        try:
            info = await self.redis_client.info()
            return {
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}

class DatabaseCache:
    """Database-backed cache for persistent caching"""
    
    def __init__(self):
        self.metrics = CacheMetrics()
        self._ensure_cache_table()
    
    def _ensure_cache_table(self):
        """Ensure cache table exists"""
        try:
            db = SessionLocal()
            
            # Create cache table if it doesn't exist
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key VARCHAR(255) PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create index for expiration cleanup
            try:
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)"))
                db.execute(text("CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache_entries(last_accessed)"))
            except:
                pass  # Indexes might already exist
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error creating cache table: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from database cache"""
        start_time = time.time()
        
        try:
            db = SessionLocal()
            
            self.metrics.total_requests += 1
            
            # Get cache entry
            result = db.execute(text("""
                SELECT value FROM cache_entries 
                WHERE cache_key = :key 
                AND (expires_at IS NULL OR expires_at > NOW())
            """), {'key': key}).fetchone()
            
            if not result:
                self.metrics.misses += 1
                db.close()
                return None
            
            # Update access statistics
            db.execute(text("""
                UPDATE cache_entries 
                SET access_count = access_count + 1, last_accessed = NOW()
                WHERE cache_key = :key
            """), {'key': key})
            db.commit()
            
            # Deserialize value
            try:
                value = json.loads(result[0])
                self.metrics.hits += 1
                
                # Update metrics
                get_time = (time.time() - start_time) * 1000
                self.metrics.avg_get_time_ms = (
                    (self.metrics.avg_get_time_ms * (self.metrics.hits - 1) + get_time) / self.metrics.hits
                )
                
                db.close()
                return value
                
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to deserialize database cache value for key: {key}")
                self.metrics.misses += 1
                db.close()
                return None
                
        except Exception as e:
            logger.error(f"Database cache get error for key {key}: {e}")
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in database cache"""
        start_time = time.time()
        
        try:
            db = SessionLocal()
            
            # Serialize value
            try:
                serialized = json.dumps(value, default=str)
            except (TypeError, ValueError):
                logger.error(f"Failed to serialize value for key: {key}")
                db.close()
                return False
            
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Upsert cache entry
            db.execute(text("""
                INSERT INTO cache_entries (cache_key, value, expires_at, access_count, last_accessed)
                VALUES (:key, :value, :expires_at, 1, NOW())
                ON CONFLICT (cache_key) DO UPDATE SET
                    value = :value,
                    expires_at = :expires_at,
                    last_accessed = NOW()
            """), {
                'key': key,
                'value': serialized,
                'expires_at': expires_at
            })
            
            db.commit()
            db.close()
            
            self.metrics.sets += 1
            
            # Update metrics
            set_time = (time.time() - start_time) * 1000
            self.metrics.avg_set_time_ms = (
                (self.metrics.avg_set_time_ms * (self.metrics.sets - 1) + set_time) / self.metrics.sets
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Database cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from database cache"""
        try:
            db = SessionLocal()
            
            result = db.execute(text("DELETE FROM cache_entries WHERE cache_key = :key"), {'key': key})
            deleted = result.rowcount > 0
            
            if deleted:
                self.metrics.deletes += 1
            
            db.commit()
            db.close()
            
            return deleted
            
        except Exception as e:
            logger.error(f"Database cache delete error for key {key}: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        try:
            db = SessionLocal()
            
            result = db.execute(text("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < NOW()"))
            deleted = result.rowcount
            
            db.commit()
            db.close()
            
            self.metrics.evictions += deleted
            logger.info(f"Cleaned up {deleted} expired cache entries")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Database cache cleanup error: {e}")
            return 0

class MultiLevelCache:
    """Multi-level cache with in-memory, Redis, and database backends"""
    
    def __init__(self):
        self.l1_cache = InMemoryCache(max_size=5000, default_ttl=300)  # 5 minutes L1
        self.l2_cache = RedisCache()
        self.l3_cache = DatabaseCache()
        
        self.l2_enabled = False
        self.l3_enabled = True
        
        # Cache level preferences
        self.l1_ttl = 300    # 5 minutes
        self.l2_ttl = 3600   # 1 hour
        self.l3_ttl = 86400  # 24 hours
    
    async def initialize(self):
        """Initialize cache backends"""
        logger.info("Initializing multi-level cache")
        
        # Initialize Redis (L2)
        if await self.l2_cache.connect():
            self.l2_enabled = True
            logger.info("L2 (Redis) cache enabled")
        else:
            logger.info("L2 (Redis) cache disabled")
        
        logger.info("Multi-level cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (tries L1, L2, L3 in order)"""
        # Try L1 (in-memory)
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 (Redis)
        if self.l2_enabled:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Populate L1
                self.l1_cache.set(key, value, ttl=self.l1_ttl)
                return value
        
        # Try L3 (Database)
        if self.l3_enabled:
            value = await self.l3_cache.get(key)
            if value is not None:
                # Populate L1 and L2
                self.l1_cache.set(key, value, ttl=self.l1_ttl)
                if self.l2_enabled:
                    await self.l2_cache.set(key, value, ttl=self.l2_ttl)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in all cache levels"""
        results = []
        
        # Set in L1
        l1_ttl = min(ttl or self.l1_ttl, self.l1_ttl)
        results.append(self.l1_cache.set(key, value, ttl=l1_ttl))
        
        # Set in L2
        if self.l2_enabled:
            l2_ttl = min(ttl or self.l2_ttl, self.l2_ttl)
            results.append(await self.l2_cache.set(key, value, ttl=l2_ttl))
        
        # Set in L3
        if self.l3_enabled:
            l3_ttl = ttl or self.l3_ttl
            results.append(await self.l3_cache.set(key, value, ttl=l3_ttl))
        
        return any(results)
    
    async def delete(self, key: str) -> bool:
        """Delete item from all cache levels"""
        results = []
        
        results.append(self.l1_cache.delete(key))
        
        if self.l2_enabled:
            results.append(await self.l2_cache.delete(key))
        
        if self.l3_enabled:
            results.append(await self.l3_cache.delete(key))
        
        return any(results)
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern from all levels"""
        total_deleted = 0
        
        # L1 cache doesn't support pattern matching, so clear all
        if '*' in pattern:
            self.l1_cache.clear()
        
        # L2 cache
        if self.l2_enabled:
            total_deleted += await self.l2_cache.clear_pattern(pattern)
        
        return total_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'l1_cache': self.l1_cache.get_stats(),
            'l2_enabled': self.l2_enabled,
            'l3_enabled': self.l3_enabled
        }
        
        if self.l2_enabled:
            stats['l2_cache'] = self.l2_cache.metrics
        
        if self.l3_enabled:
            stats['l3_cache'] = self.l3_cache.metrics
        
        # Calculate combined hit rate
        total_hits = stats['l1_cache'].hits
        total_requests = stats['l1_cache'].total_requests
        
        if self.l2_enabled:
            total_hits += self.l2_cache.metrics.hits
            total_requests += self.l2_cache.metrics.total_requests
        
        if self.l3_enabled:
            total_hits += self.l3_cache.metrics.hits
            total_requests += self.l3_cache.metrics.total_requests
        
        stats['combined'] = {
            'hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'total_requests': total_requests,
            'total_hits': total_hits
        }
        
        return stats

# Global cache instance
cache_manager = MultiLevelCache()

def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    use_args: bool = True,
    use_kwargs: bool = True,
    cache_none: bool = False
):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_prefix:
                prefix = key_prefix
            else:
                prefix = f"{func.__module__}.{func.__name__}"
            
            cache_args = args if use_args else ()
            cache_kwargs = kwargs if use_kwargs else {}
            
            cache_key = CacheKeyGenerator.generate_key(prefix, *cache_args, **cache_kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result (unless it's None and cache_none is False)
            if result is not None or cache_none:
                await cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run async cache operations
            async def _async_cached_call():
                # Generate cache key
                if key_prefix:
                    prefix = key_prefix
                else:
                    prefix = f"{func.__module__}.{func.__name__}"
                
                cache_args = args if use_args else ()
                cache_kwargs = kwargs if use_kwargs else {}
                
                cache_key = CacheKeyGenerator.generate_key(prefix, *cache_args, **cache_kwargs)
                
                # Try to get from cache
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                if result is not None or cache_none:
                    await cache_manager.set(cache_key, result, ttl=ttl)
                
                return result
            
            # Run in current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_cached_call())
                        return future.result()
                else:
                    return loop.run_until_complete(_async_cached_call())
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(_async_cached_call())
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

async def warm_cache():
    """Pre-populate cache with commonly accessed data"""
    logger.info("Starting cache warm-up")
    
    try:
        # Warm up with recent sentiment data
        from .sentiment import get_recent_sentiment_summary
        
        # Common symbols to pre-cache
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ']
        
        tasks = []
        for symbol in symbols:
            # Create warm-up tasks for different timeframes
            tasks.extend([
                get_recent_sentiment_summary(symbol, days=1),
                get_recent_sentiment_summary(symbol, days=7),
                get_recent_sentiment_summary(symbol, days=30)
            ])
        
        # Execute warm-up tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Cache warm-up completed: {successful}/{len(results)} successful")
        
    except Exception as e:
        logger.error(f"Error during cache warm-up: {e}")

async def cache_maintenance():
    """Perform cache maintenance tasks"""
    while True:
        try:
            logger.info("Running cache maintenance")
            
            # Clean up expired database cache entries
            if cache_manager.l3_enabled:
                await cache_manager.l3_cache.cleanup_expired()
            
            # Log cache statistics
            stats = cache_manager.get_stats()
            logger.info(f"Cache stats - Combined hit rate: {stats['combined']['hit_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Error during cache maintenance: {e}")
        
        # Run maintenance every 30 minutes
        await asyncio.sleep(1800)

# Cache initialization and startup
async def initialize_cache():
    """Initialize cache system"""
    await cache_manager.initialize()
    
    # Start maintenance task
    asyncio.create_task(cache_maintenance())
    
    # Warm up cache
    asyncio.create_task(warm_cache())
    
    logger.info("Cache system initialized")

# Export key components
__all__ = [
    'cache_manager', 'cached', 'CacheKeyGenerator',
    'initialize_cache', 'MultiLevelCache'
]