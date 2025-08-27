"""
SuperNova Performance Monitor
Comprehensive performance monitoring and optimization system
"""

import asyncio
import time
import psutil
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI and HTTP performance monitoring
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Database and caching imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# TimescaleDB for metrics storage
try:
    from .db import get_timescale_session, is_timescale_available
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False

from .config import settings

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class ResourceMetrics:
    """System resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    thread_count: int
    timestamp: datetime

@dataclass
class EndpointMetrics:
    """API endpoint performance metrics"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    request_size_bytes: int
    response_size_bytes: int
    timestamp: datetime
    user_agent: str = ""
    ip_address: str = ""
    query_count: int = 0
    cache_hit: bool = False

@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    operation: str
    query_time_ms: float
    rows_affected: int
    connection_pool_size: int
    active_connections: int
    query_hash: str
    timestamp: datetime
    table_name: str = ""
    index_used: bool = False

class PerformanceCollector:
    """Centralized performance metrics collector"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.endpoint_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_metrics: deque = deque(maxlen=1000)
        self.database_metrics: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Resource monitoring
        self.process = psutil.Process()
        self.last_io_counters = None
        self.last_net_counters = None
        
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric"""
        with self.lock:
            self.metrics.append(metric)
    
    def add_endpoint_metric(self, metric: EndpointMetrics):
        """Add an API endpoint metric"""
        with self.lock:
            key = f"{metric.method}:{metric.endpoint}"
            self.endpoint_metrics[key].append(metric)
            self.request_count += 1
            if metric.status_code >= 400:
                self.error_count += 1
            if metric.cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def add_resource_metric(self, metric: ResourceMetrics):
        """Add a system resource metric"""
        with self.lock:
            self.resource_metrics.append(metric)
    
    def add_database_metric(self, metric: DatabaseMetrics):
        """Add a database metric"""
        with self.lock:
            self.database_metrics.append(metric)
    
    def collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU and Memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            try:
                io_counters = self.process.io_counters()
                if self.last_io_counters:
                    disk_read_mb = (io_counters.read_bytes - self.last_io_counters.read_bytes) / 1024 / 1024
                    disk_write_mb = (io_counters.write_bytes - self.last_io_counters.write_bytes) / 1024 / 1024
                else:
                    disk_read_mb = 0.0
                    disk_write_mb = 0.0
                self.last_io_counters = io_counters
            except:
                disk_read_mb = 0.0
                disk_write_mb = 0.0
            
            # Network I/O
            try:
                net_counters = psutil.net_io_counters()
                if self.last_net_counters:
                    net_sent_mb = (net_counters.bytes_sent - self.last_net_counters.bytes_sent) / 1024 / 1024
                    net_recv_mb = (net_counters.bytes_recv - self.last_net_counters.bytes_recv) / 1024 / 1024
                else:
                    net_sent_mb = 0.0
                    net_recv_mb = 0.0
                self.last_net_counters = net_counters
            except:
                net_sent_mb = 0.0
                net_recv_mb = 0.0
            
            # Process info
            connections = len(self.process.connections())
            threads = self.process.num_threads()
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_info.rss / 1024 / 1024,
                memory_available_mb=psutil.virtual_memory().available / 1024 / 1024,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                active_connections=connections,
                thread_count=threads,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def get_endpoint_stats(self, endpoint: str = None) -> Dict[str, Any]:
        """Get endpoint performance statistics"""
        with self.lock:
            if endpoint:
                metrics = self.endpoint_metrics.get(endpoint, [])
            else:
                # Aggregate all endpoints
                metrics = []
                for endpoint_metrics in self.endpoint_metrics.values():
                    metrics.extend(endpoint_metrics)
            
            if not metrics:
                return {}
            
            response_times = [m.response_time_ms for m in metrics]
            status_codes = [m.status_code for m in metrics]
            request_sizes = [m.request_size_bytes for m in metrics]
            response_sizes = [m.response_size_bytes for m in metrics]
            
            return {
                "request_count": len(metrics),
                "avg_response_time_ms": statistics.mean(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "p99_response_time_ms": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0,
                "error_rate": sum(1 for sc in status_codes if sc >= 400) / len(status_codes),
                "avg_request_size_bytes": statistics.mean(request_sizes) if request_sizes else 0,
                "avg_response_size_bytes": statistics.mean(response_sizes) if response_sizes else 0,
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics)
            }
    
    def get_resource_stats(self, minutes: int = 5) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [m for m in self.resource_metrics if m.timestamp >= cutoff]
            
            if not recent_metrics:
                return {}
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            memory_used = [m.memory_used_mb for m in recent_metrics]
            
            return {
                "avg_cpu_percent": statistics.mean(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "avg_memory_percent": statistics.mean(memory_values),
                "max_memory_percent": max(memory_values),
                "current_memory_mb": memory_used[-1] if memory_used else 0,
                "peak_memory_mb": max(memory_used),
                "measurement_count": len(recent_metrics),
                "time_range_minutes": minutes
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        with self.lock:
            if not self.database_metrics:
                return {}
            
            query_times = [m.query_time_ms for m in self.database_metrics]
            operations = [m.operation for m in self.database_metrics]
            
            # Group by operation type
            op_stats = defaultdict(list)
            for metric in self.database_metrics:
                op_stats[metric.operation].append(metric.query_time_ms)
            
            return {
                "total_queries": len(self.database_metrics),
                "avg_query_time_ms": statistics.mean(query_times),
                "slow_queries": sum(1 for qt in query_times if qt > 1000),  # > 1 second
                "operations": dict(op_stats),
                "operation_breakdown": {
                    op: {
                        "count": len(times),
                        "avg_time_ms": statistics.mean(times),
                        "max_time_ms": max(times)
                    }
                    for op, times in op_stats.items()
                }
            }

# Global performance collector instance
performance_collector = PerformanceCollector()

class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for performance monitoring"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request information
        method = request.method
        url_path = str(request.url.path)
        user_agent = request.headers.get("user-agent", "")
        client_ip = request.client.host if request.client else ""
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            body = await request.body()
            request_size = len(body)
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            response_size = len(response.body)
        
        # Check cache headers
        cache_hit = 'X-Cache-Hit' in response.headers
        
        # Create endpoint metric
        endpoint_metric = EndpointMetrics(
            endpoint=url_path,
            method=method,
            response_time_ms=response_time_ms,
            status_code=response.status_code,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            timestamp=datetime.now(),
            user_agent=user_agent,
            ip_address=client_ip,
            cache_hit=cache_hit
        )
        
        # Add to collector
        performance_collector.add_endpoint_metric(endpoint_metric)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        response.headers["X-Process-Time"] = str(int(time.time()))
        
        return response

def performance_timer(category: str = "function", name: str = None):
    """Decorator for timing function execution"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                metric_name = name or f"{func.__module__}.{func.__name__}"
                
                metric = PerformanceMetric(
                    name=metric_name,
                    value=execution_time,
                    unit="ms",
                    timestamp=datetime.now(),
                    category=category,
                    tags={"function": func.__name__, "module": func.__module__}
                )
                performance_collector.add_metric(metric)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                metric_name = name or f"{func.__module__}.{func.__name__}"
                
                metric = PerformanceMetric(
                    name=metric_name,
                    value=execution_time,
                    unit="ms",
                    timestamp=datetime.now(),
                    category=category,
                    tags={"function": func.__name__, "module": func.__module__}
                )
                performance_collector.add_metric(metric)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@contextmanager
def performance_context(name: str, category: str = "operation"):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = (time.time() - start_time) * 1000
        
        metric = PerformanceMetric(
            name=name,
            value=execution_time,
            unit="ms",
            timestamp=datetime.now(),
            category=category
        )
        performance_collector.add_metric(metric)

class DatabasePerformanceMonitor:
    """Monitor database query performance"""
    
    def __init__(self):
        self.slow_query_threshold = 1000  # 1 second
        
    @contextmanager
    def monitor_query(self, operation: str, table_name: str = "", query_hash: str = ""):
        """Monitor database query performance"""
        start_time = time.time()
        connection_count = 0
        pool_size = 0
        
        try:
            # Get connection pool info if available
            from .db import SessionLocal
            if hasattr(SessionLocal, 'pool'):
                pool = SessionLocal.pool
                connection_count = pool.checked_in() + pool.checked_out()
                pool_size = pool.size()
                
            yield
            
        finally:
            query_time = (time.time() - start_time) * 1000
            
            # Create database metric
            db_metric = DatabaseMetrics(
                operation=operation,
                query_time_ms=query_time,
                rows_affected=0,  # Would be filled by actual implementation
                connection_pool_size=pool_size,
                active_connections=connection_count,
                query_hash=query_hash,
                timestamp=datetime.now(),
                table_name=table_name
            )
            
            performance_collector.add_database_metric(db_metric)
            
            # Log slow queries
            if query_time > self.slow_query_threshold:
                logger.warning(
                    f"Slow query detected: {operation} on {table_name} "
                    f"took {query_time:.2f}ms (threshold: {self.slow_query_threshold}ms)"
                )

# Global database monitor instance
db_monitor = DatabasePerformanceMonitor()

class PerformanceDashboard:
    """Performance dashboard and reporting"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        resource_stats = self.collector.get_resource_stats()
        endpoint_stats = self.collector.get_endpoint_stats()
        db_stats = self.collector.get_database_stats()
        
        # Determine health status
        health_score = 100
        issues = []
        
        # Check CPU usage
        if resource_stats.get('avg_cpu_percent', 0) > 80:
            health_score -= 20
            issues.append("High CPU usage")
        
        # Check memory usage
        if resource_stats.get('avg_memory_percent', 0) > 85:
            health_score -= 25
            issues.append("High memory usage")
        
        # Check response times
        if endpoint_stats.get('avg_response_time_ms', 0) > 1000:
            health_score -= 15
            issues.append("Slow API response times")
        
        # Check error rates
        if endpoint_stats.get('error_rate', 0) > 0.05:  # 5%
            health_score -= 20
            issues.append("High error rate")
        
        # Check database performance
        if db_stats.get('slow_queries', 0) > 0:
            health_score -= 10
            issues.append("Slow database queries")
        
        health_status = "excellent" if health_score >= 90 else \
                       "good" if health_score >= 75 else \
                       "warning" if health_score >= 50 else "critical"
        
        return {
            "health_score": max(0, health_score),
            "status": health_status,
            "issues": issues,
            "uptime_seconds": (datetime.now() - self.collector.start_time).total_seconds(),
            "total_requests": self.collector.request_count,
            "error_count": self.collector.error_count,
            "cache_hit_rate": self.collector.cache_hits / (self.collector.cache_hits + self.collector.cache_misses) if (self.collector.cache_hits + self.collector.cache_misses) > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "system_health": self.get_system_health(),
            "resource_metrics": self.collector.get_resource_stats(),
            "api_performance": self.collector.get_endpoint_stats(),
            "database_performance": self.collector.get_database_stats(),
            "top_endpoints": self._get_top_endpoints(),
            "recommendations": self._get_performance_recommendations()
        }
    
    def _get_top_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top endpoints by request count and response time"""
        endpoint_stats = []
        
        with self.collector.lock:
            for endpoint, metrics in self.collector.endpoint_metrics.items():
                if not metrics:
                    continue
                    
                response_times = [m.response_time_ms for m in metrics]
                
                stats = {
                    "endpoint": endpoint,
                    "request_count": len(metrics),
                    "avg_response_time_ms": statistics.mean(response_times),
                    "max_response_time_ms": max(response_times),
                    "error_count": sum(1 for m in metrics if m.status_code >= 400)
                }
                endpoint_stats.append(stats)
        
        # Sort by request count
        endpoint_stats.sort(key=lambda x: x['request_count'], reverse=True)
        
        return endpoint_stats[:limit]
    
    def _get_performance_recommendations(self) -> List[Dict[str, str]]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        resource_stats = self.collector.get_resource_stats()
        endpoint_stats = self.collector.get_endpoint_stats()
        db_stats = self.collector.get_database_stats()
        
        # CPU recommendations
        if resource_stats.get('avg_cpu_percent', 0) > 70:
            recommendations.append({
                "category": "CPU",
                "priority": "HIGH",
                "title": "High CPU Usage",
                "description": "CPU usage is consistently high. Consider optimizing algorithms or scaling horizontally.",
                "action": "Profile CPU-intensive operations and optimize or add caching"
            })
        
        # Memory recommendations
        if resource_stats.get('avg_memory_percent', 0) > 80:
            recommendations.append({
                "category": "Memory",
                "priority": "HIGH", 
                "title": "High Memory Usage",
                "description": "Memory usage is high. Check for memory leaks or large data structures.",
                "action": "Implement memory profiling and optimize data structures"
            })
        
        # API recommendations
        if endpoint_stats.get('avg_response_time_ms', 0) > 500:
            recommendations.append({
                "category": "API",
                "priority": "MEDIUM",
                "title": "Slow API Responses",
                "description": "API response times are slower than optimal.",
                "action": "Implement response caching and optimize database queries"
            })
        
        # Database recommendations
        if db_stats.get('slow_queries', 0) > 5:
            recommendations.append({
                "category": "Database",
                "priority": "HIGH",
                "title": "Slow Database Queries",
                "description": "Multiple slow database queries detected.",
                "action": "Add database indexes and optimize query structure"
            })
        
        # Cache recommendations
        if self.collector.cache_hits / (self.collector.cache_hits + self.collector.cache_misses) < 0.8:
            recommendations.append({
                "category": "Caching",
                "priority": "MEDIUM",
                "title": "Low Cache Hit Rate",
                "description": "Cache hit rate is below optimal level.",
                "action": "Review caching strategy and increase cache TTL for stable data"
            })
        
        return recommendations

# Global dashboard instance
dashboard = PerformanceDashboard(performance_collector)

class PerformanceOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self):
        self.optimization_enabled = True
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(minutes=15)
    
    async def auto_optimize(self):
        """Run automatic optimization routines"""
        if not self.optimization_enabled:
            return
        
        if datetime.now() - self.last_optimization < self.optimization_interval:
            return
        
        try:
            logger.info("Running automatic performance optimization")
            
            # Garbage collection optimization
            import gc
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear old metrics to save memory
            cutoff = datetime.now() - timedelta(hours=1)
            
            with performance_collector.lock:
                # Keep only recent metrics
                performance_collector.metrics = deque(
                    [m for m in performance_collector.metrics if m.timestamp >= cutoff],
                    maxlen=performance_collector.metrics.maxlen
                )
                
                # Clean up old endpoint metrics
                for endpoint, metrics in performance_collector.endpoint_metrics.items():
                    recent_metrics = [m for m in metrics if m.timestamp >= cutoff]
                    performance_collector.endpoint_metrics[endpoint] = deque(
                        recent_metrics, maxlen=metrics.maxlen
                    )
            
            self.last_optimization = datetime.now()
            logger.info("Automatic optimization completed")
            
        except Exception as e:
            logger.error(f"Error during automatic optimization: {e}")

# Global optimizer instance
optimizer = PerformanceOptimizer()

async def start_performance_monitoring():
    """Start background performance monitoring tasks"""
    logger.info("Starting performance monitoring")
    
    async def resource_monitor():
        """Background task for resource monitoring"""
        while True:
            try:
                resource_metric = performance_collector.collect_system_metrics()
                if resource_metric:
                    performance_collector.add_resource_metric(resource_metric)
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def optimization_monitor():
        """Background task for automatic optimization"""
        while True:
            try:
                await optimizer.auto_optimize()
            except Exception as e:
                logger.error(f"Error in optimization monitor: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    # Start monitoring tasks
    asyncio.create_task(resource_monitor())
    asyncio.create_task(optimization_monitor())
    
    logger.info("Performance monitoring started")

def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of current performance metrics"""
    return {
        "system_health": dashboard.get_system_health(),
        "resource_utilization": performance_collector.get_resource_stats(minutes=5),
        "api_performance": performance_collector.get_endpoint_stats(),
        "database_performance": performance_collector.get_database_stats(),
        "cache_performance": {
            "hit_rate": performance_collector.cache_hits / (performance_collector.cache_hits + performance_collector.cache_misses) if (performance_collector.cache_hits + performance_collector.cache_misses) > 0 else 0,
            "total_hits": performance_collector.cache_hits,
            "total_misses": performance_collector.cache_misses
        },
        "uptime_seconds": (datetime.now() - performance_collector.start_time).total_seconds()
    }

# Export key components
__all__ = [
    'PerformanceCollector', 'PerformanceMiddleware', 'PerformanceDashboard',
    'performance_timer', 'performance_context', 'db_monitor',
    'start_performance_monitoring', 'get_performance_summary',
    'performance_collector', 'dashboard', 'optimizer'
]