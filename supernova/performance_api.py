"""
SuperNova Performance API
Real-time performance metrics and monitoring endpoints
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import time
from pydantic import BaseModel, Field

from .performance_monitor import performance_collector, dashboard, optimizer
from .cache_manager import cache_manager
from .async_processor import async_processor
from .api_optimization import optimization_metrics

# Pydantic models for API responses
class HealthStatus(BaseModel):
    status: str = Field(..., description="Overall health status")
    health_score: int = Field(..., description="Health score (0-100)")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(..., description="Status check timestamp")
    issues: List[str] = Field(default=[], description="Current issues")

class ResourceMetrics(BaseModel):
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage") 
    memory_used_mb: float = Field(..., description="Memory used in MB")
    disk_io_read_mb: float = Field(..., description="Disk read I/O in MB")
    disk_io_write_mb: float = Field(..., description="Disk write I/O in MB")
    network_sent_mb: float = Field(..., description="Network sent in MB")
    network_recv_mb: float = Field(..., description="Network received in MB")
    active_connections: int = Field(..., description="Active network connections")
    thread_count: int = Field(..., description="Thread count")

class EndpointPerformance(BaseModel):
    request_count: int = Field(..., description="Total request count")
    avg_response_time_ms: float = Field(..., description="Average response time in ms")
    min_response_time_ms: float = Field(..., description="Minimum response time in ms")
    max_response_time_ms: float = Field(..., description="Maximum response time in ms")
    p95_response_time_ms: float = Field(..., description="95th percentile response time")
    p99_response_time_ms: float = Field(..., description="99th percentile response time")
    error_rate: float = Field(..., description="Error rate (0.0-1.0)")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0)")

class DatabasePerformance(BaseModel):
    total_queries: int = Field(..., description="Total database queries")
    avg_query_time_ms: float = Field(..., description="Average query time in ms")
    slow_queries: int = Field(..., description="Number of slow queries")
    operation_breakdown: Dict[str, Dict[str, Any]] = Field(..., description="Query breakdown by operation")

class CachePerformance(BaseModel):
    hit_rate: float = Field(..., description="Overall cache hit rate")
    l1_hit_rate: float = Field(..., description="L1 cache hit rate")
    l2_enabled: bool = Field(..., description="L2 cache enabled status")
    l3_enabled: bool = Field(..., description="L3 cache enabled status")
    memory_usage_mb: float = Field(..., description="Cache memory usage in MB")

class AsyncProcessorStats(BaseModel):
    current_workers: int = Field(..., description="Current number of workers")
    queue_size: int = Field(..., description="Task queue size")
    processed_tasks: int = Field(..., description="Total processed tasks")
    success_rate: float = Field(..., description="Task success rate")
    avg_execution_time: float = Field(..., description="Average task execution time")

class PerformanceReport(BaseModel):
    health: HealthStatus
    resources: ResourceMetrics
    api: EndpointPerformance
    database: DatabasePerformance
    cache: CachePerformance
    async_processor: AsyncProcessorStats
    generated_at: datetime
    recommendations: List[Dict[str, str]] = Field(default=[], description="Performance recommendations")

# Create performance API router
performance_api = APIRouter(prefix="/performance", tags=["Performance Monitoring"])

@performance_api.get("/health", response_model=HealthStatus)
async def get_system_health():
    """
    Get overall system health status with key metrics.
    
    Returns comprehensive health check including:
    - Overall health score (0-100)
    - System uptime
    - Current issues and warnings
    - Performance status indicators
    """
    try:
        health_data = dashboard.get_system_health()
        
        return HealthStatus(
            status=health_data.get('status', 'unknown'),
            health_score=health_data.get('health_score', 0),
            uptime_seconds=health_data.get('uptime_seconds', 0),
            timestamp=datetime.fromisoformat(health_data.get('timestamp', datetime.now().isoformat())),
            issues=health_data.get('issues', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

@performance_api.get("/metrics/resources", response_model=ResourceMetrics)
async def get_resource_metrics():
    """
    Get current system resource utilization metrics.
    
    Returns detailed resource usage including:
    - CPU and memory utilization
    - Disk and network I/O
    - Active connections and threads
    """
    try:
        # Collect current resource metrics
        current_metrics = performance_collector.collect_system_metrics()
        
        if not current_metrics:
            raise HTTPException(status_code=503, detail="Unable to collect resource metrics")
        
        return ResourceMetrics(
            cpu_percent=current_metrics.cpu_percent,
            memory_percent=current_metrics.memory_percent,
            memory_used_mb=current_metrics.memory_used_mb,
            disk_io_read_mb=current_metrics.disk_io_read_mb,
            disk_io_write_mb=current_metrics.disk_io_write_mb,
            network_sent_mb=current_metrics.network_sent_mb,
            network_recv_mb=current_metrics.network_recv_mb,
            active_connections=current_metrics.active_connections,
            thread_count=current_metrics.thread_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting resource metrics: {str(e)}")

@performance_api.get("/metrics/api", response_model=EndpointPerformance)
async def get_api_performance(endpoint: Optional[str] = Query(None, description="Specific endpoint to analyze")):
    """
    Get API endpoint performance metrics.
    
    Args:
        endpoint: Optional specific endpoint to analyze (returns aggregate if not specified)
    
    Returns detailed API performance including:
    - Request counts and response times
    - Error rates and percentiles
    - Cache hit rates
    """
    try:
        stats = performance_collector.get_endpoint_stats(endpoint)
        
        if not stats:
            # Return default empty stats
            stats = {
                'request_count': 0,
                'avg_response_time_ms': 0.0,
                'min_response_time_ms': 0.0,
                'max_response_time_ms': 0.0,
                'p95_response_time_ms': 0.0,
                'p99_response_time_ms': 0.0,
                'error_rate': 0.0,
                'cache_hit_rate': 0.0
            }
        
        return EndpointPerformance(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting API performance: {str(e)}")

@performance_api.get("/metrics/database", response_model=DatabasePerformance)
async def get_database_performance():
    """
    Get database performance metrics.
    
    Returns detailed database performance including:
    - Query counts and execution times
    - Slow query detection
    - Operation breakdown by type
    """
    try:
        stats = performance_collector.get_database_stats()
        
        if not stats:
            stats = {
                'total_queries': 0,
                'avg_query_time_ms': 0.0,
                'slow_queries': 0,
                'operation_breakdown': {}
            }
        
        return DatabasePerformance(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database performance: {str(e)}")

@performance_api.get("/metrics/cache", response_model=CachePerformance)
async def get_cache_performance():
    """
    Get cache system performance metrics.
    
    Returns comprehensive cache performance including:
    - Hit rates across all cache levels
    - Memory usage and efficiency
    - Cache level status
    """
    try:
        stats = cache_manager.get_stats()
        
        # Extract key metrics
        combined_stats = stats.get('combined', {})
        l1_stats = stats.get('l1_cache', {})
        
        return CachePerformance(
            hit_rate=combined_stats.get('hit_rate', 0.0),
            l1_hit_rate=l1_stats.hit_rate if hasattr(l1_stats, 'hit_rate') else 0.0,
            l2_enabled=stats.get('l2_enabled', False),
            l3_enabled=stats.get('l3_enabled', False),
            memory_usage_mb=l1_stats.memory_usage_mb if hasattr(l1_stats, 'memory_usage_mb') else 0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache performance: {str(e)}")

@performance_api.get("/metrics/async", response_model=AsyncProcessorStats)
async def get_async_processor_stats():
    """
    Get async task processor performance metrics.
    
    Returns async processor performance including:
    - Worker count and queue size
    - Task success rates and execution times
    - Processing throughput
    """
    try:
        stats = async_processor.get_stats()
        
        return AsyncProcessorStats(
            current_workers=stats.get('current_workers', 0),
            queue_size=stats.get('queue_size', 0),
            processed_tasks=stats.get('processed_tasks', 0),
            success_rate=stats.get('success_rate', 0.0),
            avg_execution_time=stats.get('avg_execution_time', 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting async processor stats: {str(e)}")

@performance_api.get("/report", response_model=PerformanceReport)
async def get_performance_report():
    """
    Get comprehensive performance report with all metrics and recommendations.
    
    Returns complete system performance analysis including:
    - All performance metrics
    - Health status and recommendations
    - Resource utilization analysis
    """
    try:
        # Collect all metrics
        health = await get_system_health()
        resources = await get_resource_metrics()
        api_perf = await get_api_performance()
        db_perf = await get_database_performance()
        cache_perf = await get_cache_performance()
        async_stats = await get_async_processor_stats()
        
        # Get recommendations
        recommendations = dashboard._get_performance_recommendations()
        
        return PerformanceReport(
            health=health,
            resources=resources,
            api=api_perf,
            database=db_perf,
            cache=cache_perf,
            async_processor=async_stats,
            generated_at=datetime.now(),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating performance report: {str(e)}")

@performance_api.get("/metrics/stream")
async def stream_performance_metrics(
    interval: int = Query(5, ge=1, le=60, description="Streaming interval in seconds"),
    duration: int = Query(300, ge=10, le=3600, description="Stream duration in seconds")
):
    """
    Stream real-time performance metrics.
    
    Args:
        interval: How often to send metrics (1-60 seconds)
        duration: How long to stream (10-3600 seconds)
    
    Returns:
        Server-Sent Events stream of performance metrics
    """
    
    async def generate_metrics():
        """Generate streaming metrics"""
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            try:
                # Collect current metrics
                health = dashboard.get_system_health()
                resource_stats = performance_collector.get_resource_stats(minutes=1)
                api_stats = performance_collector.get_endpoint_stats()
                
                # Create streaming data
                stream_data = {
                    "timestamp": datetime.now().isoformat(),
                    "health_score": health.get('health_score', 0),
                    "cpu_percent": resource_stats.get('avg_cpu_percent', 0),
                    "memory_percent": resource_stats.get('avg_memory_percent', 0),
                    "api_response_time": api_stats.get('avg_response_time_ms', 0),
                    "api_error_rate": api_stats.get('error_rate', 0),
                    "cache_hit_rate": performance_collector.cache_hits / (performance_collector.cache_hits + performance_collector.cache_misses) if (performance_collector.cache_hits + performance_collector.cache_misses) > 0 else 0,
                    "active_connections": resource_stats.get('current_memory_mb', 0)
                }
                
                # Send as Server-Sent Event
                yield f"data: {json.dumps(stream_data)}\n\n"
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                error_data = {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
        
        # End stream
        yield f"data: {json.dumps({'stream_ended': True, 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(
        generate_metrics(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )

@performance_api.get("/endpoints/top")
async def get_top_endpoints(
    limit: int = Query(10, ge=1, le=50, description="Number of top endpoints to return"),
    sort_by: str = Query("request_count", description="Sort criteria: request_count, response_time, error_rate")
):
    """
    Get top API endpoints by various performance metrics.
    
    Args:
        limit: Maximum number of endpoints to return
        sort_by: Sorting criteria (request_count, response_time, error_rate)
    
    Returns:
        List of top endpoints with performance metrics
    """
    try:
        top_endpoints = dashboard._get_top_endpoints(limit=limit)
        
        # Sort by requested criteria
        if sort_by == "response_time":
            top_endpoints.sort(key=lambda x: x.get('avg_response_time_ms', 0), reverse=True)
        elif sort_by == "error_rate":
            top_endpoints.sort(key=lambda x: x.get('error_count', 0), reverse=True)
        # Default is request_count (already sorted)
        
        return {
            "endpoints": top_endpoints,
            "sort_by": sort_by,
            "total_endpoints": len(top_endpoints),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting top endpoints: {str(e)}")

@performance_api.post("/optimize/trigger")
async def trigger_optimization(background_tasks: BackgroundTasks):
    """
    Trigger manual performance optimization.
    
    Runs performance optimization routines including:
    - Memory cleanup and garbage collection
    - Cache optimization
    - Resource rebalancing
    """
    try:
        # Add optimization to background tasks
        background_tasks.add_task(optimizer.auto_optimize)
        
        return {
            "message": "Performance optimization triggered",
            "status": "queued",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering optimization: {str(e)}")

@performance_api.get("/alerts/thresholds")
async def get_performance_thresholds():
    """
    Get current performance alert thresholds.
    
    Returns:
        Dictionary of performance thresholds and alert levels
    """
    return {
        "cpu_warning": 70.0,
        "cpu_critical": 85.0,
        "memory_warning": 80.0,
        "memory_critical": 90.0,
        "response_time_warning": 1000.0,  # ms
        "response_time_critical": 5000.0,  # ms
        "error_rate_warning": 0.05,  # 5%
        "error_rate_critical": 0.1,  # 10%
        "cache_hit_rate_warning": 0.7,  # 70%
        "cache_hit_rate_critical": 0.5,  # 50%
        "disk_usage_warning": 80.0,
        "disk_usage_critical": 95.0
    }

@performance_api.get("/history/summary")
async def get_performance_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    interval: str = Query("1h", description="Aggregation interval: 5m, 15m, 1h, 6h, 1d")
):
    """
    Get historical performance data summary.
    
    Args:
        hours: Number of hours of history to retrieve (1-168)
        interval: Data aggregation interval
    
    Returns:
        Historical performance trends and statistics
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get historical data (simplified - would typically query TimescaleDB)
        resource_stats = performance_collector.get_resource_stats(minutes=hours * 60)
        api_stats = performance_collector.get_endpoint_stats()
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": hours,
                "interval": interval
            },
            "summary": {
                "avg_cpu_percent": resource_stats.get('avg_cpu_percent', 0),
                "max_cpu_percent": resource_stats.get('max_cpu_percent', 0),
                "avg_memory_percent": resource_stats.get('avg_memory_percent', 0),
                "max_memory_percent": resource_stats.get('max_memory_percent', 0),
                "avg_response_time_ms": api_stats.get('avg_response_time_ms', 0),
                "total_requests": api_stats.get('request_count', 0),
                "error_rate": api_stats.get('error_rate', 0)
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance history: {str(e)}")

@performance_api.get("/debug/stats")
async def get_debug_stats():
    """
    Get detailed debug statistics for troubleshooting.
    
    Returns:
        Internal performance statistics and debug information
    """
    try:
        return {
            "performance_collector": {
                "metrics_count": len(performance_collector.metrics),
                "endpoint_metrics": len(performance_collector.endpoint_metrics),
                "resource_metrics": len(performance_collector.resource_metrics),
                "database_metrics": len(performance_collector.database_metrics),
                "start_time": performance_collector.start_time.isoformat()
            },
            "cache_manager": cache_manager.get_stats(),
            "async_processor": async_processor.get_stats(),
            "optimization_metrics": optimization_metrics.get_stats(),
            "system_info": {
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "platform": __import__('platform').platform(),
                "processor": __import__('platform').processor()
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting debug stats: {str(e)}")

# Export the router
__all__ = ['performance_api']