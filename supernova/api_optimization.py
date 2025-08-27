"""
SuperNova API Optimization
Advanced API performance optimization with compression, async processing, and response optimization
"""

import asyncio
import gzip
import json
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# Data processing
import pandas as pd
import numpy as np
from pydantic import BaseModel

from .config import settings
from .cache_manager import cache_manager, cached, CacheKeyGenerator
from .performance_monitor import performance_collector, performance_timer

logger = logging.getLogger(__name__)

class CompressionMiddleware(BaseHTTPMiddleware):
    """Advanced compression middleware with dynamic compression levels"""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 500, compression_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Skip compression for certain content types
        if hasattr(response, 'media_type') and response.media_type:
            skip_types = {'image/', 'video/', 'audio/', 'application/octet-stream'}
            if any(response.media_type.startswith(skip_type) for skip_type in skip_types):
                return response
        
        # Check if client accepts compression
        accept_encoding = request.headers.get('accept-encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
        
        # Get response content
        if hasattr(response, 'body') and len(response.body) > self.minimum_size:
            try:
                # Compress content
                compressed_content = gzip.compress(
                    response.body, 
                    compresslevel=self.compression_level
                )
                
                # Update response
                response.body = compressed_content
                response.headers['content-encoding'] = 'gzip'
                response.headers['content-length'] = str(len(compressed_content))
                
                # Add compression ratio header for monitoring
                original_size = len(response.body)
                compression_ratio = len(compressed_content) / original_size
                response.headers['X-Compression-Ratio'] = f"{compression_ratio:.3f}"
                
            except Exception as e:
                logger.error(f"Compression error: {e}")
        
        return response

class ResponseOptimizationMiddleware(BaseHTTPMiddleware):
    """Optimize API responses with caching headers, pagination, and field filtering"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add caching headers for GET requests
        response = await call_next(request)
        
        if request.method == "GET":
            # Add cache headers
            cache_duration = self._get_cache_duration(request.url.path)
            if cache_duration > 0:
                response.headers['Cache-Control'] = f'public, max-age={cache_duration}'
                response.headers['ETag'] = self._generate_etag(request.url.path, str(request.query_params))
        
        # Add performance headers
        response_time = (time.time() - start_time) * 1000
        response.headers['X-Response-Time'] = f"{response_time:.2f}ms"
        response.headers['X-Server-Time'] = datetime.now().isoformat()
        
        return response
    
    def _get_cache_duration(self, path: str) -> int:
        """Get appropriate cache duration for different endpoints"""
        cache_rules = {
            '/sentiment/': 300,      # 5 minutes
            '/backtest/': 3600,      # 1 hour
            '/market_data/': 60,     # 1 minute
            '/optimization/': 1800,  # 30 minutes
            '/health': 30,           # 30 seconds
            '/metrics': 60           # 1 minute
        }
        
        for endpoint, duration in cache_rules.items():
            if endpoint in path:
                return duration
        
        return 0  # No caching by default
    
    def _generate_etag(self, path: str, params: str) -> str:
        """Generate ETag for response caching"""
        import hashlib
        content = f"{path}:{params}:{int(time.time() // 60)}"  # Change every minute
        return f'"{hashlib.md5(content.encode()).hexdigest()}"'

class AsyncResponseManager:
    """Manage asynchronous response processing for heavy operations"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.background_tasks: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def start_background_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Start a background task and return task ID"""
        with self.lock:
            self.background_tasks[task_id] = {
                'status': 'running',
                'started_at': datetime.now(),
                'result': None,
                'error': None
            }
        
        def _execute_task():
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    self.background_tasks[task_id].update({
                        'status': 'completed',
                        'completed_at': datetime.now(),
                        'result': result
                    })
            except Exception as e:
                logger.error(f"Background task {task_id} failed: {e}")
                with self.lock:
                    self.background_tasks[task_id].update({
                        'status': 'failed',
                        'completed_at': datetime.now(),
                        'error': str(e)
                    })
        
        self.executor.submit(_execute_task)
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of background task"""
        with self.lock:
            return self.background_tasks.get(task_id)
    
    def cleanup_old_tasks(self, hours: int = 24):
        """Clean up tasks older than specified hours"""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        
        with self.lock:
            to_remove = []
            for task_id, task_info in self.background_tasks.items():
                if task_info.get('completed_at') and task_info['completed_at'] < cutoff:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.background_tasks[task_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old background tasks")

# Global response manager
response_manager = AsyncResponseManager()

class PaginationHelper:
    """Helper for efficient pagination"""
    
    @staticmethod
    def paginate_results(
        data: List[Any], 
        page: int = 1, 
        page_size: int = 50, 
        max_page_size: int = 1000
    ) -> Dict[str, Any]:
        """Paginate results with metadata"""
        
        # Validate parameters
        page = max(1, page)
        page_size = min(max_page_size, max(1, page_size))
        
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return {
            'data': data[start_idx:end_idx],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_items': total_items,
                'total_pages': total_pages,
                'has_previous': page > 1,
                'has_next': page < total_pages,
                'previous_page': page - 1 if page > 1 else None,
                'next_page': page + 1 if page < total_pages else None
            }
        }
    
    @staticmethod
    def cursor_paginate(
        data: List[Dict], 
        cursor_field: str = 'id',
        cursor_value: Optional[str] = None,
        limit: int = 50,
        sort_desc: bool = True
    ) -> Dict[str, Any]:
        """Cursor-based pagination for better performance"""
        
        if cursor_value is not None:
            # Filter data based on cursor
            if sort_desc:
                data = [item for item in data if str(item[cursor_field]) < cursor_value]
            else:
                data = [item for item in data if str(item[cursor_field]) > cursor_value]
        
        # Sort data
        data = sorted(data, key=lambda x: x[cursor_field], reverse=sort_desc)
        
        # Apply limit
        page_data = data[:limit]
        
        # Generate next cursor
        next_cursor = None
        if len(page_data) == limit and len(data) > limit:
            next_cursor = str(page_data[-1][cursor_field])
        
        return {
            'data': page_data,
            'cursor': {
                'next_cursor': next_cursor,
                'has_more': next_cursor is not None
            }
        }

class ResponseFormatter:
    """Format API responses for optimal performance and usability"""
    
    @staticmethod
    def format_numerical_data(data: Dict[str, Any], precision: int = 4) -> Dict[str, Any]:
        """Format numerical data to reduce response size"""
        
        def round_value(value):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    return round(value, precision)
            elif isinstance(value, dict):
                return {k: round_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [round_value(item) for item in value]
            return value
        
        return round_value(data)
    
    @staticmethod
    def filter_fields(data: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Filter response fields to reduce payload size"""
        if not fields:
            return data
        
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if key in fields}
        elif isinstance(data, list):
            return [ResponseFormatter.filter_fields(item, fields) for item in data]
        else:
            return data
    
    @staticmethod
    def compress_time_series(
        data: List[Dict], 
        timestamp_field: str = 'timestamp',
        compression_ratio: float = 0.5
    ) -> List[Dict]:
        """Compress time series data by removing intermediate points"""
        if not data or len(data) <= 2:
            return data
        
        # Calculate step size for compression
        step = max(1, int(len(data) * (1 - compression_ratio)))
        
        # Always include first and last points
        compressed = [data[0]]
        
        # Add intermediate points based on step
        for i in range(step, len(data) - step, step):
            compressed.append(data[i])
        
        # Always include last point
        if len(data) > 1:
            compressed.append(data[-1])
        
        return compressed

class StreamingResponseHelper:
    """Helper for streaming large responses"""
    
    @staticmethod
    async def stream_json_array(data_generator, chunk_size: int = 100):
        """Stream JSON array data in chunks"""
        
        yield '{"data": ['
        
        first_chunk = True
        chunk = []
        
        async for item in data_generator:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                if not first_chunk:
                    yield ','
                else:
                    first_chunk = False
                
                chunk_json = json.dumps(chunk)[1:-1]  # Remove array brackets
                yield chunk_json
                chunk = []
        
        # Send remaining items
        if chunk:
            if not first_chunk:
                yield ','
            chunk_json = json.dumps(chunk)[1:-1]
            yield chunk_json
        
        yield '], "streaming": true}'
    
    @staticmethod
    async def stream_csv_data(data_generator, headers: List[str]):
        """Stream CSV data"""
        
        # Send headers
        yield ','.join(headers) + '\n'
        
        # Send data rows
        async for row in data_generator:
            if isinstance(row, dict):
                csv_row = ','.join(str(row.get(header, '')) for header in headers)
            else:
                csv_row = ','.join(str(value) for value in row)
            
            yield csv_row + '\n'

def optimize_json_response(
    data: Any,
    fields: Optional[List[str]] = None,
    numerical_precision: int = 4,
    compress_arrays: bool = False
) -> Dict[str, Any]:
    """Optimize JSON response for performance"""
    
    # Filter fields if specified
    if fields and isinstance(data, (dict, list)):
        data = ResponseFormatter.filter_fields(data, fields)
    
    # Format numerical data
    if isinstance(data, dict):
        data = ResponseFormatter.format_numerical_data(data, numerical_precision)
    
    # Compress time series arrays
    if compress_arrays and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 100:
                if all(isinstance(item, dict) for item in value):
                    data[key] = ResponseFormatter.compress_time_series(value)
    
    return data

@cached(ttl=300, key_prefix="api_response")
async def get_cached_response(cache_key: str, data_func: Callable, *args, **kwargs):
    """Get cached API response"""
    return await data_func(*args, **kwargs)

class BatchRequestHandler:
    """Handle batch requests efficiently"""
    
    @staticmethod
    async def process_batch(requests: List[Dict], processor_func: Callable, max_concurrent: int = 10):
        """Process batch of requests concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request_data):
            async with semaphore:
                try:
                    return await processor_func(request_data)
                except Exception as e:
                    return {'error': str(e), 'request': request_data}
        
        # Process all requests concurrently
        tasks = [process_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

class APIOptimizationMetrics:
    """Track API optimization metrics"""
    
    def __init__(self):
        self.compression_savings = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_requests = 0
        self.streaming_responses = 0
        self.field_filtering_usage = 0
        
    def record_compression(self, original_size: int, compressed_size: int):
        """Record compression metrics"""
        self.compression_savings += original_size - compressed_size
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0
        
        return {
            'compression_savings_bytes': self.compression_savings,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'batch_requests': self.batch_requests,
            'streaming_responses': self.streaming_responses,
            'field_filtering_usage': self.field_filtering_usage
        }

# Global metrics instance
optimization_metrics = APIOptimizationMetrics()

def setup_api_optimizations(app: FastAPI):
    """Setup all API optimizations"""
    
    # Add compression middleware
    app.add_middleware(CompressionMiddleware, minimum_size=500, compression_level=6)
    
    # Add response optimization middleware
    app.add_middleware(ResponseOptimizationMiddleware)
    
    # Add CORS with optimization
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600  # Cache preflight requests for 1 hour
    )
    
    logger.info("API optimizations configured")

# Optimized endpoint decorators
def optimized_endpoint(
    cache_ttl: int = 300,
    enable_compression: bool = True,
    enable_field_filtering: bool = True,
    max_page_size: int = 1000
):
    """Decorator for optimized API endpoints"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Extract request parameters
                request = kwargs.get('request')
                fields = None
                
                if request and enable_field_filtering:
                    fields = request.query_params.get('fields')
                    if fields:
                        fields = [f.strip() for f in fields.split(',')]
                        optimization_metrics.field_filtering_usage += 1
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Optimize response
                if isinstance(result, dict):
                    result = optimize_json_response(
                        result,
                        fields=fields,
                        compress_arrays=True
                    )
                
                # Record performance
                execution_time = (time.time() - start_time) * 1000
                performance_collector.add_metric({
                    'name': f'optimized_endpoint.{func.__name__}',
                    'value': execution_time,
                    'unit': 'ms',
                    'category': 'api_optimization'
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimized endpoint {func.__name__}: {e}")
                raise
        
        return wrapper
    
    return decorator

async def background_optimization_tasks():
    """Run background optimization tasks"""
    
    while True:
        try:
            # Clean up old background tasks
            response_manager.cleanup_old_tasks(hours=24)
            
            # Log optimization metrics
            stats = optimization_metrics.get_stats()
            logger.info(f"API Optimization Stats: {stats}")
            
            # Wait 1 hour before next cycle
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in background optimization tasks: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

# Export key components
__all__ = [
    'setup_api_optimizations', 'optimized_endpoint', 'optimize_json_response',
    'PaginationHelper', 'ResponseFormatter', 'StreamingResponseHelper',
    'BatchRequestHandler', 'response_manager', 'optimization_metrics'
]