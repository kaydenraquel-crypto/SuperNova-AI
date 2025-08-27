"""
SuperNova Async Processor
High-performance async processing with connection pooling, task queues, and resource optimization
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Coroutine, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
import threading
import heapq
from enum import Enum

# Database connection pooling
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

# HTTP client pooling
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector, ClientResponseError

# Resource monitoring
import psutil

from .config import settings
from .performance_monitor import performance_collector, performance_timer
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class AsyncTask:
    """Async task with priority and metadata"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        return self.priority.value < other.priority.value

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0

class ConnectionPoolManager:
    """Manage various connection pools"""
    
    def __init__(self):
        self.http_session: Optional[ClientSession] = None
        self.db_engine = None
        self.async_db_session = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize all connection pools"""
        await self._init_http_pool()
        await self._init_database_pool()
        logger.info("Connection pools initialized")
    
    async def _init_http_pool(self):
        """Initialize HTTP connection pool"""
        try:
            # Configure TCP connector with connection pooling
            connector = TCPConnector(
                limit=100,                # Total connection pool size
                limit_per_host=20,        # Connections per host
                ttl_dns_cache=300,        # DNS cache TTL (5 minutes)
                use_dns_cache=True,
                keepalive_timeout=30,     # Keep alive timeout
                enable_cleanup_closed=True,
                force_close=False,
                ssl=False                 # Disable SSL verification for internal APIs
            )
            
            # Configure timeout
            timeout = ClientTimeout(
                total=30,                 # Total timeout
                connect=5,                # Connection timeout
                sock_read=10,             # Socket read timeout
                sock_connect=5            # Socket connect timeout
            )
            
            # Create session
            self.http_session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'SuperNova-AsyncProcessor/1.0',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                },
                raise_for_status=False,
                read_bufsize=8192         # Read buffer size
            )
            
            logger.info("HTTP connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize HTTP pool: {e}")
    
    async def _init_database_pool(self):
        """Initialize async database connection pool"""
        try:
            # Create async engine for database operations
            database_url = settings.DATABASE_URL
            
            # Convert SQLite URL to async if needed
            if database_url.startswith('sqlite:'):
                database_url = database_url.replace('sqlite:', 'sqlite+aiosqlite:')
            
            self.db_engine = create_async_engine(
                database_url,
                pool_size=10,             # Connection pool size
                max_overflow=20,          # Additional connections beyond pool_size
                pool_timeout=30,          # Timeout waiting for connection
                pool_recycle=3600,        # Recycle connections after 1 hour
                pool_pre_ping=True,       # Validate connections before use
                echo=False                # Don't log SQL queries
            )
            
            # Create async session factory
            self.async_db_session = async_sessionmaker(
                self.db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Async database pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
    
    async def get_http_session(self) -> Optional[ClientSession]:
        """Get HTTP session from pool"""
        return self.http_session
    
    @asynccontextmanager
    async def get_db_session(self):
        """Get database session from pool"""
        if not self.async_db_session:
            raise RuntimeError("Database pool not initialized")
        
        async with self.async_db_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
    
    async def close(self):
        """Close all connection pools"""
        if self.http_session:
            await self.http_session.close()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("Connection pools closed")

class ResourceMonitor:
    """Monitor system resources for dynamic scaling"""
    
    def __init__(self):
        self.cpu_threshold = 80.0         # CPU usage threshold (%)
        self.memory_threshold = 85.0      # Memory usage threshold (%)
        self.load_threshold = 0.8         # Task queue load threshold
        self.monitoring_interval = 10     # Monitoring interval (seconds)
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io_percent': self._get_disk_io_percent(),
                'network_io_mbps': self._get_network_io_mbps(),
                'process_count': len(psutil.pids()),
                'thread_count': threading.active_count()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _get_disk_io_percent(self) -> float:
        """Get disk I/O utilization percentage (approximate)"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # This is a simplified calculation
                total_io = disk_io.read_bytes + disk_io.write_bytes
                return min(100.0, total_io / (1024 * 1024 * 100))  # Normalize to percentage
            return 0.0
        except:
            return 0.0
    
    def _get_network_io_mbps(self) -> float:
        """Get network I/O in Mbps (approximate)"""
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                total_bytes = net_io.bytes_sent + net_io.bytes_recv
                return total_bytes / (1024 * 1024)  # Convert to MB
            return 0.0
        except:
            return 0.0
    
    def should_scale_down(self, metrics: Dict[str, float], queue_size: int) -> bool:
        """Determine if system should scale down"""
        return (
            metrics.get('cpu_percent', 100) < 30 and
            metrics.get('memory_percent', 100) < 50 and
            queue_size < 5
        )
    
    def should_scale_up(self, metrics: Dict[str, float], queue_size: int) -> bool:
        """Determine if system should scale up"""
        return (
            metrics.get('cpu_percent', 0) > self.cpu_threshold or
            metrics.get('memory_percent', 0) > self.memory_threshold or
            queue_size > 50
        )

class AsyncTaskProcessor:
    """High-performance async task processor with dynamic scaling"""
    
    def __init__(self, max_workers: int = 50, max_queue_size: int = 10000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.current_workers = 0
        self.min_workers = 2
        
        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(max_queue_size)
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.results: Dict[str, TaskResult] = {}
        self.worker_tasks: List[asyncio.Task] = []
        
        # Resource management
        self.connection_pool = ConnectionPoolManager()
        self.resource_monitor = ResourceMonitor()
        
        # Metrics
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        
        # Control flags
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        # Locks
        self._worker_lock = asyncio.Lock()
        self._result_lock = asyncio.Lock()
    
    async def start(self):
        """Start the async processor"""
        if self.is_running:
            return
        
        logger.info("Starting async task processor")
        
        # Initialize connection pools
        await self.connection_pool.initialize()
        
        # Start initial workers
        await self._scale_workers(self.min_workers)
        
        # Start monitoring and scaling task
        asyncio.create_task(self._monitor_and_scale())
        
        # Start result cleanup task
        asyncio.create_task(self._cleanup_results())
        
        self.is_running = True
        logger.info(f"Async processor started with {self.current_workers} workers")
    
    async def stop(self):
        """Stop the async processor"""
        if not self.is_running:
            return
        
        logger.info("Stopping async task processor")
        
        # Signal shutdown
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close connection pools
        await self.connection_pool.close()
        
        logger.info("Async processor stopped")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit async task for processing"""
        
        if not self.is_running:
            raise RuntimeError("Processor not running")
        
        # Generate task ID if not provided
        if not task_id:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        # Create task
        task = AsyncTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            callback=callback
        )
        
        try:
            # Add to queue (with priority)
            await self.task_queue.put((priority.value, time.time(), task))
            
            # Track active task
            self.active_tasks[task_id] = task
            
            logger.debug(f"Task {task_id} submitted with priority {priority.name}")
            return task_id
            
        except asyncio.QueueFull:
            raise RuntimeError(f"Task queue full (max size: {self.max_queue_size})")
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of completed task"""
        start_time = time.time()
        
        while True:
            async with self._result_lock:
                if task_id in self.results:
                    return self.results[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def _worker(self, worker_id: int):
        """Worker coroutine to process tasks"""
        logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    priority, timestamp, task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                result = await self._execute_task(task)
                
                # Store result
                async with self._result_lock:
                    self.results[task.id] = result
                
                # Remove from active tasks
                self.active_tasks.pop(task.id, None)
                
                # Execute callback if provided
                if task.callback and result.success:
                    try:
                        await task.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for task {task.id}: {e}")
                
                # Update metrics
                self.processed_tasks += 1
                self.total_execution_time += result.execution_time
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: AsyncTask) -> TaskResult:
        """Execute a single task with retry logic"""
        result = TaskResult(task_id=task.id, success=False)
        retry_count = 0
        
        while retry_count <= task.max_retries:
            start_time = time.time()
            
            try:
                # Execute task with timeout
                if asyncio.iscoroutinefunction(task.func):
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            task.func(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    else:
                        task_result = await task.func(*task.args, **task.kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: task.func(*task.args, **task.kwargs)),
                            timeout=task.timeout
                        )
                    else:
                        task_result = await loop.run_in_executor(None, lambda: task.func(*task.args, **task.kwargs))
                
                # Task succeeded
                result.success = True
                result.result = task_result
                result.execution_time = time.time() - start_time
                result.completed_at = datetime.now()
                result.retry_count = retry_count
                
                # Record performance metric
                performance_collector.add_metric({
                    'name': f'async_task.{task.func.__name__}',
                    'value': result.execution_time * 1000,  # Convert to ms
                    'unit': 'ms',
                    'category': 'async_processing',
                    'tags': {'priority': task.priority.name, 'retry_count': str(retry_count)}
                })
                
                break
                
            except asyncio.TimeoutError:
                error_msg = f"Task {task.id} timed out after {task.timeout}s"
                logger.warning(error_msg)
                result.error = error_msg
                self.failed_tasks += 1
                break
                
            except Exception as e:
                error_msg = f"Task {task.id} failed (attempt {retry_count + 1}): {str(e)}"
                logger.error(error_msg)
                
                result.error = str(e)
                result.retry_count = retry_count
                
                if retry_count >= task.max_retries:
                    self.failed_tasks += 1
                    break
                else:
                    # Wait before retry
                    await asyncio.sleep(task.retry_delay * (2 ** retry_count))
                    retry_count += 1
        
        return result
    
    async def _scale_workers(self, target_workers: int):
        """Scale workers to target number"""
        async with self._worker_lock:
            current = self.current_workers
            
            if target_workers > current:
                # Scale up
                new_workers = target_workers - current
                for i in range(new_workers):
                    worker_id = current + i + 1
                    worker_task = asyncio.create_task(self._worker(worker_id))
                    self.worker_tasks.append(worker_task)
                
                self.current_workers = target_workers
                logger.info(f"Scaled up to {target_workers} workers (+{new_workers})")
                
            elif target_workers < current:
                # Scale down
                workers_to_remove = current - target_workers
                for _ in range(workers_to_remove):
                    if self.worker_tasks:
                        worker_task = self.worker_tasks.pop()
                        worker_task.cancel()
                
                self.current_workers = target_workers
                logger.info(f"Scaled down to {target_workers} workers (-{workers_to_remove})")
    
    async def _monitor_and_scale(self):
        """Monitor system resources and scale workers accordingly"""
        logger.info("Starting resource monitoring and auto-scaling")
        
        while self.is_running:
            try:
                # Get system metrics
                metrics = self.resource_monitor.get_system_metrics()
                queue_size = self.task_queue.qsize()
                
                # Determine scaling action
                target_workers = self.current_workers
                
                if self.resource_monitor.should_scale_up(metrics, queue_size):
                    target_workers = min(self.max_workers, self.current_workers + 2)
                elif self.resource_monitor.should_scale_down(metrics, queue_size):
                    target_workers = max(self.min_workers, self.current_workers - 1)
                
                # Scale if needed
                if target_workers != self.current_workers:
                    await self._scale_workers(target_workers)
                
                # Log metrics periodically
                if self.processed_tasks % 100 == 0:
                    avg_execution_time = (
                        self.total_execution_time / self.processed_tasks 
                        if self.processed_tasks > 0 else 0
                    )
                    
                    logger.info(
                        f"Processor stats - Workers: {self.current_workers}, "
                        f"Queue: {queue_size}, Processed: {self.processed_tasks}, "
                        f"Failed: {self.failed_tasks}, Avg time: {avg_execution_time:.3f}s, "
                        f"CPU: {metrics.get('cpu_percent', 0):.1f}%, "
                        f"Memory: {metrics.get('memory_percent', 0):.1f}%"
                    )
                
                await asyncio.sleep(self.resource_monitor.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring and scaling: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_results(self):
        """Clean up old task results"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                async with self._result_lock:
                    expired_tasks = [
                        task_id for task_id, result in self.results.items()
                        if result.completed_at and result.completed_at < cutoff_time
                    ]
                    
                    for task_id in expired_tasks:
                        del self.results[task_id]
                    
                    if expired_tasks:
                        logger.info(f"Cleaned up {len(expired_tasks)} old task results")
                
                # Run cleanup every 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in result cleanup: {e}")
                await asyncio.sleep(300)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'current_workers': self.current_workers,
            'max_workers': self.max_workers,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'processed_tasks': self.processed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': (self.processed_tasks - self.failed_tasks) / self.processed_tasks if self.processed_tasks > 0 else 0,
            'avg_execution_time': self.total_execution_time / self.processed_tasks if self.processed_tasks > 0 else 0,
            'results_cached': len(self.results),
            'is_running': self.is_running
        }

# Global processor instance
async_processor = AsyncTaskProcessor(max_workers=50)

# Convenience functions
async def submit_async_task(
    func: Callable,
    *args,
    priority: TaskPriority = TaskPriority.MEDIUM,
    timeout: Optional[float] = None,
    **kwargs
) -> str:
    """Submit async task for processing"""
    return await async_processor.submit_task(func, *args, priority=priority, timeout=timeout, **kwargs)

async def get_task_result(task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
    """Get result of completed task"""
    return await async_processor.get_task_result(task_id, timeout)

@performance_timer(category="async_processing")
async def process_batch_async(
    items: List[Any],
    processor_func: Callable,
    max_concurrent: int = 10,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> List[Any]:
    """Process batch of items asynchronously"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await processor_func(item)
    
    # Create tasks
    tasks = [process_item(item) for item in items]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

async def initialize_async_processor():
    """Initialize the global async processor"""
    await async_processor.start()
    logger.info("Global async processor initialized")

async def shutdown_async_processor():
    """Shutdown the global async processor"""
    await async_processor.stop()
    logger.info("Global async processor shutdown")

# Export key components
__all__ = [
    'async_processor', 'AsyncTask', 'TaskResult', 'TaskPriority',
    'submit_async_task', 'get_task_result', 'process_batch_async',
    'initialize_async_processor', 'shutdown_async_processor',
    'ConnectionPoolManager', 'AsyncTaskProcessor'
]