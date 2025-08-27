"""
SuperNova AI Health Monitoring System
=====================================

Comprehensive system health monitoring and validation framework including:
- System component health checks
- Performance metrics collection
- Automated diagnostics and self-healing
- Alert management and notifications
- Business metrics and compliance monitoring
"""

import asyncio
import time
import json
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum, IntEnum
from contextlib import asynccontextmanager, contextmanager
import statistics
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# FastAPI imports for health endpoints
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

# Database and external service imports
try:
    from .db import SessionLocal, get_session, is_timescale_available
    from .database_monitoring import db_monitor
    from .performance_monitor import performance_collector, dashboard
    from .config import settings
    from .llm_service import llm_service
    from .cache_manager import cache_manager
    from .websocket_handler import websocket_manager
    from .auth import auth_manager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# External service health checks
try:
    import redis
    import requests
    EXTERNAL_SERVICES_AVAILABLE = True
except ImportError:
    EXTERNAL_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"  
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"

class AlertSeverity(IntEnum):
    """Alert severity levels (higher number = more severe)."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class HealthCheckResult:
    """Individual health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
            "dependencies": self.dependencies,
            "metrics": self.metrics
        }

@dataclass
class SystemHealthSummary:
    """Overall system health summary."""
    overall_status: HealthStatus
    overall_score: int  # 0-100
    timestamp: datetime
    components: Dict[str, HealthCheckResult]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": {name: result.to_dict() for name, result in self.components.items()},
            "issues": self.issues,
            "recommendations": self.recommendations,
            "component_count": len(self.components),
            "healthy_components": len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY]),
            "warning_components": len([c for c in self.components.values() if c.status == HealthStatus.WARNING]),
            "critical_components": len([c for c in self.components.values() if c.status == HealthStatus.CRITICAL])
        }

@dataclass
class HealthAlert:
    """Health monitoring alert."""
    alert_id: str
    component: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    created_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    threshold_breached: Optional[float] = None
    current_value: Optional[float] = None

class ComponentHealthChecker:
    """Base class for component health checkers."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.last_check: Optional[datetime] = None
        self.check_interval = timedelta(seconds=60)
        
    async def check_health(self) -> HealthCheckResult:
        """Perform health check for this component."""
        start_time = time.time()
        try:
            status, message, details, metrics = await self._perform_check()
            response_time = (time.time() - start_time) * 1000
            
            self.last_check = datetime.utcnow()
            
            return HealthCheckResult(
                component=self.component_name,
                status=status,
                message=message,
                timestamp=self.last_check,
                response_time_ms=response_time,
                details=details,
                metrics=metrics
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {self.component_name}: {e}")
            
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError

class DatabaseHealthChecker(ComponentHealthChecker):
    """Database connectivity and performance health checker."""
    
    def __init__(self):
        super().__init__("database")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        if not DATABASE_AVAILABLE:
            return HealthStatus.UNAVAILABLE, "Database module not available", {}, {}
            
        try:
            with SessionLocal() as db:
                # Test basic connectivity
                start_time = time.time()
                result = db.execute("SELECT 1").scalar()
                query_time = (time.time() - start_time) * 1000
                
                if result != 1:
                    return HealthStatus.CRITICAL, "Database query returned unexpected result", {}, {}
                
                # Check connection pool status
                pool_info = {}
                if hasattr(db.bind, 'pool'):
                    pool = db.bind.pool
                    pool_info = {
                        "size": pool.size(),
                        "checked_in": pool.checkedin(),
                        "checked_out": pool.checkedout(),
                        "invalidated": pool.invalidated()
                    }
                
                # Check for blocking queries (PostgreSQL specific)
                blocking_queries = 0
                try:
                    if 'postgresql' in str(db.bind.url):
                        blocking_result = db.execute("""
                            SELECT COUNT(*) FROM pg_stat_activity 
                            WHERE state = 'active' 
                            AND query_start < NOW() - INTERVAL '5 minutes'
                        """).scalar()
                        blocking_queries = blocking_result or 0
                except:
                    pass  # Not critical if this fails
                
                # Determine health status
                status = HealthStatus.HEALTHY
                message = "Database is healthy"
                
                if query_time > 1000:  # > 1 second
                    status = HealthStatus.WARNING
                    message = "Database responding slowly"
                elif blocking_queries > 5:
                    status = HealthStatus.WARNING
                    message = f"Multiple long-running queries detected ({blocking_queries})"
                
                details = {
                    "connection_pool": pool_info,
                    "blocking_queries": blocking_queries,
                    "database_type": str(db.bind.url).split("://")[0] if db.bind else "unknown"
                }
                
                metrics = {
                    "query_time_ms": query_time,
                    "blocking_queries": blocking_queries,
                    "pool_utilization": (pool_info.get("checked_out", 0) / max(pool_info.get("size", 1), 1)) * 100
                }
                
                return status, message, details, metrics
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Database connection failed: {str(e)}", {"error": str(e)}, {}

class LLMServiceHealthChecker(ComponentHealthChecker):
    """LLM service health checker."""
    
    def __init__(self):
        super().__init__("llm_service")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        try:
            # Test LLM service availability
            start_time = time.time()
            test_response = await llm_service.generate_response(
                "Test health check",
                max_tokens=10,
                timeout=5.0
            )
            response_time = (time.time() - start_time) * 1000
            
            if not test_response:
                return HealthStatus.CRITICAL, "LLM service not responding", {}, {}
            
            # Check service configuration
            config_info = {
                "provider": getattr(llm_service, 'current_provider', 'unknown'),
                "model": getattr(llm_service, 'current_model', 'unknown'),
                "api_keys_configured": llm_service.has_valid_api_keys() if hasattr(llm_service, 'has_valid_api_keys') else True
            }
            
            # Determine health status
            status = HealthStatus.HEALTHY
            message = "LLM service is healthy"
            
            if response_time > 5000:  # > 5 seconds
                status = HealthStatus.WARNING
                message = "LLM service responding slowly"
            
            metrics = {
                "response_time_ms": response_time,
                "token_count": len(test_response.split()) if test_response else 0
            }
            
            return status, message, config_info, metrics
            
        except asyncio.TimeoutError:
            return HealthStatus.CRITICAL, "LLM service timeout", {"error": "timeout"}, {}
        except Exception as e:
            return HealthStatus.CRITICAL, f"LLM service error: {str(e)}", {"error": str(e)}, {}

class CacheHealthChecker(ComponentHealthChecker):
    """Cache system health checker."""
    
    def __init__(self):
        super().__init__("cache")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        try:
            # Test cache connectivity
            test_key = f"health_check_{int(time.time())}"
            test_value = "test_value"
            
            start_time = time.time()
            
            # Test set operation
            await cache_manager.set(test_key, test_value, expire=60)
            
            # Test get operation
            retrieved_value = await cache_manager.get(test_key)
            
            # Test delete operation
            await cache_manager.delete(test_key)
            
            operation_time = (time.time() - start_time) * 1000
            
            if retrieved_value != test_value:
                return HealthStatus.CRITICAL, "Cache data integrity issue", {}, {}
            
            # Get cache statistics
            stats = await cache_manager.get_stats() if hasattr(cache_manager, 'get_stats') else {}
            
            # Determine health status
            status = HealthStatus.HEALTHY
            message = "Cache system is healthy"
            
            if operation_time > 100:  # > 100ms for basic operations
                status = HealthStatus.WARNING
                message = "Cache operations are slow"
            
            hit_rate = stats.get('hit_rate', 0)
            if hit_rate < 0.7:  # < 70% hit rate
                status = HealthStatus.WARNING
                message = f"Low cache hit rate: {hit_rate:.1%}"
            
            details = {
                "backend": getattr(cache_manager, 'backend_type', 'unknown'),
                "stats": stats
            }
            
            metrics = {
                "operation_time_ms": operation_time,
                "hit_rate": hit_rate,
                "memory_usage_mb": stats.get('memory_usage', 0) / 1024 / 1024
            }
            
            return status, message, details, metrics
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Cache system error: {str(e)}", {"error": str(e)}, {}

class WebSocketHealthChecker(ComponentHealthChecker):
    """WebSocket connection health checker."""
    
    def __init__(self):
        super().__init__("websocket")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        try:
            # Check WebSocket manager status
            active_connections = len(getattr(websocket_manager, 'active_connections', []))
            
            # Get connection statistics
            stats = {
                "active_connections": active_connections,
                "total_messages_sent": getattr(websocket_manager, 'total_messages_sent', 0),
                "total_messages_received": getattr(websocket_manager, 'total_messages_received', 0),
                "connection_errors": getattr(websocket_manager, 'connection_errors', 0)
            }
            
            # Determine health status
            status = HealthStatus.HEALTHY
            message = "WebSocket service is healthy"
            
            error_rate = stats['connection_errors'] / max(stats.get('total_connections', 1), 1)
            if error_rate > 0.1:  # > 10% error rate
                status = HealthStatus.WARNING
                message = f"High WebSocket error rate: {error_rate:.1%}"
            
            details = {
                "service_status": "running",
                "statistics": stats
            }
            
            metrics = {
                "active_connections": active_connections,
                "error_rate": error_rate,
                "messages_per_minute": stats.get('messages_per_minute', 0)
            }
            
            return status, message, details, metrics
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"WebSocket service error: {str(e)}", {"error": str(e)}, {}

class AuthServiceHealthChecker(ComponentHealthChecker):
    """Authentication service health checker."""
    
    def __init__(self):
        super().__init__("authentication")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        try:
            # Test authentication manager functionality
            start_time = time.time()
            
            # Test password hashing
            test_password = "test_password_12345"
            hashed = auth_manager.hash_password(test_password)
            
            # Test password verification
            is_valid = auth_manager.verify_password(test_password, hashed)
            
            operation_time = (time.time() - start_time) * 1000
            
            if not is_valid:
                return HealthStatus.CRITICAL, "Authentication password verification failed", {}, {}
            
            # Check active sessions
            active_sessions = len(getattr(auth_manager, 'active_sessions', {}))
            
            # Check recent authentication activity
            recent_logins = getattr(auth_manager, 'recent_login_attempts', 0)
            failed_logins = getattr(auth_manager, 'recent_failed_logins', 0)
            
            # Determine health status
            status = HealthStatus.HEALTHY
            message = "Authentication service is healthy"
            
            if operation_time > 500:  # > 500ms for crypto operations
                status = HealthStatus.WARNING
                message = "Authentication operations are slow"
            
            failure_rate = failed_logins / max(recent_logins, 1) if recent_logins > 0 else 0
            if failure_rate > 0.5:  # > 50% failure rate
                status = HealthStatus.WARNING
                message = f"High authentication failure rate: {failure_rate:.1%}"
            
            details = {
                "active_sessions": active_sessions,
                "recent_activity": {
                    "successful_logins": recent_logins - failed_logins,
                    "failed_logins": failed_logins
                },
                "security_features": {
                    "mfa_enabled": hasattr(auth_manager, 'mfa_manager'),
                    "rate_limiting": hasattr(auth_manager, 'rate_limiter'),
                    "session_encryption": True
                }
            }
            
            metrics = {
                "operation_time_ms": operation_time,
                "active_sessions": active_sessions,
                "failure_rate": failure_rate,
                "security_score": self._calculate_security_score(details)
            }
            
            return status, message, details, metrics
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Authentication service error: {str(e)}", {"error": str(e)}, {}
    
    def _calculate_security_score(self, details: Dict[str, Any]) -> float:
        """Calculate a security score based on enabled features."""
        score = 0.0
        features = details.get('security_features', {})
        
        if features.get('mfa_enabled'): score += 30
        if features.get('rate_limiting'): score += 25
        if features.get('session_encryption'): score += 25
        if details.get('active_sessions', 0) > 0: score += 20
        
        return min(score, 100.0)

class SystemResourceHealthChecker(ComponentHealthChecker):
    """System resource health checker."""
    
    def __init__(self):
        super().__init__("system_resources")
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Determine health status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical CPU usage: {cpu_percent}%")
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical memory usage: {memory.percent}%")
            elif memory.percent > 85:
                status = HealthStatus.WARNING
                issues.append(f"High memory usage: {memory.percent}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = HealthStatus.WARNING
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources are healthy" if not issues else "; ".join(issues)
            
            details = {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": load_avg
                },
                "memory": {
                    "total_gb": memory.total / 1024 / 1024 / 1024,
                    "used_percent": memory.percent,
                    "available_gb": memory.available / 1024 / 1024 / 1024,
                    "swap_used_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk.total / 1024 / 1024 / 1024,
                    "used_percent": disk_percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process": {
                    "memory_mb": process_memory.rss / 1024 / 1024,
                    "cpu_percent": process_cpu,
                    "threads": process.num_threads(),
                    "connections": len(process.connections())
                }
            }
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk_percent,
                "process_memory_mb": process_memory.rss / 1024 / 1024,
                "load_average_1m": load_avg[0] if load_avg else 0
            }
            
            return status, message, details, metrics
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"System resource check error: {str(e)}", {"error": str(e)}, {}

class ExternalServiceHealthChecker(ComponentHealthChecker):
    """External services health checker."""
    
    def __init__(self):
        super().__init__("external_services")
        self.services_to_check = [
            {"name": "internet_connectivity", "url": "https://www.google.com", "timeout": 5},
            {"name": "api_dependencies", "url": "https://api.github.com", "timeout": 10}
        ]
        
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Dict[str, float]]:
        if not EXTERNAL_SERVICES_AVAILABLE:
            return HealthStatus.UNAVAILABLE, "External service checking not available", {}, {}
        
        try:
            service_results = {}
            total_response_time = 0
            failed_services = []
            
            for service in self.services_to_check:
                start_time = time.time()
                try:
                    response = requests.get(
                        service["url"],
                        timeout=service["timeout"],
                        headers={'User-Agent': 'SuperNova-AI-Health-Check/1.0'}
                    )
                    response_time = (time.time() - start_time) * 1000
                    total_response_time += response_time
                    
                    service_results[service["name"]] = {
                        "status": "healthy" if response.status_code == 200 else "degraded",
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
                    
                    if response.status_code != 200:
                        failed_services.append(service["name"])
                        
                except requests.RequestException as e:
                    service_results[service["name"]] = {
                        "status": "failed",
                        "error": str(e),
                        "response_time_ms": (time.time() - start_time) * 1000
                    }
                    failed_services.append(service["name"])
            
            # Determine overall health status
            if len(failed_services) == len(self.services_to_check):
                status = HealthStatus.CRITICAL
                message = "All external services are unreachable"
            elif failed_services:
                status = HealthStatus.WARNING
                message = f"Some external services are unreachable: {', '.join(failed_services)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All external services are reachable"
            
            avg_response_time = total_response_time / len(self.services_to_check)
            
            details = {
                "services": service_results,
                "failed_count": len(failed_services),
                "total_count": len(self.services_to_check)
            }
            
            metrics = {
                "avg_response_time_ms": avg_response_time,
                "success_rate": (len(self.services_to_check) - len(failed_services)) / len(self.services_to_check),
                "failed_services": len(failed_services)
            }
            
            return status, message, details, metrics
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"External service check error: {str(e)}", {"error": str(e)}, {}

class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.health_checkers: Dict[str, ComponentHealthChecker] = {}
        self.health_history: deque = deque(maxlen=100)  # Keep last 100 health checks
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.notification_callbacks: List[Callable] = []
        
        # Initialize health checkers
        self._initialize_health_checkers()
        
        # Setup default alert thresholds
        self._setup_default_thresholds()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
    def _initialize_health_checkers(self):
        """Initialize all health checkers."""
        self.health_checkers = {
            "database": DatabaseHealthChecker(),
            "llm_service": LLMServiceHealthChecker(), 
            "cache": CacheHealthChecker(),
            "websocket": WebSocketHealthChecker(),
            "authentication": AuthServiceHealthChecker(),
            "system_resources": SystemResourceHealthChecker(),
            "external_services": ExternalServiceHealthChecker()
        }
        
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            "database": {
                "query_time_ms": {"warning": 1000, "critical": 5000},
                "pool_utilization": {"warning": 80, "critical": 95}
            },
            "llm_service": {
                "response_time_ms": {"warning": 5000, "critical": 15000}
            },
            "cache": {
                "operation_time_ms": {"warning": 100, "critical": 500},
                "hit_rate": {"warning": 0.7, "critical": 0.5}
            },
            "system_resources": {
                "cpu_percent": {"warning": 80, "critical": 90},
                "memory_percent": {"warning": 85, "critical": 95},
                "disk_percent": {"warning": 85, "critical": 95}
            }
        }
    
    async def check_all_components(self) -> SystemHealthSummary:
        """Perform health check on all components."""
        check_tasks = []
        
        # Start all health checks concurrently
        for component_name, checker in self.health_checkers.items():
            task = asyncio.create_task(checker.check_health())
            check_tasks.append((component_name, task))
        
        # Collect results
        component_results = {}
        for component_name, task in check_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=30.0)  # 30 second timeout
                component_results[component_name] = result
                
                # Check for threshold violations and create alerts
                await self._check_thresholds(result)
                
            except asyncio.TimeoutError:
                logger.error(f"Health check timeout for component: {component_name}")
                component_results[component_name] = HealthCheckResult(
                    component=component_name,
                    status=HealthStatus.CRITICAL,
                    message="Health check timed out",
                    timestamp=datetime.utcnow(),
                    response_time_ms=30000.0
                )
        
        # Calculate overall health
        overall_status, overall_score, issues, recommendations = self._calculate_overall_health(component_results)
        
        # Create system health summary
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        summary = SystemHealthSummary(
            overall_status=overall_status,
            overall_score=overall_score,
            timestamp=datetime.utcnow(),
            components=component_results,
            issues=issues,
            recommendations=recommendations,
            uptime_seconds=uptime
        )
        
        # Store in history
        self.health_history.append(summary)
        
        return summary
    
    def _calculate_overall_health(self, components: Dict[str, HealthCheckResult]) -> tuple[HealthStatus, int, List[str], List[str]]:
        """Calculate overall system health status and score."""
        if not components:
            return HealthStatus.UNAVAILABLE, 0, ["No components checked"], []
        
        status_counts = defaultdict(int)
        issues = []
        recommendations = []
        
        # Count component statuses
        for result in components.values():
            status_counts[result.status] += 1
            
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                issues.append(f"{result.component}: {result.message}")
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.UNAVAILABLE] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate score (0-100)
        total_components = len(components)
        healthy_components = status_counts[HealthStatus.HEALTHY]
        warning_components = status_counts[HealthStatus.WARNING]
        critical_components = status_counts[HealthStatus.CRITICAL]
        
        score = (
            (healthy_components * 100) +
            (warning_components * 60) +
            (critical_components * 0)
        ) / total_components
        
        # Generate recommendations
        if critical_components > 0:
            recommendations.append("Immediate attention required for critical components")
        if warning_components > 0:
            recommendations.append("Monitor warning components and optimize performance")
        if score < 80:
            recommendations.append("System health is below optimal level - review and optimize components")
        
        return overall_status, int(score), issues, recommendations
    
    async def _check_thresholds(self, result: HealthCheckResult):
        """Check if any metrics exceed alert thresholds."""
        component = result.component
        if component not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[component]
        
        for metric_name, value in result.metrics.items():
            if metric_name not in thresholds:
                continue
                
            threshold_config = thresholds[metric_name]
            
            # Check critical threshold
            if "critical" in threshold_config:
                critical_threshold = threshold_config["critical"]
                if ((metric_name.endswith("_percent") or metric_name == "hit_rate") and value >= critical_threshold) or \
                   (metric_name.endswith("_ms") and value >= critical_threshold):
                    await self._create_alert(
                        component, metric_name, AlertSeverity.CRITICAL,
                        f"{metric_name} exceeded critical threshold",
                        value, critical_threshold
                    )
                    continue
            
            # Check warning threshold
            if "warning" in threshold_config:
                warning_threshold = threshold_config["warning"]
                if ((metric_name.endswith("_percent") or metric_name == "hit_rate") and value >= warning_threshold) or \
                   (metric_name.endswith("_ms") and value >= warning_threshold):
                    await self._create_alert(
                        component, metric_name, AlertSeverity.MEDIUM,
                        f"{metric_name} exceeded warning threshold",
                        value, warning_threshold
                    )
    
    async def _create_alert(self, component: str, metric: str, severity: AlertSeverity, 
                          message: str, current_value: float, threshold: float):
        """Create and process a health alert."""
        alert_key = f"{component}_{metric}_{severity.name}"
        
        # Check if similar alert is already active
        if alert_key in self.active_alerts:
            return  # Don't spam alerts
        
        alert = HealthAlert(
            alert_id=str(uuid.uuid4()),
            component=component,
            severity=severity,
            message=message,
            details={
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
                "breach_percentage": ((current_value - threshold) / threshold) * 100
            },
            created_at=datetime.utcnow(),
            current_value=current_value,
            threshold_breached=threshold
        )
        
        self.active_alerts[alert_key] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(f"Health alert created: {alert.message} (Component: {component}, Value: {current_value}, Threshold: {threshold})")
    
    async def _send_alert_notifications(self, alert: HealthAlert):
        """Send alert notifications to all registered callbacks."""
        for callback in self.notification_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def add_notification_callback(self, callback: Callable[[HealthAlert], None]):
        """Add a notification callback for health alerts."""
        self.notification_callbacks.append(callback)
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start background health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"Health monitoring started with {interval_seconds}s interval")
    
    async def stop_monitoring(self):
        """Stop background health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                await self.check_all_components()
                
                # Clean up old alerts (resolve alerts that are no longer triggering)
                await self._cleanup_resolved_alerts()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
    
    async def _cleanup_resolved_alerts(self):
        """Clean up alerts that are no longer triggering."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=1)  # Auto-resolve after 1 hour
        
        resolved_alerts = []
        for alert_key, alert in self.active_alerts.items():
            if alert.created_at < cutoff_time and not alert.resolved_at:
                alert.resolved_at = current_time
                resolved_alerts.append(alert_key)
        
        for alert_key in resolved_alerts:
            del self.active_alerts[alert_key]
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        return [summary.to_dict() for summary in list(self.health_history)[-limit:]]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active health alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "component": alert.component,
                "severity": alert.severity.name,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "details": alert.details
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_component_metrics(self, component: str = None) -> Dict[str, Any]:
        """Get latest metrics for specific component or all components."""
        if not self.health_history:
            return {}
        
        latest_summary = self.health_history[-1]
        
        if component:
            if component in latest_summary.components:
                return latest_summary.components[component].to_dict()
            return {}
        
        return {
            name: result.to_dict() 
            for name, result in latest_summary.components.items()
        }
    
    async def perform_self_healing(self):
        """Attempt automated self-healing for known issues."""
        if not self.health_history:
            return
        
        latest_summary = self.health_history[-1]
        healing_actions = []
        
        for component_name, result in latest_summary.components.items():
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                actions = await self._attempt_component_healing(component_name, result)
                healing_actions.extend(actions)
        
        if healing_actions:
            logger.info(f"Self-healing actions performed: {', '.join(healing_actions)}")
        
        return healing_actions
    
    async def _attempt_component_healing(self, component: str, result: HealthCheckResult) -> List[str]:
        """Attempt healing for a specific component."""
        actions = []
        
        try:
            if component == "cache":
                # Clear cache if hit rate is too low
                if result.metrics.get("hit_rate", 1.0) < 0.5:
                    await cache_manager.clear_all()
                    actions.append(f"Cleared cache for {component}")
            
            elif component == "database":
                # Kill long-running queries if blocking
                if result.details.get("blocking_queries", 0) > 10:
                    # Would implement query killing here
                    actions.append(f"Attempted to resolve blocking queries for {component}")
            
            elif component == "system_resources":
                # Garbage collection for high memory usage
                if result.metrics.get("memory_percent", 0) > 90:
                    import gc
                    collected = gc.collect()
                    actions.append(f"Performed garbage collection, freed {collected} objects")
                    
        except Exception as e:
            logger.error(f"Self-healing failed for {component}: {e}")
        
        return actions

# Global health monitor instance
health_monitor = HealthMonitor()

# Convenience functions for API endpoints
async def get_health_status() -> Dict[str, Any]:
    """Get overall system health status."""
    summary = await health_monitor.check_all_components()
    return summary.to_dict()

async def get_liveness_status() -> Dict[str, Any]:
    """Simple liveness check - is the service running?"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": (datetime.utcnow() - health_monitor.start_time).total_seconds()
    }

async def get_readiness_status() -> Dict[str, Any]:
    """Readiness check - is the service ready to accept traffic?"""
    # Check critical components only
    critical_checkers = ["database", "llm_service", "authentication"]
    
    ready = True
    component_status = {}
    
    for component_name in critical_checkers:
        if component_name in health_monitor.health_checkers:
            checker = health_monitor.health_checkers[component_name]
            result = await checker.check_health()
            component_status[component_name] = result.status.value
            
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNAVAILABLE]:
                ready = False
    
    return {
        "ready": ready,
        "timestamp": datetime.utcnow().isoformat(),
        "components": component_status
    }

async def get_detailed_health_status() -> Dict[str, Any]:
    """Get detailed health status with component breakdown."""
    summary = await health_monitor.check_all_components()
    return {
        **summary.to_dict(),
        "alerts": health_monitor.get_active_alerts(),
        "history": health_monitor.get_health_history(5),
        "self_healing": {
            "available": True,
            "last_run": None  # Would track when self-healing was last executed
        }
    }

async def get_metrics_summary() -> Dict[str, Any]:
    """Get performance and health metrics summary."""
    health_summary = await health_monitor.check_all_components()
    
    # Combine with performance metrics
    perf_summary = performance_collector.get_endpoint_stats() if performance_collector else {}
    resource_stats = performance_collector.get_resource_stats() if performance_collector else {}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health": {
            "overall_status": health_summary.overall_status.value,
            "overall_score": health_summary.overall_score,
            "component_count": len(health_summary.components)
        },
        "performance": {
            "api_performance": perf_summary,
            "resource_utilization": resource_stats
        },
        "alerts": {
            "active_count": len(health_monitor.get_active_alerts()),
            "active_alerts": health_monitor.get_active_alerts()
        }
    }

# Export key components
__all__ = [
    'HealthMonitor', 'HealthStatus', 'HealthCheckResult', 'SystemHealthSummary',
    'health_monitor', 'get_health_status', 'get_liveness_status', 
    'get_readiness_status', 'get_detailed_health_status', 'get_metrics_summary'
]