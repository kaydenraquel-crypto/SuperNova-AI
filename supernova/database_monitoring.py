"""
Database Monitoring and Alerting System

Comprehensive database monitoring including:
- Real-time performance metrics collection
- Health monitoring and status checks
- Alert generation and notification
- Dashboard metrics for visualization
- Proactive issue detection
- Resource usage monitoring
"""

from __future__ import annotations
import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import json
import statistics
from contextlib import contextmanager
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from sqlalchemy import text, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import Pool

try:
    from .config import settings
    from .database_config import db_config, get_session
    from .database_optimization import db_optimizer
    import requests
    WEB_REQUESTS_AVAILABLE = True
except ImportError:
    WEB_REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class DatabaseMetrics:
    """Database performance metrics."""
    timestamp: datetime
    database: str
    
    # Connection metrics
    active_connections: int
    max_connections: int
    connection_utilization: float
    
    # Query metrics
    queries_per_second: float
    slow_queries_per_second: float
    avg_query_duration: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_gb: float
    disk_usage_percent: float
    
    # Cache metrics
    cache_hit_ratio: float
    buffer_hit_ratio: float
    
    # Lock metrics
    lock_waits: int
    deadlocks: int
    blocked_queries: int
    
    # Replication metrics (if applicable)
    replication_lag_seconds: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = None

@dataclass
class DatabaseAlert:
    """Database alert information."""
    alert_id: str
    database: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None

class AlertRule:
    """Database alert rule configuration."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: str,  # '>', '<', '>=', '<=', '==', '!='
        threshold: float,
        severity: AlertSeverity,
        database: str = None,
        description: str = "",
        cooldown_minutes: int = 5
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.database = database  # None means all databases
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None
    
    def evaluate(self, metrics: DatabaseMetrics) -> bool:
        """Evaluate if alert should be triggered."""
        if self.database and metrics.database != self.database:
            return False
        
        # Check cooldown period
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return False
        
        # Get metric value
        value = getattr(metrics, self.metric_name, None)
        if value is None:
            return False
        
        # Evaluate condition
        result = False
        if self.condition == '>':
            result = value > self.threshold
        elif self.condition == '<':
            result = value < self.threshold
        elif self.condition == '>=':
            result = value >= self.threshold
        elif self.condition == '<=':
            result = value <= self.threshold
        elif self.condition == '==':
            result = value == self.threshold
        elif self.condition == '!=':
            result = value != self.threshold
        
        return result

class NotificationChannel:
    """Base class for notification channels."""
    
    def send_notification(self, alert: DatabaseAlert) -> bool:
        """Send notification for alert."""
        raise NotImplementedError

class EmailNotification(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, 
                 recipients: List[str], use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls
    
    def send_notification(self, alert: DatabaseAlert) -> bool:
        """Send email notification."""
        try:
            subject = f"[SuperNova] Database Alert: {alert.alert_type} ({alert.severity.value.upper()})"
            
            body = f"""
Database Alert Triggered

Database: {alert.database}
Alert Type: {alert.alert_type}
Severity: {alert.severity.value.upper()}
Time: {alert.created_at.isoformat()}

Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

---
SuperNova AI Database Monitoring System
            """.strip()
            
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

class WebhookNotification(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send_notification(self, alert: DatabaseAlert) -> bool:
        """Send webhook notification."""
        if not WEB_REQUESTS_AVAILABLE:
            logger.error("Requests library not available for webhook notifications")
            return False
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'database': alert.database,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'message': alert.message,
                'details': alert.details,
                'created_at': alert.created_at.isoformat(),
                'threshold_value': alert.threshold_value,
                'current_value': alert.current_value
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

class SlackNotification(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_notification(self, alert: DatabaseAlert) -> bool:
        """Send Slack notification."""
        if not WEB_REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Slack notifications")
            return False
        
        try:
            # Map severity to colors
            colors = {
                AlertSeverity.LOW: "#36a64f",      # green
                AlertSeverity.MEDIUM: "#ff9500",   # orange
                AlertSeverity.HIGH: "#ff5500",     # red-orange
                AlertSeverity.CRITICAL: "#ff0000" # red
            }
            
            payload = {
                "attachments": [
                    {
                        "color": colors.get(alert.severity, "#cccccc"),
                        "title": f"Database Alert: {alert.alert_type}",
                        "fields": [
                            {"title": "Database", "value": alert.database, "short": True},
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Time", "value": alert.created_at.isoformat(), "short": True}
                        ]
                    }
                ]
            }
            
            if alert.current_value is not None and alert.threshold_value is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Value",
                    "value": f"Current: {alert.current_value}, Threshold: {alert.threshold_value}",
                    "short": True
                })
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

class DatabaseMonitor:
    """Database monitoring system."""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, DatabaseAlert] = {}
        self.alert_rules: List[AlertRule] = []
        self.notification_channels: List[NotificationChannel] = []
        
        # Monitoring configuration
        self.monitoring_interval = getattr(settings, 'DB_MONITORING_INTERVAL', 60)  # seconds
        self.metrics_retention_hours = getattr(settings, 'DB_METRICS_RETENTION_HOURS', 24)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup notification channels
        self._setup_notification_channels()
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                "High Connection Usage",
                "connection_utilization",
                ">",
                0.8,
                AlertSeverity.HIGH,
                description="Database connection pool usage above 80%"
            ),
            AlertRule(
                "Critical Connection Usage",
                "connection_utilization",
                ">",
                0.95,
                AlertSeverity.CRITICAL,
                description="Database connection pool usage above 95%"
            ),
            AlertRule(
                "High Query Duration",
                "avg_query_duration",
                ">",
                2.0,
                AlertSeverity.MEDIUM,
                description="Average query duration above 2 seconds"
            ),
            AlertRule(
                "Critical Query Duration",
                "avg_query_duration",
                ">",
                5.0,
                AlertSeverity.HIGH,
                description="Average query duration above 5 seconds"
            ),
            AlertRule(
                "Many Slow Queries",
                "slow_queries_per_second",
                ">",
                5.0,
                AlertSeverity.MEDIUM,
                description="More than 5 slow queries per second"
            ),
            AlertRule(
                "Low Cache Hit Ratio",
                "cache_hit_ratio",
                "<",
                0.8,
                AlertSeverity.MEDIUM,
                description="Cache hit ratio below 80%"
            ),
            AlertRule(
                "High Disk Usage",
                "disk_usage_percent",
                ">",
                85.0,
                AlertSeverity.HIGH,
                description="Disk usage above 85%"
            ),
            AlertRule(
                "Critical Disk Usage",
                "disk_usage_percent",
                ">",
                95.0,
                AlertSeverity.CRITICAL,
                description="Disk usage above 95%"
            ),
            AlertRule(
                "Deadlocks Detected",
                "deadlocks",
                ">",
                0,
                AlertSeverity.HIGH,
                description="Database deadlocks detected"
            ),
            AlertRule(
                "High Memory Usage",
                "memory_usage_mb",
                ">",
                8192,  # 8GB
                AlertSeverity.MEDIUM,
                description="Memory usage above 8GB"
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def _setup_notification_channels(self) -> None:
        """Setup notification channels from settings."""
        # Email notifications
        if hasattr(settings, 'ALERT_EMAIL_SMTP_HOST') and hasattr(settings, 'ALERT_EMAIL_RECIPIENTS'):
            try:
                email_channel = EmailNotification(
                    smtp_host=settings.ALERT_EMAIL_SMTP_HOST,
                    smtp_port=getattr(settings, 'ALERT_EMAIL_SMTP_PORT', 587),
                    username=getattr(settings, 'ALERT_EMAIL_USERNAME', ''),
                    password=getattr(settings, 'ALERT_EMAIL_PASSWORD', ''),
                    recipients=settings.ALERT_EMAIL_RECIPIENTS,
                    use_tls=getattr(settings, 'ALERT_EMAIL_USE_TLS', True)
                )
                self.notification_channels.append(email_channel)
            except Exception as e:
                logger.error(f"Failed to setup email notifications: {e}")
        
        # Webhook notifications
        if hasattr(settings, 'ALERT_WEBHOOK_URL'):
            webhook_channel = WebhookNotification(settings.ALERT_WEBHOOK_URL)
            self.notification_channels.append(webhook_channel)
        
        # Slack notifications
        if hasattr(settings, 'ALERT_SLACK_WEBHOOK_URL'):
            slack_channel = SlackNotification(settings.ALERT_SLACK_WEBHOOK_URL)
            self.notification_channels.append(slack_channel)
    
    def start_monitoring(self) -> None:
        """Start database monitoring."""
        if self.monitoring_active:
            logger.warning("Database monitoring is already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring thread
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_all_metrics()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Database monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop database monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Database monitoring stopped")
    
    def _collect_all_metrics(self) -> None:
        """Collect metrics from all configured databases."""
        urls = db_config.get_database_urls()
        
        for database in urls.keys():
            if database == 'test':
                continue
            
            try:
                metrics = self._collect_database_metrics(database)
                if metrics:
                    # Store metrics
                    self.metrics_history[database].append(metrics)
                    
                    # Evaluate alert rules
                    self._evaluate_alert_rules(metrics)
                    
            except Exception as e:
                logger.error(f"Failed to collect metrics for {database}: {e}")
        
        # Cleanup old metrics
        self._cleanup_old_metrics()
        
        # Cleanup resolved alerts
        self._cleanup_resolved_alerts()
    
    def _collect_database_metrics(self, database: str) -> Optional[DatabaseMetrics]:
        """Collect metrics for a specific database."""
        try:
            with get_session(database) as session:
                timestamp = datetime.utcnow()
                
                # Connection metrics
                active_connections = self._get_active_connections(session, database)
                max_connections = self._get_max_connections(session, database)
                connection_utilization = active_connections / max_connections if max_connections > 0 else 0
                
                # Query metrics
                queries_per_second = self._get_queries_per_second(session, database)
                slow_queries_per_second = self._get_slow_queries_per_second(session, database)
                avg_query_duration = self._get_avg_query_duration(session, database)
                
                # Resource metrics
                cpu_usage = self._get_cpu_usage(session, database)
                memory_usage = self._get_memory_usage(session, database)
                disk_usage_gb, disk_usage_percent = self._get_disk_usage(session, database)
                
                # Cache metrics
                cache_hit_ratio = self._get_cache_hit_ratio(session, database)
                buffer_hit_ratio = self._get_buffer_hit_ratio(session, database)
                
                # Lock metrics
                lock_waits = self._get_lock_waits(session, database)
                deadlocks = self._get_deadlocks(session, database)
                blocked_queries = self._get_blocked_queries(session, database)
                
                # Replication metrics
                replication_lag = self._get_replication_lag(session, database)
                
                return DatabaseMetrics(
                    timestamp=timestamp,
                    database=database,
                    active_connections=active_connections,
                    max_connections=max_connections,
                    connection_utilization=connection_utilization,
                    queries_per_second=queries_per_second,
                    slow_queries_per_second=slow_queries_per_second,
                    avg_query_duration=avg_query_duration,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_mb=memory_usage,
                    disk_usage_gb=disk_usage_gb,
                    disk_usage_percent=disk_usage_percent,
                    cache_hit_ratio=cache_hit_ratio,
                    buffer_hit_ratio=buffer_hit_ratio,
                    lock_waits=lock_waits,
                    deadlocks=deadlocks,
                    blocked_queries=blocked_queries,
                    replication_lag_seconds=replication_lag
                )
                
        except Exception as e:
            logger.error(f"Failed to collect metrics for {database}: {e}")
            return None
    
    def _get_active_connections(self, session, database: str) -> int:
        """Get number of active connections."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state IN ('active', 'idle in transaction')"
                ))
                return result.scalar() or 0
            else:
                return 1  # SQLite
        except Exception:
            return 0
    
    def _get_max_connections(self, session, database: str) -> int:
        """Get maximum number of connections."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text("SHOW max_connections"))
                return int(result.scalar() or 100)
            else:
                return 1  # SQLite
        except Exception:
            return 100  # Default
    
    def _get_queries_per_second(self, session, database: str) -> float:
        """Get queries per second rate."""
        try:
            # This would typically come from query monitoring
            if hasattr(db_optimizer, 'query_monitor'):
                recent_queries = db_optimizer.query_monitor.recent_queries
                if recent_queries:
                    # Count queries in last minute
                    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                    recent_count = sum(
                        1 for q in recent_queries 
                        if q.get('timestamp', datetime.min) > one_minute_ago
                    )
                    return recent_count / 60.0  # Convert to per second
            return 0.0
        except Exception:
            return 0.0
    
    def _get_slow_queries_per_second(self, session, database: str) -> float:
        """Get slow queries per second rate."""
        try:
            if hasattr(db_optimizer, 'query_monitor'):
                recent_queries = db_optimizer.query_monitor.recent_queries
                if recent_queries:
                    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                    slow_count = sum(
                        1 for q in recent_queries 
                        if q.get('timestamp', datetime.min) > one_minute_ago and q.get('is_slow', False)
                    )
                    return slow_count / 60.0
            return 0.0
        except Exception:
            return 0.0
    
    def _get_avg_query_duration(self, session, database: str) -> float:
        """Get average query duration."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT avg(mean_time) FROM pg_stat_statements WHERE calls > 0"
                ))
                return (result.scalar() or 0.0) / 1000.0  # Convert ms to seconds
            return 0.0
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self, session, database: str) -> float:
        """Get CPU usage percentage."""
        try:
            # This would typically require system monitoring
            # For now, return 0 as placeholder
            return 0.0
        except Exception:
            return 0.0
    
    def _get_memory_usage(self, session, database: str) -> float:
        """Get memory usage in MB."""
        try:
            if 'postgresql' in str(session.bind.url):
                # Get shared buffer usage
                result = session.execute(text(
                    "SELECT setting::int * 8192 / 1024 / 1024 FROM pg_settings WHERE name = 'shared_buffers'"
                ))
                return result.scalar() or 0.0
            return 0.0
        except Exception:
            return 0.0
    
    def _get_disk_usage(self, session, database: str) -> Tuple[float, float]:
        """Get disk usage in GB and percentage."""
        try:
            if 'postgresql' in str(session.bind.url):
                # Get database size
                result = session.execute(text(
                    "SELECT pg_database_size(current_database()) / 1024.0 / 1024.0 / 1024.0"
                ))
                size_gb = result.scalar() or 0.0
                
                # For percentage, we'd need filesystem info
                # For now, assume 1TB total and calculate percentage
                percentage = (size_gb / 1000.0) * 100
                return size_gb, min(percentage, 100.0)
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    def _get_cache_hit_ratio(self, session, database: str) -> float:
        """Get cache hit ratio."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    """
                    SELECT 
                        sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1.0) 
                    FROM pg_statio_user_tables
                    """
                ))
                return result.scalar() or 0.0
            return 1.0  # SQLite doesn't have cache metrics
        except Exception:
            return 0.0
    
    def _get_buffer_hit_ratio(self, session, database: str) -> float:
        """Get buffer hit ratio."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    """
                    SELECT 
                        blks_hit / (blks_hit + blks_read + 1.0) 
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                    """
                ))
                return result.scalar() or 0.0
            return 1.0  # SQLite
        except Exception:
            return 0.0
    
    def _get_lock_waits(self, session, database: str) -> int:
        """Get number of lock waits."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
                ))
                return result.scalar() or 0
            return 0
        except Exception:
            return 0
    
    def _get_deadlocks(self, session, database: str) -> int:
        """Get number of deadlocks."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT deadlocks FROM pg_stat_database WHERE datname = current_database()"
                ))
                return result.scalar() or 0
            return 0
        except Exception:
            return 0
    
    def _get_blocked_queries(self, session, database: str) -> int:
        """Get number of blocked queries."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL"
                ))
                return result.scalar() or 0
            return 0
        except Exception:
            return 0
    
    def _get_replication_lag(self, session, database: str) -> Optional[float]:
        """Get replication lag in seconds."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    """
                    SELECT 
                        EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) 
                    AS lag_seconds
                    """
                ))
                return result.scalar()
            return None
        except Exception:
            return None
    
    def _evaluate_alert_rules(self, metrics: DatabaseMetrics) -> None:
        """Evaluate alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                if rule.evaluate(metrics):
                    self._trigger_alert(rule, metrics)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _trigger_alert(self, rule: AlertRule, metrics: DatabaseMetrics) -> None:
        """Trigger an alert."""
        alert_id = f"{metrics.database}_{rule.name}_{int(metrics.timestamp.timestamp())}"
        
        # Check if alert is already active
        if alert_id in self.active_alerts:
            return
        
        # Get current value
        current_value = getattr(metrics, rule.metric_name, None)
        
        # Create alert
        alert = DatabaseAlert(
            alert_id=alert_id,
            database=metrics.database,
            alert_type=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description} (Current: {current_value}, Threshold: {rule.threshold})",
            details={
                'metric_name': rule.metric_name,
                'condition': rule.condition,
                'threshold': rule.threshold,
                'current_value': current_value,
                'rule_description': rule.description
            },
            created_at=metrics.timestamp,
            threshold_value=rule.threshold,
            current_value=current_value
        )
        
        # Store active alert
        self.active_alerts[alert_id] = alert
        
        # Update rule last triggered time
        rule.last_triggered = metrics.timestamp
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    def _send_alert_notifications(self, alert: DatabaseAlert) -> None:
        """Send alert notifications through all configured channels."""
        for channel in self.notification_channels:
            try:
                success = channel.send_notification(alert)
                if not success:
                    logger.error(f"Failed to send notification via {type(channel).__name__}")
            except Exception as e:
                logger.error(f"Error sending notification via {type(channel).__name__}: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics based on retention policy."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        
        for database, metrics_queue in self.metrics_history.items():
            while metrics_queue and metrics_queue[0].timestamp < cutoff_time:
                metrics_queue.popleft()
    
    def _cleanup_resolved_alerts(self) -> None:
        """Remove resolved alerts older than 24 hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.status == AlertStatus.RESOLVED and 
               alert.resolved_at and alert.resolved_at < cutoff_time
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def acknowledge_alert(self, alert_id: str, user: str = None) -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        
        if user:
            alert.details['acknowledged_by'] = user
        
        logger.info(f"Alert {alert_id} acknowledged by {user or 'system'}")
        return True
    
    def resolve_alert(self, alert_id: str, user: str = None) -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        if user:
            alert.details['resolved_by'] = user
        
        logger.info(f"Alert {alert_id} resolved by {user or 'system'}")
        return True
    
    def get_current_metrics(self, database: str = None) -> Dict[str, DatabaseMetrics]:
        """Get current metrics for databases."""
        current_metrics = {}
        
        for db_name, metrics_queue in self.metrics_history.items():
            if database and db_name != database:
                continue
            
            if metrics_queue:
                current_metrics[db_name] = metrics_queue[-1]
        
        return current_metrics
    
    def get_metrics_history(self, database: str, hours: int = 1) -> List[DatabaseMetrics]:
        """Get metrics history for a database."""
        if database not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history[database]
            if metrics.timestamp > cutoff_time
        ]
    
    def get_active_alerts(self, database: str = None, severity: AlertSeverity = None) -> List[DatabaseAlert]:
        """Get active alerts."""
        alerts = list(self.active_alerts.values())
        
        if database:
            alerts = [alert for alert in alerts if alert.database == database]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1
        }
        
        return sorted(alerts, key=lambda x: (severity_order[x.severity], x.created_at), reverse=True)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        total_alerts = len(self.active_alerts)
        alerts_by_severity = defaultdict(int)
        
        for alert in self.active_alerts.values():
            alerts_by_severity[alert.severity.value] += 1
        
        return {
            'monitoring_active': self.monitoring_active,
            'monitoring_interval': self.monitoring_interval,
            'databases_monitored': list(self.metrics_history.keys()),
            'total_active_alerts': total_alerts,
            'alerts_by_severity': dict(alerts_by_severity),
            'notification_channels': len(self.notification_channels),
            'alert_rules_count': len(self.alert_rules),
            'metrics_retention_hours': self.metrics_retention_hours
        }

# Global monitor instance
db_monitor = DatabaseMonitor()

# Convenience functions
def start_database_monitoring() -> None:
    """Start database monitoring."""
    db_monitor.start_monitoring()

def stop_database_monitoring() -> None:
    """Stop database monitoring."""
    db_monitor.stop_monitoring()

def get_monitoring_status() -> Dict[str, Any]:
    """Get monitoring status."""
    return db_monitor.get_monitoring_status()

def get_current_metrics(database: str = None) -> Dict[str, DatabaseMetrics]:
    """Get current database metrics."""
    return db_monitor.get_current_metrics(database)

def get_active_alerts(database: str = None) -> List[DatabaseAlert]:
    """Get active database alerts."""
    return db_monitor.get_active_alerts(database)

def acknowledge_alert(alert_id: str, user: str = None) -> bool:
    """Acknowledge an alert."""
    return db_monitor.acknowledge_alert(alert_id, user)

def resolve_alert(alert_id: str, user: str = None) -> bool:
    """Resolve an alert."""
    return db_monitor.resolve_alert(alert_id, user)