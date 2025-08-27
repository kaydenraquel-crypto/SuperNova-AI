"""
SuperNova AI Advanced Alerting System
=====================================

Multi-level alerting and notification system with:
- Configurable alert rules and thresholds
- Multiple notification channels (email, Slack, webhooks, PagerDuty)
- Alert aggregation and deduplication
- Escalation policies and maintenance windows
- Alert fatigue prevention
- Business logic alerts
"""

import asyncio
import json
import logging
import smtplib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, IntEnum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor

# External service imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from .config import settings
    from .health_monitor import HealthAlert, AlertSeverity
    from .performance_monitor import performance_collector
    from .database_monitoring import db_monitor
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertLevel(IntEnum):
    """Alert severity levels."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    DISCORD = "discord"
    TEAMS = "teams"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    metric_path: str  # e.g., "database.query_time_ms", "system.cpu_percent"
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    level: AlertLevel
    evaluation_window: timedelta = timedelta(minutes=5)
    min_duration: timedelta = timedelta(minutes=2)
    cooldown_period: timedelta = timedelta(minutes=15)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    custom_message: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Other rules this depends on
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule should trigger based on current value."""
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return value == self.threshold
        elif self.operator == "!=":
            return value != self.threshold
        return False

@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    title: str
    message: str
    level: AlertLevel
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    component: str
    metric_name: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    fingerprint: Optional[str] = None  # For deduplication
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    escalated_at: Optional[datetime] = None
    escalation_level: int = 0
    suppressed_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    notifications_sent: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate fingerprint for deduplication."""
        if not self.fingerprint:
            content = f"{self.rule_id}_{self.component}_{self.metric_name}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()

@dataclass
class NotificationConfig:
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[timedelta] = None
    severity_filter: Set[AlertLevel] = field(default_factory=set)
    
class NotificationHandler:
    """Base class for notification handlers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_sent: Dict[str, datetime] = {}
        
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        raise NotImplementedError
    
    def _check_rate_limit(self, alert_id: str, rate_limit: timedelta) -> bool:
        """Check if rate limit allows sending notification."""
        if not rate_limit:
            return True
            
        last_sent = self.last_sent.get(alert_id)
        if not last_sent:
            return True
            
        return datetime.utcnow() - last_sent >= rate_limit
    
    def _record_sent(self, alert_id: str):
        """Record that notification was sent."""
        self.last_sent[alert_id] = datetime.utcnow()

class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            smtp_config = self.config
            
            # Create email content
            subject = f"[SuperNova AI] {alert.level.name} Alert: {alert.title}"
            
            body = self._create_email_body(alert)
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = smtp_config.get('from_address', 'alerts@supernova-ai.com')
            msg['To'] = ', '.join(smtp_config.get('recipients', []))
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port', 587))
            if smtp_config.get('use_tls', True):
                server.starttls()
                
            if smtp_config.get('username'):
                server.login(smtp_config.get('username'), smtp_config.get('password'))
            
            server.send_message(msg)
            server.quit()
            
            self._record_sent(alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create email body content."""
        body = f"""
SuperNova AI System Alert

Alert Details:
- Alert ID: {alert.alert_id}
- Level: {alert.level.name}
- Component: {alert.component}
- Message: {alert.message}
- Created: {alert.created_at.isoformat()}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}

Status: {alert.status.value.title()}

Additional Information:
{json.dumps(alert.metadata, indent=2)}

This is an automated message from SuperNova AI monitoring system.
        """.strip()
        
        return body

class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Slack notifications")
            return False
            
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Color coding based on severity
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.LOW: "#36a64f",
                AlertLevel.MEDIUM: "#ffbb33",
                AlertLevel.HIGH: "#ff8800",
                AlertLevel.CRITICAL: "#ff0000"
            }
            
            # Create Slack message
            payload = {
                "username": "SuperNova AI Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "#cccccc"),
                        "title": f"{alert.level.name} Alert: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Component", "value": alert.component, "short": True},
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                            {"title": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
                            {"title": "Alert ID", "value": alert.alert_id, "short": True}
                        ],
                        "footer": "SuperNova AI Monitoring",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self._record_sent(alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

class WebhookNotificationHandler(NotificationHandler):
    """Generic webhook notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for webhook notifications")
            return False
            
        try:
            url = self.config.get('url')
            if not url:
                logger.error("Webhook URL not configured")
                return False
            
            # Create payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "level": alert.level.name,
                "status": alert.status.value,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata,
                "service": "supernova-ai"
            }
            
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            self._record_sent(alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

class PagerDutyNotificationHandler(NotificationHandler):
    """PagerDuty notification handler."""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for PagerDuty notifications")
            return False
            
        try:
            integration_key = self.config.get('integration_key')
            if not integration_key:
                logger.error("PagerDuty integration key not configured")
                return False
            
            # Map alert levels to PagerDuty severity
            severity_map = {
                AlertLevel.INFO: "info",
                AlertLevel.LOW: "info", 
                AlertLevel.MEDIUM: "warning",
                AlertLevel.HIGH: "error",
                AlertLevel.CRITICAL: "critical"
            }
            
            # Create PagerDuty event
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"{alert.component}: {alert.title}",
                    "source": "supernova-ai-monitor",
                    "severity": severity_map.get(alert.level, "error"),
                    "component": alert.component,
                    "group": "system-health",
                    "class": alert.metric_name,
                    "custom_details": {
                        "alert_id": alert.alert_id,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold_value,
                        "metadata": alert.metadata
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            
            self._record_sent(alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False

class AlertManager:
    """Main alert management system."""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: Dict[NotificationChannel, NotificationHandler] = {}
        self.escalation_policies: Dict[str, Dict] = {}
        self.maintenance_windows: List[Dict] = []
        self.suppression_rules: List[Dict] = []
        
        # Metrics tracking
        self.metric_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_evaluation: Dict[str, datetime] = {}
        
        # Background processing
        self.processing_active = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup default rules and handlers
        self._setup_default_rules()
        self._setup_notification_handlers()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        default_rules = [
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                description="System CPU usage is above normal levels",
                metric_path="system.cpu_percent",
                operator=">",
                threshold=85.0,
                level=AlertLevel.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="memory_critical",
                name="Critical Memory Usage",
                description="System memory usage is critically high",
                metric_path="system.memory_percent",
                operator=">",
                threshold=95.0,
                level=AlertLevel.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY]
            ),
            AlertRule(
                rule_id="disk_high",
                name="High Disk Usage",
                description="Disk space usage is high",
                metric_path="system.disk_percent",
                operator=">",
                threshold=90.0,
                level=AlertLevel.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="api_response_slow",
                name="Slow API Response Times",
                description="API response times are above acceptable levels",
                metric_path="api.response_time_p95",
                operator=">",
                threshold=2000.0,  # 2 seconds
                level=AlertLevel.MEDIUM,
                notification_channels=[NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="database_slow",
                name="Slow Database Queries",
                description="Database query times are high",
                metric_path="database.avg_query_time_ms",
                operator=">",
                threshold=1000.0,  # 1 second
                level=AlertLevel.MEDIUM,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="High Error Rate",
                description="API error rate is above normal levels",
                metric_path="api.error_rate",
                operator=">",
                threshold=0.05,  # 5%
                level=AlertLevel.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="cache_hit_low",
                name="Low Cache Hit Rate",
                description="Cache hit rate is below optimal levels",
                metric_path="cache.hit_rate",
                operator="<",
                threshold=0.7,  # 70%
                level=AlertLevel.MEDIUM,
                notification_channels=[NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="llm_response_timeout",
                name="LLM Service Timeout",
                description="LLM service response time is too high",
                metric_path="llm.response_time_ms",
                operator=">",
                threshold=10000.0,  # 10 seconds
                level=AlertLevel.HIGH,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _setup_notification_handlers(self):
        """Setup notification handlers based on configuration."""
        if not CONFIG_AVAILABLE:
            return
        
        # Email handler
        if hasattr(settings, 'ALERT_EMAIL_SMTP_HOST'):
            email_config = {
                'host': settings.ALERT_EMAIL_SMTP_HOST,
                'port': getattr(settings, 'ALERT_EMAIL_SMTP_PORT', 587),
                'username': getattr(settings, 'ALERT_EMAIL_USERNAME', ''),
                'password': getattr(settings, 'ALERT_EMAIL_PASSWORD', ''),
                'from_address': getattr(settings, 'ALERT_EMAIL_FROM', 'alerts@supernova-ai.com'),
                'recipients': getattr(settings, 'ALERT_EMAIL_RECIPIENTS', []),
                'use_tls': getattr(settings, 'ALERT_EMAIL_USE_TLS', True)
            }
            self.notification_handlers[NotificationChannel.EMAIL] = EmailNotificationHandler(email_config)
        
        # Slack handler
        if hasattr(settings, 'ALERT_SLACK_WEBHOOK_URL'):
            slack_config = {'webhook_url': settings.ALERT_SLACK_WEBHOOK_URL}
            self.notification_handlers[NotificationChannel.SLACK] = SlackNotificationHandler(slack_config)
        
        # Webhook handler
        if hasattr(settings, 'ALERT_WEBHOOK_URL'):
            webhook_config = {
                'url': settings.ALERT_WEBHOOK_URL,
                'headers': getattr(settings, 'ALERT_WEBHOOK_HEADERS', {'Content-Type': 'application/json'})
            }
            self.notification_handlers[NotificationChannel.WEBHOOK] = WebhookNotificationHandler(webhook_config)
        
        # PagerDuty handler
        if hasattr(settings, 'PAGERDUTY_INTEGRATION_KEY'):
            pagerduty_config = {'integration_key': settings.PAGERDUTY_INTEGRATION_KEY}
            self.notification_handlers[NotificationChannel.PAGERDUTY] = PagerDutyNotificationHandler(pagerduty_config)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def enable_alert_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_alert_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            return True
        return False
    
    async def record_metric(self, metric_path: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value for evaluation."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Store metric value
        metric_entry = {'value': value, 'timestamp': timestamp}
        self.metric_buffer[metric_path].append(metric_entry)
        
        # Trigger immediate evaluation for this metric
        await self._evaluate_metric_rules(metric_path, value, timestamp)
    
    async def _evaluate_metric_rules(self, metric_path: str, value: float, timestamp: datetime):
        """Evaluate all rules for a specific metric."""
        for rule in self.alert_rules.values():
            if not rule.enabled or rule.metric_path != metric_path:
                continue
            
            # Check if rule should be evaluated (not in cooldown)
            last_eval = self.last_evaluation.get(rule.rule_id)
            if last_eval and timestamp - last_eval < rule.cooldown_period:
                continue
            
            # Evaluate rule
            if rule.evaluate(value):
                await self._check_rule_duration(rule, value, timestamp)
            
            self.last_evaluation[rule.rule_id] = timestamp
    
    async def _check_rule_duration(self, rule: AlertRule, current_value: float, timestamp: datetime):
        """Check if rule has been triggering for minimum duration."""
        # Get recent values for this metric
        metric_data = self.metric_buffer[rule.metric_path]
        
        # Find values within the evaluation window that trigger the rule
        window_start = timestamp - rule.evaluation_window
        triggering_values = [
            entry for entry in metric_data
            if entry['timestamp'] >= window_start and rule.evaluate(entry['value'])
        ]
        
        # Check if rule has been triggering for minimum duration
        if triggering_values:
            earliest_trigger = min(entry['timestamp'] for entry in triggering_values)
            trigger_duration = timestamp - earliest_trigger
            
            if trigger_duration >= rule.min_duration:
                await self._create_alert(rule, current_value, timestamp)
    
    async def _create_alert(self, rule: AlertRule, current_value: float, timestamp: datetime):
        """Create a new alert."""
        # Check if similar alert already exists (deduplication)
        fingerprint = self._generate_fingerprint(rule, current_value)
        
        if fingerprint in [alert.fingerprint for alert in self.active_alerts.values()]:
            logger.debug(f"Alert already exists for rule {rule.rule_id}, skipping duplicate")
            return
        
        # Check suppression rules
        if self._is_suppressed(rule, current_value, timestamp):
            logger.debug(f"Alert suppressed for rule {rule.rule_id}")
            return
        
        # Generate alert message
        message = rule.custom_message or f"{rule.description}. Current value: {current_value}, threshold: {rule.threshold}"
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            title=rule.name,
            message=message,
            level=rule.level,
            status=AlertStatus.ACTIVE,
            created_at=timestamp,
            updated_at=timestamp,
            component=self._extract_component_from_metric(rule.metric_path),
            metric_name=rule.metric_path.split('.')[-1],
            current_value=current_value,
            threshold_value=rule.threshold,
            fingerprint=fingerprint,
            metadata={
                'rule_description': rule.description,
                'operator': rule.operator,
                'tags': rule.tags
            }
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)
        
        logger.warning(f"Alert created: {alert.title} (ID: {alert.alert_id})")
    
    def _generate_fingerprint(self, rule: AlertRule, current_value: float) -> str:
        """Generate fingerprint for alert deduplication."""
        content = f"{rule.rule_id}_{rule.metric_path}_{rule.operator}_{rule.threshold}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_component_from_metric(self, metric_path: str) -> str:
        """Extract component name from metric path."""
        return metric_path.split('.')[0] if '.' in metric_path else 'system'
    
    def _is_suppressed(self, rule: AlertRule, current_value: float, timestamp: datetime) -> bool:
        """Check if alert should be suppressed."""
        # Check maintenance windows
        for window in self.maintenance_windows:
            if window['start'] <= timestamp <= window['end']:
                if not window.get('components') or rule.metric_path.startswith(tuple(window['components'])):
                    return True
        
        # Check suppression rules
        for suppression in self.suppression_rules:
            if suppression.get('rule_id') == rule.rule_id:
                if timestamp < suppression.get('until', datetime.min):
                    return True
        
        return False
    
    async def _send_notifications(self, alert: Alert, channels: List[NotificationChannel]):
        """Send notifications through specified channels."""
        notification_tasks = []
        
        for channel in channels:
            if channel in self.notification_handlers:
                handler = self.notification_handlers[channel]
                task = asyncio.create_task(self._send_single_notification(handler, alert, channel))
                notification_tasks.append(task)
        
        # Send notifications concurrently
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Track successful notifications
            for i, result in enumerate(results):
                if result is True:
                    alert.notifications_sent.append(channels[i].value)
                elif isinstance(result, Exception):
                    logger.error(f"Notification failed for {channels[i].value}: {result}")
    
    async def _send_single_notification(self, handler: NotificationHandler, alert: Alert, channel: NotificationChannel) -> bool:
        """Send a single notification."""
        try:
            return await handler.send_notification(alert)
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.value}: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user
        alert.updated_at = datetime.utcnow()
        
        logger.info(f"Alert acknowledged: {alert.title} by {user}")
        return True
    
    async def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert.title} by {user}")
        return True
    
    async def suppress_alerts(self, rule_id: str = None, component: str = None, 
                            duration: timedelta = timedelta(hours=1)):
        """Suppress alerts for specified duration."""
        suppression = {
            'rule_id': rule_id,
            'component': component,
            'until': datetime.utcnow() + duration,
            'created_at': datetime.utcnow()
        }
        
        self.suppression_rules.append(suppression)
        
        logger.info(f"Alerts suppressed: rule_id={rule_id}, component={component}, duration={duration}")
    
    def add_maintenance_window(self, start: datetime, end: datetime, 
                              components: List[str] = None, description: str = ""):
        """Add a maintenance window during which alerts are suppressed."""
        window = {
            'start': start,
            'end': end,
            'components': components,
            'description': description
        }
        
        self.maintenance_windows.append(window)
        
        logger.info(f"Maintenance window added: {start} to {end}")
    
    def get_active_alerts(self, level: AlertLevel = None, component: str = None) -> List[Dict[str, Any]]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if component:
            alerts = [alert for alert in alerts if alert.component == component]
        
        # Sort by level (highest first) and creation time
        alerts.sort(key=lambda x: (-x.level.value, x.created_at))
        
        return [self._alert_to_dict(alert) for alert in alerts]
    
    def get_alert_history(self, limit: int = 50, since: datetime = None) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        alerts = list(self.alert_history)
        
        if since:
            alerts = [alert for alert in alerts if alert.created_at >= since]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return [self._alert_to_dict(alert) for alert in alerts[:limit]]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_by_level = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_level[alert.level.name] += 1
        
        # Recent history stats (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [alert for alert in self.alert_history if alert.created_at >= recent_cutoff]
        
        recent_by_level = defaultdict(int)
        for alert in recent_alerts:
            recent_by_level[alert.level.name] += 1
        
        return {
            'active_alerts': {
                'total': len(self.active_alerts),
                'by_level': dict(active_by_level)
            },
            'recent_alerts_24h': {
                'total': len(recent_alerts),
                'by_level': dict(recent_by_level)
            },
            'alert_rules': {
                'total': len(self.alert_rules),
                'enabled': len([rule for rule in self.alert_rules.values() if rule.enabled]),
                'disabled': len([rule for rule in self.alert_rules.values() if not rule.enabled])
            },
            'notification_channels': list(self.notification_handlers.keys()),
            'maintenance_windows': len([w for w in self.maintenance_windows if w['end'] > datetime.utcnow()])
        }
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'title': alert.title,
            'message': alert.message,
            'level': alert.level.name,
            'status': alert.status.value,
            'component': alert.component,
            'metric_name': alert.metric_name,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'created_at': alert.created_at.isoformat(),
            'updated_at': alert.updated_at.isoformat(),
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'acknowledged_by': alert.acknowledged_by,
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'notifications_sent': alert.notifications_sent,
            'metadata': alert.metadata
        }
    
    async def start_processing(self):
        """Start background alert processing."""
        if self.processing_active:
            logger.warning("Alert processing is already active")
            return
        
        self.processing_active = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Alert processing started")
    
    async def stop_processing(self):
        """Stop background alert processing."""
        if not self.processing_active:
            return
        
        self.processing_active = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert processing stopped")
    
    async def _processing_loop(self):
        """Background processing loop for alert management."""
        while self.processing_active:
            try:
                # Clean up old suppression rules
                current_time = datetime.utcnow()
                self.suppression_rules = [
                    rule for rule in self.suppression_rules
                    if rule['until'] > current_time
                ]
                
                # Clean up old maintenance windows
                self.maintenance_windows = [
                    window for window in self.maintenance_windows
                    if window['end'] > current_time
                ]
                
                # Check for alerts that should auto-resolve
                await self._check_auto_resolution()
                
                # Clean up old metric data
                self._cleanup_metric_buffer()
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
            
            try:
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
    
    async def _check_auto_resolution(self):
        """Check if any alerts should be automatically resolved."""
        current_time = datetime.utcnow()
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve alerts that haven't been triggered recently
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                alerts_to_resolve.append(alert_id)
                continue
            
            # Check if recent metric values no longer trigger the rule
            recent_values = [
                entry for entry in self.metric_buffer[rule.metric_path]
                if current_time - entry['timestamp'] < timedelta(minutes=10)
            ]
            
            if recent_values:
                still_triggering = any(rule.evaluate(entry['value']) for entry in recent_values)
                if not still_triggering:
                    alerts_to_resolve.append(alert_id)
        
        # Resolve alerts
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id, "auto-resolution")
    
    def _cleanup_metric_buffer(self):
        """Clean up old metric data to prevent memory leaks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=2)
        
        for metric_path, data in self.metric_buffer.items():
            # Keep only recent data
            recent_data = deque([
                entry for entry in data
                if entry['timestamp'] > cutoff_time
            ], maxlen=data.maxlen)
            
            self.metric_buffer[metric_path] = recent_data

# Global alert manager instance
alert_manager = AlertManager()

# Convenience functions
async def create_alert_rule(name: str, metric_path: str, operator: str, 
                          threshold: float, level: AlertLevel, 
                          notification_channels: List[NotificationChannel] = None) -> str:
    """Create a new alert rule."""
    rule_id = f"custom_{name.lower().replace(' ', '_')}_{int(time.time())}"
    
    rule = AlertRule(
        rule_id=rule_id,
        name=name,
        description=f"Custom alert for {metric_path}",
        metric_path=metric_path,
        operator=operator,
        threshold=threshold,
        level=level,
        notification_channels=notification_channels or [NotificationChannel.EMAIL]
    )
    
    alert_manager.add_alert_rule(rule)
    return rule_id

async def record_performance_metrics():
    """Record performance metrics from various sources for alert evaluation."""
    try:
        # System resource metrics
        if hasattr(performance_collector, 'collect_system_metrics'):
            resource_metrics = performance_collector.collect_system_metrics()
            if resource_metrics:
                await alert_manager.record_metric("system.cpu_percent", resource_metrics.cpu_percent)
                await alert_manager.record_metric("system.memory_percent", resource_metrics.memory_percent)
                await alert_manager.record_metric("system.disk_percent", (resource_metrics.disk_io_read_mb + resource_metrics.disk_io_write_mb) / 1024)
        
        # API performance metrics
        endpoint_stats = performance_collector.get_endpoint_stats() if performance_collector else {}
        if endpoint_stats:
            await alert_manager.record_metric("api.response_time_p95", endpoint_stats.get("p95_response_time_ms", 0))
            await alert_manager.record_metric("api.error_rate", endpoint_stats.get("error_rate", 0))
        
        # Database metrics
        if hasattr(db_monitor, 'get_current_metrics'):
            db_metrics = db_monitor.get_current_metrics()
            for db_name, metrics in db_metrics.items():
                if hasattr(metrics, 'avg_query_duration'):
                    await alert_manager.record_metric(f"database.{db_name}.avg_query_time_ms", metrics.avg_query_duration * 1000)
        
    except Exception as e:
        logger.error(f"Failed to record performance metrics: {e}")

# Export key components
__all__ = [
    'AlertManager', 'AlertRule', 'Alert', 'AlertLevel', 'AlertStatus',
    'NotificationChannel', 'alert_manager', 'create_alert_rule', 
    'record_performance_metrics'
]