"""
SuperNova AI Security Logger
Comprehensive security event logging and monitoring system
"""

import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
import aiofiles
from cryptography.fernet import Fernet

from .security_config import security_settings, SECURITY_EVENT_TYPES, RISK_LEVELS


class SecurityEventLevel(str, Enum):
    """Security event severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    timestamp: str
    event_type: str
    level: SecurityEventLevel
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    risk_score: int = 1
    event_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID
            event_data = f"{self.timestamp}{self.event_type}{self.user_id or ''}{self.client_ip or ''}"
            self.event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]


class SecurityLogger:
    """
    Comprehensive security logging system with encryption, 
    structured logging, and SIEM integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger("security")
        self.logger.setLevel(getattr(logging, security_settings.SECURITY_LOG_LEVEL))
        
        # Setup file handler if not exists
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Encryption for sensitive logs
        self.cipher = security_settings.get_fernet_cipher() if security_settings.LOG_ENCRYPTION_ENABLED else None
        
        # Event buffer for batch processing
        self.event_buffer: List[SecurityEvent] = []
        self.buffer_size = 100
        
        # Statistics
        self.event_counts = {level.value: 0 for level in SecurityEventLevel}
        self.total_events = 0
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler for security events
        security_file_handler = logging.FileHandler(log_dir / "security.log")
        security_file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        security_file_handler.setFormatter(formatter)
        
        self.logger.addHandler(security_file_handler)
        
        # Separate handler for critical events
        critical_file_handler = logging.FileHandler(log_dir / "security_critical.log")
        critical_file_handler.setLevel(logging.CRITICAL)
        critical_file_handler.setFormatter(formatter)
        
        self.logger.addHandler(critical_file_handler)
        
        # Console handler for development
        if not security_settings.is_production():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_security_event(
        self,
        event_type: str,
        level: SecurityEventLevel = SecurityEventLevel.INFO,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """Log a security event"""
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, level, details)
        
        # Create security event
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            level=level,
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            details=details or {},
            risk_score=risk_score,
            correlation_id=correlation_id
        )
        
        # Log the event
        self._log_event(event)
        
        # Update statistics
        self.event_counts[level.value] += 1
        self.total_events += 1
        
        # Add to buffer for batch processing
        self.event_buffer.append(event)
        
        # Process buffer if full
        if len(self.event_buffer) >= self.buffer_size:
            asyncio.create_task(self._process_event_buffer())
        
        # Send alerts for high-risk events
        if risk_score >= RISK_LEVELS["HIGH"]:
            asyncio.create_task(self._send_security_alert(event))
    
    def _log_event(self, event: SecurityEvent):
        """Log event to configured handlers"""
        log_data = asdict(event)
        
        # Sanitize sensitive data
        if not security_settings.LOG_SENSITIVE_DATA:
            log_data = self._sanitize_log_data(log_data)
        
        # Encrypt log data if enabled
        if self.cipher and security_settings.LOG_ENCRYPTION_ENABLED:
            log_data = self._encrypt_log_data(log_data)
        
        log_message = json.dumps(log_data, default=str)
        
        # Log based on severity level
        if event.level == SecurityEventLevel.CRITICAL:
            self.logger.critical(log_message)
        elif event.level == SecurityEventLevel.ERROR:
            self.logger.error(log_message)
        elif event.level == SecurityEventLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _sanitize_log_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data from logs"""
        sensitive_fields = [
            'password', 'secret', 'key', 'token', 'api_key',
            'ssn', 'tax_id', 'bank_account', 'credit_card'
        ]
        
        def sanitize_dict(data):
            if isinstance(data, dict):
                sanitized = {}
                for key, value in data.items():
                    key_lower = key.lower()
                    if any(sensitive in key_lower for sensitive in sensitive_fields):
                        sanitized[key] = "***REDACTED***"
                    else:
                        sanitized[key] = sanitize_dict(value)
                return sanitized
            elif isinstance(data, list):
                return [sanitize_dict(item) for item in data]
            elif isinstance(data, str) and len(data) > 20:
                # Mask long strings that might contain sensitive data
                return data[:4] + "***" + data[-4:] if len(data) > 8 else "***"
            return data
        
        return sanitize_dict(log_data)
    
    def _encrypt_log_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive log data"""
        if not self.cipher:
            return log_data
        
        try:
            log_json = json.dumps(log_data, default=str)
            encrypted_data = self.cipher.encrypt(log_json.encode())
            
            return {
                "encrypted": True,
                "timestamp": log_data.get("timestamp"),
                "event_type": log_data.get("event_type"),
                "level": log_data.get("level"),
                "event_id": log_data.get("event_id"),
                "data": encrypted_data.decode()
            }
        except Exception as e:
            # Fallback to unencrypted logging if encryption fails
            self.logger.error(f"Log encryption failed: {e}")
            return log_data
    
    def _calculate_risk_score(
        self,
        event_type: str,
        level: SecurityEventLevel,
        details: Optional[Dict[str, Any]]
    ) -> int:
        """Calculate risk score for security event"""
        
        # Base score from level
        base_scores = {
            SecurityEventLevel.INFO: 1,
            SecurityEventLevel.WARNING: 2,
            SecurityEventLevel.ERROR: 3,
            SecurityEventLevel.CRITICAL: 4
        }
        
        score = base_scores.get(level, 1)
        
        # Event type modifiers
        high_risk_events = [
            SECURITY_EVENT_TYPES["AUTHENTICATION_FAILURE"],
            SECURITY_EVENT_TYPES["AUTHORIZATION_DENIED"],
            SECURITY_EVENT_TYPES["SUSPICIOUS_ACTIVITY"],
            SECURITY_EVENT_TYPES["SECURITY_VIOLATION"]
        ]
        
        if event_type in high_risk_events:
            score += 1
        
        # Details modifiers
        if details:
            if details.get("failed_attempts", 0) > 3:
                score += 1
            if details.get("admin_action"):
                score += 1
            if details.get("data_breach"):
                score = max(score, 4)  # Always critical
        
        return min(score, 4)  # Cap at critical level
    
    async def _process_event_buffer(self):
        """Process buffered events for batch operations"""
        if not self.event_buffer:
            return
        
        events_to_process = self.event_buffer.copy()
        self.event_buffer.clear()
        
        try:
            # Send to SIEM if configured
            if security_settings.SIEM_ENABLED:
                await self._send_to_siem(events_to_process)
            
            # Perform analytics
            await self._perform_security_analytics(events_to_process)
            
        except Exception as e:
            self.logger.error(f"Error processing event buffer: {e}")
    
    async def _send_to_siem(self, events: List[SecurityEvent]):
        """Send events to SIEM system"""
        if not security_settings.SIEM_ENDPOINT or not security_settings.SIEM_API_KEY:
            return
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {security_settings.SIEM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "events": [asdict(event) for event in events],
                "source": "supernova-ai",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    security_settings.SIEM_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"SIEM integration failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"SIEM integration error: {e}")
    
    async def _perform_security_analytics(self, events: List[SecurityEvent]):
        """Perform real-time security analytics"""
        try:
            # Detect patterns
            await self._detect_brute_force_attacks(events)
            await self._detect_suspicious_activities(events)
            await self._detect_privilege_escalation(events)
            
        except Exception as e:
            self.logger.error(f"Security analytics error: {e}")
    
    async def _detect_brute_force_attacks(self, events: List[SecurityEvent]):
        """Detect potential brute force attacks"""
        failed_logins = [
            event for event in events
            if event.event_type == SECURITY_EVENT_TYPES["AUTHENTICATION_FAILURE"]
        ]
        
        # Group by IP address
        ip_failures = {}
        for event in failed_logins:
            ip = event.client_ip
            if ip:
                ip_failures[ip] = ip_failures.get(ip, 0) + 1
        
        # Alert on high failure counts
        for ip, count in ip_failures.items():
            if count >= 10:  # Threshold for brute force detection
                await self._create_security_incident(
                    title="Potential Brute Force Attack",
                    description=f"Multiple failed login attempts from IP {ip}",
                    severity="HIGH",
                    details={"ip_address": ip, "failure_count": count}
                )
    
    async def _detect_suspicious_activities(self, events: List[SecurityEvent]):
        """Detect suspicious user activities"""
        # Check for unusual access patterns
        user_activities = {}
        
        for event in events:
            if event.user_id:
                if event.user_id not in user_activities:
                    user_activities[event.user_id] = []
                user_activities[event.user_id].append(event)
        
        # Analyze patterns for each user
        for user_id, user_events in user_activities.items():
            # Check for rapid succession of different event types
            if len(set(e.event_type for e in user_events)) > 5 and len(user_events) > 20:
                await self._create_security_incident(
                    title="Suspicious User Activity",
                    description=f"User {user_id} showing unusual activity patterns",
                    severity="MEDIUM",
                    details={"user_id": user_id, "event_count": len(user_events)}
                )
    
    async def _detect_privilege_escalation(self, events: List[SecurityEvent]):
        """Detect potential privilege escalation attempts"""
        admin_actions = [
            event for event in events
            if event.event_type == SECURITY_EVENT_TYPES["ADMIN_ACTION"] or
            event.details and event.details.get("admin_action")
        ]
        
        # Check for unusual admin activities
        admin_users = {}
        for event in admin_actions:
            if event.user_id:
                admin_users[event.user_id] = admin_users.get(event.user_id, 0) + 1
        
        for user_id, count in admin_users.items():
            if count > 50:  # Threshold for suspicious admin activity
                await self._create_security_incident(
                    title="Potential Privilege Escalation",
                    description=f"User {user_id} performing excessive admin actions",
                    severity="HIGH",
                    details={"user_id": user_id, "admin_action_count": count}
                )
    
    async def _send_security_alert(self, event: SecurityEvent):
        """Send immediate security alert for high-risk events"""
        if not security_settings.SECURITY_ALERTS_ENABLED:
            return
        
        try:
            alert_data = {
                "event": asdict(event),
                "alert_type": "security_event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": event.level.value,
                "risk_score": event.risk_score
            }
            
            # Send webhook alert
            if security_settings.ALERT_WEBHOOK_URL:
                await self._send_webhook_alert(alert_data)
            
            # Send email alert
            if security_settings.ALERT_EMAIL_ENABLED:
                await self._send_email_alert(alert_data)
        
        except Exception as e:
            self.logger.error(f"Alert sending failed: {e}")
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send webhook alert"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    security_settings.ALERT_WEBHOOK_URL,
                    json=alert_data,
                    timeout=10
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Webhook alert failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Webhook alert error: {e}")
    
    async def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert"""
        # Placeholder for email alert implementation
        # In production, integrate with email service (SendGrid, SES, etc.)
        pass
    
    async def _create_security_incident(
        self,
        title: str,
        description: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Create security incident for investigation"""
        incident_data = {
            "id": hashlib.sha256(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            "title": title,
            "description": description,
            "severity": severity,
            "status": "OPEN",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "details": details
        }
        
        # Log incident
        self.log_security_event(
            event_type="SECURITY_INCIDENT",
            level=SecurityEventLevel.CRITICAL,
            details=incident_data
        )
        
        # Store incident (in production, use database)
        incident_file = Path("logs") / f"incident_{incident_data['id']}.json"
        async with aiofiles.open(incident_file, "w") as f:
            await f.write(json.dumps(incident_data, indent=2))
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security logging statistics"""
        return {
            "total_events": self.total_events,
            "event_counts_by_level": self.event_counts,
            "buffer_size": len(self.event_buffer),
            "encryption_enabled": security_settings.LOG_ENCRYPTION_ENABLED,
            "siem_integration": security_settings.SIEM_ENABLED,
            "alert_system": security_settings.SECURITY_ALERTS_ENABLED
        }
    
    async def export_security_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """Export security logs for compliance or analysis"""
        # Placeholder for log export functionality
        # In production, implement log retrieval and export
        pass
    
    def cleanup_old_logs(self, retention_days: int = None):
        """Cleanup old log files based on retention policy"""
        if retention_days is None:
            retention_days = security_settings.AUDIT_LOG_RETENTION_DAYS
        
        log_dir = Path("logs")
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for log_file in log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                self.logger.info(f"Deleted old log file: {log_file.name}")


# Global security logger instance
security_logger = SecurityLogger()


# =====================================
# SECURITY EVENT DECORATORS
# =====================================

def log_security_event(event_type: str, level: SecurityEventLevel = SecurityEventLevel.INFO):
    """Decorator to automatically log security events"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                security_logger.log_security_event(
                    event_type=event_type,
                    level=level,
                    details={"function": func.__name__, "success": True}
                )
                return result
            except Exception as e:
                security_logger.log_security_event(
                    event_type=event_type,
                    level=SecurityEventLevel.ERROR,
                    details={"function": func.__name__, "error": str(e)}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                security_logger.log_security_event(
                    event_type=event_type,
                    level=level,
                    details={"function": func.__name__, "success": True}
                )
                return result
            except Exception as e:
                security_logger.log_security_event(
                    event_type=event_type,
                    level=SecurityEventLevel.ERROR,
                    details={"function": func.__name__, "error": str(e)}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator