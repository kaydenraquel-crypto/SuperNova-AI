"""
SuperNova AI - Configuration Monitoring and Drift Detection System

This module provides comprehensive configuration monitoring with:
- Real-time configuration drift detection and alerting
- Compliance monitoring and violation detection
- Performance impact analysis from configuration changes
- Automated remediation and rollback capabilities
- Configuration security monitoring and threat detection
- Integration with monitoring and alerting systems
"""

from __future__ import annotations
import os
import json
import logging
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import yaml
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

from .config_management import config_manager, Environment, ConfigurationLevel
from .config_validation_enhanced import validator, ValidationResult, ValidationSeverity
from .config_versioning import get_version_manager, ConfigurationVersion
from .config_hot_reload import get_hot_reloader
from .secrets_management import get_secrets_manager

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Configuration drift severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance monitoring status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


class AlertType(str, Enum):
    """Configuration alert types."""
    DRIFT_DETECTED = "drift_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_RISK = "security_risk"
    PERFORMANCE_IMPACT = "performance_impact"
    VALIDATION_FAILURE = "validation_failure"
    UNAUTHORIZED_CHANGE = "unauthorized_change"


@dataclass
class ConfigurationDrift:
    """Configuration drift detection result."""
    key: str
    expected_value: Any
    actual_value: Any
    drift_type: str  # added, removed, modified, type_changed
    severity: DriftSeverity
    detected_at: datetime
    environment: Environment
    source: str
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    remediation_suggested: bool = False
    auto_remediation_safe: bool = False


@dataclass
class ComplianceViolation:
    """Compliance violation detection result."""
    rule_id: str
    rule_name: str
    violation_type: str
    affected_keys: Set[str]
    severity: DriftSeverity
    status: ComplianceStatus
    detected_at: datetime
    environment: Environment
    description: str
    remediation_steps: List[str] = field(default_factory=list)
    compliance_framework: str = "general"
    risk_level: int = 1  # 1-5 scale


@dataclass
class SecurityAlert:
    """Security-related configuration alert."""
    alert_id: str
    alert_type: str
    affected_keys: Set[str]
    severity: DriftSeverity
    detected_at: datetime
    environment: Environment
    threat_indicators: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    automatic_remediation: bool = False


@dataclass
class PerformanceImpact:
    """Performance impact assessment."""
    metric_name: str
    baseline_value: float
    current_value: float
    impact_percentage: float
    threshold_breached: bool
    affected_services: Set[str]
    detected_at: datetime
    configuration_changes: List[str] = field(default_factory=list)


class ConfigurationBaseline:
    """Manages configuration baselines for drift detection."""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.baseline_metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def create_baseline(
        self,
        name: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new configuration baseline."""
        try:
            with self.lock:
                self.baselines[name] = config.copy()
                self.baseline_metadata[name] = {
                    'created_at': datetime.utcnow().isoformat(),
                    'environment': self.environment.value,
                    'config_hash': self._calculate_hash(config),
                    'key_count': len(config),
                    'metadata': metadata or {}
                }
                
                logger.info(f"Created baseline '{name}' for {self.environment.value} environment")
                return True
                
        except Exception as e:
            logger.error(f"Error creating baseline '{name}': {e}")
            return False
    
    def update_baseline(self, name: str, config: Dict[str, Any]) -> bool:
        """Update an existing baseline."""
        try:
            with self.lock:
                if name not in self.baselines:
                    return False
                
                old_hash = self.baseline_metadata[name]['config_hash']
                new_hash = self._calculate_hash(config)
                
                self.baselines[name] = config.copy()
                self.baseline_metadata[name].update({
                    'updated_at': datetime.utcnow().isoformat(),
                    'config_hash': new_hash,
                    'key_count': len(config),
                    'previous_hash': old_hash
                })
                
                logger.info(f"Updated baseline '{name}' for {self.environment.value} environment")
                return True
                
        except Exception as e:
            logger.error(f"Error updating baseline '{name}': {e}")
            return False
    
    def get_baseline(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a baseline configuration."""
        with self.lock:
            return self.baselines.get(name, {}).copy()
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all baselines with metadata."""
        with self.lock:
            return [
                {
                    'name': name,
                    'config': config.copy(),
                    'metadata': self.baseline_metadata.get(name, {})
                }
                for name, config in self.baselines.items()
            ]
    
    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class DriftDetector:
    """Detects configuration drift from baselines."""
    
    def __init__(self, baseline_manager: ConfigurationBaseline):
        self.baseline_manager = baseline_manager
        self.drift_thresholds = {
            'critical_keys': [
                'database_url', 'redis_url', 'secret_key',
                'encryption_key', 'jwt_secret_key'
            ],
            'high_impact_keys': [
                'llm_provider', 'llm_model', 'debug',
                'ssl_required', 'mfa_enabled'
            ],
            'security_sensitive_keys': [
                'secret', 'key', 'password', 'token',
                'cert', 'ssl', 'auth'
            ]
        }
    
    def detect_drift(
        self,
        current_config: Dict[str, Any],
        baseline_name: str = "production"
    ) -> List[ConfigurationDrift]:
        """Detect configuration drift from baseline."""
        
        baseline_config = self.baseline_manager.get_baseline(baseline_name)
        if not baseline_config:
            logger.warning(f"Baseline '{baseline_name}' not found for drift detection")
            return []
        
        drifts = []
        all_keys = set(baseline_config.keys()) | set(current_config.keys())
        
        for key in all_keys:
            baseline_value = baseline_config.get(key)
            current_value = current_config.get(key)
            
            drift = self._analyze_key_drift(
                key, baseline_value, current_value,
                self.baseline_manager.environment
            )
            
            if drift:
                drifts.append(drift)
        
        return drifts
    
    def _analyze_key_drift(
        self,
        key: str,
        baseline_value: Any,
        current_value: Any,
        environment: Environment
    ) -> Optional[ConfigurationDrift]:
        """Analyze drift for a specific configuration key."""
        
        if baseline_value == current_value:
            return None  # No drift
        
        # Determine drift type
        if baseline_value is None:
            drift_type = "added"
        elif current_value is None:
            drift_type = "removed"
        elif type(baseline_value) != type(current_value):
            drift_type = "type_changed"
        else:
            drift_type = "modified"
        
        # Determine severity
        severity = self._calculate_drift_severity(key, baseline_value, current_value)
        
        # Create drift object
        drift = ConfigurationDrift(
            key=key,
            expected_value=baseline_value,
            actual_value=current_value,
            drift_type=drift_type,
            severity=severity,
            detected_at=datetime.utcnow(),
            environment=environment,
            source="drift_detector"
        )
        
        # Assess impact
        drift.impact_assessment = self._assess_drift_impact(key, drift_type, baseline_value, current_value)
        
        # Determine remediation safety
        drift.auto_remediation_safe = self._is_auto_remediation_safe(key, drift_type, severity)
        drift.remediation_suggested = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        
        return drift
    
    def _calculate_drift_severity(self, key: str, old_value: Any, new_value: Any) -> DriftSeverity:
        """Calculate severity of configuration drift."""
        
        key_lower = key.lower()
        
        # Critical keys always get high severity
        if any(pattern in key_lower for pattern in self.drift_thresholds['critical_keys']):
            return DriftSeverity.CRITICAL
        
        # High impact keys
        if any(pattern in key_lower for pattern in self.drift_thresholds['high_impact_keys']):
            return DriftSeverity.HIGH
        
        # Security sensitive keys
        if any(pattern in key_lower for pattern in self.drift_thresholds['security_sensitive_keys']):
            return DriftSeverity.HIGH
        
        # Type changes are medium severity
        if old_value is not None and new_value is not None:
            if type(old_value) != type(new_value):
                return DriftSeverity.MEDIUM
        
        # Additions/removals are lower severity unless critical
        if old_value is None or new_value is None:
            return DriftSeverity.MEDIUM
        
        # Default to low severity
        return DriftSeverity.LOW
    
    def _assess_drift_impact(
        self,
        key: str,
        drift_type: str,
        old_value: Any,
        new_value: Any
    ) -> Dict[str, Any]:
        """Assess the impact of configuration drift."""
        
        impact = {
            'requires_restart': False,
            'affects_services': [],
            'security_risk': False,
            'performance_impact': False,
            'compliance_risk': False
        }
        
        key_lower = key.lower()
        
        # Service impact mapping
        service_mapping = {
            'database': ['database', 'orm', 'migrations'],
            'redis': ['cache', 'session', 'queue'],
            'llm': ['llm_service', 'chat', 'advisor'],
            'auth': ['authentication', 'authorization'],
            'log': ['logging', 'monitoring']
        }
        
        for pattern, services in service_mapping.items():
            if pattern in key_lower:
                impact['affects_services'].extend(services)
                break
        
        # Restart requirements
        restart_patterns = [
            'database_url', 'redis_url', 'port', 'host',
            'ssl_required', 'encryption_key'
        ]
        
        if any(pattern in key_lower for pattern in restart_patterns):
            impact['requires_restart'] = True
        
        # Security risk assessment
        security_patterns = [
            'secret', 'key', 'password', 'token', 'auth',
            'ssl', 'cert', 'encryption', 'mfa'
        ]
        
        if any(pattern in key_lower for pattern in security_patterns):
            impact['security_risk'] = True
        
        # Performance impact
        performance_patterns = [
            'pool_size', 'timeout', 'concurrent', 'cache',
            'retry', 'batch_size'
        ]
        
        if any(pattern in key_lower for pattern in performance_patterns):
            impact['performance_impact'] = True
        
        # Compliance risk
        compliance_patterns = [
            'audit', 'log', 'retention', 'gdpr', 'ccpa',
            'encryption', 'backup'
        ]
        
        if any(pattern in key_lower for pattern in compliance_patterns):
            impact['compliance_risk'] = True
        
        return impact
    
    def _is_auto_remediation_safe(
        self,
        key: str,
        drift_type: str,
        severity: DriftSeverity
    ) -> bool:
        """Determine if automatic remediation is safe."""
        
        # Never auto-remediate critical severity changes
        if severity == DriftSeverity.CRITICAL:
            return False
        
        # Never auto-remediate removals of important keys
        if drift_type == "removed":
            critical_keys = self.drift_thresholds['critical_keys']
            if any(pattern in key.lower() for pattern in critical_keys):
                return False
        
        # Type changes are generally not safe for auto-remediation
        if drift_type == "type_changed":
            return False
        
        # Low severity additions/modifications can be auto-remediated
        if severity == DriftSeverity.LOW and drift_type in ["added", "modified"]:
            return True
        
        return False


class ComplianceMonitor:
    """Monitors configuration compliance with security and regulatory requirements."""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for the environment."""
        
        rules = {}
        
        # GDPR Compliance Rules
        rules['gdpr_data_protection'] = {
            'framework': 'GDPR',
            'required_keys': [
                'gdpr_compliance_enabled',
                'data_anonymization_enabled',
                'right_to_be_forgotten_enabled',
                'consent_management_enabled'
            ],
            'required_values': {
                'gdpr_compliance_enabled': True,
                'data_anonymization_enabled': True,
                'right_to_be_forgotten_enabled': True,
                'consent_management_enabled': True
            },
            'environments': ['production', 'staging']
        }
        
        # Security Compliance Rules
        rules['security_hardening'] = {
            'framework': 'Security',
            'required_keys': [
                'ssl_required',
                'mfa_enabled',
                'audit_logging_enabled',
                'encryption_enabled'
            ],
            'required_values': {
                'ssl_required': True,
                'mfa_enabled': True,
                'audit_logging_enabled': True
            },
            'forbidden_values': {
                'debug': True,
                'db_echo': True
            },
            'environments': ['production']
        }
        
        # Data Retention Compliance
        rules['data_retention'] = {
            'framework': 'Data Governance',
            'required_keys': [
                'audit_log_retention_days',
                'user_data_retention_days',
                'backup_retention_days'
            ],
            'minimum_values': {
                'audit_log_retention_days': 365,  # 1 year minimum
                'user_data_retention_days': 30,
                'backup_retention_days': 30
            },
            'maximum_values': {
                'user_data_retention_days': 2555  # 7 years maximum
            },
            'environments': ['production', 'staging']
        }
        
        # Encryption Compliance
        rules['encryption_requirements'] = {
            'framework': 'Encryption',
            'required_keys': [
                'database_encryption_enabled',
                'backup_encryption_enabled',
                'field_encryption_enabled'
            ],
            'required_values': {
                'database_encryption_enabled': True,
                'backup_encryption_enabled': True,
                'field_encryption_enabled': True
            },
            'minimum_key_lengths': {
                'secret_key': 32,
                'jwt_secret_key': 32,
                'encryption_key': 32
            },
            'environments': ['production', 'staging']
        }
        
        return rules
    
    def check_compliance(self, config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check configuration compliance against all rules."""
        
        violations = []
        
        for rule_id, rule in self.compliance_rules.items():
            # Check if rule applies to current environment
            if self.environment.value not in rule.get('environments', []):
                continue
            
            rule_violations = self._check_rule_compliance(rule_id, rule, config)
            violations.extend(rule_violations)
        
        return violations
    
    def _check_rule_compliance(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        """Check compliance for a specific rule."""
        
        violations = []
        
        # Check required keys
        required_keys = rule.get('required_keys', [])
        missing_keys = set()
        
        for key in required_keys:
            if key not in config:
                missing_keys.add(key)
        
        if missing_keys:
            violations.append(ComplianceViolation(
                rule_id=rule_id,
                rule_name=rule.get('framework', 'Unknown'),
                violation_type='missing_required_keys',
                affected_keys=missing_keys,
                severity=DriftSeverity.HIGH,
                status=ComplianceStatus.NON_COMPLIANT,
                detected_at=datetime.utcnow(),
                environment=self.environment,
                description=f"Missing required configuration keys: {', '.join(missing_keys)}",
                remediation_steps=[f"Set {key} in configuration" for key in missing_keys],
                compliance_framework=rule.get('framework', 'General')
            ))
        
        # Check required values
        required_values = rule.get('required_values', {})
        incorrect_values = set()
        
        for key, expected_value in required_values.items():
            if key in config and config[key] != expected_value:
                incorrect_values.add(key)
        
        if incorrect_values:
            violations.append(ComplianceViolation(
                rule_id=rule_id,
                rule_name=rule.get('framework', 'Unknown'),
                violation_type='incorrect_required_values',
                affected_keys=incorrect_values,
                severity=DriftSeverity.HIGH,
                status=ComplianceStatus.NON_COMPLIANT,
                detected_at=datetime.utcnow(),
                environment=self.environment,
                description=f"Incorrect values for required keys: {', '.join(incorrect_values)}",
                remediation_steps=[
                    f"Set {key} to {required_values[key]}" 
                    for key in incorrect_values
                ],
                compliance_framework=rule.get('framework', 'General')
            ))
        
        # Check forbidden values
        forbidden_values = rule.get('forbidden_values', {})
        forbidden_violations = set()
        
        for key, forbidden_value in forbidden_values.items():
            if key in config and config[key] == forbidden_value:
                forbidden_violations.add(key)
        
        if forbidden_violations:
            violations.append(ComplianceViolation(
                rule_id=rule_id,
                rule_name=rule.get('framework', 'Unknown'),
                violation_type='forbidden_values_detected',
                affected_keys=forbidden_violations,
                severity=DriftSeverity.CRITICAL,
                status=ComplianceStatus.NON_COMPLIANT,
                detected_at=datetime.utcnow(),
                environment=self.environment,
                description=f"Forbidden values detected: {', '.join(forbidden_violations)}",
                remediation_steps=[
                    f"Change {key} from {forbidden_values[key]}" 
                    for key in forbidden_violations
                ],
                compliance_framework=rule.get('framework', 'General')
            ))
        
        # Check minimum values
        minimum_values = rule.get('minimum_values', {})
        below_minimum = set()
        
        for key, min_value in minimum_values.items():
            if key in config:
                try:
                    current_value = int(config[key]) if isinstance(config[key], str) else config[key]
                    if current_value < min_value:
                        below_minimum.add(key)
                except (ValueError, TypeError):
                    continue
        
        if below_minimum:
            violations.append(ComplianceViolation(
                rule_id=rule_id,
                rule_name=rule.get('framework', 'Unknown'),
                violation_type='below_minimum_values',
                affected_keys=below_minimum,
                severity=DriftSeverity.MEDIUM,
                status=ComplianceStatus.NON_COMPLIANT,
                detected_at=datetime.utcnow(),
                environment=self.environment,
                description=f"Values below minimum requirements: {', '.join(below_minimum)}",
                remediation_steps=[
                    f"Increase {key} to at least {minimum_values[key]}"
                    for key in below_minimum
                ],
                compliance_framework=rule.get('framework', 'General')
            ))
        
        # Check key lengths (for security keys)
        minimum_key_lengths = rule.get('minimum_key_lengths', {})
        short_keys = set()
        
        for key, min_length in minimum_key_lengths.items():
            if key in config:
                key_value = str(config[key])
                if len(key_value) < min_length:
                    short_keys.add(key)
        
        if short_keys:
            violations.append(ComplianceViolation(
                rule_id=rule_id,
                rule_name=rule.get('framework', 'Unknown'),
                violation_type='insufficient_key_length',
                affected_keys=short_keys,
                severity=DriftSeverity.CRITICAL,
                status=ComplianceStatus.NON_COMPLIANT,
                detected_at=datetime.utcnow(),
                environment=self.environment,
                description=f"Keys below minimum length requirements: {', '.join(short_keys)}",
                remediation_steps=[
                    f"Generate new {key} with at least {minimum_key_lengths[key]} characters"
                    for key in short_keys
                ],
                compliance_framework=rule.get('framework', 'Security')
            ))
        
        return violations


class ConfigurationMonitor:
    """Main configuration monitoring system."""
    
    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment or Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        
        # Components
        self.baseline_manager = ConfigurationBaseline(self.environment)
        self.drift_detector = DriftDetector(self.baseline_manager)
        self.compliance_monitor = ComplianceMonitor(self.environment)
        
        # State
        self.monitoring_active = False
        self.monitoring_interval = 300  # 5 minutes
        self.last_check_time: Optional[datetime] = None
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Metrics
        self.setup_metrics()
        
        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Configuration monitor initialized for {self.environment.value} environment")
    
    def setup_metrics(self):
        """Setup Prometheus metrics for configuration monitoring."""
        
        self.metrics = {
            'drift_detections_total': Counter(
                'config_drift_detections_total',
                'Total number of configuration drift detections',
                ['environment', 'severity', 'drift_type']
            ),
            'compliance_violations_total': Counter(
                'config_compliance_violations_total',
                'Total number of compliance violations',
                ['environment', 'framework', 'violation_type']
            ),
            'monitoring_checks_total': Counter(
                'config_monitoring_checks_total',
                'Total number of monitoring checks performed',
                ['environment', 'check_type']
            ),
            'monitoring_check_duration': Histogram(
                'config_monitoring_check_duration_seconds',
                'Duration of monitoring checks',
                ['environment', 'check_type']
            ),
            'active_drifts': Gauge(
                'config_active_drifts',
                'Number of active configuration drifts',
                ['environment', 'severity']
            ),
            'compliance_score': Gauge(
                'config_compliance_score',
                'Configuration compliance score (0-1)',
                ['environment', 'framework']
            )
        }
    
    async def start_monitoring(self, interval: Optional[int] = None):
        """Start continuous configuration monitoring."""
        
        if self.monitoring_active:
            logger.warning("Configuration monitoring is already active")
            return
        
        if interval:
            self.monitoring_interval = interval
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Started configuration monitoring with {self.monitoring_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop configuration monitoring."""
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped configuration monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                await self.run_monitoring_checks()
                await asyncio.sleep(self.monitoring_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(min(self.monitoring_interval, 60))  # Fallback interval
    
    async def run_monitoring_checks(self):
        """Run all monitoring checks."""
        
        start_time = time.time()
        
        try:
            # Get current configuration
            current_config = await self._get_current_configuration()
            
            # Run drift detection
            drifts = await self._run_drift_detection(current_config)
            
            # Run compliance checks
            violations = await self._run_compliance_checks(current_config)
            
            # Process results
            await self._process_monitoring_results(drifts, violations)
            
            # Update metrics
            self._update_metrics(drifts, violations)
            
            self.last_check_time = datetime.utcnow()
            
            check_duration = time.time() - start_time
            self.metrics['monitoring_check_duration'].labels(
                environment=self.environment.value,
                check_type='full_check'
            ).observe(check_duration)
            
            self.metrics['monitoring_checks_total'].labels(
                environment=self.environment.value,
                check_type='full_check'
            ).inc()
            
            logger.debug(f"Monitoring checks completed in {check_duration:.2f}s")
        
        except Exception as e:
            logger.error(f"Error running monitoring checks: {e}")
            traceback.print_exc()
    
    async def _get_current_configuration(self) -> Dict[str, Any]:
        """Get current system configuration."""
        
        # Get configuration from config manager
        config = {}
        
        # Load from environment variables and config files
        for key, value in os.environ.items():
            if key.startswith('SUPERNOVA_') or key.lower() in [
                'debug', 'log_level', 'database_url', 'redis_url',
                'ssl_required', 'mfa_enabled', 'audit_logging_enabled'
            ]:
                config[key.lower()] = value
        
        return config
    
    async def _run_drift_detection(self, config: Dict[str, Any]) -> List[ConfigurationDrift]:
        """Run drift detection checks."""
        
        try:
            drifts = self.drift_detector.detect_drift(config)
            
            # Log significant drifts
            for drift in drifts:
                if drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    logger.warning(
                        f"Configuration drift detected: {drift.key} "
                        f"({drift.drift_type}, {drift.severity.value})"
                    )
            
            return drifts
        
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return []
    
    async def _run_compliance_checks(self, config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Run compliance monitoring checks."""
        
        try:
            violations = self.compliance_monitor.check_compliance(config)
            
            # Log violations
            for violation in violations:
                if violation.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    logger.error(
                        f"Compliance violation: {violation.rule_name} "
                        f"({violation.violation_type}, {violation.severity.value})"
                    )
            
            return violations
        
        except Exception as e:
            logger.error(f"Error in compliance checks: {e}")
            return []
    
    async def _process_monitoring_results(
        self,
        drifts: List[ConfigurationDrift],
        violations: List[ComplianceViolation]
    ):
        """Process monitoring results and trigger alerts."""
        
        # Process drifts
        for drift in drifts:
            await self._handle_configuration_drift(drift)
        
        # Process compliance violations
        for violation in violations:
            await self._handle_compliance_violation(violation)
    
    async def _handle_configuration_drift(self, drift: ConfigurationDrift):
        """Handle detected configuration drift."""
        
        # Create alert
        alert_data = {
            'type': AlertType.DRIFT_DETECTED.value,
            'severity': drift.severity.value,
            'key': drift.key,
            'expected_value': drift.expected_value,
            'actual_value': drift.actual_value,
            'drift_type': drift.drift_type,
            'environment': drift.environment.value,
            'detected_at': drift.detected_at.isoformat(),
            'impact_assessment': drift.impact_assessment,
            'auto_remediation_safe': drift.auto_remediation_safe
        }
        
        # Send alert
        await self._send_alert(alert_data)
        
        # Auto-remediation if safe and enabled
        if drift.auto_remediation_safe and self._is_auto_remediation_enabled():
            await self._attempt_auto_remediation(drift)
    
    async def _handle_compliance_violation(self, violation: ComplianceViolation):
        """Handle detected compliance violation."""
        
        # Create alert
        alert_data = {
            'type': AlertType.COMPLIANCE_VIOLATION.value,
            'severity': violation.severity.value,
            'rule_id': violation.rule_id,
            'rule_name': violation.rule_name,
            'violation_type': violation.violation_type,
            'affected_keys': list(violation.affected_keys),
            'environment': violation.environment.value,
            'detected_at': violation.detected_at.isoformat(),
            'compliance_framework': violation.compliance_framework,
            'remediation_steps': violation.remediation_steps
        }
        
        # Send alert
        await self._send_alert(alert_data)
    
    async def _send_alert(self, alert_data: Dict[str, Any]):
        """Send configuration alert."""
        
        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Log alert
        logger.info(f"Configuration alert: {alert_data['type']} ({alert_data['severity']})")
    
    def _update_metrics(
        self,
        drifts: List[ConfigurationDrift],
        violations: List[ComplianceViolation]
    ):
        """Update Prometheus metrics."""
        
        # Update drift metrics
        drift_counts = {}
        for drift in drifts:
            key = (drift.severity.value, drift.drift_type)
            drift_counts[key] = drift_counts.get(key, 0) + 1
            
            self.metrics['drift_detections_total'].labels(
                environment=self.environment.value,
                severity=drift.severity.value,
                drift_type=drift.drift_type
            ).inc()
        
        # Update active drifts gauge
        for severity in DriftSeverity:
            count = sum(1 for drift in drifts if drift.severity == severity)
            self.metrics['active_drifts'].labels(
                environment=self.environment.value,
                severity=severity.value
            ).set(count)
        
        # Update compliance metrics
        violation_counts = {}
        frameworks = set()
        
        for violation in violations:
            key = (violation.compliance_framework, violation.violation_type)
            violation_counts[key] = violation_counts.get(key, 0) + 1
            frameworks.add(violation.compliance_framework)
            
            self.metrics['compliance_violations_total'].labels(
                environment=self.environment.value,
                framework=violation.compliance_framework,
                violation_type=violation.violation_type
            ).inc()
        
        # Calculate compliance scores
        for framework in frameworks:
            framework_violations = [v for v in violations if v.compliance_framework == framework]
            total_rules = len(self.compliance_monitor.compliance_rules)
            violated_rules = len(set(v.rule_id for v in framework_violations))
            
            compliance_score = max(0, (total_rules - violated_rules) / max(total_rules, 1))
            
            self.metrics['compliance_score'].labels(
                environment=self.environment.value,
                framework=framework
            ).set(compliance_score)
    
    def _is_auto_remediation_enabled(self) -> bool:
        """Check if auto-remediation is enabled."""
        return os.getenv('CONFIG_AUTO_REMEDIATION_ENABLED', 'false').lower() == 'true'
    
    async def _attempt_auto_remediation(self, drift: ConfigurationDrift):
        """Attempt automatic remediation of configuration drift."""
        
        try:
            logger.info(f"Attempting auto-remediation for drift: {drift.key}")
            
            # Use hot-reloader to apply baseline value
            hot_reloader = get_hot_reloader()
            
            baseline_config = self.baseline_manager.get_baseline("production")
            if not baseline_config:
                logger.warning("No baseline available for auto-remediation")
                return
            
            # Create temporary configuration with corrected value
            temp_config = await self._get_current_configuration()
            temp_config[drift.key] = drift.expected_value
            
            # Apply the remediation
            event = await hot_reloader.reload_configuration(
                new_config=temp_config,
                trigger="auto_remediation",
                source=f"drift_remediation_{drift.key}"
            )
            
            if event.status == "success":
                logger.info(f"Auto-remediation successful for {drift.key}")
            else:
                logger.error(f"Auto-remediation failed for {drift.key}: {event.error_message}")
        
        except Exception as e:
            logger.error(f"Error in auto-remediation for {drift.key}: {e}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Remove an alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.monitoring_active,
            'environment': self.environment.value,
            'monitoring_interval': self.monitoring_interval,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'baselines_count': len(self.baseline_manager.baselines),
            'alert_handlers_count': len(self.alert_handlers),
            'auto_remediation_enabled': self._is_auto_remediation_enabled()
        }


# Global monitor instance
_monitor: Optional[ConfigurationMonitor] = None


def get_configuration_monitor() -> ConfigurationMonitor:
    """Get global configuration monitor instance."""
    global _monitor
    
    if _monitor is None:
        environment = Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        _monitor = ConfigurationMonitor(environment)
    
    return _monitor


# Convenience functions
async def start_configuration_monitoring(interval: int = 300):
    """Start configuration monitoring."""
    monitor = get_configuration_monitor()
    await monitor.start_monitoring(interval)


async def stop_configuration_monitoring():
    """Stop configuration monitoring."""
    monitor = get_configuration_monitor()
    await monitor.stop_monitoring()


def add_configuration_alert_handler(handler: Callable[[Dict[str, Any]], None]):
    """Add configuration alert handler."""
    monitor = get_configuration_monitor()
    monitor.add_alert_handler(handler)


def create_configuration_baseline(name: str, config: Optional[Dict[str, Any]] = None):
    """Create configuration baseline."""
    monitor = get_configuration_monitor()
    
    if config is None:
        # Use current configuration
        import asyncio
        config = asyncio.run(monitor._get_current_configuration())
    
    return monitor.baseline_manager.create_baseline(name, config)


if __name__ == "__main__":
    """CLI interface for configuration monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI Configuration Monitor')
    parser.add_argument('action', choices=['start', 'status', 'baseline', 'check'])
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval in seconds')
    parser.add_argument('--baseline-name', default='production', help='Baseline name')
    
    args = parser.parse_args()
    
    async def main():
        monitor = get_configuration_monitor()
        
        if args.action == 'start':
            print(f"Starting configuration monitoring (interval: {args.interval}s)")
            await monitor.start_monitoring(args.interval)
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await monitor.stop_monitoring()
                print("Monitoring stopped")
        
        elif args.action == 'status':
            status = monitor.get_monitoring_status()
            print(json.dumps(status, indent=2))
        
        elif args.action == 'baseline':
            current_config = await monitor._get_current_configuration()
            success = monitor.baseline_manager.create_baseline(args.baseline_name, current_config)
            print(f"Baseline creation: {'Success' if success else 'Failed'}")
        
        elif args.action == 'check':
            current_config = await monitor._get_current_configuration()
            
            drifts = monitor.drift_detector.detect_drift(current_config, args.baseline_name)
            violations = monitor.compliance_monitor.check_compliance(current_config)
            
            print(f"Configuration Check Results:")
            print(f"Drifts detected: {len(drifts)}")
            print(f"Compliance violations: {len(violations)}")
            
            for drift in drifts:
                print(f"  DRIFT: {drift.key} ({drift.severity.value})")
            
            for violation in violations:
                print(f"  VIOLATION: {violation.rule_name} ({violation.severity.value})")
    
    asyncio.run(main())