"""
SuperNova AI - Secure Configuration Hot-Reloading System

This module provides secure configuration hot-reloading with:
- Real-time configuration file monitoring and updates
- Safety checks and validation before applying changes
- Rollback capabilities and change history
- Security controls and access restrictions
- Integration with configuration management and secrets systems
- Minimal downtime configuration updates
"""

from __future__ import annotations
import os
import sys
import json
import logging
import asyncio
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import traceback
from contextlib import asynccontextmanager
import tempfile
import shutil

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml
import aiofiles

from .config_management import config_manager, Environment, ConfigurationLevel, ConfigurationChangeEvent
from .config_validation_enhanced import validator, ValidationResult, ValidationSeverity
from .secrets_management import get_secrets_manager
from .environment_config import environment_manager

logger = logging.getLogger(__name__)


class ReloadTrigger(str, Enum):
    """Configuration reload triggers."""
    FILE_CHANGE = "file_change"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EXTERNAL_SIGNAL = "external_signal"
    HEALTH_CHECK = "health_check"


class ReloadStatus(str, Enum):
    """Configuration reload status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


@dataclass
class ReloadEvent:
    """Configuration reload event."""
    trigger: ReloadTrigger
    timestamp: datetime
    status: ReloadStatus
    changes: Dict[str, Any] = field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False
    processing_time: float = 0.0
    affected_services: Set[str] = field(default_factory=set)


@dataclass
class SafetyCheck:
    """Safety check configuration."""
    name: str
    enabled: bool = True
    critical: bool = False
    timeout: float = 5.0
    retry_count: int = 1
    check_function: Optional[Callable] = None


class ConfigurationBackup:
    """Manages configuration backups for rollback."""
    
    def __init__(self, backup_dir: Path, max_backups: int = 10):
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, config: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Create a configuration backup."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        backup_id = f"config_backup_{timestamp}"
        
        backup_data = {
            'backup_id': backup_id,
            'timestamp': datetime.utcnow().isoformat(),
            'config': config,
            'metadata': metadata or {}
        }
        
        backup_file = self.backup_dir / f"{backup_id}.json"
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            logger.info(f"Created configuration backup: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore configuration from backup."""
        backup_file = self.backup_dir / f"{backup_id}.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} not found")
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            logger.info(f"Restored configuration from backup: {backup_id}")
            return backup_data['config']
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("config_backup_*.json")):
            try:
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)
                
                backups.append({
                    'backup_id': backup_data['backup_id'],
                    'timestamp': backup_data['timestamp'],
                    'file_path': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'metadata': backup_data.get('metadata', {})
                })
            except Exception as e:
                logger.warning(f"Failed to read backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def _cleanup_old_backups(self):
        """Remove old backups beyond the limit."""
        backups = self.list_backups()
        
        if len(backups) > self.max_backups:
            for backup in backups[self.max_backups:]:
                try:
                    os.unlink(backup['file_path'])
                    logger.debug(f"Removed old backup: {backup['backup_id']}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup['backup_id']}: {e}")


class SafetyCheckManager:
    """Manages safety checks before configuration reloads."""
    
    def __init__(self):
        self.safety_checks: Dict[str, SafetyCheck] = {}
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default safety checks."""
        
        # Database connectivity check
        self.add_safety_check(SafetyCheck(
            name="database_connectivity",
            enabled=True,
            critical=True,
            timeout=10.0,
            check_function=self._check_database_connectivity
        ))
        
        # LLM provider connectivity check
        self.add_safety_check(SafetyCheck(
            name="llm_connectivity",
            enabled=True,
            critical=False,
            timeout=15.0,
            check_function=self._check_llm_connectivity
        ))
        
        # Cache connectivity check
        self.add_safety_check(SafetyCheck(
            name="cache_connectivity",
            enabled=True,
            critical=False,
            timeout=5.0,
            check_function=self._check_cache_connectivity
        ))
        
        # Memory usage check
        self.add_safety_check(SafetyCheck(
            name="memory_usage",
            enabled=True,
            critical=False,
            timeout=2.0,
            check_function=self._check_memory_usage
        ))
    
    def add_safety_check(self, check: SafetyCheck):
        """Add a safety check."""
        self.safety_checks[check.name] = check
        logger.debug(f"Added safety check: {check.name}")
    
    def remove_safety_check(self, name: str):
        """Remove a safety check."""
        if name in self.safety_checks:
            del self.safety_checks[name]
            logger.debug(f"Removed safety check: {name}")
    
    async def run_safety_checks(
        self,
        new_config: Dict[str, Any],
        skip_non_critical: bool = False
    ) -> Tuple[bool, List[str]]:
        """Run all enabled safety checks."""
        
        failed_checks = []
        critical_failed = False
        
        for check in self.safety_checks.values():
            if not check.enabled:
                continue
            
            if skip_non_critical and not check.critical:
                continue
            
            try:
                logger.debug(f"Running safety check: {check.name}")
                
                # Run the check with timeout
                success = await asyncio.wait_for(
                    self._run_single_check(check, new_config),
                    timeout=check.timeout
                )
                
                if not success:
                    failed_checks.append(check.name)
                    if check.critical:
                        critical_failed = True
                        logger.error(f"Critical safety check failed: {check.name}")
                    else:
                        logger.warning(f"Safety check failed: {check.name}")
                else:
                    logger.debug(f"Safety check passed: {check.name}")
            
            except asyncio.TimeoutError:
                failed_checks.append(f"{check.name} (timeout)")
                if check.critical:
                    critical_failed = True
                logger.error(f"Safety check timed out: {check.name}")
            
            except Exception as e:
                failed_checks.append(f"{check.name} (error: {str(e)})")
                if check.critical:
                    critical_failed = True
                logger.error(f"Safety check error in {check.name}: {e}")
        
        passed = not critical_failed
        return passed, failed_checks
    
    async def _run_single_check(self, check: SafetyCheck, new_config: Dict[str, Any]) -> bool:
        """Run a single safety check with retries."""
        
        for attempt in range(check.retry_count + 1):
            try:
                if check.check_function:
                    result = await check.check_function(new_config)
                    if result:
                        return True
                    
                    if attempt < check.retry_count:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    # No check function provided, assume pass
                    return True
            
            except Exception as e:
                if attempt == check.retry_count:
                    logger.error(f"Safety check {check.name} failed after {attempt + 1} attempts: {e}")
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return False
    
    async def _check_database_connectivity(self, config: Dict[str, Any]) -> bool:
        """Check database connectivity."""
        try:
            # Test database connection with new configuration
            database_url = config.get('database_url')
            if not database_url:
                return False
            
            # Use environment manager to test connectivity
            from sqlalchemy import create_engine, text
            
            engine = create_engine(database_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            return False
    
    async def _check_llm_connectivity(self, config: Dict[str, Any]) -> bool:
        """Check LLM provider connectivity."""
        try:
            llm_provider = config.get('llm_provider', '').lower()
            
            if llm_provider == 'openai':
                api_key = config.get('openai_api_key')
                if not api_key or 'test' in api_key.lower():
                    return False
                
                # Basic API key format check
                if not api_key.startswith('sk-'):
                    return False
            
            elif llm_provider == 'anthropic':
                api_key = config.get('anthropic_api_key')
                if not api_key or 'test' in api_key.lower():
                    return False
            
            # For now, just check basic configuration
            # In production, you might want to make a test API call
            return True
            
        except Exception as e:
            logger.error(f"LLM connectivity check failed: {e}")
            return False
    
    async def _check_cache_connectivity(self, config: Dict[str, Any]) -> bool:
        """Check cache connectivity."""
        try:
            cache_backend = config.get('cache_backend', 'memory')
            
            if cache_backend == 'redis':
                redis_url = config.get('redis_url')
                if not redis_url:
                    return False
                
                # Test Redis connectivity
                import redis
                redis_client = redis.from_url(redis_url)
                redis_client.ping()
                redis_client.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache connectivity check failed: {e}")
            return False
    
    async def _check_memory_usage(self, config: Dict[str, Any]) -> bool:
        """Check current memory usage."""
        try:
            import psutil
            
            # Check if memory usage is below 90%
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")
                return False
            
            return True
            
        except ImportError:
            # psutil not available, skip check
            return True
        except Exception as e:
            logger.error(f"Memory usage check failed: {e}")
            return False


class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.last_modified = {}
        self.debounce_time = 2.0  # seconds
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only watch configuration files
        if not self._is_config_file(file_path):
            return
        
        # Debounce rapid changes
        current_time = datetime.utcnow().timestamp()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < self.debounce_time:
                return
        
        self.last_modified[file_path] = current_time
        
        # Schedule reload
        asyncio.create_task(self.hot_reloader.handle_file_change(file_path))
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        config_extensions = {'.env', '.yaml', '.yml', '.json', '.toml', '.ini'}
        config_patterns = {'env', 'config', 'settings'}
        
        # Check extension
        if file_path.suffix.lower() in config_extensions:
            return True
        
        # Check filename patterns
        filename_lower = file_path.name.lower()
        if any(pattern in filename_lower for pattern in config_patterns):
            return True
        
        return False


class ConfigurationHotReloader:
    """Main configuration hot-reloading system."""
    
    def __init__(
        self,
        environment: Optional[Environment] = None,
        backup_dir: Optional[Path] = None,
        enabled: bool = True,
        validation_required: bool = True,
        safety_checks_enabled: bool = True
    ):
        self.environment = environment or Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        self.enabled = enabled and (self.environment != Environment.PRODUCTION)  # Disabled in production by default
        self.validation_required = validation_required
        self.safety_checks_enabled = safety_checks_enabled
        
        # Components
        self.backup_manager = ConfigurationBackup(
            backup_dir or Path('./config_backups'),
            max_backups=20
        )
        self.safety_check_manager = SafetyCheckManager()
        
        # State
        self.current_config: Dict[str, Any] = {}
        self.reload_history: List[ReloadEvent] = []
        self.max_history = 50
        
        # File watching
        self.observer: Optional[Observer] = None
        self.watcher: Optional[ConfigurationWatcher] = None
        
        # Change listeners
        self.change_listeners: List[Callable[[ReloadEvent], None]] = []
        
        # Thread safety
        self.reload_lock = threading.RLock()
        
        # Statistics
        self.reload_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(f"Configuration hot-reloader initialized for {self.environment.value} environment")
    
    def start(self, watch_paths: Optional[List[Path]] = None):
        """Start the hot-reloader."""
        if not self.enabled:
            logger.info("Hot-reloader is disabled")
            return
        
        if watch_paths is None:
            watch_paths = [
                Path('.'),  # Current directory for .env files
                Path('./config'),  # Config directory
                Path('./supernova')  # Source directory
            ]
        
        # Start file watcher
        self.observer = Observer()
        self.watcher = ConfigurationWatcher(self)
        
        for watch_path in watch_paths:
            if watch_path.exists():
                self.observer.schedule(self.watcher, str(watch_path), recursive=True)
                logger.info(f"Watching configuration path: {watch_path}")
        
        self.observer.start()
        logger.info("Configuration hot-reloader started")
    
    def stop(self):
        """Stop the hot-reloader."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        logger.info("Configuration hot-reloader stopped")
    
    def add_change_listener(self, listener: Callable[[ReloadEvent], None]):
        """Add a configuration change listener."""
        self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[ReloadEvent], None]):
        """Remove a configuration change listener."""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    async def handle_file_change(self, file_path: Path):
        """Handle a configuration file change."""
        try:
            logger.info(f"Configuration file changed: {file_path}")
            
            # Load the new configuration
            new_config = self._load_configuration_from_file(file_path)
            if not new_config:
                logger.warning(f"No configuration loaded from {file_path}")
                return
            
            # Trigger reload
            await self.reload_configuration(
                new_config=new_config,
                trigger=ReloadTrigger.FILE_CHANGE,
                source=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"Error handling file change for {file_path}: {e}")
    
    def _load_configuration_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from a file."""
        try:
            if not file_path.exists():
                return None
            
            config = {}
            
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    config.update(data)
            
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    config.update(data)
            
            elif file_path.name.startswith('.env') or 'env' in file_path.name.lower():
                # Load .env file
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip().strip('"').strip("'")
                            config[key] = value
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return None
    
    async def reload_configuration(
        self,
        new_config: Optional[Dict[str, Any]] = None,
        trigger: ReloadTrigger = ReloadTrigger.MANUAL,
        source: Optional[str] = None,
        force: bool = False
    ) -> ReloadEvent:
        """Reload configuration with safety checks."""
        
        start_time = datetime.utcnow()
        
        with self.reload_lock:
            self.reload_count += 1
            
            try:
                # Create reload event
                event = ReloadEvent(
                    trigger=trigger,
                    timestamp=start_time,
                    status=ReloadStatus.FAILED  # Default to failed, update on success
                )
                
                # Load current configuration if new config not provided
                if new_config is None:
                    new_config = self._load_current_configuration()
                
                # Skip if no changes detected
                if not force and self._is_configuration_unchanged(new_config):
                    event.status = ReloadStatus.SKIPPED
                    logger.info("Configuration unchanged, skipping reload")
                    self._add_to_history(event)
                    return event
                
                # Create backup of current configuration
                backup_id = None
                if self.current_config:
                    backup_id = self.backup_manager.create_backup(
                        self.current_config,
                        {
                            'trigger': trigger.value,
                            'source': source,
                            'timestamp': start_time.isoformat()
                        }
                    )
                
                # Validate new configuration
                if self.validation_required:
                    validation_result = await validator.validate_environment(
                        environment=self.environment,
                        config=new_config,
                        strict=self.environment == Environment.PRODUCTION
                    )
                    
                    event.validation_result = validation_result
                    
                    if not validation_result.passed:
                        event.error_message = f"Configuration validation failed: {len(validation_result.errors)} errors, {len(validation_result.critical)} critical issues"
                        logger.error(event.error_message)
                        self.failure_count += 1
                        self._add_to_history(event)
                        return event
                
                # Run safety checks
                if self.safety_checks_enabled:
                    safety_passed, failed_checks = await self.safety_check_manager.run_safety_checks(
                        new_config,
                        skip_non_critical=force
                    )
                    
                    if not safety_passed:
                        event.error_message = f"Safety checks failed: {', '.join(failed_checks)}"
                        logger.error(event.error_message)
                        self.failure_count += 1
                        self._add_to_history(event)
                        return event
                
                # Calculate changes
                changes = self._calculate_changes(self.current_config, new_config)
                event.changes = changes
                
                # Apply configuration changes
                success, affected_services = await self._apply_configuration_changes(new_config, changes)
                event.affected_services = affected_services
                
                if success:
                    # Update current configuration
                    self.current_config = new_config.copy()
                    event.status = ReloadStatus.SUCCESS
                    self.success_count += 1
                    
                    logger.info(f"Configuration reloaded successfully ({len(changes)} changes)")
                    
                    # Notify listeners
                    for listener in self.change_listeners:
                        try:
                            listener(event)
                        except Exception as e:
                            logger.error(f"Error in change listener: {e}")
                
                else:
                    # Rollback if backup exists
                    if backup_id:
                        try:
                            rollback_config = self.backup_manager.restore_backup(backup_id)
                            await self._apply_configuration_changes(rollback_config, {})
                            event.rollback_performed = True
                            event.status = ReloadStatus.ROLLED_BACK
                            logger.info("Configuration rolled back due to application failure")
                        except Exception as rollback_error:
                            logger.error(f"Rollback failed: {rollback_error}")
                    
                    event.error_message = "Failed to apply configuration changes"
                    self.failure_count += 1
                
            except Exception as e:
                event.error_message = f"Reload error: {str(e)}"
                logger.error(f"Configuration reload error: {e}")
                traceback.print_exc()
                self.failure_count += 1
            
            finally:
                # Calculate processing time
                event.processing_time = (datetime.utcnow() - start_time).total_seconds()
                self._add_to_history(event)
        
        return event
    
    def _load_current_configuration(self) -> Dict[str, Any]:
        """Load current configuration from all sources."""
        config = {}
        
        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith('SUPERNOVA_') or key.lower() in [
                'database_url', 'redis_url', 'log_level', 'debug',
                'openai_api_key', 'anthropic_api_key'
            ]:
                config[key.lower()] = value
        
        # Load from .env files
        env_files = [
            f".env.{self.environment.value}",
            f".env.{self.environment.value}.local",
            ".env.local",
            ".env"
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip().strip('"').strip("'")
                            if key not in config:  # Don't override env vars
                                config[key] = value
        
        return config
    
    def _is_configuration_unchanged(self, new_config: Dict[str, Any]) -> bool:
        """Check if configuration has actually changed."""
        if not self.current_config:
            return False
        
        # Compare configurations (excluding timestamps and volatile values)
        current_stable = {k: v for k, v in self.current_config.items() if k not in ['timestamp', 'pid', 'uptime']}
        new_stable = {k: v for k, v in new_config.items() if k not in ['timestamp', 'pid', 'uptime']}
        
        return current_stable == new_stable
    
    def _calculate_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate configuration changes."""
        changes = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added keys
        for key, value in new_config.items():
            if key not in old_config:
                changes['added'][key] = value
        
        # Find removed keys
        for key, value in old_config.items():
            if key not in new_config:
                changes['removed'][key] = value
        
        # Find modified keys
        for key, value in new_config.items():
            if key in old_config and old_config[key] != value:
                changes['modified'][key] = {
                    'old': old_config[key],
                    'new': value
                }
        
        return changes
    
    async def _apply_configuration_changes(
        self,
        new_config: Dict[str, Any],
        changes: Dict[str, Any]
    ) -> Tuple[bool, Set[str]]:
        """Apply configuration changes to the system."""
        affected_services = set()
        
        try:
            # Update configuration manager
            for key, value in new_config.items():
                # Determine configuration level based on key
                if any(pattern in key.lower() for pattern in ['secret', 'key', 'password', 'token']):
                    level = ConfigurationLevel.SECRET
                elif any(pattern in key.lower() for pattern in ['database', 'redis', 'host', 'url']):
                    level = ConfigurationLevel.CONFIDENTIAL
                else:
                    level = ConfigurationLevel.PUBLIC
                
                # Set configuration
                success = await config_manager.set_configuration(
                    key=key,
                    value=value,
                    level=level,
                    hot_reloadable=True,
                    description=f"Hot-reloaded configuration for {key}"
                )
                
                if not success:
                    logger.error(f"Failed to set configuration: {key}")
                    return False, affected_services
            
            # Determine affected services based on changes
            if changes.get('added') or changes.get('modified'):
                all_changed_keys = set(changes.get('added', {}).keys()) | set(changes.get('modified', {}).keys())
                
                # Map configuration keys to services
                service_mapping = {
                    'database': ['database', 'orm', 'migrations'],
                    'redis': ['cache', 'session', 'queue'],
                    'llm': ['llm_service', 'chat', 'advisor'],
                    'log': ['logging', 'monitoring'],
                    'security': ['auth', 'encryption', 'rate_limiting'],
                    'api': ['web_api', 'websocket']
                }
                
                for key in all_changed_keys:
                    for service_key, services in service_mapping.items():
                        if service_key in key:
                            affected_services.update(services)
            
            logger.info(f"Configuration changes applied successfully, affected services: {affected_services}")
            return True, affected_services
            
        except Exception as e:
            logger.error(f"Error applying configuration changes: {e}")
            return False, affected_services
    
    def _add_to_history(self, event: ReloadEvent):
        """Add reload event to history."""
        self.reload_history.append(event)
        
        # Trim history
        if len(self.reload_history) > self.max_history:
            self.reload_history.pop(0)
    
    async def manual_reload(self, force: bool = False) -> ReloadEvent:
        """Manually trigger configuration reload."""
        logger.info("Manual configuration reload triggered")
        
        return await self.reload_configuration(
            trigger=ReloadTrigger.MANUAL,
            force=force
        )
    
    async def rollback_to_backup(self, backup_id: str) -> ReloadEvent:
        """Rollback configuration to a specific backup."""
        try:
            backup_config = self.backup_manager.restore_backup(backup_id)
            
            return await self.reload_configuration(
                new_config=backup_config,
                trigger=ReloadTrigger.MANUAL,
                source=f"backup:{backup_id}",
                force=True
            )
            
        except Exception as e:
            logger.error(f"Failed to rollback to backup {backup_id}: {e}")
            raise
    
    def get_reload_history(self, limit: Optional[int] = None) -> List[ReloadEvent]:
        """Get reload history."""
        history = self.reload_history
        if limit:
            history = history[-limit:]
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hot-reload statistics."""
        return {
            'enabled': self.enabled,
            'environment': self.environment.value,
            'total_reloads': self.reload_count,
            'successful_reloads': self.success_count,
            'failed_reloads': self.failure_count,
            'success_rate': self.success_count / max(self.reload_count, 1),
            'current_config_keys': len(self.current_config),
            'backup_count': len(self.backup_manager.list_backups()),
            'safety_checks_enabled': self.safety_checks_enabled,
            'validation_required': self.validation_required,
            'watching_enabled': self.observer is not None and self.observer.is_alive(),
            'last_reload': self.reload_history[-1].timestamp.isoformat() if self.reload_history else None
        }


# Global hot-reloader instance
_hot_reloader: Optional[ConfigurationHotReloader] = None


def get_hot_reloader() -> ConfigurationHotReloader:
    """Get global hot-reloader instance."""
    global _hot_reloader
    
    if _hot_reloader is None:
        environment = Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        _hot_reloader = ConfigurationHotReloader(
            environment=environment,
            enabled=os.getenv('CONFIG_HOT_RELOAD_ENABLED', 'false').lower() == 'true'
        )
    
    return _hot_reloader


# Convenience functions
async def reload_configuration(force: bool = False) -> ReloadEvent:
    """Manually reload configuration."""
    reloader = get_hot_reloader()
    return await reloader.manual_reload(force=force)


async def rollback_configuration(backup_id: str) -> ReloadEvent:
    """Rollback configuration to backup."""
    reloader = get_hot_reloader()
    return await reloader.rollback_to_backup(backup_id)


def start_hot_reloader():
    """Start configuration hot-reloading."""
    reloader = get_hot_reloader()
    reloader.start()


def stop_hot_reloader():
    """Stop configuration hot-reloading."""
    reloader = get_hot_reloader()
    reloader.stop()


def get_reload_statistics() -> Dict[str, Any]:
    """Get hot-reload statistics."""
    reloader = get_hot_reloader()
    return reloader.get_statistics()


# Context manager for temporary configuration changes
@asynccontextmanager
async def temporary_config_override(**overrides):
    """Context manager for temporary configuration overrides."""
    reloader = get_hot_reloader()
    original_config = reloader.current_config.copy()
    
    try:
        # Apply temporary overrides
        temp_config = original_config.copy()
        temp_config.update(overrides)
        
        event = await reloader.reload_configuration(
            new_config=temp_config,
            trigger=ReloadTrigger.MANUAL,
            source="temporary_override"
        )
        
        if event.status != ReloadStatus.SUCCESS:
            raise RuntimeError(f"Failed to apply temporary configuration: {event.error_message}")
        
        yield event
    
    finally:
        # Restore original configuration
        await reloader.reload_configuration(
            new_config=original_config,
            trigger=ReloadTrigger.MANUAL,
            source="restore_from_temporary"
        )


if __name__ == "__main__":
    """CLI interface for hot-reloader management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI Configuration Hot-Reloader')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'reload', 'history', 'backups'])
    parser.add_argument('--force', action='store_true', help='Force reload even if validation fails')
    parser.add_argument('--backup-id', help='Backup ID for rollback')
    
    args = parser.parse_args()
    
    async def main():
        reloader = get_hot_reloader()
        
        if args.action == 'start':
            reloader.start()
            print("Hot-reloader started")
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                reloader.stop()
                print("Hot-reloader stopped")
        
        elif args.action == 'stop':
            reloader.stop()
            print("Hot-reloader stopped")
        
        elif args.action == 'status':
            stats = reloader.get_statistics()
            print(json.dumps(stats, indent=2))
        
        elif args.action == 'reload':
            event = await reloader.manual_reload(force=args.force)
            print(f"Reload {event.status.value}: {event.error_message or 'Success'}")
        
        elif args.action == 'history':
            history = reloader.get_reload_history(limit=10)
            for event in history:
                print(f"{event.timestamp} - {event.status.value} - {event.trigger.value}")
        
        elif args.action == 'backups':
            backups = reloader.backup_manager.list_backups()
            for backup in backups:
                print(f"{backup['backup_id']} - {backup['timestamp']}")
    
    asyncio.run(main())