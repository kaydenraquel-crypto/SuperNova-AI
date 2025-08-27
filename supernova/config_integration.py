"""
SuperNova AI - Configuration System Integration Hub

This module provides the main integration point for all configuration components:
- Configuration management and environment handling
- Configuration validation and type checking
- Configuration hot-reloading with safety checks
- Configuration versioning and rollback capabilities
- Configuration monitoring and drift detection
- Secrets management integration
- Security framework integration
- Deployment configuration management
"""

from __future__ import annotations
import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import traceback

from .config_management import config_manager, Environment, ConfigurationLevel
from .config_validation_enhanced import validator, ValidationResult, ValidationSeverity
from .config_hot_reload import get_hot_reloader, ConfigurationHotReloader, ReloadEvent
from .config_versioning import get_version_manager, ConfigurationVersionManager, VersionType
from .config_monitoring import get_configuration_monitor, ConfigurationMonitor
from .secrets_management import get_secrets_manager, SecretsManager
from .environment_config import environment_manager

logger = logging.getLogger(__name__)


class ConfigurationSystemStatus(str, Enum):
    """Overall system status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    MAINTENANCE = "maintenance"


@dataclass
class SystemHealthCheck:
    """System health check result."""
    component: str
    status: ConfigurationSystemStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationSystemInfo:
    """Configuration system information."""
    environment: Environment
    status: ConfigurationSystemStatus
    components: Dict[str, Dict[str, Any]]
    health_checks: List[SystemHealthCheck]
    metrics: Dict[str, Any]
    uptime: timedelta
    last_validation: Optional[datetime] = None
    last_reload: Optional[datetime] = None
    last_version_check: Optional[datetime] = None


class ConfigurationSystemIntegrator:
    """Main configuration system integrator and coordinator."""
    
    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment or Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        self.start_time = datetime.utcnow()
        
        # Component references
        self._config_manager = config_manager
        self._validator = validator
        self._hot_reloader: Optional[ConfigurationHotReloader] = None
        self._version_manager: Optional[ConfigurationVersionManager] = None
        self._monitor: Optional[ConfigurationMonitor] = None
        self._secrets_manager: Optional[SecretsManager] = None
        
        # System state
        self._initialized = False
        self._status = ConfigurationSystemStatus.INITIALIZING
        self._health_checks: List[SystemHealthCheck] = []
        
        # Integration callbacks
        self._integration_callbacks: List[Callable[[str, Any], None]] = []
        
        logger.info(f"Configuration system integrator created for {self.environment.value} environment")
    
    async def initialize(
        self,
        enable_hot_reload: bool = True,
        enable_monitoring: bool = True,
        enable_versioning: bool = True,
        enable_secrets: bool = True
    ) -> bool:
        """Initialize all configuration system components."""
        
        try:
            logger.info("Initializing configuration system components...")
            
            # Initialize secrets management first (other components may need it)
            if enable_secrets:
                self._secrets_manager = get_secrets_manager()
                logger.info("✓ Secrets management initialized")
            
            # Initialize version management
            if enable_versioning:
                self._version_manager = get_version_manager()
                logger.info("✓ Version management initialized")
            
            # Initialize hot reloader
            if enable_hot_reload:
                self._hot_reloader = get_hot_reloader()
                
                # Register integration callback for hot reload events
                self._hot_reloader.add_change_listener(self._handle_reload_event)
                logger.info("✓ Hot reloader initialized")
            
            # Initialize monitoring
            if enable_monitoring:
                self._monitor = get_configuration_monitor()
                
                # Register integration callback for monitoring alerts
                self._monitor.add_alert_handler(self._handle_monitoring_alert)
                logger.info("✓ Configuration monitoring initialized")
            
            # Run initial validation
            await self._run_initial_validation()
            
            # Create initial baseline if monitoring enabled
            if self._monitor:
                await self._create_initial_baseline()
            
            # Start monitoring if enabled
            if self._monitor:
                await self._monitor.start_monitoring()
                logger.info("✓ Configuration monitoring started")
            
            # Start hot reloader file watching if enabled
            if self._hot_reloader:
                self._hot_reloader.start()
                logger.info("✓ Hot reloader file watching started")
            
            self._status = ConfigurationSystemStatus.HEALTHY
            self._initialized = True
            
            logger.info("✅ Configuration system initialization completed successfully")
            
            # Trigger integration callbacks
            for callback in self._integration_callbacks:
                try:
                    callback("system_initialized", self.get_system_info())
                except Exception as e:
                    logger.error(f"Error in integration callback: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize configuration system: {e}")
            traceback.print_exc()
            self._status = ConfigurationSystemStatus.UNHEALTHY
            return False
    
    async def shutdown(self):
        """Shutdown all configuration system components."""
        
        try:
            logger.info("Shutting down configuration system...")
            
            # Stop monitoring
            if self._monitor:
                await self._monitor.stop_monitoring()
                logger.info("✓ Configuration monitoring stopped")
            
            # Stop hot reloader
            if self._hot_reloader:
                self._hot_reloader.stop()
                logger.info("✓ Hot reloader stopped")
            
            # Shutdown secrets manager
            if self._secrets_manager:
                self._secrets_manager.shutdown()
                logger.info("✓ Secrets manager shutdown")
            
            self._status = ConfigurationSystemStatus.MAINTENANCE
            self._initialized = False
            
            logger.info("✅ Configuration system shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during configuration system shutdown: {e}")
    
    async def _run_initial_validation(self):
        """Run initial configuration validation."""
        try:
            # Get current configuration
            current_config = self._get_current_configuration()
            
            # Validate configuration
            result = await self._validator.validate_environment(
                environment=self.environment,
                config=current_config,
                strict=self.environment == Environment.PRODUCTION
            )
            
            if result.passed:
                logger.info("✅ Initial configuration validation passed")
            else:
                logger.warning(f"⚠️ Initial configuration validation issues: {len(result.errors)} errors, {len(result.warnings)} warnings")
                
                # Log critical issues
                for issue in result.critical:
                    logger.critical(f"Critical configuration issue: {issue.key} - {issue.message}")
                
                # Set status to degraded if there are errors but not critical
                if result.critical:
                    self._status = ConfigurationSystemStatus.UNHEALTHY
                elif result.errors:
                    self._status = ConfigurationSystemStatus.DEGRADED
        
        except Exception as e:
            logger.error(f"Error in initial validation: {e}")
            self._status = ConfigurationSystemStatus.UNHEALTHY
    
    async def _create_initial_baseline(self):
        """Create initial configuration baseline for monitoring."""
        try:
            current_config = self._get_current_configuration()
            
            # Create baseline with environment-specific name
            baseline_name = f"{self.environment.value}_baseline_{datetime.utcnow().strftime('%Y%m%d')}"
            
            success = self._monitor.baseline_manager.create_baseline(
                name=baseline_name,
                config=current_config,
                metadata={
                    'created_by': 'system_integrator',
                    'environment': self.environment.value,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            if success:
                logger.info(f"✅ Created initial baseline: {baseline_name}")
            else:
                logger.warning("⚠️ Failed to create initial baseline")
        
        except Exception as e:
            logger.error(f"Error creating initial baseline: {e}")
    
    def _get_current_configuration(self) -> Dict[str, Any]:
        """Get current system configuration."""
        config = {}
        
        # Load from environment variables
        for key, value in os.environ.items():
            if (key.startswith('SUPERNOVA_') or 
                key.lower() in ['debug', 'log_level', 'database_url', 'redis_url']):
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
                try:
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip().lower()
                                value = value.strip().strip('"').strip("'")
                                if key not in config:  # Don't override env vars
                                    config[key] = value
                except Exception as e:
                    logger.warning(f"Error reading {env_file}: {e}")
        
        return config
    
    def _handle_reload_event(self, event: ReloadEvent):
        """Handle configuration reload events."""
        try:
            logger.info(f"Configuration reload event: {event.status.value} ({event.trigger.value})")
            
            # Update system status based on reload result
            if event.status.value == 'failed':
                self._status = ConfigurationSystemStatus.DEGRADED
            elif event.status.value == 'success':
                # Keep current status or improve to healthy
                if self._status == ConfigurationSystemStatus.DEGRADED:
                    self._status = ConfigurationSystemStatus.HEALTHY
            
            # Create version if reload was successful and versioning is enabled
            if (event.status.value == 'success' and 
                self._version_manager and 
                len(event.changes) > 0):
                
                asyncio.create_task(self._create_version_for_reload(event))
            
            # Trigger integration callbacks
            for callback in self._integration_callbacks:
                try:
                    callback("configuration_reloaded", {
                        'event': event,
                        'system_status': self._status.value
                    })
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
        
        except Exception as e:
            logger.error(f"Error handling reload event: {e}")
    
    async def _create_version_for_reload(self, event: ReloadEvent):
        """Create configuration version for successful reload."""
        try:
            if not self._version_manager:
                return
            
            current_config = self._get_current_configuration()
            
            # Determine version type based on changes
            version_type = VersionType.PATCH
            if len(event.changes.get('added', {})) > 5 or len(event.changes.get('removed', {})) > 2:
                version_type = VersionType.MINOR
            
            # Create version
            version = await self._version_manager.create_version(
                config=current_config,
                message=f"Hot reload: {event.trigger.value} ({len(event.changes)} changes)",
                version_type=version_type,
                author="system",
                tags={'hot-reload', event.trigger.value},
                validate=False  # Already validated during reload
            )
            
            if version:
                logger.info(f"✅ Created version {version.version} for hot reload")
            
        except Exception as e:
            logger.error(f"Error creating version for reload: {e}")
    
    def _handle_monitoring_alert(self, alert_data: Dict[str, Any]):
        """Handle configuration monitoring alerts."""
        try:
            alert_type = alert_data.get('type', 'unknown')
            severity = alert_data.get('severity', 'low')
            
            logger.warning(f"Configuration monitoring alert: {alert_type} ({severity})")
            
            # Update system status based on alert severity
            if severity in ['critical', 'high']:
                self._status = ConfigurationSystemStatus.DEGRADED
            
            # Trigger integration callbacks
            for callback in self._integration_callbacks:
                try:
                    callback("monitoring_alert", alert_data)
                except Exception as e:
                    logger.error(f"Error in monitoring alert callback: {e}")
        
        except Exception as e:
            logger.error(f"Error handling monitoring alert: {e}")
    
    async def run_health_checks(self) -> List[SystemHealthCheck]:
        """Run comprehensive system health checks."""
        health_checks = []
        
        try:
            # Check configuration manager
            health_checks.append(SystemHealthCheck(
                component="config_manager",
                status=ConfigurationSystemStatus.HEALTHY if self._config_manager else ConfigurationSystemStatus.UNHEALTHY,
                message="Configuration manager operational",
                timestamp=datetime.utcnow()
            ))
            
            # Check validator
            try:
                test_config = {'test': 'value'}
                result = await self._validator.validate_environment(
                    self.environment, test_config
                )
                health_checks.append(SystemHealthCheck(
                    component="validator",
                    status=ConfigurationSystemStatus.HEALTHY,
                    message="Configuration validator operational",
                    timestamp=datetime.utcnow(),
                    details={'validation_time': result.validation_time}
                ))
            except Exception as e:
                health_checks.append(SystemHealthCheck(
                    component="validator",
                    status=ConfigurationSystemStatus.UNHEALTHY,
                    message=f"Validator error: {str(e)}",
                    timestamp=datetime.utcnow()
                ))
            
            # Check hot reloader
            if self._hot_reloader:
                stats = self._hot_reloader.get_statistics()
                status = ConfigurationSystemStatus.HEALTHY if stats['enabled'] else ConfigurationSystemStatus.DEGRADED
                health_checks.append(SystemHealthCheck(
                    component="hot_reloader",
                    status=status,
                    message="Hot reloader operational",
                    timestamp=datetime.utcnow(),
                    details=stats
                ))
            
            # Check version manager
            if self._version_manager:
                versions = self._version_manager.list_versions(limit=1)
                health_checks.append(SystemHealthCheck(
                    component="version_manager",
                    status=ConfigurationSystemStatus.HEALTHY,
                    message="Version manager operational",
                    timestamp=datetime.utcnow(),
                    details={'latest_version': versions[0].version if versions else None}
                ))
            
            # Check monitor
            if self._monitor:
                monitor_status = self._monitor.get_monitoring_status()
                status = (ConfigurationSystemStatus.HEALTHY if monitor_status['monitoring_active'] 
                         else ConfigurationSystemStatus.DEGRADED)
                health_checks.append(SystemHealthCheck(
                    component="monitor",
                    status=status,
                    message="Configuration monitor operational",
                    timestamp=datetime.utcnow(),
                    details=monitor_status
                ))
            
            # Check secrets manager
            if self._secrets_manager:
                secrets_health = await self._secrets_manager.get_secret_health()
                if 'error' in secrets_health:
                    status = ConfigurationSystemStatus.UNHEALTHY
                    message = f"Secrets manager error: {secrets_health['error']}"
                else:
                    status = ConfigurationSystemStatus.HEALTHY
                    message = "Secrets manager operational"
                
                health_checks.append(SystemHealthCheck(
                    component="secrets_manager",
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    details=secrets_health
                ))
        
        except Exception as e:
            logger.error(f"Error running health checks: {e}")
            health_checks.append(SystemHealthCheck(
                component="health_check_system",
                status=ConfigurationSystemStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.utcnow()
            ))
        
        self._health_checks = health_checks
        return health_checks
    
    def get_system_info(self) -> ConfigurationSystemInfo:
        """Get comprehensive system information."""
        
        components = {}
        
        # Config Manager
        components['config_manager'] = {
            'initialized': self._config_manager is not None,
            'environment': self.environment.value
        }
        
        # Validator
        components['validator'] = {
            'available': True
        }
        
        # Hot Reloader
        if self._hot_reloader:
            components['hot_reloader'] = self._hot_reloader.get_statistics()
        else:
            components['hot_reloader'] = {'enabled': False}
        
        # Version Manager
        if self._version_manager:
            versions = self._version_manager.list_versions(limit=5)
            components['version_manager'] = {
                'initialized': True,
                'total_versions': len(self._version_manager.versions_metadata),
                'latest_versions': [
                    {'version': v.version, 'timestamp': v.timestamp.isoformat()}
                    for v in versions
                ]
            }
        else:
            components['version_manager'] = {'initialized': False}
        
        # Monitor
        if self._monitor:
            components['monitor'] = self._monitor.get_monitoring_status()
        else:
            components['monitor'] = {'initialized': False}
        
        # Secrets Manager
        if self._secrets_manager:
            components['secrets_manager'] = {
                'initialized': True,
                'backup_stores': len(self._secrets_manager.backup_stores)
            }
        else:
            components['secrets_manager'] = {'initialized': False}
        
        # Calculate metrics
        metrics = {
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
            'components_initialized': sum(1 for c in components.values() if c.get('initialized', c.get('available', False))),
            'total_components': len(components),
            'health_checks_count': len(self._health_checks),
            'system_healthy': self._status == ConfigurationSystemStatus.HEALTHY
        }
        
        return ConfigurationSystemInfo(
            environment=self.environment,
            status=self._status,
            components=components,
            health_checks=self._health_checks,
            metrics=metrics,
            uptime=datetime.utcnow() - self.start_time,
            last_validation=None,  # Would track this in real implementation
            last_reload=None,      # Would track this in real implementation
            last_version_check=None # Would track this in real implementation
        )
    
    def add_integration_callback(self, callback: Callable[[str, Any], None]):
        """Add integration callback for system events."""
        self._integration_callbacks.append(callback)
    
    def remove_integration_callback(self, callback: Callable[[str, Any], None]):
        """Remove integration callback."""
        if callback in self._integration_callbacks:
            self._integration_callbacks.remove(callback)
    
    @asynccontextmanager
    async def maintenance_mode(self):
        """Context manager for maintenance mode operations."""
        old_status = self._status
        self._status = ConfigurationSystemStatus.MAINTENANCE
        
        logger.info("Entering maintenance mode")
        
        try:
            yield
        finally:
            self._status = old_status
            logger.info("Exiting maintenance mode")
    
    async def emergency_shutdown(self, reason: str):
        """Emergency shutdown of configuration system."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        self._status = ConfigurationSystemStatus.UNHEALTHY
        
        try:
            # Stop all monitoring immediately
            if self._monitor:
                await self._monitor.stop_monitoring()
            
            # Stop hot reloader
            if self._hot_reloader:
                self._hot_reloader.stop()
            
            # Trigger emergency callbacks
            for callback in self._integration_callbacks:
                try:
                    callback("emergency_shutdown", {'reason': reason})
                except Exception as e:
                    logger.error(f"Error in emergency callback: {e}")
            
            logger.critical("Emergency shutdown completed")
        
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}")


# Global system integrator instance
_integrator: Optional[ConfigurationSystemIntegrator] = None


def get_configuration_system() -> ConfigurationSystemIntegrator:
    """Get global configuration system integrator."""
    global _integrator
    
    if _integrator is None:
        environment = Environment(os.getenv('SUPERNOVA_ENV', 'development'))
        _integrator = ConfigurationSystemIntegrator(environment)
    
    return _integrator


# Convenience functions
async def initialize_configuration_system(**kwargs) -> bool:
    """Initialize the complete configuration system."""
    integrator = get_configuration_system()
    return await integrator.initialize(**kwargs)


async def shutdown_configuration_system():
    """Shutdown the complete configuration system."""
    integrator = get_configuration_system()
    await integrator.shutdown()


async def get_system_health() -> List[SystemHealthCheck]:
    """Get system health status."""
    integrator = get_configuration_system()
    return await integrator.run_health_checks()


def get_system_information() -> ConfigurationSystemInfo:
    """Get comprehensive system information."""
    integrator = get_configuration_system()
    return integrator.get_system_info()


def add_system_callback(callback: Callable[[str, Any], None]):
    """Add system integration callback."""
    integrator = get_configuration_system()
    integrator.add_integration_callback(callback)


if __name__ == "__main__":
    """CLI interface for configuration system management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI Configuration System')
    parser.add_argument('action', choices=['init', 'status', 'health', 'shutdown'])
    parser.add_argument('--enable-monitoring', action='store_true', help='Enable monitoring')
    parser.add_argument('--enable-hot-reload', action='store_true', help='Enable hot reload')
    parser.add_argument('--enable-versioning', action='store_true', help='Enable versioning')
    parser.add_argument('--enable-secrets', action='store_true', help='Enable secrets management')
    
    args = parser.parse_args()
    
    async def main():
        integrator = get_configuration_system()
        
        if args.action == 'init':
            print("Initializing configuration system...")
            success = await integrator.initialize(
                enable_monitoring=args.enable_monitoring,
                enable_hot_reload=args.enable_hot_reload,
                enable_versioning=args.enable_versioning,
                enable_secrets=args.enable_secrets
            )
            print(f"Initialization: {'Success' if success else 'Failed'}")
            
            if success:
                print("Configuration system is running. Press Ctrl+C to shutdown.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    await integrator.shutdown()
                    print("System shutdown completed.")
        
        elif args.action == 'status':
            info = integrator.get_system_info()
            print(json.dumps({
                'status': info.status.value,
                'environment': info.environment.value,
                'uptime': str(info.uptime),
                'components': info.components,
                'metrics': info.metrics
            }, indent=2))
        
        elif args.action == 'health':
            health_checks = await integrator.run_health_checks()
            print("System Health Checks:")
            print("-" * 50)
            for check in health_checks:
                status_icon = "✅" if check.status == ConfigurationSystemStatus.HEALTHY else "❌"
                print(f"{status_icon} {check.component}: {check.message}")
        
        elif args.action == 'shutdown':
            await integrator.shutdown()
            print("Configuration system shutdown completed.")
    
    asyncio.run(main())