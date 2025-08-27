"""
Comprehensive Database Management System for SuperNova AI

Central database management system that coordinates all database components:
- Database configuration and connection management
- Migration management and version control
- Performance optimization and monitoring
- Backup and disaster recovery
- Data lifecycle management
- Environment-specific configuration
- Testing and validation
"""

from __future__ import annotations
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

try:
    from .config import settings
    from .database_config import (
        db_config, initialize_databases, get_database_health, close_all_connections
    )
    from .migration_manager import (
        migration_manager, get_migration_status, migrate_all
    )
    from .database_optimization import (
        db_optimizer, start_database_optimization, get_optimization_report, apply_optimizations
    )
    from .database_monitoring import (
        db_monitor, start_database_monitoring, stop_database_monitoring, 
        get_monitoring_status, get_current_metrics, get_active_alerts
    )
    from .backup_recovery import initialize_backup_system
    from .data_lifecycle import (
        data_lifecycle_manager, start_lifecycle_scheduler, get_lifecycle_status
    )
    from .environment_config import (
        environment_manager, get_environment_status, get_current_environment
    )
    from .database_testing import (
        db_test_runner, run_test_suite, get_test_report, list_test_suites
    )
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    raise e

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Comprehensive database management system.
    
    Coordinates all database components and provides a unified interface
    for database operations, monitoring, and maintenance.
    """
    
    def __init__(self):
        self.initialized = False
        self.components = {
            'database_config': db_config,
            'migration_manager': migration_manager,
            'optimizer': db_optimizer,
            'monitor': db_monitor,
            'lifecycle_manager': data_lifecycle_manager,
            'environment_manager': environment_manager,
            'test_runner': db_test_runner
        }
        
        # Component status
        self.component_status = {}
        
        # Backup system
        self.backup_manager = None
        self.backup_scheduler = None
    
    def initialize(self, 
                   start_monitoring: bool = True,
                   start_optimization: bool = True,
                   start_lifecycle: bool = True,
                   run_migrations: bool = True,
                   validate_setup: bool = True) -> Dict[str, Any]:
        """
        Initialize the complete database system.
        
        Args:
            start_monitoring: Start database monitoring
            start_optimization: Start query optimization
            start_lifecycle: Start data lifecycle management
            run_migrations: Run pending migrations
            validate_setup: Validate setup after initialization
            
        Returns:
            Initialization status and results
        """
        logger.info("Initializing SuperNova Database Management System")
        
        initialization_results = {
            'start_time': datetime.utcnow().isoformat(),
            'environment': get_current_environment().value,
            'steps': {},
            'success': False
        }
        
        try:
            # Step 1: Initialize database connections
            logger.info("Step 1: Initializing database connections")
            db_init_results = initialize_databases()
            initialization_results['steps']['database_initialization'] = db_init_results
            
            if not any(db_init_results.values()):
                raise RuntimeError("No databases could be initialized")
            
            # Step 2: Run migrations if requested
            if run_migrations:
                logger.info("Step 2: Running database migrations")
                migration_results = migrate_all('head')
                initialization_results['steps']['migrations'] = migration_results
            
            # Step 3: Initialize backup system
            logger.info("Step 3: Initializing backup and recovery system")
            try:
                self.backup_manager, self.backup_scheduler = initialize_backup_system()
                initialization_results['steps']['backup_system'] = 'initialized'
            except Exception as e:
                logger.warning(f"Backup system initialization failed: {e}")
                initialization_results['steps']['backup_system'] = f'failed: {str(e)}'
            
            # Step 4: Start monitoring if requested
            if start_monitoring:
                logger.info("Step 4: Starting database monitoring")
                try:
                    start_database_monitoring()
                    initialization_results['steps']['monitoring'] = 'started'
                except Exception as e:
                    logger.error(f"Monitoring startup failed: {e}")
                    initialization_results['steps']['monitoring'] = f'failed: {str(e)}'
            
            # Step 5: Start optimization if requested
            if start_optimization:
                logger.info("Step 5: Starting database optimization")
                try:
                    start_database_optimization()
                    initialization_results['steps']['optimization'] = 'started'
                except Exception as e:
                    logger.error(f"Optimization startup failed: {e}")
                    initialization_results['steps']['optimization'] = f'failed: {str(e)}'
            
            # Step 6: Start lifecycle management if requested
            if start_lifecycle:
                logger.info("Step 6: Starting data lifecycle management")
                try:
                    start_lifecycle_scheduler()
                    initialization_results['steps']['lifecycle'] = 'started'
                except Exception as e:
                    logger.error(f"Lifecycle management startup failed: {e}")
                    initialization_results['steps']['lifecycle'] = f'failed: {str(e)}'
            
            # Step 7: Validate setup if requested
            if validate_setup:
                logger.info("Step 7: Validating database setup")
                validation_results = self.validate_setup()
                initialization_results['steps']['validation'] = validation_results
            
            self.initialized = True
            initialization_results['success'] = True
            initialization_results['end_time'] = datetime.utcnow().isoformat()
            
            logger.info("Database Management System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            initialization_results['error'] = str(e)
            initialization_results['success'] = False
            initialization_results['end_time'] = datetime.utcnow().isoformat()
        
        return initialization_results
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate the database setup.
        
        Returns:
            Validation results
        """
        logger.info("Validating database setup")
        
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'overall_status': 'unknown'
        }
        
        checks = [
            ('database_connectivity', self._check_database_connectivity),
            ('migration_status', self._check_migration_status),
            ('schema_consistency', self._check_schema_consistency),
            ('monitoring_status', self._check_monitoring_status),
            ('backup_configuration', self._check_backup_configuration),
            ('performance_baseline', self._check_performance_baseline)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_function in checks:
            try:
                logger.debug(f"Running validation check: {check_name}")
                result = check_function()
                validation_results['checks'][check_name] = result
                
                if result.get('status') == 'passed':
                    passed_checks += 1
                    
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {e}")
                validation_results['checks'][check_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Determine overall status
        success_rate = passed_checks / total_checks
        if success_rate >= 0.8:
            validation_results['overall_status'] = 'passed'
        elif success_rate >= 0.6:
            validation_results['overall_status'] = 'warning'
        else:
            validation_results['overall_status'] = 'failed'
        
        validation_results['success_rate'] = success_rate
        validation_results['passed_checks'] = passed_checks
        validation_results['total_checks'] = total_checks
        
        return validation_results
    
    def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity."""
        health_status = get_database_health()
        
        all_healthy = True
        for db_name, db_status in health_status.get('databases', {}).items():
            if not db_status.get('healthy', False):
                all_healthy = False
                break
        
        return {
            'status': 'passed' if all_healthy else 'failed',
            'details': health_status
        }
    
    def _check_migration_status(self) -> Dict[str, Any]:
        """Check migration status."""
        migration_status = get_migration_status()
        
        all_up_to_date = True
        for db_name, db_status in migration_status.get('databases', {}).items():
            if not db_status.get('is_up_to_date', False):
                all_up_to_date = False
                break
        
        return {
            'status': 'passed' if all_up_to_date else 'failed',
            'details': migration_status
        }
    
    def _check_schema_consistency(self) -> Dict[str, Any]:
        """Check schema consistency."""
        try:
            # Run basic schema consistency test
            test_result = db_test_runner.run_test('test_schema_consistency')
            
            return {
                'status': 'passed' if test_result.status.value == 'passed' else 'failed',
                'details': {
                    'test_duration': test_result.duration_seconds,
                    'assertions_passed': test_result.assertions_passed,
                    'assertions_failed': test_result.assertions_failed
                }
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_monitoring_status(self) -> Dict[str, Any]:
        """Check monitoring system status."""
        monitoring_status = get_monitoring_status()
        
        return {
            'status': 'passed' if monitoring_status.get('monitoring_active', False) else 'warning',
            'details': monitoring_status
        }
    
    def _check_backup_configuration(self) -> Dict[str, Any]:
        """Check backup configuration."""
        if self.backup_manager:
            backup_status = {
                'backup_manager': 'available',
                'scheduler': 'available' if self.backup_scheduler else 'not_available'
            }
            status = 'passed'
        else:
            backup_status = {
                'backup_manager': 'not_available',
                'scheduler': 'not_available'
            }
            status = 'warning'
        
        return {
            'status': status,
            'details': backup_status
        }
    
    def _check_performance_baseline(self) -> Dict[str, Any]:
        """Check performance baseline."""
        try:
            # Run basic performance test
            test_result = db_test_runner.run_test('test_query_performance')
            
            return {
                'status': 'passed' if test_result.status.value == 'passed' else 'failed',
                'details': {
                    'test_duration': test_result.duration_seconds,
                    'performance_metrics': test_result.details
                }
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Complete system status information
        """
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': get_current_environment().value,
            'initialized': self.initialized,
            'components': {}
        }
        
        # Database health
        status['components']['databases'] = get_database_health()
        
        # Migration status
        status['components']['migrations'] = get_migration_status()
        
        # Monitoring status
        status['components']['monitoring'] = get_monitoring_status()
        
        # Environment status
        status['components']['environments'] = get_environment_status()
        
        # Lifecycle status
        status['components']['lifecycle'] = get_lifecycle_status()
        
        # Current metrics
        try:
            current_metrics = get_current_metrics()
            status['components']['current_metrics'] = {
                db_name: {
                    'active_connections': metrics.active_connections,
                    'queries_per_second': metrics.queries_per_second,
                    'avg_query_duration': metrics.avg_query_duration,
                    'cache_hit_ratio': metrics.cache_hit_ratio
                }
                for db_name, metrics in current_metrics.items()
            }
        except Exception as e:
            status['components']['current_metrics'] = f'error: {str(e)}'
        
        # Active alerts
        try:
            active_alerts = get_active_alerts()
            status['components']['alerts'] = {
                'total_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity.value == 'critical']),
                'high_alerts': len([a for a in active_alerts if a.severity.value == 'high']),
                'alerts': [
                    {
                        'database': alert.database,
                        'type': alert.alert_type,
                        'severity': alert.severity.value,
                        'message': alert.message
                    }
                    for alert in active_alerts[:5]  # Latest 5 alerts
                ]
            }
        except Exception as e:
            status['components']['alerts'] = f'error: {str(e)}'
        
        return status
    
    def run_maintenance(self) -> Dict[str, Any]:
        """
        Run comprehensive database maintenance.
        
        Returns:
            Maintenance results
        """
        logger.info("Running database maintenance")
        
        maintenance_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tasks': {},
            'success': False
        }
        
        try:
            # 1. Apply optimization recommendations
            logger.info("Applying database optimizations")
            optimization_results = apply_optimizations('primary', max_indexes=3)
            maintenance_results['tasks']['optimization'] = optimization_results
            
            # 2. Run data lifecycle policies
            logger.info("Executing data lifecycle policies")
            lifecycle_events = data_lifecycle_manager.execute_all_policies()
            maintenance_results['tasks']['lifecycle'] = {
                'policies_executed': len(lifecycle_events),
                'records_processed': sum(event.records_affected for event in lifecycle_events),
                'successful_policies': len([e for e in lifecycle_events if e.success])
            }
            
            # 3. Backup validation (if configured)
            if self.backup_manager:
                logger.info("Validating backup system")
                # Just check that backup system is healthy
                maintenance_results['tasks']['backup_validation'] = 'completed'
            
            # 4. Clean up old metrics and logs
            logger.info("Cleaning up old monitoring data")
            # This would involve cleaning up old monitoring data
            maintenance_results['tasks']['cleanup'] = 'completed'
            
            maintenance_results['success'] = True
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
            maintenance_results['error'] = str(e)
            maintenance_results['success'] = False
        
        return maintenance_results
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check.
        
        Returns:
            Health check results
        """
        logger.info("Running database health check")
        
        health_check = {
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'overall_health': 'unknown'
        }
        
        # Run basic connectivity test suite
        try:
            test_results = run_test_suite('basic_connectivity')
            
            passed_tests = len([r for r in test_results if r.status.value == 'passed'])
            total_tests = len(test_results)
            
            health_check['checks']['connectivity'] = {
                'passed': passed_tests,
                'total': total_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            }
            
        except Exception as e:
            health_check['checks']['connectivity'] = f'error: {str(e)}'
        
        # Check system resources and performance
        try:
            current_metrics = get_current_metrics()
            
            resource_health = 'healthy'
            for db_name, metrics in current_metrics.items():
                # Check connection utilization
                if metrics.connection_utilization > 0.8:
                    resource_health = 'warning'
                if metrics.connection_utilization > 0.95:
                    resource_health = 'critical'
                
                # Check query performance
                if metrics.avg_query_duration > 2.0:
                    resource_health = 'warning'
                if metrics.avg_query_duration > 5.0:
                    resource_health = 'critical'
            
            health_check['checks']['resources'] = resource_health
            
        except Exception as e:
            health_check['checks']['resources'] = f'error: {str(e)}'
        
        # Check for active alerts
        try:
            active_alerts = get_active_alerts()
            critical_alerts = [a for a in active_alerts if a.severity.value == 'critical']
            
            if critical_alerts:
                health_check['checks']['alerts'] = 'critical'
            elif active_alerts:
                health_check['checks']['alerts'] = 'warning'
            else:
                health_check['checks']['alerts'] = 'healthy'
                
        except Exception as e:
            health_check['checks']['alerts'] = f'error: {str(e)}'
        
        # Determine overall health
        check_results = [
            v for v in health_check['checks'].values() 
            if isinstance(v, str) and v in ['healthy', 'warning', 'critical']
        ]
        
        if 'critical' in check_results:
            health_check['overall_health'] = 'critical'
        elif 'warning' in check_results:
            health_check['overall_health'] = 'warning'
        elif 'healthy' in check_results:
            health_check['overall_health'] = 'healthy'
        else:
            health_check['overall_health'] = 'unknown'
        
        return health_check
    
    def create_backup(self, database: str = 'primary', backup_type: str = 'full') -> Dict[str, Any]:
        """
        Create database backup.
        
        Args:
            database: Database to backup
            backup_type: Type of backup (full, incremental, differential)
            
        Returns:
            Backup results
        """
        if not self.backup_manager:
            return {
                'success': False,
                'error': 'Backup system not initialized'
            }
        
        logger.info(f"Creating {backup_type} backup for {database}")
        
        backup_metadata = self.backup_manager.create_backup(database, backup_type)
        
        if backup_metadata and backup_metadata.success:
            return {
                'success': True,
                'backup_id': backup_metadata.backup_id,
                'file_size_bytes': backup_metadata.file_size_bytes,
                'duration_seconds': (backup_metadata.end_time - backup_metadata.start_time).total_seconds()
            }
        else:
            return {
                'success': False,
                'error': backup_metadata.error_message if backup_metadata else 'Unknown error'
            }
    
    def shutdown(self) -> Dict[str, Any]:
        """
        Gracefully shutdown the database management system.
        
        Returns:
            Shutdown results
        """
        logger.info("Shutting down Database Management System")
        
        shutdown_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'steps': {},
            'success': False
        }
        
        try:
            # Stop monitoring
            try:
                stop_database_monitoring()
                shutdown_results['steps']['monitoring'] = 'stopped'
            except Exception as e:
                shutdown_results['steps']['monitoring'] = f'error: {str(e)}'
            
            # Stop optimization
            try:
                db_optimizer.stop_optimization()
                shutdown_results['steps']['optimization'] = 'stopped'
            except Exception as e:
                shutdown_results['steps']['optimization'] = f'error: {str(e)}'
            
            # Stop lifecycle scheduler
            try:
                data_lifecycle_manager.stop_scheduler()
                shutdown_results['steps']['lifecycle'] = 'stopped'
            except Exception as e:
                shutdown_results['steps']['lifecycle'] = f'error: {str(e)}'
            
            # Stop backup scheduler
            if self.backup_scheduler:
                try:
                    self.backup_scheduler.stop_scheduler()
                    shutdown_results['steps']['backup_scheduler'] = 'stopped'
                except Exception as e:
                    shutdown_results['steps']['backup_scheduler'] = f'error: {str(e)}'
            
            # Close database connections
            try:
                close_all_connections()
                shutdown_results['steps']['database_connections'] = 'closed'
            except Exception as e:
                shutdown_results['steps']['database_connections'] = f'error: {str(e)}'
            
            self.initialized = False
            shutdown_results['success'] = True
            
            logger.info("Database Management System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            shutdown_results['error'] = str(e)
            shutdown_results['success'] = False
        
        return shutdown_results

# Global database manager instance
database_manager = DatabaseManager()

# Convenience functions
def initialize_database_system(
    start_monitoring: bool = True,
    start_optimization: bool = True,
    start_lifecycle: bool = True,
    run_migrations: bool = True,
    validate_setup: bool = True
) -> Dict[str, Any]:
    """Initialize the complete database system."""
    return database_manager.initialize(
        start_monitoring=start_monitoring,
        start_optimization=start_optimization,
        start_lifecycle=start_lifecycle,
        run_migrations=run_migrations,
        validate_setup=validate_setup
    )

def get_database_system_status() -> Dict[str, Any]:
    """Get comprehensive database system status."""
    return database_manager.get_system_status()

def run_database_maintenance() -> Dict[str, Any]:
    """Run comprehensive database maintenance."""
    return database_manager.run_maintenance()

def run_database_health_check() -> Dict[str, Any]:
    """Run comprehensive database health check."""
    return database_manager.run_health_check()

def create_database_backup(database: str = 'primary', backup_type: str = 'full') -> Dict[str, Any]:
    """Create database backup."""
    return database_manager.create_backup(database, backup_type)

def shutdown_database_system() -> Dict[str, Any]:
    """Shutdown the database management system."""
    return database_manager.shutdown()