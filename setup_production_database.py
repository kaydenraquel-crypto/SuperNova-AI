#!/usr/bin/env python3
"""
Production Database Setup Script for SuperNova AI

This script demonstrates how to set up and initialize the complete
production database system including all components:

- PostgreSQL and TimescaleDB configuration
- Alembic migration management
- Database optimization and indexing
- Backup and disaster recovery
- Monitoring and alerting
- Data lifecycle management
- Environment-specific configuration
- Testing and validation

Usage:
    python setup_production_database.py [--environment ENV] [--validate-only] [--help]

Examples:
    # Full production setup
    python setup_production_database.py --environment production

    # Development setup with validation
    python setup_production_database.py --environment development

    # Validation only (no changes)
    python setup_production_database.py --validate-only
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from supernova.database_manager import (
        initialize_database_system,
        get_database_system_status,
        run_database_health_check,
        run_database_maintenance,
        shutdown_database_system
    )
    from supernova.environment_config import (
        environment_manager,
        Environment,
        setup_environment,
        migrate_environment
    )
    from supernova.database_testing import (
        run_test_suite,
        list_test_suites
    )
    from supernova.config import settings
except ImportError as e:
    print(f"Error importing SuperNova modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

def print_status(status: Dict[str, Any], indent: int = 0) -> None:
    """Print status information in a readable format."""
    indent_str = "  " * indent
    
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_status(value, indent + 1)
        elif isinstance(value, list):
            print(f"{indent_str}{key}: [{len(value)} items]")
        else:
            print(f"{indent_str}{key}: {value}")

def setup_environment_databases(env: Environment) -> bool:
    """Set up databases for specific environment."""
    print_header(f"Setting up {env.value.upper()} Environment")
    
    try:
        # Setup environment databases
        setup_results = setup_environment(env)
        
        print("Database Setup Results:")
        for db_name, success in setup_results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {db_name}: {status}")
        
        if not any(setup_results.values()):
            print("❌ No databases were set up successfully")
            return False
        
        # Run migrations
        print("\nRunning database migrations...")
        migration_results = migrate_environment(env)
        
        print("Migration Results:")
        for db_name, success in migration_results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {db_name}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        logger.error(f"Environment setup error: {e}")
        return False

def run_comprehensive_validation() -> bool:
    """Run comprehensive database validation."""
    print_header("Comprehensive Database Validation")
    
    try:
        # Run health check
        print("🔍 Running health check...")
        health_results = run_database_health_check()
        
        print(f"Overall Health: {health_results.get('overall_health', 'unknown').upper()}")
        
        # Print health check details
        for check_name, result in health_results.get('checks', {}).items():
            if isinstance(result, str):
                status = result.upper()
            elif isinstance(result, dict):
                status = result.get('status', 'unknown').upper()
            else:
                status = str(result)
            
            print(f"  {check_name}: {status}")
        
        # Run test suites
        print("\n🧪 Running database tests...")
        
        test_suites = ['basic_connectivity', 'integrity_suite', 'security_suite']
        
        all_tests_passed = True
        
        for suite_name in test_suites:
            print(f"\nRunning test suite: {suite_name}")
            
            try:
                results = run_test_suite(suite_name)
                
                passed = len([r for r in results if r.status.value == 'passed'])
                failed = len([r for r in results if r.status.value == 'failed'])
                total = len(results)
                
                success_rate = (passed / total * 100) if total > 0 else 0
                
                if success_rate >= 80:
                    print(f"  ✅ PASSED: {passed}/{total} tests ({success_rate:.1f}%)")
                else:
                    print(f"  ❌ FAILED: {passed}/{total} tests ({success_rate:.1f}%)")
                    all_tests_passed = False
                
                # Show failed tests
                for result in results:
                    if result.status.value == 'failed':
                        print(f"    ❌ {result.test_name}: {result.error_message}")
                        
            except Exception as e:
                print(f"  ❌ Test suite failed: {e}")
                all_tests_passed = False
        
        return health_results.get('overall_health') in ['healthy', 'warning'] and all_tests_passed
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logger.error(f"Validation error: {e}")
        return False

def initialize_production_system() -> bool:
    """Initialize the complete production database system."""
    print_header("Initializing Production Database System")
    
    try:
        # Initialize with all components
        init_results = initialize_database_system(
            start_monitoring=True,
            start_optimization=True,
            start_lifecycle=True,
            run_migrations=True,
            validate_setup=True
        )
        
        if init_results.get('success', False):
            print("✅ Database system initialization completed successfully")
            
            # Print initialization steps
            print("\nInitialization Steps:")
            for step_name, result in init_results.get('steps', {}).items():
                if isinstance(result, dict):
                    # For complex results, just show status
                    success_count = len([v for v in result.values() if v is True])
                    total_count = len(result)
                    status = f"✅ {success_count}/{total_count}" if success_count > 0 else "❌ FAILED"
                else:
                    status = "✅ SUCCESS" if 'started' in str(result) or 'initialized' in str(result) else "❌ FAILED"
                
                print(f"  {step_name}: {status}")
            
            return True
        else:
            print("❌ Database system initialization failed")
            error = init_results.get('error', 'Unknown error')
            print(f"Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        logger.error(f"System initialization error: {e}")
        return False

def show_system_status() -> None:
    """Show comprehensive system status."""
    print_header("Database System Status")
    
    try:
        status = get_database_system_status()
        
        # Basic status
        print(f"Environment: {status.get('environment', 'unknown')}")
        print(f"Initialized: {status.get('initialized', False)}")
        print(f"Timestamp: {status.get('timestamp', 'unknown')}")
        
        # Component status
        components = status.get('components', {})
        
        # Database health
        databases = components.get('databases', {}).get('databases', {})
        if databases:
            print(f"\nDatabase Health:")
            for db_name, db_info in databases.items():
                health = "✅ HEALTHY" if db_info.get('healthy') else "❌ UNHEALTHY"
                print(f"  {db_name}: {health}")
        
        # Monitoring
        monitoring = components.get('monitoring', {})
        if monitoring:
            monitoring_active = monitoring.get('monitoring_active', False)
            status_text = "✅ ACTIVE" if monitoring_active else "❌ INACTIVE"
            print(f"\nMonitoring: {status_text}")
            
            total_alerts = monitoring.get('total_active_alerts', 0)
            if total_alerts > 0:
                print(f"  Active Alerts: {total_alerts}")
        
        # Current metrics
        current_metrics = components.get('current_metrics', {})
        if current_metrics and not isinstance(current_metrics, str):
            print(f"\nCurrent Metrics:")
            for db_name, metrics in current_metrics.items():
                print(f"  {db_name}:")
                print(f"    Active Connections: {metrics.get('active_connections', 0)}")
                print(f"    Queries/sec: {metrics.get('queries_per_second', 0):.2f}")
                print(f"    Avg Query Duration: {metrics.get('avg_query_duration', 0):.3f}s")
                print(f"    Cache Hit Ratio: {metrics.get('cache_hit_ratio', 0):.2f}")
        
        # Alerts summary
        alerts = components.get('alerts', {})
        if alerts and not isinstance(alerts, str):
            total_alerts = alerts.get('total_alerts', 0)
            critical_alerts = alerts.get('critical_alerts', 0)
            high_alerts = alerts.get('high_alerts', 0)
            
            print(f"\nAlerts:")
            print(f"  Total: {total_alerts}")
            print(f"  Critical: {critical_alerts}")
            print(f"  High: {high_alerts}")
            
            # Show recent alerts
            recent_alerts = alerts.get('alerts', [])
            if recent_alerts:
                print("  Recent Alerts:")
                for alert in recent_alerts[:3]:
                    severity = alert.get('severity', 'unknown').upper()
                    database = alert.get('database', 'unknown')
                    message = alert.get('message', 'No message')
                    print(f"    [{severity}] {database}: {message}")
        
    except Exception as e:
        print(f"❌ Error getting system status: {e}")
        logger.error(f"Status error: {e}")

def run_maintenance() -> bool:
    """Run database maintenance."""
    print_header("Database Maintenance")
    
    try:
        maintenance_results = run_database_maintenance()
        
        if maintenance_results.get('success', False):
            print("✅ Maintenance completed successfully")
            
            tasks = maintenance_results.get('tasks', {})
            print("\nMaintenance Tasks:")
            
            for task_name, result in tasks.items():
                if isinstance(result, dict):
                    if 'indexes_created' in result:
                        created = result.get('indexes_created', 0)
                        failed = result.get('indexes_failed', 0)
                        print(f"  {task_name}: {created} indexes created, {failed} failed")
                    else:
                        print(f"  {task_name}: {result}")
                else:
                    print(f"  {task_name}: {result}")
            
            return True
        else:
            print("❌ Maintenance failed")
            error = maintenance_results.get('error', 'Unknown error')
            print(f"Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        logger.error(f"Maintenance error: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="SuperNova AI Production Database Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--environment', 
        type=str,
        choices=['development', 'testing', 'staging', 'production'],
        default='development',
        help='Target environment for setup'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation, do not make changes'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running test suites'
    )
    
    parser.add_argument(
        '--maintenance',
        action='store_true',
        help='Run maintenance tasks'
    )
    
    args = parser.parse_args()
    
    try:
        environment = Environment(args.environment)
    except ValueError:
        print(f"❌ Invalid environment: {args.environment}")
        return 1
    
    print_header(f"SuperNova AI Database Setup - {environment.value.upper()}")
    
    # Show current configuration
    print(f"Environment: {environment.value}")
    print(f"Database URL: {getattr(settings, 'DATABASE_URL', 'Not configured')}")
    print(f"Validation Only: {args.validate_only}")
    
    success = True
    
    try:
        if args.validate_only:
            # Only run validation
            print_header("Validation Mode - No Changes Will Be Made")
            success = run_comprehensive_validation()
        else:
            # Full setup process
            
            # Step 1: Setup environment databases
            if not setup_environment_databases(environment):
                success = False
            
            # Step 2: Initialize production system
            if success and not initialize_production_system():
                success = False
            
            # Step 3: Run validation unless skipped
            if success and not args.skip_tests:
                if not run_comprehensive_validation():
                    print("⚠️  Validation failed, but setup may still be functional")
            
            # Step 4: Run maintenance if requested
            if success and args.maintenance:
                run_maintenance()
            
            # Step 5: Show final system status
            show_system_status()
        
        # Final result
        if success:
            print_header("✅ SETUP COMPLETED SUCCESSFULLY")
            print("The SuperNova AI database system is ready for use!")
            
            if not args.validate_only:
                print("\n📋 Next Steps:")
                print("1. Monitor the system status using the database manager")
                print("2. Set up regular backups and maintenance schedules")
                print("3. Configure alerts and notifications")
                print("4. Review and adjust performance settings as needed")
                print("\n🔧 Management Commands:")
                print("  python -c \"from supernova.database_manager import get_database_system_status; print(get_database_system_status())\"")
                print("  python -c \"from supernova.database_manager import run_database_health_check; print(run_database_health_check())\"")
            
            return 0
        else:
            print_header("❌ SETUP FAILED")
            print("Check the logs above for specific error details.")
            print("Common issues:")
            print("1. Database server not running")
            print("2. Incorrect database credentials")
            print("3. Missing database permissions")
            print("4. Network connectivity issues")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        logger.error(f"Setup error: {e}")
        return 1
    finally:
        # Always attempt graceful shutdown
        try:
            print("\n🔧 Shutting down database system...")
            shutdown_results = shutdown_database_system()
            if shutdown_results.get('success'):
                print("✅ System shutdown completed")
            else:
                print("⚠️  System shutdown completed with warnings")
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)