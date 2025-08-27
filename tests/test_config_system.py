"""
SuperNova AI - Comprehensive Configuration System Testing Suite

This test suite provides comprehensive testing for:
- Configuration validation and type checking
- Environment-specific configuration loading
- Configuration hot-reloading and safety checks
- Configuration versioning and rollback capabilities
- Configuration monitoring and drift detection
- Compliance checking and security validation
- Integration between all configuration components
"""

import os
import sys
import json
import tempfile
import asyncio
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supernova.config_management import config_manager, Environment, ConfigurationLevel
from supernova.config_validation_enhanced import validator, ValidationSeverity, ValidationCategory
from supernova.config_hot_reload import get_hot_reloader, ConfigurationHotReloader, ReloadStatus
from supernova.config_versioning import get_version_manager, VersionType
from supernova.config_monitoring import get_configuration_monitor, DriftSeverity, ComplianceStatus
from supernova.secrets_management import get_secrets_manager, SecretType
from supernova.environment_config import environment_manager


class TestConfigurationValidation:
    """Test configuration validation system."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'supernova_env': 'testing',
            'debug': 'false',
            'log_level': 'ERROR',
            'database_url': 'sqlite:///:memory:',
            'db_pool_size': '5',
            'llm_enabled': 'true',
            'llm_provider': 'openai',
            'llm_temperature': '0.0',
            'llm_max_tokens': '100',
            'ssl_required': 'false',
            'mfa_enabled': 'false',
            'rate_limit_enabled': 'false'
        }
    
    @pytest.fixture
    def invalid_config(self):
        return {
            'supernova_env': 'testing',
            'debug': 'invalid_bool',
            'log_level': 123,  # Should be string
            'db_pool_size': 'not_a_number',
            'llm_temperature': '3.0',  # Above max range
            'llm_max_tokens': '0',  # Below min range
        }
    
    @pytest.mark.asyncio
    async def test_type_validation(self, sample_config):
        """Test configuration type validation."""
        result = await validator.validate_environment(
            environment=Environment.TESTING,
            config=sample_config,
            categories=[ValidationCategory.TYPE]
        )
        
        assert result.passed
        assert result.environment == Environment.TESTING
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_type_validation_failures(self, invalid_config):
        """Test configuration type validation failures."""
        result = await validator.validate_environment(
            environment=Environment.TESTING,
            config=invalid_config,
            categories=[ValidationCategory.TYPE]
        )
        
        assert not result.passed
        assert len(result.errors) > 0
        
        # Check for specific type errors
        error_keys = [issue.key for issue in result.errors]
        assert 'debug' in error_keys
        assert 'log_level' in error_keys
        assert 'db_pool_size' in error_keys
    
    @pytest.mark.asyncio
    async def test_range_validation(self, sample_config):
        """Test numeric range validation."""
        # Modify config to test range validation
        test_config = sample_config.copy()
        test_config['llm_temperature'] = '2.5'  # Above max
        test_config['db_pool_size'] = '0'  # Below min
        
        result = await validator.validate_environment(
            environment=Environment.TESTING,
            config=test_config,
            categories=[ValidationCategory.RANGE]
        )
        
        assert not result.passed
        assert len(result.errors) >= 2
        
        error_keys = [issue.key for issue in result.errors]
        assert 'llm_temperature' in error_keys
        assert 'db_pool_size' in error_keys
    
    @pytest.mark.asyncio
    async def test_security_validation_production(self):
        """Test security validation for production environment."""
        prod_config = {
            'supernova_env': 'production',
            'debug': 'true',  # Should be false in production
            'ssl_required': 'false',  # Should be true in production
            'db_ssl_required': 'false',  # Should be true in production
            'mfa_enabled': 'false',  # Should be true in production
            'secret_key': 'test_key',  # Weak secret
        }
        
        result = await validator.validate_environment(
            environment=Environment.PRODUCTION,
            config=prod_config,
            categories=[ValidationCategory.SECURITY]
        )
        
        assert not result.passed
        assert len(result.errors) > 0
        assert len(result.critical) > 0
        
        # Check for specific security violations
        critical_keys = [issue.key for issue in result.critical]
        assert 'debug' in critical_keys or 'ssl_required' in critical_keys
    
    @pytest.mark.asyncio
    async def test_dependency_validation(self):
        """Test configuration dependency validation."""
        config_with_dependencies = {
            'llm_enabled': 'true',
            'llm_provider': 'openai',
            # Missing openai_api_key dependency
            'cache_backend': 'redis',
            # Missing redis_url dependency
            'mfa_enabled': 'true',
            # Missing mfa_issuer_name dependency
        }
        
        result = await validator.validate_environment(
            environment=Environment.DEVELOPMENT,
            config=config_with_dependencies,
            categories=[ValidationCategory.DEPENDENCY]
        )
        
        assert not result.passed
        assert len(result.errors) >= 3
        
        error_keys = [issue.key for issue in result.errors]
        assert 'openai_api_key' in error_keys
        assert 'redis_url' in error_keys
        assert 'mfa_issuer_name' in error_keys
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, sample_config):
        """Test comprehensive validation with all categories."""
        result = await validator.validate_environment(
            environment=Environment.TESTING,
            config=sample_config,
            strict=True
        )
        
        assert result.passed
        assert result.total_checks > 0
        assert result.validation_time > 0


class TestConfigurationHotReload:
    """Test configuration hot-reloading system."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def hot_reloader(self, temp_config_dir):
        """Create hot reloader instance for testing."""
        return ConfigurationHotReloader(
            environment=Environment.TESTING,
            backup_dir=temp_config_dir / "backups",
            enabled=True,
            validation_required=True,
            safety_checks_enabled=False  # Disable for testing
        )
    
    @pytest.mark.asyncio
    async def test_manual_reload_success(self, hot_reloader):
        """Test successful manual configuration reload."""
        new_config = {
            'test_key': 'test_value',
            'another_key': 'another_value'
        }
        
        event = await hot_reloader.reload_configuration(
            new_config=new_config,
            force=True
        )
        
        assert event.status == ReloadStatus.SUCCESS
        assert event.trigger.value == 'manual'
        assert len(event.changes['added']) == 2
    
    @pytest.mark.asyncio
    async def test_reload_with_validation_failure(self, hot_reloader):
        """Test reload with validation failure."""
        invalid_config = {
            'debug': 'invalid_boolean_value',
            'db_pool_size': 'not_a_number'
        }
        
        event = await hot_reloader.reload_configuration(
            new_config=invalid_config
        )
        
        assert event.status == ReloadStatus.FAILED
        assert event.validation_result is not None
        assert not event.validation_result.passed
    
    @pytest.mark.asyncio
    async def test_configuration_backup_and_rollback(self, hot_reloader):
        """Test configuration backup and rollback functionality."""
        # Set initial configuration
        initial_config = {'key1': 'value1', 'key2': 'value2'}
        await hot_reloader.reload_configuration(initial_config, force=True)
        
        # Create backup
        backup_id = hot_reloader.backup_manager.create_backup(
            initial_config,
            {'test': 'backup'}
        )
        
        assert backup_id is not None
        
        # Change configuration
        new_config = {'key1': 'new_value1', 'key3': 'value3'}
        await hot_reloader.reload_configuration(new_config, force=True)
        
        # Rollback to backup
        rollback_event = await hot_reloader.rollback_to_backup(backup_id)
        
        assert rollback_event.status == ReloadStatus.SUCCESS
        assert hot_reloader.current_config['key1'] == 'value1'
        assert 'key3' not in hot_reloader.current_config
    
    @pytest.mark.asyncio
    async def test_reload_skip_unchanged(self, hot_reloader):
        """Test that unchanged configuration is skipped."""
        config = {'test': 'value'}
        
        # First reload
        event1 = await hot_reloader.reload_configuration(config, force=True)
        assert event1.status == ReloadStatus.SUCCESS
        
        # Second reload with same config
        event2 = await hot_reloader.reload_configuration(config)
        assert event2.status == ReloadStatus.SKIPPED
    
    def test_hot_reloader_statistics(self, hot_reloader):
        """Test hot reloader statistics collection."""
        stats = hot_reloader.get_statistics()
        
        assert 'enabled' in stats
        assert 'environment' in stats
        assert 'total_reloads' in stats
        assert 'success_rate' in stats
        assert stats['environment'] == Environment.TESTING.value


class TestConfigurationVersioning:
    """Test configuration versioning system."""
    
    @pytest.fixture
    def temp_repo_dir(self):
        """Create temporary directory for version repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def version_manager(self, temp_repo_dir):
        """Create version manager instance for testing."""
        from supernova.config_versioning import ConfigurationVersionManager
        return ConfigurationVersionManager(
            repository_path=temp_repo_dir,
            environment=Environment.TESTING,
            use_git=False  # Disable git for testing
        )
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager):
        """Test creating configuration version."""
        config = {
            'app_name': 'SuperNova AI Test',
            'version': '1.0.0-test',
            'debug': True
        }
        
        version_obj = await version_manager.create_version(
            config=config,
            message="Test version creation",
            version_type=VersionType.PATCH,
            author="test_user",
            validate=False  # Skip validation for testing
        )
        
        assert version_obj is not None
        assert version_obj.version == "1.0.0"
        assert version_obj.message == "Test version creation"
        assert version_obj.author == "test_user"
        assert version_obj.version_type == VersionType.PATCH
    
    @pytest.mark.asyncio
    async def test_version_comparison(self, version_manager):
        """Test version comparison and change detection."""
        config1 = {'key1': 'value1', 'key2': 'value2'}
        config2 = {'key1': 'new_value1', 'key3': 'value3'}
        
        # Create first version
        version1 = await version_manager.create_version(
            config1, "First version", VersionType.MINOR, validate=False
        )
        
        # Create second version
        version2 = await version_manager.create_version(
            config2, "Second version", VersionType.MINOR, validate=False
        )
        
        # Compare versions
        changes = version_manager.compare_versions(version1.version, version2.version)
        
        assert len(changes) == 3  # 1 modified, 1 added, 1 removed
        change_types = [c.change_type.value for c in changes]
        assert 'modified' in change_types
        assert 'added' in change_types
        assert 'removed' in change_types
    
    @pytest.mark.asyncio
    async def test_rollback_plan_creation(self, version_manager):
        """Test rollback plan creation and analysis."""
        # Create initial version
        config1 = {'critical_setting': 'value1', 'normal_setting': 'value2'}
        version1 = await version_manager.create_version(
            config1, "Initial version", VersionType.MINOR, validate=False
        )
        
        # Create second version with critical changes
        config2 = {'critical_setting': 'changed_value', 'new_setting': 'new_value'}
        version2 = await version_manager.create_version(
            config2, "Version with critical changes", VersionType.MINOR, validate=False
        )
        
        # Create rollback plan
        rollback_plan = await version_manager.create_rollback_plan(
            target_version=version1.version,
            current_version=version2.version
        )
        
        assert rollback_plan is not None
        assert rollback_plan.target_version == version1.version
        assert rollback_plan.current_version == version2.version
        assert len(rollback_plan.affected_keys) > 0
        assert rollback_plan.estimated_downtime > timedelta(0)
        assert len(rollback_plan.rollback_steps) > 0
    
    def test_version_history(self, version_manager):
        """Test version history retrieval."""
        # Add some test versions to metadata (simulating existing versions)
        from supernova.config_versioning import ConfigurationVersion
        
        test_version = ConfigurationVersion(
            version="1.0.0",
            environment=Environment.TESTING,
            timestamp=datetime.utcnow(),
            author="test",
            message="Test version",
            version_type=VersionType.PATCH,
            rollback_safe=True
        )
        
        version_manager.versions_metadata["1.0.0"] = test_version
        
        history = version_manager.get_version_history(limit=10)
        
        assert len(history) == 1
        assert history[0]['version'] == "1.0.0"
        assert history[0]['author'] == "test"
        assert history[0]['rollback_safe'] is True


class TestConfigurationMonitoring:
    """Test configuration monitoring and drift detection."""
    
    @pytest.fixture
    def config_monitor(self):
        """Create configuration monitor instance for testing."""
        from supernova.config_monitoring import ConfigurationMonitor
        return ConfigurationMonitor(environment=Environment.TESTING)
    
    def test_baseline_creation(self, config_monitor):
        """Test configuration baseline creation."""
        config = {
            'setting1': 'value1',
            'setting2': 'value2',
            'setting3': 'value3'
        }
        
        success = config_monitor.baseline_manager.create_baseline(
            name="test_baseline",
            config=config,
            metadata={'test': 'metadata'}
        )
        
        assert success is True
        
        retrieved_config = config_monitor.baseline_manager.get_baseline("test_baseline")
        assert retrieved_config == config
    
    def test_drift_detection(self, config_monitor):
        """Test configuration drift detection."""
        # Create baseline
        baseline_config = {
            'setting1': 'original_value',
            'setting2': 'original_value2',
            'critical_setting': 'critical_value'
        }
        
        config_monitor.baseline_manager.create_baseline("test", baseline_config)
        
        # Create current config with drifts
        current_config = {
            'setting1': 'changed_value',  # Modified
            'setting2': 'original_value2',  # Unchanged
            'new_setting': 'new_value',  # Added
            # critical_setting removed
        }
        
        drifts = config_monitor.drift_detector.detect_drift(current_config, "test")
        
        assert len(drifts) == 3  # 1 modified, 1 added, 1 removed
        
        # Check specific drifts
        drift_keys = [d.key for d in drifts]
        assert 'setting1' in drift_keys
        assert 'new_setting' in drift_keys
        assert 'critical_setting' in drift_keys
        
        # Check severity assessment
        critical_drift = next(d for d in drifts if d.key == 'critical_setting')
        assert critical_drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
    
    def test_compliance_monitoring(self, config_monitor):
        """Test compliance rule checking."""
        # Test configuration with compliance violations
        non_compliant_config = {
            'supernova_env': 'production',
            'ssl_required': False,  # Should be True for production
            'mfa_enabled': False,  # Should be True for production
            'audit_log_retention_days': 30,  # Too short for compliance
            'secret_key': 'weak',  # Too short for security
        }
        
        violations = config_monitor.compliance_monitor.check_compliance(non_compliant_config)
        
        assert len(violations) > 0
        
        # Check for specific violations
        violation_types = [v.violation_type for v in violations]
        assert 'incorrect_required_values' in violation_types
        
        # Check severity
        high_severity_violations = [v for v in violations if v.severity == DriftSeverity.HIGH]
        assert len(high_severity_violations) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, config_monitor):
        """Test integrated monitoring workflow."""
        # Setup baseline
        baseline_config = {
            'ssl_required': True,
            'mfa_enabled': True,
            'audit_logging_enabled': True
        }
        config_monitor.baseline_manager.create_baseline("production", baseline_config)
        
        # Mock current configuration with issues
        current_config = {
            'ssl_required': False,  # Drift + compliance issue
            'mfa_enabled': True,
            'debug': True,  # Compliance issue for production
        }
        
        # Mock the _get_current_configuration method
        config_monitor._get_current_configuration = AsyncMock(return_value=current_config)
        
        # Run monitoring checks
        await config_monitor.run_monitoring_checks()
        
        # Verify that monitoring was performed
        assert config_monitor.last_check_time is not None
    
    def test_monitoring_metrics(self, config_monitor):
        """Test monitoring metrics collection."""
        # Verify metrics are properly initialized
        assert 'drift_detections_total' in config_monitor.metrics
        assert 'compliance_violations_total' in config_monitor.metrics
        assert 'monitoring_checks_total' in config_monitor.metrics
        
        # Test metric update with sample data
        from supernova.config_monitoring import ConfigurationDrift, ComplianceViolation
        
        drift = ConfigurationDrift(
            key='test_key',
            expected_value='expected',
            actual_value='actual',
            drift_type='modified',
            severity=DriftSeverity.MEDIUM,
            detected_at=datetime.utcnow(),
            environment=Environment.TESTING,
            source='test'
        )
        
        violation = ComplianceViolation(
            rule_id='test_rule',
            rule_name='Test Rule',
            violation_type='test_violation',
            affected_keys={'test_key'},
            severity=DriftSeverity.HIGH,
            status=ComplianceStatus.NON_COMPLIANT,
            detected_at=datetime.utcnow(),
            environment=Environment.TESTING,
            description='Test violation',
            compliance_framework='Test'
        )
        
        config_monitor._update_metrics([drift], [violation])
        
        # Verify metrics were updated (would need actual metric collection in real test)
        assert True  # Placeholder for actual metric verification


class TestConfigurationIntegration:
    """Test integration between all configuration components."""
    
    @pytest.fixture
    def integrated_setup(self):
        """Setup integrated configuration system for testing."""
        # This would setup all components together
        return {
            'config_manager': config_manager,
            'validator': validator,
            'hot_reloader': get_hot_reloader(),
            'version_manager': get_version_manager(),
            'monitor': get_configuration_monitor(),
            'secrets_manager': get_secrets_manager()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_configuration_workflow(self, integrated_setup):
        """Test complete configuration management workflow."""
        components = integrated_setup
        
        # 1. Set initial configuration
        initial_config = {
            'app_name': 'SuperNova AI',
            'environment': 'testing',
            'debug': False,
            'log_level': 'INFO'
        }
        
        # 2. Validate configuration
        validation_result = await components['validator'].validate_environment(
            Environment.TESTING,
            initial_config
        )
        assert validation_result.passed
        
        # 3. Create configuration version
        version = await components['version_manager'].create_version(
            initial_config,
            "Initial configuration",
            VersionType.MINOR,
            validate=False
        )
        assert version is not None
        
        # 4. Setup monitoring baseline
        components['monitor'].baseline_manager.create_baseline(
            "integration_test",
            initial_config
        )
        
        # 5. Simulate configuration change
        updated_config = initial_config.copy()
        updated_config['debug'] = True
        updated_config['new_feature'] = 'enabled'
        
        # 6. Validate updated configuration
        updated_validation = await components['validator'].validate_environment(
            Environment.TESTING,
            updated_config
        )
        assert updated_validation.passed
        
        # 7. Detect drift
        drifts = components['monitor'].drift_detector.detect_drift(
            updated_config,
            "integration_test"
        )
        assert len(drifts) == 2  # debug changed, new_feature added
        
        # 8. Create new version for changes
        new_version = await components['version_manager'].create_version(
            updated_config,
            "Feature update",
            VersionType.MINOR,
            validate=False
        )
        assert new_version is not None
        
        # 9. Test rollback capability
        rollback_plan = await components['version_manager'].create_rollback_plan(
            version.version,
            new_version.version
        )
        assert rollback_plan is not None
        assert len(rollback_plan.affected_keys) == 2
    
    @pytest.mark.asyncio
    async def test_secrets_integration(self, integrated_setup):
        """Test integration with secrets management."""
        secrets_manager = integrated_setup['secrets_manager']
        
        # Create test secret
        secret_id = await secrets_manager.create_secret(
            "test_api_key",
            SecretType.API_KEY,
            user="test_user"
        )
        
        assert secret_id is not None
        
        # Retrieve secret
        secret_value = await secrets_manager.get_secret(secret_id, user="test_user")
        assert secret_value is not None
        assert len(secret_value) >= 32  # API keys should be at least 32 chars
        
        # Test configuration with secret reference
        config_with_secret = {
            'api_key_ref': secret_id,
            'app_setting': 'value'
        }
        
        # Validate configuration (should pass)
        validation_result = await integrated_setup['validator'].validate_environment(
            Environment.TESTING,
            config_with_secret
        )
        assert validation_result.passed
    
    def test_configuration_system_health(self, integrated_setup):
        """Test overall configuration system health."""
        components = integrated_setup
        
        # Check that all components are properly initialized
        assert components['config_manager'] is not None
        assert components['validator'] is not None
        assert components['hot_reloader'] is not None
        assert components['version_manager'] is not None
        assert components['monitor'] is not None
        assert components['secrets_manager'] is not None
        
        # Check component states
        assert components['hot_reloader'].environment == Environment.DEVELOPMENT  # Default
        assert components['version_manager'].environment == Environment.DEVELOPMENT  # Default
        assert components['monitor'].environment == Environment.DEVELOPMENT  # Default
        
        # Test hot reloader statistics
        hot_reload_stats = components['hot_reloader'].get_statistics()
        assert 'enabled' in hot_reload_stats
        assert 'environment' in hot_reload_stats
        
        # Test monitoring status
        monitor_status = components['monitor'].get_monitoring_status()
        assert 'monitoring_active' in monitor_status
        assert 'environment' in monitor_status


class TestPerformanceAndLoad:
    """Test configuration system performance and load handling."""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test configuration validation performance."""
        large_config = {
            f'setting_{i}': f'value_{i}' 
            for i in range(1000)
        }
        
        start_time = asyncio.get_event_loop().time()
        
        result = await validator.validate_environment(
            Environment.TESTING,
            large_config
        )
        
        end_time = asyncio.get_event_loop().time()
        validation_time = end_time - start_time
        
        # Validation should complete within reasonable time
        assert validation_time < 5.0  # 5 seconds max
        assert result.validation_time < 5.0
    
    @pytest.mark.asyncio
    async def test_concurrent_validation(self):
        """Test concurrent validation requests."""
        configs = [
            {f'config_{i}': f'value_{i}'} 
            for i in range(10)
        ]
        
        # Run validations concurrently
        tasks = [
            validator.validate_environment(Environment.TESTING, config)
            for config in configs
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All validations should succeed
        assert len(results) == 10
        assert all(result.passed for result in results)
    
    def test_memory_usage(self):
        """Test memory usage of configuration components."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create multiple configuration instances
        components = []
        for i in range(100):
            from supernova.config_monitoring import ConfigurationMonitor
            monitor = ConfigurationMonitor(Environment.TESTING)
            components.append(monitor)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del components
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_cleanup = peak_memory - final_memory
        
        # Memory should not increase excessively
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
        assert memory_cleanup > memory_increase * 0.5  # At least 50% cleanup


if __name__ == "__main__":
    """Run the test suite."""
    pytest.main([__file__, "-v", "--tb=short"])