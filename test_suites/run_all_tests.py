#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
===============================

This script runs all test suites in the correct order and generates
comprehensive reports. It can be used locally or in CI/CD pipelines.

Test Execution Order:
1. Code quality and static analysis
2. Unit tests with coverage
3. Integration tests
4. Security tests
5. Performance tests
6. Accessibility tests
7. Data integrity tests
8. Error recovery tests
9. End-to-end tests
10. Report generation and production readiness evaluation
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import os
import tempfile
import shutil


class TestSuiteRunner:
    """Comprehensive test suite runner."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Test execution results
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'suites': {},
            'overall_success': False,
            'failed_suites': [],
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0
        }
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent
        self.test_suites_dir = Path(__file__).parent
        self.artifacts_dir = Path(self.config.get('artifacts_dir', './test-artifacts'))
        self.reports_dir = Path(self.config.get('reports_dir', './test-reports'))
        
        # Create directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, 
                   timeout: int = 600, capture_output: bool = True) -> Dict[str, Any]:
        """Run a command and capture results."""
        if cwd is None:
            cwd = self.project_root
        
        self.logger.info(f"Running: {' '.join(command)} in {cwd}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            duration = time.time() - start_time
            
            return {
                'command': ' '.join(command),
                'returncode': result.returncode,
                'stdout': result.stdout if capture_output else '',
                'stderr': result.stderr if capture_output else '',
                'duration': duration,
                'success': result.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            return {
                'command': ' '.join(command),
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'duration': timeout,
                'success': False,
                'timeout': True
            }
        
        except Exception as e:
            return {
                'command': ' '.join(command),
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality and static analysis checks."""
        self.logger.info("Running code quality checks...")
        
        suite_results = {
            'name': 'Code Quality',
            'start_time': time.time(),
            'checks': {},
            'overall_success': True
        }
        
        # Python code formatting (Black)
        if self.config.get('run_black', True):
            result = self.run_command(['black', '--check', '--diff', 'supernova/', 'test_suites/'])
            suite_results['checks']['black'] = result
            if not result['success']:
                self.logger.warning("Black formatting check failed")
                suite_results['overall_success'] = False
        
        # Import sorting (isort)
        if self.config.get('run_isort', True):
            result = self.run_command(['isort', '--check-only', '--diff', 'supernova/', 'test_suites/'])
            suite_results['checks']['isort'] = result
            if not result['success']:
                self.logger.warning("isort check failed")
        
        # Linting (flake8)
        if self.config.get('run_flake8', True):
            result = self.run_command([
                'flake8', 'supernova/', 'test_suites/', 
                '--max-line-length=100', '--ignore=E203,W503'
            ])
            suite_results['checks']['flake8'] = result
            if not result['success']:
                self.logger.warning("flake8 linting failed")
                suite_results['overall_success'] = False
        
        # Type checking (mypy)
        if self.config.get('run_mypy', True):
            result = self.run_command(['mypy', 'supernova/', '--ignore-missing-imports'])
            suite_results['checks']['mypy'] = result
            if not result['success']:
                self.logger.warning("mypy type checking failed")
        
        # Security analysis (bandit)
        if self.config.get('run_bandit', True):
            result = self.run_command([
                'bandit', '-r', 'supernova/', '-f', 'json', 
                '-o', str(self.artifacts_dir / 'bandit-results.json')
            ])
            suite_results['checks']['bandit'] = result
            # Bandit may return non-zero even for warnings, so don't fail overall
        
        suite_results['duration'] = time.time() - suite_results['start_time']
        self.results['suites']['code_quality'] = suite_results
        
        return suite_results['overall_success']
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        self.logger.info("Running unit tests...")
        
        # Run unit tests with coverage
        result = self.run_command([
            'pytest',
            'test_suites/test_unit_api_comprehensive.py',
            'tests/',
            '--cov=supernova',
            '--cov-report=xml:' + str(self.artifacts_dir / 'coverage.xml'),
            '--cov-report=html:' + str(self.artifacts_dir / 'htmlcov'),
            '--cov-report=term-missing',
            '--cov-fail-under=85',
            '--junit-xml=' + str(self.artifacts_dir / 'unit-test-results.xml'),
            '-v'
        ], timeout=900)  # 15 minutes
        
        self.results['suites']['unit_tests'] = {
            'name': 'Unit Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Unit tests passed")
        else:
            self.logger.error("Unit tests failed")
            self.results['failed_suites'].append('unit_tests')
        
        return result['success']
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        self.logger.info("Running integration tests...")
        
        result = self.run_command([
            'pytest',
            'test_suites/test_integration_comprehensive.py',
            '-v',
            '-m', 'integration',
            '--junit-xml=' + str(self.artifacts_dir / 'integration-test-results.xml'),
            '--timeout=300'
        ], timeout=1200)  # 20 minutes
        
        self.results['suites']['integration_tests'] = {
            'name': 'Integration Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Integration tests passed")
        else:
            self.logger.error("Integration tests failed")
            self.results['failed_suites'].append('integration_tests')
        
        return result['success']
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        self.logger.info("Running security tests...")
        
        result = self.run_command([
            'pytest',
            'test_suites/test_security_comprehensive.py',
            '-v',
            '-m', 'security',
            '--junit-xml=' + str(self.artifacts_dir / 'security-test-results.xml')
        ], timeout=900)  # 15 minutes
        
        self.results['suites']['security_tests'] = {
            'name': 'Security Tests',
            'result': result,
            'success': result['success']
        }
        
        # Also run dependency vulnerability check
        safety_result = self.run_command([
            'safety', 'check', '--json', 
            '--output=' + str(self.artifacts_dir / 'safety-results.json')
        ])
        
        if result['success']:
            self.logger.info("Security tests passed")
        else:
            self.logger.error("Security tests failed")
            self.results['failed_suites'].append('security_tests')
        
        return result['success']
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        self.logger.info("Running performance tests...")
        
        # Skip performance tests if disabled
        if not self.config.get('run_performance_tests', True):
            self.logger.info("Performance tests skipped (disabled in config)")
            return True
        
        result = self.run_command([
            'pytest',
            'test_suites/test_performance_comprehensive.py',
            '-v',
            '-m', 'performance',
            '--junit-xml=' + str(self.artifacts_dir / 'performance-test-results.xml'),
            '--timeout=600'
        ], timeout=1800)  # 30 minutes
        
        self.results['suites']['performance_tests'] = {
            'name': 'Performance Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Performance tests passed")
        else:
            self.logger.warning("Performance tests failed (non-blocking)")
            # Performance test failures are warnings, not blockers
        
        return True  # Don't block on performance test failures
    
    def run_accessibility_tests(self) -> bool:
        """Run accessibility tests."""
        self.logger.info("Running accessibility tests...")
        
        # Skip if no display available (CI environment)
        if not self.config.get('run_accessibility_tests', True) or os.environ.get('CI'):
            self.logger.info("Accessibility tests skipped (CI environment or disabled)")
            return True
        
        result = self.run_command([
            'pytest',
            'test_suites/test_accessibility_ux.py',
            '-v',
            '-m', 'accessibility',
            '--junit-xml=' + str(self.artifacts_dir / 'accessibility-test-results.xml')
        ], timeout=900)  # 15 minutes
        
        self.results['suites']['accessibility_tests'] = {
            'name': 'Accessibility Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Accessibility tests passed")
        else:
            self.logger.warning("Accessibility tests failed (non-blocking)")
        
        return True  # Don't block on accessibility test failures
    
    def run_data_integrity_tests(self) -> bool:
        """Run data integrity tests."""
        self.logger.info("Running data integrity tests...")
        
        result = self.run_command([
            'pytest',
            'test_suites/test_data_integrity_comprehensive.py',
            '-v',
            '-m', 'data_integrity',
            '--junit-xml=' + str(self.artifacts_dir / 'data-integrity-test-results.xml')
        ], timeout=900)  # 15 minutes
        
        self.results['suites']['data_integrity_tests'] = {
            'name': 'Data Integrity Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Data integrity tests passed")
        else:
            self.logger.error("Data integrity tests failed")
            self.results['failed_suites'].append('data_integrity_tests')
        
        return result['success']
    
    def run_error_recovery_tests(self) -> bool:
        """Run error recovery tests."""
        self.logger.info("Running error recovery tests...")
        
        result = self.run_command([
            'pytest',
            'test_suites/test_error_recovery_comprehensive.py',
            '-v',
            '-m', 'error_recovery',
            '--junit-xml=' + str(self.artifacts_dir / 'error-recovery-test-results.xml')
        ], timeout=900)  # 15 minutes
        
        self.results['suites']['error_recovery_tests'] = {
            'name': 'Error Recovery Tests',
            'result': result,
            'success': result['success']
        }
        
        if result['success']:
            self.logger.info("Error recovery tests passed")
        else:
            self.logger.warning("Error recovery tests failed (non-blocking)")
        
        return True  # Don't block on error recovery test failures
    
    def generate_comprehensive_report(self) -> bool:
        """Generate comprehensive test report."""
        self.logger.info("Generating comprehensive test report...")
        
        result = self.run_command([
            'python',
            'test_suites/generate_comprehensive_report.py',
            '--artifacts-dir', str(self.artifacts_dir),
            '--output-dir', str(self.reports_dir),
            '--format', 'html,json,markdown'
        ], timeout=300)  # 5 minutes
        
        if result['success']:
            self.logger.info("Comprehensive test report generated")
        else:
            self.logger.error("Failed to generate comprehensive test report")
        
        return result['success']
    
    def evaluate_production_readiness(self) -> bool:
        """Evaluate production readiness."""
        self.logger.info("Evaluating production readiness...")
        
        result = self.run_command([
            'python',
            'test_suites/evaluate_production_readiness.py',
            '--test-results', str(self.artifacts_dir),
            '--output', str(self.reports_dir / 'production-readiness.json')
        ], timeout=300)  # 5 minutes
        
        self.results['production_ready'] = result['returncode'] == 0
        
        if result['success']:
            self.logger.info("Production readiness evaluation completed")
        else:
            self.logger.warning("Production readiness evaluation indicates issues")
        
        return True  # Don't block on readiness evaluation
    
    def run_all_tests(self) -> bool:
        """Run all test suites in order."""
        self.logger.info("Starting comprehensive test suite...")
        self.results['start_time'] = time.time()
        
        # Test execution pipeline
        test_pipeline = [
            ('Code Quality', self.run_code_quality_checks),
            ('Unit Tests', self.run_unit_tests),
            ('Integration Tests', self.run_integration_tests),
            ('Security Tests', self.run_security_tests),
            ('Performance Tests', self.run_performance_tests),
            ('Accessibility Tests', self.run_accessibility_tests),
            ('Data Integrity Tests', self.run_data_integrity_tests),
            ('Error Recovery Tests', self.run_error_recovery_tests),
        ]
        
        # Run tests
        overall_success = True
        for test_name, test_func in test_pipeline:
            self.logger.info(f"=== Running {test_name} ===")
            
            try:
                success = test_func()
                if not success and test_name in ['Unit Tests', 'Integration Tests', 'Security Tests', 'Data Integrity Tests']:
                    # Critical test failures
                    overall_success = False
                    self.logger.error(f"{test_name} failed - marked as critical failure")
                elif not success:
                    # Non-critical test failures
                    self.logger.warning(f"{test_name} failed - continuing with other tests")
            
            except Exception as e:
                self.logger.error(f"Exception during {test_name}: {e}")
                if test_name in ['Unit Tests', 'Integration Tests']:
                    overall_success = False
        
        # Generate reports
        self.logger.info("=== Generating Reports ===")
        self.generate_comprehensive_report()
        self.evaluate_production_readiness()
        
        # Finalize results
        self.results['end_time'] = time.time()
        self.results['duration'] = self.results['end_time'] - self.results['start_time']
        self.results['overall_success'] = overall_success
        
        # Save results
        with open(self.reports_dir / 'test-execution-results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return overall_success
    
    def print_summary(self):
        """Print test execution summary."""
        duration_minutes = self.results['duration'] / 60
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*60}")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Overall Success: {'✅ PASS' if self.results['overall_success'] else '❌ FAIL'}")
        
        print(f"\nTest Suite Results:")
        for suite_name, suite_data in self.results['suites'].items():
            if isinstance(suite_data, dict) and 'success' in suite_data:
                status = '✅ PASS' if suite_data['success'] else '❌ FAIL'
                print(f"  {suite_data['name']}: {status}")
        
        if self.results['failed_suites']:
            print(f"\nFailed Suites:")
            for suite in self.results['failed_suites']:
                print(f"  ❌ {suite}")
        
        print(f"\nReports Generated:")
        print(f"  📄 Artifacts: {self.artifacts_dir}")
        print(f"  📊 Reports: {self.reports_dir}")
        
        if 'production_ready' in self.results:
            ready_status = '✅ READY' if self.results['production_ready'] else '⚠️ NOT READY'
            print(f"  🚀 Production Ready: {ready_status}")


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load test configuration."""
    default_config = {
        'run_black': True,
        'run_isort': True,
        'run_flake8': True,
        'run_mypy': True,
        'run_bandit': True,
        'run_performance_tests': True,
        'run_accessibility_tests': True,
        'artifacts_dir': './test-artifacts',
        'reports_dir': './test-reports',
        'timeout_minutes': 60
    }
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Could not load config file {config_file}: {e}")
    
    return default_config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--artifacts-dir", help="Artifacts directory")
    parser.add_argument("--reports-dir", help="Reports directory")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.artifacts_dir:
        config['artifacts_dir'] = args.artifacts_dir
    if args.reports_dir:
        config['reports_dir'] = args.reports_dir
    if args.skip_slow:
        config['run_performance_tests'] = False
        config['run_accessibility_tests'] = False
    
    # Create test runner
    runner = TestSuiteRunner(config)
    
    try:
        # Run all tests
        success = runner.run_all_tests()
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logging.info("Test execution interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    
    except Exception as e:
        logging.error(f"Unexpected error during test execution: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()