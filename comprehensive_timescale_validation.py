"""
Comprehensive TimescaleDB Sentiment Feature Store Validation Suite

This validation script provides thorough testing and validation for the
SuperNova TimescaleDB sentiment feature store implementation.

VALIDATION AREAS COVERED:
- Code structure and imports validation
- Database integration and connection testing
- Hypertable creation and configuration
- Data insertion and retrieval operations
- Prefect workflow integration testing
- API endpoint functionality testing
- Configuration management validation
- Performance benchmarking
- Error handling and edge cases

Usage:
    python comprehensive_timescale_validation.py [--config-only] [--skip-db] [--performance]
"""

import asyncio
import json
import logging
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimescaleValidationSuite:
    """Comprehensive validation suite for TimescaleDB sentiment feature store"""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "recommendations": [],
            "critical_issues": [],
            "warnings": [],
            "summary": {}
        }
        self.test_counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
    def log_test_result(self, test_name: str, status: str, details: Dict = None, 
                       duration_ms: float = None):
        """Log individual test results"""
        self.test_counts["total"] += 1
        self.test_counts[status] += 1
        
        result = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "duration_ms": duration_ms
        }
        
        self.results["test_results"][test_name] = result
        
        # Log to console
        status_icon = {
            "passed": "✅",
            "failed": "❌", 
            "skipped": "⏭️"
        }
        logger.info(f"{status_icon.get(status, '❓')} {test_name}: {status.upper()}")
        
        if status == "failed" and details:
            logger.error(f"   Error: {details.get('error', 'Unknown error')}")
    
    def add_recommendation(self, recommendation: str, priority: str = "medium"):
        """Add a recommendation to the results"""
        self.results["recommendations"].append({
            "text": recommendation,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_critical_issue(self, issue: str, test_name: str = None):
        """Add a critical issue that requires immediate attention"""
        self.results["critical_issues"].append({
            "issue": issue,
            "test": test_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_warning(self, warning: str, test_name: str = None):
        """Add a warning about potential issues"""
        self.results["warnings"].append({
            "warning": warning,
            "test": test_name,
            "timestamp": datetime.now().isoformat()
        })

    # ========================================
    # CODE STRUCTURE VALIDATION
    # ========================================
    
    def validate_code_structure(self) -> bool:
        """Validate the structure and imports of sentiment-related modules"""
        logger.info("=== VALIDATING CODE STRUCTURE ===")
        
        start_time = time.time()
        all_passed = True
        
        # Required modules and their expected components
        module_checks = {
            "supernova.sentiment": [
                "SentimentResult", "SentimentSignal", "SentimentSource", "MarketRegime",
                "score_text", "generate_sentiment_signal", "TwitterConnector", 
                "RedditConnector", "NewsConnector"
            ],
            "supernova.sentiment_models": [
                "SentimentData", "SentimentAggregates", "SentimentAlerts", 
                "SentimentMetadata", "TimescaleBase", "get_timescale_engine"
            ],
            "supernova.timescale_setup": [
                "TimescaleSetupManager", "initialize_timescale_database", 
                "get_database_health", "run_maintenance"
            ],
            "supernova.workflows": [
                "sentiment_analysis_flow", "batch_sentiment_analysis_flow", 
                "is_prefect_available", "PREFECT_AVAILABLE"
            ]
        }
        
        for module_name, expected_components in module_checks.items():
            try:
                # Import the module
                module = __import__(module_name, fromlist=expected_components)
                missing_components = []
                
                # Check for expected components
                for component in expected_components:
                    if not hasattr(module, component):
                        missing_components.append(component)
                
                if missing_components:
                    self.log_test_result(
                        f"structure_{module_name.split('.')[-1]}_components",
                        "failed",
                        {"missing_components": missing_components}
                    )
                    all_passed = False
                    self.add_critical_issue(
                        f"Missing required components in {module_name}: {missing_components}",
                        f"structure_{module_name.split('.')[-1]}_components"
                    )
                else:
                    self.log_test_result(
                        f"structure_{module_name.split('.')[-1]}_components",
                        "passed",
                        {"components_found": len(expected_components)}
                    )
                    
            except ImportError as e:
                self.log_test_result(
                    f"structure_{module_name.split('.')[-1]}_import",
                    "failed",
                    {"error": str(e), "module": module_name}
                )
                all_passed = False
                self.add_critical_issue(
                    f"Cannot import required module {module_name}: {str(e)}",
                    f"structure_{module_name.split('.')[-1]}_import"
                )
        
        # Check file existence
        required_files = [
            "supernova/sentiment.py",
            "supernova/sentiment_models.py", 
            "supernova/timescale_setup.py",
            "supernova/workflows.py"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_test_result(f"structure_file_{Path(file_path).name}", "passed")
            else:
                self.log_test_result(
                    f"structure_file_{Path(file_path).name}", 
                    "failed",
                    {"error": f"File not found: {file_path}"}
                )
                all_passed = False
                self.add_critical_issue(f"Required file missing: {file_path}")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # DATABASE INTEGRATION TESTING
    # ========================================
    
    def validate_database_integration(self) -> bool:
        """Test TimescaleDB connection and basic database operations"""
        logger.info("=== VALIDATING DATABASE INTEGRATION ===")
        
        start_time = time.time()
        all_passed = True
        
        try:
            from supernova.timescale_setup import TimescaleSetupManager
            from supernova.db import is_timescale_available
            
            # Test TimescaleDB availability check
            try:
                timescale_available = is_timescale_available()
                self.log_test_result(
                    "database_timescale_availability_check",
                    "passed" if timescale_available else "failed",
                    {"available": timescale_available}
                )
                
                if not timescale_available:
                    self.add_warning("TimescaleDB not available - database tests will be limited")
                    
            except Exception as e:
                self.log_test_result(
                    "database_timescale_availability_check",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
            
            # Test setup manager initialization
            try:
                manager = TimescaleSetupManager()
                self.log_test_result("database_setup_manager_init", "passed")
                
                # Test connection (if TimescaleDB is configured)
                if is_timescale_available():
                    try:
                        success, error_msg = manager.test_connection()
                        self.log_test_result(
                            "database_connection_test",
                            "passed" if success else "failed",
                            {"success": success, "error": error_msg}
                        )
                        
                        if not success:
                            all_passed = False
                            self.add_critical_issue(f"Database connection failed: {error_msg}")
                            
                    except Exception as e:
                        self.log_test_result(
                            "database_connection_test",
                            "failed", 
                            {"error": str(e)}
                        )
                        all_passed = False
                        
                else:
                    self.log_test_result("database_connection_test", "skipped", 
                                       {"reason": "TimescaleDB not configured"})
                    
            except Exception as e:
                self.log_test_result(
                    "database_setup_manager_init",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
                
        except ImportError as e:
            self.log_test_result(
                "database_imports",
                "failed", 
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Cannot import database modules for testing")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    def validate_hypertable_setup(self) -> bool:
        """Validate hypertable creation and configuration"""
        logger.info("=== VALIDATING HYPERTABLE SETUP ===")
        
        start_time = time.time()
        all_passed = True
        
        try:
            from supernova.timescale_setup import TimescaleSetupManager
            from supernova.db import is_timescale_available
            
            if not is_timescale_available():
                self.log_test_result("hypertable_setup", "skipped", 
                                   {"reason": "TimescaleDB not available"})
                return True
            
            manager = TimescaleSetupManager()
            
            # Test schema creation
            try:
                schema_success = manager.create_database_schema()
                self.log_test_result(
                    "hypertable_schema_creation",
                    "passed" if schema_success else "failed",
                    {"success": schema_success}
                )
                
                if not schema_success:
                    all_passed = False
                    self.add_critical_issue("Failed to create database schema")
                    
            except Exception as e:
                self.log_test_result(
                    "hypertable_schema_creation",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
            
            # Test hypertable creation
            try:
                hypertable_success = manager.create_hypertables()
                self.log_test_result(
                    "hypertable_creation",
                    "passed" if hypertable_success else "failed",
                    {"success": hypertable_success}
                )
                
                if not hypertable_success:
                    all_passed = False
                    self.add_critical_issue("Failed to create hypertables")
                    
            except Exception as e:
                self.log_test_result(
                    "hypertable_creation",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
            
            # Test continuous aggregates
            try:
                agg_success = manager.create_continuous_aggregates()
                self.log_test_result(
                    "hypertable_continuous_aggregates",
                    "passed" if agg_success else "failed",
                    {"success": agg_success}
                )
                
                if not agg_success:
                    self.add_warning("Continuous aggregates creation failed - may impact query performance")
                    
            except Exception as e:
                self.log_test_result(
                    "hypertable_continuous_aggregates",
                    "failed",
                    {"error": str(e)}
                )
                self.add_warning("Continuous aggregates setup failed")
            
            # Test compression policies
            try:
                compression_success = manager.setup_compression_policies()
                self.log_test_result(
                    "hypertable_compression_policies",
                    "passed" if compression_success else "failed",
                    {"success": compression_success}
                )
                
                if not compression_success:
                    self.add_warning("Compression policies setup failed - may impact storage efficiency")
                    
            except Exception as e:
                self.log_test_result(
                    "hypertable_compression_policies",
                    "failed",
                    {"error": str(e)}
                )
                self.add_warning("Compression policies setup failed")
                
        except Exception as e:
            self.log_test_result(
                "hypertable_validation",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Hypertable validation failed with exception")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # DATA OPERATIONS TESTING
    # ========================================
    
    def validate_data_operations(self) -> bool:
        """Test data insertion, retrieval, and manipulation"""
        logger.info("=== VALIDATING DATA OPERATIONS ===")
        
        start_time = time.time()
        all_passed = True
        
        try:
            from supernova.sentiment_models import (
                SentimentData, SentimentSignal, get_timescale_session, 
                from_sentiment_signal
            )
            from supernova.sentiment import SentimentSource, MarketRegime
            from supernova.db import is_timescale_available
            
            if not is_timescale_available():
                self.log_test_result("data_operations", "skipped",
                                   {"reason": "TimescaleDB not available"})
                return True
            
            # Test model creation
            try:
                test_signal = SentimentSignal(
                    overall_score=0.5,
                    confidence=0.8,
                    source_breakdown={SentimentSource.TWITTER: 0.6},
                    figure_influence=0.1,
                    news_impact=0.4,
                    social_momentum=0.2,
                    contrarian_indicator=0.0,
                    regime_adjusted_score=0.5,
                    timestamp=datetime.now(timezone.utc)
                )
                
                sentiment_data = from_sentiment_signal(test_signal, "TEST")
                
                self.log_test_result(
                    "data_model_creation",
                    "passed",
                    {"model_created": True, "symbol": sentiment_data.symbol}
                )
                
            except Exception as e:
                self.log_test_result(
                    "data_model_creation",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
                self.add_critical_issue("Failed to create sentiment data model")
            
            # Test database insertion (if we have a valid model)
            if 'sentiment_data' in locals():
                try:
                    TimescaleSession = get_timescale_session()
                    
                    with TimescaleSession() as session:
                        session.add(sentiment_data)
                        session.commit()
                        
                        self.log_test_result(
                            "data_insertion",
                            "passed",
                            {"inserted_symbol": sentiment_data.symbol}
                        )
                        
                        # Test retrieval
                        retrieved = session.query(SentimentData).filter(
                            SentimentData.symbol == "TEST"
                        ).first()
                        
                        if retrieved:
                            self.log_test_result(
                                "data_retrieval",
                                "passed",
                                {"retrieved_symbol": retrieved.symbol, "score": retrieved.overall_score}
                            )
                        else:
                            self.log_test_result(
                                "data_retrieval",
                                "failed",
                                {"error": "No data retrieved"}
                            )
                            all_passed = False
                        
                        # Cleanup test data
                        session.delete(retrieved)
                        session.commit()
                        
                except Exception as e:
                    self.log_test_result(
                        "data_insertion",
                        "failed",
                        {"error": str(e)}
                    )
                    all_passed = False
                    self.add_critical_issue("Failed to insert/retrieve test data")
                    
        except ImportError as e:
            self.log_test_result(
                "data_operations_imports",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Cannot import required modules for data operations testing")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # PREFECT WORKFLOW TESTING
    # ========================================
    
    def validate_prefect_integration(self) -> bool:
        """Test Prefect workflow integration"""
        logger.info("=== VALIDATING PREFECT INTEGRATION ===")
        
        start_time = time.time()
        all_passed = True
        
        try:
            from supernova.workflows import (
                is_prefect_available, PREFECT_AVAILABLE,
                sentiment_analysis_flow, batch_sentiment_analysis_flow
            )
            
            # Test Prefect availability
            prefect_available = is_prefect_available()
            prefect_installed = PREFECT_AVAILABLE
            
            self.log_test_result(
                "prefect_availability_check",
                "passed",
                {
                    "prefect_installed": prefect_installed,
                    "prefect_configured": prefect_available
                }
            )
            
            if not prefect_installed:
                self.add_warning("Prefect not installed - workflow features limited")
            
            if not prefect_available:
                self.add_warning("Prefect not enabled in configuration")
            
            # Test flow definitions exist
            flow_tests = [
                ("sentiment_analysis_flow", sentiment_analysis_flow),
                ("batch_sentiment_analysis_flow", batch_sentiment_analysis_flow)
            ]
            
            for flow_name, flow_func in flow_tests:
                if callable(flow_func):
                    self.log_test_result(
                        f"prefect_flow_{flow_name}",
                        "passed",
                        {"callable": True}
                    )
                else:
                    self.log_test_result(
                        f"prefect_flow_{flow_name}",
                        "failed",
                        {"error": "Flow not callable"}
                    )
                    all_passed = False
            
            # Test task imports
            try:
                from supernova.workflows import (
                    fetch_twitter_data_task, fetch_reddit_data_task,
                    fetch_news_data_task, analyze_sentiment_batch_task
                )
                
                self.log_test_result(
                    "prefect_task_imports",
                    "passed",
                    {"tasks_imported": 4}
                )
                
            except ImportError as e:
                self.log_test_result(
                    "prefect_task_imports",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
                
        except ImportError as e:
            self.log_test_result(
                "prefect_module_import",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Cannot import Prefect workflow modules")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # API ENDPOINTS TESTING
    # ========================================
    
    def validate_api_endpoints(self) -> bool:
        """Test API endpoints for sentiment functionality"""
        logger.info("=== VALIDATING API ENDPOINTS ===")
        
        start_time = time.time()
        all_passed = True
        
        try:
            from supernova.api import app
            from fastapi.testclient import TestClient
            
            # Create test client
            client = TestClient(app)
            
            # Test basic API health
            try:
                # Test if the app can be created
                self.log_test_result(
                    "api_app_creation",
                    "passed",
                    {"app_created": True}
                )
                
                # Check for sentiment endpoints in routes
                sentiment_routes = []
                for route in app.routes:
                    if hasattr(route, 'path') and 'sentiment' in route.path:
                        sentiment_routes.append(route.path)
                
                if sentiment_routes:
                    self.log_test_result(
                        "api_sentiment_routes",
                        "passed",
                        {"routes_found": len(sentiment_routes), "routes": sentiment_routes}
                    )
                else:
                    self.log_test_result(
                        "api_sentiment_routes",
                        "failed",
                        {"error": "No sentiment routes found"}
                    )
                    all_passed = False
                    self.add_critical_issue("Sentiment API endpoints not found")
                
            except Exception as e:
                self.log_test_result(
                    "api_validation",
                    "failed",
                    {"error": str(e)}
                )
                all_passed = False
                
        except ImportError as e:
            self.log_test_result(
                "api_imports",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Cannot import API modules for testing")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # CONFIGURATION VALIDATION
    # ========================================
    
    def validate_configuration(self) -> bool:
        """Validate configuration and environment setup"""
        logger.info("=== VALIDATING CONFIGURATION ===")
        
        start_time = time.time()
        all_passed = True
        
        # Test configuration imports
        try:
            from supernova.config import settings
            
            self.log_test_result(
                "config_import",
                "passed",
                {"settings_available": True}
            )
            
            # Check critical TimescaleDB settings
            timescale_settings = [
                'TIMESCALE_HOST', 'TIMESCALE_PORT', 'TIMESCALE_DB',
                'TIMESCALE_USER', 'TIMESCALE_PASSWORD'
            ]
            
            missing_settings = []
            for setting in timescale_settings:
                if not hasattr(settings, setting):
                    missing_settings.append(setting)
            
            if missing_settings:
                self.log_test_result(
                    "config_timescale_settings",
                    "failed",
                    {"missing_settings": missing_settings}
                )
                all_passed = False
                self.add_critical_issue(f"Missing TimescaleDB settings: {missing_settings}")
            else:
                self.log_test_result(
                    "config_timescale_settings",
                    "passed",
                    {"all_settings_present": True}
                )
            
            # Check optional configuration
            optional_settings = [
                'ENABLE_PREFECT', 'PREFECT_TWITTER_RETRIES', 'PREFECT_NEWS_RETRIES',
                'TIMESCALE_COMPRESSION_AFTER', 'TIMESCALE_RETENTION_POLICY'
            ]
            
            present_optional = []
            for setting in optional_settings:
                if hasattr(settings, setting):
                    present_optional.append(setting)
            
            self.log_test_result(
                "config_optional_settings",
                "passed",
                {"optional_settings_found": len(present_optional), "settings": present_optional}
            )
            
            # Check API key availability
            api_settings = ['X_BEARER_TOKEN', 'REDDIT_CLIENT_ID', 'NEWSAPI_KEY']
            available_apis = []
            for setting in api_settings:
                if hasattr(settings, setting) and getattr(settings, setting):
                    available_apis.append(setting)
            
            if not available_apis:
                self.add_warning("No social media API keys configured - sentiment data will be limited")
            
            self.log_test_result(
                "config_api_keys",
                "passed",
                {"api_keys_configured": len(available_apis), "apis": available_apis}
            )
            
        except ImportError as e:
            self.log_test_result(
                "config_import",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
            self.add_critical_issue("Cannot import configuration module")
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # PERFORMANCE TESTING
    # ========================================
    
    def validate_performance(self) -> bool:
        """Run performance benchmarks and tests"""
        logger.info("=== VALIDATING PERFORMANCE ===")
        
        start_time = time.time()
        all_passed = True
        
        # Test sentiment analysis performance
        try:
            from supernova.sentiment import score_text
            
            test_texts = [
                "Apple stock surged today on positive earnings report",
                "Market volatility continues as investors worry about inflation",
                "Tesla beats expectations with strong Q3 results and raised guidance",
                "Federal Reserve hints at potential rate cuts in upcoming meeting",
                "Tech sector shows resilience despite broader market concerns"
            ]
            
            analysis_times = []
            
            for text in test_texts:
                text_start = time.time()
                result = score_text(text)
                text_duration = (time.time() - text_start) * 1000
                analysis_times.append(text_duration)
            
            avg_analysis_time = sum(analysis_times) / len(analysis_times)
            
            self.log_test_result(
                "performance_sentiment_analysis",
                "passed",
                {
                    "avg_analysis_time_ms": round(avg_analysis_time, 2),
                    "max_time_ms": round(max(analysis_times), 2),
                    "min_time_ms": round(min(analysis_times), 2),
                    "texts_analyzed": len(test_texts)
                }
            )
            
            self.results["performance_metrics"]["sentiment_analysis"] = {
                "avg_time_ms": avg_analysis_time,
                "max_time_ms": max(analysis_times),
                "min_time_ms": min(analysis_times)
            }
            
            if avg_analysis_time > 1000:  # More than 1 second
                self.add_warning("Sentiment analysis is slow - consider optimization")
                
        except Exception as e:
            self.log_test_result(
                "performance_sentiment_analysis",
                "failed",
                {"error": str(e)}
            )
            all_passed = False
        
        # Test database query performance (if available)
        try:
            from supernova.db import is_timescale_available
            from supernova.sentiment_models import get_timescale_session, SentimentData
            
            if is_timescale_available():
                TimescaleSession = get_timescale_session()
                
                with TimescaleSession() as session:
                    query_start = time.time()
                    # Simple count query
                    count = session.query(SentimentData).count()
                    query_duration = (time.time() - query_start) * 1000
                    
                    self.log_test_result(
                        "performance_database_query",
                        "passed",
                        {
                            "query_time_ms": round(query_duration, 2),
                            "record_count": count
                        }
                    )
                    
                    self.results["performance_metrics"]["database_query"] = {
                        "query_time_ms": query_duration,
                        "record_count": count
                    }
                    
            else:
                self.log_test_result(
                    "performance_database_query",
                    "skipped",
                    {"reason": "TimescaleDB not available"}
                )
                
        except Exception as e:
            self.log_test_result(
                "performance_database_query",
                "failed",
                {"error": str(e)}
            )
        
        duration = (time.time() - start_time) * 1000
        return all_passed
    
    # ========================================
    # MAIN VALIDATION RUNNER
    # ========================================
    
    def run_comprehensive_validation(self, skip_db: bool = False, 
                                   performance: bool = True) -> Dict:
        """Run the complete validation suite"""
        logger.info("🚀 STARTING COMPREHENSIVE TIMESCALEDB SENTIMENT VALIDATION")
        logger.info("=" * 70)
        
        validation_start = time.time()
        
        # Run all validation components
        validation_results = {}
        
        # 1. Code Structure Validation
        validation_results["code_structure"] = self.validate_code_structure()
        
        # 2. Configuration Validation
        validation_results["configuration"] = self.validate_configuration()
        
        # 3. Database Integration (unless skipped)
        if not skip_db:
            validation_results["database_integration"] = self.validate_database_integration()
            validation_results["hypertable_setup"] = self.validate_hypertable_setup()
            validation_results["data_operations"] = self.validate_data_operations()
        else:
            logger.info("⏭️ Skipping database tests as requested")
            self.log_test_result("database_tests", "skipped", {"reason": "User requested skip"})
        
        # 4. Prefect Integration
        validation_results["prefect_integration"] = self.validate_prefect_integration()
        
        # 5. API Endpoints
        validation_results["api_endpoints"] = self.validate_api_endpoints()
        
        # 6. Performance Testing (if enabled)
        if performance:
            validation_results["performance"] = self.validate_performance()
        else:
            logger.info("⏭️ Skipping performance tests as requested")
        
        # Calculate overall validation time
        total_duration = (time.time() - validation_start) * 1000
        
        # Compile final results
        overall_success = all(validation_results.values())
        
        # Generate summary
        self.results["summary"] = {
            "overall_success": overall_success,
            "total_tests": self.test_counts["total"],
            "passed_tests": self.test_counts["passed"],
            "failed_tests": self.test_counts["failed"],
            "skipped_tests": self.test_counts["skipped"],
            "success_rate": (self.test_counts["passed"] / self.test_counts["total"] * 100) if self.test_counts["total"] > 0 else 0,
            "total_duration_ms": total_duration,
            "validation_components": validation_results
        }
        
        # Generate final recommendations
        self._generate_final_recommendations(overall_success)
        
        # Print summary
        self._print_validation_summary()
        
        return self.results
    
    def _generate_final_recommendations(self, overall_success: bool):
        """Generate final recommendations based on validation results"""
        
        if not overall_success:
            self.add_recommendation(
                "Critical issues were found that require immediate attention before production use",
                "high"
            )
        
        if self.test_counts["failed"] > 0:
            self.add_recommendation(
                f"Address {self.test_counts['failed']} failed tests before deployment",
                "high"
            )
        
        if len(self.results["warnings"]) > 0:
            self.add_recommendation(
                f"Review {len(self.results['warnings'])} warnings for potential improvements",
                "medium"
            )
        
        # Database-specific recommendations
        if "database_integration" in self.results["test_results"]:
            if self.results["test_results"]["database_integration"]["status"] == "failed":
                self.add_recommendation(
                    "Configure TimescaleDB connection for full sentiment feature functionality",
                    "high"
                )
        
        # Performance recommendations
        if "performance_metrics" in self.results and "sentiment_analysis" in self.results["performance_metrics"]:
            avg_time = self.results["performance_metrics"]["sentiment_analysis"]["avg_time_ms"]
            if avg_time > 500:
                self.add_recommendation(
                    "Consider optimizing sentiment analysis performance for production workloads",
                    "medium"
                )
        
        # API recommendations
        api_tests = [k for k in self.results["test_results"].keys() if k.startswith("api_")]
        failed_api_tests = [k for k in api_tests if self.results["test_results"][k]["status"] == "failed"]
        
        if failed_api_tests:
            self.add_recommendation(
                "Fix API endpoint issues to ensure full functionality",
                "high"
            )
        
        # General recommendations
        self.add_recommendation(
            "Set up monitoring and alerting for TimescaleDB sentiment feature store in production",
            "medium"
        )
        
        self.add_recommendation(
            "Consider implementing automated backups for TimescaleDB sentiment data",
            "low"
        )
        
        self.add_recommendation(
            "Schedule regular performance monitoring and optimization reviews",
            "low"
        )
    
    def _print_validation_summary(self):
        """Print a comprehensive validation summary"""
        logger.info("=" * 70)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        summary = self.results["summary"]
        
        # Overall status
        status_emoji = "✅" if summary["overall_success"] else "❌"
        logger.info(f"{status_emoji} Overall Status: {'PASSED' if summary['overall_success'] else 'FAILED'}")
        logger.info("")
        
        # Test statistics
        logger.info("📈 Test Statistics:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Passed: {summary['passed_tests']} ✅")
        logger.info(f"   Failed: {summary['failed_tests']} ❌")
        logger.info(f"   Skipped: {summary['skipped_tests']} ⏭️")
        logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Duration: {summary['total_duration_ms']:.0f}ms")
        logger.info("")
        
        # Component results
        logger.info("🔍 Component Results:")
        for component, success in summary["validation_components"].items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"   {component}: {status}")
        logger.info("")
        
        # Critical issues
        if self.results["critical_issues"]:
            logger.info("🚨 Critical Issues:")
            for i, issue in enumerate(self.results["critical_issues"], 1):
                logger.info(f"   {i}. {issue['issue']}")
            logger.info("")
        
        # Warnings
        if self.results["warnings"]:
            logger.info("⚠️ Warnings:")
            for i, warning in enumerate(self.results["warnings"], 1):
                logger.info(f"   {i}. {warning['warning']}")
            logger.info("")
        
        # High priority recommendations
        high_priority_recs = [r for r in self.results["recommendations"] if r["priority"] == "high"]
        if high_priority_recs:
            logger.info("🔥 High Priority Recommendations:")
            for i, rec in enumerate(high_priority_recs, 1):
                logger.info(f"   {i}. {rec['text']}")
            logger.info("")
        
        logger.info("=" * 70)
        logger.info("✨ Validation Complete!")
        logger.info("=" * 70)

def main():
    """Main validation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TimescaleDB Sentiment Feature Store Validation")
    parser.add_argument("--config-only", action="store_true", 
                       help="Only validate configuration, skip functional tests")
    parser.add_argument("--skip-db", action="store_true",
                       help="Skip database integration tests")
    parser.add_argument("--performance", action="store_true", default=True,
                       help="Include performance benchmarking")
    parser.add_argument("--output", type=str, default="timescale_validation_results.json",
                       help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Create validation suite
    validator = TimescaleValidationSuite()
    
    try:
        # Run validation
        if args.config_only:
            # Only run configuration validation
            logger.info("🔧 CONFIGURATION-ONLY VALIDATION MODE")
            results = {"config_validation": validator.validate_configuration()}
        else:
            # Run comprehensive validation
            results = validator.run_comprehensive_validation(
                skip_db=args.skip_db,
                performance=args.performance
            )
        
        # Save detailed results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {args.output}")
        
        # Exit with appropriate code
        if results.get("summary", {}).get("overall_success", False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⛔ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Validation failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()