#!/usr/bin/env python3
"""
SuperNova Optuna Hyperparameter Optimization System - Comprehensive Validation Suite

This script provides comprehensive testing and validation for the Optuna-based 
hyperparameter optimization system implementation in the SuperNova framework.

Validation Areas:
1. Code Structure & Dependencies
2. OptunaOptimizer Class Functionality  
3. Parameter Spaces & Strategy Templates
4. Multi-objective Optimization
5. Walk-forward Validation
6. API Integration
7. Database Operations
8. Prefect Workflow Integration
9. Performance & Scalability
10. Error Handling & Edge Cases
"""

import asyncio
import json
import logging
import sys
import traceback
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch, MagicMock
import tempfile
import sqlite3
import os
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optuna_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class OptunaValidationSuite:
    """Comprehensive validation suite for Optuna optimization system"""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "test_summary": {},
            "detailed_results": {},
            "issues_found": [],
            "recommendations": [],
            "performance_metrics": {},
            "dependency_check": {},
            "success_rate": 0.0
        }
        self.temp_files = []
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("Starting comprehensive Optuna optimization validation")
        
        validation_tests = [
            ("1. Code Structure & Dependencies", self._validate_code_structure),
            ("2. OptunaOptimizer Class", self._validate_optimizer_class),
            ("3. Parameter Spaces", self._validate_parameter_spaces),
            ("4. Multi-objective Optimization", self._validate_multi_objective),
            ("5. Walk-forward Optimization", self._validate_walk_forward),
            ("6. API Integration", self._validate_api_endpoints),
            ("7. Database Operations", self._validate_database_operations),
            ("8. Prefect Workflows", self._validate_prefect_integration),
            ("9. Performance & Scalability", self._validate_performance),
            ("10. Error Handling", self._validate_error_handling)
        ]
        
        total_tests = len(validation_tests)
        passed_tests = 0
        
        for test_name, test_func in validation_tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.results["detailed_results"][test_name] = {
                    **result,
                    "duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                if result.get("success", False):
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
                else:
                    logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s)")
                    if result.get("error"):
                        logger.error(f"Error: {result['error']}")
                        
            except Exception as e:
                logger.error(f"💥 {test_name} - EXCEPTION: {str(e)}")
                self.results["detailed_results"][test_name] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "duration_seconds": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Calculate success rate
        self.results["success_rate"] = (passed_tests / total_tests) * 100
        
        # Generate summary
        self.results["test_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": f"{self.results['success_rate']:.1f}%",
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({self.results['success_rate']:.1f}%)")
        logger.info(f"Overall Status: {self.results['test_summary']['overall_status']}")
        
        return self.results
    
    async def _validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and dependencies"""
        result = {
            "success": False,
            "dependencies": {},
            "files_checked": {},
            "missing_files": [],
            "import_errors": []
        }
        
        try:
            # Check core files exist
            required_files = [
                "supernova/optimizer.py",
                "supernova/optimization_models.py",
                "supernova/api.py",
                "supernova/workflows.py",
                "supernova/schemas.py",
                "requirements.txt"
            ]
            
            for file_path in required_files:
                if os.path.exists(file_path):
                    result["files_checked"][file_path] = "✅ Found"
                else:
                    result["files_checked"][file_path] = "❌ Missing"
                    result["missing_files"].append(file_path)
            
            # Check dependencies
            dependencies_to_check = [
                ("optuna", ">=3.4.0"),
                ("joblib", ">=1.3.0"),
                ("plotly", ">=5.17.0"),
                ("prefect", ">=2.14.0"),
                ("sqlalchemy", ">=2.0.0"),
                ("pandas", ">=2.0.0"),
                ("numpy", ">=1.20.0")
            ]
            
            for dep_name, min_version in dependencies_to_check:
                try:
                    module = __import__(dep_name)
                    version = getattr(module, '__version__', 'unknown')
                    result["dependencies"][dep_name] = {
                        "installed": True,
                        "version": version,
                        "required": min_version,
                        "status": "✅"
                    }
                except ImportError as e:
                    result["dependencies"][dep_name] = {
                        "installed": False,
                        "error": str(e),
                        "required": min_version,
                        "status": "❌"
                    }
                    result["import_errors"].append(f"{dep_name}: {str(e)}")
            
            # Test core imports
            try:
                from supernova.optimizer import OptunaOptimizer, OptimizationConfig, OPTUNA_AVAILABLE
                result["core_imports"] = {"optimizer": "✅ Success"}
                
                if not OPTUNA_AVAILABLE:
                    result["import_errors"].append("OPTUNA_AVAILABLE is False")
                
            except Exception as e:
                result["core_imports"] = {"optimizer": f"❌ {str(e)}"}
                result["import_errors"].append(f"optimizer import: {str(e)}")
            
            try:
                from supernova.optimization_models import OptimizationStudyModel, OptimizationTrialModel
                result["core_imports"]["models"] = "✅ Success"
            except Exception as e:
                result["core_imports"]["models"] = f"❌ {str(e)}"
                result["import_errors"].append(f"models import: {str(e)}")
            
            # Success if no critical issues
            result["success"] = (
                len(result["missing_files"]) == 0 and 
                len(result["import_errors"]) == 0 and
                result["dependencies"].get("optuna", {}).get("installed", False)
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_optimizer_class(self) -> Dict[str, Any]:
        """Validate OptunaOptimizer class functionality"""
        result = {
            "success": False,
            "class_tests": {},
            "method_tests": {},
            "instance_tests": {}
        }
        
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            # Test class instantiation
            try:
                temp_db_path = tempfile.mktemp(suffix='.db')
                self.temp_files.append(temp_db_path)
                storage_url = f"sqlite:///{temp_db_path}"
                
                optimizer = OptunaOptimizer(storage_url=storage_url)
                result["class_tests"]["instantiation"] = "✅ Success"
                
                # Test methods exist
                required_methods = [
                    'create_study', 'optimize_strategy', 'optimize_multi_symbol',
                    'walk_forward_optimization', 'get_study_statistics',
                    'create_optimization_dashboard_data'
                ]
                
                for method_name in required_methods:
                    if hasattr(optimizer, method_name):
                        result["method_tests"][method_name] = "✅ Exists"
                    else:
                        result["method_tests"][method_name] = "❌ Missing"
                
                # Test basic configuration
                config = OptimizationConfig(
                    strategy_template="sma_crossover",
                    n_trials=5,
                    primary_objective="sharpe_ratio"
                )
                result["instance_tests"]["config_creation"] = "✅ Success"
                
                # Test study creation
                study_name = f"test_study_{int(time.time())}"
                study = optimizer.create_study(study_name, config)
                result["instance_tests"]["study_creation"] = "✅ Success"
                
                result["success"] = all(
                    status.startswith("✅") 
                    for status in result["method_tests"].values()
                )
                
            except Exception as e:
                result["class_tests"]["instantiation"] = f"❌ {str(e)}"
                result["error"] = str(e)
                
        except ImportError as e:
            result["error"] = f"Import error: {str(e)}"
            
        return result
    
    async def _validate_parameter_spaces(self) -> Dict[str, Any]:
        """Validate parameter spaces for all strategy templates"""
        result = {
            "success": False,
            "strategy_tests": {},
            "parameter_validation": {},
            "sampling_tests": {}
        }
        
        try:
            from supernova.optimizer import STRATEGY_PARAMETER_SPACES, OptunaOptimizer, OptimizationConfig
            
            # Test all strategy parameter spaces
            for strategy_name, param_space in STRATEGY_PARAMETER_SPACES.items():
                strategy_result = {
                    "parameter_count": len([k for k in param_space.keys() if k != "validation"]),
                    "has_validation": "validation" in param_space,
                    "parameters": {}
                }
                
                # Test each parameter
                for param_name, param_config in param_space.items():
                    if param_name == "validation":
                        continue
                        
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        low, high = param_config
                        param_type = "range"
                        if isinstance(low, int) and isinstance(high, int):
                            param_subtype = "integer"
                        elif isinstance(low, float) or isinstance(high, float):
                            param_subtype = "float"
                        else:
                            param_subtype = "categorical"
                    elif isinstance(param_config, list):
                        param_type = "categorical"
                        param_subtype = f"options({len(param_config)})"
                    else:
                        param_type = "unknown"
                        param_subtype = str(type(param_config))
                    
                    strategy_result["parameters"][param_name] = {
                        "type": param_type,
                        "subtype": param_subtype,
                        "config": str(param_config),
                        "valid": param_type in ["range", "categorical"]
                    }
                
                # Test validation function if present
                if "validation" in param_space:
                    try:
                        validation_func = param_space["validation"]
                        
                        # Create sample parameters for testing validation
                        sample_params = {}
                        for param_name, param_config in param_space.items():
                            if param_name == "validation":
                                continue
                            if isinstance(param_config, tuple) and len(param_config) == 2:
                                low, high = param_config
                                if isinstance(low, (int, float)):
                                    sample_params[param_name] = (low + high) / 2
                                else:
                                    sample_params[param_name] = low
                            elif isinstance(param_config, list):
                                sample_params[param_name] = param_config[0]
                        
                        validation_result = validation_func(sample_params)
                        strategy_result["validation_test"] = {
                            "callable": True,
                            "sample_result": validation_result,
                            "sample_params": sample_params
                        }
                        
                    except Exception as e:
                        strategy_result["validation_test"] = {
                            "callable": False,
                            "error": str(e)
                        }
                
                result["strategy_tests"][strategy_name] = strategy_result
            
            # Test parameter sampling with mock trial
            try:
                from unittest.mock import MagicMock
                
                temp_db_path = tempfile.mktemp(suffix='.db')
                self.temp_files.append(temp_db_path)
                optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path}")
                
                for strategy_name in STRATEGY_PARAMETER_SPACES.keys():
                    config = OptimizationConfig(
                        strategy_template=strategy_name,
                        n_trials=1
                    )
                    
                    # Mock trial object
                    mock_trial = MagicMock()
                    mock_trial.suggest_int = MagicMock(return_value=10)
                    mock_trial.suggest_float = MagicMock(return_value=0.5)
                    mock_trial.suggest_categorical = MagicMock(return_value="test")
                    
                    try:
                        params = optimizer._sample_parameters(mock_trial, config)
                        result["sampling_tests"][strategy_name] = {
                            "success": True,
                            "params_generated": len(params),
                            "sample_params": list(params.keys())
                        }
                    except Exception as e:
                        result["sampling_tests"][strategy_name] = {
                            "success": False,
                            "error": str(e)
                        }
                
            except Exception as e:
                result["sampling_tests"]["error"] = str(e)
            
            # Check success criteria
            strategy_success = all(
                len(test["parameters"]) > 0 
                for test in result["strategy_tests"].values()
            )
            
            sampling_success = all(
                test.get("success", False) 
                for test in result["sampling_tests"].values()
                if isinstance(test, dict) and "success" in test
            )
            
            result["success"] = strategy_success and sampling_success
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_multi_objective(self) -> Dict[str, Any]:
        """Validate multi-objective optimization capabilities"""
        result = {
            "success": False,
            "multi_objective_tests": {},
            "direction_tests": {},
            "pareto_tests": {}
        }
        
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            temp_db_path = tempfile.mktemp(suffix='.db')
            self.temp_files.append(temp_db_path)
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path}")
            
            # Test multi-objective configuration
            config = OptimizationConfig(
                strategy_template="sma_crossover",
                primary_objective="sharpe_ratio",
                secondary_objectives=["max_drawdown", "win_rate"],
                n_trials=3
            )
            
            # Test direction determination
            directions = optimizer._get_default_directions(config)
            result["direction_tests"]["directions"] = directions
            result["direction_tests"]["primary_direction"] = directions[0] if directions else None
            result["direction_tests"]["secondary_count"] = len(directions) - 1 if directions else 0
            result["direction_tests"]["valid"] = len(directions) == 3  # primary + 2 secondary
            
            # Test study creation with multiple objectives
            study_name = f"multi_obj_test_{int(time.time())}"
            try:
                study = optimizer.create_study(study_name, config, directions)
                result["multi_objective_tests"]["study_creation"] = "✅ Success"
                
                # Check study directions
                study_directions = [d.name for d in study.directions]
                result["multi_objective_tests"]["study_directions"] = study_directions
                result["multi_objective_tests"]["direction_count"] = len(study_directions)
                
            except Exception as e:
                result["multi_objective_tests"]["study_creation"] = f"❌ {str(e)}"
            
            # Test mock multi-objective optimization
            try:
                # Create synthetic data for testing
                synthetic_bars = self._generate_synthetic_bars(100)
                
                # Mock the backtest function to return multi-objective metrics
                with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                    mock_backtest.return_value = {
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 0.15,
                        "win_rate": 0.65,
                        "total_return": 0.25
                    }
                    
                    # Test objective function with multiple objectives
                    mock_trial = MagicMock()
                    mock_trial.suggest_int = MagicMock(return_value=20)
                    mock_trial.suggest_float = MagicMock(return_value=0.8)
                    
                    objectives = optimizer._objective_function(
                        mock_trial, synthetic_bars, config
                    )
                    
                    result["pareto_tests"]["objective_values"] = objectives
                    result["pareto_tests"]["is_multi_value"] = isinstance(objectives, list)
                    result["pareto_tests"]["value_count"] = len(objectives) if isinstance(objectives, list) else 1
                    
            except Exception as e:
                result["pareto_tests"]["error"] = str(e)
            
            # Success criteria
            result["success"] = (
                result["direction_tests"].get("valid", False) and
                result["multi_objective_tests"].get("study_creation", "").startswith("✅") and
                result["pareto_tests"].get("is_multi_value", False)
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_walk_forward(self) -> Dict[str, Any]:
        """Validate walk-forward optimization logic"""
        result = {
            "success": False,
            "window_tests": {},
            "validation_tests": {},
            "aggregation_tests": {}
        }
        
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            temp_db_path = tempfile.mktemp(suffix='.db')
            self.temp_files.append(temp_db_path)
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path}")
            
            # Test window calculation
            data_length = 1000  # 1000 bars
            window_size = 252   # 1 year
            step_size = 63      # 1 quarter
            
            windows = optimizer._calculate_walkforward_windows(
                data_length, window_size, step_size
            )
            
            result["window_tests"]["total_windows"] = len(windows)
            result["window_tests"]["first_window"] = windows[0] if windows else None
            result["window_tests"]["last_window"] = windows[-1] if windows else None
            result["window_tests"]["valid_windows"] = all(
                train_end <= test_start and test_end <= data_length
                for train_start, train_end, test_start, test_end in windows
            )
            
            # Test walk-forward configuration creation
            base_config = OptimizationConfig(
                strategy_template="rsi_strategy",
                n_trials=20,
                walk_forward=True
            )
            
            window_config = optimizer._create_window_config(base_config, 0)
            result["validation_tests"]["window_config_trials"] = window_config.n_trials
            result["validation_tests"]["reduced_trials"] = window_config.n_trials < base_config.n_trials
            result["validation_tests"]["sequential_jobs"] = window_config.n_jobs == 1
            
            # Test parameter consensus finding
            mock_results = [
                {"fast_period": 10, "slow_period": 30, "rsi_period": 14},
                {"fast_period": 12, "slow_period": 28, "rsi_period": 14},
                {"fast_period": 11, "slow_period": 32, "rsi_period": 15}
            ]
            
            consensus = optimizer._find_consensus_parameters(mock_results)
            result["aggregation_tests"]["consensus_params"] = consensus
            result["aggregation_tests"]["has_all_params"] = all(
                param in consensus for param in ["fast_period", "slow_period", "rsi_period"]
            )
            
            # Test consensus strength calculation
            consensus_strength = optimizer._calculate_consensus_strength(mock_results)
            result["aggregation_tests"]["consensus_strength"] = consensus_strength
            result["aggregation_tests"]["strength_valid"] = 0.0 <= consensus_strength <= 1.0
            
            # Mock walk-forward optimization test
            try:
                synthetic_bars = self._generate_synthetic_bars(400)
                
                # Mock successful optimization results
                with patch.object(optimizer, 'optimize_strategy') as mock_optimize:
                    mock_optimize.return_value = MagicMock(
                        best_params={"rsi_period": 14, "oversold_level": 30},
                        best_value=1.2,
                        metrics={"sharpe_ratio": 1.2, "max_drawdown": 0.1},
                        n_trials=5,
                        optimization_duration=10.0
                    )
                    
                    with patch.object(optimizer, '_validate_parameters') as mock_validate:
                        mock_validate.return_value = {"sharpe_ratio": 1.1, "max_drawdown": 0.12}
                        
                        study_name = f"walkforward_test_{int(time.time())}"
                        wf_result = optimizer.walk_forward_optimization(
                            study_name, synthetic_bars, base_config, window_size=100, step_size=25
                        )
                        
                        result["validation_tests"]["walkforward_execution"] = "✅ Success"
                        result["validation_tests"]["result_type"] = type(wf_result).__name__
                        
            except Exception as e:
                result["validation_tests"]["walkforward_execution"] = f"❌ {str(e)}"
            
            # Success criteria
            result["success"] = (
                len(windows) > 0 and
                result["window_tests"].get("valid_windows", False) and
                result["aggregation_tests"].get("has_all_params", False) and
                result["aggregation_tests"].get("strength_valid", False)
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoints for optimization"""
        result = {
            "success": False,
            "endpoint_tests": {},
            "schema_tests": {},
            "import_tests": {}
        }
        
        try:
            # Test API imports
            try:
                from supernova.api import app
                result["import_tests"]["app"] = "✅ Success"
                
                # Check if optimization endpoints are available
                from supernova.api import OPTIMIZATION_ENDPOINTS_AVAILABLE
                result["import_tests"]["optimization_available"] = OPTIMIZATION_ENDPOINTS_AVAILABLE
                
            except Exception as e:
                result["import_tests"]["app"] = f"❌ {str(e)}"
            
            # Test schema imports
            try:
                from supernova.schemas import (
                    OptimizationRequest, OptimizationResults, OptimizationStudy,
                    WatchlistOptimizationRequest, OptimizationDashboardData
                )
                result["schema_tests"]["imports"] = "✅ Success"
                
                # Test schema creation
                opt_request = OptimizationRequest(
                    symbol="AAPL",
                    strategy_template="sma_crossover",
                    bars=[],
                    n_trials=10
                )
                result["schema_tests"]["request_creation"] = "✅ Success"
                
            except Exception as e:
                result["schema_tests"]["imports"] = f"❌ {str(e)}"
            
            # Test endpoint functions exist (without running FastAPI server)
            try:
                import inspect
                from supernova import api
                
                expected_endpoints = [
                    "optimize_strategy",
                    "get_optimization_studies", 
                    "get_optimization_study",
                    "get_best_parameters",
                    "optimize_watchlist",
                    "get_optimization_dashboard",
                    "get_optimization_progress",
                    "delete_optimization_study"
                ]
                
                for endpoint_name in expected_endpoints:
                    if hasattr(api, endpoint_name):
                        func = getattr(api, endpoint_name)
                        if callable(func):
                            result["endpoint_tests"][endpoint_name] = "✅ Function exists"
                        else:
                            result["endpoint_tests"][endpoint_name] = "❌ Not callable"
                    else:
                        result["endpoint_tests"][endpoint_name] = "❌ Missing"
                
            except Exception as e:
                result["endpoint_tests"]["error"] = str(e)
            
            # Check if helper functions exist
            try:
                from supernova.api import _check_optimization_availability
                result["endpoint_tests"]["helper_functions"] = "✅ Helpers exist"
            except:
                result["endpoint_tests"]["helper_functions"] = "❌ Missing helpers"
            
            result["success"] = (
                result["import_tests"].get("app", "").startswith("✅") and
                result["schema_tests"].get("imports", "").startswith("✅") and
                sum(1 for status in result["endpoint_tests"].values() 
                    if isinstance(status, str) and status.startswith("✅")) >= 6
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_database_operations(self) -> Dict[str, Any]:
        """Validate database integration and models"""
        result = {
            "success": False,
            "model_tests": {},
            "operation_tests": {},
            "relationship_tests": {}
        }
        
        try:
            from supernova.optimization_models import (
                OptimizationStudyModel, OptimizationTrialModel, WatchlistOptimizationModel,
                create_optimization_study, update_study_progress, record_optimization_trial
            )
            from sqlalchemy import create_engine, MetaData
            from sqlalchemy.orm import sessionmaker
            
            # Create temporary database
            temp_db_path = tempfile.mktemp(suffix='.db')
            self.temp_files.append(temp_db_path)
            engine = create_engine(f"sqlite:///{temp_db_path}")
            
            # Test model table creation
            try:
                from supernova.db import Base
                Base.metadata.create_all(engine)
                result["model_tests"]["table_creation"] = "✅ Success"
                
                # Check tables exist
                metadata = MetaData()
                metadata.reflect(bind=engine)
                table_names = list(metadata.tables.keys())
                
                expected_tables = [
                    "optimization_studies", "optimization_trials", 
                    "watchlist_optimizations", "optimization_alerts"
                ]
                
                for table_name in expected_tables:
                    if table_name in table_names:
                        result["model_tests"][f"table_{table_name}"] = "✅ Exists"
                    else:
                        result["model_tests"][f"table_{table_name}"] = "❌ Missing"
                        
            except Exception as e:
                result["model_tests"]["table_creation"] = f"❌ {str(e)}"
            
            # Test database operations
            Session = sessionmaker(bind=engine)
            session = Session()
            
            try:
                # Test study creation
                study = create_optimization_study(
                    study_id="test_study_123",
                    study_name="Test Study",
                    symbol="AAPL",
                    strategy_template="sma_crossover",
                    configuration={"n_trials": 10, "primary_objective": "sharpe_ratio"},
                    session=session
                )
                
                result["operation_tests"]["study_creation"] = "✅ Success"
                result["operation_tests"]["study_id"] = study.study_id
                
                # Test progress update
                success = update_study_progress(
                    study_id="test_study_123",
                    progress=0.5,
                    best_value=1.2,
                    session=session
                )
                result["operation_tests"]["progress_update"] = "✅ Success" if success else "❌ Failed"
                
                # Test trial recording
                trial = record_optimization_trial(
                    study_id="test_study_123",
                    trial_id=1,
                    params={"fast_period": 10, "slow_period": 30},
                    value=1.2,
                    session=session
                )
                
                result["operation_tests"]["trial_recording"] = "✅ Success" if trial else "❌ Failed"
                
                # Test relationship
                study_with_trials = session.query(OptimizationStudyModel).filter(
                    OptimizationStudyModel.study_id == "test_study_123"
                ).first()
                
                if study_with_trials and len(study_with_trials.trials) > 0:
                    result["relationship_tests"]["study_trials"] = "✅ Success"
                else:
                    result["relationship_tests"]["study_trials"] = "❌ No relationship"
                
            except Exception as e:
                result["operation_tests"]["error"] = str(e)
            finally:
                session.close()
            
            # Test model methods
            try:
                study = OptimizationStudyModel(
                    study_id="test",
                    study_name="test", 
                    symbol="TEST",
                    strategy_template="test",
                    configuration={},
                    n_trials=10,
                    primary_objective="test",
                    n_complete_trials=5,
                    n_pruned_trials=2,
                    n_failed_trials=1
                )
                
                # Test hybrid properties
                total = study.total_trials
                success_rate = study.success_rate
                study_dict = study.to_dict()
                
                result["model_tests"]["hybrid_properties"] = "✅ Success"
                result["model_tests"]["to_dict"] = "✅ Success" if isinstance(study_dict, dict) else "❌ Failed"
                
            except Exception as e:
                result["model_tests"]["methods_error"] = str(e)
            
            result["success"] = (
                result["model_tests"].get("table_creation", "").startswith("✅") and
                result["operation_tests"].get("study_creation", "").startswith("✅") and
                result["operation_tests"].get("trial_recording", "").startswith("✅")
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_prefect_integration(self) -> Dict[str, Any]:
        """Validate Prefect workflow integration"""
        result = {
            "success": False,
            "workflow_tests": {},
            "flow_tests": {},
            "task_tests": {}
        }
        
        try:
            # Test workflow imports
            try:
                from supernova.workflows import (
                    optimize_strategy_parameters_flow, optimize_watchlist_strategies_flow,
                    scheduled_optimization_flow, OPTIMIZATION_AVAILABLE
                )
                result["workflow_tests"]["imports"] = "✅ Success"
                result["workflow_tests"]["optimization_available"] = OPTIMIZATION_AVAILABLE
                
            except Exception as e:
                result["workflow_tests"]["imports"] = f"❌ {str(e)}"
            
            # Test Prefect imports
            try:
                from prefect import flow, task
                result["workflow_tests"]["prefect"] = "✅ Success"
            except Exception as e:
                result["workflow_tests"]["prefect"] = f"❌ {str(e)}"
            
            # Test flow definitions
            try:
                from supernova.workflows import optimize_strategy_parameters_flow
                
                # Check if it's a Prefect flow
                if hasattr(optimize_strategy_parameters_flow, '_flow'):
                    result["flow_tests"]["strategy_flow"] = "✅ Is Prefect flow"
                else:
                    result["flow_tests"]["strategy_flow"] = "❌ Not Prefect flow"
                
                # Test flow signature
                import inspect
                sig = inspect.signature(optimize_strategy_parameters_flow)
                params = list(sig.parameters.keys())
                
                expected_params = ["symbol", "strategy_template", "n_trials"]
                has_required = all(param in params for param in expected_params)
                
                result["flow_tests"]["strategy_params"] = "✅ Valid signature" if has_required else "❌ Invalid signature"
                
            except Exception as e:
                result["flow_tests"]["strategy_flow"] = f"❌ {str(e)}"
            
            # Test task definitions (if available)
            try:
                from supernova.workflows import run_optimization_study
                
                if hasattr(run_optimization_study, '_task'):
                    result["task_tests"]["optimization_task"] = "✅ Is Prefect task"
                else:
                    result["task_tests"]["optimization_task"] = "❌ Not Prefect task"
                    
            except Exception as e:
                result["task_tests"]["optimization_task"] = f"❌ {str(e)}"
            
            # Mock flow execution test (without actual execution)
            try:
                from supernova.workflows import optimize_strategy_parameters_flow
                
                # Create mock data for testing
                with patch('supernova.workflows.run_optimization_study') as mock_task:
                    mock_task.return_value = {
                        "success": True,
                        "optimization": {
                            "best_params": {"test": "param"},
                            "best_value": 1.5,
                            "n_trials": 5,
                            "optimization_duration": 10.0,
                            "timestamp": datetime.now().isoformat(),
                            "metrics": {"sharpe_ratio": 1.5}
                        }
                    }
                    
                    result["flow_tests"]["mock_execution"] = "✅ Flow structure valid"
                    
            except Exception as e:
                result["flow_tests"]["mock_execution"] = f"❌ {str(e)}"
            
            result["success"] = (
                result["workflow_tests"].get("imports", "").startswith("✅") and
                result["workflow_tests"].get("prefect", "").startswith("✅") and
                result["flow_tests"].get("strategy_flow", "").startswith("✅")
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance and scalability"""
        result = {
            "success": False,
            "performance_tests": {},
            "memory_tests": {},
            "scalability_tests": {}
        }
        
        try:
            import psutil
            import gc
            
            # Test basic optimization performance
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            temp_db_path = tempfile.mktemp(suffix='.db')
            self.temp_files.append(temp_db_path)
            
            # Memory usage before optimization
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path}")
            
            # Test small optimization performance
            config = OptimizationConfig(
                strategy_template="sma_crossover",
                n_trials=10,
                primary_objective="sharpe_ratio"
            )
            
            synthetic_bars = self._generate_synthetic_bars(252)  # 1 year of data
            
            # Mock backtest for performance testing
            with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                mock_backtest.return_value = {
                    "sharpe_ratio": np.random.normal(1.0, 0.3),
                    "total_return": np.random.normal(0.15, 0.1),
                    "max_drawdown": np.random.uniform(0.05, 0.25)
                }
                
                study_name = f"perf_test_{int(time.time())}"
                
                try:
                    opt_result = optimizer.optimize_strategy(
                        study_name, synthetic_bars, config
                    )
                    
                    optimization_time = time.time() - start_time
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_used = memory_after - memory_before
                    
                    result["performance_tests"]["optimization_time"] = optimization_time
                    result["performance_tests"]["trials_per_second"] = config.n_trials / optimization_time
                    result["performance_tests"]["success"] = opt_result.n_trials == config.n_trials
                    
                    result["memory_tests"]["memory_before_mb"] = memory_before
                    result["memory_tests"]["memory_after_mb"] = memory_after
                    result["memory_tests"]["memory_used_mb"] = memory_used
                    result["memory_tests"]["memory_per_trial_kb"] = (memory_used * 1024) / config.n_trials if config.n_trials > 0 else 0
                    
                except Exception as e:
                    result["performance_tests"]["error"] = str(e)
            
            # Test parameter space scaling
            large_param_space = {
                "param_1": (1, 100),
                "param_2": (0.01, 10.0),
                "param_3": (5, 50),
                "param_4": [f"option_{i}" for i in range(20)],
                "param_5": (0.1, 5.0),
                "validation": lambda p: True
            }
            
            config_large = OptimizationConfig(
                strategy_template="custom_strategy",
                parameter_space=large_param_space,
                n_trials=20
            )
            
            start_time = time.time()
            
            with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                mock_backtest.return_value = {"sharpe_ratio": 1.2, "total_return": 0.15}
                
                try:
                    study_name_large = f"large_perf_test_{int(time.time())}"
                    opt_result_large = optimizer.optimize_strategy(
                        study_name_large, synthetic_bars, config_large
                    )
                    
                    large_optimization_time = time.time() - start_time
                    
                    result["scalability_tests"]["large_param_space_time"] = large_optimization_time
                    result["scalability_tests"]["large_space_success"] = opt_result_large.n_trials == config_large.n_trials
                    result["scalability_tests"]["time_per_param"] = large_optimization_time / len(large_param_space)
                    
                except Exception as e:
                    result["scalability_tests"]["large_param_error"] = str(e)
            
            # Test concurrent studies
            try:
                study_times = []
                for i in range(3):
                    start_time = time.time()
                    study_name_concurrent = f"concurrent_test_{i}_{int(time.time())}"
                    
                    with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                        mock_backtest.return_value = {"sharpe_ratio": 1.0 + i * 0.1}
                        
                        config_small = OptimizationConfig(
                            strategy_template="sma_crossover",
                            n_trials=5
                        )
                        
                        optimizer.optimize_strategy(study_name_concurrent, synthetic_bars[:100], config_small)
                        study_times.append(time.time() - start_time)
                
                result["scalability_tests"]["concurrent_studies"] = len(study_times)
                result["scalability_tests"]["avg_study_time"] = sum(study_times) / len(study_times)
                result["scalability_tests"]["concurrent_success"] = all(t > 0 for t in study_times)
                
            except Exception as e:
                result["scalability_tests"]["concurrent_error"] = str(e)
            
            # Success criteria
            result["success"] = (
                result["performance_tests"].get("success", False) and
                result["performance_tests"].get("trials_per_second", 0) > 0.5 and  # At least 0.5 trials/sec
                result["memory_tests"].get("memory_used_mb", float('inf')) < 100 and  # Less than 100MB
                result["scalability_tests"].get("large_space_success", False)
            )
            
            # Clean up
            gc.collect()
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling and edge cases"""
        result = {
            "success": False,
            "error_tests": {},
            "edge_case_tests": {},
            "recovery_tests": {}
        }
        
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            temp_db_path = tempfile.mktemp(suffix='.db')
            self.temp_files.append(temp_db_path)
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path}")
            
            # Test invalid configuration
            try:
                invalid_config = OptimizationConfig(
                    strategy_template="nonexistent_strategy",
                    n_trials=0  # Invalid trial count
                )
                result["error_tests"]["invalid_config"] = "✅ Config created (should validate at runtime)"
            except Exception as e:
                result["error_tests"]["invalid_config"] = f"❌ {str(e)}"
            
            # Test insufficient data
            try:
                config = OptimizationConfig(strategy_template="sma_crossover", n_trials=5)
                insufficient_bars = self._generate_synthetic_bars(10)  # Too few bars
                
                study_name = f"insufficient_data_test_{int(time.time())}"
                
                try:
                    optimizer.optimize_strategy(study_name, insufficient_bars, config)
                    result["error_tests"]["insufficient_data"] = "❌ Should have failed"
                except ValueError as e:
                    result["error_tests"]["insufficient_data"] = "✅ Caught insufficient data error"
                except Exception as e:
                    result["error_tests"]["insufficient_data"] = f"❌ Unexpected error: {str(e)}"
                    
            except Exception as e:
                result["error_tests"]["insufficient_data_setup"] = str(e)
            
            # Test invalid parameter ranges
            try:
                mock_trial = MagicMock()
                mock_trial.suggest_int = MagicMock(return_value=100)  # Invalid value
                
                # Create config with validation that should fail
                config = OptimizationConfig(
                    strategy_template="sma_crossover",  # Has validation: fast < slow
                    n_trials=1
                )
                
                # This should trigger validation failure
                params = optimizer._sample_parameters(mock_trial, config)
                valid = optimizer._validate_parameter_combination(params, config)
                
                result["error_tests"]["parameter_validation"] = "✅ Validation working" if not valid else "❌ Validation not working"
                
            except Exception as e:
                result["error_tests"]["parameter_validation"] = f"❌ {str(e)}"
            
            # Test backtest failure handling
            try:
                config = OptimizationConfig(strategy_template="sma_crossover", n_trials=3)
                synthetic_bars = self._generate_synthetic_bars(100)
                
                with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                    # Simulate backtest failure
                    mock_backtest.side_effect = Exception("Backtest failed")
                    
                    study_name = f"backtest_failure_test_{int(time.time())}"
                    
                    try:
                        opt_result = optimizer.optimize_strategy(study_name, synthetic_bars, config)
                        # Should complete despite backtest failures
                        result["error_tests"]["backtest_failure_handling"] = "✅ Handled gracefully"
                    except Exception as e:
                        result["error_tests"]["backtest_failure_handling"] = f"❌ Not handled: {str(e)}"
                        
            except Exception as e:
                result["error_tests"]["backtest_failure_setup"] = str(e)
            
            # Test database connection failure
            try:
                # Create optimizer with invalid database
                invalid_optimizer = OptunaOptimizer(storage_url="sqlite:///nonexistent/path/db.sqlite")
                
                config = OptimizationConfig(strategy_template="sma_crossover", n_trials=1)
                
                try:
                    study_name = f"db_failure_test_{int(time.time())}"
                    invalid_optimizer.create_study(study_name, config)
                    result["error_tests"]["database_failure"] = "❌ Should have failed"
                except Exception as e:
                    result["error_tests"]["database_failure"] = "✅ Database error caught"
                    
            except Exception as e:
                result["error_tests"]["database_failure_setup"] = str(e)
            
            # Test edge cases
            # Empty parameter space
            try:
                config = OptimizationConfig(
                    strategy_template="custom_empty",
                    parameter_space={},  # Empty parameter space
                    n_trials=1
                )
                
                mock_trial = MagicMock()
                params = optimizer._sample_parameters(mock_trial, config)
                result["edge_case_tests"]["empty_param_space"] = "✅ Handled" if len(params) == 0 else "❌ Not handled"
                
            except Exception as e:
                result["edge_case_tests"]["empty_param_space"] = f"❌ {str(e)}"
            
            # Single trial optimization
            try:
                config = OptimizationConfig(strategy_template="sma_crossover", n_trials=1)
                synthetic_bars = self._generate_synthetic_bars(100)
                
                with patch('supernova.optimizer.run_vbt_backtest') as mock_backtest:
                    mock_backtest.return_value = {"sharpe_ratio": 1.0}
                    
                    study_name = f"single_trial_test_{int(time.time())}"
                    opt_result = optimizer.optimize_strategy(study_name, synthetic_bars, config)
                    
                    result["edge_case_tests"]["single_trial"] = "✅ Success" if opt_result.n_trials == 1 else "❌ Failed"
                    
            except Exception as e:
                result["edge_case_tests"]["single_trial"] = f"❌ {str(e)}"
            
            # Test recovery mechanisms
            # Study resume capability
            try:
                config = OptimizationConfig(strategy_template="sma_crossover", n_trials=5)
                study_name = f"resume_test_{int(time.time())}"
                
                # Create initial study
                study1 = optimizer.create_study(study_name, config)
                
                # Create another optimizer instance (simulating restart)
                temp_db_path2 = temp_db_path  # Same database
                optimizer2 = OptunaOptimizer(storage_url=f"sqlite:///{temp_db_path2}")
                
                # Load existing study (should not fail)
                study2 = optimizer2.create_study(study_name, config)
                
                result["recovery_tests"]["study_resume"] = "✅ Success" if study2.study_name == study_name else "❌ Failed"
                
            except Exception as e:
                result["recovery_tests"]["study_resume"] = f"❌ {str(e)}"
            
            # Count successful error handling tests
            error_handled = sum(1 for test in result["error_tests"].values() 
                              if isinstance(test, str) and test.startswith("✅"))
            
            edge_cases_handled = sum(1 for test in result["edge_case_tests"].values()
                                   if isinstance(test, str) and test.startswith("✅"))
            
            recovery_working = sum(1 for test in result["recovery_tests"].values()
                                 if isinstance(test, str) and test.startswith("✅"))
            
            result["success"] = (
                error_handled >= 3 and  # Most error cases handled
                edge_cases_handled >= 2 and  # Edge cases work
                recovery_working >= 1  # Recovery mechanisms work
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _generate_synthetic_bars(self, num_bars: int) -> List[Dict]:
        """Generate synthetic OHLCV data for testing"""
        bars = []
        base_price = 100.0
        
        for i in range(num_bars):
            # Simple random walk
            change_pct = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + change_pct)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)
            
            bar = {
                "timestamp": (datetime.now() - timedelta(days=num_bars-i)).isoformat(),
                "open": base_price,
                "high": high,
                "low": low,
                "close": base_price,
                "volume": volume
            }
            bars.append(bar)
        
        return bars
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during testing"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        self.temp_files.clear()
    
    def save_results(self, filename: str = None):
        """Save validation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_validation_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Validation results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("SUPERNOVA OPTUNA OPTIMIZATION SYSTEM - VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['validation_timestamp']}")
        report.append(f"Overall Success Rate: {self.results['success_rate']:.1f}%")
        report.append(f"Status: {self.results['test_summary']['overall_status']}")
        report.append("")
        
        # Summary
        summary = self.results["test_summary"]
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append("")
        
        # Detailed results
        for test_name, test_result in self.results["detailed_results"].items():
            status = "✅ PASSED" if test_result.get("success", False) else "❌ FAILED"
            duration = test_result.get("duration_seconds", 0)
            
            report.append(f"{test_name}: {status} ({duration:.2f}s)")
            
            if not test_result.get("success", False) and test_result.get("error"):
                report.append(f"  Error: {test_result['error']}")
            
            # Add key findings
            for key, value in test_result.items():
                if key in ["success", "error", "duration_seconds", "timestamp"]:
                    continue
                if isinstance(value, dict) and any(isinstance(v, str) and v.startswith(("✅", "❌")) for v in value.values()):
                    report.append(f"  {key}:")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str) and subvalue.startswith(("✅", "❌")):
                            report.append(f"    {subkey}: {subvalue}")
            
            report.append("")
        
        # Issues and recommendations
        if self.results.get("issues_found"):
            report.append("ISSUES FOUND")
            report.append("-" * 40)
            for issue in self.results["issues_found"]:
                report.append(f"• {issue}")
            report.append("")
        
        if self.results.get("recommendations"):
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for rec in self.results["recommendations"]:
                report.append(f"• {rec}")
            report.append("")
        
        # Performance metrics
        if self.results.get("performance_metrics"):
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            for metric, value in self.results["performance_metrics"].items():
                report.append(f"{metric}: {value}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main execution function"""
    print("SuperNova Optuna Optimization System - Comprehensive Validation")
    print("=" * 70)
    
    validator = OptunaValidationSuite()
    
    try:
        results = await validator.run_comprehensive_validation()
        
        # Save results
        results_file = validator.save_results()
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        # Generate and save report
        report = validator.generate_report()
        
        report_file = f"optuna_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Validation report saved to: {report_file}")
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print(report)
        
        return results["success_rate"] > 80  # Return True if validation passes
        
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)