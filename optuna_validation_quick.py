#!/usr/bin/env python3
"""
SuperNova Optuna Hyperparameter Optimization - Quick Validation Script

A streamlined validation script that tests core functionality without 
requiring heavy dependencies like pandas, numpy, etc.
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class QuickOptunaValidator:
    """Lightweight validation for Optuna optimization system"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "issues": [],
            "success": False
        }
        
    async def run_validation(self):
        """Run quick validation tests"""
        logger.info("Starting Quick Optuna Validation")
        logger.info("=" * 50)
        
        tests = [
            ("Code Structure", self._test_code_structure),
            ("Dependencies", self._test_dependencies),
            ("Optimizer Import", self._test_optimizer_import),
            ("Models Import", self._test_models_import),
            ("API Import", self._test_api_import),
            ("Workflows Import", self._test_workflows_import),
            ("Parameter Spaces", self._test_parameter_spaces),
            ("Basic Functionality", self._test_basic_functionality),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                result = await test_func()
                self.results["tests"][test_name] = result
                
                if result["passed"]:
                    logger.info(f"✅ {test_name} - PASSED")
                    self.results["summary"]["passed"] += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED: {result.get('error', 'Unknown error')}")
                    self.results["summary"]["failed"] += 1
                    self.results["issues"].append(f"{test_name}: {result.get('error', 'Failed')}")
                
            except Exception as e:
                logger.error(f"💥 {test_name} - EXCEPTION: {e}")
                self.results["tests"][test_name] = {
                    "passed": False,
                    "error": str(e),
                    "exception": True
                }
                self.results["summary"]["failed"] += 1
                self.results["issues"].append(f"{test_name}: Exception - {e}")
        
        self.results["summary"]["total"] = len(tests)
        self.results["success"] = self.results["summary"]["failed"] == 0
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Passed: {self.results['summary']['passed']}")
        logger.info(f"Failed: {self.results['summary']['failed']}")
        logger.info(f"Total: {self.results['summary']['total']}")
        logger.info(f"Success: {self.results['success']}")
        
        if self.results["issues"]:
            logger.info("\nISSUES FOUND:")
            for issue in self.results["issues"]:
                logger.info(f"• {issue}")
        
        return self.results
    
    async def _test_code_structure(self):
        """Test if required files exist"""
        required_files = [
            "supernova/optimizer.py",
            "supernova/optimization_models.py", 
            "supernova/api.py",
            "supernova/workflows.py",
            "supernova/schemas.py",
            "requirements.txt"
        ]
        
        missing = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing.append(file_path)
        
        if missing:
            return {
                "passed": False,
                "error": f"Missing files: {', '.join(missing)}",
                "missing_files": missing
            }
        
        return {
            "passed": True,
            "message": "All required files found",
            "files_checked": required_files
        }
    
    async def _test_dependencies(self):
        """Test if core dependencies are available"""
        dependencies = [
            "optuna",
            "sqlalchemy", 
            "fastapi",
            "pydantic",
            "prefect"
        ]
        
        missing = []
        available = []
        
        for dep in dependencies:
            try:
                __import__(dep)
                available.append(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            return {
                "passed": False,
                "error": f"Missing dependencies: {', '.join(missing)}",
                "missing": missing,
                "available": available
            }
        
        return {
            "passed": True,
            "message": "All core dependencies available",
            "dependencies": available
        }
    
    async def _test_optimizer_import(self):
        """Test optimizer module import"""
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig, OPTUNA_AVAILABLE
            
            if not OPTUNA_AVAILABLE:
                return {
                    "passed": False,
                    "error": "OPTUNA_AVAILABLE is False - optuna not properly installed"
                }
            
            # Test class instantiation
            temp_db = tempfile.mktemp(suffix='.db')
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db}")
            
            # Test config creation
            config = OptimizationConfig(
                strategy_template="sma_crossover",
                n_trials=5
            )
            
            # Cleanup
            if os.path.exists(temp_db):
                os.remove(temp_db)
            
            return {
                "passed": True,
                "message": "Optimizer classes imported and instantiated successfully",
                "classes": ["OptunaOptimizer", "OptimizationConfig"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to import optimizer: {str(e)}"
            }
    
    async def _test_models_import(self):
        """Test optimization models import"""
        try:
            from supernova.optimization_models import (
                OptimizationStudyModel, 
                OptimizationTrialModel,
                WatchlistOptimizationModel,
                create_optimization_study
            )
            
            return {
                "passed": True,
                "message": "Optimization models imported successfully",
                "models": [
                    "OptimizationStudyModel", 
                    "OptimizationTrialModel", 
                    "WatchlistOptimizationModel"
                ]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to import models: {str(e)}"
            }
    
    async def _test_api_import(self):
        """Test API imports"""
        try:
            from supernova.api import app, OPTIMIZATION_ENDPOINTS_AVAILABLE
            
            if not OPTIMIZATION_ENDPOINTS_AVAILABLE:
                return {
                    "passed": False,
                    "error": "OPTIMIZATION_ENDPOINTS_AVAILABLE is False"
                }
            
            # Check some key endpoint functions exist
            from supernova import api
            endpoints = [
                "optimize_strategy",
                "get_optimization_studies", 
                "get_optimization_dashboard"
            ]
            
            missing_endpoints = []
            for endpoint in endpoints:
                if not hasattr(api, endpoint):
                    missing_endpoints.append(endpoint)
            
            if missing_endpoints:
                return {
                    "passed": False,
                    "error": f"Missing API endpoints: {', '.join(missing_endpoints)}"
                }
            
            return {
                "passed": True,
                "message": "API imports successful",
                "endpoints_found": endpoints
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to import API: {str(e)}"
            }
    
    async def _test_workflows_import(self):
        """Test workflow imports"""
        try:
            from supernova.workflows import (
                optimize_strategy_parameters_flow,
                optimize_watchlist_strategies_flow,
                OPTIMIZATION_AVAILABLE
            )
            
            if not OPTIMIZATION_AVAILABLE:
                return {
                    "passed": False,
                    "error": "OPTIMIZATION_AVAILABLE is False in workflows"
                }
            
            return {
                "passed": True,
                "message": "Workflow imports successful",
                "flows": [
                    "optimize_strategy_parameters_flow",
                    "optimize_watchlist_strategies_flow"
                ]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to import workflows: {str(e)}"
            }
    
    async def _test_parameter_spaces(self):
        """Test strategy parameter spaces"""
        try:
            from supernova.optimizer import STRATEGY_PARAMETER_SPACES
            
            expected_strategies = [
                "sma_crossover",
                "rsi_strategy", 
                "macd_strategy",
                "bb_strategy",
                "sentiment_strategy"
            ]
            
            missing_strategies = []
            for strategy in expected_strategies:
                if strategy not in STRATEGY_PARAMETER_SPACES:
                    missing_strategies.append(strategy)
            
            if missing_strategies:
                return {
                    "passed": False,
                    "error": f"Missing strategy parameter spaces: {', '.join(missing_strategies)}"
                }
            
            # Test parameter space structure
            for strategy, params in STRATEGY_PARAMETER_SPACES.items():
                if not isinstance(params, dict):
                    return {
                        "passed": False,
                        "error": f"Invalid parameter space for {strategy}: not a dict"
                    }
                
                if len(params) == 0 or (len(params) == 1 and "validation" in params):
                    return {
                        "passed": False,
                        "error": f"Empty parameter space for {strategy}"
                    }
            
            return {
                "passed": True,
                "message": "All strategy parameter spaces found and valid",
                "strategies": list(STRATEGY_PARAMETER_SPACES.keys()),
                "total_strategies": len(STRATEGY_PARAMETER_SPACES)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to test parameter spaces: {str(e)}"
            }
    
    async def _test_basic_functionality(self):
        """Test basic optimizer functionality"""
        try:
            from supernova.optimizer import OptunaOptimizer, OptimizationConfig
            
            # Create temporary database
            temp_db = tempfile.mktemp(suffix='.db')
            optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db}")
            
            # Test study creation
            config = OptimizationConfig(
                strategy_template="sma_crossover",
                n_trials=3,
                primary_objective="sharpe_ratio"
            )
            
            study_name = f"test_study_{int(time.time())}"
            study = optimizer.create_study(study_name, config)
            
            if not study or study.study_name != study_name:
                return {
                    "passed": False,
                    "error": "Study creation failed or returned invalid study"
                }
            
            # Test parameter sampling (mock)
            from unittest.mock import MagicMock
            mock_trial = MagicMock()
            mock_trial.suggest_int = MagicMock(return_value=10)
            mock_trial.suggest_float = MagicMock(return_value=0.5)
            
            params = optimizer._sample_parameters(mock_trial, config)
            
            if not isinstance(params, dict) or len(params) == 0:
                return {
                    "passed": False,
                    "error": "Parameter sampling failed or returned empty params"
                }
            
            # Cleanup
            if os.path.exists(temp_db):
                os.remove(temp_db)
            
            return {
                "passed": True,
                "message": "Basic functionality working",
                "study_created": study_name,
                "params_sampled": list(params.keys())
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Basic functionality test failed: {str(e)}"
            }
    
    def save_results(self, filename: str = None):
        """Save validation results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_quick_validation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

async def main():
    """Main function"""
    logger.info("SuperNova Optuna Optimization - Quick Validation")
    
    validator = QuickOptunaValidator()
    results = await validator.run_validation()
    
    # Save results
    results_file = validator.save_results()
    
    # Return success status
    return results["success"]

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)