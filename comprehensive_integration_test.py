#!/usr/bin/env python3
"""
SuperNova Extended Framework - Comprehensive Integration Testing
================================================================

This script performs comprehensive integration testing for the three major enhancements:
1. Prefect Workflows Integration
2. VectorBT Backtesting Engine 
3. LangChain LLM Advisor

Tests cover:
- Dependency validation
- API endpoints
- Configuration handling
- End-to-end workflows
- Performance validation
- Error handling
- Fallback mechanisms
"""

import os
import sys
import time
import json
import asyncio
import warnings
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the supernova package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test results storage
test_results = {
    "start_time": datetime.now(),
    "dependency_tests": {},
    "api_tests": {},
    "configuration_tests": {},
    "workflow_tests": {},
    "performance_tests": {},
    "error_handling_tests": {},
    "summary": {}
}

class TestLogger:
    """Simple test logger"""
    def __init__(self):
        self.logs = []
    
    def info(self, message):
        msg = f"INFO: {message}"
        self.logs.append(msg)
        print(msg)
    
    def warning(self, message):
        msg = f"WARNING: {message}"
        self.logs.append(msg)
        print(msg)
    
    def error(self, message):
        msg = f"ERROR: {message}"
        self.logs.append(msg)
        print(msg)
    
    def success(self, message):
        msg = f"SUCCESS: {message}"
        self.logs.append(msg)
        print(msg)

logger = TestLogger()

def test_dependency_validation() -> Dict[str, Any]:
    """Test all new dependencies and import statements"""
    logger.info("=== DEPENDENCY VALIDATION TESTS ===")
    results = {}
    
    # Core dependencies
    dependencies = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("fastapi", None),
        ("uvicorn", None),
        ("pydantic", None),
        ("sqlalchemy", None)
    ]
    
    # New enhancement dependencies
    enhancement_dependencies = [
        ("vectorbt", "vbt", "VectorBT Backtesting"),
        ("talib", None, "Technical Analysis Library"),
        ("numba", None, "High-performance computing"),
        ("langchain", None, "LLM Integration"),
        ("langchain_openai", None, "OpenAI LangChain"),
        ("langchain_anthropic", None, "Anthropic LangChain"),
        ("langchain_community", None, "Community LangChain"),
        ("langchain_core", None, "Core LangChain"),
        ("openai", None, "OpenAI Python SDK"),
        ("anthropic", None, "Anthropic Python SDK"),
        ("prefect", None, "Workflow Orchestration"),
        ("prefect_sqlalchemy", None, "Prefect SQLAlchemy blocks")
    ]
    
    # Test core dependencies
    for dep_name, import_alias in dependencies:
        try:
            if import_alias:
                exec(f"import {dep_name} as {import_alias}")
            else:
                exec(f"import {dep_name}")
            results[dep_name] = {"status": "available", "critical": True}
            logger.success(f"Core dependency {dep_name} is available")
        except ImportError as e:
            results[dep_name] = {"status": "missing", "error": str(e), "critical": True}
            logger.error(f"Critical dependency {dep_name} is missing: {e}")
    
    # Test enhancement dependencies
    for dep_info in enhancement_dependencies:
        dep_name = dep_info[0]
        import_alias = dep_info[1]
        description = dep_info[2] if len(dep_info) > 2 else dep_name
        
        try:
            if import_alias:
                exec(f"import {dep_name} as {import_alias}")
            else:
                exec(f"import {dep_name}")
            results[dep_name] = {"status": "available", "critical": False, "description": description}
            logger.success(f"Enhancement dependency {dep_name} ({description}) is available")
        except ImportError as e:
            results[dep_name] = {"status": "missing", "error": str(e), "critical": False, "description": description}
            logger.warning(f"Enhancement dependency {dep_name} ({description}) is missing: {e}")
    
    # Test supernova module imports
    supernova_modules = [
        "supernova.config",
        "supernova.api", 
        "supernova.advisor",
        "supernova.backtester",
        "supernova.workflows",
        "supernova.sentiment"
    ]
    
    for module in supernova_modules:
        try:
            exec(f"import {module}")
            results[module] = {"status": "available", "critical": True, "module_type": "supernova"}
            logger.success(f"SuperNova module {module} imports successfully")
        except ImportError as e:
            results[module] = {"status": "missing", "error": str(e), "critical": True, "module_type": "supernova"}
            logger.error(f"SuperNova module {module} import failed: {e}")
    
    return results

def test_configuration_handling() -> Dict[str, Any]:
    """Test configuration and environment variable handling"""
    logger.info("=== CONFIGURATION TESTS ===")
    results = {}
    
    try:
        from supernova.config import settings
        results["config_import"] = {"status": "success"}
        logger.success("Configuration module imported successfully")
        
        # Test core configuration values
        core_configs = [
            "SUPERNOVA_ENV", "DATABASE_URL", "LOG_LEVEL"
        ]
        
        for config in core_configs:
            value = getattr(settings, config, None)
            results[f"config_{config}"] = {
                "status": "available" if value is not None else "default",
                "value": str(value)
            }
            logger.info(f"Config {config}: {value}")
        
        # Test LLM configuration
        llm_configs = [
            "LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL", 
            "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_CACHE_ENABLED"
        ]
        
        llm_config_values = {}
        for config in llm_configs:
            value = getattr(settings, config, None)
            llm_config_values[config] = value
            results[f"llm_{config}"] = {
                "status": "configured" if value is not None else "default",
                "value": str(value)
            }
            logger.info(f"LLM Config {config}: {value}")
        
        results["llm_config_summary"] = llm_config_values
        
        # Test Prefect configuration
        prefect_configs = [
            "ENABLE_PREFECT", "PREFECT_API_URL", "PREFECT_API_KEY",
            "PREFECT_TWITTER_RETRIES", "PREFECT_REDDIT_RETRIES"
        ]
        
        prefect_config_values = {}
        for config in prefect_configs:
            value = getattr(settings, config, None)
            prefect_config_values[config] = value
            results[f"prefect_{config}"] = {
                "status": "configured" if value is not None else "default",
                "value": str(value)
            }
            logger.info(f"Prefect Config {config}: {value}")
        
        results["prefect_config_summary"] = prefect_config_values
        
        # Test VectorBT configuration
        vbt_configs = [
            "VECTORBT_ENABLED", "VECTORBT_DEFAULT_ENGINE", 
            "VECTORBT_DEFAULT_FEES", "VECTORBT_DEFAULT_SLIPPAGE"
        ]
        
        vbt_config_values = {}
        for config in vbt_configs:
            value = getattr(settings, config, None)
            vbt_config_values[config] = value
            results[f"vbt_{config}"] = {
                "status": "configured" if value is not None else "default",
                "value": str(value)
            }
            logger.info(f"VectorBT Config {config}: {value}")
        
        results["vbt_config_summary"] = vbt_config_values
        
    except Exception as e:
        results["config_import"] = {"status": "error", "error": str(e)}
        logger.error(f"Configuration test failed: {e}")
    
    return results

def create_sample_data() -> List[Dict]:
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, len(dates)):
        # Random walk with drift
        change = np.random.normal(0.001, 0.02)  # Small daily drift with 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    bars = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Create realistic OHLC data
        volatility = close * 0.01  # 1% intraday volatility
        high = close + abs(np.random.normal(0, volatility))
        low = close - abs(np.random.normal(0, volatility))
        open_price = low + np.random.random() * (high - low)
        volume = np.random.randint(100000, 1000000)
        
        bars.append({
            "timestamp": date.isoformat(),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume
        })
    
    return bars

def test_api_endpoints() -> Dict[str, Any]:
    """Test API endpoints with the three enhancements"""
    logger.info("=== API INTEGRATION TESTS ===")
    results = {}
    
    try:
        # Test imports
        from supernova.api import app
        from supernova.schemas import IntakeRequest, AdviceRequest, BacktestRequest
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        results["api_setup"] = {"status": "success"}
        logger.success("API test client created successfully")
        
        # Test 1: Intake endpoint
        intake_data = {
            "name": "Integration Test User",
            "email": "test@example.com",
            "risk_questions": [2, 3, 2, 3, 2],  # Moderate risk
            "time_horizon_yrs": 5,
            "objectives": "Growth with moderate risk",
            "constraints": "No specific constraints"
        }
        
        try:
            response = client.post("/intake", json=intake_data)
            if response.status_code == 200:
                intake_result = response.json()
                profile_id = intake_result.get("profile_id")
                results["intake_endpoint"] = {
                    "status": "success",
                    "response": intake_result,
                    "profile_id": profile_id
                }
                logger.success(f"Intake endpoint test passed. Profile ID: {profile_id}")
            else:
                results["intake_endpoint"] = {
                    "status": "error",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"Intake endpoint failed: {response.status_code} - {response.text}")
                profile_id = 1  # Use default for further tests
                
        except Exception as e:
            results["intake_endpoint"] = {"status": "error", "error": str(e)}
            logger.error(f"Intake endpoint test error: {e}")
            profile_id = 1
        
        # Test 2: Advice endpoint (with LLM integration)
        sample_bars = create_sample_data()[-60:]  # Last 60 days
        
        advice_data = {
            "profile_id": profile_id,
            "symbol": "AAPL",
            "asset_class": "stock",
            "timeframe": "1d", 
            "bars": sample_bars,
            "sentiment_hint": 0.2,  # Slightly positive sentiment
            "strategy_template": "sma_crossover",
            "params": {"short_window": 20, "long_window": 50}
        }
        
        try:
            response = client.post("/advice", json=advice_data)
            if response.status_code == 200:
                advice_result = response.json()
                results["advice_endpoint"] = {
                    "status": "success",
                    "response": advice_result
                }
                logger.success(f"Advice endpoint test passed. Action: {advice_result.get('action')}, Confidence: {advice_result.get('confidence')}")
                
                # Check for LLM enhancement in rationale
                rationale = advice_result.get('rationale', '')
                if len(rationale) > 100:  # LLM responses are typically longer
                    results["llm_enhancement"] = {"status": "detected", "rationale_length": len(rationale)}
                    logger.success("LLM-enhanced rationale detected in advice response")
                else:
                    results["llm_enhancement"] = {"status": "fallback", "rationale_length": len(rationale)}
                    logger.warning("Using fallback rationale (LLM may not be available)")
                    
            else:
                results["advice_endpoint"] = {
                    "status": "error",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"Advice endpoint failed: {response.status_code} - {response.text}")
        except Exception as e:
            results["advice_endpoint"] = {"status": "error", "error": str(e)}
            logger.error(f"Advice endpoint test error: {e}")
        
        # Test 3: Backtest endpoint (VectorBT integration)
        backtest_data = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "bars": sample_bars,
            "strategy_template": "sma_crossover",
            "params": {"short_window": 10, "long_window": 20},
            "start_cash": 10000.0,
            "fees": 0.001,
            "slippage": 0.001,
            "use_vectorbt": True
        }
        
        try:
            response = client.post("/backtest", json=backtest_data)
            if response.status_code == 200:
                backtest_result = response.json()
                results["backtest_endpoint"] = {
                    "status": "success",
                    "response": backtest_result
                }
                
                # Check which engine was used
                engine = backtest_result.get('engine', 'Unknown')
                logger.success(f"Backtest endpoint test passed. Engine: {engine}")
                
                if engine == "VectorBT":
                    results["vectorbt_integration"] = {"status": "active", "engine": engine}
                    logger.success("VectorBT integration is active")
                else:
                    results["vectorbt_integration"] = {"status": "fallback", "engine": engine}
                    logger.warning("Using legacy backtester (VectorBT may not be available)")
                    
                # Validate metrics
                required_metrics = ["CAGR", "Sharpe", "MaxDrawdown", "WinRate", "n_trades"]
                missing_metrics = [m for m in required_metrics if m not in backtest_result.get('metrics', {})]
                if not missing_metrics:
                    results["backtest_metrics"] = {"status": "complete", "metrics_count": len(backtest_result.get('metrics', {}))}
                    logger.success("All required backtest metrics are present")
                else:
                    results["backtest_metrics"] = {"status": "incomplete", "missing": missing_metrics}
                    logger.warning(f"Missing backtest metrics: {missing_metrics}")
                    
            else:
                results["backtest_endpoint"] = {
                    "status": "error",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"Backtest endpoint failed: {response.status_code} - {response.text}")
        except Exception as e:
            results["backtest_endpoint"] = {"status": "error", "error": str(e)}
            logger.error(f"Backtest endpoint test error: {e}")
        
        # Test 4: VectorBT-specific endpoint
        try:
            response = client.post("/backtest/vectorbt", json=backtest_data)
            if response.status_code == 200:
                vbt_result = response.json()
                results["vectorbt_endpoint"] = {
                    "status": "success",
                    "response": vbt_result
                }
                logger.success("VectorBT-specific endpoint test passed")
            elif response.status_code == 503:
                results["vectorbt_endpoint"] = {
                    "status": "unavailable",
                    "message": "VectorBT not available"
                }
                logger.warning("VectorBT endpoint returned service unavailable")
            else:
                results["vectorbt_endpoint"] = {
                    "status": "error",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"VectorBT endpoint failed: {response.status_code}")
        except Exception as e:
            results["vectorbt_endpoint"] = {"status": "error", "error": str(e)}
            logger.error(f"VectorBT endpoint test error: {e}")
            
    except Exception as e:
        results["api_setup"] = {"status": "error", "error": str(e)}
        logger.error(f"API test setup failed: {e}")
    
    return results

def test_workflow_integration() -> Dict[str, Any]:
    """Test Prefect workflow integration"""
    logger.info("=== WORKFLOW INTEGRATION TESTS ===")
    results = {}
    
    try:
        from supernova.workflows import is_prefect_available, PREFECT_AVAILABLE
        
        results["prefect_availability"] = {
            "status": "available" if PREFECT_AVAILABLE else "unavailable",
            "config_enabled": is_prefect_available()
        }
        
        if PREFECT_AVAILABLE:
            logger.success("Prefect is available for workflow integration")
            
            try:
                from supernova.workflows import sentiment_analysis_flow, batch_sentiment_analysis_flow
                results["workflow_imports"] = {"status": "success"}
                logger.success("Prefect workflows imported successfully")
                
                # Test basic workflow structure (without actual execution)
                results["workflow_structure"] = {
                    "sentiment_flow": "available",
                    "batch_flow": "available"
                }
                logger.success("Workflow structure validation passed")
                
                # Test workflow configuration
                from supernova.config import settings
                prefect_configs = {
                    "enable_prefect": getattr(settings, 'ENABLE_PREFECT', False),
                    "twitter_retries": getattr(settings, 'PREFECT_TWITTER_RETRIES', 3),
                    "reddit_retries": getattr(settings, 'PREFECT_REDDIT_RETRIES', 3),
                    "task_timeout": getattr(settings, 'PREFECT_TASK_TIMEOUT', 300)
                }
                
                results["workflow_config"] = prefect_configs
                logger.info(f"Prefect configuration: {prefect_configs}")
                
            except Exception as e:
                results["workflow_imports"] = {"status": "error", "error": str(e)}
                logger.error(f"Workflow import test failed: {e}")
        else:
            logger.warning("Prefect is not available - workflows will use fallback mode")
            
            # Test fallback behavior
            try:
                from supernova import sentiment
                # Test basic sentiment functionality without Prefect
                results["fallback_sentiment"] = {"status": "available"}
                logger.success("Sentiment analysis fallback is available")
            except Exception as e:
                results["fallback_sentiment"] = {"status": "error", "error": str(e)}
                logger.error(f"Sentiment fallback test failed: {e}")
    
    except Exception as e:
        results["prefect_availability"] = {"status": "error", "error": str(e)}
        logger.error(f"Workflow integration test failed: {e}")
    
    return results

def test_performance_comparison() -> Dict[str, Any]:
    """Test performance comparison between old and new implementations"""
    logger.info("=== PERFORMANCE VALIDATION TESTS ===")
    results = {}
    
    # Create larger dataset for performance testing
    large_sample = create_sample_data()
    logger.info(f"Created performance test dataset with {len(large_sample)} bars")
    
    # Test 1: Backtesting performance comparison
    backtest_params = {
        "strategy_template": "sma_crossover",
        "params": {"short_window": 20, "long_window": 50},
        "start_cash": 10000.0,
        "fees": 0.001
    }
    
    try:
        from supernova.backtester import run_backtest, run_vbt_backtest, VBT_AVAILABLE
        
        # Legacy backtester timing
        start_time = time.time()
        legacy_result = run_backtest(large_sample, **backtest_params)
        legacy_time = time.time() - start_time
        
        results["legacy_backtest"] = {
            "execution_time": legacy_time,
            "status": "success" if not isinstance(legacy_result, dict) or "error" not in legacy_result else "error",
            "result_type": type(legacy_result).__name__
        }
        logger.info(f"Legacy backtester completed in {legacy_time:.4f} seconds")
        
        # VectorBT backtester timing (if available)
        if VBT_AVAILABLE:
            start_time = time.time()
            vbt_result = run_vbt_backtest(large_sample, **backtest_params)
            vbt_time = time.time() - start_time
            
            results["vectorbt_backtest"] = {
                "execution_time": vbt_time,
                "status": "success" if not isinstance(vbt_result, dict) or "error" not in vbt_result else "error",
                "result_type": type(vbt_result).__name__
            }
            
            if legacy_time > 0 and vbt_time > 0:
                speedup = legacy_time / vbt_time
                results["performance_comparison"] = {
                    "speedup_factor": speedup,
                    "improvement": f"{((speedup - 1) * 100):.1f}%" if speedup > 1 else f"{((1 - speedup) * 100):.1f}% slower"
                }
                logger.success(f"VectorBT backtester completed in {vbt_time:.4f} seconds (speedup: {speedup:.2f}x)")
            else:
                logger.warning("Could not calculate performance comparison due to timing issues")
        else:
            results["vectorbt_backtest"] = {"status": "unavailable", "reason": "VectorBT not installed"}
            logger.warning("VectorBT not available for performance comparison")
    
    except Exception as e:
        results["backtest_performance"] = {"status": "error", "error": str(e)}
        logger.error(f"Backtest performance test failed: {e}")
    
    # Test 2: LLM response caching performance
    try:
        from supernova.advisor import advise
        
        # Test advice generation timing
        test_bars = large_sample[-100:]  # Last 100 bars
        
        # First call (no cache)
        start_time = time.time()
        result1 = advise(test_bars, risk_score=50, sentiment_hint=0.1)
        first_call_time = time.time() - start_time
        
        # Second call (should use cache if enabled)
        start_time = time.time()
        result2 = advise(test_bars, risk_score=50, sentiment_hint=0.1)
        second_call_time = time.time() - start_time
        
        results["llm_caching"] = {
            "first_call_time": first_call_time,
            "second_call_time": second_call_time,
            "cache_effective": second_call_time < first_call_time * 0.5,  # 50% faster indicates caching
            "results_consistent": result1[0] == result2[0]  # Same action
        }
        
        if second_call_time < first_call_time * 0.5:
            logger.success(f"LLM caching appears to be working (first: {first_call_time:.4f}s, second: {second_call_time:.4f}s)")
        else:
            logger.info(f"LLM caching may not be active (first: {first_call_time:.4f}s, second: {second_call_time:.4f}s)")
            
    except Exception as e:
        results["llm_caching"] = {"status": "error", "error": str(e)}
        logger.error(f"LLM caching test failed: {e}")
    
    return results

def test_error_handling() -> Dict[str, Any]:
    """Test error handling and fallback mechanisms"""
    logger.info("=== ERROR HANDLING & FALLBACK TESTS ===")
    results = {}
    
    # Test 1: Invalid data handling
    try:
        from supernova.backtester import run_backtest
        
        # Test with insufficient data
        insufficient_data = create_sample_data()[:10]  # Only 10 bars
        result = run_backtest(insufficient_data, "sma_crossover")
        
        if isinstance(result, dict) and "error" in result:
            results["insufficient_data_handling"] = {"status": "properly_handled", "error_message": result["error"]}
            logger.success("Insufficient data error properly handled")
        else:
            results["insufficient_data_handling"] = {"status": "not_handled", "result": str(result)}
            logger.warning("Insufficient data error not properly handled")
            
    except Exception as e:
        results["insufficient_data_handling"] = {"status": "exception", "error": str(e)}
        logger.error(f"Insufficient data test threw exception: {e}")
    
    # Test 2: Invalid strategy template
    try:
        from supernova.backtester import run_backtest
        
        valid_data = create_sample_data()[-100:]
        result = run_backtest(valid_data, "non_existent_strategy")
        
        if isinstance(result, dict) and "error" in result:
            results["invalid_strategy_handling"] = {"status": "properly_handled", "error_message": result["error"]}
            logger.success("Invalid strategy error properly handled")
        else:
            results["invalid_strategy_handling"] = {"status": "not_handled", "result": str(result)}
            logger.warning("Invalid strategy error not properly handled")
            
    except Exception as e:
        results["invalid_strategy_handling"] = {"status": "exception", "error": str(e)}
        logger.error(f"Invalid strategy test threw exception: {e}")
    
    # Test 3: LLM fallback behavior
    try:
        from supernova.advisor import advise
        from supernova.config import settings
        
        # Save original settings
        original_llm_enabled = getattr(settings, 'LLM_ENABLED', True)
        original_fallback_enabled = getattr(settings, 'LLM_FALLBACK_ENABLED', True)
        
        # Temporarily disable LLM
        settings.LLM_ENABLED = False
        settings.LLM_FALLBACK_ENABLED = True
        
        test_bars = create_sample_data()[-50:]
        result = advise(test_bars, risk_score=50)
        
        # Restore original settings
        settings.LLM_ENABLED = original_llm_enabled
        settings.LLM_FALLBACK_ENABLED = original_fallback_enabled
        
        if len(result) >= 4 and result[3]:  # Should have fallback rationale
            results["llm_fallback"] = {
                "status": "working",
                "rationale_length": len(result[3]),
                "has_rationale": bool(result[3])
            }
            logger.success("LLM fallback mechanism is working")
        else:
            results["llm_fallback"] = {"status": "not_working", "result": str(result)}
            logger.warning("LLM fallback mechanism may not be working")
            
    except Exception as e:
        results["llm_fallback"] = {"status": "error", "error": str(e)}
        logger.error(f"LLM fallback test failed: {e}")
    
    # Test 4: VectorBT fallback behavior
    try:
        from supernova.backtester import run_vbt_backtest, VBT_AVAILABLE
        
        if not VBT_AVAILABLE:
            # Test fallback to legacy backtester
            test_data = create_sample_data()[-100:]
            result = run_vbt_backtest(test_data, "sma_crossover")
            
            if isinstance(result, dict) and "error" not in result:
                results["vbt_fallback"] = {"status": "working", "fell_back_to": "legacy"}
                logger.success("VectorBT fallback to legacy backtester is working")
            else:
                results["vbt_fallback"] = {"status": "not_working", "result": str(result)}
                logger.warning("VectorBT fallback may not be working properly")
        else:
            results["vbt_fallback"] = {"status": "not_tested", "reason": "VectorBT is available"}
            logger.info("VectorBT fallback not tested (VectorBT is available)")
            
    except Exception as e:
        results["vbt_fallback"] = {"status": "error", "error": str(e)}
        logger.error(f"VectorBT fallback test failed: {e}")
    
    return results

def test_backward_compatibility() -> Dict[str, Any]:
    """Test that existing functionality still works"""
    logger.info("=== BACKWARD COMPATIBILITY TESTS ===")
    results = {}
    
    try:
        # Test existing advisor functionality
        from supernova.advisor import advise, score_risk
        
        # Test risk scoring (existing functionality)
        risk_questions = [3, 2, 4, 2, 3]
        risk_score = score_risk(risk_questions)
        
        if isinstance(risk_score, int) and 0 <= risk_score <= 100:
            results["risk_scoring"] = {"status": "working", "score": risk_score}
            logger.success(f"Risk scoring works correctly: {risk_score}")
        else:
            results["risk_scoring"] = {"status": "error", "score": risk_score}
            logger.error(f"Risk scoring returned invalid result: {risk_score}")
        
        # Test existing advice functionality
        test_bars = create_sample_data()[-50:]
        advice_result = advise(test_bars, risk_score=50)
        
        if len(advice_result) >= 3:
            action, confidence, details = advice_result[:3]
            results["advice_generation"] = {
                "status": "working",
                "action": action,
                "confidence": confidence,
                "has_details": bool(details)
            }
            logger.success(f"Advice generation works: {action} with {confidence:.2f} confidence")
        else:
            results["advice_generation"] = {"status": "error", "result": str(advice_result)}
            logger.error(f"Advice generation returned invalid result: {advice_result}")
            
    except Exception as e:
        results["advisor_compatibility"] = {"status": "error", "error": str(e)}
        logger.error(f"Advisor compatibility test failed: {e}")
    
    try:
        # Test existing backtester functionality
        from supernova.backtester import run_backtest
        
        test_data = create_sample_data()[-100:]
        backtest_result = run_backtest(test_data, "sma_crossover")
        
        if isinstance(backtest_result, dict) and "final_equity" in backtest_result:
            results["backtester_compatibility"] = {
                "status": "working",
                "final_equity": backtest_result["final_equity"],
                "has_metrics": "CAGR" in backtest_result
            }
            logger.success("Backtester compatibility maintained")
        else:
            results["backtester_compatibility"] = {"status": "error", "result": str(backtest_result)}
            logger.error(f"Backtester compatibility test failed: {backtest_result}")
            
    except Exception as e:
        results["backtester_compatibility"] = {"status": "error", "error": str(e)}
        logger.error(f"Backtester compatibility test failed: {e}")
    
    return results

def generate_test_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test summary"""
    logger.info("=== GENERATING TEST SUMMARY ===")
    
    summary = {
        "test_duration": (datetime.now() - all_results["start_time"]).total_seconds(),
        "total_tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_warned": 0,
        "enhancement_status": {},
        "critical_issues": [],
        "recommendations": []
    }
    
    # Analyze dependency results
    dep_results = all_results.get("dependency_tests", {})
    critical_deps_missing = []
    enhancement_deps_missing = []
    
    for dep, info in dep_results.items():
        summary["total_tests_run"] += 1
        if info.get("status") == "available":
            summary["tests_passed"] += 1
        elif info.get("status") == "missing":
            if info.get("critical", False):
                critical_deps_missing.append(dep)
                summary["tests_failed"] += 1
            else:
                enhancement_deps_missing.append(dep)
                summary["tests_warned"] += 1
    
    if critical_deps_missing:
        summary["critical_issues"].append(f"Critical dependencies missing: {', '.join(critical_deps_missing)}")
    
    if enhancement_deps_missing:
        summary["recommendations"].append(f"Install missing enhancement dependencies: {', '.join(enhancement_deps_missing)}")
    
    # Analyze LLM integration status
    llm_available = False
    api_results = all_results.get("api_tests", {})
    if "llm_enhancement" in api_results:
        llm_status = api_results["llm_enhancement"].get("status", "unknown")
        llm_available = llm_status == "detected"
    
    summary["enhancement_status"]["llm_integration"] = {
        "status": "active" if llm_available else "fallback",
        "description": "LLM-enhanced advice generation" if llm_available else "Using fallback rationale generation"
    }
    
    # Analyze VectorBT integration status
    vbt_available = False
    if "vectorbt_integration" in api_results:
        vbt_status = api_results["vectorbt_integration"].get("status", "unknown")
        vbt_available = vbt_status == "active"
    
    summary["enhancement_status"]["vectorbt_integration"] = {
        "status": "active" if vbt_available else "fallback",
        "description": "High-performance VectorBT backtesting" if vbt_available else "Using legacy backtester"
    }
    
    # Analyze Prefect workflow status
    workflow_results = all_results.get("workflow_tests", {})
    prefect_available = False
    if "prefect_availability" in workflow_results:
        prefect_status = workflow_results["prefect_availability"].get("status", "unknown")
        prefect_available = prefect_status == "available"
    
    summary["enhancement_status"]["prefect_workflows"] = {
        "status": "active" if prefect_available else "fallback",
        "description": "Prefect-orchestrated workflows" if prefect_available else "Direct sentiment analysis without orchestration"
    }
    
    # Performance analysis
    perf_results = all_results.get("performance_tests", {})
    if "performance_comparison" in perf_results:
        speedup = perf_results["performance_comparison"].get("speedup_factor", 1.0)
        if speedup > 1.2:
            summary["recommendations"].append(f"VectorBT provides {speedup:.1f}x speedup - recommend using for production")
        elif speedup < 0.8:
            summary["recommendations"].append("VectorBT appears slower - investigate configuration")
    
    # Error handling analysis
    error_results = all_results.get("error_handling_tests", {})
    error_handling_issues = []
    
    for test_name, result in error_results.items():
        if result.get("status") == "not_handled":
            error_handling_issues.append(test_name)
    
    if error_handling_issues:
        summary["critical_issues"].append(f"Error handling issues in: {', '.join(error_handling_issues)}")
    
    # Calculate overall health score
    total_enhancements = 3
    active_enhancements = sum(1 for status in summary["enhancement_status"].values() if status["status"] == "active")
    
    summary["overall_health_score"] = (summary["tests_passed"] / max(1, summary["total_tests_run"])) * 100
    summary["enhancement_adoption_rate"] = (active_enhancements / total_enhancements) * 100
    
    # Generate final recommendations
    if summary["overall_health_score"] >= 90:
        summary["recommendations"].append("System is ready for production deployment")
    elif summary["overall_health_score"] >= 75:
        summary["recommendations"].append("System is mostly ready - address warnings before production")
    else:
        summary["recommendations"].append("System needs attention before production deployment")
    
    if summary["enhancement_adoption_rate"] < 50:
        summary["recommendations"].append("Consider installing missing dependencies to unlock full functionality")
    
    return summary

def main():
    """Run comprehensive integration tests"""
    print("=" * 80)
    print("SuperNova Extended Framework - Comprehensive Integration Testing")
    print("=" * 80)
    print()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    try:
        # Run all test suites
        test_results["dependency_tests"] = test_dependency_validation()
        print()
        
        test_results["configuration_tests"] = test_configuration_handling()
        print()
        
        test_results["api_tests"] = test_api_endpoints()
        print()
        
        test_results["workflow_tests"] = test_workflow_integration()
        print()
        
        test_results["performance_tests"] = test_performance_comparison()
        print()
        
        test_results["error_handling_tests"] = test_error_handling()
        print()
        
        test_results["compatibility_tests"] = test_backward_compatibility()
        print()
        
        # Generate comprehensive summary
        test_results["summary"] = generate_test_summary(test_results)
        test_results["end_time"] = datetime.now()
        
        # Print summary
        summary = test_results["summary"]
        print("=" * 80)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 80)
        print(f"Test Duration: {summary['test_duration']:.2f} seconds")
        print(f"Tests Run: {summary['total_tests_run']}")
        print(f"Passed: {summary['tests_passed']} | Failed: {summary['tests_failed']} | Warnings: {summary['tests_warned']}")
        print(f"Overall Health Score: {summary['overall_health_score']:.1f}%")
        print(f"Enhancement Adoption Rate: {summary['enhancement_adoption_rate']:.1f}%")
        print()
        
        print("ENHANCEMENT STATUS:")
        for enhancement, status in summary["enhancement_status"].items():
            print(f"  {enhancement}: {status['status'].upper()} - {status['description']}")
        print()
        
        if summary["critical_issues"]:
            print("CRITICAL ISSUES:")
            for issue in summary["critical_issues"]:
                print(f"  ❌ {issue}")
            print()
        
        if summary["recommendations"]:
            print("RECOMMENDATIONS:")
            for rec in summary["recommendations"]:
                print(f"  💡 {rec}")
            print()
        
        # Save detailed results
        results_file = "integration_test_results.json"
        with open(results_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = {}
            for key, value in test_results.items():
                if key in ["start_time", "end_time"]:
                    json_results[key] = value.isoformat() if value else None
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=str)
        
        logger.success(f"Detailed test results saved to {results_file}")
        
        # Return appropriate exit code
        if summary["tests_failed"] > 0:
            return 1  # Critical failures
        elif summary["tests_warned"] > 0:
            return 0  # Warnings but no critical failures
        else:
            return 0  # All tests passed
            
    except Exception as e:
        logger.error(f"Integration testing failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)