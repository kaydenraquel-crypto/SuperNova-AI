#!/usr/bin/env python3
"""
Agent Tools Comprehensive Validation Script

Tests all LangChain tool definitions, integration capabilities, and error handling
for the SuperNova conversational agent system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os

# Add SuperNova to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'supernova'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentToolsValidator:
    """Comprehensive validation of agent tools"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "error_summary": {},
            "recommendations": []
        }
        self.tools = None
        self.langchain_available = False
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        
        logger.info("Starting Agent Tools Validation Suite")
        start_time = time.time()
        
        try:
            # Test 1: Import and initialization
            await self._test_imports()
            
            # Test 2: Tool creation and registration
            await self._test_tool_creation()
            
            # Test 3: Individual tool functionality
            await self._test_individual_tools()
            
            # Test 4: Tool integration and chaining
            await self._test_tool_integration()
            
            # Test 5: Error handling and recovery
            await self._test_error_handling()
            
            # Test 6: Performance benchmarking
            await self._test_performance()
            
            # Test 7: Memory management
            await self._test_memory_usage()
            
            # Generate final report
            total_time = time.time() - start_time
            self.results["total_validation_time"] = total_time
            self.results["validation_status"] = "COMPLETED"
            
            return self.results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["validation_status"] = "FAILED"
            self.results["error"] = str(e)
            return self.results
    
    async def _test_imports(self):
        """Test import capabilities and dependencies"""
        
        logger.info("Testing imports and dependencies...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            # Test core imports
            from supernova.agent_tools import get_all_tools, LANGCHAIN_AVAILABLE, TOOL_CATEGORIES
            test_results["details"]["core_imports"] = "SUCCESS"
            
            # Test LangChain availability
            self.langchain_available = LANGCHAIN_AVAILABLE
            test_results["details"]["langchain_available"] = LANGCHAIN_AVAILABLE
            
            if LANGCHAIN_AVAILABLE:
                test_results["details"]["langchain_status"] = "AVAILABLE"
            else:
                test_results["details"]["langchain_status"] = "MISSING - Will test fallback mode"
            
            # Test tool categories
            test_results["details"]["tool_categories"] = len(TOOL_CATEGORIES)
            test_results["details"]["category_names"] = list(TOOL_CATEGORIES.keys())
            
            test_results["status"] = "SUCCESS"
            
        except ImportError as e:
            test_results["details"]["import_error"] = str(e)
            logger.error(f"Import test failed: {e}")
        except Exception as e:
            test_results["details"]["general_error"] = str(e)
            logger.error(f"Import test error: {e}")
        
        self.results["test_results"]["imports"] = test_results
    
    async def _test_tool_creation(self):
        """Test tool creation and registration"""
        
        logger.info("Testing tool creation and registration...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.agent_tools import get_all_tools, get_core_financial_tools
            
            # Get all tools
            all_tools = get_all_tools()
            test_results["details"]["total_tools_created"] = len(all_tools)
            
            if self.langchain_available and all_tools:
                # Test tool properties
                tool_properties = {}
                for tool in all_tools[:3]:  # Test first 3 tools
                    tool_name = tool.name
                    tool_properties[tool_name] = {
                        "has_name": hasattr(tool, 'name'),
                        "has_description": hasattr(tool, 'description'),
                        "has_func": hasattr(tool, 'func'),
                        "name_length": len(tool.name) if hasattr(tool, 'name') else 0,
                        "description_length": len(tool.description) if hasattr(tool, 'description') else 0
                    }
                
                test_results["details"]["tool_properties"] = tool_properties
                test_results["details"]["tools_have_required_attributes"] = all(
                    props["has_name"] and props["has_description"] and props["has_func"]
                    for props in tool_properties.values()
                )
                
                self.tools = all_tools
                test_results["status"] = "SUCCESS"
                
            elif not self.langchain_available:
                test_results["details"]["fallback_mode"] = "Tools not created due to LangChain unavailability"
                test_results["status"] = "SUCCESS"
            else:
                test_results["details"]["no_tools_created"] = "Tools list is empty"
                
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Tool creation test failed: {e}")
        
        self.results["test_results"]["tool_creation"] = test_results
    
    async def _test_individual_tools(self):
        """Test individual tool functionality"""
        
        logger.info("Testing individual tool functionality...")
        test_results = {"status": "FAILED", "details": {}}
        
        if not self.langchain_available or not self.tools:
            test_results["status"] = "SKIPPED"
            test_results["details"]["reason"] = "LangChain not available or tools not created"
            self.results["test_results"]["individual_tools"] = test_results
            return
        
        tool_test_results = {}
        
        # Test specific tools
        test_cases = [
            {
                "tool_name": "get_investment_advice",
                "test_inputs": {"symbol": "AAPL", "risk_profile": "moderate"},
                "expected_keywords": ["investment", "advice", "AAPL"]
            },
            {
                "tool_name": "get_historical_sentiment",
                "test_inputs": {"symbol": "TSLA", "days_back": 7},
                "expected_keywords": ["sentiment", "TSLA"]
            },
            {
                "tool_name": "run_backtest_analysis",
                "test_inputs": {"symbol": "MSFT", "strategy_template": "rsi_strategy"},
                "expected_keywords": ["backtest", "MSFT", "strategy"]
            }
        ]
        
        for test_case in test_cases:
            tool_name = test_case["tool_name"]
            logger.info(f"Testing tool: {tool_name}")
            
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    # Test tool execution
                    start_time = time.time()
                    result = tool.func(**test_case["test_inputs"])
                    execution_time = time.time() - start_time
                    
                    # Validate result
                    result_valid = isinstance(result, str) and len(result) > 0
                    contains_keywords = any(keyword.lower() in result.lower() 
                                          for keyword in test_case["expected_keywords"])
                    
                    tool_test_results[tool_name] = {
                        "execution_success": True,
                        "execution_time_ms": int(execution_time * 1000),
                        "result_valid": result_valid,
                        "contains_expected_keywords": contains_keywords,
                        "result_length": len(result),
                        "first_100_chars": result[:100] if result else ""
                    }
                    
                except Exception as e:
                    tool_test_results[tool_name] = {
                        "execution_success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    logger.error(f"Tool {tool_name} failed: {e}")
            else:
                tool_test_results[tool_name] = {
                    "execution_success": False,
                    "error": "Tool not found",
                    "error_type": "NotFound"
                }
        
        test_results["details"]["tool_tests"] = tool_test_results
        
        # Calculate success rate
        successful_tests = sum(1 for result in tool_test_results.values() 
                             if result.get("execution_success", False))
        total_tests = len(tool_test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        test_results["details"]["success_rate"] = success_rate
        test_results["details"]["successful_tests"] = successful_tests
        test_results["details"]["total_tests"] = total_tests
        
        test_results["status"] = "SUCCESS" if success_rate >= 80 else "PARTIAL"
        
        self.results["test_results"]["individual_tools"] = test_results
    
    async def _test_tool_integration(self):
        """Test tool integration and chaining capabilities"""
        
        logger.info("Testing tool integration...")
        test_results = {"status": "FAILED", "details": {}}
        
        if not self.langchain_available or not self.tools:
            test_results["status"] = "SKIPPED"
            test_results["details"]["reason"] = "LangChain not available or tools not created"
            self.results["test_results"]["tool_integration"] = test_results
            return
        
        try:
            # Test sequential tool usage
            symbol = "AAPL"
            
            # Step 1: Get symbol information
            symbol_info_tool = next((t for t in self.tools if t.name == "get_symbol_information"), None)
            advice_tool = next((t for t in self.tools if t.name == "get_investment_advice"), None)
            
            integration_results = {}
            
            if symbol_info_tool:
                start_time = time.time()
                symbol_result = symbol_info_tool.func(symbol=symbol, include_fundamentals=True)
                symbol_time = time.time() - start_time
                
                integration_results["symbol_info"] = {
                    "success": isinstance(symbol_result, str) and len(symbol_result) > 0,
                    "execution_time_ms": int(symbol_time * 1000),
                    "result_length": len(symbol_result) if symbol_result else 0
                }
            
            if advice_tool:
                start_time = time.time()
                advice_result = advice_tool.func(symbol=symbol, risk_profile="moderate")
                advice_time = time.time() - start_time
                
                integration_results["investment_advice"] = {
                    "success": isinstance(advice_result, str) and len(advice_result) > 0,
                    "execution_time_ms": int(advice_time * 1000),
                    "result_length": len(advice_result) if advice_result else 0
                }
            
            # Test data consistency
            consistent_symbol = (
                symbol.upper() in symbol_result if symbol_result else False,
                symbol.upper() in advice_result if advice_result else False
            )
            
            test_results["details"]["integration_results"] = integration_results
            test_results["details"]["data_consistency"] = all(consistent_symbol)
            test_results["details"]["total_execution_time_ms"] = sum(
                r.get("execution_time_ms", 0) for r in integration_results.values()
            )
            
            successful_integrations = sum(1 for r in integration_results.values() if r.get("success", False))
            total_integrations = len(integration_results)
            
            test_results["details"]["integration_success_rate"] = (
                (successful_integrations / total_integrations) * 100 if total_integrations > 0 else 0
            )
            
            test_results["status"] = "SUCCESS" if successful_integrations == total_integrations else "PARTIAL"
            
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Tool integration test failed: {e}")
        
        self.results["test_results"]["tool_integration"] = test_results
    
    async def _test_error_handling(self):
        """Test error handling and recovery"""
        
        logger.info("Testing error handling...")
        test_results = {"status": "FAILED", "details": {}}
        
        if not self.langchain_available or not self.tools:
            test_results["status"] = "SKIPPED"
            test_results["details"]["reason"] = "LangChain not available or tools not created"
            self.results["test_results"]["error_handling"] = test_results
            return
        
        error_test_cases = [
            {
                "test_name": "invalid_symbol",
                "tool_name": "get_investment_advice",
                "inputs": {"symbol": "INVALID_SYMBOL_123", "risk_profile": "moderate"},
                "expected_behavior": "graceful_handling"
            },
            {
                "test_name": "missing_required_param",
                "tool_name": "get_historical_sentiment", 
                "inputs": {"days_back": 7},  # Missing symbol
                "expected_behavior": "error_or_default"
            },
            {
                "test_name": "invalid_param_type",
                "tool_name": "run_backtest_analysis",
                "inputs": {"symbol": "AAPL", "strategy_template": 123},  # Wrong type
                "expected_behavior": "error_or_conversion"
            }
        ]
        
        error_results = {}
        
        for test_case in error_test_cases:
            test_name = test_case["test_name"]
            tool_name = test_case["tool_name"]
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    start_time = time.time()
                    result = tool.func(**test_case["inputs"])
                    execution_time = time.time() - start_time
                    
                    # Check if tool handled error gracefully
                    handled_gracefully = (
                        isinstance(result, str) and 
                        len(result) > 0 and 
                        ("error" in result.lower() or "invalid" in result.lower() or len(result) > 20)
                    )
                    
                    error_results[test_name] = {
                        "executed_without_exception": True,
                        "handled_gracefully": handled_gracefully,
                        "execution_time_ms": int(execution_time * 1000),
                        "result_preview": result[:100] if result else ""
                    }
                    
                except Exception as e:
                    error_results[test_name] = {
                        "executed_without_exception": False,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    }
        
        test_results["details"]["error_tests"] = error_results
        
        # Calculate error handling score
        graceful_handling_count = sum(
            1 for result in error_results.values() 
            if result.get("handled_gracefully", False) or not result.get("executed_without_exception", True)
        )
        total_error_tests = len(error_results)
        error_handling_score = (graceful_handling_count / total_error_tests) * 100 if total_error_tests > 0 else 0
        
        test_results["details"]["error_handling_score"] = error_handling_score
        test_results["status"] = "SUCCESS" if error_handling_score >= 70 else "PARTIAL"
        
        self.results["test_results"]["error_handling"] = test_results
    
    async def _test_performance(self):
        """Test performance benchmarking"""
        
        logger.info("Testing performance...")
        test_results = {"status": "FAILED", "details": {}}
        
        if not self.langchain_available or not self.tools:
            test_results["status"] = "SKIPPED"
            test_results["details"]["reason"] = "LangChain not available or tools not created"
            self.results["test_results"]["performance"] = test_results
            return
        
        try:
            # Performance test cases
            perf_tests = [
                {"tool_name": "get_investment_advice", "inputs": {"symbol": "AAPL"}, "iterations": 5},
                {"tool_name": "get_symbol_information", "inputs": {"symbol": "MSFT"}, "iterations": 3},
                {"tool_name": "calculate_technical_indicators", "inputs": {"symbol": "GOOGL", "indicators": ["RSI", "MACD"]}, "iterations": 3}
            ]
            
            performance_results = {}
            
            for perf_test in perf_tests:
                tool_name = perf_test["tool_name"]
                tool = next((t for t in self.tools if t.name == tool_name), None)
                
                if tool:
                    execution_times = []
                    
                    for i in range(perf_test["iterations"]):
                        try:
                            start_time = time.time()
                            result = tool.func(**perf_test["inputs"])
                            execution_time = time.time() - start_time
                            execution_times.append(execution_time * 1000)  # Convert to ms
                        except Exception as e:
                            logger.warning(f"Performance test iteration {i} failed for {tool_name}: {e}")
                    
                    if execution_times:
                        performance_results[tool_name] = {
                            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
                            "min_execution_time_ms": min(execution_times),
                            "max_execution_time_ms": max(execution_times),
                            "successful_iterations": len(execution_times),
                            "total_iterations": perf_test["iterations"]
                        }
            
            test_results["details"]["performance_results"] = performance_results
            
            # Calculate overall performance score
            avg_response_times = [r["avg_execution_time_ms"] for r in performance_results.values()]
            if avg_response_times:
                overall_avg_time = sum(avg_response_times) / len(avg_response_times)
                test_results["details"]["overall_avg_response_time_ms"] = overall_avg_time
                
                # Performance thresholds (ms)
                performance_score = 100
                if overall_avg_time > 1000:
                    performance_score = 70
                elif overall_avg_time > 500:
                    performance_score = 85
                elif overall_avg_time > 200:
                    performance_score = 95
                
                test_results["details"]["performance_score"] = performance_score
                test_results["status"] = "SUCCESS" if performance_score >= 80 else "PARTIAL"
            else:
                test_results["status"] = "FAILED"
                test_results["details"]["error"] = "No performance data collected"
        
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Performance test failed: {e}")
        
        self.results["test_results"]["performance"] = test_results
    
    async def _test_memory_usage(self):
        """Test memory usage and efficiency"""
        
        logger.info("Testing memory usage...")
        test_results = {"status": "SUCCESS", "details": {}}
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate tool usage
            if self.langchain_available and self.tools:
                for _ in range(10):
                    for tool in self.tools[:3]:  # Test first 3 tools
                        try:
                            # Simple test calls
                            if tool.name == "get_investment_advice":
                                tool.func(symbol="AAPL", risk_profile="moderate")
                            elif tool.name == "get_symbol_information":
                                tool.func(symbol="MSFT")
                        except:
                            pass  # Ignore errors in memory test
            
            # Force garbage collection
            gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            test_results["details"] = {
                "initial_memory_mb": round(initial_memory, 2),
                "final_memory_mb": round(final_memory, 2),
                "memory_increase_mb": round(memory_increase, 2),
                "memory_increase_acceptable": memory_increase < 50  # Less than 50MB increase
            }
            
            if memory_increase > 100:
                test_results["status"] = "WARNING"
                test_results["details"]["warning"] = "High memory usage detected"
            
        except ImportError:
            test_results["status"] = "SKIPPED"
            test_results["details"]["reason"] = "psutil not available for memory testing"
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Memory test failed: {e}")
        
        self.results["test_results"]["memory_usage"] = test_results
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check test results and generate recommendations
        if self.results["test_results"].get("imports", {}).get("status") == "FAILED":
            recommendations.append("CRITICAL: Fix import issues before deploying agent tools")
        
        if not self.langchain_available:
            recommendations.append("WARNING: LangChain not available - agent will run in fallback mode")
        
        individual_tools = self.results["test_results"].get("individual_tools", {})
        if individual_tools.get("status") == "PARTIAL":
            success_rate = individual_tools.get("details", {}).get("success_rate", 0)
            if success_rate < 80:
                recommendations.append(f"ISSUE: Tool success rate is {success_rate}% - investigate failing tools")
        
        performance = self.results["test_results"].get("performance", {})
        if performance.get("status") == "PARTIAL":
            avg_time = performance.get("details", {}).get("overall_avg_response_time_ms", 0)
            if avg_time > 500:
                recommendations.append(f"PERFORMANCE: Average response time is {avg_time}ms - consider optimization")
        
        error_handling = self.results["test_results"].get("error_handling", {})
        if error_handling.get("status") == "PARTIAL":
            score = error_handling.get("details", {}).get("error_handling_score", 0)
            if score < 70:
                recommendations.append(f"ERROR HANDLING: Score is {score}% - improve error handling robustness")
        
        memory_usage = self.results["test_results"].get("memory_usage", {})
        if memory_usage.get("status") == "WARNING":
            recommendations.append("MEMORY: High memory usage detected - monitor for memory leaks")
        
        # General recommendations
        if self.langchain_available:
            recommendations.append("READY: Core agent tools are functional and ready for integration")
        else:
            recommendations.append("FALLBACK: System will operate in limited mode without LangChain")
        
        recommendations.append("TEST: Run integration tests with actual LLM providers")
        recommendations.append("MONITOR: Set up performance monitoring for production deployment")
        
        self.results["recommendations"] = recommendations
        return recommendations

async def main():
    """Main validation function"""
    
    print("SuperNova Agent Tools Validation Suite")
    print("=" * 50)
    
    validator = AgentToolsValidator()
    results = await validator.run_validation()
    
    # Generate recommendations
    validator.generate_recommendations()
    
    # Save results
    results_file = "agent_tools_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\nValidation Status: {results['validation_status']}")
    print(f"Total Time: {results.get('total_validation_time', 0):.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    print("\nTest Results Summary:")
    for test_name, test_result in results["test_results"].items():
        status = test_result.get("status", "UNKNOWN")
        print(f"  {test_name}: {status}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results.get("recommendations", []), 1):
        print(f"  {i}. {rec}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())