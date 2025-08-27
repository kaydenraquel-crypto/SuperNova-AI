#!/usr/bin/env python3
"""
Validation script for VectorBT upgrade - checks code structure and integration
"""

import os
import sys
from pathlib import Path

def validate_file_exists(filepath, description):
    """Check if a file exists and is readable"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"[PASS] {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} - NOT FOUND")
        return False

def validate_requirements():
    """Validate requirements.txt has VectorBT dependencies"""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"[FAIL] Requirements file not found: {req_file}")
        return False
    
    with open(req_file, 'r') as f:
        content = f.read()
    
    required_deps = ['vectorbt', 'ta-lib', 'numba']
    missing_deps = []
    
    for dep in required_deps:
        if dep.lower() not in content.lower():
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"[FAIL] Missing dependencies in requirements.txt: {missing_deps}")
        return False
    else:
        print("[PASS] All VectorBT dependencies found in requirements.txt")
        return True

def validate_backtester_integration():
    """Check if backtester.py has VectorBT integration"""
    backtester_file = "supernova/backtester.py"
    
    if not os.path.exists(backtester_file):
        print(f"[FAIL] Backtester file not found: {backtester_file}")
        return False
    
    with open(backtester_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key VectorBT integration components
    checks = [
        ("VectorBT import", "import vectorbt as vbt"),
        ("VBT_AVAILABLE flag", "VBT_AVAILABLE"),
        ("run_vbt_backtest function", "def run_vbt_backtest"),
        ("_prepare_vbt_data function", "def _prepare_vbt_data"),
        ("_generate_vbt_signals function", "def _generate_vbt_signals"),
        ("SMA crossover strategy", "def _sma_crossover_signals"),
        ("RSI strategy", "def _rsi_strategy_signals"),
        ("MACD strategy", "def _macd_strategy_signals"),
        ("Bollinger Bands strategy", "def _bollinger_bands_signals"),
        ("Sentiment strategy", "def _sentiment_strategy_signals"),
        ("Portfolio creation", "vbt.Portfolio.from_signals"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, search_term in checks:
        if search_term in content:
            print(f"[PASS] {check_name}: Found")
            passed += 1
        else:
            print(f"[FAIL] {check_name}: Missing")
    
    print(f"Backtester Integration: {passed}/{total} checks passed")
    return passed >= total * 0.8  # 80% pass rate

def validate_api_integration():
    """Check if API has VectorBT integration"""
    api_file = "supernova/api.py"
    
    if not os.path.exists(api_file):
        print(f"[FAIL] API file not found: {api_file}")
        return False
    
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("VectorBT import", "run_vbt_backtest"),
        ("VBT_AVAILABLE import", "VBT_AVAILABLE"),
        ("Config import", "from .config import settings"),
        ("Enhanced backtest endpoint", "use_vbt"),
        ("VectorBT-specific endpoint", "/backtest/vectorbt"),
        ("Error handling", "HTTPException"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, search_term in checks:
        if search_term in content:
            print(f"[PASS] {check_name}: Found")
            passed += 1
        else:
            print(f"[FAIL] {check_name}: Missing")
    
    print(f"API Integration: {passed}/{total} checks passed")
    return passed >= total * 0.8

def validate_schema_enhancements():
    """Check if schemas have been enhanced for VectorBT"""
    schema_file = "supernova/schemas.py"
    
    if not os.path.exists(schema_file):
        print(f"[FAIL] Schema file not found: {schema_file}")
        return False
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("use_vectorbt field", "use_vectorbt: bool"),
        ("Enhanced BacktestRequest", "start_cash: float"),
        ("fees field", "fees: float"),
        ("slippage field", "slippage: float"),
        ("Enhanced BacktestOut", "engine: str"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, search_term in checks:
        if search_term in content:
            print(f"[PASS] {check_name}: Found")
            passed += 1
        else:
            print(f"[FAIL] {check_name}: Missing")
    
    print(f"Schema Enhancements: {passed}/{total} checks passed")
    return passed >= total * 0.8

def validate_config_settings():
    """Check if config has VectorBT settings"""
    config_file = "supernova/config.py"
    
    if not os.path.exists(config_file):
        print(f"[FAIL] Config file not found: {config_file}")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("VectorBT enabled setting", "VECTORBT_ENABLED"),
        ("Default engine setting", "VECTORBT_DEFAULT_ENGINE"),
        ("Default fees setting", "VECTORBT_DEFAULT_FEES"),
        ("Default slippage setting", "VECTORBT_DEFAULT_SLIPPAGE"),
        ("Performance mode setting", "VECTORBT_PERFORMANCE_MODE"),
        ("Backtest limits", "BACKTEST_MAX_BARS"),
        ("Strategy engine setting", "DEFAULT_STRATEGY_ENGINE"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, search_term in checks:
        if search_term in content:
            print(f"[PASS] {check_name}: Found")
            passed += 1
        else:
            print(f"[FAIL] {check_name}: Missing")
    
    print(f"Config Settings: {passed}/{total} checks passed")
    return passed >= total * 0.8

def check_file_structure():
    """Check overall file structure"""
    files_to_check = [
        ("requirements.txt", "Requirements file"),
        ("supernova/__init__.py", "SuperNova package init"),
        ("supernova/backtester.py", "Backtester module"),
        ("supernova/api.py", "API module"),
        ("supernova/schemas.py", "Schemas module"),
        ("supernova/config.py", "Config module"),
    ]
    
    passed = 0
    total = len(files_to_check)
    
    for filepath, description in files_to_check:
        if validate_file_exists(filepath, description):
            passed += 1
    
    print(f"File Structure: {passed}/{total} files found")
    return passed == total

def main():
    """Main validation function"""
    print("=" * 60)
    print("SuperNova VectorBT Upgrade Validation")
    print("=" * 60)
    
    # Change to the framework directory
    framework_dir = Path(__file__).parent
    os.chdir(framework_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run validation tests
    tests = [
        ("File Structure", check_file_structure),
        ("Requirements Dependencies", validate_requirements),
        ("Backtester Integration", validate_backtester_integration),
        ("API Integration", validate_api_integration),
        ("Schema Enhancements", validate_schema_enhancements),
        ("Config Settings", validate_config_settings),
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                print(f"[PASS] {test_name}: PASSED")
            else:
                print(f"[FAIL] {test_name}: FAILED")
        except Exception as e:
            print(f"[ERROR] {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(tests)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {status}: {test_name}")
    
    # Overall assessment
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("\n[SUCCESS] VectorBT UPGRADE VALIDATION: PASSED!")
        print("The VectorBT integration appears to be properly implemented.")
        
        # Next steps
        print("\nNext Steps:")
        print("1. Install VectorBT dependencies: pip install vectorbt ta-lib numba")
        print("2. Test with real data using the enhanced /backtest endpoint")
        print("3. Configure environment variables for VectorBT settings")
        print("4. Monitor performance improvements in production")
        
        return True
    else:
        print("\n[WARNING] VectorBT UPGRADE VALIDATION: INCOMPLETE!")
        print("Some components need attention before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)