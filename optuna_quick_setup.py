#!/usr/bin/env python3
"""
SuperNova Optuna Optimization - Quick Setup & Test Script

This script helps set up the optimization system and run basic functionality tests.
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, Any

def run_command(cmd: str, description: str = "") -> tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        print(f"Running: {description or cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ Success")
            return True, result.stdout
        else:
            print(f"✗ Failed: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout")
        return False, "Command timed out"
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, str(e)

def check_file_exists(filepath: str) -> bool:
    """Check if file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists

def test_basic_imports():
    """Test basic Python imports"""
    print("\n=== Testing Basic Imports ===")
    
    imports_to_test = [
        ("import json", "JSON support"),
        ("import sqlite3", "SQLite support"),
        ("import asyncio", "Asyncio support"),
        ("from datetime import datetime", "Datetime support"),
        ("from typing import Dict, List", "Type hints support")
    ]
    
    passed = 0
    for import_stmt, desc in imports_to_test:
        try:
            exec(import_stmt)
            print(f"✓ {desc}")
            passed += 1
        except ImportError as e:
            print(f"✗ {desc}: {e}")
    
    return passed == len(imports_to_test)

def test_optimization_imports():
    """Test optimization-specific imports"""
    print("\n=== Testing Optimization Imports ===")
    
    tests = []
    
    # Test core optimization imports
    try:
        from supernova.optimizer import OptunaOptimizer, OptimizationConfig
        print("✓ Core optimizer classes")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Core optimizer classes: {e}")
        tests.append(False)
    
    # Test models
    try:
        from supernova.optimization_models import OptimizationStudyModel
        print("✓ Database models")  
        tests.append(True)
    except ImportError as e:
        print(f"✗ Database models: {e}")
        tests.append(False)
    
    # Test API
    try:
        from supernova.api import app
        print("✓ API application")
        tests.append(True)
    except ImportError as e:
        print(f"✗ API application: {e}")
        tests.append(False)
    
    # Test workflows
    try:
        from supernova.workflows import optimize_strategy_parameters_flow
        print("✓ Workflow functions")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Workflow functions: {e}")
        tests.append(False)
    
    return all(tests)

def test_database_creation():
    """Test database table creation"""
    print("\n=== Testing Database Setup ===")
    
    try:
        # Create temporary database
        temp_db = tempfile.mktemp(suffix='.db')
        
        from sqlalchemy import create_engine
        from supernova.db import Base
        from supernova.optimization_models import OptimizationStudyModel
        
        # Create engine and tables
        engine = create_engine(f"sqlite:///{temp_db}")
        Base.metadata.create_all(engine)
        
        print("✓ Database tables created successfully")
        
        # Clean up
        if os.path.exists(temp_db):
            os.remove(temp_db)
        
        return True
        
    except Exception as e:
        print(f"✗ Database creation failed: {e}")
        return False

def test_optimizer_basic():
    """Test basic optimizer functionality"""
    print("\n=== Testing Optimizer Basics ===")
    
    try:
        from supernova.optimizer import OptunaOptimizer, OptimizationConfig, OPTUNA_AVAILABLE
        
        if not OPTUNA_AVAILABLE:
            print("✗ Optuna not available")
            return False
        
        print("✓ Optuna is available")
        
        # Test config creation
        config = OptimizationConfig(
            strategy_template="sma_crossover",
            n_trials=5,
            primary_objective="sharpe_ratio"
        )
        print("✓ Configuration created")
        
        # Test optimizer instantiation
        temp_db = tempfile.mktemp(suffix='.db')
        optimizer = OptunaOptimizer(storage_url=f"sqlite:///{temp_db}")
        print("✓ Optimizer instantiated")
        
        # Test study creation
        study_name = f"test_study_{int(datetime.now().timestamp())}"
        study = optimizer.create_study(study_name, config)
        print("✓ Study created")
        
        # Clean up
        if os.path.exists(temp_db):
            os.remove(temp_db)
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False

def generate_setup_script():
    """Generate a setup script for the user"""
    setup_script = """#!/bin/bash
# SuperNova Optuna Optimization Setup Script

echo "Setting up SuperNova Optuna Optimization System..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Set up database
echo "Setting up database..."
python -c "
from supernova.db import engine
from supernova.optimization_models import Base
Base.metadata.create_all(engine)
print('Database tables created successfully')
"

echo "Setup complete!"
echo ""
echo "To use the system:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the API server: uvicorn supernova.api:app --reload"
echo "3. Access the optimization endpoints at http://localhost:8000"
"""
    
    with open("setup_optuna.sh", "w") as f:
        f.write(setup_script)
    
    print("✓ Setup script generated: setup_optuna.sh")

def main():
    """Main setup and test function"""
    print("SuperNova Optuna Optimization - Quick Setup & Test")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_status": "UNKNOWN"
    }
    
    # Step 1: Check file structure
    print("\n=== Checking File Structure ===")
    required_files = [
        "supernova/optimizer.py",
        "supernova/optimization_models.py", 
        "supernova/api.py",
        "supernova/workflows.py",
        "requirements.txt"
    ]
    
    files_present = sum(1 for f in required_files if check_file_exists(f))
    results["tests"]["file_structure"] = f"{files_present}/{len(required_files)} files present"
    
    # Step 2: Test basic imports
    basic_imports_ok = test_basic_imports()
    results["tests"]["basic_imports"] = "PASS" if basic_imports_ok else "FAIL"
    
    # Step 3: Test optimization imports  
    opt_imports_ok = test_optimization_imports()
    results["tests"]["optimization_imports"] = "PASS" if opt_imports_ok else "FAIL"
    
    # Step 4: Test database creation
    if opt_imports_ok:
        db_ok = test_database_creation()
        results["tests"]["database_setup"] = "PASS" if db_ok else "FAIL"
    else:
        print("\n⚠️ Skipping database test due to import failures")
        results["tests"]["database_setup"] = "SKIPPED"
        db_ok = False
    
    # Step 5: Test optimizer basics
    if opt_imports_ok:
        optimizer_ok = test_optimizer_basic()
        results["tests"]["optimizer_basic"] = "PASS" if optimizer_ok else "FAIL"
    else:
        print("\n⚠️ Skipping optimizer test due to import failures")
        results["tests"]["optimizer_basic"] = "SKIPPED"
        optimizer_ok = False
    
    # Generate setup script
    print("\n=== Generating Setup Resources ===")
    generate_setup_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for result in results["tests"].values() if result == "PASS")
    total_tests = len([r for r in results["tests"].values() if r != "SKIPPED"])
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in results["tests"].items():
        status_symbol = {"PASS": "✓", "FAIL": "✗", "SKIPPED": "-"}.get(result, "?")
        print(f"{status_symbol} {test_name}: {result}")
    
    # Determine overall status
    if passed_tests == total_tests and files_present == len(required_files):
        results["overall_status"] = "READY"
        print("\n🎉 System Status: READY FOR USE!")
        print("\nNext steps:")
        print("1. Run: chmod +x setup_optuna.sh && ./setup_optuna.sh")
        print("2. Start API: uvicorn supernova.api:app --reload")
        print("3. Test optimization endpoints")
        
    elif passed_tests >= total_tests * 0.6:
        results["overall_status"] = "NEEDS_SETUP"
        print("\n⚠️ System Status: NEEDS DEPENDENCY INSTALLATION")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run this script again to verify setup")
        
    else:
        results["overall_status"] = "ISSUES_FOUND" 
        print("\n❌ System Status: MAJOR ISSUES FOUND")
        print("\nIssues to resolve:")
        print("- Missing core files or major import errors")
        print("- Check file structure and dependencies")
    
    # Save results
    import json
    with open("optuna_setup_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: optuna_setup_results.json")
    
    return results["overall_status"] == "READY"

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        sys.exit(1)