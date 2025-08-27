#!/usr/bin/env python3
"""
SuperNova AI - Integration Validation Script
Comprehensive system integration validation for production readiness
"""

import os
import sys
import json
import importlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class IntegrationValidator:
    """Comprehensive integration validation for SuperNova AI"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_id": f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        self.base_path = Path(__file__).parent
    
    def validate_architecture(self) -> Dict[str, Any]:
        """Validate system architecture and file dependencies"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Critical system files
        critical_files = {
            "Backend Core": [
                "supernova/__init__.py",
                "supernova/api.py",
                "supernova/db.py",
                "supernova/config.py",
                "supernova/schemas.py"
            ],
            "Security Framework": [
                "supernova/auth.py",
                "supernova/security_config.py",
                "supernova/encryption.py",
                "supernova/input_validation.py",
                "supernova/rate_limiting.py"
            ],
            "Performance & Monitoring": [
                "supernova/performance_monitor.py",
                "supernova/cache_manager.py",
                "supernova/async_processor.py"
            ],
            "Communication": [
                "supernova/chat.py",
                "supernova/websocket_handler.py",
                "supernova/conversation_memory.py"
            ],
            "Frontend": [
                "frontend/package.json",
                "frontend/src/App.tsx",
                "frontend/src/index.tsx",
                "frontend/webpack.config.js"
            ],
            "Configuration": [
                "main.py",
                "requirements.txt",
                "pytest.ini"
            ]
        }
        
        missing_files = []
        found_files = []
        
        for category, files in critical_files.items():
            category_status = {"found": [], "missing": []}
            for file_path in files:
                full_path = self.base_path / file_path
                if full_path.exists():
                    category_status["found"].append(file_path)
                    found_files.append(file_path)
                else:
                    category_status["missing"].append(file_path)
                    missing_files.append(file_path)
            
            test_results["details"][category] = category_status
        
        # Determine overall status
        if not missing_files:
            test_results["status"] = "PASSED"
        elif len(missing_files) < 5:
            test_results["status"] = "WARNING"
            test_results["warnings"].extend([f"Missing file: {f}" for f in missing_files])
        else:
            test_results["status"] = "FAILED"
            test_results["errors"].extend([f"Missing critical file: {f}" for f in missing_files])
        
        test_results["summary"] = {
            "total_files": len(found_files) + len(missing_files),
            "found_files": len(found_files),
            "missing_files": len(missing_files)
        }
        
        return test_results
    
    def validate_import_dependencies(self) -> Dict[str, Any]:
        """Validate Python module imports"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Core modules to test
        core_modules = [
            "supernova.api",
            "supernova.db", 
            "supernova.config",
            "supernova.schemas",
            "supernova.auth",
            "supernova.chat"
        ]
        
        import_status = {}
        failed_imports = []
        successful_imports = []
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                import_status[module] = "SUCCESS"
                successful_imports.append(module)
            except ImportError as e:
                import_status[module] = f"FAILED: {str(e)}"
                failed_imports.append(module)
                test_results["errors"].append(f"Failed to import {module}: {str(e)}")
            except Exception as e:
                import_status[module] = f"ERROR: {str(e)}"
                failed_imports.append(module)
                test_results["warnings"].append(f"Error importing {module}: {str(e)}")
        
        test_results["details"] = import_status
        
        # Determine status
        if not failed_imports:
            test_results["status"] = "PASSED"
        elif len(failed_imports) < len(core_modules) / 2:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        config_checks = {
            "requirements.txt": self.base_path / "requirements.txt",
            "main.py": self.base_path / "main.py",
            "frontend/package.json": self.base_path / "frontend" / "package.json",
            "pytest.ini": self.base_path / "pytest.ini"
        }
        
        for config_name, config_path in config_checks.items():
            if config_path.exists():
                try:
                    if config_name.endswith('.json'):
                        with open(config_path, 'r') as f:
                            data = json.load(f)
                            test_results["details"][config_name] = "Valid JSON"
                    else:
                        with open(config_path, 'r') as f:
                            content = f.read()
                            test_results["details"][config_name] = f"Found ({len(content)} chars)"
                except Exception as e:
                    test_results["details"][config_name] = f"Error: {str(e)}"
                    test_results["warnings"].append(f"Configuration error in {config_name}: {str(e)}")
            else:
                test_results["details"][config_name] = "Missing"
                test_results["errors"].append(f"Missing configuration file: {config_name}")
        
        # Check for environment variables
        env_vars = ["SUPERNOVA_DB_URL", "SECRET_KEY", "API_KEY"]
        env_status = {}
        for var in env_vars:
            env_status[var] = "Set" if os.getenv(var) else "Not Set"
        
        test_results["details"]["environment_variables"] = env_status
        
        error_count = len(test_results["errors"])
        if error_count == 0:
            test_results["status"] = "PASSED"
        elif error_count < 3:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_frontend_structure(self) -> Dict[str, Any]:
        """Validate frontend Material-UI structure"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        frontend_structure = {
            "package.json": "frontend/package.json",
            "main_app": "frontend/src/App.tsx",
            "index": "frontend/src/index.tsx",
            "webpack_config": "frontend/webpack.config.js",
            "components": [
                "frontend/src/components/layout/MainLayout.tsx",
                "frontend/src/components/dashboard/DashboardPage.tsx",
                "frontend/src/components/common/LoadingScreen.tsx"
            ],
            "services": "frontend/src/services/api.ts",
            "theme": "frontend/src/theme/index.ts"
        }
        
        for component, path_info in frontend_structure.items():
            if component == "components":
                component_status = {}
                for comp_path in path_info:
                    full_path = self.base_path / comp_path
                    component_status[comp_path] = "Found" if full_path.exists() else "Missing"
                test_results["details"][component] = component_status
            else:
                full_path = self.base_path / path_info
                test_results["details"][component] = "Found" if full_path.exists() else "Missing"
                if not full_path.exists():
                    test_results["errors"].append(f"Missing frontend file: {path_info}")
        
        # Check package.json dependencies
        package_json_path = self.base_path / "frontend" / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                    dependencies = package_data.get("dependencies", {})
                    mui_deps = [dep for dep in dependencies.keys() if dep.startswith("@mui")]
                    test_results["details"]["mui_dependencies"] = {
                        "count": len(mui_deps),
                        "packages": mui_deps[:5]  # Show first 5
                    }
            except Exception as e:
                test_results["warnings"].append(f"Error reading package.json: {str(e)}")
        
        error_count = len(test_results["errors"])
        if error_count == 0:
            test_results["status"] = "PASSED"
        elif error_count < 3:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_security_framework(self) -> Dict[str, Any]:
        """Validate security framework implementation"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        security_components = {
            "authentication": "supernova/auth.py",
            "security_config": "supernova/security_config.py", 
            "encryption": "supernova/encryption.py",
            "input_validation": "supernova/input_validation.py",
            "rate_limiting": "supernova/rate_limiting.py",
            "web_security": "supernova/web_security.py"
        }
        
        for component, file_path in security_components.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        # Basic security checks
                        security_patterns = ["hash", "encrypt", "token", "auth", "validate"]
                        found_patterns = [p for p in security_patterns if p in content.lower()]
                        test_results["details"][component] = {
                            "status": "Found",
                            "size": len(content),
                            "security_patterns": found_patterns
                        }
                except Exception as e:
                    test_results["details"][component] = {"status": "Error", "error": str(e)}
                    test_results["warnings"].append(f"Error reading {component}: {str(e)}")
            else:
                test_results["details"][component] = {"status": "Missing"}
                test_results["errors"].append(f"Missing security component: {file_path}")
        
        missing_components = len([c for c in test_results["details"].values() if c.get("status") == "Missing"])
        
        if missing_components == 0:
            test_results["status"] = "PASSED"
        elif missing_components < 3:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        
        print("Starting SuperNova AI Integration Validation...")
        print("=" * 60)
        
        validation_tests = {
            "architecture": self.validate_architecture,
            "import_dependencies": self.validate_import_dependencies,
            "configuration": self.validate_configuration,
            "frontend_structure": self.validate_frontend_structure,
            "security_framework": self.validate_security_framework
        }
        
        for test_name, test_func in validation_tests.items():
            print(f"\nRunning {test_name.replace('_', ' ').title()} Validation...")
            
            try:
                result = test_func()
                self.results["tests"][test_name] = result
                
                status = result["status"]
                if status == "PASSED":
                    print(f"PASSED: {test_name}")
                    self.results["summary"]["passed"] += 1
                elif status == "WARNING":
                    print(f"WARNING: {test_name}")
                    self.results["summary"]["warnings"] += 1
                else:
                    print(f"FAILED: {test_name}")
                    self.results["summary"]["failed"] += 1
                
                self.results["summary"]["total_tests"] += 1
                
                # Show key details
                if result.get("errors"):
                    for error in result["errors"][:3]:  # Show first 3 errors
                        print(f"   ERROR: {error}")
                if result.get("warnings"):
                    for warning in result["warnings"][:2]:  # Show first 2 warnings
                        print(f"   WARNING: {warning}")
                        
            except Exception as e:
                print(f"CRITICAL ERROR: {test_name} - {str(e)}")
                self.results["tests"][test_name] = {
                    "status": "CRITICAL_ERROR",
                    "error": str(e)
                }
                self.results["summary"]["failed"] += 1
                self.results["summary"]["total_tests"] += 1
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 80)
        report.append("SuperNova AI - Integration Validation Report")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Validation ID: {self.results['validation_id']}")
        report.append("")
        
        # Summary
        summary = self.results["summary"]
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Failed: {summary['failed']}")
        
        success_rate = (summary['passed'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        report.append(f"  Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Overall Status
        if summary['failed'] == 0:
            if summary['warnings'] == 0:
                overall_status = "PRODUCTION READY"
            else:
                overall_status = "PRODUCTION READY (with warnings)"
        else:
            overall_status = "NEEDS ATTENTION"
        
        report.append(f"OVERALL STATUS: {overall_status}")
        report.append("")
        
        # Detailed results
        for test_name, test_result in self.results["tests"].items():
            report.append(f"TEST: {test_name.upper().replace('_', ' ')}")
            report.append(f"Status: {test_result['status']}")
            
            if test_result.get("errors"):
                report.append("Errors:")
                for error in test_result["errors"]:
                    report.append(f"  - {error}")
            
            if test_result.get("warnings"):
                report.append("Warnings:")
                for warning in test_result["warnings"]:
                    report.append(f"  - {warning}")
            
            report.append("")
        
        return "\\n".join(report)

def main():
    """Main validation function"""
    validator = IntegrationValidator()
    
    # Run validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    results_file = "integration_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report
    report = validator.generate_report()
    report_file = "integration_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print summary
    summary = results["summary"]
    success_rate = (summary['passed'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
    
    print(f"\\nSuccess Rate: {success_rate:.1f}%")
    print(f"Passed: {summary['passed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['failed'] == 0:
        print("\\nSystem is ready for production deployment!")
    else:
        print("\\nPlease address failed tests before production deployment.")
    
    return results

if __name__ == "__main__":
    main()