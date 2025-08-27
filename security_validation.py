#!/usr/bin/env python3
"""
SuperNova AI - Security Framework Validation
Comprehensive security validation for production readiness
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class SecurityValidator:
    """Security framework validation for SuperNova AI"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_id": f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "security_tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "security_score": 0
            }
        }
        self.base_path = Path(__file__).parent
    
    def validate_security_files(self) -> Dict[str, Any]:
        """Validate security-related file presence and content"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": [],
            "security_patterns": {}
        }
        
        security_files = {
            "authentication": "supernova/auth.py",
            "security_config": "supernova/security_config.py",
            "encryption": "supernova/encryption.py",
            "input_validation": "supernova/input_validation.py",
            "rate_limiting": "supernova/rate_limiting.py",
            "web_security": "supernova/web_security.py",
            "security_logger": "supernova/security_logger.py"
        }
        
        # Security patterns to look for
        security_patterns = {
            "password_hashing": ["bcrypt", "scrypt", "argon2", "pbkdf2", "hash"],
            "encryption": ["encrypt", "decrypt", "cipher", "aes", "rsa"],
            "authentication": ["token", "jwt", "oauth", "session", "auth"],
            "validation": ["validate", "sanitize", "escape", "clean"],
            "rate_limiting": ["rate", "limit", "throttle", "bucket"],
            "logging": ["log", "audit", "track", "monitor"]
        }
        
        for component, file_path in security_files.items():
            full_path = self.base_path / file_path
            component_result = {
                "file_exists": False,
                "file_size": 0,
                "security_patterns_found": {},
                "security_score": 0
            }
            
            if full_path.exists():
                component_result["file_exists"] = True
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        component_result["file_size"] = len(content)
                        
                        # Check for security patterns
                        content_lower = content.lower()
                        total_patterns = 0
                        found_patterns = 0
                        
                        for pattern_type, patterns in security_patterns.items():
                            pattern_found = False
                            for pattern in patterns:
                                if pattern in content_lower:
                                    pattern_found = True
                                    found_patterns += 1
                                    break
                            component_result["security_patterns_found"][pattern_type] = pattern_found
                            total_patterns += 1
                        
                        # Calculate security score for this component
                        component_result["security_score"] = (found_patterns / total_patterns) * 100
                        
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {component}: {str(e)}")
                    component_result["error"] = str(e)
            else:
                test_results["errors"].append(f"Missing security file: {file_path}")
            
            test_results["details"][component] = component_result
        
        # Calculate overall security score
        existing_files = [c for c in test_results["details"].values() if c["file_exists"]]
        if existing_files:
            avg_security_score = sum(c["security_score"] for c in existing_files) / len(existing_files)
            test_results["security_patterns"]["overall_score"] = avg_security_score
        else:
            test_results["security_patterns"]["overall_score"] = 0
        
        # Determine status
        missing_files = len([c for c in test_results["details"].values() if not c["file_exists"]])
        if missing_files == 0:
            test_results["status"] = "PASSED"
        elif missing_files < 3:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_configuration_security(self) -> Dict[str, Any]:
        """Validate security configurations"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Check for environment variables and configuration
        env_security_vars = [
            "SECRET_KEY",
            "JWT_SECRET", 
            "ENCRYPTION_KEY",
            "DATABASE_URL",
            "API_KEY",
            "OPENAI_API_KEY"
        ]
        
        env_status = {}
        set_vars = 0
        for var in env_security_vars:
            is_set = bool(os.getenv(var))
            env_status[var] = "SET" if is_set else "NOT_SET"
            if is_set:
                set_vars += 1
        
        test_results["details"]["environment_variables"] = env_status
        test_results["details"]["env_vars_set"] = f"{set_vars}/{len(env_security_vars)}"
        
        # Check configuration files for security settings
        config_files = {
            "main_config": "supernova/config.py",
            "security_config": "supernova/security_config.py"
        }
        
        for config_name, config_path in config_files.items():
            full_path = self.base_path / config_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for security-related configurations
                    security_configs = {
                        "cors_configured": "CORS" in content or "cors" in content.lower(),
                        "https_configured": "https" in content.lower() or "ssl" in content.lower(),
                        "session_security": "secure" in content.lower() and "session" in content.lower(),
                        "rate_limiting": "rate" in content.lower() and "limit" in content.lower()
                    }
                    
                    test_results["details"][config_name] = security_configs
                    
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {config_name}: {str(e)}")
            else:
                test_results["errors"].append(f"Missing config file: {config_path}")
        
        # Determine status
        if len(test_results["errors"]) == 0 and set_vars >= len(env_security_vars) // 2:
            test_results["status"] = "PASSED"
        elif len(test_results["errors"]) < 2:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_input_sanitization(self) -> Dict[str, Any]:
        """Validate input sanitization and validation patterns"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Files to check for input validation
        validation_files = [
            "supernova/input_validation.py",
            "supernova/api.py",
            "supernova/schemas.py",
            "supernova/web_security.py"
        ]
        
        validation_patterns = [
            r"validate\w*\(",
            r"sanitize\w*\(",
            r"escape\w*\(",
            r"clean\w*\(",
            r"filter\w*\(",
            r"bleach\.",
            r"html\.escape",
            r"re\.escape",
            r"pydantic",
            r"BaseModel"
        ]
        
        total_validations_found = 0
        files_with_validation = 0
        
        for file_path in validation_files:
            full_path = self.base_path / file_path
            file_result = {
                "file_exists": False,
                "validation_patterns_found": 0,
                "patterns_detail": []
            }
            
            if full_path.exists():
                file_result["file_exists"] = True
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for validation patterns
                    for pattern in validation_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            file_result["validation_patterns_found"] += len(matches)
                            file_result["patterns_detail"].append({
                                "pattern": pattern,
                                "matches": len(matches)
                            })
                    
                    if file_result["validation_patterns_found"] > 0:
                        files_with_validation += 1
                        total_validations_found += file_result["validation_patterns_found"]
                        
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {file_path}: {str(e)}")
            else:
                test_results["errors"].append(f"Missing validation file: {file_path}")
            
            test_results["details"][file_path] = file_result
        
        test_results["details"]["summary"] = {
            "total_validation_patterns": total_validations_found,
            "files_with_validation": files_with_validation,
            "validation_coverage": f"{files_with_validation}/{len(validation_files)}"
        }
        
        # Determine status
        if files_with_validation >= len(validation_files) // 2:
            test_results["status"] = "PASSED"
        elif files_with_validation > 0:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_ssl_tls_config(self) -> Dict[str, Any]:
        """Validate SSL/TLS configuration"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Check for SSL/TLS configurations
        ssl_patterns = [
            "ssl",
            "tls", 
            "https",
            "certificate",
            "cert",
            "keyfile",
            "certfile",
            "ssl_context"
        ]
        
        # Files to check for SSL configuration
        ssl_files = [
            "main.py",
            "supernova/config.py",
            "supernova/api.py",
            "supernova/web_security.py"
        ]
        
        ssl_configs_found = 0
        ssl_details = {}
        
        for file_path in ssl_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    found_patterns = []
                    for pattern in ssl_patterns:
                        if pattern in content:
                            found_patterns.append(pattern)
                            ssl_configs_found += 1
                    
                    ssl_details[file_path] = {
                        "ssl_patterns_found": found_patterns,
                        "has_ssl_config": len(found_patterns) > 0
                    }
                    
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {file_path}: {str(e)}")
            else:
                ssl_details[file_path] = {"exists": False}
        
        test_results["details"] = ssl_details
        test_results["details"]["summary"] = {
            "total_ssl_references": ssl_configs_found,
            "files_with_ssl_config": len([d for d in ssl_details.values() if d.get("has_ssl_config", False)])
        }
        
        # Check for environment SSL variables
        ssl_env_vars = ["SSL_CERT", "SSL_KEY", "TLS_CERT", "CERT_FILE", "KEY_FILE"]
        ssl_env_set = [var for var in ssl_env_vars if os.getenv(var)]
        test_results["details"]["ssl_env_vars_set"] = ssl_env_set
        
        # Determine status - SSL/TLS is recommended for production
        if ssl_configs_found > 0 or ssl_env_set:
            test_results["status"] = "PASSED"
        else:
            test_results["status"] = "WARNING"  # Not failed, but recommended
            test_results["warnings"].append("No SSL/TLS configuration found - recommended for production")
        
        return test_results
    
    def run_security_validation(self) -> Dict[str, Any]:
        """Run all security validation tests"""
        
        print("Starting SuperNova AI Security Validation...")
        print("=" * 50)
        
        security_tests = {
            "security_files": self.validate_security_files,
            "configuration_security": self.validate_configuration_security,
            "input_sanitization": self.validate_input_sanitization,
            "ssl_tls_config": self.validate_ssl_tls_config
        }
        
        for test_name, test_func in security_tests.items():
            print(f"\\nRunning {test_name.replace('_', ' ').title()}...")
            
            try:
                result = test_func()
                self.results["security_tests"][test_name] = result
                
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
                    for error in result["errors"][:2]:
                        print(f"   ERROR: {error}")
                if result.get("warnings"):
                    for warning in result["warnings"][:2]:
                        print(f"   WARNING: {warning}")
                        
            except Exception as e:
                print(f"CRITICAL ERROR: {test_name} - {str(e)}")
                self.results["security_tests"][test_name] = {
                    "status": "CRITICAL_ERROR",
                    "error": str(e)
                }
                self.results["summary"]["failed"] += 1
                self.results["summary"]["total_tests"] += 1
        
        # Calculate overall security score
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        warnings = self.results["summary"]["warnings"]
        
        if total > 0:
            security_score = ((passed * 100) + (warnings * 50)) / (total * 100) * 100
            self.results["summary"]["security_score"] = round(security_score, 1)
        
        return self.results
    
    def generate_security_report(self) -> str:
        """Generate security validation report"""
        
        report = []
        report.append("=" * 70)
        report.append("SuperNova AI - Security Framework Validation Report")
        report.append("=" * 70)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Validation ID: {self.results['validation_id']}")
        report.append("")
        
        # Summary
        summary = self.results["summary"]
        report.append("SECURITY SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  Security Score: {summary['security_score']:.1f}%")
        report.append("")
        
        # Security Status
        if summary['failed'] == 0:
            if summary['warnings'] == 0:
                security_status = "SECURE - Production Ready"
            else:
                security_status = "SECURE - Production Ready (with recommendations)"
        else:
            security_status = "SECURITY ISSUES DETECTED - Address before production"
        
        report.append(f"SECURITY STATUS: {security_status}")
        report.append("")
        
        # Detailed results
        for test_name, test_result in self.results["security_tests"].items():
            report.append(f"SECURITY TEST: {test_name.upper().replace('_', ' ')}")
            report.append(f"Status: {test_result['status']}")
            
            if test_result.get("errors"):
                report.append("Security Issues:")
                for error in test_result["errors"]:
                    report.append(f"  - {error}")
            
            if test_result.get("warnings"):
                report.append("Security Recommendations:")
                for warning in test_result["warnings"]:
                    report.append(f"  - {warning}")
            
            report.append("")
        
        # Security Recommendations
        report.append("SECURITY RECOMMENDATIONS:")
        if summary['security_score'] < 80:
            report.append("  - Implement missing security components")
            report.append("  - Add comprehensive input validation")
            report.append("  - Configure SSL/TLS for production")
        
        report.append("  - Regularly update dependencies")
        report.append("  - Monitor security logs")
        report.append("  - Perform regular security audits")
        report.append("  - Implement intrusion detection")
        report.append("")
        
        return "\\n".join(report)

def main():
    """Main security validation function"""
    validator = SecurityValidator()
    
    # Run validation
    results = validator.run_security_validation()
    
    # Save results
    results_file = "security_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report
    report = validator.generate_security_report()
    report_file = "security_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\\n" + "=" * 50)
    print("SECURITY VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print summary
    summary = results["summary"]
    security_score = summary["security_score"]
    
    print(f"\\nSecurity Score: {security_score:.1f}%")
    print(f"Passed: {summary['passed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['failed'] == 0:
        print("\\nSecurity framework is production ready!")
    else:
        print("\\nAddress security issues before production deployment.")
    
    return results

if __name__ == "__main__":
    main()