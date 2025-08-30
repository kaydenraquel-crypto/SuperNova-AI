#!/usr/bin/env python3
"""
Security configuration validation script for SuperNova AI
Validates that all security-critical settings are properly configured
"""

import os
import sys
import secrets
import re
from typing import List, Tuple

def validate_environment_variables() -> List[Tuple[str, str]]:
    """Validate required environment variables are set and secure."""
    errors = []
    
    # Required variables
    required_vars = [
        'SECRET_KEY',
        'JWT_SECRET', 
        'TIMESCALE_PASSWORD',
        'TIMESCALE_HOST',
        'TIMESCALE_USER',
        'TIMESCALE_DB'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            errors.append((var, f"Environment variable {var} is required but not set"))
            continue
            
        # Security validations
        if var in ['SECRET_KEY', 'JWT_SECRET']:
            if len(value) < 32:
                errors.append((var, f"{var} must be at least 32 characters long"))
            if value in ['your-secret-key-change-in-production', 'your-jwt-secret-change-in-production']:
                errors.append((var, f"{var} is using default/placeholder value - must be changed"))
                
        if var == 'TIMESCALE_PASSWORD':
            if len(value) < 12:
                errors.append((var, "TIMESCALE_PASSWORD must be at least 12 characters long"))
            if value in ['supernova123', 'password', '123456']:
                errors.append((var, "TIMESCALE_PASSWORD is using weak/default value"))
    
    return errors

def validate_ssl_configuration() -> List[Tuple[str, str]]:
    """Validate SSL/TLS configuration."""
    errors = []
    
    ssl_required = os.getenv('SSL_REQUIRED', 'false').lower() == 'true'
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production' and not ssl_required:
        errors.append(('SSL_REQUIRED', 'SSL must be required in production environment'))
    
    return errors

def check_hardcoded_secrets() -> List[Tuple[str, str]]:
    """Check for hardcoded secrets in configuration files."""
    errors = []
    
    # Check docker-compose files
    compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
    
    for file_path in compose_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for default secret patterns
            dangerous_patterns = [
                r'your-secret-key-change-in-production',
                r'your-jwt-secret-change-in-production', 
                r'supernova123',
                r'admin123',
                r'password123'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content):
                    errors.append((file_path, f"Found hardcoded secret pattern: {pattern}"))
    
    return errors

def main():
    """Run all security validations."""
    print("SuperNova AI Security Configuration Validator")
    print("=" * 50)
    
    all_errors = []
    
    # Run validations
    print("Checking environment variables...")
    all_errors.extend(validate_environment_variables())
    
    print("Checking SSL configuration...")
    all_errors.extend(validate_ssl_configuration())
    
    print("Checking for hardcoded secrets...")
    all_errors.extend(check_hardcoded_secrets())
    
    # Report results
    if all_errors:
        print(f"\n❌ SECURITY VALIDATION FAILED: {len(all_errors)} issues found")
        print("\nSecurity Issues:")
        for component, error in all_errors:
            print(f"  - {component}: {error}")
        print("\nPlease fix these issues before deploying to production!")
        sys.exit(1)
    else:
        print("\n✅ SECURITY VALIDATION PASSED: No critical issues found")
        print("Configuration appears secure for production deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()