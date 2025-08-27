#!/usr/bin/env python3
"""
SuperNova AI - Final Production Readiness Checklist
Comprehensive validation for production deployment
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ProductionReadinessChecker:
    """Final production readiness validation for SuperNova AI"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checklist_id": f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "categories": {},
            "overall_status": "PENDING",
            "readiness_score": 0,
            "critical_issues": [],
            "recommendations": [],
            "deployment_approved": False
        }
        self.base_path = Path(__file__).parent
        self.critical_failures = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def check_category(self, category_name: str, checks: List[Dict]) -> Dict[str, Any]:
        """Execute a category of checks"""
        category_result = {
            "status": "PENDING",
            "checks": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "total": len(checks)
        }
        
        print(f"\\nChecking {category_name}:")
        print("-" * 40)
        
        for check in checks:
            check_name = check["name"]
            check_func = check["function"]
            is_critical = check.get("critical", False)
            
            try:
                result = check_func()
                status = result.get("status", "FAILED")
                details = result.get("details", {})
                error = result.get("error")
                
                if status == "PASSED":
                    print(f"  PASS: {check_name}")
                    category_result["passed"] += 1
                    self.passed_checks += 1
                elif status == "WARNING":
                    print(f"  WARN: {check_name} - {error or 'Warning detected'}")
                    category_result["warnings"] += 1
                    if not is_critical:
                        self.passed_checks += 1
                else:
                    print(f"  FAIL: {check_name} - {error or 'Check failed'}")
                    category_result["failed"] += 1
                    if is_critical:
                        self.critical_failures.append(f"{category_name}: {check_name}")
                
                category_result["checks"][check_name] = {
                    "status": status,
                    "details": details,
                    "error": error,
                    "critical": is_critical
                }
                
                self.total_checks += 1
                
            except Exception as e:
                print(f"  FAIL: {check_name} - Exception: {str(e)}")
                category_result["failed"] += 1
                category_result["checks"][check_name] = {
                    "status": "FAILED",
                    "error": str(e),
                    "critical": is_critical
                }
                self.total_checks += 1
                
                if is_critical:
                    self.critical_failures.append(f"{category_name}: {check_name}")
        
        # Determine category status
        if category_result["failed"] == 0:
            if category_result["warnings"] == 0:
                category_result["status"] = "PASSED"
            else:
                category_result["status"] = "WARNING"
        else:
            category_result["status"] = "FAILED"
        
        return category_result
    
    # ============================================================================
    # ARCHITECTURE & DEPENDENCIES CHECKS
    # ============================================================================
    
    def check_core_files_exist(self) -> Dict[str, Any]:
        """Check that all core system files exist"""
        critical_files = [
            "main.py",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            "supernova/__init__.py",
            "supernova/api.py",
            "supernova/db.py",
            "supernova/config.py",
            "frontend/package.json",
            "frontend/src/App.tsx"
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not (self.base_path / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return {
                "status": "FAILED",
                "error": f"Missing critical files: {', '.join(missing_files)}",
                "details": {"missing_count": len(missing_files)}
            }
        
        return {
            "status": "PASSED",
            "details": {"all_critical_files_present": True, "file_count": len(critical_files)}
        }
    
    def check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies"""
        try:
            requirements_file = self.base_path / "requirements.txt"
            if not requirements_file.exists():
                return {"status": "FAILED", "error": "requirements.txt not found"}
            
            with open(requirements_file, 'r') as f:
                requirements = f.read()
                
            # Check for critical dependencies
            critical_deps = ["fastapi", "uvicorn", "sqlalchemy", "pydantic"]
            missing_deps = []
            
            for dep in critical_deps:
                if dep.lower() not in requirements.lower():
                    missing_deps.append(dep)
            
            if missing_deps:
                return {
                    "status": "FAILED",
                    "error": f"Missing critical dependencies: {', '.join(missing_deps)}"
                }
            
            return {
                "status": "PASSED",
                "details": {"requirements_file_exists": True, "critical_deps_present": True}
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def check_frontend_dependencies(self) -> Dict[str, Any]:
        """Check frontend dependencies"""
        try:
            package_json = self.base_path / "frontend" / "package.json"
            if not package_json.exists():
                return {"status": "WARNING", "error": "Frontend package.json not found"}
            
            with open(package_json, 'r') as f:
                package_data = json.load(f)
            
            dependencies = package_data.get("dependencies", {})
            critical_deps = ["react", "@mui/material", "axios"]
            missing_deps = []
            
            for dep in critical_deps:
                if dep not in dependencies:
                    missing_deps.append(dep)
            
            if missing_deps:
                return {
                    "status": "WARNING",
                    "error": f"Missing frontend dependencies: {', '.join(missing_deps)}"
                }
            
            return {
                "status": "PASSED",
                "details": {"package_json_exists": True, "critical_deps_present": True}
            }
            
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    # ============================================================================
    # SECURITY CHECKS
    # ============================================================================
    
    def check_environment_security(self) -> Dict[str, Any]:
        """Check environment security configuration"""
        security_vars = [
            "SECRET_KEY",
            "JWT_SECRET",
            "DATABASE_URL"
        ]
        
        missing_vars = []
        weak_vars = []
        
        for var in security_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif len(value) < 16:  # Check for weak keys
                weak_vars.append(var)
        
        issues = []
        if missing_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
        if weak_vars:
            issues.append(f"Weak security keys: {', '.join(weak_vars)}")
        
        if issues:
            return {
                "status": "FAILED" if missing_vars else "WARNING",
                "error": "; ".join(issues)
            }
        
        return {
            "status": "PASSED",
            "details": {"security_vars_configured": True}
        }
    
    def check_security_headers(self) -> Dict[str, Any]:
        """Check security headers configuration"""
        try:
            nginx_conf = self.base_path / "docker" / "nginx.conf"
            if not nginx_conf.exists():
                return {"status": "WARNING", "error": "Nginx configuration not found"}
            
            with open(nginx_conf, 'r') as f:
                nginx_content = f.read()
            
            security_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "Strict-Transport-Security"
            ]
            
            missing_headers = []
            for header in security_headers:
                if header not in nginx_content:
                    missing_headers.append(header)
            
            if missing_headers:
                return {
                    "status": "WARNING",
                    "error": f"Missing security headers: {', '.join(missing_headers)}"
                }
            
            return {
                "status": "PASSED",
                "details": {"security_headers_configured": True}
            }
            
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    # ============================================================================
    # PERFORMANCE CHECKS
    # ============================================================================
    
    def check_performance_configuration(self) -> Dict[str, Any]:
        """Check performance configuration"""
        try:
            # Check Docker configuration for resource limits
            docker_compose = self.base_path / "docker-compose.prod.yml"
            if docker_compose.exists():
                with open(docker_compose, 'r') as f:
                    compose_content = f.read()
                
                has_resource_limits = "resources:" in compose_content
                has_scaling = "replicas:" in compose_content
                
                return {
                    "status": "PASSED",
                    "details": {
                        "resource_limits_configured": has_resource_limits,
                        "scaling_configured": has_scaling
                    }
                }
            else:
                return {
                    "status": "WARNING",
                    "error": "Production Docker compose file not found"
                }
                
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    def check_caching_configuration(self) -> Dict[str, Any]:
        """Check caching configuration"""
        try:
            cache_manager = self.base_path / "supernova" / "cache_manager.py"
            if not cache_manager.exists():
                return {"status": "WARNING", "error": "Cache manager not found"}
            
            with open(cache_manager, 'r') as f:
                cache_content = f.read()
            
            has_redis = "redis" in cache_content.lower()
            has_caching = "cache" in cache_content.lower()
            
            if has_redis and has_caching:
                return {
                    "status": "PASSED",
                    "details": {"caching_system_implemented": True, "redis_integration": True}
                }
            else:
                return {
                    "status": "WARNING",
                    "error": "Caching system not fully implemented"
                }
                
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    # ============================================================================
    # DATABASE CHECKS
    # ============================================================================
    
    def check_database_migrations(self) -> Dict[str, Any]:
        """Check database migrations"""
        try:
            migrations_dir = self.base_path / "migrations"
            if not migrations_dir.exists():
                return {"status": "FAILED", "error": "Migrations directory not found"}
            
            migration_files = list(migrations_dir.glob("*.sql"))
            migrate_script = migrations_dir / "migrate.py"
            
            if not migration_files:
                return {"status": "FAILED", "error": "No migration files found"}
            
            if not migrate_script.exists():
                return {"status": "WARNING", "error": "Migration script not found"}
            
            return {
                "status": "PASSED",
                "details": {
                    "migration_files_count": len(migration_files),
                    "migration_script_exists": migrate_script.exists()
                }
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def check_database_backup_strategy(self) -> Dict[str, Any]:
        """Check database backup configuration"""
        try:
            # Check for backup scripts in Docker configuration
            docker_compose = self.base_path / "docker-compose.prod.yml"
            if docker_compose.exists():
                with open(docker_compose, 'r') as f:
                    compose_content = f.read()
                
                has_backup_service = "backup:" in compose_content
                has_backup_volumes = "backup" in compose_content
                
                if has_backup_service:
                    return {
                        "status": "PASSED",
                        "details": {"backup_service_configured": True}
                    }
                else:
                    return {
                        "status": "WARNING",
                        "error": "Database backup strategy not configured"
                    }
            else:
                return {"status": "WARNING", "error": "Production configuration not found"}
                
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    # ============================================================================
    # DEPLOYMENT CHECKS
    # ============================================================================
    
    def check_docker_configuration(self) -> Dict[str, Any]:
        """Check Docker configuration"""
        try:
            dockerfile = self.base_path / "Dockerfile"
            docker_compose = self.base_path / "docker-compose.yml"
            docker_compose_prod = self.base_path / "docker-compose.prod.yml"
            
            if not dockerfile.exists():
                return {"status": "FAILED", "error": "Dockerfile not found"}
            
            if not docker_compose.exists():
                return {"status": "FAILED", "error": "docker-compose.yml not found"}
            
            # Check for production configuration
            has_prod_config = docker_compose_prod.exists()
            
            # Check Dockerfile for security best practices
            with open(dockerfile, 'r') as f:
                dockerfile_content = f.read()
            
            has_non_root_user = "USER " in dockerfile_content and "root" not in dockerfile_content.split("USER")[-1]
            has_healthcheck = "HEALTHCHECK" in dockerfile_content
            
            return {
                "status": "PASSED",
                "details": {
                    "dockerfile_exists": True,
                    "docker_compose_exists": True,
                    "production_config_exists": has_prod_config,
                    "non_root_user_configured": has_non_root_user,
                    "healthcheck_configured": has_healthcheck
                }
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def check_deployment_scripts(self) -> Dict[str, Any]:
        """Check deployment automation"""
        try:
            deploy_script = self.base_path / "deploy.sh"
            if not deploy_script.exists():
                return {"status": "WARNING", "error": "Deployment script not found"}
            
            with open(deploy_script, 'r') as f:
                deploy_content = f.read()
            
            has_backup = "backup" in deploy_content.lower()
            has_health_check = "health" in deploy_content.lower()
            has_rollback = "rollback" in deploy_content.lower()
            
            return {
                "status": "PASSED",
                "details": {
                    "deployment_script_exists": True,
                    "backup_functionality": has_backup,
                    "health_check_functionality": has_health_check,
                    "rollback_functionality": has_rollback
                }
            }
            
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    # ============================================================================
    # MONITORING CHECKS
    # ============================================================================
    
    def check_monitoring_configuration(self) -> Dict[str, Any]:
        """Check monitoring and observability"""
        try:
            # Check for monitoring services in Docker compose
            docker_compose_prod = self.base_path / "docker-compose.prod.yml"
            if not docker_compose_prod.exists():
                return {"status": "WARNING", "error": "Production configuration not found"}
            
            with open(docker_compose_prod, 'r') as f:
                compose_content = f.read()
            
            has_prometheus = "prometheus:" in compose_content
            has_grafana = "grafana:" in compose_content
            has_logging = "elasticsearch:" in compose_content or "logstash:" in compose_content
            
            monitoring_services = sum([has_prometheus, has_grafana, has_logging])
            
            if monitoring_services >= 2:
                return {
                    "status": "PASSED",
                    "details": {
                        "prometheus_configured": has_prometheus,
                        "grafana_configured": has_grafana,
                        "logging_stack_configured": has_logging
                    }
                }
            else:
                return {
                    "status": "WARNING",
                    "error": "Insufficient monitoring services configured"
                }
                
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    def check_logging_configuration(self) -> Dict[str, Any]:
        """Check logging configuration"""
        try:
            # Check for logging configuration in the application
            config_file = self.base_path / "supernova" / "config.py"
            if not config_file.exists():
                return {"status": "WARNING", "error": "Config file not found"}
            
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            has_logging_config = "logging" in config_content.lower()
            has_log_level = "log_level" in config_content.lower()
            
            return {
                "status": "PASSED" if has_logging_config else "WARNING",
                "details": {
                    "logging_configured": has_logging_config,
                    "log_level_configured": has_log_level
                },
                "error": None if has_logging_config else "Logging not properly configured"
            }
            
        except Exception as e:
            return {"status": "WARNING", "error": str(e)}
    
    def run_production_readiness_check(self) -> Dict[str, Any]:
        """Run complete production readiness checklist"""
        
        print("SuperNova AI - Production Readiness Checklist")
        print("=" * 70)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Checklist ID: {self.results['checklist_id']}")
        
        # Define checklist categories
        checklist_categories = {
            "Architecture & Dependencies": [
                {"name": "Core Files Present", "function": self.check_core_files_exist, "critical": True},
                {"name": "Python Dependencies", "function": self.check_python_dependencies, "critical": True},
                {"name": "Frontend Dependencies", "function": self.check_frontend_dependencies, "critical": False}
            ],
            "Security Configuration": [
                {"name": "Environment Security", "function": self.check_environment_security, "critical": True},
                {"name": "Security Headers", "function": self.check_security_headers, "critical": False}
            ],
            "Performance Configuration": [
                {"name": "Performance Settings", "function": self.check_performance_configuration, "critical": False},
                {"name": "Caching Configuration", "function": self.check_caching_configuration, "critical": False}
            ],
            "Database Configuration": [
                {"name": "Database Migrations", "function": self.check_database_migrations, "critical": True},
                {"name": "Backup Strategy", "function": self.check_database_backup_strategy, "critical": False}
            ],
            "Deployment Configuration": [
                {"name": "Docker Configuration", "function": self.check_docker_configuration, "critical": True},
                {"name": "Deployment Scripts", "function": self.check_deployment_scripts, "critical": False}
            ],
            "Monitoring & Observability": [
                {"name": "Monitoring Services", "function": self.check_monitoring_configuration, "critical": False},
                {"name": "Logging Configuration", "function": self.check_logging_configuration, "critical": False}
            ]
        }
        
        # Run all categories
        for category_name, checks in checklist_categories.items():
            category_result = self.check_category(category_name, checks)
            self.results["categories"][category_name] = category_result
        
        # Calculate overall readiness score
        if self.total_checks > 0:
            self.results["readiness_score"] = (self.passed_checks / self.total_checks) * 100
        
        # Determine overall status
        if not self.critical_failures:
            if self.results["readiness_score"] >= 90:
                self.results["overall_status"] = "PRODUCTION READY"
                self.results["deployment_approved"] = True
            elif self.results["readiness_score"] >= 75:
                self.results["overall_status"] = "READY WITH MINOR ISSUES"
                self.results["deployment_approved"] = True
            else:
                self.results["overall_status"] = "NEEDS IMPROVEMENTS"
        else:
            self.results["overall_status"] = "CRITICAL ISSUES DETECTED"
            self.results["critical_issues"] = self.critical_failures
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.results
    
    def generate_recommendations(self):
        """Generate deployment recommendations"""
        recommendations = []
        
        if self.results["readiness_score"] < 90:
            recommendations.append("Address failed checks to improve production readiness")
        
        if self.critical_failures:
            recommendations.append("CRITICAL: Fix all critical issues before deployment")
        
        # Category-specific recommendations
        for category_name, category_result in self.results["categories"].items():
            if category_result["status"] == "FAILED":
                recommendations.append(f"Address issues in {category_name}")
            elif category_result["warnings"] > 0:
                recommendations.append(f"Review warnings in {category_name}")
        
        # General production recommendations
        recommendations.extend([
            "Set up monitoring alerts for critical metrics",
            "Configure automated backups with tested restore procedures",
            "Implement health checks for all services",
            "Set up SSL certificates for production domains",
            "Configure rate limiting and DDoS protection",
            "Test disaster recovery procedures",
            "Set up log aggregation and analysis",
            "Configure secrets management for production",
            "Set up automated security scanning",
            "Implement blue-green deployment strategy"
        ])
        
        self.results["recommendations"] = recommendations[:10]  # Top 10 recommendations
    
    def generate_report(self) -> str:
        """Generate production readiness report"""
        results = self.results
        report = []
        
        report.append("=" * 80)
        report.append("SuperNova AI - Production Readiness Assessment Report")
        report.append("=" * 80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Assessment ID: {results['checklist_id']}")
        report.append("")
        
        # Overall Status
        report.append(f"OVERALL STATUS: {results['overall_status']}")
        report.append(f"READINESS SCORE: {results['readiness_score']:.1f}%")
        report.append(f"DEPLOYMENT APPROVED: {'YES' if results['deployment_approved'] else 'NO'}")
        report.append("")
        
        # Critical Issues
        if results.get("critical_issues"):
            report.append("CRITICAL ISSUES (MUST FIX BEFORE DEPLOYMENT):")
            for issue in results["critical_issues"]:
                report.append(f"  FAILED: {issue}")
            report.append("")
        
        # Category Results
        for category_name, category_result in results["categories"].items():
            report.append(f"{category_name.upper()}:")
            report.append(f"  Status: {category_result['status']}")
            report.append(f"  Passed: {category_result['passed']}/{category_result['total']}")
            
            # Failed checks
            failed_checks = [name for name, check in category_result["checks"].items() 
                           if check["status"] == "FAILED"]
            if failed_checks:
                report.append(f"  Failed: {', '.join(failed_checks)}")
            
            report.append("")
        
        # Recommendations
        report.append("TOP RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            report.append(f"  {i}. {rec}")
        report.append("")
        
        # Deployment Instructions
        if results["deployment_approved"]:
            report.append("DEPLOYMENT INSTRUCTIONS:")
            report.append("  1. Review and address any warnings")
            report.append("  2. Configure production environment variables")
            report.append("  3. Set up SSL certificates")
            report.append("  4. Run: ./deploy.sh")
            report.append("  5. Verify all services are healthy")
            report.append("  6. Run end-to-end tests in production")
            report.append("  7. Monitor system performance and logs")
        else:
            report.append("DEPLOYMENT BLOCKED:")
            report.append("  Fix all critical issues before attempting deployment")
            report.append("  Re-run this checklist after fixes are applied")
        
        report.append("")
        report.append("=" * 80)
        
        return "\\n".join(report)

def main():
    """Main checklist execution"""
    checker = ProductionReadinessChecker()
    
    # Run checklist
    results = checker.run_production_readiness_check()
    
    # Save results
    results_file = "production_readiness_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report
    report = checker.generate_report()
    report_file = "production_readiness_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\\n" + "=" * 70)
    print("PRODUCTION READINESS ASSESSMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print final status
    overall_status = results["overall_status"]
    readiness_score = results["readiness_score"]
    deployment_approved = results["deployment_approved"]
    
    print(f"\\nFINAL ASSESSMENT:")
    print(f"   Status: {overall_status}")
    print(f"   Readiness Score: {readiness_score:.1f}%")
    print(f"   Deployment Approved: {'YES' if deployment_approved else 'NO'}")
    
    if deployment_approved:
        print("\\nSuperNova AI is PRODUCTION READY for deployment!")
        print("   Follow the deployment instructions in the report.")
    else:
        print("\\nSuperNova AI requires fixes before production deployment.")
        print("   Address critical issues and re-run this checklist.")
    
    return 0 if deployment_approved else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)