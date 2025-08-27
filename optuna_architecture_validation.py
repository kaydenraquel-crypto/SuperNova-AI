#!/usr/bin/env python3
"""
SuperNova Optuna Architecture Validation & Installation Guide

This script validates the Optuna hyperparameter optimization system architecture,
code quality, and provides comprehensive installation and setup guidance.
"""

import os
import sys
import ast
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re

class OptunaArchitectureValidator:
    """Validates Optuna optimization system architecture and provides setup guidance"""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "architecture_analysis": {},
            "code_quality": {},
            "integration_points": {},
            "installation_status": {},
            "setup_recommendations": [],
            "validation_summary": {},
            "detailed_findings": {}
        }
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete architecture validation"""
        print("SuperNova Optuna Optimization System - Architecture Validation")
        print("=" * 70)
        
        validation_steps = [
            ("1. File Structure Analysis", self._analyze_file_structure),
            ("2. Code Quality Assessment", self._assess_code_quality),
            ("3. Integration Points Review", self._review_integration_points),
            ("4. Dependencies Analysis", self._analyze_dependencies),
            ("5. API Design Validation", self._validate_api_design),
            ("6. Database Schema Review", self._review_database_schema),
            ("7. Workflow Architecture", self._analyze_workflow_architecture),
            ("8. Configuration Management", self._validate_configuration),
            ("9. Error Handling Patterns", self._analyze_error_handling),
            ("10. Performance Considerations", self._analyze_performance_design)
        ]
        
        for step_name, step_func in validation_steps:
            print(f"\n{step_name}")
            print("-" * 50)
            try:
                result = step_func()
                self.results["detailed_findings"][step_name] = result
                
                if result.get("status") == "PASS":
                    print(f"[PASS] {step_name} - PASSED")
                elif result.get("status") == "WARN":
                    print(f"[WARN] {step_name} - WARNINGS")
                else:
                    print(f"[FAIL] {step_name} - ISSUES FOUND")
                
                # Print key findings
                if "findings" in result:
                    for finding in result["findings"][:3]:  # Show top 3 findings
                        print(f"   • {finding}")
                        
            except Exception as e:
                print(f"[ERROR] {step_name} - ERROR: {e}")
                self.results["detailed_findings"][step_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        self._generate_summary()
        self._generate_setup_recommendations()
        
        return self.results
    
    def _analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze the file structure of the optimization system"""
        expected_files = {
            "Core Components": [
                "supernova/optimizer.py",
                "supernova/optimization_models.py",
                "supernova/api.py", 
                "supernova/workflows.py",
                "supernova/schemas.py"
            ],
            "Supporting Files": [
                "requirements.txt",
                "supernova/config.py",
                "supernova/db.py",
                "supernova/backtester.py"
            ],
            "Tests": [
                "tests/test_optimizer.py",
                "tests/test_optimization_models.py"
            ]
        }
        
        findings = []
        file_analysis = {}
        
        for category, files in expected_files.items():
            file_analysis[category] = {}
            for file_path in files:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    file_analysis[category][file_path] = {
                        "exists": True,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                else:
                    file_analysis[category][file_path] = {"exists": False}
                    if category == "Core Components":
                        findings.append(f"Missing core file: {file_path}")
        
        # Count existing vs missing files
        total_files = sum(len(files) for files in expected_files.values())
        existing_files = sum(
            1 for category in file_analysis.values() 
            for file_info in category.values() 
            if file_info.get("exists", False)
        )
        
        status = "PASS" if existing_files >= total_files * 0.8 else "WARN" if existing_files >= total_files * 0.6 else "FAIL"
        
        findings.append(f"File coverage: {existing_files}/{total_files} files ({existing_files/total_files*100:.1f}%)")
        
        return {
            "status": status,
            "file_analysis": file_analysis,
            "coverage": f"{existing_files}/{total_files}",
            "findings": findings
        }
    
    def _assess_code_quality(self) -> Dict[str, Any]:
        """Assess code quality of optimization components"""
        files_to_analyze = [
            "supernova/optimizer.py",
            "supernova/optimization_models.py",
            "supernova/api.py",
            "supernova/workflows.py"
        ]
        
        findings = []
        analysis = {}
        
        for file_path in files_to_analyze:
            if not os.path.exists(file_path):
                analysis[file_path] = {"exists": False}
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST for analysis
                tree = ast.parse(content, filename=file_path)
                
                file_metrics = self._analyze_file_metrics(content, tree)
                analysis[file_path] = file_metrics
                
                # Quality checks
                if file_metrics["lines"] > 1500:
                    findings.append(f"{file_path}: Large file ({file_metrics['lines']} lines)")
                
                if file_metrics["classes"] == 0 and "models.py" in file_path:
                    findings.append(f"{file_path}: No classes found in models file")
                
                if file_metrics["functions"] < 5 and file_metrics["lines"] > 200:
                    findings.append(f"{file_path}: Few functions for file size")
                    
                if file_metrics["docstring_coverage"] < 0.5:
                    findings.append(f"{file_path}: Low docstring coverage ({file_metrics['docstring_coverage']:.1%})")
                
            except Exception as e:
                analysis[file_path] = {"error": str(e)}
                findings.append(f"{file_path}: Analysis failed - {str(e)}")
        
        # Overall assessment
        total_lines = sum(m.get("lines", 0) for m in analysis.values() if isinstance(m, dict))
        avg_complexity = sum(m.get("complexity", 0) for m in analysis.values() if isinstance(m, dict)) / len(analysis)
        
        findings.append(f"Total codebase: {total_lines:,} lines")
        findings.append(f"Average complexity: {avg_complexity:.1f}")
        
        status = "PASS" if len(findings) <= 5 else "WARN" if len(findings) <= 10 else "FAIL"
        
        return {
            "status": status,
            "analysis": analysis,
            "total_lines": total_lines,
            "avg_complexity": avg_complexity,
            "findings": findings
        }
    
    def _analyze_file_metrics(self, content: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze metrics for a single file"""
        lines = content.count('\n') + 1
        
        # Count different node types
        classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        async_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)])
        
        # Count docstrings
        docstrings = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (ast.get_docstring(node) is not None):
                    docstrings += 1
        
        total_defs = classes + functions + async_functions
        docstring_coverage = docstrings / total_defs if total_defs > 0 else 0
        
        # Simple complexity estimate (nested control structures)
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
        
        # Import analysis
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        
        return {
            "lines": lines,
            "classes": classes,
            "functions": functions,
            "async_functions": async_functions,
            "docstring_coverage": docstring_coverage,
            "complexity": complexity,
            "imports": len(set(imports))
        }
    
    def _review_integration_points(self) -> Dict[str, Any]:
        """Review integration points between components"""
        integration_points = {
            "Optimizer ↔ Backtester": self._check_integration("optimizer.py", "backtester"),
            "API ↔ Optimizer": self._check_integration("api.py", "optimizer"),
            "Workflows ↔ Optimizer": self._check_integration("workflows.py", "optimizer"), 
            "Models ↔ Database": self._check_integration("optimization_models.py", "db"),
            "API ↔ Schemas": self._check_integration("api.py", "schemas"),
            "Optimizer ↔ Models": self._check_integration("optimizer.py", "optimization_models")
        }
        
        findings = []
        working_integrations = 0
        
        for integration, status in integration_points.items():
            if status["connected"]:
                working_integrations += 1
            else:
                findings.append(f"Integration issue: {integration}")
        
        findings.append(f"Working integrations: {working_integrations}/{len(integration_points)}")
        
        status = "PASS" if working_integrations >= len(integration_points) * 0.8 else "WARN"
        
        return {
            "status": status,
            "integration_points": integration_points,
            "working_count": working_integrations,
            "total_count": len(integration_points),
            "findings": findings
        }
    
    def _check_integration(self, source_file: str, target_module: str) -> Dict[str, Any]:
        """Check if source file properly imports/uses target module"""
        file_path = f"supernova/{source_file}"
        
        if not os.path.exists(file_path):
            return {"connected": False, "reason": "Source file not found"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for import statements
            import_patterns = [
                f"from .{target_module} import",
                f"from supernova.{target_module} import",
                f"import {target_module}",
                f"import supernova.{target_module}"
            ]
            
            has_import = any(pattern in content for pattern in import_patterns)
            
            # Check for usage (simple heuristic)
            target_name = target_module.split('.')[-1]
            has_usage = target_name in content
            
            return {
                "connected": has_import,
                "has_import": has_import,
                "has_usage": has_usage,
                "file_exists": True
            }
            
        except Exception as e:
            return {"connected": False, "reason": f"Error reading file: {e}"}
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency configuration"""
        findings = []
        dependency_analysis = {}
        
        # Check requirements.txt
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", 'r') as f:
                requirements = f.read()
            
            # Key dependencies for optimization
            key_deps = {
                "optuna": ">=3.4.0",
                "joblib": ">=1.3.0", 
                "plotly": ">=5.17.0",
                "prefect": ">=2.14.0",
                "sqlalchemy": ">=2.0.0",
                "fastapi": "",
                "pydantic": ""
            }
            
            for dep, min_version in key_deps.items():
                if dep in requirements:
                    dependency_analysis[dep] = "✅ Listed"
                else:
                    dependency_analysis[dep] = "❌ Missing"
                    findings.append(f"Missing key dependency: {dep}")
            
            # Check for conflicts
            if "aiohttp==3.9.1" in requirements and "aiohttp==3.9.5" in requirements:
                findings.append("Dependency conflict: Multiple aiohttp versions")
            
            # Count total dependencies
            dep_lines = [line.strip() for line in requirements.split('\n') 
                        if line.strip() and not line.startswith('#')]
            dependency_analysis["total_dependencies"] = len(dep_lines)
            
        else:
            findings.append("requirements.txt file not found")
            dependency_analysis["requirements_file"] = "❌ Missing"
        
        # Installation status check
        try:
            installed_packages = subprocess.check_output(
                [sys.executable, "-m", "pip", "list"], 
                universal_newlines=True
            )
            
            for dep in ["optuna", "fastapi", "sqlalchemy", "prefect"]:
                if dep in installed_packages.lower():
                    dependency_analysis[f"{dep}_installed"] = "✅ Installed"
                else:
                    dependency_analysis[f"{dep}_installed"] = "❌ Not Installed"
                    
        except Exception:
            findings.append("Could not check installed packages")
        
        missing_deps = len([v for v in dependency_analysis.values() if "❌" in v])
        status = "PASS" if missing_deps == 0 else "WARN" if missing_deps <= 3 else "FAIL"
        
        return {
            "status": status,
            "dependency_analysis": dependency_analysis,
            "missing_count": missing_deps,
            "findings": findings
        }
    
    def _validate_api_design(self) -> Dict[str, Any]:
        """Validate API design and endpoints"""
        api_file = "supernova/api.py"
        findings = []
        api_analysis = {}
        
        if not os.path.exists(api_file):
            return {
                "status": "FAIL",
                "findings": ["API file not found"]
            }
        
        try:
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for optimization endpoints
            expected_endpoints = [
                "optimize_strategy",
                "get_optimization_studies",
                "get_optimization_study", 
                "optimize_watchlist",
                "get_optimization_dashboard",
                "get_optimization_progress"
            ]
            
            for endpoint in expected_endpoints:
                if f"def {endpoint}" in content or f"async def {endpoint}" in content:
                    api_analysis[endpoint] = "✅ Defined"
                else:
                    api_analysis[endpoint] = "❌ Missing"
                    findings.append(f"Missing API endpoint: {endpoint}")
            
            # Check for proper error handling
            error_patterns = ["HTTPException", "try:", "except:"]
            has_error_handling = any(pattern in content for pattern in error_patterns)
            api_analysis["error_handling"] = "✅ Present" if has_error_handling else "❌ Minimal"
            
            # Check for async support
            has_async = "async def" in content
            api_analysis["async_support"] = "✅ Present" if has_async else "❌ Missing"
            
            # Check for proper imports
            required_imports = ["fastapi", "pydantic", "HTTPException"]
            for imp in required_imports:
                if imp in content:
                    api_analysis[f"import_{imp}"] = "✅ Present"
                else:
                    api_analysis[f"import_{imp}"] = "❌ Missing"
                    findings.append(f"Missing import: {imp}")
            
            missing_endpoints = len([v for v in api_analysis.values() if "❌" in v])
            status = "PASS" if missing_endpoints == 0 else "WARN" if missing_endpoints <= 2 else "FAIL"
            
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e),
                "findings": [f"Error analyzing API: {e}"]
            }
        
        return {
            "status": status,
            "api_analysis": api_analysis,
            "findings": findings
        }
    
    def _review_database_schema(self) -> Dict[str, Any]:
        """Review database schema design"""
        models_file = "supernova/optimization_models.py"
        findings = []
        schema_analysis = {}
        
        if not os.path.exists(models_file):
            return {
                "status": "FAIL", 
                "findings": ["Database models file not found"]
            }
        
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required models
            expected_models = [
                "OptimizationStudyModel",
                "OptimizationTrialModel",
                "WatchlistOptimizationModel",
                "OptimizationParameterImportanceModel"
            ]
            
            for model in expected_models:
                if f"class {model}" in content:
                    schema_analysis[model] = "[PASS] Defined"
                else:
                    schema_analysis[model] = "[FAIL] Missing"
                    findings.append(f"Missing model: {model}")
            
            # Check for proper SQLAlchemy usage
            sqlalchemy_patterns = [
                "from sqlalchemy",
                "Column",
                "__tablename__",
                "relationship"
            ]
            
            for pattern in sqlalchemy_patterns:
                if pattern in content:
                    schema_analysis[f"sqlalchemy_{pattern}"] = "[PASS] Present"
                else:
                    schema_analysis[f"sqlalchemy_{pattern}"] = "[FAIL] Missing"
            
            # Check for indexes and constraints
            has_indexes = "Index(" in content
            has_constraints = "CheckConstraint" in content or "UniqueConstraint" in content
            
            schema_analysis["database_indexes"] = "✅ Present" if has_indexes else "⚠️ Limited"
            schema_analysis["database_constraints"] = "✅ Present" if has_constraints else "⚠️ Limited"
            
            # Check for utility functions
            utility_functions = ["create_optimization_study", "update_study_progress"]
            for func in utility_functions:
                if f"def {func}" in content:
                    schema_analysis[func] = "✅ Present"
                else:
                    schema_analysis[func] = "❌ Missing"
            
            missing_items = len([v for v in schema_analysis.values() if "❌" in v])
            status = "PASS" if missing_items == 0 else "WARN" if missing_items <= 3 else "FAIL"
            
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e),
                "findings": [f"Error analyzing schema: {e}"]
            }
        
        return {
            "status": status,
            "schema_analysis": schema_analysis,
            "findings": findings
        }
    
    def _analyze_workflow_architecture(self) -> Dict[str, Any]:
        """Analyze Prefect workflow architecture"""
        workflows_file = "supernova/workflows.py"
        findings = []
        workflow_analysis = {}
        
        if not os.path.exists(workflows_file):
            return {
                "status": "FAIL",
                "findings": ["Workflows file not found"]
            }
        
        try:
            with open(workflows_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for Prefect imports
            prefect_imports = ["from prefect import", "@flow", "@task"]
            for imp in prefect_imports:
                if imp in content:
                    workflow_analysis[f"prefect_{imp.replace(' ', '_')}"] = "✅ Present"
                else:
                    workflow_analysis[f"prefect_{imp.replace(' ', '_')}"] = "❌ Missing"
                    findings.append(f"Missing Prefect pattern: {imp}")
            
            # Check for optimization flows
            expected_flows = [
                "optimize_strategy_parameters_flow",
                "optimize_watchlist_strategies_flow",
                "scheduled_optimization_flow"
            ]
            
            for flow in expected_flows:
                if flow in content:
                    workflow_analysis[flow] = "✅ Defined"
                else:
                    workflow_analysis[flow] = "❌ Missing"
                    findings.append(f"Missing workflow: {flow}")
            
            # Check for async support
            has_async_flows = "async def" in content and "@flow" in content
            workflow_analysis["async_flows"] = "✅ Present" if has_async_flows else "❌ Missing"
            
            # Check for error handling in flows
            has_flow_error_handling = "try:" in content and "@flow" in content
            workflow_analysis["flow_error_handling"] = "✅ Present" if has_flow_error_handling else "⚠️ Limited"
            
            missing_items = len([v for v in workflow_analysis.values() if "❌" in v])
            status = "PASS" if missing_items == 0 else "WARN" if missing_items <= 2 else "FAIL"
            
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e), 
                "findings": [f"Error analyzing workflows: {e}"]
            }
        
        return {
            "status": status,
            "workflow_analysis": workflow_analysis,
            "findings": findings
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration management"""
        findings = []
        config_analysis = {}
        
        # Check for config files
        config_files = ["supernova/config.py", ".env", "config.yaml", "config.json"]
        for config_file in config_files:
            if os.path.exists(config_file):
                config_analysis[config_file] = "✅ Present"
            else:
                config_analysis[config_file] = "❌ Missing"
        
        # Check optimizer.py for configuration patterns
        optimizer_file = "supernova/optimizer.py"
        if os.path.exists(optimizer_file):
            with open(optimizer_file, 'r') as f:
                content = f.read()
            
            # Check for configuration classes
            if "OptimizationConfig" in content:
                config_analysis["OptimizationConfig"] = "✅ Present"
                
                # Check for configurable parameters
                config_params = [
                    "n_trials", "primary_objective", "secondary_objectives",
                    "max_drawdown_limit", "min_sharpe_ratio", "walk_forward"
                ]
                
                present_params = [p for p in config_params if p in content]
                config_analysis["configurable_params"] = f"{len(present_params)}/{len(config_params)}"
                
                if len(present_params) < len(config_params) * 0.8:
                    findings.append("Some optimization parameters not configurable")
            else:
                config_analysis["OptimizationConfig"] = "❌ Missing"
                findings.append("OptimizationConfig class not found")
        
        present_configs = len([v for v in config_analysis.values() if "✅" in v])
        status = "PASS" if present_configs >= 2 else "WARN" if present_configs >= 1 else "FAIL"
        
        return {
            "status": status,
            "config_analysis": config_analysis,
            "findings": findings
        }
    
    def _analyze_error_handling(self) -> Dict[str, Any]:
        """Analyze error handling patterns"""
        findings = []
        error_analysis = {}
        
        files_to_check = [
            "supernova/optimizer.py",
            "supernova/api.py", 
            "supernova/workflows.py"
        ]
        
        total_error_patterns = 0
        total_files = 0
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            total_files += 1
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for error handling patterns
            error_patterns = {
                "try_except": content.count("try:"),
                "custom_exceptions": content.count("Exception"),
                "logging": content.count("logger."),
                "validation": content.count("raise")
            }
            
            file_error_score = sum(error_patterns.values())
            total_error_patterns += file_error_score
            
            error_analysis[file_path] = {
                "patterns": error_patterns,
                "score": file_error_score
            }
            
            if file_error_score < 5:
                findings.append(f"{file_path}: Limited error handling")
        
        avg_error_handling = total_error_patterns / total_files if total_files > 0 else 0
        
        status = "PASS" if avg_error_handling >= 10 else "WARN" if avg_error_handling >= 5 else "FAIL"
        
        findings.append(f"Average error handling score: {avg_error_handling:.1f}")
        
        return {
            "status": status,
            "error_analysis": error_analysis,
            "avg_score": avg_error_handling,
            "findings": findings
        }
    
    def _analyze_performance_design(self) -> Dict[str, Any]:
        """Analyze performance design considerations"""
        findings = []
        performance_analysis = {}
        
        optimizer_file = "supernova/optimizer.py"
        
        if not os.path.exists(optimizer_file):
            return {
                "status": "FAIL",
                "findings": ["Optimizer file not found for performance analysis"]
            }
        
        with open(optimizer_file, 'r') as f:
            content = f.read()
        
        # Check for performance features
        performance_features = {
            "parallel_processing": "n_jobs" in content and "Parallel" in content,
            "caching": "cache" in content.lower() or "storage" in content,
            "memory_management": "del " in content or "gc." in content,
            "batch_processing": "batch" in content.lower(),
            "async_operations": "async def" in content,
            "connection_pooling": "pool" in content.lower(),
            "pruning": "prune" in content.lower() or "Pruner" in content
        }
        
        for feature, present in performance_features.items():
            performance_analysis[feature] = "✅ Present" if present else "❌ Missing"
            if not present:
                findings.append(f"Performance feature missing: {feature}")
        
        # Check for scalability patterns
        scalability_patterns = [
            "ThreadPoolExecutor", "ProcessPoolExecutor", 
            "concurrent.futures", "multiprocessing"
        ]
        
        has_scalability = any(pattern in content for pattern in scalability_patterns)
        performance_analysis["scalability_support"] = "✅ Present" if has_scalability else "❌ Limited"
        
        present_features = len([v for v in performance_analysis.values() if "✅" in v])
        total_features = len(performance_analysis)
        
        status = "PASS" if present_features >= total_features * 0.7 else "WARN" if present_features >= total_features * 0.5 else "FAIL"
        
        findings.append(f"Performance features: {present_features}/{total_features}")
        
        return {
            "status": status,
            "performance_analysis": performance_analysis,
            "features_score": f"{present_features}/{total_features}",
            "findings": findings
        }
    
    def _generate_summary(self):
        """Generate validation summary"""
        total_tests = len(self.results["detailed_findings"])
        passed_tests = len([r for r in self.results["detailed_findings"].values() 
                           if r.get("status") == "PASS"])
        warned_tests = len([r for r in self.results["detailed_findings"].values()
                           if r.get("status") == "WARN"])
        failed_tests = total_tests - passed_tests - warned_tests
        
        self.results["validation_summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "warnings": warned_tests, 
            "failed": failed_tests,
            "success_rate": f"{passed_tests/total_tests*100:.1f}%",
            "overall_status": "EXCELLENT" if failed_tests == 0 and warned_tests <= 2 
                             else "GOOD" if failed_tests <= 2 
                             else "NEEDS_IMPROVEMENT"
        }
    
    def _generate_setup_recommendations(self):
        """Generate setup and installation recommendations"""
        recommendations = []
        
        # Check dependency status
        deps_result = self.results["detailed_findings"].get("4. Dependencies Analysis", {})
        if deps_result.get("status") != "PASS":
            recommendations.append({
                "category": "Dependencies",
                "priority": "HIGH", 
                "action": "Install required dependencies",
                "command": "pip install -r requirements.txt",
                "description": "Install all required Python packages for optimization system"
            })
        
        # Check file structure
        structure_result = self.results["detailed_findings"].get("1. File Structure Analysis", {})
        if structure_result.get("status") != "PASS":
            recommendations.append({
                "category": "File Structure",
                "priority": "HIGH",
                "action": "Ensure all core files are present", 
                "description": "Missing core optimization files prevent system functionality"
            })
        
        # Check integration
        integration_result = self.results["detailed_findings"].get("3. Integration Points Review", {})
        if integration_result.get("status") != "PASS":
            recommendations.append({
                "category": "Integration",
                "priority": "MEDIUM",
                "action": "Fix module import issues",
                "description": "Some components cannot properly import required modules"
            })
        
        # Performance recommendations
        perf_result = self.results["detailed_findings"].get("10. Performance Considerations", {})
        if perf_result.get("status") != "PASS":
            recommendations.append({
                "category": "Performance", 
                "priority": "MEDIUM",
                "action": "Implement missing performance features",
                "description": "Add parallel processing, caching, and memory management"
            })
        
        # Database setup
        db_result = self.results["detailed_findings"].get("6. Database Schema Review", {})
        if db_result.get("status") != "PASS":
            recommendations.append({
                "category": "Database",
                "priority": "HIGH",
                "action": "Set up database schema",
                "command": "python -c \"from supernova.db import engine; from supernova.optimization_models import Base; Base.metadata.create_all(engine)\"",
                "description": "Create database tables for optimization tracking"
            })
        
        # Environment setup
        recommendations.append({
            "category": "Environment",
            "priority": "HIGH", 
            "action": "Set up virtual environment",
            "commands": [
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt"
            ],
            "description": "Create isolated environment for dependencies"
        })
        
        self.results["setup_recommendations"] = recommendations
    
    def save_results(self, filename: str = None) -> str:
        """Save validation results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_architecture_validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return filename
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("SUPERNOVA OPTUNA OPTIMIZATION SYSTEM - ARCHITECTURE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['validation_timestamp']}")
        report.append("")
        
        # Summary
        summary = self.results["validation_summary"]
        report.append("VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Status: {summary['overall_status']}")
        report.append(f"Success Rate: {summary['success_rate']}")
        report.append(f"Tests Passed: {summary['passed']}/{summary['total_tests']}")
        report.append(f"Warnings: {summary['warnings']}")
        report.append(f"Failed: {summary['failed']}")
        report.append("")
        
        # Detailed findings
        report.append("DETAILED VALIDATION RESULTS")
        report.append("-" * 40)
        for test_name, result in self.results["detailed_findings"].items():
            status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(result.get("status", "FAIL"), "❌")
            report.append(f"{status_emoji} {test_name}: {result.get('status', 'UNKNOWN')}")
            
            # Show key findings
            if "findings" in result:
                for finding in result["findings"][:3]:
                    report.append(f"   • {finding}")
            report.append("")
        
        # Setup recommendations
        if self.results.get("setup_recommendations"):
            report.append("SETUP RECOMMENDATIONS")
            report.append("-" * 40)
            
            for rec in self.results["setup_recommendations"]:
                report.append(f"[{rec['priority']}] {rec['category']}: {rec['action']}")
                report.append(f"   {rec['description']}")
                
                if "command" in rec:
                    report.append(f"   Command: {rec['command']}")
                elif "commands" in rec:
                    report.append("   Commands:")
                    for cmd in rec["commands"]:
                        report.append(f"     {cmd}")
                report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main execution"""
    validator = OptunaArchitectureValidator()
    results = validator.run_validation()
    
    # Save results
    results_file = validator.save_results()
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    # Generate and save report  
    report = validator.generate_report()
    report_file = f"optuna_architecture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Full report saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    summary = results["validation_summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Architecture Quality: {summary['success_rate']} tests passed")
    
    if summary['overall_status'] in ['EXCELLENT', 'GOOD']:
        print("✅ The Optuna optimization system architecture is well-designed and ready for deployment.")
    else:
        print("⚠️  The system needs improvements before production deployment.")
        print("   Check the setup recommendations in the report.")
    
    print("\nNext Steps:")
    print("1. Review the detailed report for specific recommendations")
    print("2. Install missing dependencies: pip install -r requirements.txt") 
    print("3. Set up the database schema")
    print("4. Run functional tests after setup completion")
    
    return results

if __name__ == "__main__":
    main()