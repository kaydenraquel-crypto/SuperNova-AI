"""
Simple TimescaleDB Sentiment Feature Store Validation

A lightweight validation script that checks the structure and components
without requiring full database connectivity or all dependencies.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import importlib.util
import ast


class SimpleTimescaleValidator:
    """Simple validation for TimescaleDB sentiment implementation"""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_type": "structure_and_code",
            "test_results": {},
            "summary": {},
            "recommendations": [],
            "issues": []
        }
        
    def log_result(self, test_name: str, status: str, details: dict = None):
        """Log test result"""
        self.results["test_results"][test_name] = {
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        status_icon = {"passed": "[PASS]", "failed": "[FAIL]", "warning": "[WARN]", "info": "[INFO]"}
        print(f"{status_icon.get(status, '[?]')} {test_name}: {status.upper()}")
        if details and status in ["failed", "warning"]:
            print(f"   {details.get('message', details)}")
    
    def validate_file_structure(self):
        """Check if all required files exist"""
        print("\n=== FILE STRUCTURE VALIDATION ===")
        
        required_files = {
            "supernova/sentiment.py": "Core sentiment analysis module",
            "supernova/sentiment_models.py": "TimescaleDB data models", 
            "supernova/timescale_setup.py": "Database setup and management",
            "supernova/workflows.py": "Prefect workflow integration",
            "supernova/api.py": "API endpoints (should contain sentiment endpoints)"
        }
        
        all_exist = True
        for file_path, description in required_files.items():
            if Path(file_path).exists():
                self.log_result(f"file_{Path(file_path).name}", "passed", 
                              {"description": description})
            else:
                self.log_result(f"file_{Path(file_path).name}", "failed", 
                              {"message": f"Missing file: {file_path}", "description": description})
                all_exist = False
                self.results["issues"].append(f"Required file missing: {file_path}")
        
        return all_exist
    
    def validate_sentiment_module(self):
        """Validate sentiment.py module structure"""
        print("\n=== SENTIMENT MODULE VALIDATION ===")
        
        sentiment_file = Path("supernova/sentiment.py")
        if not sentiment_file.exists():
            self.log_result("sentiment_module", "failed", {"message": "sentiment.py not found"})
            return False
            
        try:
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key classes and functions
            required_components = [
                "class SentimentResult",
                "class SentimentSignal", 
                "class SentimentSource",
                "class MarketRegime",
                "class TwitterConnector",
                "class RedditConnector",
                "class NewsConnector",
                "def score_text",
                "def generate_sentiment_signal",
                "async def generate_batch_sentiment_signals"
            ]
            
            found_components = []
            missing_components = []
            
            for component in required_components:
                if component in content:
                    found_components.append(component)
                else:
                    missing_components.append(component)
            
            if missing_components:
                self.log_result("sentiment_components", "warning", 
                              {"missing": missing_components, "found": len(found_components)})
                self.results["issues"].append(f"Missing sentiment components: {missing_components}")
            else:
                self.log_result("sentiment_components", "passed", 
                              {"found": len(found_components)})
            
            # Check for social media integration
            social_features = ["X_BEARER_TOKEN", "REDDIT_CLIENT_ID", "tweepy", "praw"]
            social_found = [feature for feature in social_features if feature in content]
            
            self.log_result("sentiment_social_integration", "passed" if social_found else "warning",
                          {"features_found": social_found, "note": "Social media integration available"})
            
            # Check for advanced NLP features
            nlp_features = ["finbert", "vader", "spacy", "transformers"]
            nlp_found = [feature for feature in nlp_features if feature.lower() in content.lower()]
            
            self.log_result("sentiment_nlp_features", "passed" if nlp_found else "info",
                          {"features_found": nlp_found, "note": "Advanced NLP features available"})
            
            return len(missing_components) == 0
            
        except Exception as e:
            self.log_result("sentiment_module", "failed", {"error": str(e)})
            return False
    
    def validate_sentiment_models(self):
        """Validate sentiment_models.py structure"""
        print("\n=== SENTIMENT MODELS VALIDATION ===")
        
        models_file = Path("supernova/sentiment_models.py")
        if not models_file.exists():
            self.log_result("sentiment_models", "failed", {"message": "sentiment_models.py not found"})
            return False
            
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for TimescaleDB model classes
            required_models = [
                "class TimescaleBase",
                "class SentimentData", 
                "class SentimentAggregates",
                "class SentimentAlerts",
                "class SentimentMetadata"
            ]
            
            found_models = []
            missing_models = []
            
            for model in required_models:
                if model in content:
                    found_models.append(model)
                else:
                    missing_models.append(model)
            
            if missing_models:
                self.log_result("sentiment_models_classes", "failed", 
                              {"missing": missing_models, "found": len(found_models)})
                self.results["issues"].append(f"Missing model classes: {missing_models}")
            else:
                self.log_result("sentiment_models_classes", "passed", 
                              {"found": len(found_models)})
            
            # Check for database functions
            db_functions = [
                "get_timescale_engine",
                "get_timescale_session", 
                "get_timescale_url",
                "from_sentiment_signal"
            ]
            
            found_functions = [func for func in db_functions if func in content]
            missing_functions = [func for func in db_functions if func not in content]
            
            if missing_functions:
                self.log_result("sentiment_models_functions", "warning",
                              {"missing": missing_functions, "found": found_functions})
            else:
                self.log_result("sentiment_models_functions", "passed",
                              {"found": found_functions})
            
            # Check for SQLAlchemy integration
            sqlalchemy_features = ["DeclarativeBase", "mapped_column", "Mapped", "TIMESTAMP", "JSONB"]
            sqlalchemy_found = [feature for feature in sqlalchemy_features if feature in content]
            
            self.log_result("sentiment_models_sqlalchemy", "passed" if sqlalchemy_found else "warning",
                          {"features_found": sqlalchemy_found})
            
            return len(missing_models) == 0
            
        except Exception as e:
            self.log_result("sentiment_models", "failed", {"error": str(e)})
            return False
    
    def validate_timescale_setup(self):
        """Validate timescale_setup.py structure"""
        print("\n=== TIMESCALE SETUP VALIDATION ===")
        
        setup_file = Path("supernova/timescale_setup.py")
        if not setup_file.exists():
            self.log_result("timescale_setup", "failed", {"message": "timescale_setup.py not found"})
            return False
            
        try:
            with open(setup_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for setup manager
            required_components = [
                "class TimescaleSetupManager",
                "def initialize_timescale_database",
                "def get_database_health",
                "def run_maintenance",
                "def create_hypertables",
                "def create_continuous_aggregates",
                "def setup_compression_policies",
                "def setup_retention_policies"
            ]
            
            found_components = []
            missing_components = []
            
            for component in required_components:
                if component in content:
                    found_components.append(component)
                else:
                    missing_components.append(component)
            
            if missing_components:
                self.log_result("timescale_setup_components", "warning",
                              {"missing": missing_components, "found": len(found_components)})
            else:
                self.log_result("timescale_setup_components", "passed",
                              {"found": len(found_components)})
            
            # Check for TimescaleDB-specific features
            timescale_features = [
                "create_hypertable", "add_compression_policy", "add_retention_policy",
                "continuous_aggregate", "time_bucket", "timescaledb_information"
            ]
            
            timescale_found = [feature for feature in timescale_features if feature in content]
            
            self.log_result("timescale_setup_features", "passed" if timescale_found else "warning",
                          {"timescale_features": timescale_found})
            
            return len(missing_components) == 0
            
        except Exception as e:
            self.log_result("timescale_setup", "failed", {"error": str(e)})
            return False
    
    def validate_workflows(self):
        """Validate workflows.py Prefect integration"""
        print("\n=== WORKFLOWS VALIDATION ===")
        
        workflows_file = Path("supernova/workflows.py")
        if not workflows_file.exists():
            self.log_result("workflows", "failed", {"message": "workflows.py not found"})
            return False
            
        try:
            with open(workflows_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for Prefect components
            prefect_components = [
                "@task", "@flow", "from prefect import",
                "sentiment_analysis_flow", "batch_sentiment_analysis_flow",
                "fetch_twitter_data_task", "fetch_reddit_data_task",
                "analyze_sentiment_batch_task", "save_sentiment_to_timescale_task"
            ]
            
            found_prefect = []
            missing_prefect = []
            
            for component in prefect_components:
                if component in content:
                    found_prefect.append(component)
                else:
                    missing_prefect.append(component)
            
            if missing_prefect:
                self.log_result("workflows_prefect", "warning",
                              {"missing": missing_prefect, "found": len(found_prefect)})
            else:
                self.log_result("workflows_prefect", "passed",
                              {"found": len(found_prefect)})
            
            # Check for error handling and fallbacks
            error_handling = ["try:", "except:", "PREFECT_AVAILABLE", "is_prefect_available"]
            error_found = [item for item in error_handling if item in content]
            
            self.log_result("workflows_error_handling", "passed" if error_found else "warning",
                          {"error_handling_found": error_found})
            
            return True
            
        except Exception as e:
            self.log_result("workflows", "failed", {"error": str(e)})
            return False
    
    def validate_api_endpoints(self):
        """Check API endpoints for sentiment features"""
        print("\n=== API ENDPOINTS VALIDATION ===")
        
        api_file = Path("supernova/api.py")
        if not api_file.exists():
            self.log_result("api_endpoints", "failed", {"message": "api.py not found"})
            return False
            
        try:
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for sentiment endpoints
            sentiment_endpoints = [
                "/sentiment/historical", "SentimentHistoryResponse", 
                "SentimentDataPoint", "get_sentiment_historical",
                "TimescaleHealthResponse", "_check_timescale_availability"
            ]
            
            found_endpoints = []
            missing_endpoints = []
            
            for endpoint in sentiment_endpoints:
                if endpoint in content:
                    found_endpoints.append(endpoint)
                else:
                    missing_endpoints.append(endpoint)
            
            if missing_endpoints:
                self.log_result("api_sentiment_endpoints", "warning",
                              {"missing": missing_endpoints, "found": len(found_endpoints)})
            else:
                self.log_result("api_sentiment_endpoints", "passed",
                              {"found": len(found_endpoints)})
            
            # Check for TimescaleDB integration
            timescale_api_features = [
                "get_timescale_session", "SentimentData", "SentimentAggregates",
                "is_timescale_available", "TimescaleSession"
            ]
            
            timescale_found = [feature for feature in timescale_api_features if feature in content]
            
            self.log_result("api_timescale_integration", "passed" if timescale_found else "warning",
                          {"timescale_features": timescale_found})
            
            return True
            
        except Exception as e:
            self.log_result("api_endpoints", "failed", {"error": str(e)})
            return False
    
    def validate_schemas(self):
        """Check if schemas are properly defined"""
        print("\n=== SCHEMAS VALIDATION ===")
        
        schemas_file = Path("supernova/schemas.py")
        if not schemas_file.exists():
            self.log_result("schemas", "failed", {"message": "schemas.py not found"})
            return False
            
        try:
            with open(schemas_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for sentiment-related schemas
            sentiment_schemas = [
                "SentimentDataPoint", "SentimentHistoryRequest", "SentimentHistoryResponse",
                "SentimentAggregateRequest", "SentimentAggregateResponse", 
                "TimescaleHealthResponse", "BulkSentimentInsertRequest"
            ]
            
            found_schemas = [schema for schema in sentiment_schemas if schema in content]
            missing_schemas = [schema for schema in sentiment_schemas if schema not in content]
            
            if missing_schemas:
                self.log_result("schemas_sentiment", "warning",
                              {"missing": missing_schemas, "found": found_schemas})
            else:
                self.log_result("schemas_sentiment", "passed",
                              {"found": found_schemas})
            
            return True
            
        except Exception as e:
            self.log_result("schemas", "failed", {"error": str(e)})
            return False
    
    def analyze_code_quality(self):
        """Analyze code quality and patterns"""
        print("\n=== CODE QUALITY ANALYSIS ===")
        
        # Check for common patterns and best practices
        files_to_check = [
            "supernova/sentiment.py",
            "supernova/sentiment_models.py", 
            "supernova/timescale_setup.py",
            "supernova/workflows.py"
        ]
        
        quality_metrics = {
            "has_docstrings": 0,
            "has_type_hints": 0,
            "has_error_handling": 0,
            "has_logging": 0,
            "total_files": len(files_to_check)
        }
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for documentation
                    if '"""' in content and 'Args:' in content:
                        quality_metrics["has_docstrings"] += 1
                    
                    # Check for type hints
                    if 'typing import' in content or '-> ' in content or ': str' in content:
                        quality_metrics["has_type_hints"] += 1
                    
                    # Check for error handling
                    if 'try:' in content and 'except' in content:
                        quality_metrics["has_error_handling"] += 1
                    
                    # Check for logging
                    if 'logging' in content and 'logger' in content:
                        quality_metrics["has_logging"] += 1
                        
                except Exception:
                    pass
        
        # Calculate percentages
        for metric, count in quality_metrics.items():
            if metric != "total_files":
                percentage = (count / quality_metrics["total_files"]) * 100
                status = "passed" if percentage >= 75 else "warning" if percentage >= 50 else "info"
                self.log_result(f"code_quality_{metric}", status, 
                              {"percentage": round(percentage, 1), "files": count})
        
        return True
    
    def generate_recommendations(self):
        """Generate implementation recommendations"""
        print("\n=== GENERATING RECOMMENDATIONS ===")
        
        # Analyze results to generate recommendations
        test_results = self.results["test_results"]
        
        # Check for failed tests
        failed_tests = [name for name, result in test_results.items() if result["status"] == "failed"]
        warning_tests = [name for name, result in test_results.items() if result["status"] == "warning"]
        
        recommendations = []
        
        if failed_tests:
            recommendations.append({
                "priority": "HIGH",
                "category": "Critical Issues",
                "text": f"Fix {len(failed_tests)} failed validation tests before deployment",
                "details": failed_tests
            })
        
        if warning_tests:
            recommendations.append({
                "priority": "MEDIUM", 
                "category": "Improvements",
                "text": f"Address {len(warning_tests)} warnings for better functionality",
                "details": warning_tests
            })
        
        # Specific recommendations based on findings
        if any("sentiment_social_integration" in name and result["status"] == "warning" 
               for name, result in test_results.items()):
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Social Media",
                "text": "Configure social media API keys (Twitter, Reddit) for comprehensive sentiment analysis",
                "details": ["Set X_BEARER_TOKEN", "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"]
            })
        
        if any("timescale" in name and result["status"] in ["failed", "warning"]
               for name, result in test_results.items()):
            recommendations.append({
                "priority": "HIGH",
                "category": "Database",
                "text": "Set up TimescaleDB connection and configuration",
                "details": ["Install TimescaleDB", "Configure connection parameters", "Run database setup"]
            })
        
        if any("prefect" in name and result["status"] == "warning"
               for name, result in test_results.items()):
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Workflows",
                "text": "Install and configure Prefect for workflow orchestration",
                "details": ["pip install prefect", "Set ENABLE_PREFECT=True in config"]
            })
        
        # Performance and monitoring recommendations
        recommendations.extend([
            {
                "priority": "LOW",
                "category": "Monitoring",
                "text": "Set up monitoring and alerting for sentiment analysis pipeline",
                "details": ["Monitor API response times", "Track sentiment data quality", "Set up error alerts"]
            },
            {
                "priority": "LOW",
                "category": "Testing",
                "text": "Implement comprehensive testing suite",
                "details": ["Unit tests for all components", "Integration tests", "Performance benchmarks"]
            },
            {
                "priority": "MEDIUM",
                "category": "Documentation",
                "text": "Create deployment and configuration documentation",
                "details": ["Installation guide", "Configuration reference", "API documentation"]
            }
        ])
        
        self.results["recommendations"] = recommendations
        
        # Print recommendations
        for rec in recommendations:
            priority_icon = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]"}
            print(f"{priority_icon.get(rec['priority'], '[REC]')} {rec['priority']} - {rec['category']}: {rec['text']}")
    
    def run_validation(self):
        """Run complete validation suite"""
        print("STARTING SIMPLE TIMESCALEDB SENTIMENT VALIDATION")
        print("=" * 60)
        
        validation_start = datetime.now()
        
        # Run validation components
        results = {
            "file_structure": self.validate_file_structure(),
            "sentiment_module": self.validate_sentiment_module(),
            "sentiment_models": self.validate_sentiment_models(),
            "timescale_setup": self.validate_timescale_setup(),
            "workflows": self.validate_workflows(),
            "api_endpoints": self.validate_api_endpoints(),
            "schemas": self.validate_schemas(),
            "code_quality": self.analyze_code_quality()
        }
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Calculate summary
        total_tests = len(self.results["test_results"])
        passed_tests = len([r for r in self.results["test_results"].values() if r["status"] == "passed"])
        failed_tests = len([r for r in self.results["test_results"].values() if r["status"] == "failed"])
        warning_tests = len([r for r in self.results["test_results"].values() if r["status"] == "warning"])
        
        validation_duration = (datetime.now() - validation_start).total_seconds()
        
        self.results["summary"] = {
            "overall_status": "PASSED" if failed_tests == 0 else "ISSUES_FOUND",
            "total_tests": total_tests,
            "passed_tests": passed_tests, 
            "failed_tests": failed_tests,
            "warning_tests": warning_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "validation_duration": validation_duration,
            "component_results": results
        }
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        
        # Overall status
        status_text = "[PASS]" if summary["overall_status"] == "PASSED" else "[ISSUES]"
        print(f"{status_text} Overall Status: {summary['overall_status']}")
        print()
        
        # Statistics
        print("Test Statistics:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Warnings: {summary['warning_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Duration: {summary['validation_duration']:.1f}s")
        print()
        
        # Component results
        print("Component Results:")
        for component, success in summary["component_results"].items():
            status = "[PASS] PASSED" if success else "[ISSUES] ISSUES"
            print(f"   {component}: {status}")
        print()
        
        # Issues summary
        if self.results["issues"]:
            print("Issues Found:")
            for issue in self.results["issues"]:
                print(f"   - {issue}")
            print()
        
        # High priority recommendations
        high_priority = [r for r in self.results["recommendations"] if r["priority"] == "HIGH"]
        if high_priority:
            print("High Priority Actions:")
            for rec in high_priority:
                print(f"   - {rec['text']}")
            print()
        
        print("=" * 60)
        print("Validation Complete!")
        print("=" * 60)


def main():
    """Run validation"""
    validator = SimpleTimescaleValidator()
    results = validator.run_validation()
    
    # Save results
    output_file = "simple_timescale_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Exit with appropriate code
    if results["summary"]["failed_tests"] == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())