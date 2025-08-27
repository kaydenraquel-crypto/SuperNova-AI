#!/usr/bin/env python3
"""
Production Readiness Evaluator
==============================

This script evaluates production readiness based on comprehensive test results.
It provides a scoring system and recommendations for production deployment.

Evaluation Criteria:
- Test Coverage and Success Rates
- Security Vulnerability Assessment
- Performance Benchmarks
- Data Integrity Validation
- Error Recovery Capabilities
- Accessibility Compliance
- Code Quality Metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class ProductionReadinessCriteria:
    """Production readiness evaluation criteria."""
    name: str
    weight: float  # Weight in overall score (0.0 to 1.0)
    threshold: float  # Minimum score required
    current_score: float = 0.0
    status: str = "unknown"  # pass, warning, fail
    details: Dict[str, Any] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.recommendations is None:
            self.recommendations = []
        
        # Determine status based on threshold
        if self.current_score >= self.threshold:
            self.status = "pass"
        elif self.current_score >= self.threshold * 0.8:  # 80% of threshold
            self.status = "warning"
        else:
            self.status = "fail"


@dataclass
class ProductionReadinessReport:
    """Complete production readiness assessment."""
    timestamp: str
    project_name: str = "SuperNova AI"
    version: str = "1.0.0"
    environment: str = "production"
    
    # Individual criteria scores
    criteria: List[ProductionReadinessCriteria] = None
    
    # Overall assessment
    overall_score: int = 0
    weighted_score: float = 0.0
    production_ready: bool = False
    readiness_level: str = "not_ready"  # not_ready, conditional, ready
    
    # Issues and recommendations
    critical_issues: List[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None
    
    # Deployment gates
    gates_passed: int = 0
    total_gates: int = 0
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = []
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []


class ProductionReadinessEvaluator:
    """Evaluate production readiness from test results."""
    
    def __init__(self, test_results_dir: str):
        self.test_results_dir = Path(test_results_dir)
        self.logger = logging.getLogger(__name__)
        
        # Define production readiness criteria
        self.criteria_definitions = [
            {
                "name": "Unit Test Coverage",
                "weight": 0.15,
                "threshold": 85.0,
                "description": "Code coverage from unit tests"
            },
            {
                "name": "Unit Test Success Rate",
                "weight": 0.15,
                "threshold": 95.0,
                "description": "Unit test pass rate"
            },
            {
                "name": "Integration Test Success Rate",
                "weight": 0.20,
                "threshold": 90.0,
                "description": "Integration test pass rate"
            },
            {
                "name": "Security Assessment",
                "weight": 0.20,
                "threshold": 80.0,
                "description": "Security vulnerability assessment"
            },
            {
                "name": "Performance Benchmarks",
                "weight": 0.10,
                "threshold": 75.0,
                "description": "Performance test results"
            },
            {
                "name": "Data Integrity",
                "weight": 0.10,
                "threshold": 90.0,
                "description": "Data integrity validation"
            },
            {
                "name": "Error Recovery",
                "weight": 0.05,
                "threshold": 70.0,
                "description": "Error handling and recovery capabilities"
            },
            {
                "name": "Accessibility Compliance",
                "weight": 0.05,
                "threshold": 80.0,
                "description": "WCAG accessibility compliance"
            }
        ]
    
    def load_test_results(self) -> Dict[str, Any]:
        """Load test results from artifacts."""
        results = {}
        
        # Look for test result files
        for artifact_dir in self.test_results_dir.iterdir():
            if not artifact_dir.is_dir():
                continue
            
            # Load JSON reports
            for json_file in artifact_dir.glob("*report.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    results[json_file.stem] = data
                except Exception as e:
                    self.logger.warning(f"Could not load {json_file}: {e}")
            
            # Load XML test results
            for xml_file in artifact_dir.glob("*-test-results.xml"):
                results[xml_file.stem] = {"file": str(xml_file)}
        
        return results
    
    def evaluate_unit_test_coverage(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate unit test coverage."""
        criteria = ProductionReadinessCriteria(
            name="Unit Test Coverage",
            weight=0.15,
            threshold=85.0
        )
        
        # Look for coverage data
        coverage_score = 0.0
        
        # Check various sources for coverage data
        for key, data in test_results.items():
            if 'coverage' in key.lower() and isinstance(data, dict):
                if 'line_rate' in data:
                    coverage_score = data['line_rate'] * 100
                    break
                elif 'percent_covered' in data:
                    coverage_score = data['percent_covered']
                    break
        
        # Default coverage estimation if not found
        if coverage_score == 0.0:
            # Look for unit test results to estimate coverage
            unit_test_data = test_results.get('unit-test-results', {})
            if isinstance(unit_test_data, dict) and 'success_rate' in unit_test_data:
                # Estimate coverage based on test success (rough approximation)
                coverage_score = min(85.0, unit_test_data['success_rate'] * 0.9)
        
        criteria.current_score = coverage_score
        
        # Add recommendations based on coverage
        if coverage_score < 70:
            criteria.recommendations.extend([
                "Increase unit test coverage to at least 85%",
                "Focus on testing critical business logic",
                "Add tests for edge cases and error conditions"
            ])
        elif coverage_score < 85:
            criteria.recommendations.extend([
                "Improve unit test coverage to meet 85% threshold",
                "Review uncovered code paths for testing opportunities"
            ])
        
        criteria.details = {
            "coverage_percentage": coverage_score,
            "threshold": criteria.threshold,
            "meets_threshold": coverage_score >= criteria.threshold
        }
        
        return criteria
    
    def evaluate_test_success_rates(self, test_results: Dict[str, Any]) -> List[ProductionReadinessCriteria]:
        """Evaluate test success rates."""
        criteria_list = []
        
        # Unit test success rate
        unit_criteria = ProductionReadinessCriteria(
            name="Unit Test Success Rate",
            weight=0.15,
            threshold=95.0
        )
        
        # Integration test success rate
        integration_criteria = ProductionReadinessCriteria(
            name="Integration Test Success Rate", 
            weight=0.20,
            threshold=90.0
        )
        
        # Extract test success rates
        for key, data in test_results.items():
            if 'unit' in key.lower() and isinstance(data, dict):
                if 'success_rate' in data:
                    unit_criteria.current_score = data['success_rate']
                elif 'passed' in data and 'total' in data:
                    unit_criteria.current_score = (data['passed'] / data['total']) * 100 if data['total'] > 0 else 0
            
            elif 'integration' in key.lower() and isinstance(data, dict):
                if 'success_rate' in data:
                    integration_criteria.current_score = data['success_rate']
                elif 'passed' in data and 'total' in data:
                    integration_criteria.current_score = (data['passed'] / data['total']) * 100 if data['total'] > 0 else 0
        
        # Add recommendations
        if unit_criteria.current_score < 95:
            unit_criteria.recommendations.extend([
                "Fix failing unit tests to achieve 95% success rate",
                "Review and improve test reliability",
                "Address flaky tests and race conditions"
            ])
        
        if integration_criteria.current_score < 90:
            integration_criteria.recommendations.extend([
                "Fix failing integration tests to achieve 90% success rate",
                "Improve test environment stability",
                "Address integration issues between components"
            ])
        
        criteria_list.extend([unit_criteria, integration_criteria])
        return criteria_list
    
    def evaluate_security_assessment(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate security assessment."""
        criteria = ProductionReadinessCriteria(
            name="Security Assessment",
            weight=0.20,
            threshold=80.0
        )
        
        # Count security issues
        total_issues = 0
        high_issues = 0
        medium_issues = 0
        low_issues = 0
        
        # Look for security scan results
        for key, data in test_results.items():
            if 'security' in key.lower() and isinstance(data, dict):
                # Check for various security tools results
                if 'bandit' in str(data).lower():
                    results = data.get('results', [])
                    for issue in results:
                        severity = issue.get('issue_severity', '').upper()
                        if severity == 'HIGH':
                            high_issues += 1
                        elif severity == 'MEDIUM':
                            medium_issues += 1
                        else:
                            low_issues += 1
                    total_issues += len(results)
                
                elif 'safety' in str(data).lower():
                    # Safety results (dependency vulnerabilities)
                    vulnerabilities = data if isinstance(data, list) else data.get('vulnerabilities', [])
                    high_issues += len(vulnerabilities)  # Assume all dependency issues are high
                    total_issues += len(vulnerabilities)
                
                elif 'trivy' in str(data).lower():
                    # Container security scan results
                    results = data.get('Results', [])
                    for result in results:
                        vulnerabilities = result.get('Vulnerabilities', [])
                        for vuln in vulnerabilities:
                            severity = vuln.get('Severity', '').upper()
                            if severity in ['CRITICAL', 'HIGH']:
                                high_issues += 1
                            elif severity == 'MEDIUM':
                                medium_issues += 1
                            else:
                                low_issues += 1
                        total_issues += len(vulnerabilities)
        
        # Calculate security score (100 - weighted penalty for issues)
        security_score = 100.0
        if total_issues > 0:
            # Apply penalties based on severity
            penalty = (high_issues * 15) + (medium_issues * 5) + (low_issues * 1)
            security_score = max(0, 100 - penalty)
        
        criteria.current_score = security_score
        
        # Add recommendations based on issues
        if high_issues > 0:
            criteria.recommendations.extend([
                f"Address {high_issues} high-severity security issues immediately",
                "Review and patch vulnerable dependencies",
                "Implement security best practices"
            ])
        
        if medium_issues > 5:
            criteria.recommendations.extend([
                f"Address {medium_issues} medium-severity security issues",
                "Prioritize security issues based on risk assessment"
            ])
        
        if total_issues > 20:
            criteria.recommendations.extend([
                "Implement automated security scanning in CI/CD",
                "Establish security review process",
                "Regular security audits and penetration testing"
            ])
        
        criteria.details = {
            "total_issues": total_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
            "security_score": security_score
        }
        
        return criteria
    
    def evaluate_performance_benchmarks(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate performance benchmarks."""
        criteria = ProductionReadinessCriteria(
            name="Performance Benchmarks",
            weight=0.10,
            threshold=75.0
        )
        
        # Look for performance test results
        performance_score = 75.0  # Default score
        
        for key, data in test_results.items():
            if 'performance' in key.lower() and isinstance(data, dict):
                # Extract performance metrics
                metrics = data.get('metrics', {})
                if metrics:
                    response_times = metrics.get('response_times', {})
                    avg_response_time = response_times.get('mean', 0.0)
                    p95_response_time = response_times.get('p95', 0.0)
                    success_rate = metrics.get('success_rate', 100.0)
                    requests_per_second = metrics.get('requests_per_second', 0.0)
                    
                    # Calculate performance score
                    performance_score = 100.0
                    
                    # Penalize high response times
                    if avg_response_time > 2.0:  # > 2 seconds
                        performance_score -= 20
                    elif avg_response_time > 1.0:  # > 1 second
                        performance_score -= 10
                    
                    if p95_response_time > 5.0:  # > 5 seconds
                        performance_score -= 15
                    elif p95_response_time > 3.0:  # > 3 seconds
                        performance_score -= 10
                    
                    # Penalize low success rate
                    if success_rate < 95:
                        performance_score -= (100 - success_rate) * 2
                    
                    # Penalize low throughput
                    if requests_per_second < 10:
                        performance_score -= 10
                    elif requests_per_second < 50:
                        performance_score -= 5
                    
                    performance_score = max(0, performance_score)
                    break
        
        criteria.current_score = performance_score
        
        # Add performance recommendations
        if performance_score < 60:
            criteria.recommendations.extend([
                "Optimize application performance to meet benchmarks",
                "Profile and optimize slow database queries",
                "Implement caching strategies",
                "Review and optimize critical code paths"
            ])
        elif performance_score < 75:
            criteria.recommendations.extend([
                "Fine-tune performance to meet production requirements",
                "Monitor and optimize resource usage"
            ])
        
        return criteria
    
    def evaluate_data_integrity(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate data integrity validation."""
        criteria = ProductionReadinessCriteria(
            name="Data Integrity",
            weight=0.10,
            threshold=90.0
        )
        
        # Look for data integrity test results
        integrity_score = 90.0  # Default passing score
        
        for key, data in test_results.items():
            if 'data' in key.lower() and 'integrity' in key.lower() and isinstance(data, dict):
                # Extract integrity metrics
                if 'overall_health_score' in data:
                    integrity_score = data['overall_health_score']
                elif 'integrity_status' in data:
                    status = data['integrity_status']
                    if status == 'excellent':
                        integrity_score = 95
                    elif status == 'good':
                        integrity_score = 85
                    elif status == 'needs_attention':
                        integrity_score = 70
                    else:
                        integrity_score = 50
                break
        
        criteria.current_score = integrity_score
        
        # Add recommendations based on integrity score
        if integrity_score < 75:
            criteria.recommendations.extend([
                "Address data integrity violations",
                "Implement data validation constraints",
                "Review and fix referential integrity issues",
                "Add comprehensive data consistency checks"
            ])
        elif integrity_score < 90:
            criteria.recommendations.extend([
                "Improve data integrity validation",
                "Implement automated integrity monitoring"
            ])
        
        return criteria
    
    def evaluate_error_recovery(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate error recovery capabilities."""
        criteria = ProductionReadinessCriteria(
            name="Error Recovery",
            weight=0.05,
            threshold=70.0
        )
        
        # Look for error recovery test results
        recovery_score = 70.0  # Default score
        
        for key, data in test_results.items():
            if 'error' in key.lower() and 'recovery' in key.lower() and isinstance(data, dict):
                if 'resilience_score' in data:
                    recovery_score = data['resilience_score']
                elif 'recovery_success' in data and data['recovery_success']:
                    recovery_score = 80.0
                break
        
        criteria.current_score = recovery_score
        
        # Add recommendations
        if recovery_score < 60:
            criteria.recommendations.extend([
                "Implement comprehensive error handling",
                "Add circuit breaker patterns for external services",
                "Implement retry mechanisms with exponential backoff",
                "Add graceful degradation for non-critical features"
            ])
        elif recovery_score < 70:
            criteria.recommendations.extend([
                "Improve error recovery mechanisms",
                "Add monitoring and alerting for error conditions"
            ])
        
        return criteria
    
    def evaluate_accessibility_compliance(self, test_results: Dict[str, Any]) -> ProductionReadinessCriteria:
        """Evaluate accessibility compliance."""
        criteria = ProductionReadinessCriteria(
            name="Accessibility Compliance",
            weight=0.05,
            threshold=80.0
        )
        
        # Look for accessibility test results
        accessibility_score = 80.0  # Default score
        
        for key, data in test_results.items():
            if 'accessibility' in key.lower() and isinstance(data, dict):
                # Calculate score based on WCAG compliance
                wcag_compliance = data.get('wcag_compliance', {})
                if wcag_compliance:
                    level_aa_status = wcag_compliance.get('level_aa_status', '')
                    if 'partial' in level_aa_status.lower():
                        accessibility_score = 75
                    elif 'compliant' in level_aa_status.lower():
                        accessibility_score = 90
                    else:
                        accessibility_score = 60
                break
        
        criteria.current_score = accessibility_score
        
        # Add recommendations
        if accessibility_score < 70:
            criteria.recommendations.extend([
                "Implement comprehensive accessibility features",
                "Ensure WCAG 2.1 AA compliance",
                "Add proper ARIA labels and semantic markup",
                "Test with screen readers and assistive technologies"
            ])
        elif accessibility_score < 80:
            criteria.recommendations.extend([
                "Improve accessibility compliance to meet WCAG 2.1 AA standards",
                "Add accessibility testing to CI/CD pipeline"
            ])
        
        return criteria
    
    def evaluate_production_readiness(self) -> ProductionReadinessReport:
        """Evaluate overall production readiness."""
        # Load test results
        test_results = self.load_test_results()
        
        # Initialize report
        report = ProductionReadinessReport(
            timestamp=datetime.now().isoformat()
        )
        
        # Evaluate each criteria
        report.criteria = []
        
        # Unit test coverage
        report.criteria.append(self.evaluate_unit_test_coverage(test_results))
        
        # Test success rates
        report.criteria.extend(self.evaluate_test_success_rates(test_results))
        
        # Security assessment
        report.criteria.append(self.evaluate_security_assessment(test_results))
        
        # Performance benchmarks
        report.criteria.append(self.evaluate_performance_benchmarks(test_results))
        
        # Data integrity
        report.criteria.append(self.evaluate_data_integrity(test_results))
        
        # Error recovery
        report.criteria.append(self.evaluate_error_recovery(test_results))
        
        # Accessibility compliance
        report.criteria.append(self.evaluate_accessibility_compliance(test_results))
        
        # Calculate overall scores
        self.calculate_overall_scores(report)
        
        # Generate recommendations
        self.generate_recommendations(report)
        
        return report
    
    def calculate_overall_scores(self, report: ProductionReadinessReport):
        """Calculate overall production readiness scores."""
        total_weight = sum(criteria.weight for criteria in report.criteria)
        weighted_sum = sum(criteria.current_score * criteria.weight for criteria in report.criteria)
        
        if total_weight > 0:
            report.weighted_score = weighted_sum / total_weight
            report.overall_score = int(report.weighted_score)
        
        # Count deployment gates
        report.total_gates = len(report.criteria)
        report.gates_passed = sum(1 for criteria in report.criteria if criteria.status == "pass")
        
        # Determine readiness level
        critical_failures = sum(1 for criteria in report.criteria if criteria.status == "fail" and criteria.weight >= 0.15)
        
        if report.overall_score >= 85 and critical_failures == 0:
            report.readiness_level = "ready"
            report.production_ready = True
        elif report.overall_score >= 70 and critical_failures == 0:
            report.readiness_level = "conditional"
            report.production_ready = False
        else:
            report.readiness_level = "not_ready"
            report.production_ready = False
        
        # Collect critical issues and warnings
        for criteria in report.criteria:
            if criteria.status == "fail":
                if criteria.weight >= 0.15:  # High-weight criteria
                    report.critical_issues.append(
                        f"{criteria.name}: {criteria.current_score:.1f}% (required: {criteria.threshold}%)"
                    )
                else:
                    report.warnings.append(
                        f"{criteria.name}: {criteria.current_score:.1f}% (required: {criteria.threshold}%)"
                    )
            elif criteria.status == "warning":
                report.warnings.append(
                    f"{criteria.name}: {criteria.current_score:.1f}% (recommended: {criteria.threshold}%)"
                )
    
    def generate_recommendations(self, report: ProductionReadinessReport):
        """Generate production readiness recommendations."""
        # Collect recommendations from criteria
        all_recommendations = []
        for criteria in report.criteria:
            all_recommendations.extend(criteria.recommendations)
        
        # Add overall recommendations based on readiness level
        if report.readiness_level == "not_ready":
            all_recommendations.extend([
                "Address all critical issues before considering production deployment",
                "Implement comprehensive testing strategy",
                "Establish monitoring and alerting systems",
                "Conduct thorough security review"
            ])
        elif report.readiness_level == "conditional":
            all_recommendations.extend([
                "Address remaining issues and implement monitoring",
                "Plan for gradual rollout with close monitoring",
                "Establish rollback procedures"
            ])
        else:
            all_recommendations.extend([
                "Ready for production deployment",
                "Maintain monitoring and continuous improvement",
                "Regular security and performance reviews"
            ])
        
        # Remove duplicates and prioritize
        report.recommendations = list(dict.fromkeys(all_recommendations))
    
    def generate_json_report(self, report: ProductionReadinessReport) -> str:
        """Generate JSON report."""
        return json.dumps(asdict(report), indent=2, default=str)
    
    def generate_summary_report(self, report: ProductionReadinessReport) -> str:
        """Generate summary report for console output."""
        summary = f"""
Production Readiness Assessment
==============================

Project: {report.project_name}
Version: {report.version}
Timestamp: {report.timestamp}

Overall Score: {report.overall_score}/100
Readiness Level: {report.readiness_level.replace('_', ' ').title()}
Production Ready: {'Yes' if report.production_ready else 'No'}

Deployment Gates: {report.gates_passed}/{report.total_gates} passed

Criteria Assessment:
"""
        
        for criteria in report.criteria:
            status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[criteria.status]
            summary += f"  {status_icon} {criteria.name}: {criteria.current_score:.1f}% (threshold: {criteria.threshold}%)\n"
        
        if report.critical_issues:
            summary += f"\nCritical Issues ({len(report.critical_issues)}):\n"
            for issue in report.critical_issues:
                summary += f"  ❌ {issue}\n"
        
        if report.warnings:
            summary += f"\nWarnings ({len(report.warnings)}):\n"
            for warning in report.warnings:
                summary += f"  ⚠️ {warning}\n"
        
        summary += f"\nTop Recommendations:\n"
        for i, rec in enumerate(report.recommendations[:5], 1):
            summary += f"  {i}. {rec}\n"
        
        return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate production readiness from test results")
    parser.add_argument("--test-results", required=True, help="Directory containing test results")
    parser.add_argument("--output", required=True, help="Output file for readiness report (JSON)")
    parser.add_argument("--format", default="json", choices=["json", "summary"], help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Evaluate production readiness
    evaluator = ProductionReadinessEvaluator(args.test_results)
    report = evaluator.evaluate_production_readiness()
    
    # Generate output
    if args.format == "json":
        content = evaluator.generate_json_report(report)
    else:
        content = evaluator.generate_summary_report(report)
    
    # Write output file
    with open(args.output, 'w') as f:
        f.write(content)
    
    # Print summary to console
    summary = evaluator.generate_summary_report(report)
    print(summary)
    
    # Exit with appropriate code
    if report.production_ready:
        sys.exit(0)
    elif report.readiness_level == "conditional":
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Error


if __name__ == "__main__":
    main()