#!/usr/bin/env python3
"""
Comprehensive Test Report Generator
==================================

This script generates comprehensive test reports from all test suite results including:
- Unit test coverage and results
- Integration test results
- Performance test metrics
- Security scan results
- Accessibility test results
- Data integrity validation
- Error recovery test results

Supports multiple output formats: HTML, JSON, Markdown, PDF
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
import logging

try:
    from jinja2 import Environment, FileSystemLoader, Template
    from markdown import markdown
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from junitparser import JUnitXml
    import pandas as pd
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install jinja2 markdown plotly pandas junitparser")
    sys.exit(1)


@dataclass
class TestSuiteResult:
    """Data structure for test suite results."""
    name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    duration: float = 0.0
    coverage: Optional[float] = None
    success_rate: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.total_tests > 0:
            self.success_rate = (self.passed_tests / self.total_tests) * 100
        if self.details is None:
            self.details = {}


@dataclass
class SecurityResult:
    """Data structure for security scan results."""
    tool_name: str
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    info_issues: int = 0
    total_issues: int = 0
    score: Optional[float] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class PerformanceResult:
    """Data structure for performance test results."""
    test_name: str
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    requests_per_second: float = 0.0
    success_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ComprehensiveTestReport:
    """Comprehensive test report data structure."""
    timestamp: str
    project_name: str = "SuperNova AI"
    version: str = "1.0.0"
    environment: str = "test"
    
    # Test suite results
    unit_tests: Optional[TestSuiteResult] = None
    integration_tests: Optional[TestSuiteResult] = None
    frontend_tests: Optional[TestSuiteResult] = None
    security_tests: Optional[TestSuiteResult] = None
    accessibility_tests: Optional[TestSuiteResult] = None
    data_integrity_tests: Optional[TestSuiteResult] = None
    error_recovery_tests: Optional[TestSuiteResult] = None
    e2e_tests: Optional[TestSuiteResult] = None
    
    # Specialized results
    security_results: List[SecurityResult] = None
    performance_results: List[PerformanceResult] = None
    
    # Overall metrics
    overall_success_rate: float = 0.0
    total_tests_run: int = 0
    total_tests_passed: int = 0
    total_tests_failed: int = 0
    production_ready: bool = False
    readiness_score: int = 0
    
    def __post_init__(self):
        if self.security_results is None:
            self.security_results = []
        if self.performance_results is None:
            self.performance_results = []


class TestReportGenerator:
    """Generate comprehensive test reports from test artifacts."""
    
    def __init__(self, artifacts_dir: str, output_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        if template_dir.exists():
            self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        else:
            self.jinja_env = None
    
    def parse_junit_xml(self, xml_file: Path) -> TestSuiteResult:
        """Parse JUnit XML test results."""
        try:
            xml = JUnitXml.fromfile(str(xml_file))
            
            total_tests = xml.tests
            failures = xml.failures
            errors = xml.errors
            skipped = xml.skipped
            passed = total_tests - failures - errors - skipped
            duration = xml.time
            
            return TestSuiteResult(
                name=xml_file.stem,
                total_tests=total_tests,
                passed_tests=passed,
                failed_tests=failures + errors,
                skipped_tests=skipped,
                duration=duration or 0.0
            )
        
        except Exception as e:
            self.logger.error(f"Error parsing JUnit XML {xml_file}: {e}")
            return TestSuiteResult(name=xml_file.stem)
    
    def parse_coverage_xml(self, xml_file: Path) -> float:
        """Parse coverage XML to get coverage percentage."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Look for coverage element with line-rate attribute
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None and 'line-rate' in coverage_elem.attrib:
                return float(coverage_elem.attrib['line-rate']) * 100
            
            return 0.0
        
        except Exception as e:
            self.logger.error(f"Error parsing coverage XML {xml_file}: {e}")
            return 0.0
    
    def parse_security_json(self, json_file: Path) -> List[SecurityResult]:
        """Parse security scan JSON results."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            results = []
            
            if 'bandit' in json_file.name:
                # Parse Bandit results
                issues = data.get('results', [])
                high = sum(1 for issue in issues if issue.get('issue_severity') == 'HIGH')
                medium = sum(1 for issue in issues if issue.get('issue_severity') == 'MEDIUM')
                low = sum(1 for issue in issues if issue.get('issue_severity') == 'LOW')
                
                results.append(SecurityResult(
                    tool_name='Bandit (SAST)',
                    high_issues=high,
                    medium_issues=medium,
                    low_issues=low,
                    total_issues=len(issues),
                    details=data
                ))
            
            elif 'safety' in json_file.name:
                # Parse Safety results
                issues = data if isinstance(data, list) else data.get('vulnerabilities', [])
                total = len(issues)
                
                results.append(SecurityResult(
                    tool_name='Safety (Dependencies)',
                    high_issues=total,  # Assume all dependency issues are high
                    total_issues=total,
                    details=data
                ))
            
            elif 'trivy' in json_file.name:
                # Parse Trivy results
                if isinstance(data, dict) and 'Results' in data:
                    total_issues = 0
                    high_issues = 0
                    medium_issues = 0
                    low_issues = 0
                    
                    for result in data['Results']:
                        vulnerabilities = result.get('Vulnerabilities', [])
                        total_issues += len(vulnerabilities)
                        
                        for vuln in vulnerabilities:
                            severity = vuln.get('Severity', '').upper()
                            if severity in ['CRITICAL', 'HIGH']:
                                high_issues += 1
                            elif severity == 'MEDIUM':
                                medium_issues += 1
                            else:
                                low_issues += 1
                    
                    results.append(SecurityResult(
                        tool_name='Trivy (Container)',
                        high_issues=high_issues,
                        medium_issues=medium_issues,
                        low_issues=low_issues,
                        total_issues=total_issues,
                        details=data
                    ))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing security JSON {json_file}: {e}")
            return []
    
    def parse_performance_json(self, json_file: Path) -> List[PerformanceResult]:
        """Parse performance test JSON results."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            results = []
            
            # Look for performance metrics structure
            if 'metrics' in data:
                metrics = data['metrics']
                response_times = metrics.get('response_times', {})
                
                result = PerformanceResult(
                    test_name=data.get('test_name', 'Performance Test'),
                    response_time_avg=response_times.get('mean', 0.0),
                    response_time_p95=response_times.get('p95', 0.0),
                    response_time_p99=response_times.get('p99', 0.0),
                    requests_per_second=metrics.get('requests_per_second', 0.0),
                    success_rate=metrics.get('success_rate', 0.0),
                    memory_usage_mb=metrics.get('memory_usage', {}).get('mean_mb', 0.0),
                    cpu_usage_percent=metrics.get('cpu_usage', {}).get('mean_percent', 0.0),
                    details=data
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing performance JSON {json_file}: {e}")
            return []
    
    def collect_test_results(self) -> ComprehensiveTestReport:
        """Collect all test results from artifacts."""
        report = ComprehensiveTestReport(
            timestamp=datetime.now().isoformat(),
            project_name="SuperNova AI",
            version="1.0.0"
        )
        
        # Find and parse test result files
        for artifact_dir in self.artifacts_dir.iterdir():
            if not artifact_dir.is_dir():
                continue
            
            # Parse JUnit XML files
            for xml_file in artifact_dir.glob("*.xml"):
                if 'unit-test' in xml_file.name:
                    report.unit_tests = self.parse_junit_xml(xml_file)
                elif 'integration-test' in xml_file.name:
                    report.integration_tests = self.parse_junit_xml(xml_file)
                elif 'security-test' in xml_file.name:
                    report.security_tests = self.parse_junit_xml(xml_file)
                elif 'accessibility-test' in xml_file.name:
                    report.accessibility_tests = self.parse_junit_xml(xml_file)
                elif 'data-integrity-test' in xml_file.name:
                    report.data_integrity_tests = self.parse_junit_xml(xml_file)
                elif 'error-recovery-test' in xml_file.name:
                    report.error_recovery_tests = self.parse_junit_xml(xml_file)
                elif 'e2e-test' in xml_file.name:
                    report.e2e_tests = self.parse_junit_xml(xml_file)
            
            # Parse coverage XML files
            for xml_file in artifact_dir.glob("coverage.xml"):
                if report.unit_tests:
                    report.unit_tests.coverage = self.parse_coverage_xml(xml_file)
            
            # Parse security JSON files
            for json_file in artifact_dir.glob("*-results.json"):
                if any(sec_tool in json_file.name for sec_tool in ['bandit', 'safety', 'trivy']):
                    security_results = self.parse_security_json(json_file)
                    report.security_results.extend(security_results)
            
            # Parse performance JSON files
            for json_file in artifact_dir.glob("*performance*.json"):
                performance_results = self.parse_performance_json(json_file)
                report.performance_results.extend(performance_results)
        
        # Calculate overall metrics
        self.calculate_overall_metrics(report)
        
        return report
    
    def calculate_overall_metrics(self, report: ComprehensiveTestReport):
        """Calculate overall test metrics and production readiness."""
        test_suites = [
            report.unit_tests,
            report.integration_tests,
            report.frontend_tests,
            report.security_tests,
            report.accessibility_tests,
            report.data_integrity_tests,
            report.error_recovery_tests,
            report.e2e_tests
        ]
        
        valid_suites = [suite for suite in test_suites if suite and suite.total_tests > 0]
        
        if valid_suites:
            report.total_tests_run = sum(suite.total_tests for suite in valid_suites)
            report.total_tests_passed = sum(suite.passed_tests for suite in valid_suites)
            report.total_tests_failed = sum(suite.failed_tests for suite in valid_suites)
            
            if report.total_tests_run > 0:
                report.overall_success_rate = (report.total_tests_passed / report.total_tests_run) * 100
        
        # Calculate production readiness score
        readiness_factors = []
        
        # Test coverage and success rates
        if report.unit_tests and report.unit_tests.success_rate > 0:
            readiness_factors.append(min(100, report.unit_tests.success_rate))
        
        if report.integration_tests and report.integration_tests.success_rate > 0:
            readiness_factors.append(min(100, report.integration_tests.success_rate))
        
        # Security score (inverse of issues)
        security_score = 100
        for security_result in report.security_results:
            # Penalize based on issue severity
            penalty = (security_result.high_issues * 10 + 
                      security_result.medium_issues * 5 + 
                      security_result.low_issues * 1)
            security_score = max(0, security_score - penalty)
        
        if report.security_results:
            readiness_factors.append(security_score)
        
        # Code coverage score
        if report.unit_tests and report.unit_tests.coverage:
            readiness_factors.append(report.unit_tests.coverage)
        
        # Performance score (inverse of response times)
        if report.performance_results:
            avg_response_time = sum(p.response_time_avg for p in report.performance_results) / len(report.performance_results)
            performance_score = max(0, 100 - (avg_response_time * 10))  # Penalize high response times
            readiness_factors.append(performance_score)
        
        # Calculate overall readiness score
        if readiness_factors:
            report.readiness_score = int(sum(readiness_factors) / len(readiness_factors))
        
        # Determine production readiness
        report.production_ready = (
            report.readiness_score >= 80 and
            report.overall_success_rate >= 95 and
            security_score >= 70
        )
    
    def generate_html_report(self, report: ComprehensiveTestReport) -> str:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report.project_name }} - Comprehensive Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 0.9em; margin-top: 5px; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        .test-suite { background: #f8f9fa; margin: 15px 0; padding: 20px; border-radius: 8px; }
        .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
        .table th { background: #f8f9fa; font-weight: 600; }
        .charts { margin: 30px 0; }
        .chart-container { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ report.project_name }} Test Report</h1>
            <p>Generated on {{ report.timestamp }}</p>
            <p>Environment: {{ report.environment }} | Version: {{ report.version }}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {% if report.overall_success_rate >= 95 %}status-good{% elif report.overall_success_rate >= 80 %}status-warning{% else %}status-danger{% endif %}">
                    {{ "%.1f"|format(report.overall_success_rate) }}%
                </div>
                <div class="metric-label">Overall Success Rate</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{{ report.total_tests_run }}</div>
                <div class="metric-label">Total Tests Run</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value {% if report.readiness_score >= 80 %}status-good{% elif report.readiness_score >= 60 %}status-warning{% else %}status-danger{% endif %}">
                    {{ report.readiness_score }}
                </div>
                <div class="metric-label">Production Readiness Score</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value {% if report.production_ready %}status-good{% else %}status-danger{% endif %}">
                    {% if report.production_ready %}✅ Ready{% else %}❌ Not Ready{% endif %}
                </div>
                <div class="metric-label">Production Status</div>
            </div>
        </div>

        <h2>Test Suite Results</h2>
        
        {% for suite_name, suite in [
            ('Unit Tests', report.unit_tests),
            ('Integration Tests', report.integration_tests),
            ('Security Tests', report.security_tests),
            ('Accessibility Tests', report.accessibility_tests),
            ('Data Integrity Tests', report.data_integrity_tests),
            ('Error Recovery Tests', report.error_recovery_tests),
            ('E2E Tests', report.e2e_tests)
        ] %}
            {% if suite %}
            <div class="test-suite">
                <h3>{{ suite_name }}</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ suite.success_rate }}%"></div>
                </div>
                <p>{{ suite.passed_tests }}/{{ suite.total_tests }} tests passed ({{ "%.1f"|format(suite.success_rate) }}%)</p>
                {% if suite.coverage %}
                    <p>Code Coverage: {{ "%.1f"|format(suite.coverage) }}%</p>
                {% endif %}
                <p>Duration: {{ "%.2f"|format(suite.duration) }}s</p>
            </div>
            {% endif %}
        {% endfor %}

        {% if report.security_results %}
        <h2>Security Scan Results</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Tool</th>
                    <th>High Issues</th>
                    <th>Medium Issues</th>
                    <th>Low Issues</th>
                    <th>Total Issues</th>
                </tr>
            </thead>
            <tbody>
                {% for security in report.security_results %}
                <tr>
                    <td>{{ security.tool_name }}</td>
                    <td class="{% if security.high_issues > 0 %}status-danger{% else %}status-good{% endif %}">{{ security.high_issues }}</td>
                    <td class="{% if security.medium_issues > 0 %}status-warning{% else %}status-good{% endif %}">{{ security.medium_issues }}</td>
                    <td>{{ security.low_issues }}</td>
                    <td>{{ security.total_issues }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if report.performance_results %}
        <h2>Performance Test Results</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Avg Response Time</th>
                    <th>P95 Response Time</th>
                    <th>Requests/sec</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
                {% for perf in report.performance_results %}
                <tr>
                    <td>{{ perf.test_name }}</td>
                    <td>{{ "%.3f"|format(perf.response_time_avg) }}s</td>
                    <td>{{ "%.3f"|format(perf.response_time_p95) }}s</td>
                    <td>{{ "%.1f"|format(perf.requests_per_second) }}</td>
                    <td class="{% if perf.success_rate >= 95 %}status-good{% elif perf.success_rate >= 80 %}status-warning{% else %}status-danger{% endif %}">{{ "%.1f"|format(perf.success_rate) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <div class="charts">
            {{ test_results_chart|safe }}
            {{ security_chart|safe }}
        </div>
    </div>
</body>
</html>
        """
        
        # Generate charts
        test_results_chart = self.generate_test_results_chart(report)
        security_chart = self.generate_security_chart(report)
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            report=report,
            test_results_chart=test_results_chart,
            security_chart=security_chart
        )
        
        return html_content
    
    def generate_test_results_chart(self, report: ComprehensiveTestReport) -> str:
        """Generate test results visualization chart."""
        try:
            # Collect test suite data
            suite_names = []
            success_rates = []
            
            test_suites = [
                ('Unit Tests', report.unit_tests),
                ('Integration Tests', report.integration_tests),
                ('Security Tests', report.security_tests),
                ('Accessibility Tests', report.accessibility_tests),
                ('Data Integrity Tests', report.data_integrity_tests),
                ('Error Recovery Tests', report.error_recovery_tests),
                ('E2E Tests', report.e2e_tests)
            ]
            
            for name, suite in test_suites:
                if suite and suite.total_tests > 0:
                    suite_names.append(name)
                    success_rates.append(suite.success_rate)
            
            if not suite_names:
                return ""
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=suite_names,
                    y=success_rates,
                    marker_color=['#28a745' if rate >= 95 else '#ffc107' if rate >= 80 else '#dc3545' for rate in success_rates],
                    text=[f"{rate:.1f}%" for rate in success_rates],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Test Suite Success Rates",
                xaxis_title="Test Suite",
                yaxis_title="Success Rate (%)",
                yaxis_range=[0, 100],
                height=400
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        except Exception as e:
            self.logger.error(f"Error generating test results chart: {e}")
            return ""
    
    def generate_security_chart(self, report: ComprehensiveTestReport) -> str:
        """Generate security issues visualization chart."""
        try:
            if not report.security_results:
                return ""
            
            # Collect security data
            tool_names = []
            high_issues = []
            medium_issues = []
            low_issues = []
            
            for security in report.security_results:
                tool_names.append(security.tool_name)
                high_issues.append(security.high_issues)
                medium_issues.append(security.medium_issues)
                low_issues.append(security.low_issues)
            
            # Create stacked bar chart
            fig = go.Figure(data=[
                go.Bar(name='High', x=tool_names, y=high_issues, marker_color='#dc3545'),
                go.Bar(name='Medium', x=tool_names, y=medium_issues, marker_color='#ffc107'),
                go.Bar(name='Low', x=tool_names, y=low_issues, marker_color='#28a745')
            ])
            
            fig.update_layout(
                title="Security Issues by Tool",
                xaxis_title="Security Tool",
                yaxis_title="Number of Issues",
                barmode='stack',
                height=400
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        except Exception as e:
            self.logger.error(f"Error generating security chart: {e}")
            return ""
    
    def generate_json_report(self, report: ComprehensiveTestReport) -> str:
        """Generate JSON report."""
        return json.dumps(asdict(report), indent=2, default=str)
    
    def generate_markdown_report(self, report: ComprehensiveTestReport) -> str:
        """Generate Markdown report."""
        md_template = f"""# {report.project_name} - Comprehensive Test Report

**Generated:** {report.timestamp}  
**Environment:** {report.environment}  
**Version:** {report.version}

## 📊 Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Overall Success Rate | {report.overall_success_rate:.1f}% | {'✅' if report.overall_success_rate >= 95 else '⚠️' if report.overall_success_rate >= 80 else '❌'} |
| Total Tests Run | {report.total_tests_run} | - |
| Tests Passed | {report.total_tests_passed} | - |
| Tests Failed | {report.total_tests_failed} | - |
| Production Readiness Score | {report.readiness_score}/100 | {'✅' if report.readiness_score >= 80 else '⚠️' if report.readiness_score >= 60 else '❌'} |
| Production Ready | {'Yes' if report.production_ready else 'No'} | {'✅' if report.production_ready else '❌'} |

## 🧪 Test Suite Results

"""
        
        test_suites = [
            ('Unit Tests', report.unit_tests),
            ('Integration Tests', report.integration_tests),
            ('Security Tests', report.security_tests),
            ('Accessibility Tests', report.accessibility_tests),
            ('Data Integrity Tests', report.data_integrity_tests),
            ('Error Recovery Tests', report.error_recovery_tests),
            ('E2E Tests', report.e2e_tests)
        ]
        
        for name, suite in test_suites:
            if suite and suite.total_tests > 0:
                status = '✅' if suite.success_rate >= 95 else '⚠️' if suite.success_rate >= 80 else '❌'
                md_template += f"""
### {name} {status}

- **Success Rate:** {suite.success_rate:.1f}%
- **Tests:** {suite.passed_tests}/{suite.total_tests} passed
- **Duration:** {suite.duration:.2f}s
"""
                if suite.coverage:
                    md_template += f"- **Coverage:** {suite.coverage:.1f}%\n"
        
        # Add security results
        if report.security_results:
            md_template += """
## 🔐 Security Scan Results

| Tool | High | Medium | Low | Total |
|------|------|--------|-----|-------|
"""
            for security in report.security_results:
                md_template += f"| {security.tool_name} | {security.high_issues} | {security.medium_issues} | {security.low_issues} | {security.total_issues} |\n"
        
        # Add performance results
        if report.performance_results:
            md_template += """
## ⚡ Performance Test Results

| Test Name | Avg Response Time | P95 Response Time | RPS | Success Rate |
|-----------|-------------------|-------------------|-----|--------------|
"""
            for perf in report.performance_results:
                md_template += f"| {perf.test_name} | {perf.response_time_avg:.3f}s | {perf.response_time_p95:.3f}s | {perf.requests_per_second:.1f} | {perf.success_rate:.1f}% |\n"
        
        return md_template
    
    def generate_summary_markdown(self, report: ComprehensiveTestReport) -> str:
        """Generate a summary markdown for PR comments."""
        summary = f"""## Test Results Summary

**Overall Success Rate:** {report.overall_success_rate:.1f}% {'✅' if report.overall_success_rate >= 95 else '⚠️' if report.overall_success_rate >= 80 else '❌'}  
**Production Readiness:** {report.readiness_score}/100 {'✅' if report.production_ready else '❌'}

### Test Suites
"""
        
        test_suites = [
            ('Unit', report.unit_tests),
            ('Integration', report.integration_tests),
            ('Security', report.security_tests),
            ('Accessibility', report.accessibility_tests),
            ('Data Integrity', report.data_integrity_tests),
            ('Error Recovery', report.error_recovery_tests)
        ]
        
        for name, suite in test_suites:
            if suite and suite.total_tests > 0:
                status = '✅' if suite.success_rate >= 95 else '⚠️' if suite.success_rate >= 80 else '❌'
                summary += f"- **{name}:** {suite.passed_tests}/{suite.total_tests} ({suite.success_rate:.1f}%) {status}\n"
        
        # Security summary
        if report.security_results:
            total_high = sum(s.high_issues for s in report.security_results)
            total_medium = sum(s.medium_issues for s in report.security_results)
            total_low = sum(s.low_issues for s in report.security_results)
            
            summary += f"""
### Security Issues
- **High:** {total_high} {'❌' if total_high > 0 else '✅'}
- **Medium:** {total_medium} {'⚠️' if total_medium > 0 else '✅'}  
- **Low:** {total_low}
"""
        
        return summary
    
    def generate_reports(self, formats: List[str] = None):
        """Generate reports in specified formats."""
        if formats is None:
            formats = ['html', 'json', 'markdown']
        
        # Collect test results
        self.logger.info("Collecting test results...")
        report = self.collect_test_results()
        
        # Generate reports
        for format_type in formats:
            self.logger.info(f"Generating {format_type.upper()} report...")
            
            if format_type == 'html':
                content = self.generate_html_report(report)
                output_file = self.output_dir / "comprehensive-test-report.html"
            elif format_type == 'json':
                content = self.generate_json_report(report)
                output_file = self.output_dir / "comprehensive-test-report.json"
            elif format_type == 'markdown':
                content = self.generate_markdown_report(report)
                output_file = self.output_dir / "comprehensive-test-report.md"
            else:
                self.logger.warning(f"Unsupported format: {format_type}")
                continue
            
            # Write report file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Generated {format_type.upper()} report: {output_file}")
        
        # Generate summary for PR comments
        summary_content = self.generate_summary_markdown(report)
        summary_file = self.output_dir / "summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.logger.info(f"Generated summary report: {summary_file}")
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive test reports")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing test artifacts")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    parser.add_argument("--format", default="html,json,markdown", help="Report formats (comma-separated)")
    
    args = parser.parse_args()
    
    formats = [f.strip() for f in args.format.split(',')]
    
    generator = TestReportGenerator(args.artifacts_dir, args.output_dir)
    report = generator.generate_reports(formats)
    
    # Print summary to console
    print(f"\n🧪 Test Report Summary")
    print(f"{'='*50}")
    print(f"Overall Success Rate: {report.overall_success_rate:.1f}%")
    print(f"Total Tests Run: {report.total_tests_run}")
    print(f"Tests Passed: {report.total_tests_passed}")
    print(f"Tests Failed: {report.total_tests_failed}")
    print(f"Production Readiness Score: {report.readiness_score}/100")
    print(f"Production Ready: {'Yes' if report.production_ready else 'No'}")
    
    if report.security_results:
        total_security_issues = sum(s.total_issues for s in report.security_results)
        print(f"Total Security Issues: {total_security_issues}")
    
    print(f"\nReports generated in: {args.output_dir}")


if __name__ == "__main__":
    main()