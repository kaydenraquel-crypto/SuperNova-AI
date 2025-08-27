#!/usr/bin/env python3
"""
SuperNova AI - WebSocket Communication Validation
Test real-time WebSocket functionality
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class WebSocketValidator:
    """WebSocket real-time communication validation for SuperNova AI"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_id": f"websocket_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "websocket_tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "websocket_score": 0
            }
        }
        self.base_path = Path(__file__).parent
    
    def validate_websocket_infrastructure(self) -> Dict[str, Any]:
        """Validate WebSocket infrastructure files"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        websocket_files = {
            "websocket_handler": "supernova/websocket_handler.py",
            "websocket_optimizer": "supernova/websocket_optimizer.py",
            "chat_handler": "supernova/chat.py",
            "api_websocket_routes": "supernova/api.py"
        }
        
        websocket_patterns = {
            "websocket_support": ["websocket", "ws", "WebSocket"],
            "async_support": ["async def", "await", "asyncio"],
            "connection_management": ["connect", "disconnect", "connection"],
            "message_handling": ["message", "send", "receive", "broadcast"],
            "real_time": ["real", "time", "live", "stream"],
            "chat_features": ["chat", "typing", "presence", "notification"]
        }
        
        found_files = 0
        total_ws_score = 0
        
        for component, file_path in websocket_files.items():
            full_path = self.base_path / file_path
            component_result = {
                "file_exists": False,
                "file_size": 0,
                "websocket_patterns": {},
                "websocket_score": 0
            }
            
            if full_path.exists():
                component_result["file_exists"] = True
                found_files += 1
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        component_result["file_size"] = len(content)
                    
                    content_lower = content.lower()
                    pattern_scores = []
                    
                    for pattern_type, patterns in websocket_patterns.items():
                        pattern_found = any(p.lower() in content_lower for p in patterns)
                        component_result["websocket_patterns"][pattern_type] = pattern_found
                        pattern_scores.append(1 if pattern_found else 0)
                    
                    # Calculate WebSocket score for this component
                    if pattern_scores:
                        component_result["websocket_score"] = (sum(pattern_scores) / len(pattern_scores)) * 100
                        total_ws_score += component_result["websocket_score"]
                        
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {component}: {str(e)}")
                    component_result["error"] = str(e)
            else:
                test_results["errors"].append(f"Missing WebSocket file: {file_path}")
            
            test_results["details"][component] = component_result
        
        # Calculate overall WebSocket score
        if found_files > 0:
            test_results["overall_websocket_score"] = total_ws_score / found_files
        else:
            test_results["overall_websocket_score"] = 0
        
        # Determine status
        if found_files >= 3:  # At least 3 out of 4 WebSocket files
            test_results["status"] = "PASSED"
        elif found_files >= 2:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_websocket_message_types(self) -> Dict[str, Any]:
        """Validate WebSocket message type support"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        message_types = [
            "chat",
            "typing", 
            "presence",
            "notification",
            "system",
            "market_update",
            "chart_data",
            "voice_message",
            "file_share"
        ]
        
        websocket_files = [
            "supernova/websocket_handler.py",
            "supernova/chat.py",
            "supernova/schemas.py"
        ]
        
        found_message_types = set()
        file_coverage = {}
        
        for file_path in websocket_files:
            full_path = self.base_path / file_path
            file_result = {
                "file_exists": False,
                "message_types_found": [],
                "message_type_count": 0
            }
            
            if full_path.exists():
                file_result["file_exists"] = True
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for msg_type in message_types:
                        if msg_type in content:
                            file_result["message_types_found"].append(msg_type)
                            found_message_types.add(msg_type)
                    
                    file_result["message_type_count"] = len(file_result["message_types_found"])
                    
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {file_path}: {str(e)}")
            else:
                test_results["errors"].append(f"Missing WebSocket file: {file_path}")
            
            file_coverage[file_path] = file_result
        
        test_results["details"] = file_coverage
        test_results["summary"] = {
            "total_message_types": len(message_types),
            "supported_message_types": len(found_message_types),
            "message_type_coverage": f"{len(found_message_types)}/{len(message_types)}",
            "supported_types": list(found_message_types)
        }
        
        # Determine status
        coverage_ratio = len(found_message_types) / len(message_types)
        if coverage_ratio >= 0.7:  # 70% or more message types supported
            test_results["status"] = "PASSED"
        elif coverage_ratio >= 0.5:  # 50% or more
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_connection_management(self) -> Dict[str, Any]:
        """Validate WebSocket connection management"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        connection_patterns = {
            "connect_handling": ["connect", "on_connect", "connection_opened"],
            "disconnect_handling": ["disconnect", "on_disconnect", "connection_closed"],
            "error_handling": ["error", "exception", "try", "catch"],
            "connection_tracking": ["connections", "active", "track", "manage"],
            "heartbeat": ["ping", "pong", "heartbeat", "keepalive"],
            "authentication": ["auth", "token", "verify", "validate"]
        }
        
        websocket_files = [
            "supernova/websocket_handler.py",
            "supernova/websocket_optimizer.py"
        ]
        
        total_connection_features = 0
        files_with_connection_mgmt = 0
        
        for file_path in websocket_files:
            full_path = self.base_path / file_path
            file_result = {
                "file_exists": False,
                "connection_features": {},
                "connection_management_score": 0
            }
            
            if full_path.exists():
                file_result["file_exists"] = True
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    features_found = 0
                    for feature, patterns in connection_patterns.items():
                        feature_found = any(p in content for p in patterns)
                        file_result["connection_features"][feature] = feature_found
                        if feature_found:
                            features_found += 1
                    
                    file_result["connection_management_score"] = (features_found / len(connection_patterns)) * 100
                    total_connection_features += features_found
                    
                    if features_found > 0:
                        files_with_connection_mgmt += 1
                        
                except Exception as e:
                    test_results["warnings"].append(f"Error reading {file_path}: {str(e)}")
            else:
                test_results["errors"].append(f"Missing connection management file: {file_path}")
            
            test_results["details"][file_path] = file_result
        
        test_results["summary"] = {
            "total_connection_features": total_connection_features,
            "files_with_connection_mgmt": files_with_connection_mgmt,
            "connection_coverage": f"{files_with_connection_mgmt}/{len(websocket_files)}"
        }
        
        # Determine status
        if files_with_connection_mgmt >= len(websocket_files):
            test_results["status"] = "PASSED"
        elif files_with_connection_mgmt >= 1:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def validate_frontend_websocket_integration(self) -> Dict[str, Any]:
        """Validate frontend WebSocket integration"""
        test_results = {
            "status": "PENDING",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        frontend_websocket_patterns = [
            "websocket",
            "socket.io",
            "ws://",
            "wss://",
            "useWebSocket",
            "WebSocket",
            "socketio-client"
        ]
        
        frontend_files = [
            "frontend/src/hooks/useWebSocket.tsx",
            "frontend/src/services/api.ts",
            "frontend/src/components/chat",
            "frontend/package.json"
        ]
        
        websocket_implementations = 0
        files_with_websocket = 0
        
        for file_path in frontend_files:
            full_path = self.base_path / file_path
            file_result = {
                "file_exists": False,
                "websocket_patterns_found": [],
                "websocket_implementation_count": 0
            }
            
            # Handle directory case
            if file_path.endswith("/chat"):
                if full_path.exists() and full_path.is_dir():
                    file_result["file_exists"] = True
                    try:
                        # Check all files in chat directory
                        chat_files = list(full_path.glob("*.tsx")) + list(full_path.glob("*.ts"))
                        ws_count = 0
                        for chat_file in chat_files:
                            with open(chat_file, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                for pattern in frontend_websocket_patterns:
                                    if pattern.lower() in content:
                                        if pattern not in file_result["websocket_patterns_found"]:
                                            file_result["websocket_patterns_found"].append(pattern)
                                        ws_count += content.count(pattern.lower())
                        
                        file_result["websocket_implementation_count"] = ws_count
                        websocket_implementations += ws_count
                        
                        if ws_count > 0:
                            files_with_websocket += 1
                            
                    except Exception as e:
                        test_results["warnings"].append(f"Error reading chat directory: {str(e)}")
            else:
                if full_path.exists():
                    file_result["file_exists"] = True
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                        
                        ws_count = 0
                        for pattern in frontend_websocket_patterns:
                            count = content.count(pattern.lower())
                            if count > 0:
                                file_result["websocket_patterns_found"].append(pattern)
                                ws_count += count
                        
                        file_result["websocket_implementation_count"] = ws_count
                        websocket_implementations += ws_count
                        
                        if ws_count > 0:
                            files_with_websocket += 1
                            
                    except Exception as e:
                        test_results["warnings"].append(f"Error reading {file_path}: {str(e)}")
                else:
                    test_results["errors"].append(f"Missing frontend WebSocket file: {file_path}")
            
            test_results["details"][file_path] = file_result
        
        test_results["summary"] = {
            "total_websocket_implementations": websocket_implementations,
            "files_with_websocket": files_with_websocket,
            "frontend_websocket_coverage": f"{files_with_websocket}/{len(frontend_files)}"
        }
        
        # Determine status
        if files_with_websocket >= 2:
            test_results["status"] = "PASSED"
        elif files_with_websocket >= 1:
            test_results["status"] = "WARNING"
        else:
            test_results["status"] = "FAILED"
        
        return test_results
    
    def run_websocket_validation(self) -> Dict[str, Any]:
        """Run all WebSocket validation tests"""
        
        print("Starting SuperNova AI WebSocket Validation...")
        print("=" * 50)
        
        websocket_tests = {
            "websocket_infrastructure": self.validate_websocket_infrastructure,
            "websocket_message_types": self.validate_websocket_message_types,
            "connection_management": self.validate_connection_management,
            "frontend_websocket_integration": self.validate_frontend_websocket_integration
        }
        
        for test_name, test_func in websocket_tests.items():
            print(f"\\nRunning {test_name.replace('_', ' ').title()}...")
            
            try:
                result = test_func()
                self.results["websocket_tests"][test_name] = result
                
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
                self.results["websocket_tests"][test_name] = {
                    "status": "CRITICAL_ERROR",
                    "error": str(e)
                }
                self.results["summary"]["failed"] += 1
                self.results["summary"]["total_tests"] += 1
        
        # Calculate overall WebSocket score
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        warnings = self.results["summary"]["warnings"]
        
        if total > 0:
            websocket_score = ((passed * 100) + (warnings * 75)) / (total * 100) * 100
            self.results["summary"]["websocket_score"] = round(websocket_score, 1)
        
        return self.results
    
    def generate_websocket_report(self) -> str:
        """Generate WebSocket validation report"""
        
        report = []
        report.append("=" * 70)
        report.append("SuperNova AI - WebSocket Communication Validation Report")
        report.append("=" * 70)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Validation ID: {self.results['validation_id']}")
        report.append("")
        
        # Summary
        summary = self.results["summary"]
        report.append("WEBSOCKET SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  WebSocket Score: {summary['websocket_score']:.1f}%")
        report.append("")
        
        # WebSocket Status
        if summary['failed'] == 0:
            if summary['websocket_score'] >= 90:
                ws_status = "EXCELLENT - Real-time features fully implemented"
            elif summary['websocket_score'] >= 75:
                ws_status = "GOOD - Real-time communication ready"
            else:
                ws_status = "BASIC - Consider enhancing real-time features"
        else:
            ws_status = "WEBSOCKET ISSUES DETECTED - Address before production"
        
        report.append(f"WEBSOCKET STATUS: {ws_status}")
        report.append("")
        
        # Detailed results
        for test_name, test_result in self.results["websocket_tests"].items():
            report.append(f"WEBSOCKET TEST: {test_name.upper().replace('_', ' ')}")
            report.append(f"Status: {test_result['status']}")
            
            if test_result.get("errors"):
                report.append("WebSocket Issues:")
                for error in test_result["errors"]:
                    report.append(f"  - {error}")
            
            if test_result.get("warnings"):
                report.append("WebSocket Recommendations:")
                for warning in test_result["warnings"]:
                    report.append(f"  - {warning}")
            
            report.append("")
        
        # WebSocket Recommendations
        report.append("WEBSOCKET ENHANCEMENT RECOMMENDATIONS:")
        report.append("  - Implement comprehensive message types")
        report.append("  - Add connection pooling and load balancing")
        report.append("  - Implement heartbeat/ping-pong mechanism")
        report.append("  - Add WebSocket authentication and authorization")
        report.append("  - Implement message queuing for offline users")
        report.append("  - Add real-time data streaming optimization")
        report.append("  - Configure WebSocket scaling for production")
        report.append("")
        
        return "\\n".join(report)

def main():
    """Main WebSocket validation function"""
    validator = WebSocketValidator()
    
    # Run validation
    results = validator.run_websocket_validation()
    
    # Save results
    results_file = "websocket_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report
    report = validator.generate_websocket_report()
    report_file = "websocket_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\\n" + "=" * 50)
    print("WEBSOCKET VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print summary
    summary = results["summary"]
    websocket_score = summary["websocket_score"]
    
    print(f"\\nWebSocket Score: {websocket_score:.1f}%")
    print(f"Passed: {summary['passed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['failed'] == 0 and websocket_score >= 75:
        print("\\nWebSocket real-time communication is production ready!")
    else:
        print("\\nConsider WebSocket improvements before production deployment.")
    
    return results

if __name__ == "__main__":
    main()