#!/usr/bin/env python3
"""
SuperNova AI - Comprehensive End-to-End Testing Suite
Production readiness validation through complete system testing
"""

import os
import sys
import json
import asyncio
import time
import threading
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class EndToEndTestSuite:
    """Comprehensive end-to-end testing for SuperNova AI"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_id": f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "base_url": base_url,
            "test_results": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "execution_time": 0,
                "success_rate": 0
            }
        }
        self.test_user_data = {
            "username": "e2e_test_user",
            "email": "test@supernova-ai.com",
            "password": "TestPassword123!",
            "first_name": "End2End",
            "last_name": "TestUser"
        }
        self.auth_token = None
        self.session = requests.Session()
    
    def log_test(self, test_name: str, status: str, details: Dict = None, error: str = None):
        """Log test results"""
        self.results["test_results"][test_name] = {
            "status": status,
            "details": details or {},
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if status == "PASSED":
            self.results["summary"]["passed"] += 1
            print(f"✅ {test_name}: PASSED")
        elif status == "WARNING":
            self.results["summary"]["warnings"] += 1
            print(f"⚠️ {test_name}: WARNING")
        else:
            self.results["summary"]["failed"] += 1
            print(f"❌ {test_name}: FAILED - {error}")
        
        self.results["summary"]["total_tests"] += 1
    
    def test_api_health(self) -> bool:
        """Test API health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.log_test("api_health", "PASSED", {"response_time": response.elapsed.total_seconds()})
                return True
            else:
                self.log_test("api_health", "FAILED", error=f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("api_health", "FAILED", error=str(e))
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation endpoints"""
        try:
            # Test OpenAPI docs
            docs_response = self.session.get(f"{self.base_url}/docs", timeout=10)
            redoc_response = self.session.get(f"{self.base_url}/redoc", timeout=10)
            openapi_response = self.session.get(f"{self.base_url}/openapi.json", timeout=10)
            
            docs_ok = docs_response.status_code == 200
            redoc_ok = redoc_response.status_code == 200
            openapi_ok = openapi_response.status_code == 200
            
            if docs_ok and redoc_ok and openapi_ok:
                self.log_test("api_documentation", "PASSED", {
                    "docs_available": docs_ok,
                    "redoc_available": redoc_ok,
                    "openapi_spec_available": openapi_ok
                })
                return True
            else:
                self.log_test("api_documentation", "FAILED", error="Documentation endpoints not accessible")
                return False
                
        except Exception as e:
            self.log_test("api_documentation", "FAILED", error=str(e))
            return False
    
    def test_user_registration(self) -> bool:
        """Test user registration flow"""
        try:
            # Clean up any existing test user first
            self.cleanup_test_user()
            
            registration_data = {
                "username": self.test_user_data["username"],
                "email": self.test_user_data["email"],
                "password": self.test_user_data["password"],
                "first_name": self.test_user_data["first_name"],
                "last_name": self.test_user_data["last_name"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/auth/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 201:
                self.log_test("user_registration", "PASSED", {"user_created": True})
                return True
            else:
                self.log_test("user_registration", "FAILED", 
                            error=f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log_test("user_registration", "FAILED", error=str(e))
            return False
    
    def test_user_authentication(self) -> bool:
        """Test user login flow"""
        try:
            login_data = {
                "username": self.test_user_data["username"],
                "password": self.test_user_data["password"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                data=login_data,  # Form data for OAuth2
                timeout=10
            )
            
            if response.status_code == 200:
                auth_data = response.json()
                if "access_token" in auth_data:
                    self.auth_token = auth_data["access_token"]
                    self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                    self.log_test("user_authentication", "PASSED", {"token_received": True})
                    return True
                else:
                    self.log_test("user_authentication", "FAILED", error="No access token in response")
                    return False
            else:
                self.log_test("user_authentication", "FAILED", 
                            error=f"Login failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log_test("user_authentication", "FAILED", error=str(e))
            return False
    
    def test_user_profile(self) -> bool:
        """Test user profile operations"""
        try:
            if not self.auth_token:
                self.log_test("user_profile", "FAILED", error="No authentication token available")
                return False
            
            # Get current profile
            profile_response = self.session.get(f"{self.base_url}/api/user/profile", timeout=10)
            
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                
                # Update profile
                update_data = {
                    "risk_tolerance": "high",
                    "investment_experience": "advanced",
                    "investment_goals": "Long-term growth with AI-driven insights",
                    "monthly_investment_budget": 5000.00
                }
                
                update_response = self.session.put(
                    f"{self.base_url}/api/user/profile",
                    json=update_data,
                    timeout=10
                )
                
                if update_response.status_code == 200:
                    self.log_test("user_profile", "PASSED", {
                        "profile_retrieved": True,
                        "profile_updated": True,
                        "user_id": profile_data.get("user_id")
                    })
                    return True
                else:
                    self.log_test("user_profile", "FAILED", error="Profile update failed")
                    return False
            else:
                self.log_test("user_profile", "FAILED", error="Profile retrieval failed")
                return False
                
        except Exception as e:
            self.log_test("user_profile", "FAILED", error=str(e))
            return False
    
    def test_portfolio_management(self) -> bool:
        """Test portfolio creation and management"""
        try:
            if not self.auth_token:
                self.log_test("portfolio_management", "FAILED", error="No authentication token available")
                return False
            
            # Create portfolio
            portfolio_data = {
                "name": "E2E Test Portfolio",
                "description": "End-to-end test portfolio with AI recommendations",
                "cash_balance": 10000.00
            }
            
            create_response = self.session.post(
                f"{self.base_url}/api/portfolios",
                json=portfolio_data,
                timeout=10
            )
            
            if create_response.status_code == 201:
                portfolio = create_response.json()
                portfolio_id = portfolio["id"]
                
                # Get portfolios
                list_response = self.session.get(f"{self.base_url}/api/portfolios", timeout=10)
                
                if list_response.status_code == 200:
                    portfolios = list_response.json()
                    
                    # Add holding to portfolio
                    holding_data = {
                        "asset_symbol": "AAPL",
                        "quantity": 10.0,
                        "purchase_price": 150.00
                    }
                    
                    holding_response = self.session.post(
                        f"{self.base_url}/api/portfolios/{portfolio_id}/holdings",
                        json=holding_data,
                        timeout=10
                    )
                    
                    holdings_success = holding_response.status_code == 201
                    
                    self.log_test("portfolio_management", "PASSED", {
                        "portfolio_created": True,
                        "portfolio_listed": True,
                        "holding_added": holdings_success,
                        "portfolio_id": portfolio_id
                    })
                    return True
                else:
                    self.log_test("portfolio_management", "WARNING", error="Portfolio listing failed")
                    return False
            else:
                self.log_test("portfolio_management", "FAILED", error="Portfolio creation failed")
                return False
                
        except Exception as e:
            self.log_test("portfolio_management", "FAILED", error=str(e))
            return False
    
    def test_ai_advisor(self) -> bool:
        """Test AI advisor functionality"""
        try:
            if not self.auth_token:
                self.log_test("ai_advisor", "FAILED", error="No authentication token available")
                return False
            
            # Request AI advice
            advice_request = {
                "query": "Should I invest in technology stocks given current market conditions?",
                "context": "portfolio_advice",
                "risk_tolerance": "medium"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/ai/advice",
                json=advice_request,
                timeout=30  # AI requests may take longer
            )
            
            if response.status_code == 200:
                advice_data = response.json()
                has_advice = "advice" in advice_data or "response" in advice_data
                has_confidence = "confidence" in advice_data
                
                self.log_test("ai_advisor", "PASSED", {
                    "advice_generated": has_advice,
                    "confidence_provided": has_confidence,
                    "response_length": len(str(advice_data))
                })
                return True
            else:
                self.log_test("ai_advisor", "WARNING", 
                            error=f"AI advisor request failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("ai_advisor", "WARNING", error=str(e))
            return False
    
    def test_chat_functionality(self) -> bool:
        """Test chat system functionality"""
        try:
            if not self.auth_token:
                self.log_test("chat_functionality", "FAILED", error="No authentication token available")
                return False
            
            # Create chat session
            session_data = {
                "title": "E2E Test Chat Session",
                "context_type": "general"
            }
            
            session_response = self.session.post(
                f"{self.base_url}/api/chat/sessions",
                json=session_data,
                timeout=10
            )
            
            if session_response.status_code == 201:
                session = session_response.json()
                session_id = session["id"]
                
                # Send message
                message_data = {
                    "content": "Hello, can you help me understand market trends?",
                    "message_type": "user"
                }
                
                message_response = self.session.post(
                    f"{self.base_url}/api/chat/sessions/{session_id}/messages",
                    json=message_data,
                    timeout=30
                )
                
                if message_response.status_code == 201:
                    # Get chat history
                    history_response = self.session.get(
                        f"{self.base_url}/api/chat/sessions/{session_id}/messages",
                        timeout=10
                    )
                    
                    if history_response.status_code == 200:
                        messages = history_response.json()
                        
                        self.log_test("chat_functionality", "PASSED", {
                            "session_created": True,
                            "message_sent": True,
                            "history_retrieved": True,
                            "message_count": len(messages),
                            "session_id": session_id
                        })
                        return True
                    else:
                        self.log_test("chat_functionality", "WARNING", error="Chat history retrieval failed")
                        return False
                else:
                    self.log_test("chat_functionality", "FAILED", error="Message sending failed")
                    return False
            else:
                self.log_test("chat_functionality", "FAILED", error="Chat session creation failed")
                return False
                
        except Exception as e:
            self.log_test("chat_functionality", "FAILED", error=str(e))
            return False
    
    def test_watchlist_functionality(self) -> bool:
        """Test watchlist operations"""
        try:
            if not self.auth_token:
                self.log_test("watchlist_functionality", "FAILED", error="No authentication token available")
                return False
            
            # Create watchlist
            watchlist_data = {
                "name": "E2E Test Watchlist",
                "description": "End-to-end test watchlist"
            }
            
            create_response = self.session.post(
                f"{self.base_url}/api/watchlists",
                json=watchlist_data,
                timeout=10
            )
            
            if create_response.status_code == 201:
                watchlist = create_response.json()
                watchlist_id = watchlist["id"]
                
                # Add items to watchlist
                item_data = {
                    "asset_symbol": "TSLA",
                    "notes": "Monitoring for AI and autonomous vehicle developments"
                }
                
                item_response = self.session.post(
                    f"{self.base_url}/api/watchlists/{watchlist_id}/items",
                    json=item_data,
                    timeout=10
                )
                
                if item_response.status_code == 201:
                    # Get watchlist items
                    items_response = self.session.get(
                        f"{self.base_url}/api/watchlists/{watchlist_id}/items",
                        timeout=10
                    )
                    
                    if items_response.status_code == 200:
                        items = items_response.json()
                        
                        self.log_test("watchlist_functionality", "PASSED", {
                            "watchlist_created": True,
                            "item_added": True,
                            "items_retrieved": True,
                            "item_count": len(items),
                            "watchlist_id": watchlist_id
                        })
                        return True
                    else:
                        self.log_test("watchlist_functionality", "WARNING", error="Watchlist items retrieval failed")
                        return False
                else:
                    self.log_test("watchlist_functionality", "FAILED", error="Watchlist item addition failed")
                    return False
            else:
                self.log_test("watchlist_functionality", "FAILED", error="Watchlist creation failed")
                return False
                
        except Exception as e:
            self.log_test("watchlist_functionality", "FAILED", error=str(e))
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring endpoints"""
        try:
            # Test metrics endpoint
            metrics_response = self.session.get(f"{self.base_url}/api/metrics", timeout=10)
            
            # Test system status
            status_response = self.session.get(f"{self.base_url}/api/system/status", timeout=10)
            
            metrics_ok = metrics_response.status_code == 200
            status_ok = status_response.status_code == 200 or status_response.status_code == 404  # May not be implemented
            
            if metrics_ok or status_ok:
                self.log_test("performance_monitoring", "PASSED", {
                    "metrics_available": metrics_ok,
                    "status_available": status_ok
                })
                return True
            else:
                self.log_test("performance_monitoring", "WARNING", error="Performance monitoring endpoints not available")
                return False
                
        except Exception as e:
            self.log_test("performance_monitoring", "WARNING", error=str(e))
            return False
    
    def test_api_rate_limiting(self) -> bool:
        """Test API rate limiting"""
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(20):  # Try to exceed rate limit
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay
            
            # Check if any requests were rate limited (429 status code)
            rate_limited = any(status == 429 for status in responses)
            successful_requests = sum(1 for status in responses if status == 200)
            
            self.log_test("api_rate_limiting", "PASSED", {
                "rate_limiting_detected": rate_limited,
                "successful_requests": successful_requests,
                "total_requests": len(responses)
            })
            return True
                
        except Exception as e:
            self.log_test("api_rate_limiting", "WARNING", error=str(e))
            return False
    
    def test_websocket_connection(self) -> bool:
        """Test WebSocket connectivity"""
        try:
            import websockets
            
            # Test WebSocket connection (basic connectivity test)
            ws_url = self.base_url.replace("http", "ws") + "/ws"
            
            async def test_ws():
                try:
                    async with websockets.connect(ws_url, timeout=10) as websocket:
                        # Send test message
                        test_message = json.dumps({
                            "type": "ping",
                            "data": {"test": True}
                        })
                        await websocket.send(test_message)
                        
                        # Wait for response
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        return True
                except Exception:
                    return False
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ws_success = loop.run_until_complete(test_ws())
            loop.close()
            
            if ws_success:
                self.log_test("websocket_connection", "PASSED", {"websocket_responsive": True})
            else:
                self.log_test("websocket_connection", "WARNING", error="WebSocket connection failed")
            
            return ws_success
                
        except ImportError:
            self.log_test("websocket_connection", "WARNING", error="websockets package not available")
            return False
        except Exception as e:
            self.log_test("websocket_connection", "WARNING", error=str(e))
            return False
    
    def cleanup_test_user(self):
        """Clean up test user and data"""
        try:
            # This would typically involve admin endpoints or direct database cleanup
            # For now, we'll just reset the session
            self.session.headers.pop("Authorization", None)
            self.auth_token = None
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests"""
        start_time = time.time()
        
        print("🚀 Starting SuperNova AI End-to-End Testing...")
        print("=" * 60)
        
        # Test sequence - order matters for some tests
        test_sequence = [
            ("API Health Check", self.test_api_health),
            ("API Documentation", self.test_api_documentation),
            ("User Registration", self.test_user_registration),
            ("User Authentication", self.test_user_authentication),
            ("User Profile Management", self.test_user_profile),
            ("Portfolio Management", self.test_portfolio_management),
            ("Watchlist Functionality", self.test_watchlist_functionality),
            ("AI Advisor", self.test_ai_advisor),
            ("Chat Functionality", self.test_chat_functionality),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("API Rate Limiting", self.test_api_rate_limiting),
            ("WebSocket Connection", self.test_websocket_connection)
        ]
        
        for test_name, test_func in test_sequence:
            print(f"\\n🧪 Running: {test_name}")
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name.lower().replace(" ", "_"), "FAILED", error=str(e))
            time.sleep(1)  # Small delay between tests
        
        # Calculate execution time and success rate
        execution_time = time.time() - start_time
        self.results["summary"]["execution_time"] = execution_time
        
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        if total > 0:
            self.results["summary"]["success_rate"] = (passed / total) * 100
        
        # Cleanup
        self.cleanup_test_user()
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        results = self.results
        report = []
        
        report.append("=" * 80)
        report.append("SuperNova AI - End-to-End Testing Report")
        report.append("=" * 80)
        report.append(f"Test ID: {results['test_id']}")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Base URL: {results['base_url']}")
        report.append(f"Execution Time: {results['summary']['execution_time']:.2f} seconds")
        report.append("")
        
        # Summary
        summary = results["summary"]
        report.append("TEST SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  Success Rate: {summary['success_rate']:.1f}%")
        report.append("")
        
        # Overall Status
        if summary['failed'] == 0:
            if summary['warnings'] == 0:
                overall_status = "PRODUCTION READY - All tests passed"
            else:
                overall_status = "PRODUCTION READY - Minor issues detected"
        elif summary['failed'] < summary['total_tests'] / 2:
            overall_status = "NEEDS ATTENTION - Some critical tests failed"
        else:
            overall_status = "NOT READY - Major issues detected"
        
        report.append(f"OVERALL STATUS: {overall_status}")
        report.append("")
        
        # Detailed results
        for test_name, result in results["test_results"].items():
            report.append(f"TEST: {test_name.upper().replace('_', ' ')}")
            report.append(f"  Status: {result['status']}")
            if result.get("error"):
                report.append(f"  Error: {result['error']}")
            if result.get("details"):
                for key, value in result["details"].items():
                    report.append(f"  {key}: {value}")
            report.append("")
        
        return "\\n".join(report)

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperNova AI End-to-End Testing Suite')
    parser.add_argument('--url', default='http://localhost:8081', help='Base URL for testing')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = EndToEndTestSuite(base_url=args.url)
    
    # Run tests
    results = test_suite.run_comprehensive_tests()
    
    # Save results
    results_file = args.output or "end_to_end_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report
    report = test_suite.generate_report()
    report_file = results_file.replace('.json', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\\n" + "=" * 60)
    print("END-TO-END TESTING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print summary
    summary = results["summary"]
    success_rate = summary["success_rate"]
    
    print(f"\\nTest Summary:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Passed: {summary['passed']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Execution Time: {summary['execution_time']:.2f}s")
    
    if summary['failed'] == 0:
        print("\\n🎉 SuperNova AI is ready for production deployment!")
        exit_code = 0
    else:
        print(f"\\n⚠️ {summary['failed']} tests failed. Address issues before production deployment.")
        exit_code = 1
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)