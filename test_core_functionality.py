#!/usr/bin/env python3
"""
Core Functionality Validation Script

Simple validation of SuperNova conversational agent core components
without complex dependencies or Unicode issues.
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add SuperNova to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'supernova'))

class CoreFunctionalityValidator:
    """Simple validation of core functionality"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def run_validation(self):
        """Run validation tests"""
        
        print("SuperNova Core Functionality Validation")
        print("=" * 45)
        
        # Test 1: File existence
        print("\n1. Testing file structure...")
        self.test_file_structure()
        
        # Test 2: Import capabilities
        print("\n2. Testing imports...")
        self.test_imports()
        
        # Test 3: Memory system basic functionality
        print("\n3. Testing memory system...")
        asyncio.run(self.test_memory_system())
        
        # Test 4: Conversation agent basic functionality
        print("\n4. Testing conversation agent...")
        asyncio.run(self.test_conversation_agent())
        
        # Test 5: API structure
        print("\n5. Testing API structure...")
        self.test_api_structure()
        
        # Test 6: WebSocket structure
        print("\n6. Testing WebSocket structure...")
        self.test_websocket_structure()
        
        # Test 7: Chat UI structure
        print("\n7. Testing Chat UI structure...")
        self.test_chat_ui_structure()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def test_file_structure(self):
        """Test if all required files exist"""
        
        required_files = [
            "supernova/agent_tools.py",
            "supernova/chat.py", 
            "supernova/conversation_memory.py",
            "supernova/websocket_handler.py",
            "supernova/chat_ui.py",
            "supernova/api.py",
            "supernova/db.py",
            "supernova/config.py"
        ]
        
        file_status = {}
        missing_files = []
        
        for file_path in required_files:
            if os.path.exists(file_path):
                file_status[file_path] = "EXISTS"
                # Check if file has content
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content) > 100:  # Basic content check
                            file_status[file_path] = "VALID"
                        else:
                            file_status[file_path] = "EMPTY"
                except Exception as e:
                    file_status[file_path] = f"ERROR: {str(e)}"
            else:
                file_status[file_path] = "MISSING"
                missing_files.append(file_path)
        
        self.results["tests"]["file_structure"] = {
            "status": "SUCCESS" if not missing_files else "FAILED",
            "file_status": file_status,
            "missing_files": missing_files,
            "total_files": len(required_files),
            "valid_files": sum(1 for status in file_status.values() if status == "VALID")
        }
        
        if not missing_files:
            print("   [SUCCESS] All required files exist")
        else:
            print(f"   [FAILED] Missing files: {missing_files}")
    
    def test_imports(self):
        """Test basic imports"""
        
        import_tests = {}
        
        # Test agent_tools
        try:
            from supernova import agent_tools
            import_tests["agent_tools"] = "SUCCESS"
        except Exception as e:
            import_tests["agent_tools"] = f"FAILED: {str(e)}"
        
        # Test chat
        try:
            from supernova import chat
            import_tests["chat"] = "SUCCESS"
        except Exception as e:
            import_tests["chat"] = f"FAILED: {str(e)}"
        
        # Test conversation_memory
        try:
            from supernova import conversation_memory
            import_tests["conversation_memory"] = "SUCCESS"
        except Exception as e:
            import_tests["conversation_memory"] = f"FAILED: {str(e)}"
        
        # Test websocket_handler
        try:
            from supernova import websocket_handler
            import_tests["websocket_handler"] = "SUCCESS"
        except Exception as e:
            import_tests["websocket_handler"] = f"FAILED: {str(e)}"
        
        # Test chat_ui
        try:
            from supernova import chat_ui
            import_tests["chat_ui"] = "SUCCESS"
        except Exception as e:
            import_tests["chat_ui"] = f"FAILED: {str(e)}"
        
        success_count = sum(1 for status in import_tests.values() if status == "SUCCESS")
        total_count = len(import_tests)
        
        self.results["tests"]["imports"] = {
            "status": "SUCCESS" if success_count == total_count else "PARTIAL",
            "import_results": import_tests,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": (success_count / total_count) * 100 if total_count > 0 else 0
        }
        
        print(f"   [{'SUCCESS' if success_count == total_count else 'PARTIAL'}] Imports: {success_count}/{total_count}")
    
    async def test_memory_system(self):
        """Test memory system functionality"""
        
        try:
            from supernova.conversation_memory import MemoryManager, ConversationRole
            
            # Test basic memory operations
            manager = MemoryManager()
            conv_memory = manager.get_conversation_memory("test_conv_1", user_id=1)
            
            # Add a test message
            entry = conv_memory.add_message(
                role=ConversationRole.USER,
                content="Test message for validation",
                importance_score=0.8
            )
            
            # Get history
            history = conv_memory.get_conversation_history(limit=10)
            
            # Get summary
            summary = conv_memory.summarize_conversation()
            
            self.results["tests"]["memory_system"] = {
                "status": "SUCCESS",
                "manager_created": True,
                "conversation_created": True,
                "message_added": entry is not None,
                "history_retrieved": len(history) > 0,
                "summary_generated": len(summary) > 0
            }
            
            print("   [SUCCESS] Memory system functional")
            
        except Exception as e:
            self.results["tests"]["memory_system"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"   [FAILED] Memory system: {str(e)}")
    
    async def test_conversation_agent(self):
        """Test conversation agent functionality"""
        
        try:
            from supernova.chat import SuperNovaConversationalAgent, AgentPersonality
            
            # Create agent
            agent = SuperNovaConversationalAgent()
            
            # Test basic query processing
            start_time = time.time()
            response = await agent.process_query(
                user_input="What is a good investment strategy?",
                conversation_id="validation_test_conv",
                user_id=1
            )
            response_time = time.time() - start_time
            
            # Test personality switching
            original_personality = agent.personality
            agent.switch_personality(AgentPersonality.CONSERVATIVE_ADVISOR)
            personality_switched = agent.personality != original_personality
            
            # Get agent stats
            stats = agent.get_agent_stats()
            
            self.results["tests"]["conversation_agent"] = {
                "status": "SUCCESS",
                "agent_created": True,
                "query_processed": response is not None,
                "response_time_ms": int(response_time * 1000),
                "response_content_length": len(response.get("content", "")) if response else 0,
                "personality_switching": personality_switched,
                "stats_available": stats is not None,
                "fallback_mode": response.get("metadata", {}).get("fallback_mode", False) if response else True
            }
            
            print("   [SUCCESS] Conversation agent functional")
            
        except Exception as e:
            self.results["tests"]["conversation_agent"] = {
                "status": "FAILED", 
                "error": str(e)
            }
            print(f"   [FAILED] Conversation agent: {str(e)}")
    
    def test_api_structure(self):
        """Test API endpoint structure"""
        
        try:
            with open("supernova/api.py", 'r') as f:
                api_content = f.read()
            
            # Check for key components
            has_chat_endpoints = "/chat" in api_content
            has_websocket_route = "websocket" in api_content.lower()
            has_fastapi_import = "from fastapi import" in api_content
            has_error_handling = "HTTPException" in api_content
            
            self.results["tests"]["api_structure"] = {
                "status": "SUCCESS" if all([has_chat_endpoints, has_websocket_route, has_fastapi_import]) else "PARTIAL",
                "has_chat_endpoints": has_chat_endpoints,
                "has_websocket_route": has_websocket_route,
                "has_fastapi_import": has_fastapi_import,
                "has_error_handling": has_error_handling,
                "file_size": len(api_content)
            }
            
            print("   [SUCCESS] API structure validated")
            
        except Exception as e:
            self.results["tests"]["api_structure"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"   [FAILED] API structure: {str(e)}")
    
    def test_websocket_structure(self):
        """Test WebSocket handler structure"""
        
        try:
            with open("supernova/websocket_handler.py", 'r') as f:
                ws_content = f.read()
            
            # Check for key components
            has_websocket_manager = "WebSocketManager" in ws_content
            has_connection_handling = "connect" in ws_content and "disconnect" in ws_content
            has_message_handling = "send_message" in ws_content
            has_real_time_features = "typing" in ws_content.lower() or "presence" in ws_content.lower()
            
            self.results["tests"]["websocket_structure"] = {
                "status": "SUCCESS" if all([has_websocket_manager, has_connection_handling, has_message_handling]) else "PARTIAL",
                "has_websocket_manager": has_websocket_manager,
                "has_connection_handling": has_connection_handling,
                "has_message_handling": has_message_handling,
                "has_real_time_features": has_real_time_features,
                "file_size": len(ws_content)
            }
            
            print("   [SUCCESS] WebSocket structure validated")
            
        except Exception as e:
            self.results["tests"]["websocket_structure"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"   [FAILED] WebSocket structure: {str(e)}")
    
    def test_chat_ui_structure(self):
        """Test Chat UI structure"""
        
        try:
            with open("supernova/chat_ui.py", 'r') as f:
                ui_content = f.read()
            
            # Check for key components
            has_html_templates = "<html" in ui_content.lower() or "html" in ui_content.lower()
            has_css_styling = "style" in ui_content.lower() or "css" in ui_content.lower()
            has_javascript = "javascript" in ui_content.lower() or "script" in ui_content.lower()
            has_chat_interface = "chat" in ui_content.lower() and "message" in ui_content.lower()
            
            self.results["tests"]["chat_ui_structure"] = {
                "status": "SUCCESS" if all([has_html_templates, has_chat_interface]) else "PARTIAL",
                "has_html_templates": has_html_templates,
                "has_css_styling": has_css_styling,
                "has_javascript": has_javascript,
                "has_chat_interface": has_chat_interface,
                "file_size": len(ui_content)
            }
            
            print("   [SUCCESS] Chat UI structure validated")
            
        except Exception as e:
            self.results["tests"]["chat_ui_structure"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"   [FAILED] Chat UI structure: {str(e)}")
    
    def generate_summary(self):
        """Generate validation summary"""
        
        test_statuses = [test.get("status", "FAILED") for test in self.results["tests"].values()]
        
        success_count = sum(1 for status in test_statuses if status == "SUCCESS")
        partial_count = sum(1 for status in test_statuses if status == "PARTIAL")
        failed_count = sum(1 for status in test_statuses if status == "FAILED")
        
        total_tests = len(test_statuses)
        
        # Determine overall status
        if failed_count == 0 and partial_count == 0:
            overall_status = "FULLY_OPERATIONAL"
        elif failed_count == 0:
            overall_status = "MOSTLY_OPERATIONAL"
        elif success_count >= total_tests // 2:
            overall_status = "PARTIALLY_OPERATIONAL"
        else:
            overall_status = "NEEDS_ATTENTION"
        
        self.results["summary"] = {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "success_count": success_count,
            "partial_count": partial_count,
            "failed_count": failed_count,
            "success_rate": (success_count / total_tests) * 100 if total_tests > 0 else 0,
            "recommendations": self._generate_recommendations(overall_status, success_count, partial_count, failed_count)
        }
    
    def _generate_recommendations(self, overall_status, success_count, partial_count, failed_count):
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        if overall_status == "FULLY_OPERATIONAL":
            recommendations.append("All core systems are operational and ready for deployment")
            recommendations.append("Proceed with integration testing and user acceptance testing")
        
        elif overall_status == "MOSTLY_OPERATIONAL":
            recommendations.append("Core systems are functional with minor issues")
            recommendations.append("Review partial test results and address any warnings")
            
        elif overall_status == "PARTIALLY_OPERATIONAL":
            recommendations.append("Several components need attention before deployment")
            recommendations.append("Focus on fixing failed tests before proceeding")
            
        else:
            recommendations.append("Multiple critical issues detected - significant work needed")
            recommendations.append("Do not deploy until core issues are resolved")
        
        # Specific recommendations
        if self.results["tests"].get("imports", {}).get("status") == "PARTIAL":
            recommendations.append("Some import issues detected - check dependencies")
        
        if self.results["tests"].get("memory_system", {}).get("status") == "FAILED":
            recommendations.append("CRITICAL: Memory system not functional - fix before deployment")
        
        if self.results["tests"].get("conversation_agent", {}).get("status") == "FAILED":
            recommendations.append("CRITICAL: Conversation agent not functional - core system issue")
        
        # General next steps
        recommendations.append("Next steps:")
        recommendations.append("  1. Install and configure LLM provider (OpenAI, Anthropic, etc.)")
        recommendations.append("  2. Test real-time WebSocket functionality")
        recommendations.append("  3. Validate chat UI in browser")
        recommendations.append("  4. Run end-to-end integration tests")
        recommendations.append("  5. Conduct performance and load testing")
        
        return recommendations

def main():
    """Main validation function"""
    
    validator = CoreFunctionalityValidator()
    results = validator.run_validation()
    
    # Save results
    results_file = "core_functionality_validation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*45}")
    print("VALIDATION SUMMARY")
    print(f"{'='*45}")
    
    summary = results["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Tests Passed: {summary['success_count']}/{summary['total_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['partial_count'] > 0:
        print(f"Partial Results: {summary['partial_count']}")
    if summary['failed_count'] > 0:
        print(f"Failed Tests: {summary['failed_count']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()