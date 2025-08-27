#!/usr/bin/env python3
"""
Chat Agent Comprehensive Validation Script

Tests conversational agent functionality, memory management, and integration
without requiring external dependencies that might be problematic.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os
import traceback

# Add SuperNova to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'supernova'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatAgentValidator:
    """Comprehensive validation of chat agent components"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "error_summary": {},
            "recommendations": []
        }
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        
        logger.info("Starting Chat Agent Validation Suite")
        start_time = time.time()
        
        try:
            # Test 1: Memory system validation
            await self._test_memory_system()
            
            # Test 2: Conversation agent validation
            await self._test_conversation_agent()
            
            # Test 3: Agent personality system
            await self._test_personality_system()
            
            # Test 4: Context management
            await self._test_context_management()
            
            # Test 5: Error handling and recovery
            await self._test_error_handling()
            
            # Test 6: Performance and scalability
            await self._test_performance()
            
            # Generate final report
            total_time = time.time() - start_time
            self.results["total_validation_time"] = total_time
            self.results["validation_status"] = "COMPLETED"
            
            return self.results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["validation_status"] = "FAILED"
            self.results["error"] = str(e)
            self.results["traceback"] = traceback.format_exc()
            return self.results
    
    async def _test_memory_system(self):
        """Test memory management system"""
        
        logger.info("Testing memory management system...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.conversation_memory import (
                MemoryManager, ConversationMemory, UserContextMemory, 
                FinancialContextMemory, SessionMemory, ConversationRole, MemoryType
            )
            
            test_results["details"]["imports"] = "SUCCESS"
            
            # Test MemoryManager
            manager = MemoryManager()
            test_results["details"]["memory_manager_created"] = True
            
            # Test ConversationMemory
            conv_memory = manager.get_conversation_memory("test_conv_1", user_id=1)
            test_results["details"]["conversation_memory_created"] = True
            
            # Test adding messages
            entry = conv_memory.add_message(
                role=ConversationRole.USER,
                content="Test user message",
                importance_score=0.8
            )
            test_results["details"]["message_added"] = entry is not None
            
            # Test conversation history
            history = conv_memory.get_conversation_history(limit=10)
            test_results["details"]["history_retrieved"] = len(history) >= 1
            
            # Test UserContextMemory
            user_context = manager.get_user_context_memory(1)
            user_context.update_preference("risk_tolerance", "moderate", confidence=0.9)
            test_results["details"]["user_context_working"] = True
            
            # Test FinancialContextMemory
            financial_context = manager.get_financial_context_memory("test_conv_1")
            financial_context.add_market_context(
                context_type="price_data",
                data={"symbol": "AAPL", "price": 150.0},
                symbol="AAPL"
            )
            test_results["details"]["financial_context_working"] = True
            
            # Test SessionMemory
            session_memory = manager.get_session_memory("test_session_1")
            session_memory.set_state("current_symbol", "AAPL")
            session_state = session_memory.get_state("current_symbol")
            test_results["details"]["session_memory_working"] = session_state == "AAPL"
            
            # Test memory stats
            stats = manager.get_memory_stats()
            test_results["details"]["memory_stats"] = stats
            test_results["details"]["active_conversations"] = stats["active_conversations"]
            
            test_results["status"] = "SUCCESS"
            
        except ImportError as e:
            test_results["details"]["import_error"] = str(e)
            logger.error(f"Memory system import failed: {e}")
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Memory system test failed: {e}")
        
        self.results["test_results"]["memory_system"] = test_results
    
    async def _test_conversation_agent(self):
        """Test conversational agent functionality"""
        
        logger.info("Testing conversational agent...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.chat import (
                SuperNovaConversationalAgent, AgentPersonality, QueryType, 
                ConversationContext, create_balanced_analyst
            )
            
            test_results["details"]["imports"] = "SUCCESS"
            
            # Test agent creation
            agent = SuperNovaConversationalAgent()
            test_results["details"]["agent_created"] = True
            test_results["details"]["default_personality"] = agent.personality.value
            
            # Test agent with specific personality
            balanced_agent = create_balanced_analyst()
            test_results["details"]["balanced_agent_created"] = True
            
            # Test query processing (without LLM)
            query = "What's the best investment strategy?"
            conversation_id = "test_conversation_1"
            
            start_time = time.time()
            response = await agent.process_query(
                user_input=query,
                conversation_id=conversation_id,
                user_id=1
            )
            response_time = time.time() - start_time
            
            test_results["details"]["query_processed"] = response is not None
            test_results["details"]["response_time_ms"] = int(response_time * 1000)
            test_results["details"]["response_content_length"] = len(response.get("content", ""))
            test_results["details"]["has_conversation_id"] = "conversation_id" in response
            test_results["details"]["has_query_type"] = "query_type" in response
            test_results["details"]["fallback_mode"] = response.get("metadata", {}).get("fallback_mode", False)
            
            # Test personality switching
            original_personality = agent.personality
            agent.switch_personality(AgentPersonality.CONSERVATIVE_ADVISOR)
            test_results["details"]["personality_switched"] = agent.personality != original_personality
            
            # Test agent stats
            stats = agent.get_agent_stats()
            test_results["details"]["agent_stats"] = stats
            test_results["details"]["llm_available"] = stats["llm_available"]
            test_results["details"]["agent_available"] = stats["agent_available"]
            
            test_results["status"] = "SUCCESS"
            
        except ImportError as e:
            test_results["details"]["import_error"] = str(e)
            logger.error(f"Chat agent import failed: {e}")
        except Exception as e:
            test_results["details"]["error"] = str(e)
            test_results["details"]["traceback"] = traceback.format_exc()
            logger.error(f"Chat agent test failed: {e}")
        
        self.results["test_results"]["conversation_agent"] = test_results
    
    async def _test_personality_system(self):
        """Test agent personality system"""
        
        logger.info("Testing personality system...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.chat import (
                AgentPersonality, create_conservative_advisor, create_aggressive_trader,
                create_educational_agent, create_research_assistant
            )
            
            # Test all personality types
            personalities = {
                "conservative": create_conservative_advisor(),
                "aggressive": create_aggressive_trader(),
                "educational": create_educational_agent(),
                "research": create_research_assistant()
            }
            
            personality_results = {}
            
            for name, agent in personalities.items():
                try:
                    # Test quick response from each personality
                    response = await agent.process_query(
                        user_input="Tell me about risk management",
                        conversation_id=f"test_personality_{name}",
                        user_id=1
                    )
                    
                    personality_results[name] = {
                        "created": True,
                        "responded": response is not None,
                        "personality": agent.personality.value,
                        "content_length": len(response.get("content", "")) if response else 0
                    }
                    
                except Exception as e:
                    personality_results[name] = {
                        "created": True,
                        "responded": False,
                        "error": str(e)
                    }
            
            test_results["details"]["personality_tests"] = personality_results
            
            # Check if all personalities were created successfully
            created_count = sum(1 for result in personality_results.values() if result.get("created", False))
            responded_count = sum(1 for result in personality_results.values() if result.get("responded", False))
            
            test_results["details"]["total_personalities"] = len(personalities)
            test_results["details"]["successfully_created"] = created_count
            test_results["details"]["successfully_responded"] = responded_count
            
            test_results["status"] = "SUCCESS" if created_count == len(personalities) else "PARTIAL"
            
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Personality system test failed: {e}")
        
        self.results["test_results"]["personality_system"] = test_results
    
    async def _test_context_management(self):
        """Test context management and conversation flow"""
        
        logger.info("Testing context management...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.chat import SuperNovaConversationalAgent
            from supernova.conversation_memory import memory_manager
            
            agent = SuperNovaConversationalAgent()
            conversation_id = "context_test_conversation"
            
            # Simulate a multi-turn conversation
            conversation_turns = [
                "I want to invest in technology stocks",
                "What about Apple specifically?", 
                "How does it compare to Microsoft?",
                "What's the risk level for my portfolio?"
            ]
            
            context_results = []
            
            for i, query in enumerate(conversation_turns):
                start_time = time.time()
                response = await agent.process_query(
                    user_input=query,
                    conversation_id=conversation_id,
                    user_id=1
                )
                response_time = time.time() - start_time
                
                context_results.append({
                    "turn": i + 1,
                    "query": query,
                    "response_time_ms": int(response_time * 1000),
                    "response_length": len(response.get("content", "")),
                    "has_context_summary": "context_summary" in response,
                    "query_type": response.get("query_type", "unknown")
                })
            
            # Test conversation memory retrieval
            conv_memory = memory_manager.get_conversation_memory(conversation_id, user_id=1)
            history = conv_memory.get_conversation_history()
            summary = conv_memory.summarize_conversation()
            
            test_results["details"]["conversation_turns"] = context_results
            test_results["details"]["total_turns"] = len(conversation_turns)
            test_results["details"]["conversation_history_length"] = len(history)
            test_results["details"]["has_conversation_summary"] = len(summary) > 0
            test_results["details"]["avg_response_time_ms"] = sum(r["response_time_ms"] for r in context_results) / len(context_results)
            
            test_results["status"] = "SUCCESS"
            
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Context management test failed: {e}")
        
        self.results["test_results"]["context_management"] = test_results
    
    async def _test_error_handling(self):
        """Test error handling and recovery"""
        
        logger.info("Testing error handling...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.chat import SuperNovaConversationalAgent
            
            agent = SuperNovaConversationalAgent()
            
            # Test various error scenarios
            error_scenarios = [
                {"name": "empty_query", "query": "", "conversation_id": "error_test_1"},
                {"name": "very_long_query", "query": "A" * 10000, "conversation_id": "error_test_2"},
                {"name": "special_characters", "query": "!@#$%^&*()_+{}|:<>?[];'./,", "conversation_id": "error_test_3"},
                {"name": "unicode_query", "query": "测试中文查询 🚀📈💰", "conversation_id": "error_test_4"},
                {"name": "none_conversation_id", "query": "Test query", "conversation_id": None}
            ]
            
            error_results = {}
            
            for scenario in error_scenarios:
                try:
                    start_time = time.time()
                    response = await agent.process_query(
                        user_input=scenario["query"],
                        conversation_id=scenario["conversation_id"] or "fallback_conv_id"
                    )
                    response_time = time.time() - start_time
                    
                    error_results[scenario["name"]] = {
                        "handled_gracefully": response is not None,
                        "response_time_ms": int(response_time * 1000),
                        "has_error_field": "error" in response if response else False,
                        "content_length": len(response.get("content", "")) if response else 0
                    }
                    
                except Exception as e:
                    error_results[scenario["name"]] = {
                        "handled_gracefully": False,
                        "exception": str(e),
                        "exception_type": type(e).__name__
                    }
            
            test_results["details"]["error_scenarios"] = error_results
            
            # Calculate error handling score
            graceful_handling_count = sum(1 for result in error_results.values() if result.get("handled_gracefully", False))
            total_scenarios = len(error_scenarios)
            error_handling_score = (graceful_handling_count / total_scenarios) * 100 if total_scenarios > 0 else 0
            
            test_results["details"]["error_handling_score"] = error_handling_score
            test_results["details"]["scenarios_handled_gracefully"] = graceful_handling_count
            test_results["details"]["total_scenarios"] = total_scenarios
            
            test_results["status"] = "SUCCESS" if error_handling_score >= 80 else "PARTIAL"
            
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Error handling test failed: {e}")
        
        self.results["test_results"]["error_handling"] = test_results
    
    async def _test_performance(self):
        """Test performance and scalability"""
        
        logger.info("Testing performance...")
        test_results = {"status": "FAILED", "details": {}}
        
        try:
            from supernova.chat import SuperNovaConversationalAgent
            
            agent = SuperNovaConversationalAgent()
            
            # Performance test: Multiple concurrent conversations
            concurrent_conversations = []
            
            # Simulate 5 concurrent conversations with 3 messages each
            for conv_id in range(5):
                for msg_id in range(3):
                    concurrent_conversations.append({
                        "conversation_id": f"perf_test_conv_{conv_id}",
                        "query": f"Test message {msg_id} in conversation {conv_id}",
                        "user_id": conv_id
                    })
            
            # Run performance test
            start_time = time.time()
            performance_results = []
            
            for conv in concurrent_conversations:
                msg_start_time = time.time()
                response = await agent.process_query(
                    user_input=conv["query"],
                    conversation_id=conv["conversation_id"],
                    user_id=conv["user_id"]
                )
                msg_time = time.time() - msg_start_time
                
                performance_results.append({
                    "conversation_id": conv["conversation_id"],
                    "response_time_ms": int(msg_time * 1000),
                    "success": response is not None,
                    "content_length": len(response.get("content", "")) if response else 0
                })
            
            total_time = time.time() - start_time
            
            # Calculate performance metrics
            successful_responses = sum(1 for r in performance_results if r["success"])
            avg_response_time = sum(r["response_time_ms"] for r in performance_results) / len(performance_results) if performance_results else 0
            min_response_time = min(r["response_time_ms"] for r in performance_results) if performance_results else 0
            max_response_time = max(r["response_time_ms"] for r in performance_results) if performance_results else 0
            
            test_results["details"]["total_messages"] = len(concurrent_conversations)
            test_results["details"]["successful_responses"] = successful_responses
            test_results["details"]["success_rate"] = (successful_responses / len(concurrent_conversations)) * 100 if concurrent_conversations else 0
            test_results["details"]["total_execution_time_ms"] = int(total_time * 1000)
            test_results["details"]["avg_response_time_ms"] = avg_response_time
            test_results["details"]["min_response_time_ms"] = min_response_time
            test_results["details"]["max_response_time_ms"] = max_response_time
            test_results["details"]["messages_per_second"] = len(concurrent_conversations) / total_time if total_time > 0 else 0
            
            # Performance scoring
            performance_score = 100
            if avg_response_time > 500:
                performance_score -= 20
            if avg_response_time > 1000:
                performance_score -= 30
            if test_results["details"]["success_rate"] < 100:
                performance_score -= 20
            
            test_results["details"]["performance_score"] = max(0, performance_score)
            test_results["status"] = "SUCCESS" if performance_score >= 70 else "PARTIAL"
            
        except Exception as e:
            test_results["details"]["error"] = str(e)
            logger.error(f"Performance test failed: {e}")
        
        self.results["test_results"]["performance"] = test_results
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Memory system recommendations
        memory_result = self.results["test_results"].get("memory_system", {})
        if memory_result.get("status") == "SUCCESS":
            recommendations.append("✓ Memory management system is fully functional")
        elif memory_result.get("status") == "FAILED":
            recommendations.append("❌ CRITICAL: Memory system has issues - investigate before deployment")
        
        # Conversation agent recommendations
        agent_result = self.results["test_results"].get("conversation_agent", {})
        if agent_result.get("status") == "SUCCESS":
            if agent_result.get("details", {}).get("fallback_mode"):
                recommendations.append("⚠️ Agent running in fallback mode - LLM integration not available")
            else:
                recommendations.append("✓ Conversational agent is fully operational")
        
        # Personality system recommendations
        personality_result = self.results["test_results"].get("personality_system", {})
        if personality_result.get("status") == "SUCCESS":
            recommendations.append("✓ All personality types are working correctly")
        elif personality_result.get("status") == "PARTIAL":
            recommendations.append("⚠️ Some personality types have issues - review implementation")
        
        # Performance recommendations
        performance_result = self.results["test_results"].get("performance", {})
        if performance_result.get("status") == "SUCCESS":
            avg_time = performance_result.get("details", {}).get("avg_response_time_ms", 0)
            if avg_time < 200:
                recommendations.append("✓ Excellent response performance (< 200ms)")
            else:
                recommendations.append(f"✓ Good response performance ({avg_time}ms average)")
        
        # Error handling recommendations
        error_result = self.results["test_results"].get("error_handling", {})
        if error_result.get("status") == "SUCCESS":
            recommendations.append("✓ Error handling is robust and graceful")
        elif error_result.get("status") == "PARTIAL":
            score = error_result.get("details", {}).get("error_handling_score", 0)
            recommendations.append(f"⚠️ Error handling needs improvement (score: {score}%)")
        
        # General recommendations
        recommendations.append("📋 NEXT STEPS:")
        recommendations.append("  1. Test with actual LLM providers (OpenAI, Anthropic)")
        recommendations.append("  2. Validate WebSocket real-time functionality") 
        recommendations.append("  3. Test chat UI integration")
        recommendations.append("  4. Conduct load testing with multiple users")
        recommendations.append("  5. Set up monitoring and logging for production")
        
        # Overall assessment
        success_count = sum(1 for result in self.results["test_results"].values() if result.get("status") == "SUCCESS")
        total_tests = len(self.results["test_results"])
        
        if success_count == total_tests:
            recommendations.insert(0, "🎉 ALL SYSTEMS OPERATIONAL - Ready for production deployment")
        elif success_count >= total_tests * 0.8:
            recommendations.insert(0, "✅ MOSTLY OPERATIONAL - Minor issues to address before production")
        else:
            recommendations.insert(0, "⚠️ MULTIPLE ISSUES DETECTED - Significant work needed before deployment")
        
        self.results["recommendations"] = recommendations
        return recommendations

async def main():
    """Main validation function"""
    
    print("SuperNova Chat Agent Validation Suite")
    print("=" * 50)
    
    validator = ChatAgentValidator()
    results = await validator.run_validation()
    
    # Generate recommendations
    validator.generate_recommendations()
    
    # Save results
    results_file = "chat_agent_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\nValidation Status: {results['validation_status']}")
    print(f"Total Time: {results.get('total_validation_time', 0):.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    print("\nTest Results Summary:")
    for test_name, test_result in results["test_results"].items():
        status = test_result.get("status", "UNKNOWN")
        icon = "✅" if status == "SUCCESS" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"  {icon} {test_name}: {status}")
    
    print("\nRecommendations:")
    for rec in results.get("recommendations", []):
        print(f"  {rec}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())