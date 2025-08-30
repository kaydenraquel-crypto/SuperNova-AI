#!/usr/bin/env python3
"""
WebSocket Integration Test for SuperNova-AI
Tests real-time communication features
"""

import asyncio
import json
import time
import websockets
import requests
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketIntegrationTest:
    """Test WebSocket connections and real-time features"""
    
    def __init__(self):
        self.ws_url = "ws://localhost:8081/ws"
        self.api_url = "http://localhost:8081"
        self.results = {
            'websocket_tests': [],
            'realtime_tests': [],
            'performance_metrics': {},
            'issues': []
        }
    
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        logger.info("Testing WebSocket connection...")
        
        try:
            async with websockets.connect(f"{self.ws_url}/test", timeout=10) as websocket:
                # Send ping message
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                if data.get('type') == 'pong':
                    self.results['websocket_tests'].append({
                        'test': 'basic_connection',
                        'status': 'passed',
                        'message': 'WebSocket ping/pong successful'
                    })
                    logger.info("✓ WebSocket connection test passed")
                else:
                    raise Exception(f"Unexpected response: {data}")
                    
        except Exception as e:
            self.results['websocket_tests'].append({
                'test': 'basic_connection',
                'status': 'failed',
                'error': str(e)
            })
            logger.warning(f"WebSocket connection test failed: {str(e)}")
    
    async def test_realtime_data_streaming(self):
        """Test real-time data streaming"""
        logger.info("Testing real-time data streaming...")
        
        try:
            async with websockets.connect(f"{self.ws_url}/market", timeout=10) as websocket:
                # Subscribe to market data
                subscribe_msg = {
                    "type": "subscribe",
                    "symbols": ["AAPL", "GOOGL", "MSFT"]
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for subscription confirmation or data
                messages_received = 0
                for _ in range(3):  # Wait for up to 3 messages
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3)
                        data = json.loads(response)
                        messages_received += 1
                        
                        if data.get('type') in ['subscribed', 'market_data', 'price_update']:
                            logger.info(f"Received: {data.get('type')}")
                    except asyncio.TimeoutError:
                        break
                
                if messages_received > 0:
                    self.results['realtime_tests'].append({
                        'test': 'data_streaming',
                        'status': 'passed',
                        'messages_received': messages_received
                    })
                    logger.info("✓ Real-time data streaming test passed")
                else:
                    raise Exception("No messages received")
                    
        except Exception as e:
            self.results['realtime_tests'].append({
                'test': 'data_streaming',
                'status': 'failed',
                'error': str(e)
            })
            logger.warning(f"Real-time data streaming test failed: {str(e)}")
    
    async def test_chat_websocket(self):
        """Test chat WebSocket functionality"""
        logger.info("Testing chat WebSocket...")
        
        try:
            async with websockets.connect(f"{self.ws_url}/chat", timeout=10) as websocket:
                # Send chat message
                chat_msg = {
                    "type": "message",
                    "content": "Hello, this is a test message",
                    "session_id": "test_session"
                }
                await websocket.send(json.dumps(chat_msg))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)
                
                if data.get('type') == 'message' or data.get('type') == 'response':
                    self.results['realtime_tests'].append({
                        'test': 'chat_websocket',
                        'status': 'passed',
                        'response_type': data.get('type')
                    })
                    logger.info("✓ Chat WebSocket test passed")
                else:
                    raise Exception(f"Unexpected chat response: {data}")
                    
        except Exception as e:
            self.results['realtime_tests'].append({
                'test': 'chat_websocket',
                'status': 'failed',
                'error': str(e)
            })
            logger.warning(f"Chat WebSocket test failed: {str(e)}")
    
    async def test_websocket_performance(self):
        """Test WebSocket performance under load"""
        logger.info("Testing WebSocket performance...")
        
        try:
            message_count = 10
            start_time = time.time()
            
            async with websockets.connect(f"{self.ws_url}/test", timeout=10) as websocket:
                # Send multiple messages rapidly
                for i in range(message_count):
                    await websocket.send(json.dumps({
                        "type": "performance_test",
                        "sequence": i,
                        "timestamp": time.time()
                    }))
                
                # Receive responses
                responses_received = 0
                for _ in range(message_count):
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2)
                        responses_received += 1
                    except asyncio.TimeoutError:
                        break
                
                end_time = time.time()
                total_time = end_time - start_time
                
                self.results['performance_metrics'] = {
                    'messages_sent': message_count,
                    'responses_received': responses_received,
                    'total_time': total_time,
                    'messages_per_second': message_count / total_time if total_time > 0 else 0
                }
                
                logger.info(f"✓ Performance: {message_count} messages in {total_time:.2f}s")
                
        except Exception as e:
            self.results['issues'].append({
                'test': 'websocket_performance',
                'error': str(e)
            })
            logger.warning(f"WebSocket performance test failed: {str(e)}")
    
    async def run_all_tests(self):
        """Run all WebSocket integration tests"""
        logger.info("Starting WebSocket Integration Tests...")
        
        tests = [
            self.test_websocket_connection,
            self.test_realtime_data_streaming,
            self.test_chat_websocket,
            self.test_websocket_performance
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {str(e)}")
                self.results['issues'].append({
                    'test': test.__name__,
                    'error': str(e)
                })
        
        return self.results
    
    def generate_report(self):
        """Generate WebSocket integration test report"""
        websocket_passed = len([t for t in self.results['websocket_tests'] if t['status'] == 'passed'])
        websocket_total = len(self.results['websocket_tests'])
        
        realtime_passed = len([t for t in self.results['realtime_tests'] if t['status'] == 'passed'])
        realtime_total = len(self.results['realtime_tests'])
        
        report = f"""
WebSocket Integration Test Report
=================================
Generated: {datetime.now().isoformat()}

WebSocket Tests: {websocket_passed}/{websocket_total} passed
Real-time Tests: {realtime_passed}/{realtime_total} passed

Performance Metrics:
-------------------
"""
        
        if self.results['performance_metrics']:
            metrics = self.results['performance_metrics']
            report += f"Messages per second: {metrics.get('messages_per_second', 0):.2f}\n"
            report += f"Response rate: {metrics.get('responses_received', 0)}/{metrics.get('messages_sent', 0)}\n"
        
        if self.results['issues']:
            report += "\nIssues Found:\n"
            report += "-------------\n"
            for issue in self.results['issues']:
                report += f"- {issue['test']}: {issue['error']}\n"
        
        # Determine overall status
        total_tests = websocket_total + realtime_total
        total_passed = websocket_passed + realtime_passed
        
        if total_tests == 0:
            status = "NO_TESTS_RUN"
        elif total_passed == total_tests:
            status = "ALL_PASSED"
        elif total_passed >= total_tests * 0.7:
            status = "MOSTLY_PASSED"
        else:
            status = "MULTIPLE_FAILURES"
        
        report += f"\nOverall Status: {status}\n"
        
        return report, status

async def main():
    """Main test runner"""
    test_suite = WebSocketIntegrationTest()
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8081/docs", timeout=5)
        if response.status_code != 200:
            logger.warning("Backend may not be fully running")
    except:
        logger.warning("Backend not accessible - WebSocket tests may fail")
    
    try:
        results = await test_suite.run_all_tests()
        report, status = test_suite.generate_report()
        
        # Save results
        with open('websocket_integration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('websocket_integration_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        if status == "ALL_PASSED":
            return 0
        elif status == "MOSTLY_PASSED":
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"WebSocket integration test suite failed: {str(e)}")
        return 3

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))