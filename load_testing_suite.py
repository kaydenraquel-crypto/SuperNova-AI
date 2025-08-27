"""
SuperNova Load Testing Suite
Comprehensive load testing and stress testing for SuperNova AI system
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

# Load testing frameworks
try:
    import locust
    from locust import HttpUser, task, between, events
    from locust.env import Environment
    from locust.stats import stats_printer, stats_history
    from locust.log import setup_logging
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

# Monitoring and reporting
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LoadTestMetrics:
    """Load test performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    requests_per_second: float = 0.0
    bytes_transferred: int = 0
    error_rate: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    def calculate_percentiles(self, response_times: List[float]):
        """Calculate response time percentiles"""
        if response_times:
            sorted_times = sorted(response_times)
            length = len(sorted_times)
            
            self.p50_response_time = sorted_times[int(length * 0.5)]
            self.p95_response_time = sorted_times[int(length * 0.95)]
            self.p99_response_time = sorted_times[int(length * 0.99)]

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8081"
    max_users: int = 100
    spawn_rate: int = 10
    test_duration: int = 300  # seconds
    min_wait_time: int = 1000  # milliseconds
    max_wait_time: int = 3000  # milliseconds
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    user_scenarios: List[str] = field(default_factory=lambda: ["basic", "advanced", "stress"])
    data_generation: bool = True
    report_output: str = "load_test_results"

class AsyncHTTPLoadTester:
    """High-performance async HTTP load tester"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = LoadTestMetrics()
        self.response_times: List[float] = []
        self.errors: List[Dict[str, Any]] = []
        self.is_running = False
        
    async def run_load_test(self):
        """Run async load test"""
        logger.info(f"Starting async load test - {self.config.max_users} users, {self.config.test_duration}s duration")
        
        self.metrics.start_time = datetime.now()
        self.is_running = True
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create user tasks
            tasks = []
            for user_id in range(self.config.max_users):
                task = asyncio.create_task(self._user_simulation(session, user_id))
                tasks.append(task)
                
                # Gradual user spawn
                if (user_id + 1) % self.config.spawn_rate == 0:
                    await asyncio.sleep(1)
            
            # Run test for specified duration
            await asyncio.sleep(self.config.test_duration)
            self.is_running = False
            
            # Wait for all tasks to complete
            logger.info("Stopping load test, waiting for tasks to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate final metrics
        self.metrics.end_time = datetime.now()
        self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        self._calculate_final_metrics()
        
        logger.info("Async load test completed")
        return self.metrics
    
    async def _user_simulation(self, session: aiohttp.ClientSession, user_id: int):
        """Simulate individual user behavior"""
        while self.is_running:
            try:
                # Select random endpoint
                endpoint_config = random.choice(self.config.endpoints or self._get_default_endpoints())
                
                # Make request
                start_time = time.time()
                
                url = f"{self.config.base_url}{endpoint_config['path']}"
                method = endpoint_config.get('method', 'GET')
                data = self._generate_test_data(endpoint_config) if self.config.data_generation else None
                
                async with session.request(method, url, json=data) as response:
                    response_data = await response.text()
                    response_time = (time.time() - start_time) * 1000
                    
                    # Record metrics
                    self.response_times.append(response_time)
                    self.metrics.total_requests += 1
                    self.metrics.bytes_transferred += len(response_data)
                    
                    if response.status >= 400:
                        self.metrics.failed_requests += 1
                        self.errors.append({
                            'user_id': user_id,
                            'url': url,
                            'status': response.status,
                            'response_time': response_time,
                            'timestamp': datetime.now()
                        })
                    else:
                        self.metrics.successful_requests += 1
                
                # Wait between requests
                wait_time = random.randint(self.config.min_wait_time, self.config.max_wait_time) / 1000
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.metrics.failed_requests += 1
                self.errors.append({
                    'user_id': user_id,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
                await asyncio.sleep(1)  # Brief pause on error
    
    def _generate_test_data(self, endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate test data for endpoints"""
        endpoint_type = endpoint_config.get('type', 'generic')
        
        if endpoint_type == 'sentiment':
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
            return {
                'symbols': random.sample(symbols, random.randint(1, 3)),
                'start_date': (datetime.now() - timedelta(days=30)).isoformat(),
                'end_date': datetime.now().isoformat(),
                'interval': random.choice(['raw', '1h', '6h', '1d'])
            }
        
        elif endpoint_type == 'backtest':
            return {
                'bars': self._generate_ohlcv_data(),
                'strategy_template': random.choice(['sma_crossover', 'rsi_strategy', 'macd_strategy']),
                'params': {'fast_period': random.randint(5, 20), 'slow_period': random.randint(20, 50)},
                'start_cash': random.randint(10000, 100000),
                'use_vectorbt': True
            }
        
        elif endpoint_type == 'optimization':
            return {
                'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL']),
                'strategy_template': 'sma_crossover',
                'n_trials': random.randint(10, 50),
                'walk_forward': random.choice([True, False])
            }
        
        elif endpoint_type == 'chat':
            queries = [
                "What's the market outlook for AAPL?",
                "Analyze the sentiment for tech stocks",
                "Should I buy or sell Tesla?",
                "What are the best dividend stocks?",
                "Explain the RSI indicator"
            ]
            return {'message': random.choice(queries)}
        
        return None
    
    def _generate_ohlcv_data(self) -> List[Dict[str, Any]]:
        """Generate sample OHLCV data"""
        bars = []
        base_price = random.uniform(100, 500)
        
        for i in range(100):
            timestamp = datetime.now() - timedelta(days=100-i)
            
            # Generate realistic OHLCV data
            open_price = base_price * random.uniform(0.98, 1.02)
            close_price = open_price * random.uniform(0.95, 1.05)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.03)
            low_price = min(open_price, close_price) * random.uniform(0.97, 1.0)
            volume = random.randint(100000, 1000000)
            
            bars.append({
                'timestamp': timestamp.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            base_price = close_price
        
        return bars
    
    def _get_default_endpoints(self) -> List[Dict[str, Any]]:
        """Get default endpoint configuration"""
        return [
            {'path': '/health', 'method': 'GET', 'weight': 10},
            {'path': '/performance/health', 'method': 'GET', 'weight': 5},
            {'path': '/sentiment/historical/AAPL', 'method': 'GET', 'type': 'sentiment', 'weight': 8},
            {'path': '/backtest', 'method': 'POST', 'type': 'backtest', 'weight': 3},
            {'path': '/chat', 'method': 'POST', 'type': 'chat', 'weight': 7},
            {'path': '/optimize/strategy', 'method': 'POST', 'type': 'optimization', 'weight': 2},
            {'path': '/advice', 'method': 'POST', 'type': 'advice', 'weight': 6}
        ]
    
    def _calculate_final_metrics(self):
        """Calculate final load test metrics"""
        if self.response_times:
            self.metrics.avg_response_time = statistics.mean(self.response_times)
            self.metrics.min_response_time = min(self.response_times)
            self.metrics.max_response_time = max(self.response_times)
            self.metrics.calculate_percentiles(self.response_times)
        
        if self.metrics.duration_seconds > 0:
            self.metrics.requests_per_second = self.metrics.total_requests / self.metrics.duration_seconds
        
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests

class SuperNovaLoadTestUser(HttpUser):
    """Locust user class for SuperNova load testing"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        self.strategies = ['sma_crossover', 'rsi_strategy', 'macd_strategy', 'bb_strategy']
    
    @task(10)
    def health_check(self):
        """Health check endpoint - high frequency"""
        self.client.get("/health")
    
    @task(8)
    def get_sentiment_data(self):
        """Get sentiment data - medium-high frequency"""
        symbol = random.choice(self.symbols)
        params = {
            'start_date': (datetime.now() - timedelta(days=7)).isoformat(),
            'end_date': datetime.now().isoformat(),
            'interval': random.choice(['1h', '6h', '1d']),
            'limit': random.randint(100, 500)
        }
        
        with self.client.get(f"/sentiment/historical/{symbol}", params=params, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(6)
    def get_advice(self):
        """Get trading advice - medium frequency"""
        data = {
            'profile_id': random.randint(1, 100),
            'bars': self._generate_sample_bars(),
            'symbol': random.choice(self.symbols),
            'timeframe': '1h',
            'sentiment_hint': random.uniform(-1, 1),
            'strategy_template': random.choice(self.strategies)
        }
        
        with self.client.post("/advice", json=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def chat_interaction(self):
        """Chat interaction - medium frequency"""
        queries = [
            f"What's the sentiment for {random.choice(self.symbols)}?",
            "Should I buy tech stocks now?",
            "What are the market trends today?",
            "Analyze the crypto market",
            "Give me investment advice"
        ]
        
        data = {
            'message': random.choice(queries),
            'profile_id': random.randint(1, 100)
        }
        
        with self.client.post("/chat", json=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def run_backtest(self):
        """Run backtest - lower frequency (expensive operation)"""
        data = {
            'bars': self._generate_sample_bars(50),  # Smaller dataset for load testing
            'strategy_template': random.choice(self.strategies),
            'symbol': random.choice(self.symbols),
            'timeframe': '1h',
            'params': {
                'fast_period': random.randint(5, 15),
                'slow_period': random.randint(20, 40),
                'rsi_period': random.randint(10, 20),
                'rsi_upper': random.randint(70, 80),
                'rsi_lower': random.randint(20, 30)
            },
            'start_cash': 10000,
            'use_vectorbt': True
        }
        
        with self.client.post("/backtest", json=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def optimization_request(self):
        """Optimization request - lowest frequency (very expensive)"""
        data = {
            'symbol': random.choice(self.symbols),
            'strategy_template': random.choice(self.strategies),
            'n_trials': random.randint(5, 20),  # Small for load testing
            'bars': self._generate_sample_bars(30),
            'primary_objective': 'sharpe_ratio',
            'walk_forward': False
        }
        
        with self.client.post("/optimize/strategy", json=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(4)
    def performance_metrics(self):
        """Get performance metrics"""
        endpoints = [
            "/performance/health",
            "/performance/metrics/resources",
            "/performance/metrics/api",
            "/performance/metrics/cache"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    def _generate_sample_bars(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate sample OHLCV data for testing"""
        bars = []
        base_price = random.uniform(100, 500)
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(hours=count-i)
            
            open_price = base_price * random.uniform(0.99, 1.01)
            close_price = open_price * random.uniform(0.97, 1.03)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
            
            bars.append({
                'timestamp': timestamp.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(10000, 100000)
            })
            
            base_price = close_price
        
        return bars

class LoadTestRunner:
    """Main load test runner with multiple testing strategies"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.results: Dict[str, Any] = {}
        
    async def run_async_load_test(
        self,
        max_users: int = 50,
        duration: int = 120,
        spawn_rate: int = 5
    ) -> LoadTestMetrics:
        """Run async load test"""
        config = LoadTestConfig(
            base_url=self.base_url,
            max_users=max_users,
            test_duration=duration,
            spawn_rate=spawn_rate
        )
        
        tester = AsyncHTTPLoadTester(config)
        metrics = await tester.run_load_test()
        
        self.results['async_load_test'] = {
            'metrics': metrics,
            'errors': tester.errors
        }
        
        return metrics
    
    def run_locust_load_test(
        self,
        max_users: int = 100,
        spawn_rate: float = 10,
        duration: int = 300
    ) -> Dict[str, Any]:
        """Run Locust-based load test"""
        if not LOCUST_AVAILABLE:
            raise RuntimeError("Locust not available. Install with: pip install locust")
        
        logger.info(f"Starting Locust load test - {max_users} users, {duration}s duration")
        
        # Setup Locust environment
        setup_logging("INFO", None)
        env = Environment(user_classes=[SuperNovaLoadTestUser])
        
        # Configure stats collection
        stats_history.enable(csv_writer=True)
        
        # Start load test
        env.create_local_runner()
        env.runner.start(max_users, spawn_rate=spawn_rate)
        
        # Run for specified duration
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            time.sleep(1)
        
        # Stop test
        env.runner.quit()
        
        # Collect results
        stats = env.stats
        results = {
            'total_requests': stats.total.num_requests,
            'total_failures': stats.total.num_failures,
            'avg_response_time': stats.total.avg_response_time,
            'min_response_time': stats.total.min_response_time,
            'max_response_time': stats.total.max_response_time,
            'requests_per_second': stats.total.total_rps,
            'failure_rate': stats.total.fail_ratio,
            'endpoint_stats': {}
        }
        
        # Collect per-endpoint stats
        for endpoint, stat in stats.entries.items():
            results['endpoint_stats'][f"{endpoint.method} {endpoint.name}"] = {
                'requests': stat.num_requests,
                'failures': stat.num_failures,
                'avg_response_time': stat.avg_response_time,
                'min_response_time': stat.min_response_time,
                'max_response_time': stat.max_response_time,
                'rps': stat.total_rps
            }
        
        self.results['locust_load_test'] = results
        logger.info("Locust load test completed")
        
        return results
    
    async def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test to find breaking points"""
        logger.info("Starting stress test to find system limits")
        
        stress_results = []
        user_counts = [10, 25, 50, 100, 200, 500, 1000]
        
        for user_count in user_counts:
            logger.info(f"Stress testing with {user_count} concurrent users")
            
            try:
                metrics = await self.run_async_load_test(
                    max_users=user_count,
                    duration=60,  # Shorter duration for stress test
                    spawn_rate=user_count // 5  # Aggressive spawn rate
                )
                
                stress_results.append({
                    'user_count': user_count,
                    'successful': True,
                    'requests_per_second': metrics.requests_per_second,
                    'avg_response_time': metrics.avg_response_time,
                    'error_rate': metrics.error_rate,
                    'p95_response_time': metrics.p95_response_time
                })
                
                # Check if system is failing
                if metrics.error_rate > 0.1 or metrics.p95_response_time > 10000:  # 10s
                    logger.warning(f"System degradation detected at {user_count} users")
                    break
                    
            except Exception as e:
                logger.error(f"Stress test failed at {user_count} users: {e}")
                stress_results.append({
                    'user_count': user_count,
                    'successful': False,
                    'error': str(e)
                })
                break
        
        # Analyze results to find capacity limits
        successful_tests = [r for r in stress_results if r.get('successful', False)]
        
        capacity_analysis = {
            'max_stable_users': 0,
            'breaking_point': 0,
            'recommended_capacity': 0,
            'performance_degradation_threshold': 0
        }
        
        if successful_tests:
            # Find maximum stable user count (error rate < 1%, response time reasonable)
            stable_tests = [
                r for r in successful_tests 
                if r['error_rate'] < 0.01 and r['avg_response_time'] < 2000
            ]
            
            if stable_tests:
                capacity_analysis['max_stable_users'] = max(r['user_count'] for r in stable_tests)
                capacity_analysis['recommended_capacity'] = int(capacity_analysis['max_stable_users'] * 0.7)  # 70% of max
            
            capacity_analysis['breaking_point'] = max(r['user_count'] for r in successful_tests)
        
        self.results['stress_test'] = {
            'test_results': stress_results,
            'capacity_analysis': capacity_analysis
        }
        
        logger.info(f"Stress test completed. Max stable users: {capacity_analysis['max_stable_users']}")
        
        return self.results['stress_test']
    
    async def run_endurance_test(self, duration: int = 3600) -> Dict[str, Any]:
        """Run endurance test for sustained load"""
        logger.info(f"Starting endurance test - {duration} seconds duration")
        
        # Use moderate load for endurance testing
        user_count = 50
        metrics = await self.run_async_load_test(
            max_users=user_count,
            duration=duration,
            spawn_rate=5
        )
        
        # Monitor resource usage during test
        resource_samples = []
        if PSUTIL_AVAILABLE:
            sample_count = duration // 30  # Sample every 30 seconds
            for _ in range(sample_count):
                resource_samples.append({
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                })
                await asyncio.sleep(29)  # Wait for next sample
        
        self.results['endurance_test'] = {
            'duration_seconds': duration,
            'user_count': user_count,
            'metrics': metrics,
            'resource_samples': resource_samples
        }
        
        logger.info("Endurance test completed")
        
        return self.results['endurance_test']
    
    def generate_report(self, output_file: str = "load_test_report.json"):
        """Generate comprehensive load test report"""
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'base_url': self.base_url,
                'tests_run': list(self.results.keys())
            },
            'results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Load test report saved to {output_file}")
        
        # Print summary
        self._print_summary()
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance recommendations based on test results"""
        recommendations = []
        
        # Analyze async load test results
        if 'async_load_test' in self.results:
            metrics = self.results['async_load_test']['metrics']
            
            if hasattr(metrics, 'avg_response_time') and metrics.avg_response_time > 1000:
                recommendations.append({
                    'category': 'Response Time',
                    'priority': 'HIGH',
                    'issue': f'Average response time is {metrics.avg_response_time:.0f}ms',
                    'recommendation': 'Optimize slow endpoints, add caching, or scale horizontally'
                })
            
            if hasattr(metrics, 'error_rate') and metrics.error_rate > 0.05:
                recommendations.append({
                    'category': 'Reliability',
                    'priority': 'CRITICAL',
                    'issue': f'Error rate is {metrics.error_rate:.2%}',
                    'recommendation': 'Investigate and fix errors causing failures'
                })
        
        # Analyze stress test results
        if 'stress_test' in self.results:
            capacity = self.results['stress_test']['capacity_analysis']
            
            if capacity['max_stable_users'] < 100:
                recommendations.append({
                    'category': 'Scalability',
                    'priority': 'HIGH',
                    'issue': f'System can only handle {capacity["max_stable_users"]} concurrent users',
                    'recommendation': 'Consider horizontal scaling, database optimization, or caching improvements'
                })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Monitoring',
                'priority': 'MEDIUM',
                'issue': 'Continuous performance monitoring needed',
                'recommendation': 'Implement real-time performance monitoring and alerting'
            },
            {
                'category': 'Optimization',
                'priority': 'MEDIUM',
                'issue': 'Regular performance testing needed',
                'recommendation': 'Schedule regular load testing to catch performance regressions'
            }
        ])
        
        return recommendations
    
    def _print_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*60)
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            
            if test_name == 'async_load_test':
                metrics = results['metrics']
                if hasattr(metrics, 'total_requests'):
                    print(f"  Total Requests: {metrics.total_requests}")
                    print(f"  Success Rate: {(1-metrics.error_rate):.1%}")
                    print(f"  Avg Response Time: {metrics.avg_response_time:.0f}ms")
                    print(f"  P95 Response Time: {metrics.p95_response_time:.0f}ms")
                    print(f"  Requests/Second: {metrics.requests_per_second:.1f}")
            
            elif test_name == 'stress_test':
                capacity = results['capacity_analysis']
                print(f"  Max Stable Users: {capacity['max_stable_users']}")
                print(f"  Breaking Point: {capacity['breaking_point']}")
                print(f"  Recommended Capacity: {capacity['recommended_capacity']}")
            
            elif test_name == 'locust_load_test':
                print(f"  Total Requests: {results['total_requests']}")
                print(f"  Failure Rate: {results['failure_rate']:.1%}")
                print(f"  Avg Response Time: {results['avg_response_time']:.0f}ms")
                print(f"  Requests/Second: {results['requests_per_second']:.1f}")
        
        print("\n" + "="*60)

# Convenience functions
async def run_quick_load_test(base_url: str = "http://localhost:8081", users: int = 20, duration: int = 60):
    """Run a quick load test"""
    runner = LoadTestRunner(base_url)
    metrics = await runner.run_async_load_test(max_users=users, duration=duration)
    return metrics

async def run_comprehensive_load_test(base_url: str = "http://localhost:8081"):
    """Run comprehensive load test suite"""
    runner = LoadTestRunner(base_url)
    
    # Run all test types
    print("Running async load test...")
    await runner.run_async_load_test(max_users=100, duration=300)
    
    print("Running stress test...")
    await runner.run_stress_test()
    
    if LOCUST_AVAILABLE:
        print("Running Locust load test...")
        runner.run_locust_load_test(max_users=50, duration=180)
    
    # Generate report
    report = runner.generate_report("comprehensive_load_test_report.json")
    
    return report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        # Run comprehensive test
        asyncio.run(run_comprehensive_load_test())
    else:
        # Run quick test
        asyncio.run(run_quick_load_test(users=10, duration=30))