"""
Performance Testing Suite with Locust
=====================================

Load testing and performance validation for SuperNova AI financial platform
using Locust for realistic user behavior simulation.
"""
import time
import random
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta

from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
import numpy as np


class FinancialPlatformUser(HttpUser):
    """Simulate realistic user behavior on the financial platform."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def on_start(self):
        """Initialize user session."""
        self.user_id = None
        self.profile_id = None
        self.portfolios = []
        self.watchlist_items = []
        self.auth_token = None
        
        # Create user profile
        self.create_user_profile()
    
    def create_user_profile(self):
        """Create a user profile for testing."""
        user_data = {
            "name": f"LoadTestUser_{random.randint(1000, 9999)}",
            "email": f"loadtest_{random.randint(1000, 9999)}@example.com",
            "income": random.uniform(50000, 200000),
            "risk_questions": [random.randint(1, 5) for _ in range(5)]
        }
        
        with self.client.post("/intake", json=user_data, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                self.profile_id = data.get("profile_id")
                response.success()
            else:
                response.failure(f"Failed to create user profile: {response.status_code}")
    
    @task(3)
    def get_portfolio_advice(self):
        """Request portfolio advice - high frequency task."""
        if not self.profile_id:
            raise RescheduleTask()
        
        advice_data = {
            "profile_id": self.profile_id,
            "symbols": random.sample(["VTI", "BND", "VEA", "VWO", "QQQ", "SPY"], k=random.randint(3, 5))
        }
        
        start_time = time.time()
        with self.client.post("/advice", json=advice_data, catch_response=True) as response:
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                # Validate response structure
                data = response.json()
                if "allocations" in data and "risk_metrics" in data:
                    response.success()
                    
                    # Performance assertion
                    if elapsed_time > 3.0:  # 3 second threshold
                        events.request.fire(
                            request_type="POST",
                            name="/advice (SLOW)",
                            response_time=elapsed_time * 1000,
                            response_length=len(response.content),
                            exception=f"Slow response: {elapsed_time:.2f}s"
                        )
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Failed to get advice: {response.status_code}")
    
    @task(2)
    def manage_watchlist(self):
        """Add/remove items from watchlist."""
        if not self.profile_id:
            raise RescheduleTask()
        
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "BRK.B"]
        action = random.choice(["add", "remove"])
        
        if action == "add" or not self.watchlist_items:
            # Add to watchlist
            symbol = random.choice(symbols)
            watchlist_data = {
                "profile_id": self.profile_id,
                "symbol": symbol
            }
            
            with self.client.post("/watchlist", json=watchlist_data, catch_response=True) as response:
                if response.status_code == 200:
                    self.watchlist_items.append(symbol)
                    response.success()
                else:
                    response.failure(f"Failed to add to watchlist: {response.status_code}")
        
        else:
            # Remove from watchlist
            if self.watchlist_items:
                symbol = random.choice(self.watchlist_items)
                with self.client.delete(f"/watchlist/{symbol}", catch_response=True) as response:
                    if response.status_code == 200:
                        self.watchlist_items.remove(symbol)
                        response.success()
                    else:
                        response.failure(f"Failed to remove from watchlist: {response.status_code}")
    
    @task(1)
    def run_backtest(self):
        """Run backtesting - intensive task."""
        if not self.profile_id:
            raise RescheduleTask()
        
        backtest_data = {
            "profile_id": self.profile_id,
            "symbols": random.sample(["VTI", "VEA", "VWO", "BND"], k=3),
            "start_date": (datetime.now() - timedelta(days=random.randint(365, 1095))).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_capital": random.uniform(10000, 100000)
        }
        
        start_time = time.time()
        with self.client.post("/backtest", json=backtest_data, catch_response=True) as response:
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "performance" in data and "metrics" in data:
                    response.success()
                    
                    # Performance assertion for intensive tasks
                    if elapsed_time > 10.0:  # 10 second threshold for backtesting
                        events.request.fire(
                            request_type="POST",
                            name="/backtest (SLOW)",
                            response_time=elapsed_time * 1000,
                            response_length=len(response.content),
                            exception=f"Slow backtest: {elapsed_time:.2f}s"
                        )
                else:
                    response.failure("Invalid backtest response structure")
            else:
                response.failure(f"Backtest failed: {response.status_code}")
    
    @task(2)
    def get_market_data(self):
        """Fetch market data - moderate frequency."""
        symbols = ["VTI", "SPY", "QQQ", "BND", "GLD", "VEA"]
        symbol = random.choice(symbols)
        
        start_time = time.time()
        with self.client.get(f"/market-data/{symbol}", catch_response=True) as response:
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and len(data["bars"]) > 0:
                    response.success()
                    
                    # Fast response expected for market data
                    if elapsed_time > 1.0:  # 1 second threshold
                        events.request.fire(
                            request_type="GET",
                            name=f"/market-data/{symbol} (SLOW)",
                            response_time=elapsed_time * 1000,
                            response_length=len(response.content),
                            exception=f"Slow market data: {elapsed_time:.2f}s"
                        )
                else:
                    response.failure("Empty or invalid market data response")
            else:
                response.failure(f"Market data request failed: {response.status_code}")
    
    @task(1)
    def get_analytics_insights(self):
        """Fetch analytics insights."""
        if not self.profile_id:
            raise RescheduleTask()
        
        with self.client.get(f"/analytics/insights/{self.profile_id}", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "insights" in data:
                    response.success()
                else:
                    response.failure("Invalid analytics response structure")
            else:
                response.failure(f"Analytics request failed: {response.status_code}")
    
    @task(1)
    def sentiment_analysis(self):
        """Get sentiment analysis for symbols."""
        symbols = ["AAPL", "TSLA", "NVDA", "AMD", "PLTR"]
        symbol = random.choice(symbols)
        
        with self.client.get(f"/sentiment/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "sentiment_score" in data:
                    response.success()
                else:
                    response.failure("Invalid sentiment response structure")
            else:
                response.failure(f"Sentiment analysis failed: {response.status_code}")


class DataIntensiveUser(HttpUser):
    """Simulate users performing data-intensive operations."""
    
    weight = 1  # Lower weight, fewer of these users
    wait_time = between(5, 15)  # Longer wait times for intensive operations
    
    def on_start(self):
        self.profile_id = None
        self.create_user_profile()
    
    def create_user_profile(self):
        """Create user profile for data-intensive testing."""
        user_data = {
            "name": f"DataIntensiveUser_{random.randint(1000, 9999)}",
            "email": f"dataintensive_{random.randint(1000, 9999)}@example.com",
            "income": random.uniform(100000, 500000),
            "risk_questions": [random.randint(3, 5) for _ in range(5)]  # Higher risk tolerance
        }
        
        with self.client.post("/intake", json=user_data, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                self.profile_id = data.get("profile_id")
                response.success()
            else:
                response.failure(f"Failed to create data intensive user: {response.status_code}")
    
    @task(2)
    def large_portfolio_analysis(self):
        """Analyze large portfolios with many assets."""
        if not self.profile_id:
            raise RescheduleTask()
        
        # Large number of symbols for stress testing
        all_symbols = [
            "VTI", "VXUS", "BND", "BNDX", "VEA", "VWO", "VNQ", "VNQI",
            "QQQ", "SPY", "IWM", "EFA", "EEM", "AGG", "LQD", "HYG",
            "GLD", "SLV", "DBA", "USO", "TLT", "IEF", "SHY", "VTEB"
        ]
        
        large_portfolio = {
            "profile_id": self.profile_id,
            "symbols": random.sample(all_symbols, k=random.randint(15, 20)),
            "analysis_type": "comprehensive"
        }
        
        start_time = time.time()
        with self.client.post("/portfolio/analyze", json=large_portfolio, catch_response=True) as response:
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                response.success()
                
                # Performance tracking for large analyses
                if elapsed_time > 30.0:  # 30 second threshold
                    events.request.fire(
                        request_type="POST",
                        name="/portfolio/analyze (VERY SLOW)",
                        response_time=elapsed_time * 1000,
                        response_length=len(response.content),
                        exception=f"Very slow analysis: {elapsed_time:.2f}s"
                    )
            else:
                response.failure(f"Large portfolio analysis failed: {response.status_code}")
    
    @task(1)
    def historical_data_analysis(self):
        """Request large amounts of historical data."""
        symbol = random.choice(["VTI", "SPY", "QQQ"])
        
        # Request 5+ years of data
        start_date = datetime.now() - timedelta(days=random.randint(1800, 2500))
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "resolution": "1D"
        }
        
        start_time = time.time()
        with self.client.get(f"/market-data/{symbol}/historical", 
                           params=params, catch_response=True) as response:
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and len(data["bars"]) > 1000:  # Expect significant data
                    response.success()
                else:
                    response.failure("Insufficient historical data returned")
                
                # Track performance of data-heavy requests
                if elapsed_time > 5.0:
                    events.request.fire(
                        request_type="GET",
                        name=f"/market-data/{symbol}/historical (SLOW)",
                        response_time=elapsed_time * 1000,
                        response_length=len(response.content),
                        exception=f"Slow historical data: {elapsed_time:.2f}s"
                    )
            else:
                response.failure(f"Historical data request failed: {response.status_code}")


class APIStressUser(HttpUser):
    """Simulate high-frequency API usage for stress testing."""
    
    weight = 2
    wait_time = between(0.1, 1.0)  # Very short wait times for stress testing
    
    @task(5)
    def rapid_health_checks(self):
        """Rapidly hit health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(3)
    def rapid_quotes(self):
        """Rapidly request quote data."""
        symbol = random.choice(["VTI", "SPY", "QQQ", "BND"])
        
        with self.client.get(f"/quote/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:  # Rate limited
                response.failure("Rate limited - may need to adjust rate limiting")
            else:
                response.failure(f"Quote request failed: {response.status_code}")
    
    @task(1)
    def concurrent_calculations(self):
        """Submit multiple calculation requests rapidly."""
        for _ in range(3):  # Multiple rapid requests
            calculation_data = {
                "operation": "portfolio_metrics",
                "data": {
                    "returns": [random.uniform(-0.05, 0.05) for _ in range(252)],
                    "risk_free_rate": 0.02
                }
            }
            
            with self.client.post("/calculate", json=calculation_data, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Calculation failed: {response.status_code}")
            
            time.sleep(0.1)  # Brief pause between requests


# Custom event listeners for performance monitoring
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom performance monitoring."""
    if response_time > 5000:  # 5 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")
    
    if exception:
        print(f"REQUEST FAILED: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize performance monitoring."""
    print("=== SuperNova Performance Test Started ===")
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate performance report."""
    stats = environment.runner.stats
    
    print("=== SuperNova Performance Test Results ===")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")
    
    # Performance thresholds
    performance_issues = []
    
    if stats.total.avg_response_time > 2000:  # 2 seconds average
        performance_issues.append(f"High average response time: {stats.total.avg_response_time:.2f}ms")
    
    if stats.total.get_response_time_percentile(0.95) > 5000:  # 5 seconds 95th percentile
        performance_issues.append(f"High 95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    
    if stats.total.num_failures / stats.total.num_requests > 0.01:  # >1% failure rate
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        performance_issues.append(f"High failure rate: {failure_rate:.2f}%")
    
    if performance_issues:
        print("\n⚠️ PERFORMANCE ISSUES DETECTED:")
        for issue in performance_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All performance thresholds met!")
    
    # Save detailed results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_requests": stats.total.num_requests,
        "total_failures": stats.total.num_failures,
        "avg_response_time": stats.total.avg_response_time,
        "p95_response_time": stats.total.get_response_time_percentile(0.95),
        "p99_response_time": stats.total.get_response_time_percentile(0.99),
        "requests_per_second": stats.total.current_rps,
        "performance_issues": performance_issues,
        "detailed_stats": {}
    }
    
    # Add per-endpoint stats
    for name, endpoint_stats in stats.entries.items():
        if endpoint_stats.num_requests > 0:
            results["detailed_stats"][name[1]] = {  # name[1] is the endpoint path
                "requests": endpoint_stats.num_requests,
                "failures": endpoint_stats.num_failures,
                "avg_response_time": endpoint_stats.avg_response_time,
                "p95_response_time": endpoint_stats.get_response_time_percentile(0.95),
                "rps": endpoint_stats.current_rps
            }
    
    with open("performance_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: performance_test_results.json")


# Locust configuration for different test scenarios
class QuickTestUser(HttpUser):
    """Quick smoke test user for CI/CD pipeline."""
    
    tasks = [FinancialPlatformUser.get_portfolio_advice, FinancialPlatformUser.get_market_data]
    wait_time = between(1, 2)
    
    def on_start(self):
        self.profile_id = 1  # Use default test profile


if __name__ == "__main__":
    # Example usage:
    # locust -f test_performance_locust.py --host=http://localhost:8000
    # locust -f test_performance_locust.py --host=http://localhost:8000 --users=50 --spawn-rate=5 --run-time=300s
    pass