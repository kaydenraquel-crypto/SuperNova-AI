"""
Comprehensive Performance and Load Testing Suite
===============================================

This module provides extensive performance testing including:
- API endpoint load testing
- Database performance testing
- Memory usage and leak detection
- Concurrent user simulation
- Response time benchmarks
- Resource utilization monitoring
"""
import pytest
import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import tracemalloc
from typing import List, Dict, Any, Optional
import statistics
import json
from datetime import datetime, timedelta
import numpy as np

from fastapi.testclient import TestClient
import requests
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from supernova.api import app
from supernova.db import SessionLocal, Base
from supernova.schemas import OHLCVBar


class PerformanceMetrics:
    """Class to track and analyze performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.error_count: int = 0
        self.success_count: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def add_response_time(self, time: float):
        self.response_times.append(time)
    
    def add_memory_usage(self, usage: float):
        self.memory_usage.append(usage)
    
    def add_cpu_usage(self, usage: float):
        self.cpu_usage.append(usage)
    
    def increment_success(self):
        self.success_count += 1
    
    def increment_error(self):
        self.error_count += 1
    
    def start_timer(self):
        self.start_time = time.time()
    
    def stop_timer(self):
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        total_requests = self.success_count + self.error_count
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'success_rate': (self.success_count / total_requests * 100) if total_requests > 0 else 0,
            'duration_seconds': duration,
            'requests_per_second': total_requests / duration if duration > 0 else 0,
            'response_times': {
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0,
                'mean': statistics.mean(self.response_times) if self.response_times else 0,
                'median': statistics.median(self.response_times) if self.response_times else 0,
                'p95': np.percentile(self.response_times, 95) if self.response_times else 0,
                'p99': np.percentile(self.response_times, 99) if self.response_times else 0,
                'std_dev': statistics.stdev(self.response_times) if len(self.response_times) > 1 else 0,
            },
            'memory_usage': {
                'min_mb': min(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
                'max_mb': max(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
                'mean_mb': statistics.mean(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
            },
            'cpu_usage': {
                'min_percent': min(self.cpu_usage) if self.cpu_usage else 0,
                'max_percent': max(self.cpu_usage) if self.cpu_usage else 0,
                'mean_percent': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            }
        }


class LoadTestRunner:
    """Load test runner for API endpoints."""
    
    def __init__(self, base_url: str = "http://testserver"):
        self.base_url = base_url
        self.metrics = PerformanceMetrics()
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a single HTTP request and record metrics."""
        start_time = time.time()
        
        try:
            with TestClient(app) as client:
                if method.upper() == 'GET':
                    response = client.get(endpoint, headers=headers)
                elif method.upper() == 'POST':
                    response = client.post(endpoint, json=data, headers=headers)
                elif method.upper() == 'PUT':
                    response = client.put(endpoint, json=data, headers=headers)
                elif method.upper() == 'DELETE':
                    response = client.delete(endpoint, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)
                
                # Record memory and CPU usage
                process = psutil.Process()
                memory_info = process.memory_info()
                self.metrics.add_memory_usage(memory_info.rss)
                self.metrics.add_cpu_usage(process.cpu_percent())
                
                if response.status_code < 400:
                    self.metrics.increment_success()
                else:
                    self.metrics.increment_error()
                
                return {
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'response_size': len(response.content) if hasattr(response, 'content') else 0
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.add_response_time(response_time)
            self.metrics.increment_error()
            
            return {
                'status_code': 0,
                'response_time': response_time,
                'error': str(e)
            }
    
    def run_concurrent_load_test(self, method: str, endpoint: str, concurrent_users: int, 
                                requests_per_user: int, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Run concurrent load test with multiple users."""
        self.metrics.start_timer()
        
        def user_session(user_id: int) -> List[Dict[str, Any]]:
            """Simulate a user session with multiple requests."""
            results = []
            for i in range(requests_per_user):
                result = self.make_request(method, endpoint, data)
                result['user_id'] = user_id
                result['request_number'] = i + 1
                results.append(result)
                
                # Add small delay between requests from same user
                time.sleep(0.1)
            
            return results
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            future_to_user = {
                executor.submit(user_session, user_id): user_id 
                for user_id in range(concurrent_users)
            }
            
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    print(f"User {user_id} generated exception: {e}")
                    self.metrics.increment_error()
        
        self.metrics.stop_timer()
        
        return {
            'test_config': {
                'method': method,
                'endpoint': endpoint,
                'concurrent_users': concurrent_users,
                'requests_per_user': requests_per_user,
                'total_requests': concurrent_users * requests_per_user
            },
            'metrics': self.metrics.get_summary(),
            'detailed_results': all_results
        }


@pytest.mark.performance
class TestAPIPerformance:
    """Test API endpoint performance under various loads."""
    
    def test_intake_endpoint_performance(self, sample_ohlcv_data):
        """Test intake endpoint performance."""
        runner = LoadTestRunner()
        
        test_data = {
            "name": "Performance Test User",
            "email": "perf@test.com",
            "risk_questions": [3, 3, 2, 3, 2]
        }
        
        # Single user baseline
        result = runner.run_concurrent_load_test(
            'POST', '/intake', 1, 10, test_data
        )
        
        metrics = result['metrics']
        assert metrics['success_rate'] > 95
        assert metrics['response_times']['mean'] < 1.0  # Should be under 1 second
        assert metrics['requests_per_second'] > 5
    
    def test_advice_endpoint_performance(self, sample_ohlcv_data):
        """Test advice endpoint performance with realistic data."""
        runner = LoadTestRunner()
        
        # Create a test profile first
        profile_data = {"name": "Perf User", "email": "perf@test.com"}
        profile_response = runner.make_request('POST', '/intake', profile_data)
        
        test_data = {
            "profile_id": 1,  # Assume first profile
            "symbol": "PERF_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:100]]
        }
        
        result = runner.run_concurrent_load_test(
            'POST', '/advice', 5, 5, test_data
        )
        
        metrics = result['metrics']
        assert metrics['success_rate'] > 90
        assert metrics['response_times']['mean'] < 3.0  # Should be under 3 seconds
        assert metrics['response_times']['p95'] < 5.0   # 95th percentile under 5 seconds
    
    def test_backtest_endpoint_performance(self, sample_ohlcv_data):
        """Test backtest endpoint performance."""
        runner = LoadTestRunner()
        
        test_data = {
            "strategy_template": "ma_crossover",
            "params": {"fast": 10, "slow": 20},
            "symbol": "PERF_BACKTEST",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        result = runner.run_concurrent_load_test(
            'POST', '/backtest', 3, 3, test_data
        )
        
        metrics = result['metrics']
        assert metrics['success_rate'] > 85
        assert metrics['response_times']['mean'] < 10.0  # Should be under 10 seconds
        assert metrics['response_times']['p99'] < 30.0   # 99th percentile under 30 seconds
    
    def test_concurrent_user_simulation(self, sample_ohlcv_data):
        """Test system behavior under concurrent user load."""
        runner = LoadTestRunner()
        
        # Simulate realistic user workflow
        def user_workflow(user_id: int) -> Dict[str, Any]:
            workflow_results = {'user_id': user_id, 'steps': []}
            
            # Step 1: User intake
            intake_data = {
                "name": f"Concurrent User {user_id}",
                "email": f"user{user_id}@test.com",
                "risk_questions": [2, 3, 2, 3, 3]
            }
            
            intake_result = runner.make_request('POST', '/intake', intake_data)
            workflow_results['steps'].append({'step': 'intake', 'result': intake_result})
            
            if intake_result['status_code'] == 200:
                # Step 2: Get advice
                advice_data = {
                    "profile_id": user_id,  # Simplified assumption
                    "symbol": f"USER_{user_id}_TEST",
                    "bars": [bar.model_dump() for bar in sample_ohlcv_data[:50]]
                }
                
                advice_result = runner.make_request('POST', '/advice', advice_data)
                workflow_results['steps'].append({'step': 'advice', 'result': advice_result})
            
            return workflow_results
        
        # Test with multiple concurrent users
        concurrent_users = 20
        workflow_results = []
        
        runner.metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(user_workflow, user_id) 
                for user_id in range(concurrent_users)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    workflow_results.append(result)
                except Exception as e:
                    runner.metrics.increment_error()
        
        runner.metrics.stop_timer()
        
        # Analyze workflow results
        successful_workflows = sum(
            1 for result in workflow_results 
            if all(step['result']['status_code'] == 200 for step in result['steps'])
        )
        
        success_rate = (successful_workflows / concurrent_users) * 100
        
        assert success_rate > 80  # At least 80% of workflows should succeed
        assert runner.metrics.get_summary()['duration_seconds'] < 60  # Should complete in under 1 minute
    
    def test_data_size_performance_scaling(self):
        """Test how performance scales with different data sizes."""
        runner = LoadTestRunner()
        
        # Test with different bar counts
        data_sizes = [10, 50, 100, 500, 1000]
        results = {}
        
        for size in data_sizes:
            # Generate OHLCV data of specified size
            bars = []
            for i in range(size):
                bars.append(OHLCVBar(
                    timestamp=(datetime.now() - timedelta(hours=size-i)).isoformat() + "Z",
                    open=100 + (i * 0.1),
                    high=101 + (i * 0.1),
                    low=99 + (i * 0.1),
                    close=100.5 + (i * 0.1),
                    volume=10000
                ))
            
            test_data = {
                "profile_id": 1,
                "symbol": f"SIZE_TEST_{size}",
                "bars": [bar.model_dump() for bar in bars]
            }
            
            # Run single request test
            start_time = time.time()
            result = runner.make_request('POST', '/advice', test_data)
            response_time = time.time() - start_time
            
            results[size] = {
                'response_time': response_time,
                'status_code': result['status_code'],
                'data_size_kb': len(str(test_data)) / 1024
            }
        
        # Verify performance doesn't degrade exponentially
        for i in range(1, len(data_sizes)):
            current_size = data_sizes[i]
            previous_size = data_sizes[i-1]
            
            current_time = results[current_size]['response_time']
            previous_time = results[previous_size]['response_time']
            
            # Response time shouldn't increase more than 3x for 10x data
            size_ratio = current_size / previous_size
            time_ratio = current_time / previous_time if previous_time > 0 else 1
            
            assert time_ratio < (size_ratio * 0.5), f"Performance degrades too quickly: {time_ratio} vs {size_ratio}"
    
    @pytest.mark.slow
    def test_sustained_load_endurance(self, sample_ohlcv_data):
        """Test system behavior under sustained load."""
        runner = LoadTestRunner()
        
        test_data = {
            "profile_id": 1,
            "symbol": "ENDURANCE_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data[:50]]
        }
        
        # Run sustained load for 5 minutes
        duration_minutes = 5
        requests_per_minute = 30
        total_requests = duration_minutes * requests_per_minute
        
        runner.metrics.start_timer()
        
        for i in range(total_requests):
            runner.make_request('POST', '/advice', test_data)
            
            # Control request rate
            if i < total_requests - 1:  # Don't sleep after last request
                time.sleep(60.0 / requests_per_minute)  # Space requests evenly
        
        runner.metrics.stop_timer()
        
        metrics = runner.metrics.get_summary()
        
        # Performance shouldn't degrade significantly over time
        assert metrics['success_rate'] > 95
        assert metrics['response_times']['mean'] < 2.0
        
        # Memory usage shouldn't grow unboundedly (check for leaks)
        memory_stats = metrics['memory_usage']
        memory_growth = memory_stats['max_mb'] - memory_stats['min_mb']
        assert memory_growth < 100  # Less than 100MB growth over test


@pytest.mark.performance  
class TestDatabasePerformance:
    """Test database performance and connection handling."""
    
    def test_database_connection_pool_performance(self):
        """Test database connection pool under load."""
        def execute_query(query_id: int) -> Dict[str, Any]:
            start_time = time.time()
            try:
                db = SessionLocal()
                result = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
                db.close()
                
                return {
                    'query_id': query_id,
                    'execution_time': time.time() - start_time,
                    'result': result,
                    'success': True
                }
            except Exception as e:
                return {
                    'query_id': query_id,
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'success': False
                }
        
        # Test concurrent database queries
        concurrent_queries = 50
        query_results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
            futures = [
                executor.submit(execute_query, i) 
                for i in range(concurrent_queries)
            ]
            
            for future in as_completed(futures):
                query_results.append(future.result())
        
        # Analyze results
        successful_queries = [r for r in query_results if r['success']]
        failed_queries = [r for r in query_results if not r['success']]
        
        success_rate = len(successful_queries) / len(query_results) * 100
        avg_execution_time = statistics.mean([r['execution_time'] for r in successful_queries])
        
        assert success_rate > 95
        assert avg_execution_time < 0.1  # Should be under 100ms
    
    def test_database_transaction_performance(self, db_session):
        """Test database transaction performance."""
        from supernova.db import User, Profile
        
        def create_user_profile(user_id: int) -> Dict[str, Any]:
            start_time = time.time()
            
            try:
                db = SessionLocal()
                
                # Create user
                user = User(
                    name=f"Performance User {user_id}",
                    email=f"perf{user_id}@test.com"
                )
                db.add(user)
                db.flush()
                
                # Create profile
                profile = Profile(
                    user_id=user.id,
                    risk_score=50,
                    time_horizon_yrs=10,
                    objectives="test",
                    constraints="test"
                )
                db.add(profile)
                db.commit()
                db.close()
                
                return {
                    'user_id': user_id,
                    'execution_time': time.time() - start_time,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'user_id': user_id,
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'success': False
                }
        
        # Test concurrent transactions
        concurrent_transactions = 20
        transaction_results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_transactions) as executor:
            futures = [
                executor.submit(create_user_profile, i) 
                for i in range(concurrent_transactions)
            ]
            
            for future in as_completed(futures):
                transaction_results.append(future.result())
        
        # Analyze results
        successful_transactions = [r for r in transaction_results if r['success']]
        success_rate = len(successful_transactions) / len(transaction_results) * 100
        avg_execution_time = statistics.mean([r['execution_time'] for r in successful_transactions])
        
        assert success_rate > 90
        assert avg_execution_time < 0.5  # Should be under 500ms
    
    def test_large_dataset_query_performance(self, db_session):
        """Test query performance with large datasets."""
        from supernova.db import User
        
        # Create large dataset
        batch_size = 1000
        users = []
        
        for i in range(batch_size):
            users.append(User(
                name=f"Batch User {i}",
                email=f"batch{i}@test.com"
            ))
        
        # Insert in batch
        start_time = time.time()
        db_session.add_all(users)
        db_session.commit()
        insert_time = time.time() - start_time
        
        # Query performance tests
        start_time = time.time()
        user_count = db_session.query(User).count()
        count_query_time = time.time() - start_time
        
        start_time = time.time()
        first_100_users = db_session.query(User).limit(100).all()
        limit_query_time = time.time() - start_time
        
        start_time = time.time()
        filtered_users = db_session.query(User).filter(User.name.like('%1%')).all()
        filter_query_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 5.0  # Batch insert should be under 5 seconds
        assert count_query_time < 1.0  # Count query should be under 1 second
        assert limit_query_time < 0.5  # Limit query should be under 500ms
        assert filter_query_time < 2.0  # Filter query should be under 2 seconds
        assert user_count >= batch_size


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage and leak detection."""
    
    def test_memory_usage_during_load(self, sample_ohlcv_data):
        """Test memory usage during high load scenarios."""
        tracemalloc.start()
        
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        runner = LoadTestRunner()
        
        # Simulate heavy load
        test_data = {
            "profile_id": 1,
            "symbol": "MEMORY_TEST",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        # Make many requests
        for i in range(100):
            runner.make_request('POST', '/advice', test_data)
            
            # Check memory every 10 requests
            if i % 10 == 0:
                current_memory = tracemalloc.get_traced_memory()[0]
                memory_growth = current_memory - initial_memory
                
                # Memory growth shouldn't be excessive
                assert memory_growth < 100 * 1024 * 1024  # Less than 100MB
        
        final_memory = tracemalloc.get_traced_memory()[0]
        total_growth = final_memory - initial_memory
        
        tracemalloc.stop()
        
        # Total memory growth should be reasonable
        assert total_growth < 200 * 1024 * 1024  # Less than 200MB total
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repetitive operations."""
        tracemalloc.start()
        
        def create_temporary_objects():
            """Create and destroy temporary objects."""
            large_list = [i for i in range(10000)]
            large_dict = {i: f"value_{i}" for i in range(1000)}
            return len(large_list) + len(large_dict)
        
        # Baseline memory usage
        baseline_memory = tracemalloc.get_traced_memory()[0]
        
        # Run operation multiple times
        iterations = 50
        for i in range(iterations):
            result = create_temporary_objects()
            
            # Force garbage collection periodically
            if i % 10 == 0:
                import gc
                gc.collect()
        
        final_memory = tracemalloc.get_traced_memory()[0]
        memory_growth = final_memory - baseline_memory
        
        tracemalloc.stop()
        
        # Memory growth should be minimal after repeated operations
        assert memory_growth < 10 * 1024 * 1024  # Less than 10MB growth


@pytest.mark.performance
class TestResourceUtilization:
    """Test CPU and resource utilization under load."""
    
    def test_cpu_usage_under_load(self, sample_ohlcv_data):
        """Test CPU usage during intensive operations."""
        process = psutil.Process()
        
        # Monitor CPU usage during load test
        cpu_measurements = []
        runner = LoadTestRunner()
        
        test_data = {
            "strategy_template": "ensemble",
            "params": {
                "ma_crossover": {"fast": 10, "slow": 30},
                "rsi_breakout": {"length": 14}
            },
            "symbol": "CPU_TEST",
            "timeframe": "1h",
            "bars": [bar.model_dump() for bar in sample_ohlcv_data]
        }
        
        def monitor_cpu():
            """Monitor CPU usage in background."""
            for _ in range(30):  # Monitor for 30 seconds
                cpu_percent = process.cpu_percent(interval=1)
                cpu_measurements.append(cpu_percent)
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run CPU-intensive backtest operations
        for _ in range(5):
            runner.make_request('POST', '/backtest', test_data)
        
        monitor_thread.join()
        
        if cpu_measurements:
            avg_cpu = statistics.mean(cpu_measurements)
            max_cpu = max(cpu_measurements)
            
            # CPU usage should be reasonable
            assert avg_cpu < 80.0  # Average CPU should be under 80%
            assert max_cpu < 95.0  # Peak CPU should be under 95%
    
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        def resource_intensive_operation():
            """Perform resource-intensive operation."""
            runner = LoadTestRunner()
            
            # Create many temporary objects
            temp_data = {
                "data": [[i, j] for i in range(100) for j in range(100)]
            }
            
            for _ in range(10):
                runner.make_request('POST', '/intake', temp_data)
        
        # Run operation multiple times
        for _ in range(5):
            resource_intensive_operation()
            gc.collect()  # Force garbage collection
        
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Object count shouldn't grow excessively
        assert object_growth < 1000  # Less than 1000 new objects retained


@pytest.mark.performance
def test_generate_performance_report(tmp_path):
    """Generate comprehensive performance test report."""
    runner = LoadTestRunner()
    
    # Run multiple performance tests
    test_results = {}
    
    # Test different endpoints
    endpoints = [
        ('POST', '/intake', {"name": "Test User"}),
        ('POST', '/advice', {
            "profile_id": 1, 
            "symbol": "TEST", 
            "bars": [{"timestamp": "2024-01-01T10:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}] * 50
        }),
    ]
    
    for method, endpoint, data in endpoints:
        result = runner.run_concurrent_load_test(method, endpoint, 5, 10, data)
        test_results[f"{method}_{endpoint.replace('/', '_')}"] = result
    
    # Generate report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': None,  # Would need import sys
        },
        'test_results': test_results,
        'summary': {
            'total_tests': len(test_results),
            'overall_success_rate': statistics.mean([
                result['metrics']['success_rate'] 
                for result in test_results.values()
            ]),
            'overall_avg_response_time': statistics.mean([
                result['metrics']['response_times']['mean'] 
                for result in test_results.values()
            ]),
        }
    }
    
    # Save report
    report_file = tmp_path / "performance_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    assert report_file.exists()
    assert report['summary']['overall_success_rate'] > 80