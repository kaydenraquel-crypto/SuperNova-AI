"""
Performance Benchmark Suite for TimescaleDB Sentiment Feature Store

This script provides performance benchmarks for the sentiment analysis
and TimescaleDB operations to ensure production readiness.
"""

import asyncio
import time
import statistics
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

class TimescalePerformanceBenchmark:
    """Performance benchmarking for TimescaleDB sentiment operations"""
    
    def __init__(self):
        self.results = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "performance_metrics": {},
            "benchmarks": {},
            "recommendations": []
        }
        
    def log_benchmark(self, test_name: str, metrics: Dict[str, Any]):
        """Log benchmark results"""
        self.results["benchmarks"][test_name] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[BENCH] {test_name}: {metrics}")
    
    def benchmark_sentiment_analysis(self):
        """Benchmark sentiment analysis performance"""
        print("\n=== SENTIMENT ANALYSIS PERFORMANCE ===")
        
        # Test texts of varying lengths
        test_cases = [
            {
                "name": "short_text",
                "text": "Apple stock up today",
                "iterations": 100
            },
            {
                "name": "medium_text", 
                "text": "Apple Inc. shares surged 5% in morning trading following a strong earnings report that beat analyst expectations. The technology giant reported record quarterly revenue driven by robust iPhone sales and growing services revenue.",
                "iterations": 50
            },
            {
                "name": "long_text",
                "text": """Apple Inc. (NASDAQ: AAPL) shares experienced significant volatility in today's trading session, 
                opening 2% higher following the company's quarterly earnings release after market close yesterday. 
                The technology giant reported earnings per share of $1.46, substantially beating the consensus estimate 
                of $1.39. Revenue came in at $89.5 billion, marking a 4.3% year-over-year increase and surpassing 
                analyst projections of $87.8 billion. iPhone sales, which represent approximately 52% of total revenue, 
                showed resilience despite concerns about market saturation in key regions. The Services segment 
                continued its strong growth trajectory, generating $19.2 billion in revenue, up 16% from the same 
                quarter last year. CEO Tim Cook highlighted the company's expanding ecosystem and growing user base, 
                which now exceeds 1.8 billion active devices globally. Despite these positive results, some analysts 
                expressed caution regarding supply chain constraints and potential headwinds from regulatory challenges 
                in international markets. The stock closed up 3.2% at $178.45, bringing the year-to-date performance 
                to a gain of 12.8%.""",
                "iterations": 20
            }
        ]
        
        try:
            # Try to import sentiment analysis
            sys.path.append('.')
            from supernova.sentiment import score_text
            
            for test_case in test_cases:
                times = []
                
                for i in range(test_case["iterations"]):
                    start_time = time.time()
                    result = score_text(test_case["text"])
                    end_time = time.time()
                    
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                
                metrics = {
                    "text_length": len(test_case["text"]),
                    "iterations": test_case["iterations"],
                    "avg_time_ms": round(avg_time, 2),
                    "min_time_ms": round(min_time, 2),
                    "max_time_ms": round(max_time, 2),
                    "std_dev_ms": round(std_dev, 2),
                    "throughput_per_second": round(1000 / avg_time, 2) if avg_time > 0 else 0
                }
                
                self.log_benchmark(f"sentiment_{test_case['name']}", metrics)
                
                # Performance assessment
                if avg_time > 1000:  # More than 1 second
                    self.results["recommendations"].append({
                        "category": "Performance",
                        "priority": "HIGH", 
                        "text": f"Sentiment analysis for {test_case['name']} is slow ({avg_time:.1f}ms avg). Consider optimization."
                    })
                elif avg_time > 500:  # More than 500ms
                    self.results["recommendations"].append({
                        "category": "Performance",
                        "priority": "MEDIUM",
                        "text": f"Sentiment analysis for {test_case['name']} could be faster ({avg_time:.1f}ms avg)."
                    })
            
        except ImportError as e:
            print(f"[SKIP] Sentiment analysis benchmark: {e}")
            self.log_benchmark("sentiment_analysis", {"status": "skipped", "reason": str(e)})
        except Exception as e:
            print(f"[ERROR] Sentiment analysis benchmark: {e}")
            self.log_benchmark("sentiment_analysis", {"status": "error", "error": str(e)})
    
    def benchmark_data_model_operations(self):
        """Benchmark data model creation and serialization"""
        print("\n=== DATA MODEL PERFORMANCE ===")
        
        try:
            from supernova.sentiment import SentimentSignal, SentimentSource, MarketRegime
            
            # Test data model creation
            iterations = 1000
            creation_times = []
            
            for i in range(iterations):
                start_time = time.time()
                
                signal = SentimentSignal(
                    overall_score=0.5,
                    confidence=0.8,
                    source_breakdown={SentimentSource.TWITTER: 0.6, SentimentSource.NEWS: 0.4},
                    figure_influence=0.1,
                    news_impact=0.4,
                    social_momentum=0.2,
                    contrarian_indicator=0.0,
                    regime_adjusted_score=0.5,
                    timestamp=datetime.now(timezone.utc)
                )
                
                end_time = time.time()
                creation_times.append((end_time - start_time) * 1000)
            
            # Calculate statistics
            avg_creation = statistics.mean(creation_times)
            
            metrics = {
                "iterations": iterations,
                "avg_creation_time_ms": round(avg_creation, 4),
                "creation_throughput_per_second": round(1000 / avg_creation, 0) if avg_creation > 0 else 0
            }
            
            self.log_benchmark("data_model_creation", metrics)
            
        except ImportError as e:
            print(f"[SKIP] Data model benchmark: {e}")
            self.log_benchmark("data_model", {"status": "skipped", "reason": str(e)})
        except Exception as e:
            print(f"[ERROR] Data model benchmark: {e}")
            self.log_benchmark("data_model", {"status": "error", "error": str(e)})
    
    def benchmark_file_operations(self):
        """Benchmark file I/O operations"""
        print("\n=== FILE I/O PERFORMANCE ===")
        
        # Test reading module files
        files_to_test = [
            "supernova/sentiment.py",
            "supernova/sentiment_models.py",
            "supernova/timescale_setup.py",
            "supernova/workflows.py",
            "supernova/api.py"
        ]
        
        for file_path in files_to_test:
            if Path(file_path).exists():
                iterations = 50
                read_times = []
                
                for i in range(iterations):
                    start_time = time.time()
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    end_time = time.time()
                    read_times.append((end_time - start_time) * 1000)
                
                avg_read = statistics.mean(read_times)
                file_size = Path(file_path).stat().st_size
                
                metrics = {
                    "file_size_bytes": file_size,
                    "file_size_kb": round(file_size / 1024, 2),
                    "iterations": iterations,
                    "avg_read_time_ms": round(avg_read, 2),
                    "read_throughput_mb_per_second": round((file_size / 1024 / 1024) / (avg_read / 1000), 2) if avg_read > 0 else 0
                }
                
                self.log_benchmark(f"file_read_{Path(file_path).name}", metrics)
    
    def benchmark_json_operations(self):
        """Benchmark JSON serialization/deserialization"""
        print("\n=== JSON PERFORMANCE ===")
        
        # Create test data structure similar to sentiment results
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0.5,
            "confidence": 0.8,
            "source_breakdown": {
                "twitter": 0.6,
                "reddit": 0.4,
                "news": 0.3
            },
            "metadata": {
                "processing_time": 1.5,
                "data_sources": ["twitter", "reddit", "news"],
                "quality_metrics": {
                    "confidence_scores": [0.8, 0.7, 0.9, 0.6, 0.8],
                    "source_counts": {"twitter": 50, "reddit": 30, "news": 20}
                }
            }
        }
        
        iterations = 1000
        
        # Test serialization
        serialize_times = []
        for i in range(iterations):
            start_time = time.time()
            json_str = json.dumps(test_data, default=str)
            end_time = time.time()
            serialize_times.append((end_time - start_time) * 1000)
        
        # Test deserialization 
        json_str = json.dumps(test_data, default=str)
        deserialize_times = []
        for i in range(iterations):
            start_time = time.time()
            data = json.loads(json_str)
            end_time = time.time()
            deserialize_times.append((end_time - start_time) * 1000)
        
        metrics = {
            "data_size_bytes": len(json_str),
            "iterations": iterations,
            "avg_serialize_time_ms": round(statistics.mean(serialize_times), 4),
            "avg_deserialize_time_ms": round(statistics.mean(deserialize_times), 4),
            "serialize_throughput_per_second": round(1000 / statistics.mean(serialize_times), 0),
            "deserialize_throughput_per_second": round(1000 / statistics.mean(deserialize_times), 0)
        }
        
        self.log_benchmark("json_operations", metrics)
    
    def benchmark_memory_usage(self):
        """Estimate memory usage for common operations"""
        print("\n=== MEMORY USAGE ESTIMATION ===")
        
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            
            # Baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test memory usage with sentiment objects
            sentiment_objects = []
            
            try:
                from supernova.sentiment import SentimentSignal, SentimentSource
                
                # Create 1000 sentiment signals
                for i in range(1000):
                    signal = SentimentSignal(
                        overall_score=0.5,
                        confidence=0.8,
                        source_breakdown={SentimentSource.TWITTER: 0.6},
                        figure_influence=0.1,
                        news_impact=0.4,
                        social_momentum=0.2,
                        contrarian_indicator=0.0,
                        regime_adjusted_score=0.5,
                        timestamp=datetime.now(timezone.utc)
                    )
                    sentiment_objects.append(signal)
                
                # Measure memory after creating objects
                after_creation = process.memory_info().rss / 1024 / 1024  # MB
                memory_per_object = (after_creation - baseline_memory) / 1000  # MB per object
                
                metrics = {
                    "baseline_memory_mb": round(baseline_memory, 2),
                    "after_creation_mb": round(after_creation, 2),
                    "objects_created": 1000,
                    "estimated_memory_per_object_kb": round(memory_per_object * 1024, 2),
                    "memory_increase_mb": round(after_creation - baseline_memory, 2)
                }
                
                self.log_benchmark("memory_usage", metrics)
                
                # Cleanup
                del sentiment_objects
                gc.collect()
                
            except ImportError:
                print("[SKIP] Memory benchmark: Cannot import sentiment modules")
                
        except ImportError:
            print("[SKIP] Memory benchmark: psutil not available")
            self.log_benchmark("memory_usage", {"status": "skipped", "reason": "psutil not available"})
        except Exception as e:
            print(f"[ERROR] Memory benchmark: {e}")
            self.log_benchmark("memory_usage", {"status": "error", "error": str(e)})
    
    def assess_performance(self):
        """Assess overall performance and generate recommendations"""
        print("\n=== PERFORMANCE ASSESSMENT ===")
        
        benchmarks = self.results["benchmarks"]
        
        # Assess sentiment analysis performance
        sentiment_benchmarks = {k: v for k, v in benchmarks.items() if k.startswith("sentiment_")}
        
        if sentiment_benchmarks:
            avg_times = []
            for bench_name, bench_data in sentiment_benchmarks.items():
                if "metrics" in bench_data and "avg_time_ms" in bench_data["metrics"]:
                    avg_times.append(bench_data["metrics"]["avg_time_ms"])
            
            if avg_times:
                overall_avg = statistics.mean(avg_times)
                
                if overall_avg < 100:
                    performance_rating = "EXCELLENT"
                elif overall_avg < 300:
                    performance_rating = "GOOD"
                elif overall_avg < 1000:
                    performance_rating = "ACCEPTABLE"
                else:
                    performance_rating = "NEEDS_IMPROVEMENT"
                
                self.results["performance_metrics"]["sentiment_analysis"] = {
                    "overall_avg_time_ms": round(overall_avg, 2),
                    "performance_rating": performance_rating,
                    "recommendation": self._get_performance_recommendation(performance_rating)
                }
        
        # Assess data model performance
        if "data_model_creation" in benchmarks:
            model_metrics = benchmarks["data_model_creation"]["metrics"]
            if "avg_creation_time_ms" in model_metrics:
                creation_time = model_metrics["avg_creation_time_ms"]
                
                if creation_time < 0.1:
                    model_rating = "EXCELLENT"
                elif creation_time < 0.5:
                    model_rating = "GOOD"
                elif creation_time < 2.0:
                    model_rating = "ACCEPTABLE"
                else:
                    model_rating = "NEEDS_IMPROVEMENT"
                
                self.results["performance_metrics"]["data_model"] = {
                    "creation_time_ms": creation_time,
                    "performance_rating": model_rating,
                    "throughput_per_second": model_metrics.get("creation_throughput_per_second", 0)
                }
        
        # Generate overall performance recommendations
        self._generate_performance_recommendations()
    
    def _get_performance_recommendation(self, rating: str) -> str:
        """Get performance recommendation based on rating"""
        recommendations = {
            "EXCELLENT": "Performance is excellent. Monitor in production to maintain quality.",
            "GOOD": "Performance is good. Consider minor optimizations for high-load scenarios.",
            "ACCEPTABLE": "Performance is acceptable but could benefit from optimization.",
            "NEEDS_IMPROVEMENT": "Performance needs improvement before production deployment."
        }
        return recommendations.get(rating, "Performance assessment needed.")
    
    def _generate_performance_recommendations(self):
        """Generate detailed performance recommendations"""
        
        # CPU-bound operations recommendations
        self.results["recommendations"].extend([
            {
                "category": "Optimization",
                "priority": "MEDIUM",
                "text": "Consider implementing caching for frequently analyzed text patterns"
            },
            {
                "category": "Scalability",
                "priority": "MEDIUM", 
                "text": "Use async operations for I/O-bound tasks like API calls and database operations"
            },
            {
                "category": "Memory",
                "priority": "LOW",
                "text": "Implement object pooling for high-frequency sentiment analysis operations"
            },
            {
                "category": "Monitoring",
                "priority": "MEDIUM",
                "text": "Set up performance monitoring and alerting in production"
            }
        ])
    
    def run_benchmarks(self):
        """Run all performance benchmarks"""
        print("STARTING TIMESCALEDB SENTIMENT PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run benchmark components
        self.benchmark_sentiment_analysis()
        self.benchmark_data_model_operations()
        self.benchmark_file_operations()
        self.benchmark_json_operations()
        self.benchmark_memory_usage()
        
        # Assess overall performance
        self.assess_performance()
        
        total_time = time.time() - start_time
        
        # Add execution time
        self.results["total_benchmark_time_seconds"] = round(total_time, 2)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Total Benchmark Time: {self.results['total_benchmark_time_seconds']}s")
        print(f"Benchmarks Run: {len(self.results['benchmarks'])}")
        print()
        
        # Performance metrics summary
        if self.results["performance_metrics"]:
            print("Performance Ratings:")
            for component, metrics in self.results["performance_metrics"].items():
                rating = metrics.get("performance_rating", "N/A")
                print(f"   {component}: {rating}")
            print()
        
        # Recommendations summary
        if self.results["recommendations"]:
            high_priority = [r for r in self.results["recommendations"] if r["priority"] == "HIGH"]
            if high_priority:
                print("High Priority Performance Issues:")
                for rec in high_priority:
                    print(f"   - {rec['text']}")
                print()
        
        print("=" * 60)
        print("Performance Benchmark Complete!")
        print("=" * 60)


def main():
    """Run performance benchmarks"""
    benchmark = TimescalePerformanceBenchmark()
    results = benchmark.run_benchmarks()
    
    # Save results
    output_file = "timescale_performance_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed benchmark results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())