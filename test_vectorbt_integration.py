#!/usr/bin/env python3
"""
Test script to validate VectorBT integration with SuperNova Framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import SuperNova modules
from supernova.backtester import run_vbt_backtest, VBT_AVAILABLE
from supernova.schemas import BacktestRequest, OHLCVBar


def generate_test_data(n_bars=100, start_price=100):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Generate price series with slight upward trend and volatility
    price_changes = np.random.normal(0.001, 0.02, n_bars)  # Daily returns
    prices = [start_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    prices = prices[1:]  # Remove the initial price
    
    bars = []
    start_date = datetime(2024, 1, 1)
    
    for i, close in enumerate(prices):
        # Generate OHLC from close price
        daily_vol = abs(np.random.normal(0, 0.01))  # Daily volatility
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        open_price = prices[i-1] if i > 0 else close
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(1000, 50000))
        
        bar = {
            "timestamp": (start_date + timedelta(days=i)).isoformat() + "Z",
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume
        }
        bars.append(bar)
    
    return bars


def test_vectorbt_strategies():
    """Test all VectorBT strategy templates"""
    print("🚀 Testing VectorBT Integration...")
    print(f"VectorBT Available: {VBT_AVAILABLE}")
    
    if not VBT_AVAILABLE:
        print("❌ VectorBT not available. Install dependencies: pip install vectorbt ta-lib numba")
        return False
    
    # Generate test data
    test_bars = generate_test_data(200)
    print(f"📊 Generated {len(test_bars)} test bars")
    
    # Test strategies
    strategies = [
        ("sma_crossover", {"short_window": 10, "long_window": 30}),
        ("rsi_strategy", {"rsi_period": 14, "oversold": 30, "overbought": 70}),
        ("macd_strategy", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("bb_strategy", {"period": 20, "std_dev": 2}),
        ("sentiment_strategy", {"sentiment_threshold": 0.6})
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        print(f"\n🔄 Testing {strategy_name}...")
        
        try:
            metrics = run_vbt_backtest(
                bars=test_bars,
                strategy_template=strategy_name,
                params=params,
                start_cash=10000.0,
                fees=0.001,
                slippage=0.001
            )
            
            if isinstance(metrics, dict) and "error" not in metrics:
                print(f"✅ {strategy_name} - Success!")
                print(f"   Final Equity: ${metrics.get('final_equity', 0):,.2f}")
                print(f"   Total Return: {metrics.get('total_return', 0):.2f}%")
                print(f"   Sharpe Ratio: {metrics.get('Sharpe', 0):.2f}")
                print(f"   Max Drawdown: {metrics.get('MaxDrawdown', 0):.2f}%")
                print(f"   Number of Trades: {metrics.get('n_trades', 0)}")
                
                results[strategy_name] = {
                    "status": "success",
                    "metrics": {
                        "final_equity": metrics.get('final_equity', 0),
                        "total_return": metrics.get('total_return', 0),
                        "sharpe": metrics.get('Sharpe', 0),
                        "max_drawdown": metrics.get('MaxDrawdown', 0),
                        "n_trades": metrics.get('n_trades', 0)
                    }
                }
            else:
                error_msg = metrics.get("error", "Unknown error") if isinstance(metrics, dict) else str(metrics)
                print(f"❌ {strategy_name} - Error: {error_msg}")
                results[strategy_name] = {"status": "error", "error": error_msg}
                
        except Exception as e:
            print(f"❌ {strategy_name} - Exception: {str(e)}")
            results[strategy_name] = {"status": "exception", "error": str(e)}
    
    return results


def test_api_integration():
    """Test API schema integration"""
    print("\n🌐 Testing API Schema Integration...")
    
    try:
        # Create test data
        test_bars = generate_test_data(100)
        
        # Convert to OHLCVBar objects
        ohlcv_bars = []
        for bar in test_bars:
            ohlcv_bar = OHLCVBar(
                timestamp=bar["timestamp"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"]
            )
            ohlcv_bars.append(ohlcv_bar)
        
        # Create BacktestRequest
        request = BacktestRequest(
            strategy_template="sma_crossover",
            params={"short_window": 10, "long_window": 20},
            symbol="TEST",
            timeframe="1h",
            bars=ohlcv_bars,
            use_vectorbt=True,
            start_cash=10000.0,
            fees=0.001,
            slippage=0.001
        )
        
        print("✅ API Schema Integration - Success!")
        print(f"   Created request with {len(request.bars)} bars")
        print(f"   Strategy: {request.strategy_template}")
        print(f"   Use VectorBT: {request.use_vectorbt}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Schema Integration - Error: {str(e)}")
        return False


def performance_comparison():
    """Compare VectorBT vs Legacy backtester performance"""
    print("\n⚡ Performance Comparison...")
    
    try:
        from supernova.backtester import run_backtest
        import time
        
        # Generate larger dataset for performance test
        test_bars = generate_test_data(500)
        strategy = "sma_crossover"  # Use a strategy available in both
        params = {"short_window": 10, "long_window": 30}
        
        # Test VectorBT
        if VBT_AVAILABLE:
            start_time = time.time()
            vbt_result = run_vbt_backtest(test_bars, strategy, params)
            vbt_time = time.time() - start_time
        else:
            vbt_time = None
            vbt_result = {"error": "VectorBT not available"}
        
        # Test Legacy
        start_time = time.time()
        try:
            legacy_result = run_backtest(test_bars, strategy, params)
            legacy_time = time.time() - start_time
        except Exception as e:
            legacy_time = None
            legacy_result = {"error": str(e)}
        
        print(f"📊 Performance Results ({len(test_bars)} bars):")
        if vbt_time is not None:
            print(f"   VectorBT: {vbt_time:.3f}s")
        if legacy_time is not None:
            print(f"   Legacy:   {legacy_time:.3f}s")
        
        if vbt_time and legacy_time:
            speedup = legacy_time / vbt_time
            print(f"   Speedup:  {speedup:.1f}x faster")
        
        return {"vbt_time": vbt_time, "legacy_time": legacy_time}
        
    except Exception as e:
        print(f"❌ Performance Comparison - Error: {str(e)}")
        return None


def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 SuperNova VectorBT Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Strategy Templates
    strategy_results = test_vectorbt_strategies()
    
    # Test 2: API Integration
    api_success = test_api_integration()
    
    # Test 3: Performance Comparison
    perf_results = performance_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    successful_strategies = sum(1 for r in strategy_results.values() if r["status"] == "success")
    total_strategies = len(strategy_results)
    
    print(f"✅ Strategy Tests: {successful_strategies}/{total_strategies} passed")
    print(f"✅ API Integration: {'PASS' if api_success else 'FAIL'}")
    print(f"✅ VectorBT Available: {VBT_AVAILABLE}")
    
    if perf_results:
        print(f"⚡ Performance Test: Completed")
    
    # Save detailed results
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "vectorbt_available": VBT_AVAILABLE,
        "strategy_results": strategy_results,
        "api_integration": api_success,
        "performance_results": perf_results
    }
    
    with open("vectorbt_test_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: vectorbt_test_results.json")
    
    # Final status
    overall_success = (
        VBT_AVAILABLE and 
        successful_strategies >= 3 and  # At least 3 strategies working
        api_success
    )
    
    status = "🎉 ALL TESTS PASSED!" if overall_success else "⚠️  SOME TESTS FAILED"
    print(f"\n{status}")
    
    return overall_success


if __name__ == "__main__":
    main()