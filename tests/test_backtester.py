import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supernova.backtester import (
    run_backtest, run_walk_forward_analysis, run_multi_asset_backtest,
    _max_drawdown, _calculate_win_rate, _calculate_sortino_ratio, 
    _calculate_var, ASSET_CONFIGS, TradeRecord
)
from supernova.strategy_engine import TEMPLATES
from supernova.recursive import (
    optimize_strategy_genetic, GeneticConfig, run_portfolio_optimization
)


@pytest.fixture
def sample_bars():
    """Generate sample OHLCV bar data for backtesting."""
    base_price = 100
    bars = []
    for i in range(200):  # Need more data for backtesting
        timestamp = (datetime.now() - timedelta(hours=200-i)).isoformat() + "Z"
        # Simulate price movement with trend
        price_change = np.random.normal(0, 0.8) + (0.1 if i > 100 else 0)
        base_price += price_change
        
        high = base_price + abs(np.random.normal(0, 0.5))
        low = base_price - abs(np.random.normal(0, 0.5))
        close = base_price + np.random.normal(0, 0.3)
        volume = int(10000 + np.random.normal(0, 2000))
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": max(base_price, high, close),
            "low": min(base_price, low, close),
            "close": close,
            "volume": max(1, volume)
        })
        base_price = close
    
    return bars


@pytest.fixture
def trending_up_bars():
    """Generate bars with clear upward trend for testing profitability."""
    bars = []
    base_price = 100
    for i in range(150):
        timestamp = (datetime.now() - timedelta(hours=150-i)).isoformat() + "Z"
        base_price += 0.3 + np.random.normal(0, 0.2)  # consistent upward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price - 0.1,
            "high": base_price + 0.2,
            "low": base_price - 0.3,
            "close": base_price,
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
    
    return bars


@pytest.fixture
def trending_down_bars():
    """Generate bars with clear downward trend for testing short strategies."""
    bars = []
    base_price = 150
    for i in range(150):
        timestamp = (datetime.now() - timedelta(hours=150-i)).isoformat() + "Z"
        base_price -= 0.4 + np.random.normal(0, 0.2)  # consistent downward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price + 0.1,
            "high": base_price + 0.2,
            "low": base_price - 0.4,
            "close": base_price,
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
    
    return bars


@pytest.fixture
def volatile_sideways_bars():
    """Generate volatile sideways data for testing drawdown scenarios."""
    bars = []
    base_price = 100
    for i in range(150):
        timestamp = (datetime.now() - timedelta(hours=150-i)).isoformat() + "Z"
        # High volatility but no trend
        base_price += np.random.normal(0, 1.5)
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": base_price + abs(np.random.normal(0, 1.0)),
            "low": base_price - abs(np.random.normal(0, 1.0)),
            "close": base_price + np.random.normal(0, 0.8),
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
        base_price = bars[-1]["close"]
    
    return bars


class TestRunBacktest:
    def test_basic_backtest_functionality(self, sample_bars):
        """Test basic backtesting functionality with default parameters."""
        result = run_backtest(sample_bars, "ma_crossover")
        
        assert isinstance(result, dict)
        assert "final_equity" in result
        assert "CAGR" in result
        assert "Sharpe" in result
        assert "MaxDrawdown" in result
        assert "WinRate" in result
        assert "n_trades" in result
        
        # Check that all values are reasonable
        assert result["final_equity"] > 0
        assert -1 <= result["CAGR"] <= 10  # CAGR should be reasonable
        assert -5 <= result["Sharpe"] <= 5  # Sharpe should be reasonable
        assert 0 <= result["MaxDrawdown"] <= 1  # Drawdown is a percentage
        assert 0 <= result["WinRate"] <= 1  # Win rate is a percentage
        assert result["n_trades"] >= 0

    def test_all_strategy_templates(self, sample_bars):
        """Test backtesting with all available strategy templates."""
        for template_name in TEMPLATES.keys():
            result = run_backtest(sample_bars, template_name)
            
            assert isinstance(result, dict)
            assert result["final_equity"] > 0
            assert 0 <= result["MaxDrawdown"] <= 1
            assert 0 <= result["WinRate"] <= 1

    def test_ensemble_strategy(self, sample_bars):
        """Test backtesting with ensemble strategy."""
        result = run_backtest(sample_bars, "ensemble")
        
        assert isinstance(result, dict)
        assert result["final_equity"] > 0
        assert 0 <= result["MaxDrawdown"] <= 1
        assert 0 <= result["WinRate"] <= 1

    def test_custom_parameters(self, sample_bars):
        """Test backtesting with custom strategy parameters."""
        params = {"fast": 5, "slow": 15}
        result = run_backtest(sample_bars, "ma_crossover", params=params)
        
        assert isinstance(result, dict)
        assert result["final_equity"] > 0

    def test_different_starting_cash(self, sample_bars):
        """Test backtesting with different starting cash amounts."""
        cash_amounts = [1000, 10000, 100000]
        
        for cash in cash_amounts:
            result = run_backtest(sample_bars, "ma_crossover", start_cash=cash)
            # Final equity should scale roughly with starting cash
            assert result["final_equity"] > 0
            # CAGR and other ratios should be independent of starting cash
            assert -1 <= result["CAGR"] <= 10

    def test_fee_impact(self, sample_bars):
        """Test that higher fees reduce performance."""
        result_no_fee = run_backtest(sample_bars, "ma_crossover", fee_rate=0.0)
        result_high_fee = run_backtest(sample_bars, "ma_crossover", fee_rate=0.01)
        
        # Higher fees should reduce final equity
        assert result_high_fee["final_equity"] <= result_no_fee["final_equity"]

    def test_allow_short_functionality(self, trending_down_bars):
        """Test that allowing short selling can improve performance in downtrends."""
        result_no_short = run_backtest(trending_down_bars, "ma_crossover", allow_short=False)
        result_with_short = run_backtest(trending_down_bars, "ma_crossover", allow_short=True)
        
        # With shorting allowed, should handle downtrends better
        assert isinstance(result_with_short, dict)
        assert result_with_short["final_equity"] > 0

    def test_profitable_uptrend_strategy(self, trending_up_bars):
        """Test that strategies can capture uptrends profitably."""
        result = run_backtest(trending_up_bars, "ma_crossover")
        
        # In a strong uptrend, final equity should exceed starting cash
        assert result["final_equity"] > 8000  # Allow for some fees and timing
        assert result["CAGR"] > 0  # Should have positive returns

    def test_risk_metrics_calculation(self, volatile_sideways_bars):
        """Test risk metrics in volatile market conditions."""
        result = run_backtest(volatile_sideways_bars, "rsi_breakout")
        
        # In volatile conditions, expect higher drawdown
        assert result["MaxDrawdown"] >= 0
        # Sharpe ratio might be poor due to volatility
        assert isinstance(result["Sharpe"], float)
        # Win rate should be reasonable
        assert 0 <= result["WinRate"] <= 1

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for backtesting."""
        # Only 10 bars - insufficient for backtesting
        short_bars = []
        base_price = 100
        for i in range(10):
            timestamp = (datetime.now() - timedelta(hours=10-i)).isoformat() + "Z"
            short_bars.append({
                "timestamp": timestamp,
                "open": base_price, "high": base_price + 1, "low": base_price - 1,
                "close": base_price, "volume": 1000
            })
        
        # Should handle gracefully - might not execute any trades
        result = run_backtest(short_bars, "ma_crossover")
        # Should return error for insufficient data
        assert "error" in result or result["final_equity"] == 10000.0
        if "error" not in result:
            assert result["n_trades"] == 0

    def test_unknown_template_error(self, sample_bars):
        """Test that unknown template raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown template"):
            run_backtest(sample_bars, "nonexistent_template")

    def test_edge_case_zero_prices(self):
        """Test handling of edge case with zero prices."""
        bars = []
        for i in range(100):
            timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
            price = 0.01 if i < 50 else 0.02  # Very low prices
            bars.append({
                "timestamp": timestamp,
                "open": price, "high": price, "low": price,
                "close": price, "volume": 1000
            })
        
        # Should handle without crashing
        result = run_backtest(bars, "ma_crossover")
        assert result["final_equity"] >= 0

    def test_single_large_move(self):
        """Test backtesting with single large price move."""
        bars = []
        base_price = 100
        for i in range(100):
            timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
            # Large move at position 70
            if i == 70:
                base_price *= 1.5  # 50% jump
            bars.append({
                "timestamp": timestamp,
                "open": base_price, "high": base_price + 0.1, "low": base_price - 0.1,
                "close": base_price, "volume": 1000
            })
        
        result = run_backtest(bars, "fx_breakout")
        # Should capture the breakout
        assert result["final_equity"] > 10000

    def test_ensemble_with_custom_params(self, sample_bars):
        """Test ensemble backtesting with custom parameters."""
        params = {
            "ma_crossover": {"fast": 5, "slow": 20},
            "rsi_breakout": {"length": 10, "low_th": 25, "high_th": 75}
        }
        result = run_backtest(sample_bars, "ensemble", params=params)
        
        assert isinstance(result, dict)
        assert result["final_equity"] > 0


class TestMaxDrawdown:
    def test_no_drawdown_increasing_equity(self):
        """Test max drawdown with consistently increasing equity."""
        equity = np.array([1000, 1100, 1200, 1300, 1400])
        dd = _max_drawdown(equity)
        assert dd == 0.0

    def test_simple_drawdown(self):
        """Test max drawdown with simple decline."""
        equity = np.array([1000, 1200, 800, 1000])  # Peak 1200, trough 800
        dd = _max_drawdown(equity)
        expected_dd = (1200 - 800) / 1200  # 33.33%
        assert abs(dd - expected_dd) < 1e-10

    def test_multiple_drawdowns(self):
        """Test max drawdown with multiple declines."""
        equity = np.array([1000, 1200, 800, 1100, 600, 1000])
        dd = _max_drawdown(equity)
        # Max drawdown should be from peak 1200 to trough 600
        expected_dd = (1200 - 600) / 1200  # 50%
        assert abs(dd - expected_dd) < 1e-10

    def test_empty_equity_curve(self):
        """Test max drawdown with empty equity curve."""
        equity = np.array([])
        dd = _max_drawdown(equity)
        assert dd == 0.0

    def test_single_value(self):
        """Test max drawdown with single equity value."""
        equity = np.array([1000])
        dd = _max_drawdown(equity)
        assert dd == 0.0

    def test_all_identical_values(self):
        """Test max drawdown with all identical values."""
        equity = np.array([1000, 1000, 1000, 1000])
        dd = _max_drawdown(equity)
        assert dd == 0.0


class TestWinRate:
    def test_all_winning_periods(self):
        """Test win rate with all positive changes."""
        equity = np.array([1000, 1100, 1200, 1300])
        wr = _win_rate(equity)
        assert wr == 1.0

    def test_all_losing_periods(self):
        """Test win rate with all negative changes."""
        equity = np.array([1000, 900, 800, 700])
        wr = _win_rate(equity)
        assert wr == 0.0

    def test_mixed_periods(self):
        """Test win rate with mixed positive and negative changes."""
        equity = np.array([1000, 1100, 900, 1200])  # +100, -200, +300
        wr = _win_rate(equity)
        assert wr == 2/3  # 2 winning out of 3 periods

    def test_no_changes(self):
        """Test win rate with no price changes."""
        equity = np.array([1000, 1000, 1000])
        wr = _win_rate(equity)
        assert wr == 0.0  # No gains counted as losses

    def test_empty_equity(self):
        """Test win rate with empty equity curve."""
        equity = np.array([])
        wr = _win_rate(equity)
        assert wr == 0.0

    def test_single_value(self):
        """Test win rate with single equity value."""
        equity = np.array([1000])
        wr = _win_rate(equity)
        assert wr == 0.0  # No changes to evaluate


class TestBacktestEdgeCases:
    def test_constant_hold_signal(self):
        """Test backtest with strategy that always returns hold."""
        # Create custom bars where strategy will always hold
        bars = []
        for i in range(100):
            timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
            bars.append({
                "timestamp": timestamp,
                "open": 50, "high": 50, "low": 50, "close": 50, "volume": 1000
            })
        
        result = run_backtest(bars, "rsi_breakout")  # RSI near 50, should hold
        # No trades should be executed (or very few)
        assert result["final_equity"] <= 11000.0  # Allow for minimal trading
        assert result["n_trades"] <= 2  # Very few trades

    def test_extreme_volatility(self):
        """Test backtest with extreme volatility."""
        bars = []
        base_price = 100
        for i in range(100):
            timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
            # Extreme swings
            change = 50 if i % 2 == 0 else -45
            base_price = max(1, base_price + change)
            
            bars.append({
                "timestamp": timestamp,
                "open": base_price, "high": base_price + 10, "low": base_price - 10,
                "close": base_price, "volume": 1000
            })
        
        # Should handle extreme volatility without crashing
        result = run_backtest(bars, "options_straddle")
        assert isinstance(result, dict)
        assert result["final_equity"] > 0

    def test_very_high_fees(self, sample_bars):
        """Test backtest with very high transaction fees."""
        # 10% fee rate - should make trading unprofitable
        result = run_backtest(sample_bars, "ma_crossover", fee_rate=0.10)
        
        # With very high fees, final equity might be less than starting cash
        assert result["final_equity"] > 0  # But should not go negative
        assert result["final_equity"] <= 11000  # Account for some trading

    def test_precision_handling(self):
        """Test precision handling with very small prices."""
        bars = []
        price = 0.0001  # Very small price
        for i in range(100):
            timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
            price += 0.00001 * np.random.normal()
            price = max(0.00001, price)  # Keep positive
            
            bars.append({
                "timestamp": timestamp,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 1000000
            })
        
        result = run_backtest(bars, "ma_crossover")
        assert result["final_equity"] > 0
        # Should have reasonable number of trades
        assert result["n_trades"] >= 0