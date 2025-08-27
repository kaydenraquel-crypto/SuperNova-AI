import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supernova.backtester import (
    run_backtest, run_walk_forward_analysis, run_multi_asset_backtest,
    _calculate_sortino_ratio, _calculate_var, _calculate_win_rate,
    ASSET_CONFIGS, TradeRecord
)
from supernova.recursive import (
    optimize_strategy_genetic, GeneticConfig, run_portfolio_optimization,
    run_ensemble_optimization
)


@pytest.fixture
def enhanced_sample_bars():
    """Generate enhanced sample data with more realistic price movements."""
    np.random.seed(42)  # For reproducible tests
    bars = []
    base_price = 100.0
    
    for i in range(300):  # More data for advanced tests
        timestamp = (datetime.now() - timedelta(hours=300-i)).isoformat() + "Z"
        
        # Add some trend and volatility
        trend = 0.02 if i > 150 else -0.01
        daily_return = trend + np.random.normal(0, 0.015)
        
        base_price *= (1 + daily_return)
        base_price = max(0.1, base_price)  # Prevent negative prices
        
        # OHLC generation
        high_offset = abs(np.random.normal(0, 0.005))
        low_offset = abs(np.random.normal(0, 0.005))
        close_offset = np.random.normal(0, 0.002)
        
        high = base_price * (1 + high_offset)
        low = base_price * (1 - low_offset)
        close = base_price * (1 + close_offset)
        volume = max(1000, int(10000 + np.random.normal(0, 2000)))
        
        bars.append({
            "timestamp": timestamp,
            "open": float(base_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": volume
        })
        
        base_price = close
    
    return bars


@pytest.fixture
def multi_asset_data(enhanced_sample_bars):
    """Create multi-asset data for testing."""
    # Create correlated asset data
    asset_b_bars = []
    for i, bar in enumerate(enhanced_sample_bars):
        # Add correlation but with some noise
        correlation_factor = 0.7
        noise_factor = 0.3
        
        original_return = (bar["close"] / bar["open"]) - 1
        correlated_return = original_return * correlation_factor + np.random.normal(0, 0.01) * noise_factor
        
        new_close = bar["open"] * (1 + correlated_return)
        new_high = new_close * (1 + abs(np.random.normal(0, 0.003)))
        new_low = new_close * (1 - abs(np.random.normal(0, 0.003)))
        
        asset_b_bars.append({
            "timestamp": bar["timestamp"],
            "open": bar["open"] * 0.9,  # Different base price
            "high": float(new_high),
            "low": float(new_low),
            "close": float(new_close),
            "volume": bar["volume"]
        })
    
    return {
        "ASSET_A": enhanced_sample_bars,
        "ASSET_B": asset_b_bars
    }


class TestEnhancedBacktester:
    """Tests for enhanced backtesting features"""
    
    def test_asset_type_configurations(self, enhanced_sample_bars):
        """Test different asset type configurations"""
        asset_types = ["stock", "forex", "futures", "options", "crypto"]
        
        for asset_type in asset_types:
            result = run_backtest(enhanced_sample_bars, "ma_crossover", asset_type=asset_type)
            assert isinstance(result, dict)
            assert result["asset_type"] == asset_type
            assert "max_leverage_used" in result
            assert "margin_calls" in result
            
            # Check asset-specific parameters
            config = ASSET_CONFIGS[asset_type]
            assert result["leverage"] <= config.max_leverage
    
    def test_leverage_and_margin(self, enhanced_sample_bars):
        """Test leverage and margin functionality"""
        # Test with different leverage levels
        base_result = run_backtest(enhanced_sample_bars, "ma_crossover", leverage=1.0)
        leveraged_result = run_backtest(enhanced_sample_bars, "ma_crossover", leverage=2.0)
        
        assert isinstance(base_result, dict)
        assert isinstance(leveraged_result, dict)
        assert base_result["leverage"] == 1.0
        assert leveraged_result["leverage"] == 2.0
        
        # Margin tracking
        assert "avg_margin_usage" in leveraged_result
        assert isinstance(leveraged_result["avg_margin_usage"], (int, float))
    
    def test_slippage_and_market_impact(self, enhanced_sample_bars):
        """Test slippage and market impact simulation"""
        no_slippage = run_backtest(enhanced_sample_bars, "ma_crossover", enable_slippage=False)
        with_slippage = run_backtest(enhanced_sample_bars, "ma_crossover", enable_slippage=True)
        
        assert no_slippage["enable_slippage"] == False
        assert with_slippage["enable_slippage"] == True
        
        # With slippage enabled, should track total slippage
        assert "total_slippage" in with_slippage
        assert "slippage_ratio" in with_slippage
        
        if with_slippage["n_trades"] > 0:
            assert with_slippage["total_slippage"] >= 0
    
    def test_after_hours_trading(self, enhanced_sample_bars):
        """Test after-hours trading simulation"""
        regular_hours = run_backtest(enhanced_sample_bars, "ma_crossover", enable_after_hours=False)
        after_hours = run_backtest(enhanced_sample_bars, "ma_crossover", enable_after_hours=True)
        
        assert regular_hours["enable_after_hours"] == False
        assert after_hours["enable_after_hours"] == True
    
    def test_advanced_risk_metrics(self, enhanced_sample_bars):
        """Test advanced risk metrics calculation"""
        result = run_backtest(enhanced_sample_bars, "ma_crossover")
        
        # Check for new risk metrics
        expected_metrics = [
            "Sortino", "VaR_95", "VaR_99", "RecoveryFactor", 
            "ProfitFactor", "InformationRatio", "MaxAdverseExcursion",
            "MaxFavorableExcursion"
        ]
        
        for metric in expected_metrics:
            assert metric in result, f"Missing metric: {metric}"
            assert isinstance(result[metric], (int, float))
    
    def test_trade_level_analytics(self, enhanced_sample_bars):
        """Test detailed trade-level analytics"""
        result = run_backtest(enhanced_sample_bars, "ma_crossover")
        
        # Check trade analytics
        trade_metrics = [
            "avg_trade_pnl", "avg_win_pnl", "avg_loss_pnl",
            "largest_win", "largest_loss", "avg_hold_time"
        ]
        
        for metric in trade_metrics:
            assert metric in result, f"Missing trade metric: {metric}"
            assert isinstance(result[metric], (int, float))
        
        # Check detailed trades data
        assert "trades" in result
        assert isinstance(result["trades"], list)
        
        if result["trades"]:
            trade = result["trades"][0]
            expected_fields = [
                "timestamp", "type", "price", "shares", "pnl",
                "slippage", "market_impact", "after_hours"
            ]
            for field in expected_fields:
                assert field in trade, f"Missing trade field: {field}"
    
    def test_contract_multipliers(self, enhanced_sample_bars):
        """Test contract multipliers for different asset types"""
        result = run_backtest(
            enhanced_sample_bars, "futures_trend", 
            asset_type="futures",
            contract_multiplier=50.0
        )
        
        assert result["contract_multiplier"] == 50.0
        assert "point_value" in result
    
    def test_partial_fills(self, enhanced_sample_bars):
        """Test partial fill simulation"""
        result = run_backtest(
            enhanced_sample_bars, "ma_crossover", 
            partial_fill_prob=0.2  # 20% chance of partial fill
        )
        
        # Check if partial fill field exists in trades
        if result["trades"]:
            assert all("partial_fill" in t for t in result["trades"])
    
    def test_margin_calls(self, enhanced_sample_bars):
        """Test margin call simulation"""
        result = run_backtest(
            enhanced_sample_bars, "ma_crossover",
            leverage=5.0,  # High leverage to potentially trigger margin calls
            asset_type="crypto"  # Crypto has lower margin requirements
        )
        
        assert "margin_calls" in result
        assert isinstance(result["margin_calls"], int)
        assert result["margin_calls"] >= 0
    
    def test_fees_and_costs_tracking(self, enhanced_sample_bars):
        """Test comprehensive fees and costs tracking"""
        result = run_backtest(enhanced_sample_bars, "ma_crossover", fee_rate=0.001)
        
        cost_metrics = [
            "total_fees_paid", "total_slippage", 
            "fees_ratio", "slippage_ratio"
        ]
        
        for metric in cost_metrics:
            assert metric in result, f"Missing cost metric: {metric}"
            assert isinstance(result[metric], (int, float))
    
    def test_equity_curve_and_returns(self, enhanced_sample_bars):
        """Test equity curve and returns data"""
        result = run_backtest(enhanced_sample_bars, "ma_crossover")
        
        assert "equity_curve" in result
        assert "daily_returns" in result
        assert "underwater_curve" in result
        
        assert isinstance(result["equity_curve"], list)
        assert isinstance(result["daily_returns"], list)
        assert isinstance(result["underwater_curve"], list)
        
        if result["equity_curve"]:
            assert len(result["equity_curve"]) > 0
            assert all(isinstance(x, (int, float)) for x in result["equity_curve"])


class TestWalkForwardAnalysis:
    """Tests for walk-forward analysis"""
    
    def test_basic_walk_forward(self, enhanced_sample_bars):
        """Test basic walk-forward analysis"""
        result = run_walk_forward_analysis(
            enhanced_sample_bars, "ma_crossover",
            window_size=50, step_size=10, optimization_period=30
        )
        
        assert "individual_periods" in result
        assert "aggregate_metrics" in result
        assert "analysis_type" in result
        assert result["analysis_type"] == "walk_forward"
        
        if result["individual_periods"]:
            period_result = result["individual_periods"][0]
            assert "optimized_params" in period_result
            assert "period_start" in period_result
            assert "period_end" in period_result
    
    def test_insufficient_data_walk_forward(self, enhanced_sample_bars):
        """Test walk-forward with insufficient data"""
        short_bars = enhanced_sample_bars[:50]
        result = run_walk_forward_analysis(
            short_bars, "ma_crossover",
            window_size=252, step_size=21, optimization_period=126
        )
        
        assert "error" in result


class TestMultiAssetBacktest:
    """Tests for multi-asset backtesting"""
    
    def test_multi_asset_basic(self, multi_asset_data):
        """Test basic multi-asset backtesting"""
        result = run_multi_asset_backtest(
            multi_asset_data, "cross_asset_momentum"
        )
        
        if "error" not in result:
            assert "assets_analyzed" in result
            assert "correlation_matrix" in result
            assert len(result["assets_analyzed"]) == 2
            assert "final_positions" in result
    
    def test_risk_parity_strategy(self, multi_asset_data):
        """Test risk parity multi-asset strategy"""
        result = run_multi_asset_backtest(
            multi_asset_data, "risk_parity"
        )
        
        if "error" not in result:
            assert "strategy" in result
            assert result["strategy"] == "risk_parity"
    
    def test_unsupported_template_multi_asset(self, multi_asset_data):
        """Test multi-asset with unsupported template"""
        with pytest.raises(ValueError):
            run_multi_asset_backtest(multi_asset_data, "ma_crossover")


class TestGeneticOptimization:
    """Tests for genetic algorithm optimization"""
    
    def test_basic_genetic_optimization(self, enhanced_sample_bars):
        """Test basic genetic optimization"""
        param_ranges = {
            "fast": (5, 20, 'int'),
            "slow": (20, 50, 'int')
        }
        
        config = GeneticConfig(population_size=10, generations=3)  # Small for testing
        
        result = optimize_strategy_genetic(
            enhanced_sample_bars, "ma_crossover", param_ranges, config
        )
        
        assert "optimized_params" in result
        assert "best_fitness" in result
        assert "final_metrics" in result
        assert "generations_run" in result
        
        # Check optimized parameters are within range
        assert 5 <= result["optimized_params"]["fast"] <= 20
        assert 20 <= result["optimized_params"]["slow"] <= 50
    
    def test_genetic_config_options(self, enhanced_sample_bars):
        """Test different genetic algorithm configurations"""
        param_ranges = {
            "length": (10, 30, 'int'),
            "low_th": (20, 40, 'int'),
            "high_th": (60, 80, 'int')
        }
        
        config = GeneticConfig(
            population_size=8, 
            generations=2, 
            fitness_function="calmar"
        )
        
        result = optimize_strategy_genetic(
            enhanced_sample_bars, "rsi_breakout", param_ranges, config
        )
        
        assert isinstance(result, dict)
        assert "optimization_config" in result
        assert result["optimization_config"]["fitness_function"] == "calmar"
    
    def test_invalid_template_genetic(self, enhanced_sample_bars):
        """Test genetic optimization with invalid template"""
        param_ranges = {"param": (1, 10, 'int')}
        
        with pytest.raises(ValueError):
            optimize_strategy_genetic(
                enhanced_sample_bars, "nonexistent_strategy", param_ranges
            )


class TestPortfolioOptimization:
    """Tests for portfolio optimization"""
    
    def test_risk_parity_optimization(self, multi_asset_data):
        """Test risk parity portfolio optimization"""
        result = run_portfolio_optimization(
            multi_asset_data, optimization_method="risk_parity"
        )
        
        assert "weights" in result
        assert "expected_return" in result
        assert "expected_volatility" in result
        assert "sharpe_ratio" in result
        assert "optimization_method" in result
        
        # Weights should sum to 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point errors
    
    def test_min_variance_optimization(self, multi_asset_data):
        """Test minimum variance optimization"""
        result = run_portfolio_optimization(
            multi_asset_data, optimization_method="min_variance"
        )
        
        assert result["optimization_method"] == "min_variance"
        assert "correlation_matrix" in result
    
    def test_portfolio_constraints(self, multi_asset_data):
        """Test portfolio optimization with constraints"""
        constraints = {
            "max_weight": 0.6,
            "min_weight": 0.1,
            "target_volatility": 0.12
        }
        
        result = run_portfolio_optimization(
            multi_asset_data, 
            optimization_method="risk_parity",
            constraints=constraints
        )
        
        assert result["constraints"] == constraints
        
        # Check weight constraints
        for weight in result["weights"].values():
            assert constraints["min_weight"] <= weight <= constraints["max_weight"]


class TestEnsembleOptimization:
    """Tests for ensemble strategy optimization"""
    
    def test_ensemble_optimization(self, enhanced_sample_bars):
        """Test ensemble strategy optimization"""
        strategies = ["ma_crossover", "rsi_breakout", "macd_trend"]
        
        result = run_ensemble_optimization(enhanced_sample_bars, strategies)
        
        if "error" not in result:
            assert "ensemble_weights" in result
            assert "individual_results" in result
            assert "correlation_matrix" in result
            assert "ensemble_sharpe" in result
            assert "ensemble_cagr" in result
            
            # Weights should sum to 1
            total_weight = sum(result["ensemble_weights"].values())
            assert abs(total_weight - 1.0) < 0.01
    
    def test_insufficient_strategies_ensemble(self, enhanced_sample_bars):
        """Test ensemble with insufficient successful strategies"""
        # Use a strategy that might fail
        strategies = ["nonexistent_strategy"]
        
        result = run_ensemble_optimization(enhanced_sample_bars, strategies)
        
        # Should return error due to insufficient strategies
        assert "error" in result


class TestAdvancedRiskMetrics:
    """Tests for advanced risk metric calculations"""
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015, -0.02, 0.008])
        sortino = _calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino) or sortino == float('inf')
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        var_95 = _calculate_var(returns, 0.95)
        var_99 = _calculate_var(returns, 0.99)
        
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 <= var_95  # 99% VaR should be more extreme (more negative)
    
    def test_win_rate_calculation(self):
        """Test enhanced win rate calculation"""
        trades = [
            TradeRecord(timestamp="2023-01-01", type="buy", price=100.0, shares=10, pnl=50.0),
            TradeRecord(timestamp="2023-01-02", type="sell", price=105.0, shares=10, pnl=-20.0),
            TradeRecord(timestamp="2023-01-03", type="buy", price=95.0, shares=10, pnl=30.0),
        ]
        
        win_rate = _calculate_win_rate(trades)
        assert win_rate == 2/3  # 2 winning trades out of 3
    
    def test_empty_trades_win_rate(self):
        """Test win rate with no trades"""
        win_rate = _calculate_win_rate([])
        assert win_rate == 0.0
    
    def test_all_winning_trades(self):
        """Test win rate with all winning trades"""
        trades = [
            TradeRecord(timestamp="2023-01-01", type="buy", price=100.0, shares=10, pnl=50.0),
            TradeRecord(timestamp="2023-01-02", type="sell", price=105.0, shares=10, pnl=20.0),
            TradeRecord(timestamp="2023-01-03", type="buy", price=95.0, shares=10, pnl=30.0),
        ]
        
        win_rate = _calculate_win_rate(trades)
        assert win_rate == 1.0
    
    def test_all_losing_trades(self):
        """Test win rate with all losing trades"""
        trades = [
            TradeRecord(timestamp="2023-01-01", type="buy", price=100.0, shares=10, pnl=-50.0),
            TradeRecord(timestamp="2023-01-02", type="sell", price=105.0, shares=10, pnl=-20.0),
            TradeRecord(timestamp="2023-01-03", type="buy", price=95.0, shares=10, pnl=-30.0),
        ]
        
        win_rate = _calculate_win_rate(trades)
        assert win_rate == 0.0


class TestAssetConfigurations:
    """Tests for asset-specific configurations"""
    
    def test_asset_config_parameters(self):
        """Test that all asset configurations have required parameters"""
        required_attrs = [
            'min_margin', 'overnight_margin', 'max_leverage', 
            'base_slippage_bps', 'impact_coefficient'
        ]
        
        for asset_type, config in ASSET_CONFIGS.items():
            for attr in required_attrs:
                assert hasattr(config, attr), f"{asset_type} missing {attr}"
    
    def test_forex_config(self):
        """Test forex-specific configuration"""
        forex_config = ASSET_CONFIGS["forex"]
        
        assert forex_config.max_leverage >= 20.0  # Forex should allow high leverage
        assert forex_config.min_margin <= 0.05   # Low margin requirements
        assert forex_config.settlement_days == 0  # T+0 settlement
    
    def test_options_config(self):
        """Test options-specific configuration"""
        options_config = ASSET_CONFIGS["options"]
        
        assert options_config.contract_size == 100.0  # Standard options contract
        assert options_config.point_value == 100.0
        assert options_config.settlement_days == 1   # T+1 settlement
    
    def test_futures_config(self):
        """Test futures-specific configuration"""
        futures_config = ASSET_CONFIGS["futures"]
        
        assert futures_config.max_leverage >= 10.0  # Futures allow leverage
        assert futures_config.contract_size > 1.0   # Futures have multipliers
        assert futures_config.settlement_days == 0  # Daily settlement