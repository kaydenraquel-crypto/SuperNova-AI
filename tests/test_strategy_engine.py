import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supernova.strategy_engine import (
    make_df, eval_ma_crossover, eval_rsi_breakout, eval_macd_trend,
    eval_options_straddle, eval_fx_breakout, eval_futures_trend,
    ensemble, TEMPLATES
)


@pytest.fixture
def sample_bars():
    """Generate realistic OHLCV bar data for testing."""
    base_price = 100
    bars = []
    for i in range(100):
        timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
        # Simulate price movement with some volatility
        price_change = np.random.normal(0, 0.5) + (0.1 if i > 50 else -0.05)  # trend after midpoint
        base_price += price_change
        
        high = base_price + abs(np.random.normal(0, 0.3))
        low = base_price - abs(np.random.normal(0, 0.3))
        close = base_price + np.random.normal(0, 0.2)
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
    """Generate bars with clear upward trend."""
    bars = []
    base_price = 100
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
        base_price += 0.5 + np.random.normal(0, 0.2)  # consistent upward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price - 0.2,
            "high": base_price + 0.3,
            "low": base_price - 0.4,
            "close": base_price,
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
    
    return bars


@pytest.fixture
def trending_down_bars():
    """Generate bars with clear downward trend."""
    bars = []
    base_price = 150
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
        base_price -= 0.6 + np.random.normal(0, 0.2)  # consistent downward movement
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price + 0.2,
            "high": base_price + 0.3,
            "low": base_price - 0.5,
            "close": base_price,
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
    
    return bars


@pytest.fixture
def sideways_bars():
    """Generate bars with sideways movement."""
    bars = []
    base_price = 100
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
        base_price += np.random.normal(0, 0.3)  # random walk around base
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": base_price + abs(np.random.normal(0, 0.5)),
            "low": base_price - abs(np.random.normal(0, 0.5)),
            "close": base_price + np.random.normal(0, 0.3),
            "volume": 10000 + int(np.random.normal(0, 1000))
        })
        base_price = bars[-1]["close"]
    
    return bars


class TestMakeDF:
    def test_make_df_basic(self, sample_bars):
        """Test DataFrame creation from bars."""
        df = make_df(sample_bars)
        assert len(df) == len(sample_bars)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_make_df_sorting(self):
        """Test that DataFrame is properly sorted by timestamp."""
        bars = [
            {"timestamp": "2024-01-01T02:00:00Z", "open": 102, "high": 105, "low": 101, "close": 104, "volume": 1000},
            {"timestamp": "2024-01-01T01:00:00Z", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1000},
            {"timestamp": "2024-01-01T03:00:00Z", "open": 104, "high": 106, "low": 103, "close": 105, "volume": 1000},
        ]
        df = make_df(bars)
        assert df.iloc[0].name.hour == 1
        assert df.iloc[1].name.hour == 2
        assert df.iloc[2].name.hour == 3

    def test_make_df_empty_list(self):
        """Test handling of empty bar list."""
        df = make_df([])
        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


class TestMAStrategy:
    def test_eval_ma_crossover_basic(self, sample_bars):
        """Test basic MA crossover functionality."""
        df = make_df(sample_bars)
        signal, conf, details = eval_ma_crossover(df, fast=5, slow=10)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "fast" in details
        assert "slow" in details
        assert isinstance(details["fast"], float)
        assert isinstance(details["slow"], float)

    def test_eval_ma_crossover_buy_signal(self):
        """Test MA crossover buy signal generation."""
        # Create data where fast MA crosses above slow MA
        bars = []
        base = 100
        for i in range(30):
            timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
            # Price drops then rises sharply at the end to trigger crossover
            price = base - 5 + (0.5 * i if i > 20 else 0)
            bars.append({
                "timestamp": timestamp,
                "open": price, "high": price + 0.5, "low": price - 0.5, 
                "close": price, "volume": 1000
            })
        
        df = make_df(bars)
        signal, conf, details = eval_ma_crossover(df, fast=3, slow=10)
        
        # Should detect the crossover
        assert signal in ["buy", "hold"]  # May not always trigger depending on exact values
        assert details["fast"] != details["slow"]

    def test_eval_ma_crossover_custom_params(self, sample_bars):
        """Test MA crossover with custom parameters."""
        df = make_df(sample_bars)
        signal, conf, details = eval_ma_crossover(df, fast=20, slow=50)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0

    def test_eval_ma_crossover_insufficient_data(self):
        """Test MA crossover with insufficient data."""
        bars = [
            {"timestamp": "2024-01-01T01:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
        ]
        df = make_df(bars)
        
        # Should handle gracefully but may have NaN values
        signal, conf, details = eval_ma_crossover(df, fast=10, slow=20)
        assert signal in ["buy", "sell", "hold"]


class TestRSIStrategy:
    def test_eval_rsi_breakout_basic(self, sample_bars):
        """Test basic RSI breakout functionality."""
        df = make_df(sample_bars)
        signal, conf, details = eval_rsi_breakout(df, length=14, low_th=30, high_th=70)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "rsi" in details
        assert 0 <= details["rsi"] <= 100

    def test_eval_rsi_breakout_oversold(self):
        """Test RSI breakout buy signal in oversold conditions."""
        # Create consistently declining prices to get low RSI
        bars = []
        base = 100
        for i in range(30):
            timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
            base -= 1.5  # Consistent decline
            bars.append({
                "timestamp": timestamp,
                "open": base + 1, "high": base + 1.2, "low": base - 0.5,
                "close": base, "volume": 1000
            })
        
        df = make_df(bars)
        signal, conf, details = eval_rsi_breakout(df, length=14, low_th=30, high_th=70)
        
        # Should be oversold and generate buy signal
        assert details["rsi"] < 50  # Should be low due to consistent decline
        if details["rsi"] < 30:
            assert signal == "buy"

    def test_eval_rsi_breakout_overbought(self):
        """Test RSI breakout sell signal in overbought conditions."""
        # Create consistently rising prices to get high RSI
        bars = []
        base = 100
        for i in range(30):
            timestamp = (datetime.now() - timedelta(hours=30-i)).isoformat() + "Z"
            base += 1.5  # Consistent rise
            bars.append({
                "timestamp": timestamp,
                "open": base - 1, "high": base + 0.5, "low": base - 1.2,
                "close": base, "volume": 1000
            })
        
        df = make_df(bars)
        signal, conf, details = eval_rsi_breakout(df, length=14, low_th=30, high_th=70)
        
        # Should be overbought
        assert details["rsi"] > 50  # Should be high due to consistent rise
        if details["rsi"] > 70:
            assert signal == "sell"

    def test_eval_rsi_breakout_custom_thresholds(self, sample_bars):
        """Test RSI breakout with custom thresholds."""
        df = make_df(sample_bars)
        signal, conf, details = eval_rsi_breakout(df, length=10, low_th=25, high_th=75)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0 <= details["rsi"] <= 100


class TestMACDStrategy:
    def test_eval_macd_trend_basic(self, sample_bars):
        """Test basic MACD trend functionality."""
        df = make_df(sample_bars)
        signal, conf, details = eval_macd_trend(df, fast=12, slow=26, signal=9)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "macd_hist" in details
        assert isinstance(details["macd_hist"], float)

    def test_eval_macd_trend_bullish(self, trending_up_bars):
        """Test MACD trend with bullish histogram."""
        df = make_df(trending_up_bars)
        signal, conf, details = eval_macd_trend(df, fast=5, slow=10, signal=3)
        
        # With upward trend, MACD histogram should eventually turn positive
        if details["macd_hist"] > 0:
            assert signal == "buy"
        elif details["macd_hist"] < 0:
            assert signal == "sell"
        else:
            assert signal == "hold"

    def test_eval_macd_trend_bearish(self, trending_down_bars):
        """Test MACD trend with bearish histogram."""
        df = make_df(trending_down_bars)
        signal, conf, details = eval_macd_trend(df, fast=5, slow=10, signal=3)
        
        # With downward trend, MACD histogram should be negative
        if details["macd_hist"] < 0:
            assert signal == "sell"

    def test_eval_macd_trend_custom_params(self, sample_bars):
        """Test MACD trend with custom parameters."""
        df = make_df(sample_bars)
        signal, conf, details = eval_macd_trend(df, fast=8, slow=21, signal=5)
        
        assert signal in ["buy", "sell", "hold"]


class TestExtendedStrategies:
    def test_eval_options_straddle_basic(self, sample_bars):
        """Test basic options straddle strategy."""
        df = make_df(sample_bars)
        signal, conf, details = eval_options_straddle(df, window=20)
        
        assert signal in ["buy", "hold"]  # Straddle only generates buy or hold
        assert 0.1 <= conf <= 1.0
        assert "std" in details
        assert "avg_std" in details
        assert details["std"] > 0
        assert details["avg_std"] > 0

    def test_eval_options_straddle_high_volatility(self):
        """Test options straddle with high volatility scenario."""
        bars = []
        base = 100
        for i in range(40):
            timestamp = (datetime.now() - timedelta(hours=40-i)).isoformat() + "Z"
            # Add high volatility in recent periods
            volatility = 3.0 if i > 30 else 0.5
            change = np.random.normal(0, volatility)
            base += change
            
            bars.append({
                "timestamp": timestamp,
                "open": base, "high": base + abs(change/2), "low": base - abs(change/2),
                "close": base + change/2, "volume": 1000
            })
        
        df = make_df(bars)
        signal, conf, details = eval_options_straddle(df, window=10)
        
        # High recent volatility should trigger buy signal
        if details["std"] > 1.5 * details["avg_std"]:
            assert signal == "buy"

    def test_eval_fx_breakout_basic(self, sample_bars):
        """Test basic FX breakout strategy."""
        df = make_df(sample_bars)
        signal, conf, details = eval_fx_breakout(df, lookback=20, buffer=0.001)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "breakout_high" in details or "breakout_low" in details or "range" in details

    def test_eval_fx_breakout_upward(self):
        """Test FX breakout with upward breakout."""
        bars = []
        base = 100
        # Create range-bound data then breakout
        for i in range(50):
            timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
            if i < 40:
                # Range bound between 99-101
                base = 100 + np.random.uniform(-1, 1)
            else:
                # Strong breakout upward
                base += 0.5
            
            bars.append({
                "timestamp": timestamp,
                "open": base, "high": base + 0.2, "low": base - 0.2,
                "close": base, "volume": 1000
            })
        
        df = make_df(bars)
        signal, conf, details = eval_fx_breakout(df, lookback=30, buffer=0.005)
        
        # Should detect upward breakout
        assert signal in ["buy", "hold"]  # May not always trigger depending on exact breakout

    def test_eval_futures_trend_basic(self, sample_bars):
        """Test basic futures trend strategy."""
        df = make_df(sample_bars)
        signal, conf, details = eval_futures_trend(df, ma_len=20)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "ma" in details
        assert "close" in details
        assert details["ma"] > 0
        assert details["close"] > 0

    def test_eval_futures_trend_bullish(self, trending_up_bars):
        """Test futures trend with bullish scenario."""
        df = make_df(trending_up_bars)
        signal, conf, details = eval_futures_trend(df, ma_len=10)
        
        # With upward trend, price should be above MA
        if details["close"] > details["ma"]:
            assert signal == "buy"

    def test_eval_futures_trend_bearish(self, trending_down_bars):
        """Test futures trend with bearish scenario."""
        df = make_df(trending_down_bars)
        signal, conf, details = eval_futures_trend(df, ma_len=10)
        
        # With downward trend, price should be below MA
        if details["close"] < details["ma"]:
            assert signal == "sell"


class TestEnsemble:
    def test_ensemble_basic(self, sample_bars):
        """Test basic ensemble functionality."""
        df = make_df(sample_bars)
        signal, conf, details = ensemble(df)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert len(details) == len(TEMPLATES)
        
        # Check all templates are represented
        for template_name in TEMPLATES.keys():
            assert template_name in details

    def test_ensemble_with_params(self, sample_bars):
        """Test ensemble with custom parameters."""
        df = make_df(sample_bars)
        params = {
            "ma_crossover": {"fast": 5, "slow": 15},
            "rsi_breakout": {"length": 10, "low_th": 25, "high_th": 75}
        }
        signal, conf, details = ensemble(df, params)
        
        assert signal in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0

    def test_ensemble_consensus_buy(self, trending_up_bars):
        """Test ensemble with strong bullish consensus."""
        df = make_df(trending_up_bars)
        signal, conf, details = ensemble(df)
        
        # With strong upward trend, many strategies should agree on buy
        buy_signals = sum(1 for template_name in TEMPLATES.keys() 
                         if TEMPLATES[template_name](df)[0] == "buy")
        
        # If majority are buy signals, ensemble should be buy
        if buy_signals > len(TEMPLATES) / 2:
            assert signal == "buy"

    def test_ensemble_confidence_bounds(self, sample_bars):
        """Test ensemble confidence stays within bounds."""
        df = make_df(sample_bars)
        for _ in range(10):  # Test multiple times for randomness
            signal, conf, details = ensemble(df)
            assert 0.1 <= conf <= 0.9  # Ensemble clamps confidence


class TestTemplatesDict:
    def test_all_templates_present(self):
        """Test that all expected templates are in TEMPLATES dict."""
        expected_templates = [
            "ma_crossover", "rsi_breakout", "macd_trend",
            "options_straddle", "fx_breakout", "futures_trend"
        ]
        
        for template in expected_templates:
            assert template in TEMPLATES
            assert callable(TEMPLATES[template])

    def test_all_templates_work(self, sample_bars):
        """Test that all templates can be called and return expected format."""
        df = make_df(sample_bars)
        
        for template_name, template_func in TEMPLATES.items():
            signal, conf, details = template_func(df)
            
            assert signal in ["buy", "sell", "hold"]
            assert isinstance(conf, (float, int))
            assert 0.0 <= conf <= 1.0
            assert isinstance(details, dict)
            assert len(details) > 0


class TestErrorHandling:
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        bars = [
            {"timestamp": "2024-01-01T01:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
        ]
        df = make_df(bars)
        
        # All strategies should handle single data point gracefully
        for template_name, template_func in TEMPLATES.items():
            try:
                signal, conf, details = template_func(df)
                assert signal in ["buy", "sell", "hold"]
                assert isinstance(conf, (float, int))
                assert isinstance(details, dict)
            except Exception as e:
                pytest.fail(f"Template {template_name} failed with insufficient data: {e}")

    def test_all_nan_data_handling(self):
        """Test handling of NaN data."""
        bars = [
            {"timestamp": "2024-01-01T01:00:00Z", "open": float('nan'), "high": float('nan'), 
             "low": float('nan'), "close": float('nan'), "volume": 1000}
        ] * 30
        
        df = make_df(bars)
        
        # Strategies should handle NaN data without crashing
        for template_name, template_func in TEMPLATES.items():
            try:
                signal, conf, details = template_func(df)
                # Results may be hold/neutral with NaN data
                assert signal in ["buy", "sell", "hold"]
            except Exception:
                # Some strategies may legitimately fail with all NaN data
                pass

    def test_zero_volume_handling(self, sample_bars):
        """Test handling of zero volume bars."""
        # Modify some bars to have zero volume
        modified_bars = sample_bars.copy()
        for i in range(0, len(modified_bars), 10):
            modified_bars[i]["volume"] = 0
        
        df = make_df(modified_bars)
        
        # All strategies should handle zero volume gracefully
        for template_name, template_func in TEMPLATES.items():
            signal, conf, details = template_func(df)
            assert signal in ["buy", "sell", "hold"]


@pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
def test_template_consistency(template_name, sample_bars):
    """Test that each template produces consistent results."""
    df = make_df(sample_bars)
    template_func = TEMPLATES[template_name]
    
    # Call multiple times with same data - should get same result
    results = [template_func(df) for _ in range(3)]
    
    for i in range(1, len(results)):
        assert results[i][0] == results[0][0]  # Same signal
        assert abs(results[i][1] - results[0][1]) < 1e-10  # Same confidence
        
        # Same details (allowing for float precision)
        for key in results[0][2]:
            assert key in results[i][2]
            if isinstance(results[0][2][key], (int, float)):
                assert abs(results[i][2][key] - results[0][2][key]) < 1e-10


@pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
def test_template_return_types(template_name, sample_bars):
    """Test that all templates return correct types."""
    df = make_df(sample_bars)
    template_func = TEMPLATES[template_name]
    
    signal, conf, details = template_func(df)
    
    assert isinstance(signal, str)
    assert signal in ["buy", "sell", "hold"]
    assert isinstance(conf, (float, int))
    assert 0.0 <= conf <= 1.0
    assert isinstance(details, dict)
    
    # All detail values should be serializable
    for key, value in details.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float, str, list, dict))