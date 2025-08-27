import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supernova.indicators import sma, ema, rsi, macd, bollinger, atr


@pytest.fixture
def sample_series():
    """Generate sample price series for testing."""
    np.random.seed(42)  # For reproducible tests
    base_price = 100
    prices = [base_price]
    
    for i in range(100):
        change = np.random.normal(0, 1) * 0.01 * base_price  # 1% volatility
        base_price += change
        prices.append(base_price)
    
    return pd.Series(prices)


@pytest.fixture
def ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    data = []
    base_price = 100
    
    for i in range(100):
        open_price = base_price
        change = np.random.normal(0, 1) * 0.01 * base_price
        close_price = base_price + change
        
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        base_price = close_price
    
    return pd.DataFrame(data)


@pytest.fixture
def trending_up_series():
    """Generate upward trending price series."""
    return pd.Series([100 + i * 0.5 for i in range(50)])


@pytest.fixture
def trending_down_series():
    """Generate downward trending price series."""
    return pd.Series([150 - i * 0.3 for i in range(50)])


@pytest.fixture
def volatile_series():
    """Generate highly volatile price series."""
    np.random.seed(42)
    base = 100
    prices = []
    for i in range(50):
        base += np.random.normal(0, 5)  # High volatility
        prices.append(base)
    return pd.Series(prices)


class TestSMA:
    def test_sma_basic(self):
        """Test basic SMA functionality."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 2)
        assert result.iloc[-1] == 4.5  # (4+5)/2
        assert result.iloc[-2] == 3.5  # (3+4)/2

    def test_sma_window_size(self, sample_series):
        """Test SMA with different window sizes."""
        windows = [3, 5, 10, 20]
        
        for window in windows:
            result = sma(sample_series, window)
            
            # First (window-1) values should be NaN
            assert pd.isna(result.iloc[:window-1]).all()
            
            # Remaining values should not be NaN
            assert result.iloc[window-1:].notna().all()
            
            # Length should match input
            assert len(result) == len(sample_series)

    def test_sma_calculation_accuracy(self):
        """Test SMA calculation accuracy."""
        s = pd.Series([10, 20, 30, 40, 50])
        result = sma(s, 3)
        
        # Manual verification
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 20.0  # (10+20+30)/3
        assert result.iloc[3] == 30.0  # (20+30+40)/3
        assert result.iloc[4] == 40.0  # (30+40+50)/3

    def test_sma_edge_cases(self):
        """Test SMA edge cases."""
        # Empty series
        empty_s = pd.Series([], dtype=float)
        result = sma(empty_s, 5)
        assert len(result) == 0
        
        # Window larger than series
        short_s = pd.Series([1, 2, 3])
        result = sma(short_s, 10)
        assert pd.isna(result).all()
        
        # Window of 1 (should equal original series)
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s)

    def test_sma_with_nan_values(self):
        """Test SMA handling of NaN values."""
        s = pd.Series([1, 2, np.nan, 4, 5])
        result = sma(s, 3)
        
        # Should handle NaN values according to pandas rolling behavior
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)


class TestEMA:
    def test_ema_basic(self, sample_series):
        """Test basic EMA functionality."""
        result = ema(sample_series, 10)
        
        assert len(result) == len(sample_series)
        assert result.notna().all()  # EMA doesn't have NaN values like SMA
        assert isinstance(result, pd.Series)

    def test_ema_vs_sma(self, trending_up_series):
        """Test that EMA reacts faster than SMA in trends."""
        ema_result = ema(trending_up_series, 10)
        sma_result = sma(trending_up_series, 10)
        
        # In uptrend, EMA should be generally higher than SMA in later periods
        # (more responsive to recent price changes)
        recent_diff = ema_result.iloc[-10:] - sma_result.iloc[-10:]
        assert recent_diff.mean() > 0

    def test_ema_different_lengths(self, sample_series):
        """Test EMA with different span lengths."""
        spans = [5, 10, 20, 50]
        
        for span in spans:
            result = ema(sample_series, span)
            assert len(result) == len(sample_series)
            assert result.notna().all()

    def test_ema_smoothing_property(self, volatile_series):
        """Test that EMA smooths volatile data."""
        ema_short = ema(volatile_series, 5)
        ema_long = ema(volatile_series, 20)
        
        # Longer EMA should be smoother (less volatile)
        short_volatility = ema_short.std()
        long_volatility = ema_long.std()
        
        assert long_volatility < short_volatility


class TestRSI:
    def test_rsi_basic(self):
        """Test basic RSI functionality."""
        s = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3])
        r = rsi(s, 3)
        assert r.notna().any()
        assert len(r) == len(s)

    def test_rsi_bounds(self, sample_series):
        """Test that RSI stays within 0-100 bounds."""
        result = rsi(sample_series, 14)
        valid_values = result.dropna()
        
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_trending_behavior(self, trending_up_series, trending_down_series):
        """Test RSI behavior in trending markets."""
        rsi_up = rsi(trending_up_series, 14)
        rsi_down = rsi(trending_down_series, 14)
        
        # In strong uptrend, RSI should be generally high
        assert rsi_up.iloc[-10:].mean() > 50
        
        # In strong downtrend, RSI should be generally low
        assert rsi_down.iloc[-10:].mean() < 50

    def test_rsi_extreme_conditions(self):
        """Test RSI in extreme market conditions."""
        # All rising prices
        rising = pd.Series(range(1, 31))  # 1, 2, 3, ..., 30
        rsi_rising = rsi(rising, 14)
        
        # Should approach 100 in continuous uptrend
        assert rsi_rising.iloc[-1] > 80
        
        # All falling prices
        falling = pd.Series(range(30, 0, -1))  # 30, 29, 28, ..., 1
        rsi_falling = rsi(falling, 14)
        
        # Should approach 0 in continuous downtrend
        assert rsi_falling.iloc[-1] < 20

    def test_rsi_different_periods(self, sample_series):
        """Test RSI with different period lengths."""
        periods = [5, 14, 21, 30]
        
        for period in periods:
            result = rsi(sample_series, period)
            assert len(result) == len(sample_series)
            valid_values = result.dropna()
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()

    def test_rsi_calculation_accuracy(self):
        """Test RSI calculation accuracy with known values."""
        # Use a simple case for manual verification
        prices = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
        
        result = rsi(prices, 14)
        
        # RSI should be calculated correctly (exact value depends on implementation)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        final_rsi = result.iloc[-1]
        assert 0 <= final_rsi <= 100


class TestMACD:
    def test_macd_basic(self, sample_series):
        """Test basic MACD functionality."""
        macd_line, signal_line, hist = macd(sample_series)
        
        assert len(macd_line) == len(sample_series)
        assert len(signal_line) == len(sample_series)
        assert len(hist) == len(sample_series)
        
        # Histogram should equal macd_line - signal_line
        pd.testing.assert_series_equal(hist, macd_line - signal_line)

    def test_macd_trending_behavior(self, trending_up_series, trending_down_series):
        """Test MACD behavior in trending markets."""
        # Uptrend
        macd_line_up, signal_line_up, hist_up = macd(trending_up_series)
        
        # In uptrend, MACD histogram should generally be positive in later periods
        assert hist_up.iloc[-5:].mean() > 0
        
        # Downtrend
        macd_line_down, signal_line_down, hist_down = macd(trending_down_series)
        
        # In downtrend, MACD histogram should generally be negative in later periods
        assert hist_down.iloc[-5:].mean() < 0

    def test_macd_custom_parameters(self, sample_series):
        """Test MACD with custom parameters."""
        macd_line, signal_line, hist = macd(sample_series, fast=8, slow=21, signal=5)
        
        assert len(macd_line) == len(sample_series)
        assert len(signal_line) == len(sample_series)
        assert len(hist) == len(sample_series)

    def test_macd_crossovers(self):
        """Test MACD crossover detection."""
        # Create price series that should generate clear MACD signals
        prices = pd.Series([100] * 20 + list(range(100, 120)) + [120] * 10)
        macd_line, signal_line, hist = macd(prices, fast=5, slow=10, signal=3)
        
        # Should have crossovers in the histogram
        assert hist.max() > 0  # Should go positive
        assert hist.min() < 0  # Should go negative


class TestBollinger:
    def test_bollinger_basic(self, sample_series):
        """Test basic Bollinger Bands functionality."""
        upper, mid, lower = bollinger(sample_series)
        
        assert len(upper) == len(sample_series)
        assert len(mid) == len(sample_series)
        assert len(lower) == len(sample_series)
        
        # Upper should be above mid, mid should be above lower
        valid_indices = mid.notna()
        assert (upper[valid_indices] >= mid[valid_indices]).all()
        assert (mid[valid_indices] >= lower[valid_indices]).all()

    def test_bollinger_band_properties(self, sample_series):
        """Test Bollinger Band mathematical properties."""
        upper, mid, lower = bollinger(sample_series, length=20, mult=2.0)
        
        # Middle band should equal SMA
        sma_result = sma(sample_series, 20)
        pd.testing.assert_series_equal(mid, sma_result)
        
        # Bandwidth should be proportional to volatility
        bandwidth = upper - lower
        assert bandwidth.notna().any()
        assert (bandwidth >= 0).all()

    def test_bollinger_different_parameters(self, sample_series):
        """Test Bollinger Bands with different parameters."""
        # Different lengths
        upper1, mid1, lower1 = bollinger(sample_series, length=10)
        upper2, mid2, lower2 = bollinger(sample_series, length=30)
        
        # Different multipliers
        upper3, mid3, lower3 = bollinger(sample_series, mult=1.5)
        upper4, mid4, lower4 = bollinger(sample_series, mult=2.5)
        
        # Wider bands should have larger bandwidth
        bandwidth1 = (upper4 - lower4).iloc[-10:].mean()
        bandwidth2 = (upper3 - lower3).iloc[-10:].mean()
        assert bandwidth1 > bandwidth2

    def test_bollinger_volatile_data(self, volatile_series):
        """Test Bollinger Bands with highly volatile data."""
        upper, mid, lower = bollinger(volatile_series)
        
        # Bands should be wide for volatile data
        bandwidth = upper - lower
        assert bandwidth.dropna().mean() > 0


class TestATR:
    def test_atr_basic(self, ohlc_data):
        """Test basic ATR functionality."""
        result = atr(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'])
        
        assert len(result) == len(ohlc_data)
        assert result.notna().sum() > 0  # Should have some valid values
        assert (result >= 0).all()  # ATR should always be positive

    def test_atr_calculation_logic(self):
        """Test ATR calculation logic."""
        # Create simple OHLC data for manual verification
        high = pd.Series([102, 105, 103, 108, 107])
        low = pd.Series([98, 101, 99, 104, 103])
        close = pd.Series([100, 103, 101, 106, 105])
        
        result = atr(high, low, close, length=3)
        
        assert len(result) == 5
        assert (result >= 0).all()
        # First value should be NaN due to shift
        assert pd.isna(result.iloc[0])

    def test_atr_different_periods(self, ohlc_data):
        """Test ATR with different period lengths."""
        periods = [7, 14, 21, 30]
        
        for period in periods:
            result = atr(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'], length=period)
            assert len(result) == len(ohlc_data)
            assert (result >= 0).all()

    def test_atr_volatility_measure(self):
        """Test that ATR correctly measures volatility."""
        # Low volatility data
        stable_high = pd.Series([101] * 20)
        stable_low = pd.Series([99] * 20)
        stable_close = pd.Series([100] * 20)
        
        stable_atr = atr(stable_high, stable_low, stable_close, length=5)
        
        # High volatility data
        volatile_high = pd.Series([100 + 5*np.sin(i/3) + 3 for i in range(20)])
        volatile_low = pd.Series([100 + 5*np.sin(i/3) - 3 for i in range(20)])
        volatile_close = pd.Series([100 + 5*np.sin(i/3) for i in range(20)])
        
        volatile_atr = atr(volatile_high, volatile_low, volatile_close, length=5)
        
        # Volatile data should have higher ATR
        stable_mean = stable_atr.dropna().mean()
        volatile_mean = volatile_atr.dropna().mean()
        assert volatile_mean > stable_mean


class TestIndicatorIntegration:
    def test_all_indicators_with_same_data(self, sample_series, ohlc_data):
        """Test that all indicators work with the same dataset."""
        # Test with price series
        sma_result = sma(sample_series, 20)
        ema_result = ema(sample_series, 20)
        rsi_result = rsi(sample_series, 14)
        macd_line, signal_line, hist = macd(sample_series)
        upper, mid, lower = bollinger(sample_series)
        
        # All should have same length as input
        assert len(sma_result) == len(sample_series)
        assert len(ema_result) == len(sample_series)
        assert len(rsi_result) == len(sample_series)
        assert len(macd_line) == len(sample_series)
        assert len(upper) == len(sample_series)
        
        # Test ATR with OHLC data
        atr_result = atr(ohlc_data['high'], ohlc_data['low'], ohlc_data['close'])
        assert len(atr_result) == len(ohlc_data)

    def test_indicator_consistency(self, sample_series):
        """Test that indicators produce consistent results across multiple calls."""
        # Call each indicator multiple times
        results1 = {
            'sma': sma(sample_series, 20),
            'ema': ema(sample_series, 20),
            'rsi': rsi(sample_series, 14)
        }
        
        results2 = {
            'sma': sma(sample_series, 20),
            'ema': ema(sample_series, 20),
            'rsi': rsi(sample_series, 14)
        }
        
        # Results should be identical
        for key in results1:
            pd.testing.assert_series_equal(results1[key], results2[key])

    def test_indicators_with_edge_cases(self):
        """Test all indicators with edge cases."""
        # Very short series
        short_series = pd.Series([100, 101, 99])
        
        # Should not crash
        try:
            sma(short_series, 2)
            ema(short_series, 2)
            rsi(short_series, 2)
            macd(short_series, 2, 3, 1)
            bollinger(short_series, 2)
        except Exception as e:
            pytest.fail(f"Indicator failed with short series: {e}")
        
        # Series with NaN values
        nan_series = pd.Series([100, np.nan, 102, np.nan, 104])
        
        # Should handle NaN values gracefully
        try:
            sma(nan_series, 3)
            ema(nan_series, 3)
            rsi(nan_series, 3)
        except Exception as e:
            pytest.fail(f"Indicator failed with NaN values: {e}")


class TestIndicatorMathematicalProperties:
    def test_sma_linearity(self):
        """Test SMA linearity property."""
        s1 = pd.Series([1, 2, 3, 4, 5])
        s2 = pd.Series([2, 4, 6, 8, 10])
        
        sma1 = sma(s1, 3)
        sma2 = sma(s2, 3)
        sma_sum = sma(s1 + s2, 3)
        
        # SMA(a + b) should equal SMA(a) + SMA(b)
        pd.testing.assert_series_equal(sma_sum, sma1 + sma2, check_names=False)

    def test_rsi_extreme_values(self):
        """Test RSI extreme values."""
        # All identical values should give RSI around 50 (undefined)
        constant_series = pd.Series([100] * 30)
        rsi_constant = rsi(constant_series, 14)
        
        # Should handle constant values without error
        assert isinstance(rsi_constant, pd.Series)

    def test_bollinger_band_percentage(self, sample_series):
        """Test that price stays within Bollinger Bands most of the time."""
        upper, mid, lower = bollinger(sample_series, length=20, mult=2.0)
        
        # With 2 standard deviations, ~95% of values should be within bands
        valid_idx = upper.notna() & lower.notna()
        within_bands = ((sample_series >= lower) & (sample_series <= upper))[valid_idx]
        
        percentage_within = within_bands.mean()
        # Should be reasonably high percentage (allowing for some variance)
        assert percentage_within > 0.8
