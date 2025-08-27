import numpy as np
import pandas as pd

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=length, adjust=False).mean()
    roll_down = down.ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, length=20, mult=2.0):
    mid = sma(close, length)
    std = close.rolling(length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    prev_close = close.shift(1)
    tr = (high - low).abs().combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    return tr.rolling(length).mean()

# Options Greeks and Advanced Analytics
from scipy import stats
from scipy.optimize import minimize_scalar

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate Delta (price sensitivity to underlying)"""
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1) - 1

def calculate_gamma(S, K, T, r, sigma):
    """Calculate Gamma (rate of change of Delta)"""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    """Calculate Theta (time decay)"""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
    else:
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
    
    return theta

def calculate_vega(S, K, T, r, sigma):
    """Calculate Vega (volatility sensitivity)"""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% vol change
    return vega

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    """Calculate Rho (interest rate sensitivity)"""
    if T <= 0:
        return 0.0
    
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100  # Divide by 100 for 1% rate change
    else:
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
    
    return rho

def implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson method"""
    if T <= 0:
        return 0.0
    
    def objective(sigma):
        if option_type == 'call':
            theoretical_price = black_scholes_call(S, K, T, r, sigma)
        else:
            theoretical_price = black_scholes_put(S, K, T, r, sigma)
        return (theoretical_price - market_price) ** 2
    
    result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
    return result.x if result.success else 0.20  # Default to 20% if optimization fails

def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all Greeks at once for efficiency"""
    if T <= 0:
        return {
            'delta': 1.0 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    return {
        'delta': calculate_delta(S, K, T, r, sigma, option_type),
        'gamma': calculate_gamma(S, K, T, r, sigma),
        'theta': calculate_theta(S, K, T, r, sigma, option_type),
        'vega': calculate_vega(S, K, T, r, sigma),
        'rho': calculate_rho(S, K, T, r, sigma, option_type)
    }

# Currency Analysis Functions
def calculate_correlation(series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling correlation between two currency pairs"""
    return series1.rolling(window).corr(series2.rolling(window))

def calculate_carry_signal(interest_rate_diff: pd.Series, volatility: pd.Series, threshold: float = 0.02) -> pd.Series:
    """Calculate carry trade signal based on interest rate differential and volatility"""
    risk_adjusted_carry = interest_rate_diff / (volatility + 1e-9)
    return np.where(risk_adjusted_carry > threshold, 1, 
                   np.where(risk_adjusted_carry < -threshold, -1, 0))

# Futures Analysis Functions
def detect_contango_backwardation(near_price: pd.Series, far_price: pd.Series) -> pd.Series:
    """Detect contango/backwardation in futures curve"""
    ratio = far_price / near_price
    return np.where(ratio > 1.0, 1,  # Contango
                   np.where(ratio < 1.0, -1, 0))  # Backwardation

def seasonal_decompose_simple(series: pd.Series, period: int = 252) -> dict:
    """Simple seasonal decomposition for futures patterns"""
    trend = series.rolling(window=period, center=True).mean()
    detrended = series - trend
    seasonal = detrended.rolling(window=period).mean()
    residual = series - trend - seasonal
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }

# Market Regime Detection
def detect_market_regime(returns: pd.Series, window: int = 20) -> pd.Series:
    """Detect market regime based on volatility clustering"""
    volatility = returns.rolling(window).std()
    vol_threshold = volatility.rolling(window * 2).quantile(0.7)
    
    regime = np.where(volatility > vol_threshold, 1,  # High volatility regime
                     np.where(volatility < vol_threshold * 0.5, -1, 0))  # Low volatility regime
    
    return pd.Series(regime, index=returns.index)

# Cross-Asset Momentum
def cross_asset_momentum(asset_returns: dict, lookback: int = 20) -> dict:
    """Calculate cross-asset momentum scores"""
    momentum_scores = {}
    
    for asset, returns in asset_returns.items():
        momentum = returns.rolling(lookback).mean() / returns.rolling(lookback).std()
        momentum_scores[asset] = momentum
    
    return momentum_scores
