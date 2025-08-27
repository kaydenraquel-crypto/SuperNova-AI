from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from .indicators import (sma, ema, rsi, macd, bollinger, atr, 
                        calculate_all_greeks, implied_volatility, black_scholes_call, black_scholes_put,
                        calculate_correlation, calculate_carry_signal, detect_contango_backwardation,
                        seasonal_decompose_simple, detect_market_regime, cross_asset_momentum)

def make_df(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()

def eval_ma_crossover(df: pd.DataFrame, fast=10, slow=20) -> Tuple[str, float, dict]:
    fast_ma = sma(df["close"], fast)
    slow_ma = sma(df["close"], slow)
    signal = "hold"
    conf = 0.5
    if fast_ma.iloc[-2] < slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
        signal, conf = "buy", 0.7
    elif fast_ma.iloc[-2] > slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
        signal, conf = "sell", 0.7
    return signal, conf, {"fast": float(fast_ma.iloc[-1]), "slow": float(slow_ma.iloc[-1])}

def eval_rsi_breakout(df: pd.DataFrame, length=14, low_th=30, high_th=70) -> Tuple[str, float, dict]:
    r = rsi(df["close"], length)
    last = r.iloc[-1]
    if last < low_th:
        return "buy", 0.6, {"rsi": float(last)}
    if last > high_th:
        return "sell", 0.6, {"rsi": float(last)}
    return "hold", 0.5, {"rsi": float(last)}

def eval_macd_trend(df: pd.DataFrame, fast=12, slow=26, signal=9) -> Tuple[str, float, dict]:
    line, sig, hist = macd(df["close"], fast, slow, signal)
    last = hist.iloc[-1]
    if last > 0:
        return "buy", 0.55, {"macd_hist": float(last)}
    if last < 0:
        return "sell", 0.55, {"macd_hist": float(last)}
    return "hold", 0.5, {"macd_hist": float(last)}

TEMPLATES = {
    "ma_crossover": eval_ma_crossover,
    "rsi_breakout": eval_rsi_breakout,
    "macd_trend": eval_macd_trend,
}

def ensemble(df: pd.DataFrame, params: dict | None = None) -> Tuple[str, float, dict]:
    votes = []
    details = {}
    for name, fn in TEMPLATES.items():
        if params and name in params:
            sig, conf, det = fn(df, **params[name])
        else:
            sig, conf, det = fn(df)
        votes.append((sig, conf))
        details[name] = det
    score = sum((1 if s == "buy" else -1 if s == "sell" else 0) * c for s, c in votes)
    action = "hold"
    if score > 0.5:
        action = "buy"
    elif score < -0.5:
        action = "sell"
    confidence = min(0.9, max(0.1, abs(score)))
    return action, confidence, details


# --- Enhanced Multi-Asset Templates: Options, FX, Futures ---

def eval_options_straddle(df: pd.DataFrame, strike: float | None = None, window: int = 20, 
                         time_to_expiry: float = 0.083, risk_free_rate: float = 0.05) -> Tuple[str, float, dict]:
    """Enhanced options straddle strategy with Greeks-based risk management"""
    
    current_price = df["close"].iloc[-1]
    strike = strike or current_price  # ATM if no strike provided
    
    # Calculate historical volatility
    returns = df["close"].pct_change().dropna()
    hist_vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
    avg_vol = returns.rolling(window).std().mean() * np.sqrt(252)
    
    # Calculate theoretical straddle price
    call_price = black_scholes_call(current_price, strike, time_to_expiry, risk_free_rate, hist_vol)
    put_price = black_scholes_put(current_price, strike, time_to_expiry, risk_free_rate, hist_vol)
    straddle_price = call_price + put_price
    
    # Calculate Greeks for risk assessment
    call_greeks = calculate_all_greeks(current_price, strike, time_to_expiry, risk_free_rate, hist_vol, 'call')
    put_greeks = calculate_all_greeks(current_price, strike, time_to_expiry, risk_free_rate, hist_vol, 'put')
    
    # Combined Greeks for straddle
    net_delta = call_greeks['delta'] + put_greeks['delta']  # Should be close to 0 for ATM
    net_gamma = call_greeks['gamma'] + put_greeks['gamma']
    net_theta = call_greeks['theta'] + put_greeks['theta']
    net_vega = call_greeks['vega'] + put_greeks['vega']
    
    # Volatility expansion signal
    vol_expansion = hist_vol > 1.3 * avg_vol
    
    # Gamma squeeze detection (high gamma with low time decay)
    gamma_squeeze = net_gamma > 0.05 and abs(net_theta) < 0.5
    
    # Position sizing based on Vega exposure
    max_vega_exposure = 1000  # Maximum acceptable vega
    position_size = min(1.0, max_vega_exposure / (abs(net_vega) + 1e-9))
    
    signal = "hold"
    confidence = 0.5
    
    # Enhanced signal logic
    if vol_expansion and gamma_squeeze and time_to_expiry > 0.02:  # At least 1 week to expiry
        signal = "buy"
        confidence = 0.75
    elif hist_vol < 0.7 * avg_vol and net_theta < -0.5:  # High time decay in low vol
        signal = "sell"
        confidence = 0.6
    
    details = {
        "historical_vol": float(hist_vol),
        "avg_vol": float(avg_vol),
        "straddle_price": float(straddle_price),
        "net_delta": float(net_delta),
        "net_gamma": float(net_gamma),
        "net_theta": float(net_theta),
        "net_vega": float(net_vega),
        "position_size": float(position_size),
        "vol_expansion": vol_expansion,
        "gamma_squeeze": gamma_squeeze
    }
    
    return signal, confidence, details

def eval_fx_breakout(df: pd.DataFrame, lookback: int = 50, buffer: float = 0.001, 
                    interest_rate_diff: float = 0.0, pair_type: str = "major") -> Tuple[str, float, dict]:
    """Enhanced FX breakout strategy with carry trade and correlation analysis"""
    
    # Basic breakout logic
    high = df["high"].rolling(lookback).max().iloc[-1]
    low = df["low"].rolling(lookback).min().iloc[-1]
    close = df["close"].iloc[-1]
    
    # Calculate volatility for risk adjustment
    returns = df["close"].pct_change().dropna()
    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    avg_volatility = returns.rolling(20).std().mean() * np.sqrt(252)
    
    # Carry trade signal
    carry_signal = calculate_carry_signal(
        pd.Series([interest_rate_diff]), 
        pd.Series([volatility]), 
        threshold=0.015
    ).iloc[0] if not pd.isna(volatility) else 0
    
    # Range detection
    range_pct = (high - low) / close
    range_expansion = range_pct > 0.05  # 5% range
    
    # Momentum confirmation
    momentum_short = sma(df["close"], 5).iloc[-1] / sma(df["close"], 20).iloc[-1] - 1
    momentum_long = sma(df["close"], 20).iloc[-1] / sma(df["close"], 50).iloc[-1] - 1
    
    # Risk adjustment for different pair types
    pair_multipliers = {
        "major": 1.0,      # EUR/USD, GBP/USD, USD/JPY, etc.
        "minor": 0.8,      # EUR/GBP, AUD/CAD, etc.
        "exotic": 0.6      # USD/ZAR, EUR/TRY, etc.
    }
    risk_mult = pair_multipliers.get(pair_type, 1.0)
    
    signal = "hold"
    confidence = 0.5
    
    # Enhanced breakout logic with carry bias
    if close > high * (1 + buffer):
        base_confidence = 0.7 * risk_mult
        carry_boost = 0.1 if carry_signal > 0 else -0.05
        momentum_boost = 0.05 if momentum_short > 0.02 else -0.02
        
        confidence = min(0.9, base_confidence + carry_boost + momentum_boost)
        signal = "buy"
        
    elif close < low * (1 - buffer):
        base_confidence = 0.7 * risk_mult
        carry_boost = 0.1 if carry_signal < 0 else -0.05
        momentum_boost = 0.05 if momentum_short < -0.02 else -0.02
        
        confidence = min(0.9, base_confidence + carry_boost + momentum_boost)
        signal = "sell"
        
    # Carry-only signal in ranging markets
    elif range_expansion and abs(carry_signal) > 0 and abs(interest_rate_diff) > 0.02:
        signal = "buy" if carry_signal > 0 else "sell"
        confidence = 0.6 * risk_mult
    
    details = {
        "breakout_high": float(high),
        "breakout_low": float(low),
        "range_pct": float(range_pct),
        "volatility": float(volatility),
        "avg_volatility": float(avg_volatility),
        "carry_signal": float(carry_signal),
        "interest_rate_diff": float(interest_rate_diff),
        "momentum_short": float(momentum_short),
        "momentum_long": float(momentum_long),
        "pair_type": pair_type,
        "risk_multiplier": float(risk_mult)
    }
    
    return signal, confidence, details

def eval_futures_trend(df: pd.DataFrame, ma_len: int = 30, contract_type: str = "commodity",
                      days_to_expiry: int = 30, far_contract_price: float | None = None) -> Tuple[str, float, dict]:
    """Enhanced futures trend strategy with roll logic and seasonal patterns"""
    
    ma = df["close"].rolling(ma_len).mean().iloc[-1]
    close = df["close"].iloc[-1]
    
    # Calculate trend strength
    returns = df["close"].pct_change().dropna()
    trend_strength = returns.rolling(20).mean().iloc[-1] / (returns.rolling(20).std().iloc[-1] + 1e-9)
    
    # Seasonal decomposition for commodity futures
    if len(df) >= 252:  # Need at least 1 year of data
        seasonal_data = seasonal_decompose_simple(df["close"])
        seasonal_component = seasonal_data['seasonal'].iloc[-1] if not pd.isna(seasonal_data['seasonal'].iloc[-1]) else 0
        trend_component = seasonal_data['trend'].iloc[-1] if not pd.isna(seasonal_data['trend'].iloc[-1]) else close
    else:
        seasonal_component = 0
        trend_component = close
    
    # Contango/Backwardation analysis
    curve_signal = 0
    if far_contract_price is not None:
        curve_signal = detect_contango_backwardation(
            pd.Series([close]), 
            pd.Series([far_contract_price])
        ).iloc[0]
    
    # Roll risk assessment
    roll_risk = 1.0
    if days_to_expiry <= 10:  # High roll risk
        roll_risk = 0.5
    elif days_to_expiry <= 20:  # Medium roll risk
        roll_risk = 0.7
    
    # Contract type risk adjustments
    type_multipliers = {
        "commodity": 1.0,      # Oil, Gold, Agricultural
        "financial": 0.9,      # Interest rate, Index futures
        "currency": 0.85,      # Currency futures
        "crypto": 0.7          # Cryptocurrency futures
    }
    type_mult = type_multipliers.get(contract_type, 1.0)
    
    signal = "hold"
    base_confidence = 0.6
    
    # Enhanced trend logic
    if close > ma:
        # Bullish trend
        seasonal_boost = 0.1 if seasonal_component > 0.02 * close else 0
        curve_boost = 0.05 if curve_signal > 0 else -0.03  # Contango slightly bearish
        trend_boost = min(0.1, abs(trend_strength) * 0.05)
        
        confidence = min(0.9, base_confidence * type_mult * roll_risk + seasonal_boost + curve_boost + trend_boost)
        signal = "buy"
        
    elif close < ma:
        # Bearish trend  
        seasonal_boost = 0.1 if seasonal_component < -0.02 * close else 0
        curve_boost = 0.05 if curve_signal < 0 else -0.03  # Backwardation slightly bullish
        trend_boost = min(0.1, abs(trend_strength) * 0.05)
        
        confidence = min(0.9, base_confidence * type_mult * roll_risk + seasonal_boost + curve_boost + trend_boost)
        signal = "sell"
    else:
        confidence = 0.5 * roll_risk
    
    # Avoid trading near expiry with high roll risk
    if days_to_expiry <= 5:
        signal = "hold"
        confidence = 0.3
    
    details = {
        "ma": float(ma),
        "close": float(close),
        "trend_strength": float(trend_strength),
        "seasonal_component": float(seasonal_component),
        "trend_component": float(trend_component),
        "curve_signal": int(curve_signal),
        "days_to_expiry": days_to_expiry,
        "roll_risk": float(roll_risk),
        "contract_type": contract_type,
        "type_multiplier": float(type_mult),
        "far_contract_price": float(far_contract_price) if far_contract_price else None
    }
    
    return signal, confidence, details

# Advanced Multi-Asset Strategies

def eval_cross_asset_momentum(asset_data: dict, lookback: int = 20, min_assets: int = 3) -> Tuple[str, float, dict]:
    """Cross-asset momentum strategy comparing multiple asset classes"""
    
    if len(asset_data) < min_assets:
        return "hold", 0.5, {"error": "Insufficient assets for cross-asset analysis"}
    
    momentum_scores = {}
    returns_data = {}
    
    for asset_name, df in asset_data.items():
        if len(df) < lookback + 10:
            continue
            
        returns = df["close"].pct_change().dropna()
        returns_data[asset_name] = returns
        
        # Calculate momentum score (risk-adjusted returns)
        momentum = returns.rolling(lookback).mean().iloc[-1]
        volatility = returns.rolling(lookback).std().iloc[-1]
        momentum_scores[asset_name] = momentum / (volatility + 1e-9)
    
    if len(momentum_scores) < min_assets:
        return "hold", 0.5, {"error": "Insufficient valid momentum scores"}
    
    # Rank assets by momentum
    ranked_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    top_asset = ranked_assets[0]
    bottom_asset = ranked_assets[-1]
    
    # Market regime detection
    avg_volatility = np.mean([returns_data[asset].rolling(20).std().iloc[-1] 
                             for asset in returns_data.keys() if len(returns_data[asset]) >= 20])
    
    regime = "normal"
    if avg_volatility > 0.03:  # High volatility regime
        regime = "high_vol"
    elif avg_volatility < 0.01:  # Low volatility regime
        regime = "low_vol"
    
    # Signal generation
    momentum_spread = top_asset[1] - bottom_asset[1]
    signal = "hold"
    confidence = 0.5
    
    if momentum_spread > 0.02:  # Strong momentum divergence
        signal = "buy"  # Favor top momentum asset
        confidence = min(0.8, 0.5 + momentum_spread * 10)
    elif momentum_spread < -0.02:
        signal = "sell"  # Favor bottom momentum asset (contrarian)
        confidence = min(0.8, 0.5 + abs(momentum_spread) * 10)
    
    # Regime adjustments
    if regime == "high_vol":
        confidence *= 0.8  # Reduce confidence in volatile markets
    elif regime == "low_vol":
        confidence *= 1.1  # Increase confidence in stable markets
    
    details = {
        "momentum_scores": {k: float(v) for k, v in momentum_scores.items()},
        "top_asset": top_asset[0],
        "top_momentum": float(top_asset[1]),
        "bottom_asset": bottom_asset[0], 
        "bottom_momentum": float(bottom_asset[1]),
        "momentum_spread": float(momentum_spread),
        "market_regime": regime,
        "avg_volatility": float(avg_volatility)
    }
    
    return signal, confidence, details

def eval_asset_rotation(asset_data: dict, lookback: int = 60, rotation_threshold: float = 0.05) -> Tuple[str, float, dict]:
    """Asset rotation strategy based on relative performance"""
    
    if len(asset_data) < 2:
        return "hold", 0.5, {"error": "Need at least 2 assets for rotation"}
    
    performance_scores = {}
    volatility_scores = {}
    
    for asset_name, df in asset_data.items():
        if len(df) < lookback + 10:
            continue
        
        # Calculate total return over lookback period
        start_price = df["close"].iloc[-(lookback+1)]
        end_price = df["close"].iloc[-1]
        total_return = (end_price / start_price) - 1
        
        # Calculate volatility
        returns = df["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Sharpe-like score
        risk_free_rate = 0.02  # 2% risk-free rate assumption
        sharpe_score = (total_return - risk_free_rate/4) / (volatility + 1e-9)  # Quarterly return
        
        performance_scores[asset_name] = sharpe_score
        volatility_scores[asset_name] = volatility
    
    if len(performance_scores) < 2:
        return "hold", 0.5, {"error": "Insufficient performance data"}
    
    # Find best performing asset
    best_asset = max(performance_scores, key=performance_scores.get)
    best_score = performance_scores[best_asset]
    
    # Calculate average performance
    avg_performance = np.mean(list(performance_scores.values()))
    
    signal = "hold"
    confidence = 0.5
    
    # Rotation signal
    if best_score > avg_performance + rotation_threshold:
        signal = "buy"  # Rotate to best performer
        confidence = min(0.85, 0.6 + (best_score - avg_performance))
    elif best_score < avg_performance - rotation_threshold:
        signal = "sell"  # Avoid worst performer
        confidence = min(0.75, 0.6 + abs(best_score - avg_performance))
    
    details = {
        "performance_scores": {k: float(v) for k, v in performance_scores.items()},
        "volatility_scores": {k: float(v) for k, v in volatility_scores.items()},
        "best_asset": best_asset,
        "best_score": float(best_score),
        "avg_performance": float(avg_performance),
        "rotation_signal": signal != "hold"
    }
    
    return signal, confidence, details

def eval_risk_parity(asset_data: dict, target_vol: float = 0.15, rebalance_threshold: float = 0.05) -> Tuple[str, float, dict]:
    """Risk parity strategy - equal risk contribution from each asset"""
    
    if len(asset_data) < 2:
        return "hold", 0.5, {"error": "Need at least 2 assets for risk parity"}
    
    weights = {}
    volatilities = {}
    correlations = {}
    
    # Calculate individual asset volatilities
    returns_matrix = {}
    for asset_name, df in asset_data.items():
        if len(df) < 30:
            continue
        returns = df["close"].pct_change().dropna()
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        volatilities[asset_name] = vol
        returns_matrix[asset_name] = returns
    
    if len(volatilities) < 2:
        return "hold", 0.5, {"error": "Insufficient volatility data"}
    
    # Calculate correlation matrix (simplified - pairwise)
    asset_names = list(volatilities.keys())
    for i, asset1 in enumerate(asset_names):
        for j, asset2 in enumerate(asset_names[i+1:], i+1):
            if len(returns_matrix[asset1]) >= 20 and len(returns_matrix[asset2]) >= 20:
                min_len = min(len(returns_matrix[asset1]), len(returns_matrix[asset2]))
                corr = returns_matrix[asset1].iloc[-min_len:].corr(returns_matrix[asset2].iloc[-min_len:])
                correlations[f"{asset1}_{asset2}"] = corr
    
    # Risk parity weights (inverse volatility)
    inv_vol_sum = sum(1/vol for vol in volatilities.values())
    for asset in volatilities:
        weights[asset] = (1/volatilities[asset]) / inv_vol_sum
    
    # Portfolio volatility estimate
    portfolio_vol = np.sqrt(sum(weights[asset]**2 * volatilities[asset]**2 for asset in weights))
    
    # Rebalancing signal
    vol_deviation = abs(portfolio_vol - target_vol) / target_vol
    
    signal = "hold"
    confidence = 0.5
    
    if vol_deviation > rebalance_threshold:
        if portfolio_vol > target_vol:
            signal = "sell"  # Reduce risk
            confidence = min(0.8, 0.5 + vol_deviation)
        else:
            signal = "buy"  # Increase risk
            confidence = min(0.8, 0.5 + vol_deviation)
    
    details = {
        "weights": {k: float(v) for k, v in weights.items()},
        "volatilities": {k: float(v) for k, v in volatilities.items()},
        "portfolio_vol": float(portfolio_vol),
        "target_vol": float(target_vol),
        "vol_deviation": float(vol_deviation),
        "rebalance_needed": vol_deviation > rebalance_threshold,
        "correlations": {k: float(v) for k, v in correlations.items()}
    }
    
    return signal, confidence, details

# Enhanced ensemble with adaptive weighting
def adaptive_ensemble(df: pd.DataFrame, asset_data: dict = None, params: dict | None = None) -> Tuple[str, float, dict]:
    """Adaptive ensemble with market regime detection and dynamic strategy weighting"""
    
    # Detect market regime
    returns = df["close"].pct_change().dropna()
    if len(returns) >= 20:
        regime_series = detect_market_regime(returns)
        current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else 0
    else:
        current_regime = 0
    
    regime_name = {-1: "low_vol", 0: "normal", 1: "high_vol"}.get(current_regime, "normal")
    
    # Strategy weights based on market regime
    regime_weights = {
        "low_vol": {
            "ma_crossover": 0.3,
            "rsi_breakout": 0.2,
            "macd_trend": 0.25,
            "options_straddle": 0.1,
            "fx_breakout": 0.1,
            "futures_trend": 0.05
        },
        "normal": {
            "ma_crossover": 0.25,
            "rsi_breakout": 0.2,
            "macd_trend": 0.2,
            "options_straddle": 0.15,
            "fx_breakout": 0.1,
            "futures_trend": 0.1
        },
        "high_vol": {
            "ma_crossover": 0.15,
            "rsi_breakout": 0.15,
            "macd_trend": 0.15,
            "options_straddle": 0.25,
            "fx_breakout": 0.15,
            "futures_trend": 0.15
        }
    }
    
    weights = regime_weights.get(regime_name, regime_weights["normal"])
    
    votes = []
    details = {"regime": regime_name, "regime_score": int(current_regime), "strategy_weights": weights}
    
    # Standard strategies
    for name, fn in TEMPLATES.items():
        if name in ["cross_asset_momentum", "asset_rotation", "risk_parity"]:
            continue  # Skip multi-asset strategies for single-asset ensemble
            
        try:
            if params and name in params:
                sig, conf, det = fn(df, **params[name])
            else:
                sig, conf, det = fn(df)
            
            weight = weights.get(name, 0.1)
            votes.append((sig, conf * weight))
            details[name] = det
        except Exception as e:
            details[name] = {"error": str(e)}
    
    # Calculate weighted score
    weighted_score = sum((1 if s == "buy" else -1 if s == "sell" else 0) * c for s, c in votes)
    total_weight = sum(c for _, c in votes)
    
    if total_weight > 0:
        normalized_score = weighted_score / total_weight
    else:
        normalized_score = 0
    
    action = "hold"
    if normalized_score > 0.3:
        action = "buy"
    elif normalized_score < -0.3:
        action = "sell"
    
    confidence = min(0.9, max(0.1, abs(normalized_score)))
    details["weighted_score"] = float(normalized_score)
    details["total_weight"] = float(total_weight)
    
    return action, confidence, details

TEMPLATES.update({
    "options_straddle": eval_options_straddle,
    "fx_breakout": eval_fx_breakout,
    "futures_trend": eval_futures_trend,
    "cross_asset_momentum": eval_cross_asset_momentum,
    "asset_rotation": eval_asset_rotation,
    "risk_parity": eval_risk_parity,
    "adaptive_ensemble": adaptive_ensemble,
})
