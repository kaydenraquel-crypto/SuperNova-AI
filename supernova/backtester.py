import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, time
import warnings
from .strategy_engine import make_df, TEMPLATES, ensemble, adaptive_ensemble

# VectorBT imports with error handling for backward compatibility
try:
    import vectorbt as vbt
    import talib
    from numba import jit
    VBT_AVAILABLE = True
except ImportError as e:
    VBT_AVAILABLE = False
    print(f"VectorBT not available: {e}. Using legacy backtester.")

@dataclass
class TradeRecord:
    """Enhanced trade record with comprehensive trade-level analytics"""
    timestamp: str
    type: str  # 'buy', 'sell', 'cover', 'short', 'margin_call'
    price: float
    shares: float
    cost: Optional[float] = None
    proceeds: Optional[float] = None
    fees: float = 0.0
    margin: float = 0.0
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position_size: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    hold_time: int = 0  # bars held
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    slippage: float = 0.0
    market_impact: float = 0.0
    partial_fill: bool = False
    after_hours: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'type': self.type,
            'price': self.price,
            'shares': self.shares,
            'cost': self.cost,
            'proceeds': self.proceeds,
            'fees': self.fees,
            'margin': self.margin,
            'pnl': self.pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'position_size': self.position_size,
            'equity_before': self.equity_before,
            'equity_after': self.equity_after,
            'max_adverse_excursion': self.max_adverse_excursion,
            'max_favorable_excursion': self.max_favorable_excursion,
            'hold_time': self.hold_time,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'partial_fill': self.partial_fill,
            'after_hours': self.after_hours
        }

@dataclass
class AssetConfig:
    """Enhanced asset configuration with professional parameters"""
    asset_type: str
    min_margin: float
    overnight_margin: float
    max_leverage: float
    tick_size: float = 0.01
    min_price_increment: float = 0.01
    trading_hours: Tuple[time, time] = (time(9, 30), time(16, 0))
    after_hours: bool = False
    pre_market: bool = False
    settlement_days: int = 2  # T+2 for stocks
    min_trade_size: float = 1.0
    max_trade_size: float = float('inf')
    base_slippage_bps: float = 5.0  # 5 basis points
    impact_coefficient: float = 0.1  # Market impact
    contract_size: float = 1.0  # For futures/options
    point_value: float = 1.0  # Dollar value per point
    margin_call_level: float = 1.25  # 125% margin requirement
    force_close_level: float = 1.1  # 110% force liquidation

# Enhanced asset configurations
ASSET_CONFIGS = {
    "stock": AssetConfig(
        asset_type="stock",
        min_margin=0.5, overnight_margin=0.5, max_leverage=2.0,
        tick_size=0.01, trading_hours=(time(9, 30), time(16, 0)),
        after_hours=True, pre_market=True, settlement_days=2,
        base_slippage_bps=2.0, impact_coefficient=0.05
    ),
    "forex": AssetConfig(
        asset_type="forex", 
        min_margin=0.02, overnight_margin=0.02, max_leverage=50.0,
        tick_size=0.00001, trading_hours=(time(17, 0), time(17, 0)),  # 24h
        after_hours=True, pre_market=True, settlement_days=0,
        base_slippage_bps=1.0, impact_coefficient=0.02,
        margin_call_level=1.15, force_close_level=1.05
    ),
    "futures": AssetConfig(
        asset_type="futures",
        min_margin=0.05, overnight_margin=0.1, max_leverage=20.0,
        tick_size=0.25, contract_size=50.0, point_value=50.0,  # E-mini S&P
        trading_hours=(time(9, 30), time(16, 15)),
        after_hours=True, settlement_days=0,
        base_slippage_bps=3.0, impact_coefficient=0.08
    ),
    "options": AssetConfig(
        asset_type="options",
        min_margin=1.0, overnight_margin=1.0, max_leverage=1.0,
        tick_size=0.01, contract_size=100.0, point_value=100.0,
        trading_hours=(time(9, 30), time(16, 0)),
        settlement_days=1, base_slippage_bps=10.0, impact_coefficient=0.15
    ),
    "crypto": AssetConfig(
        asset_type="crypto",
        min_margin=0.1, overnight_margin=0.1, max_leverage=10.0,
        tick_size=0.01, trading_hours=(time(0, 0), time(23, 59)),  # 24h
        after_hours=True, pre_market=True, settlement_days=0,
        base_slippage_bps=15.0, impact_coefficient=0.2,
        margin_call_level=1.2, force_close_level=1.05
    )
}

def run_backtest(bars: list[dict], template: str, params: dict | None = None,
                 start_cash: float = 10000.0, fee_rate: float = 0.0005, allow_short: bool = False,
                 leverage: float = 1.0, margin_req: float | None = None, 
                 contract_multiplier: float | None = None, asset_type: str = "stock",
                 enable_slippage: bool = True, enable_market_impact: bool = True,
                 enable_after_hours: bool = False, partial_fill_prob: float = 0.02,
                 max_position_size: float = 0.95, risk_free_rate: float = 0.02):
    """
    Professional-grade backtester with comprehensive features:
    - Enhanced margin & leverage with realistic asset class parameters
    - Contract multipliers and proper P&L calculations
    - Slippage, market impact, and partial fill simulation
    - After-hours and pre-market trading simulation
    - Advanced risk metrics and trade-level analytics
    - Margin calls and forced liquidation
    """
    
    df = make_df(bars)
    if len(df) < 50:
        return {"error": "Insufficient data for backtesting (minimum 50 bars required)"}
    
    # Get enhanced asset configuration
    asset_config = ASSET_CONFIGS.get(asset_type, ASSET_CONFIGS["stock"])
    
    # Override with user parameters if provided
    effective_leverage = min(leverage, asset_config.max_leverage)
    margin_requirement = margin_req if margin_req is not None else asset_config.min_margin
    contract_mult = contract_multiplier if contract_multiplier is not None else asset_config.contract_size
    point_value = asset_config.point_value
    
    # Initialize backtesting state
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    timestamps = df.index
    
    pos = 0  # -1: short, 0: flat, 1: long
    cash = start_cash
    shares = 0.0
    equity_curve = []
    margin_used = 0.0
    trades: List[TradeRecord] = []
    
    # Trade tracking for analytics
    current_trade_entry = None
    max_adverse_excursion = 0.0
    max_favorable_excursion = 0.0
    trade_hold_time = 0
    
    # Portfolio-level tracking
    daily_returns = []
    equity_peaks = []
    underwater_curve = []
    total_fees_paid = 0.0
    total_slippage = 0.0
    margin_call_count = 0

    for i in range(max(30, 50), len(df)):
        sub = df.iloc[:i+1]
        
        # Strategy evaluation
        if template == "ensemble":
            action, conf, details = ensemble(sub, params or {})
        elif template == "adaptive_ensemble":
            action, conf, details = adaptive_ensemble(sub, params=params)
        else:
            fn = TEMPLATES.get(template)
            if not fn:
                raise ValueError(f"Unknown template {template}")
            
            try:
                action, conf, details = fn(sub, **(params or {}))
            except Exception as e:
                action, conf, details = "hold", 0.5, {"error": str(e)}

        price = close[i]
        
        current_time = timestamps[i].time() if hasattr(timestamps[i], 'time') else time(12, 0)
        is_after_hours = enable_after_hours and (
            current_time < asset_config.trading_hours[0] or 
            current_time > asset_config.trading_hours[1]
        )
        
        # Calculate available buying power with enhanced margin logic
        available_cash = cash - margin_used
        current_margin_req = asset_config.overnight_margin if is_after_hours else margin_requirement
        buying_power = available_cash * effective_leverage
        
        # Calculate current equity with unrealized P&L
        if shares != 0:
            position_value = shares * price * contract_mult * point_value
            if pos == 1:  # Long position
                unrealized_pnl = (price - current_trade_entry) * shares * contract_mult * point_value if current_trade_entry else 0
            else:  # Short position
                unrealized_pnl = (current_trade_entry - price) * abs(shares) * contract_mult * point_value if current_trade_entry else 0
            
            current_equity = cash + unrealized_pnl
            
            # Track MAE and MFE for current trade
            if current_trade_entry:
                if pos == 1:  # Long position
                    adverse = (current_trade_entry - low[i]) * shares * contract_mult * point_value
                    favorable = (high[i] - current_trade_entry) * shares * contract_mult * point_value
                else:  # Short position
                    adverse = (high[i] - current_trade_entry) * abs(shares) * contract_mult * point_value
                    favorable = (current_trade_entry - low[i]) * abs(shares) * contract_mult * point_value
                
                max_adverse_excursion = min(max_adverse_excursion, adverse)
                max_favorable_excursion = max(max_favorable_excursion, favorable)
                trade_hold_time += 1
        else:
            current_equity = cash
            unrealized_pnl = 0
        
        # Enhanced margin call logic
        if margin_used > 0:
            margin_ratio = current_equity / margin_used
            
            # Margin call at 125% (or asset-specific level)
            if margin_ratio < asset_config.margin_call_level:
                margin_call_count += 1
                
                # Force liquidation at 110% (or asset-specific level)
                if margin_ratio < asset_config.force_close_level and pos != 0:
                    # Calculate slippage for forced liquidation (higher than normal)
                    forced_slippage = asset_config.base_slippage_bps * 3.0 / 10000  # 3x normal slippage
                    
                    if pos == 1:
                        liquidation_price = price * (1 - forced_slippage)
                        proceeds = shares * liquidation_price * contract_mult * point_value
                        fees = proceeds * fee_rate
                        cash += proceeds - fees
                        
                        trade_pnl = proceeds - (current_trade_entry * shares * contract_mult * point_value)
                    else:  # pos == -1
                        liquidation_price = price * (1 + forced_slippage)
                        cost = abs(shares) * liquidation_price * contract_mult * point_value
                        fees = cost * fee_rate
                        cash -= cost + fees
                        
                        trade_pnl = (current_trade_entry * abs(shares) * contract_mult * point_value) - cost
                    
                    # Record forced liquidation trade
                    trade_record = TradeRecord(
                        timestamp=str(timestamps[i]),
                        type="margin_call",
                        price=liquidation_price,
                        shares=abs(shares),
                        fees=fees,
                        pnl=trade_pnl,
                        equity_before=current_equity,
                        equity_after=cash,
                        max_adverse_excursion=max_adverse_excursion,
                        max_favorable_excursion=max_favorable_excursion,
                        hold_time=trade_hold_time,
                        entry_price=current_trade_entry,
                        exit_price=liquidation_price,
                        slippage=forced_slippage * price,
                        after_hours=is_after_hours
                    )
                    trades.append(trade_record)
                    
                    # Reset position
                    margin_used = 0.0
                    shares = 0
                    pos = 0
                    current_trade_entry = None
                    max_adverse_excursion = 0.0
                    max_favorable_excursion = 0.0
                    trade_hold_time = 0
                    
                    total_fees_paid += fees
                    total_slippage += forced_slippage * price * abs(shares)
        
        # Enhanced position sizing with risk management
        base_position_size = max_position_size  # User-defined max position size
        confidence_multiplier = conf  # Scale position by confidence
        volatility_adjustment = 1.0
        
        # Calculate recent volatility for position sizing
        if i >= 20:
            recent_returns = np.diff(np.log(close[max(0, i-20):i+1]))
            volatility = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 1 else 0.2
            # Reduce position size in high volatility
            volatility_adjustment = max(0.3, min(1.0, 0.5 / (volatility + 0.1)))
        
        position_size_multiplier = base_position_size * confidence_multiplier * volatility_adjustment
        
        if action == "buy" and pos <= 0:
            # Check if we can trade (market hours or after-hours enabled)
            if not enable_after_hours and is_after_hours:
                continue  # Skip trade if after hours and not enabled
            # Close short position first
            if pos == -1:
                # Calculate slippage and market impact for covering
                volume_impact = 0.0
                if enable_market_impact:
                    position_volume = abs(shares) * price * contract_mult
                    volume_impact = min(0.01, position_volume * asset_config.impact_coefficient / 1000000)
                
                slippage = 0.0
                if enable_slippage:
                    base_slippage = asset_config.base_slippage_bps / 10000
                    slippage = base_slippage + volume_impact
                    if is_after_hours:
                        slippage *= 2.0  # Higher slippage after hours
                
                # Check for partial fill
                fill_ratio = 1.0
                if np.random.random() < partial_fill_prob:
                    fill_ratio = np.random.uniform(0.7, 0.95)
                
                actual_shares = abs(shares) * fill_ratio
                cover_price = price * (1 + slippage)  # Adverse slippage when covering
                cover_cost = actual_shares * cover_price * contract_mult * point_value
                fees = cover_cost * fee_rate
                
                cash -= cover_cost + fees
                
                # Calculate P&L for the short trade
                if current_trade_entry:
                    trade_pnl = (current_trade_entry * actual_shares * contract_mult * point_value) - cover_cost
                else:
                    trade_pnl = -cover_cost
                
                trade_record = TradeRecord(
                    timestamp=str(timestamps[i]),
                    type="cover",
                    price=cover_price,
                    shares=actual_shares,
                    cost=cover_cost,
                    fees=fees,
                    pnl=trade_pnl,
                    equity_before=current_equity,
                    equity_after=cash,
                    max_adverse_excursion=max_adverse_excursion,
                    max_favorable_excursion=max_favorable_excursion,
                    hold_time=trade_hold_time,
                    entry_price=current_trade_entry,
                    exit_price=cover_price,
                    slippage=slippage * price,
                    market_impact=volume_impact * price,
                    partial_fill=fill_ratio < 1.0,
                    after_hours=is_after_hours
                )
                trades.append(trade_record)
                
                # Update position
                shares += actual_shares if fill_ratio < 1.0 else 0  # Remaining shares if partial fill
                if fill_ratio >= 1.0:
                    margin_used = 0.0
                    pos = 0
                    current_trade_entry = None
                    max_adverse_excursion = 0.0
                    max_favorable_excursion = 0.0
                    trade_hold_time = 0
                else:
                    shares = -abs(shares) * (1 - fill_ratio)  # Remaining short position
                
                total_fees_paid += fees
                total_slippage += slippage * price * actual_shares
            
            # Open long position
            if buying_power > 0 and pos == 0:  # Only if no current position
                max_shares = (buying_power * position_size_multiplier) / (price * contract_mult * point_value)
                
                # Apply minimum/maximum trade size constraints
                max_shares = max(asset_config.min_trade_size, 
                               min(max_shares, asset_config.max_trade_size))
                
                if max_shares > 0:
                    # Calculate slippage and market impact
                    volume_impact = 0.0
                    if enable_market_impact:
                        position_volume = max_shares * price * contract_mult
                        volume_impact = min(0.01, position_volume * asset_config.impact_coefficient / 1000000)
                    
                    slippage = 0.0
                    if enable_slippage:
                        base_slippage = asset_config.base_slippage_bps / 10000
                        slippage = base_slippage + volume_impact
                        if is_after_hours:
                            slippage *= 2.0  # Higher slippage after hours
                    
                    # Check for partial fill
                    fill_ratio = 1.0
                    if np.random.random() < partial_fill_prob:
                        fill_ratio = np.random.uniform(0.7, 0.95)
                    
                    shares_to_buy = max_shares * fill_ratio
                    buy_price = price * (1 + slippage)  # Adverse slippage when buying
                    cost = shares_to_buy * buy_price * contract_mult * point_value
                    fees = cost * fee_rate
                    margin_needed = cost * current_margin_req / effective_leverage
                    
                    if cash >= margin_needed + fees:
                        cash -= fees
                        margin_used = margin_needed
                        shares = shares_to_buy
                        pos = 1
                        current_trade_entry = buy_price
                        max_adverse_excursion = 0.0
                        max_favorable_excursion = 0.0
                        trade_hold_time = 0
                        
                        trade_record = TradeRecord(
                            timestamp=str(timestamps[i]),
                            type="buy",
                            price=buy_price,
                            shares=shares_to_buy,
                            cost=cost,
                            fees=fees,
                            margin=margin_needed,
                            equity_before=current_equity,
                            equity_after=cash + (shares_to_buy * buy_price * contract_mult * point_value),
                            entry_price=buy_price,
                            slippage=slippage * price,
                            market_impact=volume_impact * price,
                            partial_fill=fill_ratio < 1.0,
                            after_hours=is_after_hours
                        )
                        trades.append(trade_record)
                        
                        total_fees_paid += fees
                        total_slippage += slippage * price * shares_to_buy
        
        elif action == "sell" and pos >= 0:
            # Check if we can trade (market hours or after-hours enabled)
            if not enable_after_hours and is_after_hours:
                continue  # Skip trade if after hours and not enabled
            # Close long position first
            if pos == 1:
                # Calculate slippage and market impact for selling
                volume_impact = 0.0
                if enable_market_impact:
                    position_volume = shares * price * contract_mult
                    volume_impact = min(0.01, position_volume * asset_config.impact_coefficient / 1000000)
                
                slippage = 0.0
                if enable_slippage:
                    base_slippage = asset_config.base_slippage_bps / 10000
                    slippage = base_slippage + volume_impact
                    if is_after_hours:
                        slippage *= 2.0  # Higher slippage after hours
                
                # Check for partial fill
                fill_ratio = 1.0
                if np.random.random() < partial_fill_prob:
                    fill_ratio = np.random.uniform(0.7, 0.95)
                
                actual_shares = shares * fill_ratio
                sell_price = price * (1 - slippage)  # Adverse slippage when selling
                proceeds = actual_shares * sell_price * contract_mult * point_value
                fees = proceeds * fee_rate
                
                cash += proceeds - fees
                
                # Calculate P&L for the long trade
                if current_trade_entry:
                    trade_pnl = proceeds - (current_trade_entry * actual_shares * contract_mult * point_value)
                else:
                    trade_pnl = 0
                
                trade_record = TradeRecord(
                    timestamp=str(timestamps[i]),
                    type="sell",
                    price=sell_price,
                    shares=actual_shares,
                    proceeds=proceeds,
                    fees=fees,
                    pnl=trade_pnl,
                    equity_before=current_equity,
                    equity_after=cash,
                    max_adverse_excursion=max_adverse_excursion,
                    max_favorable_excursion=max_favorable_excursion,
                    hold_time=trade_hold_time,
                    entry_price=current_trade_entry,
                    exit_price=sell_price,
                    slippage=slippage * price,
                    market_impact=volume_impact * price,
                    partial_fill=fill_ratio < 1.0,
                    after_hours=is_after_hours
                )
                trades.append(trade_record)
                
                # Update position
                shares -= actual_shares
                if fill_ratio >= 1.0:
                    margin_used = 0.0
                    pos = 0
                    current_trade_entry = None
                    max_adverse_excursion = 0.0
                    max_favorable_excursion = 0.0
                    trade_hold_time = 0
                else:
                    shares = shares  # Remaining long position
                
                total_fees_paid += fees
                total_slippage += slippage * price * actual_shares
            
            # Open short position (if allowed)
            if allow_short and buying_power > 0 and pos == 0:  # Only if no current position
                max_shares = (buying_power * position_size_multiplier) / (price * contract_mult * point_value)
                
                # Apply minimum/maximum trade size constraints
                max_shares = max(asset_config.min_trade_size, 
                               min(max_shares, asset_config.max_trade_size))
                
                if max_shares > 0:
                    # Calculate slippage and market impact
                    volume_impact = 0.0
                    if enable_market_impact:
                        position_volume = max_shares * price * contract_mult
                        volume_impact = min(0.01, position_volume * asset_config.impact_coefficient / 1000000)
                    
                    slippage = 0.0
                    if enable_slippage:
                        base_slippage = asset_config.base_slippage_bps / 10000
                        slippage = base_slippage + volume_impact
                        if is_after_hours:
                            slippage *= 2.0  # Higher slippage after hours
                    
                    # Check for partial fill
                    fill_ratio = 1.0
                    if np.random.random() < partial_fill_prob:
                        fill_ratio = np.random.uniform(0.7, 0.95)
                    
                    shares_to_short = max_shares * fill_ratio
                    short_price = price * (1 - slippage)  # Favorable slippage when shorting
                    proceeds = shares_to_short * short_price * contract_mult * point_value
                    fees = proceeds * fee_rate
                    margin_needed = proceeds * current_margin_req / effective_leverage
                    
                    if cash >= margin_needed + fees:
                        cash += proceeds - fees
                        margin_used = margin_needed
                        shares = -shares_to_short  # Negative for short position
                        pos = -1
                        current_trade_entry = short_price
                        max_adverse_excursion = 0.0
                        max_favorable_excursion = 0.0
                        trade_hold_time = 0
                        
                        trade_record = TradeRecord(
                            timestamp=str(timestamps[i]),
                            type="short",
                            price=short_price,
                            shares=shares_to_short,
                            proceeds=proceeds,
                            fees=fees,
                            margin=margin_needed,
                            equity_before=current_equity,
                            equity_after=cash,
                            entry_price=short_price,
                            slippage=slippage * price,
                            market_impact=volume_impact * price,
                            partial_fill=fill_ratio < 1.0,
                            after_hours=is_after_hours
                        )
                        trades.append(trade_record)
                        
                        total_fees_paid += fees
                        total_slippage += slippage * price * shares_to_short

        # Calculate current equity with proper unrealized P&L
        if shares != 0 and current_trade_entry:
            if pos == 1:  # Long position
                unrealized_pnl = (price - current_trade_entry) * shares * contract_mult * point_value
            else:  # Short position (shares is negative)
                unrealized_pnl = (current_trade_entry - price) * abs(shares) * contract_mult * point_value
            
            current_equity = cash + unrealized_pnl
        else:
            current_equity = cash
            unrealized_pnl = 0
        
        # Record daily equity for returns calculation
        equity_curve.append(max(current_equity, 0))  # Prevent negative equity
        
        # Track underwater curve (drawdown from peak)
        if equity_curve:
            current_peak = max(equity_peaks) if equity_peaks else start_cash
            if current_equity > current_peak:
                equity_peaks.append(current_equity)
                underwater_curve.append(0)
            else:
                equity_peaks.append(current_peak)
                underwater_curve.append((current_peak - current_equity) / current_peak)
        
        # Calculate daily returns
        if len(equity_curve) > 1:
            daily_return = (equity_curve[-1] / equity_curve[-2]) - 1
            daily_returns.append(daily_return)

    # Calculate comprehensive performance metrics
    if len(equity_curve) > 1:
        returns = np.diff(np.log(np.array(equity_curve) + 1e-9))
        periods = len(returns)
        
        # Core metrics
        cagr = (equity_curve[-1] / (start_cash + 1e-9)) ** (252/periods) - 1 if periods > 0 else 0
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = (returns.mean() * 252 - risk_free_rate) / (vol + 1e-9) if vol > 0 else 0
        max_dd = _max_drawdown(np.array(equity_curve))
        equity_win_rate = _win_rate(np.array(equity_curve))
        
        # Advanced risk metrics
        sortino = _calculate_sortino_ratio(returns, risk_free_rate)
        var_95 = _calculate_var(returns, 0.95)
        var_99 = _calculate_var(returns, 0.99)
        calmar = _calculate_calmar_ratio(cagr, max_dd)
        recovery_factor = _calculate_recovery_factor(cagr * 100, max_dd * 100)
        information_ratio = _calculate_information_ratio(returns)
        
        # Trade-level analytics
        trade_analytics = _calculate_trade_analytics(trades)
        trades_win_rate = trade_analytics["profit_factor"]
        
        # Cost analysis
        total_fees_ratio = total_fees_paid / start_cash if start_cash > 0 else 0
        total_slippage_ratio = total_slippage / start_cash if start_cash > 0 else 0
        
        # Margin and leverage tracking
        margin_trades = [t for t in trades if hasattr(t, 'margin') and t.margin > 0]
        avg_margin_usage = np.mean([t.margin for t in margin_trades]) if margin_trades else 0
        max_leverage_used = max([
            (t.cost if hasattr(t, 'cost') and t.cost else t.shares * t.price * contract_mult * point_value) / 
            (t.margin + 1e-9) for t in margin_trades
        ]) if margin_trades else effective_leverage
        
    else:
        cagr = vol = sharpe = max_dd = equity_win_rate = sortino = var_95 = var_99 = 0
        calmar = recovery_factor = information_ratio = 0
        trade_analytics = _calculate_trade_analytics([])
        total_fees_ratio = total_slippage_ratio = 0
        avg_margin_usage = max_leverage_used = 0

    # Compile comprehensive results
    result = {
        # Core Performance
        "final_equity": float(equity_curve[-1]) if equity_curve else start_cash,
        "total_return": float((equity_curve[-1] / start_cash - 1) * 100) if equity_curve else 0,
        "CAGR": float(cagr * 100),
        "volatility": float(vol * 100),
        
        # Risk Metrics
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "MaxDrawdown": float(max_dd * 100),
        "RecoveryFactor": float(recovery_factor),
        "InformationRatio": float(information_ratio),
        "VaR_95": float(var_95 * 100),
        "VaR_99": float(var_99 * 100),
        
        # Trade Analytics
        "WinRate": float(equity_win_rate * 100),
        "n_trades": len(trades),
        **{k: v for k, v in trade_analytics.items()},  # Unpack all trade analytics
        
        # Cost Analysis
        "total_fees_paid": float(total_fees_paid),
        "total_slippage": float(total_slippage),
        "fees_ratio": float(total_fees_ratio),
        "slippage_ratio": float(total_slippage_ratio),
        
        # Position Management
        "avg_margin_usage": float(avg_margin_usage),
        "max_leverage_used": float(max_leverage_used),
        "margin_calls": margin_call_count,
        
        # Configuration
        "asset_type": asset_type,
        "leverage": effective_leverage,
        "contract_multiplier": contract_mult,
        "point_value": point_value,
        "enable_slippage": enable_slippage,
        "enable_market_impact": enable_market_impact,
        "enable_after_hours": enable_after_hours,
        "partial_fill_prob": partial_fill_prob,
        
        # Data Series (for plotting and analysis)
        "equity_curve": [float(x) for x in equity_curve],
        "daily_returns": [float(x) for x in daily_returns],
        "underwater_curve": [float(x) for x in underwater_curve],
        "trades": [trade.to_dict() for trade in trades]  # Detailed trade records
    }
    
    return result

def _max_drawdown(equity):
    peak = -np.inf
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / (peak + 1e-9)
        max_dd = max(max_dd, dd)
    return max_dd

def _win_rate(equity):
    diffs = np.diff(equity)
    gains = (diffs > 0).sum()
    total = len(diffs)
    return gains / total if total else 0.0

def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (return/downside deviation)"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    return (excess_returns.mean() * 252) / (downside_deviation * np.sqrt(252))

def _calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) at given confidence level"""
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, (1 - confidence_level) * 100)

def _calculate_win_rate(trades: List[TradeRecord]) -> float:
    """Calculate win rate from trade records"""
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.pnl > 0)
    return winning_trades / len(trades)

def _calculate_profit_factor(trades: List[TradeRecord]) -> float:
    """Calculate profit factor (gross profit / gross loss)"""
    if not trades:
        return 0.0
    
    gross_profit = sum(max(0, trade.pnl) for trade in trades)
    gross_loss = abs(sum(min(0, trade.pnl) for trade in trades))
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

def _calculate_recovery_factor(cagr: float, max_drawdown: float) -> float:
    """Calculate recovery factor (net profit / max drawdown)"""
    if max_drawdown == 0:
        return float('inf') if cagr > 0 else 0.0
    return cagr / max_drawdown

def _calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray = None) -> float:
    """Calculate information ratio (tracking error adjusted return)"""
    if benchmark_returns is None:
        # Use zero benchmark if none provided
        benchmark_returns = np.zeros_like(returns)
    
    excess_returns = returns - benchmark_returns
    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0.0
    
    return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

def _calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (CAGR / Max Drawdown)"""
    if max_drawdown == 0:
        return float('inf') if cagr > 0 else 0.0
    return cagr / max_drawdown

def _calculate_trade_analytics(trades: List[TradeRecord]) -> Dict[str, float]:
    """Calculate comprehensive trade-level analytics"""
    if not trades:
        return {
            "avg_trade_pnl": 0.0, "avg_win_pnl": 0.0, "avg_loss_pnl": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0, "avg_hold_time": 0.0,
            "max_adverse_excursion": 0.0, "max_favorable_excursion": 0.0,
            "profit_factor": 0.0, "payoff_ratio": 0.0, "kelly_criterion": 0.0
        }
    
    pnls = [trade.pnl for trade in trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl < 0]
    
    avg_trade_pnl = np.mean(pnls)
    avg_win_pnl = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss_pnl = np.mean(losing_trades) if losing_trades else 0.0
    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0
    
    # Hold time analysis
    hold_times = [trade.hold_time for trade in trades if trade.hold_time > 0]
    avg_hold_time = np.mean(hold_times) if hold_times else 0.0
    
    # MAE/MFE analysis
    mae_values = [trade.max_adverse_excursion for trade in trades if trade.max_adverse_excursion != 0]
    mfe_values = [trade.max_favorable_excursion for trade in trades if trade.max_favorable_excursion != 0]
    max_adverse_excursion = np.mean(mae_values) if mae_values else 0.0
    max_favorable_excursion = np.mean(mfe_values) if mfe_values else 0.0
    
    # Advanced metrics
    profit_factor = _calculate_profit_factor(trades)
    payoff_ratio = abs(avg_win_pnl / avg_loss_pnl) if avg_loss_pnl != 0 else float('inf')
    
    # Kelly Criterion calculation
    win_rate = _calculate_win_rate(trades)
    kelly_criterion = 0.0
    if avg_loss_pnl != 0 and payoff_ratio != float('inf'):
        kelly_criterion = win_rate - ((1 - win_rate) / payoff_ratio)
    
    return {
        "avg_trade_pnl": float(avg_trade_pnl),
        "avg_win_pnl": float(avg_win_pnl),
        "avg_loss_pnl": float(avg_loss_pnl),
        "largest_win": float(largest_win),
        "largest_loss": float(largest_loss),
        "avg_hold_time": float(avg_hold_time),
        "max_adverse_excursion": float(max_adverse_excursion),
        "max_favorable_excursion": float(max_favorable_excursion),
        "profit_factor": float(profit_factor),
        "payoff_ratio": float(payoff_ratio),
        "kelly_criterion": float(kelly_criterion)
    }

def run_multi_asset_backtest(asset_data: dict, template: str, params: dict | None = None,
                           start_cash: float = 10000.0, fee_rate: float = 0.0005, 
                           rebalance_frequency: int = 20) -> dict:
    """
    Run backtest on multiple assets for cross-asset strategies
    
    Args:
        asset_data: Dict of {"asset_name": bars_list} for each asset
        template: Strategy template name (cross_asset_momentum, asset_rotation, risk_parity)
        params: Strategy parameters
        start_cash: Starting capital
        fee_rate: Transaction fee rate
        rebalance_frequency: Days between rebalancing
    """
    
    if template not in ["cross_asset_momentum", "asset_rotation", "risk_parity"]:
        raise ValueError(f"Template {template} not supported for multi-asset backtesting")
    
    # Convert all asset data to DataFrames
    dfs = {}
    min_length = float('inf')
    
    for asset_name, bars in asset_data.items():
        df = make_df(bars)
        dfs[asset_name] = df
        min_length = min(min_length, len(df))
    
    if min_length < 60:  # Need minimum data
        return {"error": "Insufficient data for multi-asset backtesting"}
    
    # Get strategy function
    fn = TEMPLATES.get(template)
    if not fn:
        raise ValueError(f"Unknown template {template}")
    
    # Initialize portfolio
    cash = start_cash
    positions = {asset: 0.0 for asset in dfs.keys()}  # Shares held in each asset
    equity_curve = []
    trades = []
    last_rebalance = 0
    
    # Align data - use common date range
    start_idx = max(50, rebalance_frequency)
    
    for i in range(start_idx, min_length):
        # Get current data for all assets
        current_dfs = {}
        current_prices = {}
        
        for asset_name, df in dfs.items():
            current_dfs[asset_name] = df.iloc[:i+1]
            current_prices[asset_name] = df["close"].iloc[i]
        
        # Rebalance check
        if i - last_rebalance >= rebalance_frequency or i == start_idx:
            try:
                # Get strategy signal
                signal, confidence, details = fn(current_dfs, **(params or {}))
                
                if signal != "hold" or i == start_idx:
                    # Calculate current portfolio value
                    portfolio_value = cash + sum(positions[asset] * current_prices[asset] 
                                               for asset in positions.keys())
                    
                    # Liquidate current positions
                    liquidation_proceeds = 0.0
                    for asset in positions.keys():
                        if positions[asset] != 0:
                            proceeds = positions[asset] * current_prices[asset]
                            fees = proceeds * fee_rate
                            liquidation_proceeds += proceeds - fees
                            
                            trades.append({
                                "asset": asset,
                                "type": "liquidate",
                                "price": current_prices[asset],
                                "shares": positions[asset],
                                "proceeds": proceeds,
                                "fees": fees
                            })
                            
                            positions[asset] = 0.0
                    
                    cash += liquidation_proceeds
                    
                    # Implement strategy-specific allocation
                    if template == "cross_asset_momentum":
                        # Allocate to top momentum asset
                        top_asset = details.get("top_asset")
                        if top_asset and signal == "buy":
                            allocation_amount = cash * 0.95
                            shares_to_buy = allocation_amount / current_prices[top_asset]
                            cost = shares_to_buy * current_prices[top_asset]
                            fees = cost * fee_rate
                            
                            if cash >= cost + fees:
                                positions[top_asset] = shares_to_buy
                                cash -= cost + fees
                                
                                trades.append({
                                    "asset": top_asset,
                                    "type": "buy",
                                    "price": current_prices[top_asset],
                                    "shares": shares_to_buy,
                                    "cost": cost,
                                    "fees": fees
                                })
                    
                    elif template == "asset_rotation":
                        # Rotate to best performing asset
                        best_asset = details.get("best_asset")
                        if best_asset and details.get("rotation_signal"):
                            allocation_amount = cash * 0.95
                            shares_to_buy = allocation_amount / current_prices[best_asset]
                            cost = shares_to_buy * current_prices[best_asset]
                            fees = cost * fee_rate
                            
                            if cash >= cost + fees:
                                positions[best_asset] = shares_to_buy
                                cash -= cost + fees
                                
                                trades.append({
                                    "asset": best_asset,
                                    "type": "rotate",
                                    "price": current_prices[best_asset],
                                    "shares": shares_to_buy,
                                    "cost": cost,
                                    "fees": fees
                                })
                    
                    elif template == "risk_parity":
                        # Equal risk allocation
                        weights = details.get("weights", {})
                        total_allocation = cash * 0.95
                        
                        for asset, weight in weights.items():
                            if asset in current_prices:
                                allocation = total_allocation * weight
                                shares_to_buy = allocation / current_prices[asset]
                                cost = shares_to_buy * current_prices[asset]
                                fees = cost * fee_rate
                                
                                if cash >= cost + fees:
                                    positions[asset] = shares_to_buy
                                    cash -= cost + fees
                                    
                                    trades.append({
                                        "asset": asset,
                                        "type": "rebalance",
                                        "price": current_prices[asset],
                                        "shares": shares_to_buy,
                                        "cost": cost,
                                        "fees": fees,
                                        "weight": weight
                                    })
                    
                    last_rebalance = i
            
            except Exception as e:
                # Continue with current positions if strategy fails
                details = {"error": str(e)}
        
        # Calculate current equity
        portfolio_value = cash + sum(positions[asset] * current_prices[asset] 
                                   for asset in positions.keys())
        equity_curve.append(portfolio_value)
    
    # Calculate performance metrics
    if len(equity_curve) > 1:
        returns = np.diff(np.log(np.array(equity_curve) + 1e-9))
        periods = len(returns)
        
        cagr = (equity_curve[-1] / (start_cash + 1e-9)) ** (252/periods) - 1 if periods > 0 else 0
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = (returns.mean() * 252) / (vol + 1e-9) if vol > 0 else 0
        max_dd = _max_drawdown(np.array(equity_curve))
        win_rate = _win_rate(np.array(equity_curve))
        calmar = cagr / (max_dd + 1e-9) if max_dd > 0 else 0
    else:
        cagr = vol = sharpe = max_dd = win_rate = calmar = 0
    
    return {
        "final_equity": float(equity_curve[-1]) if equity_curve else start_cash,
        "total_return": float((equity_curve[-1] / start_cash - 1) * 100) if equity_curve else 0,
        "CAGR": float(cagr * 100),
        "volatility": float(vol * 100),
        "Sharpe": float(sharpe),
        "Calmar": float(calmar),
        "MaxDrawdown": float(max_dd * 100),
        "WinRate": float(win_rate * 100),
        "n_trades": len(trades),
        "n_rebalances": len([t for t in trades if t["type"] in ["rebalance", "rotate"]]),
        "final_positions": {k: float(v) for k, v in positions.items() if v != 0},
        "strategy": template,
        "rebalance_frequency": rebalance_frequency,
        "assets_analyzed": list(asset_data.keys()),
        "correlation_matrix": _calculate_asset_correlations(asset_data) if len(asset_data) > 1 else {}
    }

def _calculate_asset_correlations(asset_data: dict) -> dict:
    """Calculate correlation matrix between assets"""
    correlations = {}
    asset_names = list(asset_data.keys())
    
    for i, asset1 in enumerate(asset_names):
        for j, asset2 in enumerate(asset_names[i+1:], i+1):
            df1 = make_df(asset_data[asset1])
            df2 = make_df(asset_data[asset2])
            
            if len(df1) > 20 and len(df2) > 20:
                returns1 = df1["close"].pct_change().dropna()
                returns2 = df2["close"].pct_change().dropna()
                
                min_len = min(len(returns1), len(returns2))
                corr = returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:])
                correlations[f"{asset1}_{asset2}"] = float(corr) if not np.isnan(corr) else 0.0
    
    return correlations

def run_walk_forward_analysis(bars: list[dict], template: str, params: dict | None = None,
                             window_size: int = 252, step_size: int = 21, optimization_period: int = 126,
                             start_cash: float = 10000.0, fee_rate: float = 0.0005,
                             param_ranges: dict = None) -> dict:
    """
    Walk-forward analysis with parameter optimization and out-of-sample testing
    
    Args:
        bars: OHLCV data
        template: Strategy template name
        params: Base strategy parameters (will be optimized if param_ranges provided)
        window_size: Size of each analysis window in periods
        step_size: Step size between windows in periods
        optimization_period: In-sample optimization period within each window
        start_cash: Starting capital for each window
        fee_rate: Transaction fee rate
        param_ranges: Parameter ranges for optimization (optional)
    
    Returns:
        Dictionary with individual period results and aggregate metrics
    """
    
    df = make_df(bars)
    total_periods = len(df)
    
    if total_periods < window_size + optimization_period:
        return {"error": f"Insufficient data. Need at least {window_size + optimization_period} periods"}
    
    # Track results for each walk-forward period
    individual_periods = []
    aggregate_equity = [start_cash]  # Combined equity across all periods
    all_trades = []
    
    # Walk forward through the data
    start_idx = optimization_period
    
    while start_idx + window_size <= total_periods:
        end_idx = start_idx + window_size
        
        # Split data into optimization (in-sample) and testing (out-of-sample) periods
        optimization_end = start_idx + optimization_period
        
        optimization_data = bars[start_idx-optimization_period:optimization_end]
        testing_data = bars[optimization_end:end_idx]
        
        period_start = df.index[start_idx].strftime('%Y-%m-%d') if hasattr(df.index[start_idx], 'strftime') else str(df.index[start_idx])
        period_end = df.index[end_idx-1].strftime('%Y-%m-%d') if hasattr(df.index[end_idx-1], 'strftime') else str(df.index[end_idx-1])
        
        # Optimize parameters on in-sample data
        optimized_params = params.copy() if params else {}
        optimization_metrics = None
        
        if param_ranges:
            try:
                from .recursive import optimize_strategy_genetic, GeneticConfig
                
                # Quick genetic optimization for walk-forward
                config = GeneticConfig(
                    population_size=20,
                    generations=10,
                    fitness_function="sharpe"
                )
                
                opt_result = optimize_strategy_genetic(
                    optimization_data, template, param_ranges, config
                )
                
                if "optimized_params" in opt_result:
                    optimized_params = opt_result["optimized_params"]
                    optimization_metrics = opt_result["final_metrics"]
                    
            except ImportError:
                # Fallback to basic parameter sweep if genetic optimization unavailable
                best_sharpe = -np.inf
                best_params = optimized_params
                
                # Simple grid search (limited for performance)
                param_combinations = []
                for param, (min_val, max_val, param_type) in (param_ranges or {}).items():
                    if param_type == 'int':
                        values = [min_val, (min_val + max_val) // 2, max_val]
                    elif param_type == 'float':
                        values = [min_val, (min_val + max_val) / 2, max_val]
                    else:
                        values = [min_val]  # Default for other types
                    param_combinations.append((param, values))
                
                # Test limited combinations for performance
                import itertools
                for combo in list(itertools.product(*[values for _, values in param_combinations]))[:27]:  # Limit to 27 combinations
                    test_params = dict(zip([param for param, _ in param_combinations], combo))
                    test_params.update(optimized_params)
                    
                    result = run_backtest(optimization_data, template, test_params, start_cash, fee_rate)
                    if isinstance(result, dict) and "Sharpe" in result and result["Sharpe"] > best_sharpe:
                        best_sharpe = result["Sharpe"]
                        best_params = test_params
                        optimization_metrics = result
                
                optimized_params = best_params
        
        # Test optimized parameters on out-of-sample data
        if len(testing_data) >= 30:  # Minimum testing period
            testing_result = run_backtest(
                testing_data, template, optimized_params, start_cash, fee_rate
            )
            
            # Store period results
            period_result = {
                "period_start": period_start,
                "period_end": period_end,
                "optimization_period_size": optimization_period,
                "testing_period_size": len(testing_data),
                "optimized_params": optimized_params,
                "optimization_metrics": optimization_metrics,
                "testing_metrics": testing_result,
                "parameter_stability": _assess_parameter_stability(optimized_params, individual_periods)
            }
            
            individual_periods.append(period_result)
            
            # Aggregate equity curve
            if isinstance(testing_result, dict) and "equity_curve" in testing_result:
                # Scale to continue from last equity level
                current_equity = aggregate_equity[-1]
                period_equity = testing_result["equity_curve"]
                if period_equity:
                    scaled_equity = [(eq / period_equity[0]) * current_equity for eq in period_equity[1:]]
                    aggregate_equity.extend(scaled_equity)
                    
                    # Collect all trades with timing adjustments
                    if "trades" in testing_result:
                        period_trades = testing_result["trades"]
                        # Adjust trade timestamps for proper sequencing
                        for trade in period_trades:
                            adjusted_trade = trade.copy()
                            all_trades.append(adjusted_trade)
        
        # Move to next period
        start_idx += step_size
    
    # Calculate aggregate metrics
    if len(aggregate_equity) > 1:
        aggregate_returns = np.diff(np.log(np.array(aggregate_equity) + 1e-9))
        periods = len(aggregate_returns)
        
        agg_cagr = (aggregate_equity[-1] / start_cash) ** (252/periods) - 1 if periods > 0 else 0
        agg_vol = aggregate_returns.std() * np.sqrt(252) if len(aggregate_returns) > 1 else 0
        agg_sharpe = (aggregate_returns.mean() * 252) / (agg_vol + 1e-9) if agg_vol > 0 else 0
        agg_max_dd = _max_drawdown(np.array(aggregate_equity))
        agg_calmar = agg_cagr / (agg_max_dd + 1e-9) if agg_max_dd > 0 else 0
        
        aggregate_metrics = {
            "CAGR": float(agg_cagr * 100),
            "volatility": float(agg_vol * 100),
            "Sharpe": float(agg_sharpe),
            "MaxDrawdown": float(agg_max_dd * 100),
            "Calmar": float(agg_calmar),
            "final_equity": float(aggregate_equity[-1]),
            "total_return": float((aggregate_equity[-1] / start_cash - 1) * 100)
        }
    else:
        aggregate_metrics = {
            "CAGR": 0.0, "volatility": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0,
            "Calmar": 0.0, "final_equity": start_cash, "total_return": 0.0
        }
    
    # Calculate stability metrics
    stability_metrics = _calculate_walk_forward_stability(individual_periods)
    
    return {
        "analysis_type": "walk_forward",
        "individual_periods": individual_periods,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_equity_curve": [float(x) for x in aggregate_equity],
        "stability_metrics": stability_metrics,
        "total_periods_analyzed": len(individual_periods),
        "average_testing_period_performance": {
            "avg_sharpe": np.mean([p["testing_metrics"].get("Sharpe", 0) for p in individual_periods if isinstance(p["testing_metrics"], dict)]),
            "avg_cagr": np.mean([p["testing_metrics"].get("CAGR", 0) for p in individual_periods if isinstance(p["testing_metrics"], dict)]),
            "avg_max_dd": np.mean([p["testing_metrics"].get("MaxDrawdown", 0) for p in individual_periods if isinstance(p["testing_metrics"], dict)])
        } if individual_periods else {"avg_sharpe": 0, "avg_cagr": 0, "avg_max_dd": 0},
        "configuration": {
            "window_size": window_size,
            "step_size": step_size, 
            "optimization_period": optimization_period,
            "template": template,
            "param_ranges_provided": param_ranges is not None
        }
    }

def _assess_parameter_stability(current_params: dict, previous_periods: list) -> dict:
    """Assess stability of optimized parameters across periods"""
    if len(previous_periods) < 2:
        return {"stability_score": 1.0, "parameter_drift": {}}
    
    # Look at last few periods for stability assessment
    recent_periods = previous_periods[-3:] if len(previous_periods) >= 3 else previous_periods
    
    parameter_drift = {}
    total_drift = 0
    param_count = 0
    
    for param, current_value in current_params.items():
        if isinstance(current_value, (int, float)):
            historical_values = []
            for period in recent_periods:
                if param in period["optimized_params"]:
                    historical_values.append(period["optimized_params"][param])
            
            if historical_values:
                mean_historical = np.mean(historical_values)
                if mean_historical != 0:
                    drift = abs(current_value - mean_historical) / abs(mean_historical)
                    parameter_drift[param] = float(drift)
                    total_drift += drift
                    param_count += 1
    
    stability_score = max(0, 1 - (total_drift / max(1, param_count))) if param_count > 0 else 1.0
    
    return {
        "stability_score": float(stability_score),
        "parameter_drift": parameter_drift,
        "avg_drift": float(total_drift / max(1, param_count))
    }

def _calculate_walk_forward_stability(individual_periods: list) -> dict:
    """Calculate stability metrics across walk-forward periods"""
    if not individual_periods:
        return {"consistency_ratio": 0.0, "performance_stability": 0.0}
    
    # Extract performance metrics
    sharpe_values = []
    cagr_values = []
    
    for period in individual_periods:
        if isinstance(period["testing_metrics"], dict):
            sharpe_values.append(period["testing_metrics"].get("Sharpe", 0))
            cagr_values.append(period["testing_metrics"].get("CAGR", 0))
    
    # Calculate consistency (lower coefficient of variation = more consistent)
    sharpe_consistency = 1 - (np.std(sharpe_values) / (abs(np.mean(sharpe_values)) + 1e-9)) if sharpe_values else 0
    cagr_consistency = 1 - (np.std(cagr_values) / (abs(np.mean(cagr_values)) + 1e-9)) if cagr_values else 0
    
    # Calculate percentage of profitable periods
    profitable_periods = sum(1 for cagr in cagr_values if cagr > 0)
    profitability_ratio = profitable_periods / len(cagr_values) if cagr_values else 0
    
    return {
        "consistency_ratio": float(max(0, (sharpe_consistency + cagr_consistency) / 2)),
        "performance_stability": float(profitability_ratio),
        "sharpe_consistency": float(max(0, sharpe_consistency)),
        "cagr_consistency": float(max(0, cagr_consistency)),
        "profitable_periods_ratio": float(profitability_ratio)
    }

# VectorBT-based High-Performance Backtesting Functions

def run_vbt_backtest(bars: List[dict], strategy_template: str, params: dict = None, 
                     start_cash: float = 10000.0, fees: float = 0.001, slippage: float = 0.001) -> dict:
    """
    High-performance VectorBT-based backtesting with comprehensive strategy templates
    
    Args:
        bars: OHLCV data as list of dictionaries
        strategy_template: Strategy name (sma_crossover, rsi_strategy, macd_strategy, bb_strategy, sentiment_strategy)
        params: Strategy parameters
        start_cash: Starting capital
        fees: Transaction fees (as percentage)
        slippage: Slippage (as percentage)
    
    Returns:
        Dictionary with comprehensive metrics and trade statistics
    """
    
    if not VBT_AVAILABLE:
        # Fallback to legacy backtester
        return run_backtest(bars, strategy_template, params, start_cash, fees)
    
    # Convert bars to pandas DataFrame
    df = _prepare_vbt_data(bars)
    if df is None or len(df) < 50:
        return {"error": "Insufficient data for VectorBT backtesting (minimum 50 bars required)"}
    
    # Apply default parameters if none provided
    params = params or {}
    
    try:
        # Generate trading signals based on strategy
        entries, exits = _generate_vbt_signals(df, strategy_template, params)
        
        if entries is None or exits is None:
            return {"error": f"Failed to generate signals for strategy: {strategy_template}"}
        
        # Create VectorBT portfolio
        portfolio = vbt.Portfolio.from_signals(
            data=df['close'],
            entries=entries,
            exits=exits,
            init_cash=start_cash,
            fees=fees,
            slippage=slippage,
            freq='1D'  # Default to daily frequency
        )
        
        # Extract comprehensive metrics
        stats = portfolio.stats()
        trades = portfolio.trades.records
        
        # Calculate additional metrics
        returns = portfolio.returns()
        benchmark_returns = df['close'].pct_change().fillna(0)
        
        # Core performance metrics
        total_return = float(portfolio.total_return()) * 100
        cagr = float(portfolio.cagr()) * 100
        volatility = float(returns.std() * np.sqrt(252)) * 100  # Annualized
        sharpe_ratio = float(portfolio.sharpe_ratio())
        sortino_ratio = float(portfolio.sortino_ratio())
        calmar_ratio = float(portfolio.calmar_ratio())
        max_drawdown = float(portfolio.max_drawdown()) * 100
        
        # Trade statistics
        n_trades = len(trades) if trades is not None else 0
        win_rate = float(portfolio.trades.win_rate) if n_trades > 0 else 0
        profit_factor = float(portfolio.trades.profit_factor) if n_trades > 0 else 0
        expectancy = float(portfolio.trades.expectancy) if n_trades > 0 else 0
        
        # Risk metrics
        var_95 = float(np.percentile(returns.dropna(), 5)) * 100
        var_99 = float(np.percentile(returns.dropna(), 1)) * 100
        
        # Additional VectorBT specific metrics
        exposure = float(portfolio.stats()['Exposure [%]']) if 'Exposure [%]' in portfolio.stats() else 0
        sqn = float(portfolio.trades.sqn) if n_trades > 0 else 0  # System Quality Number
        
        # Trade analysis
        if n_trades > 0:
            avg_trade_duration = float(portfolio.trades.duration.mean())
            avg_win = float(portfolio.trades.winning.pnl.mean()) if portfolio.trades.winning.count > 0 else 0
            avg_loss = float(portfolio.trades.losing.pnl.mean()) if portfolio.trades.losing.count > 0 else 0
            largest_win = float(portfolio.trades.winning.pnl.max()) if portfolio.trades.winning.count > 0 else 0
            largest_loss = float(portfolio.trades.losing.pnl.min()) if portfolio.trades.losing.count > 0 else 0
        else:
            avg_trade_duration = avg_win = avg_loss = largest_win = largest_loss = 0
        
        # Information ratio (vs benchmark)
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Compile comprehensive results
        result = {
            # Core Performance
            "final_equity": float(portfolio.final_value()),
            "total_return": total_return,
            "CAGR": cagr,
            "volatility": volatility,
            
            # Risk Metrics
            "Sharpe": sharpe_ratio,
            "Sortino": sortino_ratio,
            "Calmar": calmar_ratio,
            "MaxDrawdown": max_drawdown,
            "InformationRatio": float(information_ratio),
            "VaR_95": var_95,
            "VaR_99": var_99,
            
            # Trade Analytics
            "WinRate": float(win_rate * 100),
            "n_trades": n_trades,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_trade_duration": avg_trade_duration,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            
            # VectorBT Specific
            "Exposure": exposure,
            "SQN": sqn,  # System Quality Number
            
            # Configuration
            "strategy": strategy_template,
            "parameters": params,
            "fees": fees,
            "slippage": slippage,
            "start_cash": start_cash,
            "engine": "VectorBT",
            
            # Data Series for analysis
            "equity_curve": portfolio.value().tolist(),
            "daily_returns": returns.tolist(),
            "benchmark_returns": benchmark_returns.tolist(),
            "underwater_curve": portfolio.drawdown().tolist(),
            
            # Trade records (if available)
            "trades": _format_vbt_trades(trades) if n_trades > 0 else []
        }
        
        return result
        
    except Exception as e:
        return {"error": f"VectorBT backtesting failed: {str(e)}"}

def _prepare_vbt_data(bars: List[dict]) -> pd.DataFrame:
    """Prepare OHLCV data for VectorBT"""
    try:
        df = pd.DataFrame(bars)
        
        # Ensure proper column names
        column_mapping = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Convert to numeric
        for col in required_cols + ['volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 't' in df.columns:
            df['t'] = pd.to_datetime(df['t'])
            df = df.set_index('t')
        else:
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error preparing VBT data: {e}")
        return None

def _generate_vbt_signals(df: pd.DataFrame, strategy: str, params: dict) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Generate buy/sell signals for VectorBT strategies"""
    
    try:
        if strategy == "sma_crossover":
            return _sma_crossover_signals(df, params)
        elif strategy == "rsi_strategy":
            return _rsi_strategy_signals(df, params)
        elif strategy == "macd_strategy":
            return _macd_strategy_signals(df, params)
        elif strategy == "bb_strategy":
            return _bollinger_bands_signals(df, params)
        elif strategy == "sentiment_strategy":
            return _sentiment_strategy_signals(df, params)
        else:
            # Try to use existing strategy templates
            return _legacy_strategy_signals(df, strategy, params)
            
    except Exception as e:
        print(f"Error generating signals for {strategy}: {e}")
        return None, None

def _sma_crossover_signals(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """Simple Moving Average Crossover Strategy"""
    short_window = params.get('short_window', 20)
    long_window = params.get('long_window', 50)
    
    short_ma = df['close'].rolling(window=short_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    
    # Entry: Short MA crosses above Long MA
    entries = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    
    # Exit: Short MA crosses below Long MA
    exits = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    
    return entries, exits

def _rsi_strategy_signals(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """RSI-based Mean Reversion Strategy"""
    rsi_period = params.get('rsi_period', 14)
    oversold_level = params.get('oversold', 30)
    overbought_level = params.get('overbought', 70)
    
    # Calculate RSI using talib
    rsi = pd.Series(talib.RSI(df['close'].values, timeperiod=rsi_period), index=df.index)
    
    # Entry: RSI crosses above oversold level
    entries = (rsi > oversold_level) & (rsi.shift(1) <= oversold_level)
    
    # Exit: RSI crosses above overbought level
    exits = (rsi > overbought_level) & (rsi.shift(1) <= overbought_level)
    
    return entries, exits

def _macd_strategy_signals(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """MACD Strategy"""
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    
    # Calculate MACD using talib
    macd, macdsignal, macdhist = talib.MACD(
        df['close'].values,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    
    macd = pd.Series(macd, index=df.index)
    macdsignal = pd.Series(macdsignal, index=df.index)
    
    # Entry: MACD crosses above signal line
    entries = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
    
    # Exit: MACD crosses below signal line
    exits = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))
    
    return entries, exits

def _bollinger_bands_signals(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands Mean Reversion Strategy"""
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)
    
    # Calculate Bollinger Bands using talib
    upper, middle, lower = talib.BBANDS(
        df['close'].values,
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev
    )
    
    upper = pd.Series(upper, index=df.index)
    lower = pd.Series(lower, index=df.index)
    
    # Entry: Price touches lower band
    entries = (df['close'] <= lower) & (df['close'].shift(1) > lower.shift(1))
    
    # Exit: Price touches upper band
    exits = (df['close'] >= upper) & (df['close'].shift(1) < upper.shift(1))
    
    return entries, exits

def _sentiment_strategy_signals(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """Sentiment-based Strategy (requires sentiment scores in data)"""
    sentiment_threshold = params.get('sentiment_threshold', 0.6)
    sentiment_column = params.get('sentiment_column', 'sentiment')
    
    if sentiment_column not in df.columns:
        # Create dummy sentiment data if not available
        df[sentiment_column] = np.random.normal(0.5, 0.2, len(df))
        df[sentiment_column] = df[sentiment_column].clip(0, 1)
    
    # Entry: High positive sentiment
    entries = df[sentiment_column] > sentiment_threshold
    
    # Exit: Low sentiment
    exits = df[sentiment_column] < (1 - sentiment_threshold)
    
    return entries, exits

def _legacy_strategy_signals(df: pd.DataFrame, strategy: str, params: dict) -> Tuple[pd.Series, pd.Series]:
    """Convert legacy strategy templates to VectorBT signals"""
    try:
        # Use existing strategy engine
        bars = df.reset_index().to_dict('records')
        
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        
        # Process data window by window to generate signals
        for i in range(50, len(df)):
            window_data = bars[:i+1]
            
            if strategy in TEMPLATES:
                fn = TEMPLATES[strategy]
                action, confidence, details = fn(make_df(window_data), **(params or {}))
                
                if action == "buy":
                    entries.iloc[i] = True
                elif action == "sell":
                    exits.iloc[i] = True
        
        return entries, exits
        
    except Exception as e:
        print(f"Error in legacy strategy conversion: {e}")
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

def _format_vbt_trades(trades) -> List[dict]:
    """Format VectorBT trade records for output"""
    if trades is None or len(trades) == 0:
        return []
    
    try:
        trades_list = []
        for i, trade in enumerate(trades):
            trades_list.append({
                "id": int(i),
                "entry_timestamp": str(trade.get('entry_timestamp', '')),
                "exit_timestamp": str(trade.get('exit_timestamp', '')),
                "entry_price": float(trade.get('entry_price', 0)),
                "exit_price": float(trade.get('exit_price', 0)),
                "size": float(trade.get('size', 0)),
                "pnl": float(trade.get('pnl', 0)),
                "return_pct": float(trade.get('return_pct', 0)) * 100,
                "duration": int(trade.get('duration', 0))
            })
        return trades_list[:100]  # Limit to 100 trades for performance
    
    except Exception as e:
        print(f"Error formatting VBT trades: {e}")
        return []
