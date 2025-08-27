"""
Financial Data Processing Engine for SuperNova AI
Advanced time-series analysis, risk modeling, and market data processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from scipy import stats, optimize
from scipy.signal import find_peaks
import warnings
from collections import defaultdict, deque
import time
from enum import Enum

# TimescaleDB and database imports
try:
    from .db import get_timescale_session, is_timescale_available, SessionLocal
    from .analytics_models import (
        Portfolio, Position, Transaction, PerformanceRecord, RiskMetric,
        MarketSentiment, TechnicalIndicator, BacktestAnalysis
    )
    from sqlalchemy import select, and_, or_, func, desc
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

from .config import settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataQuality(str, Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"

@dataclass
class TimeSeriesMetrics:
    """Time series analysis metrics"""
    trend_strength: float
    trend_direction: str
    volatility: float
    autocorrelation: float
    seasonality_score: float
    stationarity_p_value: float
    data_quality: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskModel:
    """Risk model results"""
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    volatility_forecast: float
    correlation_risk: float
    concentration_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # bull, bear, sideways, volatile
    confidence: float
    duration_days: int
    volatility_level: str  # low, medium, high
    trend_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FinancialDataProcessor:
    """Advanced financial data processing engine"""
    
    def __init__(self):
        """Initialize the data processing engine"""
        self.trading_days_per_year = 252
        self.risk_free_rate = getattr(settings, 'RISK_FREE_RATE', 0.02)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Market hours (can be configured)
        self.market_open = 9.5  # 9:30 AM
        self.market_close = 16.0  # 4:00 PM
        
        # Data cache for performance
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def process_time_series_data(self, 
                                data: pd.DataFrame,
                                price_column: str = 'close',
                                volume_column: str = 'volume') -> TimeSeriesMetrics:
        """
        Process time series data and extract comprehensive metrics
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Name of price column
            volume_column: Name of volume column
            
        Returns:
            TimeSeriesMetrics object
        """
        try:
            if len(data) < 30:
                return TimeSeriesMetrics(
                    trend_strength=0,
                    trend_direction='insufficient_data',
                    volatility=0,
                    autocorrelation=0,
                    seasonality_score=0,
                    stationarity_p_value=1.0,
                    data_quality='insufficient'
                )
            
            prices = data[price_column].dropna()
            returns = prices.pct_change().dropna()
            
            # Trend analysis
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices.values)
            trend_strength = abs(r_value)
            trend_direction = 'upward' if slope > 0 else 'downward' if slope < 0 else 'sideways'
            
            # Volatility
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Autocorrelation
            autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0
            autocorr = 0 if pd.isna(autocorr) else autocorr
            
            # Seasonality analysis
            seasonality_score = self._calculate_seasonality(returns)
            
            # Stationarity test (Augmented Dickey-Fuller)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(returns.dropna())
                stationarity_p_value = adf_result[1]
            except ImportError:
                # Fallback: simplified stationarity check
                stationarity_p_value = self._simple_stationarity_test(returns)
            
            # Data quality assessment
            data_quality = self._assess_data_quality(data, returns)
            
            return TimeSeriesMetrics(
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volatility=volatility,
                autocorrelation=autocorr,
                seasonality_score=seasonality_score,
                stationarity_p_value=stationarity_p_value,
                data_quality=data_quality
            )
            
        except Exception as e:
            logger.error(f"Error processing time series data: {str(e)}")
            raise
    
    def calculate_advanced_risk_metrics(self,
                                      returns: pd.Series,
                                      confidence_levels: List[float] = [0.95, 0.99],
                                      time_horizon_days: int = 1) -> RiskModel:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels for VaR/ES
            time_horizon_days: Risk horizon in days
            
        Returns:
            RiskModel object
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 30:
                return RiskModel(
                    value_at_risk_95=0,
                    value_at_risk_99=0,
                    expected_shortfall_95=0,
                    expected_shortfall_99=0,
                    volatility_forecast=0,
                    correlation_risk=0,
                    concentration_risk=0
                )
            
            # Scale returns for time horizon
            scaled_returns = returns_clean * np.sqrt(time_horizon_days)
            
            # Value at Risk (Historical simulation)
            var_95 = np.percentile(scaled_returns, 5)
            var_99 = np.percentile(scaled_returns, 1)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = scaled_returns[scaled_returns <= var_95].mean()
            es_99 = scaled_returns[scaled_returns <= var_99].mean()
            
            # Handle NaN values
            es_95 = es_95 if not pd.isna(es_95) else var_95
            es_99 = es_99 if not pd.isna(es_99) else var_99
            
            # Volatility forecast using EWMA
            volatility_forecast = self._forecast_volatility(returns_clean)
            
            # Correlation risk (simplified - measures autocorrelation)
            correlation_risk = abs(returns_clean.autocorr(lag=1))
            correlation_risk = 0 if pd.isna(correlation_risk) else correlation_risk
            
            # Concentration risk (return dispersion)
            concentration_risk = returns_clean.std() / abs(returns_clean.mean()) if returns_clean.mean() != 0 else 0
            concentration_risk = min(concentration_risk, 10)  # Cap at reasonable level
            
            return RiskModel(
                value_at_risk_95=abs(var_95),
                value_at_risk_99=abs(var_99),
                expected_shortfall_95=abs(es_95),
                expected_shortfall_99=abs(es_99),
                volatility_forecast=volatility_forecast,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def detect_market_regime(self, 
                           price_data: pd.Series,
                           volume_data: Optional[pd.Series] = None,
                           lookback_days: int = 60) -> MarketRegime:
        """
        Detect current market regime
        
        Args:
            price_data: Price time series
            volume_data: Optional volume data
            lookback_days: Days to look back for regime detection
            
        Returns:
            MarketRegime object
        """
        try:
            if len(price_data) < lookback_days:
                return MarketRegime(
                    regime_type='insufficient_data',
                    confidence=0,
                    duration_days=0,
                    volatility_level='unknown',
                    trend_strength=0
                )
            
            # Use recent data for regime detection
            recent_prices = price_data.tail(lookback_days)
            returns = recent_prices.pct_change().dropna()
            
            # Trend analysis
            x = np.arange(len(recent_prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices.values)
            trend_strength = abs(r_value)
            
            # Volatility analysis
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            vol_percentile = self._get_volatility_percentile(returns, price_data)
            
            # Volatility level classification
            if vol_percentile > 0.8:
                volatility_level = 'high'
            elif vol_percentile > 0.4:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Regime classification
            regime_type, confidence = self._classify_regime(slope, r_value, volatility, returns)
            
            # Estimate regime duration
            duration_days = self._estimate_regime_duration(returns)
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                duration_days=duration_days,
                volatility_level=volatility_level,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime(
                regime_type='error',
                confidence=0,
                duration_days=0,
                volatility_level='unknown',
                trend_strength=0
            )
    
    def calculate_portfolio_attribution(self,
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series,
                                      sector_data: Dict[str, pd.Series],
                                      position_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate performance attribution analysis
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            sector_data: Sector return data
            position_weights: Position weights in portfolio
            
        Returns:
            Dictionary with attribution analysis
        """
        try:
            # Align all data
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, keys=['portfolio', 'benchmark'])
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 10:
                return {
                    'total_excess_return': 0,
                    'asset_allocation_effect': 0,
                    'security_selection_effect': 0,
                    'interaction_effect': 0,
                    'attribution_quality': 'insufficient_data'
                }
            
            # Calculate excess returns
            portfolio_return = aligned_data['portfolio'].mean() * self.trading_days_per_year
            benchmark_return = aligned_data['benchmark'].mean() * self.trading_days_per_year
            total_excess_return = portfolio_return - benchmark_return
            
            # Asset allocation effect (simplified)
            asset_allocation_effect = 0
            for sector, sector_returns in sector_data.items():
                if sector in position_weights:
                    # Assume equal benchmark weights for simplicity
                    benchmark_weight = 1.0 / len(sector_data)
                    portfolio_weight = position_weights[sector]
                    weight_difference = portfolio_weight - benchmark_weight
                    
                    # Align sector returns with benchmark
                    sector_aligned = sector_returns.reindex(aligned_data.index).fillna(0)
                    sector_return = sector_aligned.mean() * self.trading_days_per_year
                    
                    asset_allocation_effect += weight_difference * (sector_return - benchmark_return)
            
            # Security selection effect
            security_selection_effect = total_excess_return - asset_allocation_effect
            
            # Interaction effect (residual)
            interaction_effect = 0  # Simplified for this implementation
            
            # Attribution quality
            explained_return = asset_allocation_effect + security_selection_effect
            attribution_quality = 1 - abs(total_excess_return - explained_return) / max(abs(total_excess_return), 0.001)
            attribution_quality = max(0, min(1, attribution_quality))
            
            return {
                'total_excess_return': total_excess_return,
                'asset_allocation_effect': asset_allocation_effect,
                'security_selection_effect': security_selection_effect,
                'interaction_effect': interaction_effect,
                'attribution_quality': attribution_quality,
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio attribution: {str(e)}")
            return {
                'total_excess_return': 0,
                'asset_allocation_effect': 0,
                'security_selection_effect': 0,
                'interaction_effect': 0,
                'attribution_quality': 0
            }
    
    def analyze_correlation_structure(self,
                                    returns_data: pd.DataFrame,
                                    method: str = 'pearson',
                                    rolling_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze correlation structure of assets
        
        Args:
            returns_data: DataFrame with asset returns
            method: Correlation method
            rolling_window: Optional rolling window for dynamic correlations
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            clean_data = returns_data.dropna()
            
            if len(clean_data) < 30:
                return {
                    'static_correlations': {},
                    'average_correlation': 0,
                    'correlation_dispersion': 0,
                    'dynamic_correlations': {},
                    'correlation_regimes': []
                }
            
            # Static correlation matrix
            static_corr = clean_data.corr(method=method)
            
            # Extract upper triangle (avoid duplicates)
            mask = np.triu(np.ones_like(static_corr, dtype=bool), k=1)
            correlations = static_corr.where(mask).stack().dropna()
            
            # Summary statistics
            avg_correlation = correlations.mean()
            correlation_dispersion = correlations.std()
            
            # Dynamic correlations (if rolling window specified)
            dynamic_correlations = {}
            if rolling_window and len(clean_data) > rolling_window:
                for i in range(len(clean_data.columns)):
                    for j in range(i+1, len(clean_data.columns)):
                        col1, col2 = clean_data.columns[i], clean_data.columns[j]
                        rolling_corr = clean_data[col1].rolling(rolling_window).corr(clean_data[col2])
                        dynamic_correlations[f'{col1}_{col2}'] = rolling_corr.dropna().tolist()
            
            # Correlation regime detection
            correlation_regimes = self._detect_correlation_regimes(correlations, clean_data)
            
            return {
                'static_correlations': static_corr.to_dict(),
                'average_correlation': avg_correlation,
                'correlation_dispersion': correlation_dispersion,
                'correlation_distribution': {
                    'min': correlations.min(),
                    'max': correlations.max(),
                    'median': correlations.median(),
                    'q25': correlations.quantile(0.25),
                    'q75': correlations.quantile(0.75)
                },
                'dynamic_correlations': dynamic_correlations,
                'correlation_regimes': correlation_regimes,
                'diversification_ratio': self._calculate_diversification_ratio(clean_data, static_corr)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlation structure: {str(e)}")
            return {
                'static_correlations': {},
                'average_correlation': 0,
                'correlation_dispersion': 0,
                'dynamic_correlations': {},
                'correlation_regimes': []
            }
    
    async def process_market_data_stream(self,
                                       data_stream: List[Dict[str, Any]],
                                       symbol: str) -> Dict[str, Any]:
        """
        Process real-time market data stream
        
        Args:
            data_stream: List of market data points
            symbol: Asset symbol
            
        Returns:
            Dictionary with processed streaming data
        """
        try:
            if not data_stream:
                return {'error': 'No data provided'}
            
            # Convert to DataFrame
            df = pd.DataFrame(data_stream)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Process price data
            prices = df['price'] if 'price' in df.columns else df['close']
            returns = prices.pct_change().dropna()
            
            # Real-time metrics
            current_price = prices.iloc[-1]
            price_change = prices.iloc[-1] - prices.iloc[-2] if len(prices) > 1 else 0
            price_change_pct = (price_change / prices.iloc[-2]) * 100 if len(prices) > 1 and prices.iloc[-2] != 0 else 0
            
            # Volatility metrics
            if len(returns) >= 20:
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(self.trading_days_per_year)
                volatility_percentile = self._calculate_volatility_percentile(returns)
            else:
                volatility = 0
                volatility_percentile = 50
            
            # Volume analysis (if available)
            volume_metrics = {}
            if 'volume' in df.columns:
                volume_metrics = self._analyze_volume_profile(df)
            
            # Technical indicators
            technical_signals = self._calculate_technical_signals(prices)
            
            # Market microstructure (if tick data available)
            microstructure_metrics = {}
            if len(df) > 100:
                microstructure_metrics = self._analyze_microstructure(df)
            
            return {
                'symbol': symbol,
                'timestamp': data_stream[-1]['timestamp'],
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volatility': volatility,
                'volatility_percentile': volatility_percentile,
                'volume_metrics': volume_metrics,
                'technical_signals': technical_signals,
                'microstructure_metrics': microstructure_metrics,
                'data_quality': self._assess_stream_quality(df)
            }
            
        except Exception as e:
            logger.error(f"Error processing market data stream: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods
    def _calculate_seasonality(self, returns: pd.Series) -> float:
        """Calculate seasonality score"""
        try:
            if len(returns) < 252:  # Need at least 1 year
                return 0
            
            # Group by month and calculate variance
            monthly_returns = returns.groupby(returns.index.month).mean()
            seasonality_score = monthly_returns.std()
            return min(seasonality_score, 1.0)  # Cap at 1.0
        except:
            return 0
    
    def _simple_stationarity_test(self, returns: pd.Series) -> float:
        """Simple stationarity test (rolling mean variance)"""
        try:
            if len(returns) < 60:
                return 0.5
            
            window = len(returns) // 3
            rolling_mean = returns.rolling(window).mean()
            mean_variance = rolling_mean.var()
            
            # Lower variance suggests more stationarity
            return min(mean_variance * 10, 1.0)  # Scale and cap
        except:
            return 0.5
    
    def _assess_data_quality(self, data: pd.DataFrame, returns: pd.Series) -> str:
        """Assess data quality"""
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        return_quality = 1 - (returns == 0).sum() / len(returns) if len(returns) > 0 else 0
        
        if missing_ratio < 0.05 and return_quality > 0.8 and len(data) > 100:
            return DataQuality.HIGH.value
        elif missing_ratio < 0.15 and return_quality > 0.6 and len(data) > 50:
            return DataQuality.MEDIUM.value
        elif len(data) > 30:
            return DataQuality.LOW.value
        else:
            return DataQuality.INSUFFICIENT.value
    
    def _forecast_volatility(self, returns: pd.Series, alpha: float = 0.1) -> float:
        """Forecast volatility using EWMA"""
        try:
            squared_returns = returns ** 2
            ewma_vol = squared_returns.ewm(alpha=alpha).mean().iloc[-1] ** 0.5
            return ewma_vol * np.sqrt(self.trading_days_per_year)
        except:
            return returns.std() * np.sqrt(self.trading_days_per_year)
    
    def _get_volatility_percentile(self, returns: pd.Series, full_data: pd.Series) -> float:
        """Get current volatility percentile"""
        try:
            current_vol = returns.tail(30).std()
            full_returns = full_data.pct_change().dropna()
            rolling_vols = full_returns.rolling(30).std().dropna()
            percentile = (rolling_vols < current_vol).mean()
            return percentile
        except:
            return 0.5
    
    def _classify_regime(self, slope: float, r_value: float, volatility: float, returns: pd.Series) -> Tuple[str, float]:
        """Classify market regime"""
        try:
            trend_strength = abs(r_value)
            avg_return = returns.mean()
            
            # High volatility regime
            if volatility > returns.std() * 1.5:
                return 'volatile', min(0.8, volatility / (returns.std() * 2))
            
            # Trend regimes
            if trend_strength > 0.3:
                if avg_return > 0:
                    return 'bull', trend_strength
                else:
                    return 'bear', trend_strength
            
            # Sideways regime
            return 'sideways', max(0.3, 1 - trend_strength)
        except:
            return 'unknown', 0.0
    
    def _estimate_regime_duration(self, returns: pd.Series) -> int:
        """Estimate current regime duration"""
        try:
            # Simple implementation: look for regime changes
            rolling_mean = returns.rolling(20).mean()
            sign_changes = (rolling_mean > 0) != (rolling_mean > 0).shift(1)
            last_change = sign_changes[::-1].idxmax() if sign_changes.any() else returns.index[0]
            duration = (returns.index[-1] - last_change).days
            return max(1, duration)
        except:
            return 30  # Default estimate
    
    def _detect_correlation_regimes(self, correlations: pd.Series, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect correlation regimes"""
        try:
            # Simple implementation: high vs low correlation periods
            window = min(60, len(data) // 4)
            if window < 20:
                return []
            
            rolling_avg_corr = correlations.expanding().mean()
            
            regimes = []
            if len(rolling_avg_corr) > 0:
                current_corr = rolling_avg_corr.iloc[-1]
                historical_median = rolling_avg_corr.median()
                
                if current_corr > historical_median * 1.2:
                    regime_type = 'high_correlation'
                elif current_corr < historical_median * 0.8:
                    regime_type = 'low_correlation'
                else:
                    regime_type = 'normal_correlation'
                
                regimes.append({
                    'regime_type': regime_type,
                    'current_correlation': current_corr,
                    'historical_median': historical_median
                })
            
            return regimes
        except:
            return []
    
    def _calculate_diversification_ratio(self, returns: pd.DataFrame, corr_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        try:
            if len(returns.columns) < 2:
                return 1.0
            
            # Equal weights for simplicity
            weights = np.array([1/len(returns.columns)] * len(returns.columns))
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov().values * self.trading_days_per_year, weights)))
            
            # Weighted average of individual volatilities
            individual_vols = returns.std().values * np.sqrt(self.trading_days_per_year)
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        except:
            return 1.0
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile"""
        try:
            if 'volume' not in df.columns:
                return {}
            
            volume = df['volume']
            avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'normal'
            }
        except:
            return {}
    
    def _calculate_technical_signals(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate basic technical signals"""
        try:
            signals = {}
            
            if len(prices) >= 50:
                # Moving averages
                ma_20 = prices.rolling(20).mean()
                ma_50 = prices.rolling(50).mean()
                
                current_price = prices.iloc[-1]
                signals['ma_20'] = ma_20.iloc[-1]
                signals['ma_50'] = ma_50.iloc[-1]
                signals['price_vs_ma20'] = (current_price - ma_20.iloc[-1]) / ma_20.iloc[-1] * 100
                signals['ma_crossover'] = 'bullish' if ma_20.iloc[-1] > ma_50.iloc[-1] else 'bearish'
            
            # RSI approximation
            if len(prices) >= 14:
                delta = prices.diff()
                gains = delta.where(delta > 0, 0).rolling(14).mean()
                losses = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                signals['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                
                if signals['rsi'] > 70:
                    signals['rsi_signal'] = 'overbought'
                elif signals['rsi'] < 30:
                    signals['rsi_signal'] = 'oversold'
                else:
                    signals['rsi_signal'] = 'neutral'
            
            return signals
        except:
            return {}
    
    def _analyze_microstructure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure"""
        try:
            if len(df) < 100:
                return {}
            
            # Price impact analysis
            if 'volume' in df.columns:
                price_changes = df['price'].pct_change() if 'price' in df.columns else df['close'].pct_change()
                volume_changes = df['volume'].pct_change()
                
                # Simple correlation between price change and volume
                price_volume_corr = price_changes.corr(volume_changes)
                
                return {
                    'price_volume_correlation': price_volume_corr if not pd.isna(price_volume_corr) else 0,
                    'average_trade_size': df['volume'].mean() if 'volume' in df.columns else 0
                }
            
            return {}
        except:
            return {}
    
    def _assess_stream_quality(self, df: pd.DataFrame) -> str:
        """Assess streaming data quality"""
        try:
            # Check for gaps in timestamps
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            gaps = (time_diffs > time_diffs.median() * 3).sum()
            gap_ratio = gaps / len(df)
            
            if gap_ratio < 0.05 and len(df) > 100:
                return 'high'
            elif gap_ratio < 0.15:
                return 'medium'
            else:
                return 'low'
        except:
            return 'unknown'
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate current volatility percentile"""
        try:
            current_vol = returns.tail(20).std() if len(returns) >= 20 else returns.std()
            rolling_vol = returns.rolling(20).std().dropna()
            if len(rolling_vol) > 0:
                percentile = (rolling_vol < current_vol).mean() * 100
                return percentile
            return 50.0
        except:
            return 50.0

# Utility functions
async def process_portfolio_data_async(processor: FinancialDataProcessor,
                                     portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Async wrapper for portfolio data processing"""
    loop = asyncio.get_event_loop()
    
    tasks = []
    
    # Time series analysis
    if 'price_data' in portfolio_data:
        price_series = pd.Series(portfolio_data['price_data'])
        tasks.append(
            loop.run_in_executor(
                processor.thread_pool,
                processor.process_time_series_data,
                pd.DataFrame({'close': price_series}),
                'close'
            )
        )
    
    # Risk analysis
    if 'returns' in portfolio_data:
        returns = pd.Series(portfolio_data['returns'])
        tasks.append(
            loop.run_in_executor(
                processor.thread_pool,
                processor.calculate_advanced_risk_metrics,
                returns
            )
        )
    
    # Market regime detection
    if 'price_data' in portfolio_data:
        price_series = pd.Series(portfolio_data['price_data'])
        tasks.append(
            loop.run_in_executor(
                processor.thread_pool,
                processor.detect_market_regime,
                price_series
            )
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = {}
    result_keys = ['time_series_metrics', 'risk_model', 'market_regime']
    
    for i, result in enumerate(results):
        if i < len(result_keys) and not isinstance(result, Exception):
            processed_results[result_keys[i]] = result.to_dict() if hasattr(result, 'to_dict') else result
        elif isinstance(result, Exception):
            logger.error(f"Error in processing {result_keys[i] if i < len(result_keys) else i}: {str(result)}")
    
    return processed_results