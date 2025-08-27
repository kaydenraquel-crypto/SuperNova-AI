"""
Advanced Financial Analytics Engine for SuperNova AI
Comprehensive portfolio performance analytics, risk metrics, and financial calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
import logging
from decimal import Decimal, ROUND_HALF_UP
import statistics
import warnings
from scipy import stats
from scipy.optimize import minimize
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float
    
    # Additional metrics
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class RiskAnalysis:
    """Risk analysis container"""
    portfolio_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    diversification_ratio: float
    concentration_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    security_selection: float
    asset_allocation: float
    interaction_effect: float
    total_excess_return: float
    sector_attribution: Dict[str, float]
    security_attribution: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class AdvancedAnalyticsEngine:
    """Advanced financial analytics engine with comprehensive calculations"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analytics engine
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def calculate_portfolio_performance(self, 
                                      portfolio_values: pd.Series,
                                      benchmark_values: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_values: Optional benchmark values for comparison
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        try:
            # Calculate returns
            returns = portfolio_values.pct_change().dropna()
            
            if benchmark_values is not None:
                benchmark_returns = benchmark_values.pct_change().dropna()
                # Align dates
                aligned_returns = returns.align(benchmark_returns, join='inner')
                returns, benchmark_returns = aligned_returns[0], aligned_returns[1]
                excess_returns = returns - benchmark_returns
            else:
                benchmark_returns = pd.Series(dtype=float)
                excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
            
            # Basic performance metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            periods_per_year = self.trading_days_per_year / len(returns) * len(returns)
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
            
            cumulative_return = total_return
            excess_return = excess_returns.mean() * self.trading_days_per_year if len(excess_returns) > 0 else 0
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            current_drawdown = drawdowns.iloc[-1]
            
            # Find maximum drawdown duration
            drawdown_periods = self._calculate_drawdown_duration(drawdowns)
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Beta and Alpha (if benchmark available)
            if len(benchmark_returns) > 0:
                beta = self._calculate_beta(returns, benchmark_returns)
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_returns.mean() * self.trading_days_per_year - self.risk_free_rate))
                tracking_error = (returns - benchmark_returns).std() * np.sqrt(self.trading_days_per_year)
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            else:
                beta = 0
                alpha = 0
                tracking_error = 0
                information_ratio = 0
            
            # Statistical measures
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Value at Risk and Conditional Value at Risk
            var_95 = np.percentile(returns, 5) * np.sqrt(self.trading_days_per_year)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(self.trading_days_per_year)
            
            # Trading metrics (if applicable)
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                cumulative_return=cumulative_return,
                excess_return=excess_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration_days=max_drawdown_duration,
                current_drawdown=current_drawdown,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                var_95=var_95,
                cvar_95=cvar_95,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {str(e)}")
            raise
    
    def calculate_risk_analysis(self,
                              positions: Dict[str, Dict[str, float]],
                              returns_data: pd.DataFrame,
                              confidence_level: float = 0.95) -> RiskAnalysis:
        """
        Calculate comprehensive risk analysis for portfolio
        
        Args:
            positions: Dictionary of positions with weights and values
            returns_data: DataFrame with return data for each asset
            confidence_level: Confidence level for VaR calculations
            
        Returns:
            RiskAnalysis object with risk metrics
        """
        try:
            # Extract weights and symbols
            symbols = list(positions.keys())
            weights = np.array([positions[symbol]['weight'] for symbol in symbols])
            
            # Align returns data
            returns_matrix = returns_data[symbols].dropna()
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov().values * self.trading_days_per_year
            correlation_matrix = returns_matrix.corr()
            
            # Portfolio variance and volatility
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Component VaR (marginal contribution to risk)
            marginal_var = np.dot(cov_matrix, weights) / portfolio_vol
            component_var = {symbol: weights[i] * marginal_var[i] 
                           for i, symbol in enumerate(symbols)}
            
            # Marginal VaR
            marginal_var_dict = {symbol: marginal_var[i] for i, symbol in enumerate(symbols)}
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            diversification_ratio = portfolio_vol / weighted_avg_vol
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w**2 for w in weights)
            
            # Convert correlation matrix to dict
            correlation_dict = {}
            for i, symbol1 in enumerate(symbols):
                correlation_dict[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    correlation_dict[symbol1][symbol2] = correlation_matrix.iloc[i, j]
            
            return RiskAnalysis(
                portfolio_var=portfolio_var,
                component_var=component_var,
                marginal_var=marginal_var_dict,
                correlation_matrix=correlation_dict,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk analysis: {str(e)}")
            raise
    
    def calculate_attribution_analysis(self,
                                     portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     sector_weights: Dict[str, float],
                                     sector_returns: Dict[str, pd.Series]) -> AttributionAnalysis:
        """
        Calculate performance attribution analysis
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series  
            sector_weights: Sector weight allocations
            sector_returns: Sector return series
            
        Returns:
            AttributionAnalysis object
        """
        try:
            # Align all return series
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            })
            
            for sector, returns in sector_returns.items():
                aligned_data[f'sector_{sector}'] = returns
                
            aligned_data = aligned_data.dropna()
            
            # Calculate excess returns
            excess_return = aligned_data['portfolio'].mean() - aligned_data['benchmark'].mean()
            
            # Asset allocation effect
            # Weight difference * benchmark sector return
            asset_allocation = 0
            for sector in sector_weights:
                if f'sector_{sector}' in aligned_data.columns:
                    # Assume benchmark weight is equal weight for simplicity
                    benchmark_weight = 1.0 / len(sector_weights)
                    weight_diff = sector_weights[sector] - benchmark_weight
                    sector_return = aligned_data[f'sector_{sector}'].mean()
                    asset_allocation += weight_diff * sector_return
            
            # Security selection effect
            # Portfolio weight * (security return - benchmark sector return)
            security_selection = 0
            for sector in sector_weights:
                if f'sector_{sector}' in aligned_data.columns:
                    portfolio_weight = sector_weights[sector]
                    # For simplification, assume security return equals portfolio return
                    security_return = aligned_data['portfolio'].mean()
                    benchmark_sector_return = aligned_data[f'sector_{sector}'].mean()
                    security_selection += portfolio_weight * (security_return - benchmark_sector_return)
            
            # Interaction effect
            interaction_effect = excess_return - asset_allocation - security_selection
            
            # Sector attribution breakdown
            sector_attribution = {}
            for sector in sector_weights:
                if f'sector_{sector}' in aligned_data.columns:
                    weight = sector_weights[sector]
                    sector_return = aligned_data[f'sector_{sector}'].mean()
                    benchmark_sector_return = aligned_data['benchmark'].mean()  # Simplified
                    sector_attribution[sector] = weight * (sector_return - benchmark_sector_return)
            
            # Security attribution (simplified)
            security_attribution = {sector: 0 for sector in sector_weights}
            
            return AttributionAnalysis(
                security_selection=security_selection,
                asset_allocation=asset_allocation,
                interaction_effect=interaction_effect,
                total_excess_return=excess_return,
                sector_attribution=sector_attribution,
                security_attribution=security_attribution
            )
            
        except Exception as e:
            logger.error(f"Error calculating attribution analysis: {str(e)}")
            raise
    
    def calculate_time_series_analysis(self, 
                                     price_data: pd.Series,
                                     window_size: int = 30) -> Dict[str, Any]:
        """
        Perform time series analysis on price data
        
        Args:
            price_data: Time series of price data
            window_size: Rolling window size for calculations
            
        Returns:
            Dictionary with time series analysis results
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Rolling statistics
            rolling_mean = returns.rolling(window=window_size).mean()
            rolling_std = returns.rolling(window=window_size).std()
            rolling_sharpe = (rolling_mean * self.trading_days_per_year) / (rolling_std * np.sqrt(self.trading_days_per_year))
            
            # Trend analysis
            x = np.arange(len(price_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_data.values)
            
            # Volatility clustering (GARCH-like analysis)
            squared_returns = returns ** 2
            volatility_persistence = squared_returns.autocorr(lag=1)
            
            # Seasonality detection (simplified)
            if len(price_data) >= 252:  # At least 1 year of data
                monthly_returns = returns.groupby(returns.index.month).mean()
                seasonality_score = monthly_returns.std()
            else:
                seasonality_score = 0
            
            return {
                'trend_slope': slope,
                'trend_r_squared': r_value ** 2,
                'trend_p_value': p_value,
                'volatility_persistence': volatility_persistence,
                'seasonality_score': seasonality_score,
                'rolling_sharpe': rolling_sharpe.tolist(),
                'rolling_volatility': (rolling_std * np.sqrt(self.trading_days_per_year)).tolist(),
                'current_trend': 'upward' if slope > 0 else 'downward',
                'trend_strength': abs(r_value)
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            raise
    
    def calculate_correlation_analysis(self, 
                                     returns_data: pd.DataFrame,
                                     method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation analysis between assets
        
        Args:
            returns_data: DataFrame with return data for assets
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr(method=method)
            
            # Find highest and lowest correlations
            correlation_pairs = []
            symbols = correlation_matrix.columns
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = correlation_matrix.loc[symbol1, symbol2]
                        correlation_pairs.append({
                            'asset1': symbol1,
                            'asset2': symbol2,
                            'correlation': corr_value
                        })
            
            # Sort by correlation strength
            correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Calculate average correlation
            correlations = [pair['correlation'] for pair in correlation_pairs]
            avg_correlation = np.mean(correlations) if correlations else 0
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'highest_correlations': correlation_pairs[:5],
                'lowest_correlations': sorted(correlation_pairs, key=lambda x: abs(x['correlation']))[:5],
                'average_correlation': avg_correlation,
                'correlation_distribution': {
                    'strong_positive': len([c for c in correlations if c > 0.7]),
                    'moderate_positive': len([c for c in correlations if 0.3 < c <= 0.7]),
                    'weak': len([c for c in correlations if -0.3 <= c <= 0.3]),
                    'moderate_negative': len([c for c in correlations if -0.7 <= c < -0.3]),
                    'strong_negative': len([c for c in correlations if c < -0.7])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    def calculate_volatility_forecast(self,
                                    returns: pd.Series,
                                    forecast_days: int = 30,
                                    model: str = 'ewm') -> Dict[str, Any]:
        """
        Calculate volatility forecast
        
        Args:
            returns: Return series
            forecast_days: Number of days to forecast
            model: Volatility model ('ewm', 'rolling', 'garch')
            
        Returns:
            Dictionary with volatility forecast results
        """
        try:
            if model == 'ewm':
                # Exponentially weighted moving average
                volatility = returns.ewm(span=30).std() * np.sqrt(self.trading_days_per_year)
                forecast_vol = volatility.iloc[-1]  # Use last value as forecast
                
            elif model == 'rolling':
                # Rolling window standard deviation
                volatility = returns.rolling(window=30).std() * np.sqrt(self.trading_days_per_year)
                forecast_vol = volatility.iloc[-1]
                
            else:  # simplified GARCH
                # Simple GARCH(1,1) approximation
                squared_returns = returns ** 2
                volatility = squared_returns.ewm(alpha=0.1).mean() ** 0.5 * np.sqrt(self.trading_days_per_year)
                forecast_vol = volatility.iloc[-1]
            
            # Calculate volatility regime
            vol_median = volatility.median()
            current_vol = volatility.iloc[-1]
            
            if current_vol > vol_median * 1.5:
                regime = 'high'
            elif current_vol < vol_median * 0.5:
                regime = 'low'
            else:
                regime = 'normal'
            
            return {
                'forecast_volatility': forecast_vol,
                'historical_volatility': volatility.tolist(),
                'volatility_percentile': (volatility < current_vol).mean() * 100,
                'regime': regime,
                'model_used': model,
                'forecast_confidence': 0.68  # Simplified confidence interval
            }
            
        except Exception as e:
            logger.error(f"Error in volatility forecast: {str(e)}")
            raise
    
    # Helper methods
    def _calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            aligned_returns = asset_returns.align(benchmark_returns, join='inner')
            asset_ret, bench_ret = aligned_returns[0], aligned_returns[1]
            
            covariance = np.cov(asset_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            
            return covariance / benchmark_variance if benchmark_variance != 0 else 0
        except:
            return 0
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> List[int]:
        """Calculate drawdown duration periods"""
        try:
            durations = []
            current_duration = 0
            
            for drawdown in drawdowns:
                if drawdown < 0:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
            
            # Add final duration if still in drawdown
            if current_duration > 0:
                durations.append(current_duration)
                
            return durations
        except:
            return [0]
    
    async def calculate_portfolio_analytics_async(self, 
                                                portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async wrapper for comprehensive portfolio analytics
        
        Args:
            portfolio_data: Dictionary containing portfolio data
            
        Returns:
            Dictionary with comprehensive analytics results
        """
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive calculations in executor
        tasks = []
        
        if 'values' in portfolio_data:
            values = pd.Series(portfolio_data['values'])
            benchmark = pd.Series(portfolio_data.get('benchmark', []))
            
            # Performance metrics
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self.calculate_portfolio_performance,
                    values,
                    benchmark if len(benchmark) > 0 else None
                )
            )
            
            # Time series analysis
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self.calculate_time_series_analysis,
                    values
                )
            )
        
        if 'positions' in portfolio_data and 'returns_data' in portfolio_data:
            positions = portfolio_data['positions']
            returns_data = pd.DataFrame(portfolio_data['returns_data'])
            
            # Risk analysis
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self.calculate_risk_analysis,
                    positions,
                    returns_data
                )
            )
            
            # Correlation analysis
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self.calculate_correlation_analysis,
                    returns_data
                )
            )
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        analytics_results = {}
        result_keys = ['performance_metrics', 'time_series_analysis', 'risk_analysis', 'correlation_analysis']
        
        for i, result in enumerate(results):
            if i < len(result_keys) and not isinstance(result, Exception):
                analytics_results[result_keys[i]] = result.to_dict() if hasattr(result, 'to_dict') else result
            elif isinstance(result, Exception):
                logger.error(f"Error in analytics calculation {result_keys[i] if i < len(result_keys) else i}: {str(result)}")
        
        return analytics_results

# Utility functions for analytics
def calculate_information_coefficient(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """Calculate information coefficient (IC) for prediction accuracy"""
    try:
        return np.corrcoef(predictions, actual_returns)[0, 1]
    except:
        return 0.0

def calculate_maximum_adverse_excursion(entry_prices: List[float], 
                                      exit_prices: List[float],
                                      low_prices: List[List[float]]) -> List[float]:
    """Calculate Maximum Adverse Excursion (MAE) for trades"""
    mae_values = []
    
    for i, (entry, exit) in enumerate(zip(entry_prices, exit_prices)):
        if i < len(low_prices):
            trade_lows = low_prices[i]
            max_adverse = min(trade_lows) if trade_lows else entry
            mae = (entry - max_adverse) / entry if entry != 0 else 0
            mae_values.append(mae)
        else:
            mae_values.append(0)
    
    return mae_values

def calculate_maximum_favorable_excursion(entry_prices: List[float],
                                        exit_prices: List[float], 
                                        high_prices: List[List[float]]) -> List[float]:
    """Calculate Maximum Favorable Excursion (MFE) for trades"""
    mfe_values = []
    
    for i, (entry, exit) in enumerate(zip(entry_prices, exit_prices)):
        if i < len(high_prices):
            trade_highs = high_prices[i]
            max_favorable = max(trade_highs) if trade_highs else entry
            mfe = (max_favorable - entry) / entry if entry != 0 else 0
            mfe_values.append(mfe)
        else:
            mfe_values.append(0)
    
    return mfe_values