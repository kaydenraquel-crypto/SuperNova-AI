"""SuperNova Optuna-Based Hyperparameter Optimization System

Comprehensive optimization service for automated strategy parameter tuning using Optuna.
Features multi-objective optimization, walk-forward analysis, and robust parameter search.
"""

from __future__ import annotations
import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Optuna imports with fallback
try:
    import optuna
    from optuna import Trial, Study
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.storages import RDBStorage
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_pareto_front
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = None
    Study = None

# Joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Local imports
from .backtester import run_vbt_backtest, VBT_AVAILABLE
from .config import settings
from .schemas import OHLCVBar
from .db import SessionLocal

# Configure logging
logger = logging.getLogger(__name__)

# Strategy parameter spaces definition
STRATEGY_PARAMETER_SPACES = {
    "sma_crossover": {
        "fast_period": (5, 50),
        "slow_period": (20, 200),
        "validation": lambda p: p["fast_period"] < p["slow_period"]
    },
    "rsi_strategy": {
        "rsi_period": (5, 30),
        "oversold_level": (20, 40),
        "overbought_level": (60, 80),
        "validation": lambda p: p["oversold_level"] < p["overbought_level"]
    },
    "macd_strategy": {
        "fast_period": (8, 15),
        "slow_period": (21, 35),
        "signal_period": (5, 15),
        "validation": lambda p: p["fast_period"] < p["slow_period"]
    },
    "bb_strategy": {
        "bb_period": (10, 30),
        "bb_std_dev": (1.5, 3.0),
        "rsi_period": (10, 20),
        "oversold_level": (20, 35),
        "overbought_level": (65, 80),
        "validation": lambda p: p["oversold_level"] < p["overbought_level"]
    },
    "sentiment_strategy": {
        "sentiment_threshold": (-1.0, 1.0),
        "confidence_threshold": (0.5, 0.95),
        "momentum_weight": (0.0, 1.0),
        "contrarian_weight": (0.0, 1.0),
        "validation": lambda p: True
    }
}

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs"""
    strategy_template: str
    parameter_space: Optional[Dict[str, Tuple]] = None
    
    # Optimization objectives
    primary_objective: str = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio
    secondary_objectives: List[str] = field(default_factory=lambda: ["max_drawdown", "win_rate"])
    
    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1  # parallel jobs
    
    # Validation settings
    walk_forward: bool = False
    validation_splits: int = 3
    min_samples_per_split: int = 252  # trading days
    
    # Pruning and sampling
    enable_pruning: bool = True
    sampler_type: str = "tpe"  # tpe, cmaes, random
    
    # Risk constraints
    max_drawdown_limit: float = 0.25  # 25% max drawdown
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.35
    
    # Transaction costs
    include_transaction_costs: bool = True
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001   # 0.1%

@dataclass
class OptimizationResult:
    """Results from optimization run"""
    study_name: str
    best_params: Dict[str, Any]
    best_value: float
    best_trial: int
    n_trials: int
    optimization_duration: float
    
    # Performance metrics
    metrics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]] = None
    
    # Multi-objective results
    pareto_front: Optional[List[Dict]] = None
    
    # Study statistics
    study_stats: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    convergence_info: Optional[Dict[str, Any]] = None

class OptunaOptimizer:
    """Advanced Optuna-based hyperparameter optimizer for trading strategies"""
    
    def __init__(self, storage_url: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            storage_url: Database URL for study persistence (e.g., 'sqlite:///optuna.db')
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna>=3.4.0")
        
        self.storage_url = storage_url or "sqlite:///optuna_studies.db"
        self.storage = RDBStorage(url=self.storage_url)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._active_studies: Dict[str, Study] = {}
        
    def create_study(
        self,
        study_name: str,
        config: OptimizationConfig,
        directions: Optional[List[str]] = None
    ) -> Study:
        """
        Create a new optimization study.
        
        Args:
            study_name: Unique identifier for the study
            config: Optimization configuration
            directions: List of optimization directions ('maximize', 'minimize')
        
        Returns:
            Optuna Study object
        """
        # Determine optimization directions
        if directions is None:
            directions = self._get_default_directions(config)
        
        # Configure sampler
        sampler = self._create_sampler(config)
        
        # Configure pruner
        pruner = self._create_pruner(config) if config.enable_pruning else None
        
        try:
            # Create or load existing study
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
                directions=directions,
                sampler=sampler,
                pruner=pruner
            )
            
            self._active_studies[study_name] = study
            logger.info(f"Created/loaded study '{study_name}' with {len(study.trials)} existing trials")
            
            return study
            
        except Exception as e:
            logger.error(f"Failed to create study '{study_name}': {e}")
            raise
    
    def optimize_strategy(
        self,
        study_name: str,
        bars_data: List[Dict],
        config: OptimizationConfig,
        progress_callback: Optional[Callable[[int, float], None]] = None
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using Optuna.
        
        Args:
            study_name: Unique study identifier
            bars_data: Historical OHLCV data for backtesting
            config: Optimization configuration
            progress_callback: Optional callback for progress updates
        
        Returns:
            OptimizationResult with best parameters and metrics
        """
        start_time = datetime.now()
        
        try:
            # Validate data
            if len(bars_data) < config.min_samples_per_split:
                raise ValueError(f"Insufficient data. Need at least {config.min_samples_per_split} bars")
            
            # Create study
            study = self.create_study(study_name, config)
            
            # Define objective function
            def objective(trial: Trial) -> Union[float, List[float]]:
                return self._objective_function(
                    trial, bars_data, config, progress_callback
                )
            
            # Run optimization
            logger.info(f"Starting optimization for study '{study_name}' with {config.n_trials} trials")
            
            if config.n_jobs > 1 and JOBLIB_AVAILABLE:
                # Parallel optimization
                logger.info(f"Using parallel optimization with {config.n_jobs} jobs")
                study.optimize(
                    objective,
                    n_trials=config.n_trials,
                    timeout=config.timeout,
                    n_jobs=config.n_jobs
                )
            else:
                # Sequential optimization
                study.optimize(
                    objective,
                    n_trials=config.n_trials,
                    timeout=config.timeout,
                    callbacks=[self._create_progress_callback(progress_callback)] if progress_callback else None
                )
            
            # Prepare results
            result = self._prepare_optimization_result(
                study, config, start_time, bars_data
            )
            
            logger.info(f"Optimization completed. Best value: {result.best_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for study '{study_name}': {e}")
            raise
    
    def optimize_multi_symbol(
        self,
        symbols_data: Dict[str, List[Dict]],
        config: OptimizationConfig,
        study_prefix: str = "multi_symbol",
        aggregate_results: bool = True
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize strategies for multiple symbols.
        
        Args:
            symbols_data: Dictionary mapping symbols to their OHLCV data
            config: Optimization configuration
            study_prefix: Prefix for study names
            aggregate_results: Whether to create an aggregated study
        
        Returns:
            Dictionary mapping symbols to their optimization results
        """
        results = {}
        
        try:
            if config.n_jobs > 1 and len(symbols_data) > 1:
                # Parallel symbol optimization
                logger.info(f"Optimizing {len(symbols_data)} symbols in parallel")
                
                def optimize_single_symbol(symbol, bars_data):
                    study_name = f"{study_prefix}_{symbol}"
                    return symbol, self.optimize_strategy(study_name, bars_data, config)
                
                symbol_results = Parallel(n_jobs=min(config.n_jobs, len(symbols_data)))(
                    delayed(optimize_single_symbol)(symbol, bars_data)
                    for symbol, bars_data in symbols_data.items()
                )
                
                results = dict(symbol_results)
                
            else:
                # Sequential symbol optimization
                for symbol, bars_data in symbols_data.items():
                    logger.info(f"Optimizing strategy for {symbol}")
                    study_name = f"{study_prefix}_{symbol}"
                    results[symbol] = self.optimize_strategy(study_name, bars_data, config)
            
            # Optionally create aggregated study
            if aggregate_results and len(results) > 1:
                logger.info("Creating aggregated multi-symbol study")
                results["_aggregated"] = self._create_aggregated_study(results, config)
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-symbol optimization failed: {e}")
            raise
    
    def walk_forward_optimization(
        self,
        study_name: str,
        bars_data: List[Dict],
        config: OptimizationConfig,
        window_size: int = 252,  # trading days
        step_size: int = 63     # quarter
    ) -> OptimizationResult:
        """
        Perform walk-forward optimization for robust parameter estimation.
        
        Args:
            study_name: Study identifier
            bars_data: Historical OHLCV data
            config: Optimization configuration
            window_size: Size of optimization window
            step_size: Step size for walk-forward
        
        Returns:
            OptimizationResult with walk-forward validation metrics
        """
        logger.info(f"Starting walk-forward optimization for {study_name}")
        
        try:
            # Calculate walk-forward windows
            windows = self._calculate_walkforward_windows(
                len(bars_data), window_size, step_size
            )
            
            if len(windows) < 2:
                raise ValueError("Insufficient data for walk-forward optimization")
            
            # Optimize on each window
            window_results = []
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(f"Walk-forward window {i+1}/{len(windows)}")
                
                # Extract training and testing data
                train_data = bars_data[train_start:train_end]
                test_data = bars_data[test_start:test_end]
                
                if len(train_data) < config.min_samples_per_split:
                    continue
                
                # Create window-specific study
                window_study_name = f"{study_name}_wf_{i}"
                window_config = self._create_window_config(config, i)
                
                # Optimize on training data
                window_result = self.optimize_strategy(
                    window_study_name, train_data, window_config
                )
                
                # Validate on test data
                validation_metrics = self._validate_parameters(
                    window_result.best_params,
                    test_data,
                    config
                )
                
                window_result.validation_metrics = validation_metrics
                window_results.append(window_result)
            
            # Aggregate walk-forward results
            aggregated_result = self._aggregate_walkforward_results(
                study_name, window_results, config
            )
            
            logger.info(f"Walk-forward optimization completed with {len(window_results)} windows")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {e}")
            raise
    
    def get_study_statistics(self, study_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a study."""
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage
            )
            
            trials = study.trials
            if not trials:
                return {"error": "No trials found"}
            
            # Basic statistics
            stats = {
                "study_name": study_name,
                "n_trials": len(trials),
                "n_complete_trials": len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned_trials": len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED]),
                "n_failed_trials": len([t for t in trials if t.state == optuna.trial.TrialState.FAIL]),
                "creation_time": study.system_attrs.get("creation_time"),
                "directions": [d.name for d in study.directions]
            }
            
            # Best trial information
            if study.best_trials:
                stats["best_trials"] = [
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "duration": trial.duration.total_seconds() if trial.duration else None
                    }
                    for trial in study.best_trials[:5]  # Top 5 trials
                ]
            
            # Parameter importance (if available)
            try:
                importances = optuna.importance.get_param_importances(study)
                stats["parameter_importance"] = importances
            except:
                stats["parameter_importance"] = None
            
            # Convergence information
            complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(complete_trials) >= 10:
                recent_values = [t.value for t in complete_trials[-10:]]
                stats["recent_performance"] = {
                    "last_10_mean": np.mean(recent_values),
                    "last_10_std": np.std(recent_values),
                    "improvement_trend": self._calculate_improvement_trend(complete_trials)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get study statistics: {e}")
            return {"error": str(e)}
    
    def create_optimization_dashboard_data(self, study_names: List[str]) -> Dict[str, Any]:
        """Create data structure for optimization dashboard visualization."""
        dashboard_data = {
            "studies": {},
            "summary": {
                "total_studies": len(study_names),
                "total_trials": 0,
                "best_performers": [],
                "parameter_insights": {}
            },
            "generated_at": datetime.now().isoformat()
        }
        
        try:
            for study_name in study_names:
                study_stats = self.get_study_statistics(study_name)
                if "error" not in study_stats:
                    dashboard_data["studies"][study_name] = study_stats
                    dashboard_data["summary"]["total_trials"] += study_stats.get("n_trials", 0)
            
            # Find best performers across all studies
            all_best_trials = []
            for study_data in dashboard_data["studies"].values():
                best_trials = study_data.get("best_trials", [])
                all_best_trials.extend(best_trials)
            
            # Sort by value (assuming higher is better)
            dashboard_data["summary"]["best_performers"] = sorted(
                all_best_trials, 
                key=lambda x: x.get("value", float('-inf')), 
                reverse=True
            )[:10]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to create dashboard data: {e}")
            dashboard_data["error"] = str(e)
            return dashboard_data
    
    # Private helper methods
    
    def _objective_function(
        self,
        trial: Trial,
        bars_data: List[Dict],
        config: OptimizationConfig,
        progress_callback: Optional[Callable] = None
    ) -> Union[float, List[float]]:
        """Objective function for Optuna optimization."""
        try:
            # Sample parameters
            params = self._sample_parameters(trial, config)
            
            # Validate parameter combination
            if not self._validate_parameter_combination(params, config):
                raise optuna.exceptions.TrialPruned()
            
            # Run backtest
            metrics = self._run_backtest_with_params(params, bars_data, config)
            
            # Check risk constraints
            if not self._check_risk_constraints(metrics, config):
                raise optuna.exceptions.TrialPruned()
            
            # Extract objective values
            if len(config.secondary_objectives) == 0:
                # Single objective optimization
                return metrics.get(config.primary_objective, 0.0)
            else:
                # Multi-objective optimization
                objectives = [metrics.get(config.primary_objective, 0.0)]
                for obj in config.secondary_objectives:
                    objectives.append(metrics.get(obj, 0.0))
                return objectives
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            # Return worst possible value
            if len(config.secondary_objectives) == 0:
                return float('-inf')
            else:
                return [float('-inf')] * (1 + len(config.secondary_objectives))
    
    def _sample_parameters(self, trial: Trial, config: OptimizationConfig) -> Dict[str, Any]:
        """Sample parameters for the trial."""
        params = {}
        
        # Get parameter space
        param_space = config.parameter_space or STRATEGY_PARAMETER_SPACES.get(config.strategy_template, {})
        
        for param_name, param_range in param_space.items():
            if param_name == "validation":
                continue
                
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif isinstance(low, float) or isinstance(high, float):
                    params[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, [low, high])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        
        return params
    
    def _validate_parameter_combination(self, params: Dict[str, Any], config: OptimizationConfig) -> bool:
        """Validate that parameter combination makes sense."""
        param_space = config.parameter_space or STRATEGY_PARAMETER_SPACES.get(config.strategy_template, {})
        validation_func = param_space.get("validation")
        
        if validation_func:
            return validation_func(params)
        return True
    
    def _run_backtest_with_params(
        self,
        params: Dict[str, Any],
        bars_data: List[Dict],
        config: OptimizationConfig
    ) -> Dict[str, float]:
        """Run backtest with given parameters."""
        if not VBT_AVAILABLE:
            raise RuntimeError("VectorBT not available for backtesting")
        
        try:
            # Add transaction costs if enabled
            fees = config.commission if config.include_transaction_costs else 0.0
            slippage = config.slippage if config.include_transaction_costs else 0.0
            
            metrics = run_vbt_backtest(
                bars=bars_data,
                strategy_template=config.strategy_template,
                params=params,
                start_cash=10000.0,
                fees=fees,
                slippage=slippage
            )
            
            if isinstance(metrics, dict) and "error" not in metrics:
                return metrics
            else:
                logger.warning(f"Backtest failed: {metrics}")
                return {"sharpe_ratio": -10.0, "total_return": -1.0, "max_drawdown": 1.0}
                
        except Exception as e:
            logger.warning(f"Backtest execution failed: {e}")
            return {"sharpe_ratio": -10.0, "total_return": -1.0, "max_drawdown": 1.0}
    
    def _check_risk_constraints(self, metrics: Dict[str, float], config: OptimizationConfig) -> bool:
        """Check if metrics meet risk constraints."""
        max_dd = metrics.get("Max Drawdown", metrics.get("max_drawdown", 0))
        sharpe = metrics.get("Sharpe", metrics.get("sharpe_ratio", 0))
        win_rate = metrics.get("Win Rate", metrics.get("win_rate", 0))
        
        if abs(max_dd) > config.max_drawdown_limit:
            return False
        if sharpe < config.min_sharpe_ratio:
            return False  
        if win_rate < config.min_win_rate:
            return False
            
        return True
    
    def _create_sampler(self, config: OptimizationConfig):
        """Create appropriate sampler based on configuration."""
        if config.sampler_type == "tpe":
            return TPESampler(n_startup_trials=10, n_ei_candidates=24)
        elif config.sampler_type == "cmaes":
            return CmaEsSampler()
        else:
            return optuna.samplers.RandomSampler()
    
    def _create_pruner(self, config: OptimizationConfig):
        """Create appropriate pruner based on configuration."""
        if config.n_trials < 20:
            return None  # Don't prune with few trials
        
        return MedianPruner(
            n_startup_trials=max(5, config.n_trials // 10),
            n_warmup_steps=10,
            interval_steps=5
        )
    
    def _get_default_directions(self, config: OptimizationConfig) -> List[str]:
        """Get default optimization directions based on objectives."""
        directions = []
        
        # Primary objective
        if "return" in config.primary_objective.lower() or "sharpe" in config.primary_objective.lower():
            directions.append("maximize")
        elif "drawdown" in config.primary_objective.lower() or "risk" in config.primary_objective.lower():
            directions.append("minimize")
        else:
            directions.append("maximize")  # Default
        
        # Secondary objectives
        for obj in config.secondary_objectives:
            if "drawdown" in obj.lower() or "risk" in obj.lower():
                directions.append("minimize")
            else:
                directions.append("maximize")
        
        return directions
    
    def _create_progress_callback(self, progress_callback: Optional[Callable]) -> Callable:
        """Create progress callback wrapper."""
        def callback(study: Study, trial):
            if progress_callback:
                progress = len(study.trials)
                best_value = study.best_value if study.best_trial else None
                progress_callback(progress, best_value)
        
        return callback
    
    def _prepare_optimization_result(
        self,
        study: Study,
        config: OptimizationConfig,
        start_time: datetime,
        bars_data: List[Dict]
    ) -> OptimizationResult:
        """Prepare optimization result from completed study."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params if best_trial else {}
        best_value = best_trial.value if best_trial else 0.0
        
        # Run final backtest with best parameters
        best_metrics = {}
        if best_params:
            best_metrics = self._run_backtest_with_params(
                best_params, bars_data, config
            )
        
        # Get study statistics
        study_stats = {
            "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
        
        # Get Pareto front for multi-objective
        pareto_front = None
        if len(config.secondary_objectives) > 0:
            try:
                pareto_trials = study.best_trials
                pareto_front = [
                    {
                        "params": trial.params,
                        "values": trial.values,
                        "number": trial.number
                    }
                    for trial in pareto_trials[:10]  # Top 10
                ]
            except:
                pass
        
        return OptimizationResult(
            study_name=study.study_name,
            best_params=best_params,
            best_value=best_value,
            best_trial=best_trial.number if best_trial else -1,
            n_trials=len(study.trials),
            optimization_duration=duration,
            metrics=best_metrics,
            pareto_front=pareto_front,
            study_stats=study_stats
        )
    
    def _calculate_walkforward_windows(
        self,
        data_length: int,
        window_size: int,
        step_size: int
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate walk-forward optimization windows."""
        windows = []
        
        for start in range(0, data_length - window_size - step_size, step_size):
            train_start = start
            train_end = start + window_size
            test_start = train_end
            test_end = min(test_start + step_size, data_length)
            
            if test_end - test_start >= step_size // 2:  # Minimum test size
                windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def _create_window_config(self, config: OptimizationConfig, window_idx: int) -> OptimizationConfig:
        """Create configuration for walk-forward window."""
        window_config = OptimizationConfig(
            strategy_template=config.strategy_template,
            parameter_space=config.parameter_space,
            primary_objective=config.primary_objective,
            secondary_objectives=config.secondary_objectives,
            n_trials=max(20, config.n_trials // 4),  # Fewer trials per window
            n_jobs=1,  # Sequential for windows
            enable_pruning=config.enable_pruning,
            sampler_type=config.sampler_type,
            max_drawdown_limit=config.max_drawdown_limit,
            min_sharpe_ratio=config.min_sharpe_ratio,
            min_win_rate=config.min_win_rate,
            include_transaction_costs=config.include_transaction_costs,
            commission=config.commission,
            slippage=config.slippage
        )
        return window_config
    
    def _validate_parameters(
        self,
        params: Dict[str, Any],
        test_data: List[Dict],
        config: OptimizationConfig
    ) -> Dict[str, float]:
        """Validate parameters on test data."""
        return self._run_backtest_with_params(params, test_data, config)
    
    def _aggregate_walkforward_results(
        self,
        study_name: str,
        window_results: List[OptimizationResult],
        config: OptimizationConfig
    ) -> OptimizationResult:
        """Aggregate walk-forward optimization results."""
        if not window_results:
            raise ValueError("No window results to aggregate")
        
        # Calculate average performance across windows
        avg_metrics = defaultdict(list)
        all_params = []
        
        for result in window_results:
            all_params.append(result.best_params)
            for metric, value in result.metrics.items():
                avg_metrics[metric].append(value)
            
            if result.validation_metrics:
                for metric, value in result.validation_metrics.items():
                    avg_metrics[f"val_{metric}"].append(value)
        
        # Calculate aggregated metrics
        final_metrics = {}
        for metric, values in avg_metrics.items():
            if values:
                final_metrics[f"avg_{metric}"] = np.mean(values)
                final_metrics[f"std_{metric}"] = np.std(values)
                final_metrics[f"min_{metric}"] = np.min(values)
                final_metrics[f"max_{metric}"] = np.max(values)
        
        # Find most common parameters (consensus)
        consensus_params = self._find_consensus_parameters(all_params)
        
        # Create aggregated result
        total_duration = sum(r.optimization_duration for r in window_results)
        total_trials = sum(r.n_trials for r in window_results)
        
        return OptimizationResult(
            study_name=f"{study_name}_walkforward_aggregated",
            best_params=consensus_params,
            best_value=final_metrics.get(f"avg_{config.primary_objective}", 0.0),
            best_trial=-1,
            n_trials=total_trials,
            optimization_duration=total_duration,
            metrics=final_metrics,
            validation_metrics=None,
            study_stats={
                "n_windows": len(window_results),
                "consensus_strength": self._calculate_consensus_strength(all_params)
            }
        )
    
    def _find_consensus_parameters(self, all_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find consensus parameters across walk-forward windows."""
        if not all_params:
            return {}
        
        consensus = {}
        param_names = set()
        for params in all_params:
            param_names.update(params.keys())
        
        for param_name in param_names:
            values = [params.get(param_name) for params in all_params if param_name in params]
            values = [v for v in values if v is not None]
            
            if not values:
                continue
            
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric parameter - use median
                consensus[param_name] = float(np.median(values))
            else:
                # Categorical parameter - use mode
                from collections import Counter
                counter = Counter(values)
                consensus[param_name] = counter.most_common(1)[0][0]
        
        return consensus
    
    def _calculate_consensus_strength(self, all_params: List[Dict[str, Any]]) -> float:
        """Calculate how much consensus there is across parameters."""
        if len(all_params) <= 1:
            return 1.0
        
        consensus_scores = []
        param_names = set()
        for params in all_params:
            param_names.update(params.keys())
        
        for param_name in param_names:
            values = [params.get(param_name) for params in all_params if param_name in params]
            values = [v for v in values if v is not None]
            
            if len(values) <= 1:
                consensus_scores.append(1.0)
                continue
            
            if all(isinstance(v, (int, float)) for v in values):
                # For numeric: use coefficient of variation
                std_dev = np.std(values)
                mean_val = np.mean(values)
                if mean_val != 0:
                    cv = std_dev / abs(mean_val)
                    consensus_scores.append(max(0.0, 1.0 - cv))
                else:
                    consensus_scores.append(1.0 if std_dev == 0 else 0.0)
            else:
                # For categorical: use frequency of most common value
                from collections import Counter
                counter = Counter(values)
                most_common_freq = counter.most_common(1)[0][1]
                consensus_scores.append(most_common_freq / len(values))
        
        return np.mean(consensus_scores) if consensus_scores else 0.0
    
    def _create_aggregated_study(
        self,
        symbol_results: Dict[str, OptimizationResult],
        config: OptimizationConfig
    ) -> OptimizationResult:
        """Create aggregated study from multi-symbol optimization."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated aggregation
        
        all_params = []
        all_metrics = defaultdict(list)
        
        for symbol, result in symbol_results.items():
            if symbol.startswith("_"):  # Skip metadata entries
                continue
            
            all_params.append(result.best_params)
            for metric, value in result.metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate consensus parameters
        consensus_params = self._find_consensus_parameters(all_params)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                avg_metrics[f"avg_{metric}"] = np.mean(values)
                avg_metrics[f"std_{metric}"] = np.std(values)
        
        return OptimizationResult(
            study_name="multi_symbol_aggregated",
            best_params=consensus_params,
            best_value=avg_metrics.get(f"avg_{config.primary_objective}", 0.0),
            best_trial=-1,
            n_trials=sum(r.n_trials for r in symbol_results.values() if not r.study_name.startswith("_")),
            optimization_duration=sum(r.optimization_duration for r in symbol_results.values() if not r.study_name.startswith("_")),
            metrics=avg_metrics,
            study_stats={
                "n_symbols": len([s for s in symbol_results.keys() if not s.startswith("_")]),
                "consensus_strength": self._calculate_consensus_strength(all_params)
            }
        )
    
    def _calculate_improvement_trend(self, trials: List) -> str:
        """Calculate whether optimization is still improving."""
        if len(trials) < 20:
            return "insufficient_data"
        
        # Look at last 20 trials
        recent_values = [t.value for t in trials[-20:]]
        
        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.corrcoef(x, recent_values)[0, 1]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "converged"
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Convenience functions for common optimization patterns

def optimize_sma_crossover(
    bars_data: List[Dict],
    study_name: str = "sma_crossover_opt",
    n_trials: int = 100,
    storage_url: Optional[str] = None
) -> OptimizationResult:
    """Optimize SMA crossover strategy."""
    config = OptimizationConfig(
        strategy_template="sma_crossover",
        n_trials=n_trials,
        primary_objective="sharpe_ratio",
        secondary_objectives=["max_drawdown"]
    )
    
    optimizer = OptunaOptimizer(storage_url)
    return optimizer.optimize_strategy(study_name, bars_data, config)

def optimize_rsi_strategy(
    bars_data: List[Dict],
    study_name: str = "rsi_strategy_opt", 
    n_trials: int = 100,
    storage_url: Optional[str] = None
) -> OptimizationResult:
    """Optimize RSI strategy."""
    config = OptimizationConfig(
        strategy_template="rsi_strategy",
        n_trials=n_trials,
        primary_objective="sharpe_ratio",
        secondary_objectives=["win_rate"]
    )
    
    optimizer = OptunaOptimizer(storage_url)
    return optimizer.optimize_strategy(study_name, bars_data, config)

def optimize_multi_strategy(
    symbols_data: Dict[str, List[Dict]],
    strategy_template: str,
    n_trials: int = 50,
    storage_url: Optional[str] = None
) -> Dict[str, OptimizationResult]:
    """Optimize strategy across multiple symbols."""
    config = OptimizationConfig(
        strategy_template=strategy_template,
        n_trials=n_trials,
        primary_objective="sharpe_ratio",
        n_jobs=min(4, len(symbols_data))
    )
    
    optimizer = OptunaOptimizer(storage_url)
    return optimizer.optimize_multi_symbol(symbols_data, config)

# Export main classes and functions
__all__ = [
    "OptunaOptimizer",
    "OptimizationConfig", 
    "OptimizationResult",
    "STRATEGY_PARAMETER_SPACES",
    "optimize_sma_crossover",
    "optimize_rsi_strategy", 
    "optimize_multi_strategy",
    "OPTUNA_AVAILABLE"
]