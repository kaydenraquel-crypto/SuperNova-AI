"""
SuperNova ML Optimizer
Advanced AI/ML processing optimization with memory management, model caching, and GPU acceleration
"""

import torch
import numpy as np
import pandas as pd
import asyncio
import time
import gc
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache, wraps
import psutil
import pickle
import joblib
from pathlib import Path

# Transformers and ML libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BatchEncoding
    )
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Scikit-learn optimization
try:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# VectorBT optimization
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

# Optuna optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .config import settings
from .cache_manager import cache_manager, cached
from .performance_monitor import performance_timer
from .async_processor import async_processor, TaskPriority

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """ML model performance metrics"""
    load_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    batch_size: int = 1
    throughput_per_second: float = 0.0
    accuracy: Optional[float] = None
    cache_hit_rate: float = 0.0
    gpu_utilization: float = 0.0
    model_size_mb: float = 0.0

@dataclass
class OptimizationConfig:
    """ML optimization configuration"""
    enable_gpu: bool = torch.cuda.is_available()
    batch_size: int = 32
    max_memory_mb: int = 4096
    enable_model_cache: bool = True
    enable_inference_cache: bool = True
    cache_ttl: int = 3600
    num_workers: int = 4
    enable_mixed_precision: bool = True
    enable_tensorrt: bool = False
    optimization_level: str = "balanced"  # aggressive, balanced, conservative

class GPUManager:
    """Manage GPU resources and optimization"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.memory_threshold = 0.8  # Use max 80% of GPU memory
        self.is_gpu_available = torch.cuda.is_available()
        
        if self.is_gpu_available:
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}, Memory: {self.total_memory / 1024**3:.1f} GB")
        
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for computation"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        else:
            return torch.device('cpu')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not self.is_gpu_available:
            return {'allocated': 0.0, 'cached': 0.0, 'max_allocated': 0.0}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.is_gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU cache cleared")
    
    def is_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available"""
        if not self.is_gpu_available:
            return True  # CPU has flexible memory
        
        current_usage = torch.cuda.memory_allocated()
        available = self.total_memory * self.memory_threshold - current_usage
        return available > (required_mb * 1024 * 1024)
    
    @contextmanager
    def memory_management(self):
        """Context manager for GPU memory management"""
        if self.is_gpu_available:
            initial_memory = torch.cuda.memory_allocated()
            try:
                yield
            finally:
                # Clean up and report memory usage
                final_memory = torch.cuda.memory_allocated()
                memory_used = (final_memory - initial_memory) / 1024**2  # MB
                if memory_used > 100:  # Log if >100MB used
                    logger.debug(f"GPU memory used: {memory_used:.1f} MB")
                
                # Clear cache if memory usage is high
                if torch.cuda.memory_allocated() / self.total_memory > self.memory_threshold:
                    self.clear_cache()
        else:
            yield

class ModelCache:
    """Intelligent model caching system"""
    
    def __init__(self, max_size: int = 5, max_memory_mb: int = 2048):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        
    def get_model(self, model_key: str, loader_func: Callable) -> Any:
        """Get model from cache or load it"""
        with self.lock:
            if model_key in self.models:
                self.access_times[model_key] = datetime.now()
                logger.debug(f"Model cache hit: {model_key}")
                return self.models[model_key]
            
            # Load model
            start_time = time.time()
            model = loader_func()
            load_time = time.time() - start_time
            
            # Estimate model size
            model_size_mb = self._estimate_model_size(model)
            
            # Check cache capacity
            if len(self.models) >= self.max_size or self._total_memory_usage() + model_size_mb > self.max_memory_mb:
                self._evict_lru_model()
            
            # Cache model
            self.models[model_key] = model
            self.model_info[model_key] = {
                'size_mb': model_size_mb,
                'load_time': load_time,
                'loaded_at': datetime.now()
            }
            self.access_times[model_key] = datetime.now()
            
            logger.info(f"Model cached: {model_key}, Size: {model_size_mb:.1f} MB, Load time: {load_time:.2f}s")
            return model
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB"""
        try:
            if hasattr(model, 'parameters'):
                # PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                return total_params * 4 / 1024 / 1024  # Assuming float32
            elif hasattr(model, 'get_params'):
                # Scikit-learn model
                return len(pickle.dumps(model)) / 1024 / 1024
            else:
                # Generic estimation
                return len(pickle.dumps(model)) / 1024 / 1024
        except:
            return 50.0  # Default estimate
    
    def _total_memory_usage(self) -> float:
        """Calculate total memory usage of cached models"""
        return sum(info['size_mb'] for info in self.model_info.values())
    
    def _evict_lru_model(self):
        """Evict least recently used model"""
        if not self.models:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        del self.models[lru_key]
        del self.model_info[lru_key]
        del self.access_times[lru_key]
        
        logger.info(f"Evicted LRU model: {lru_key}")
    
    def clear(self):
        """Clear all cached models"""
        with self.lock:
            self.models.clear()
            self.model_info.clear()
            self.access_times.clear()
            logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'cached_models': len(self.models),
                'total_memory_mb': self._total_memory_usage(),
                'model_info': dict(self.model_info)
            }

class SentimentModelOptimizer:
    """Optimized sentiment analysis model"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gpu_manager = GPUManager()
        self.model_cache = ModelCache()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Performance metrics
        self.metrics = ModelMetrics()
        
        # Batch processing
        self.batch_processor = None
        self.processing_queue = asyncio.Queue(maxsize=1000)
    
    async def initialize(self):
        """Initialize sentiment model with optimization"""
        logger.info("Initializing optimized sentiment model")
        
        model_name = settings.FINBERT_MODEL_PATH or "ProsusAI/finbert"
        
        def load_model():
            with self.gpu_manager.memory_management():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Move to optimal device
                model = model.to(self.gpu_manager.device)
                
                # Enable optimizations
                if self.config.enable_mixed_precision and self.gpu_manager.is_gpu_available:
                    model = model.half()  # Use FP16
                
                # Compile model for optimization (PyTorch 2.0+)
                if hasattr(torch, 'compile') and self.config.optimization_level == "aggressive":
                    try:
                        model = torch.compile(model)
                        logger.info("Model compiled with torch.compile")
                    except Exception as e:
                        logger.warning(f"Failed to compile model: {e}")
                
                # Create pipeline
                pipeline = transformers.pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.gpu_manager.is_gpu_available else -1,
                    batch_size=self.config.batch_size,
                    return_all_scores=True
                )
                
                return {'tokenizer': tokenizer, 'model': model, 'pipeline': pipeline}
        
        # Load model with caching
        components = self.model_cache.get_model(f"finbert_{model_name}", load_model)
        
        self.tokenizer = components['tokenizer']
        self.model = components['model']
        self.pipeline = components['pipeline']
        
        # Start batch processor
        self.batch_processor = asyncio.create_task(self._batch_processor())
        
        logger.info("Sentiment model initialized successfully")
    
    @cached(ttl=3600, key_prefix="sentiment_analysis")
    async def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for batch of texts with optimization"""
        
        if not texts:
            return []
        
        start_time = time.time()
        
        with self.gpu_manager.memory_management():
            # Process in optimal batch sizes
            results = []
            optimal_batch_size = min(self.config.batch_size, len(texts))
            
            for i in range(0, len(texts), optimal_batch_size):
                batch = texts[i:i + optimal_batch_size]
                
                # Run inference
                with torch.no_grad():
                    predictions = self.pipeline(batch)
                
                # Process results
                for j, pred in enumerate(predictions):
                    sentiment_scores = {item['label']: item['score'] for item in pred}
                    
                    # Determine overall sentiment
                    max_label = max(pred, key=lambda x: x['score'])['label']
                    overall_score = self._normalize_sentiment_score(max_label, sentiment_scores)
                    
                    results.append({
                        'text': batch[j],
                        'overall_score': overall_score,
                        'confidence': max(sentiment_scores.values()),
                        'scores': sentiment_scores,
                        'model': 'finbert_optimized'
                    })
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.inference_time = processing_time
        self.metrics.batch_size = len(texts)
        self.metrics.throughput_per_second = len(texts) / processing_time
        
        return results
    
    async def analyze_sentiment_single(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment for single text (uses batching internally)"""
        results = await self.analyze_sentiment_batch([text])
        return results[0] if results else {}
    
    async def _batch_processor(self):
        """Background batch processor for optimal throughput"""
        batch = []
        batch_timeout = 0.1  # 100ms timeout
        
        while True:
            try:
                # Collect batch
                start_time = time.time()
                
                while len(batch) < self.config.batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self.processing_queue.get(),
                            timeout=batch_timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                    
                    # Check timeout
                    if time.time() - start_time > batch_timeout:
                        break
                
                if batch:
                    # Process batch
                    texts = [item['text'] for item in batch]
                    results = await self.analyze_sentiment_batch(texts)
                    
                    # Send results back
                    for item, result in zip(batch, results):
                        if 'callback' in item:
                            await item['callback'](result)
                    
                    batch.clear()
                
                await asyncio.sleep(0.01)  # Brief pause
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                batch.clear()
                await asyncio.sleep(1)
    
    def _normalize_sentiment_score(self, label: str, scores: Dict[str, float]) -> float:
        """Normalize sentiment score to [-1, 1] range"""
        label_mapping = {
            'LABEL_0': -1,  # Negative
            'LABEL_1': 0,   # Neutral
            'LABEL_2': 1,   # Positive
            'negative': -1,
            'neutral': 0,
            'positive': 1
        }
        
        base_score = label_mapping.get(label.lower(), 0)
        confidence = scores.get(label, 0.5)
        
        # Adjust based on confidence
        return base_score * confidence
    
    def get_metrics(self) -> ModelMetrics:
        """Get model performance metrics"""
        memory_info = self.gpu_manager.get_memory_usage()
        
        self.metrics.memory_usage_mb = memory_info.get('allocated', 0) * 1024  # Convert GB to MB
        self.metrics.gpu_utilization = memory_info.get('allocated', 0) / memory_info.get('max_allocated', 1) * 100
        
        return self.metrics

class VectorBTOptimizer:
    """Optimized VectorBT backtesting with performance enhancements"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = ModelMetrics()
        
        # Enable numba optimizations
        if VBT_AVAILABLE:
            # Configure VectorBT for optimal performance
            vbt.settings.array_wrapper['freq'] = None  # Disable frequency inference for speed
            vbt.settings.broadcasting['index_from'] = 'strict'  # Strict indexing for performance
    
    @performance_timer(category="vectorbt")
    @cached(ttl=1800, key_prefix="vbt_backtest")
    async def run_optimized_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Run optimized VectorBT backtest"""
        
        if not VBT_AVAILABLE:
            raise RuntimeError("VectorBT not available")
        
        start_time = time.time()
        
        # Optimize data types for performance
        data = self._optimize_dataframe(data)
        
        # Run backtest in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_backtest_sync, data, strategy_func, params, **kwargs)
            results = await loop.run_in_executor(None, lambda: future.result())
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.inference_time = processing_time
        self.metrics.throughput_per_second = len(data) / processing_time
        
        return results
    
    def _run_backtest_sync(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous backtest execution"""
        
        try:
            # Generate signals
            entries, exits = strategy_func(data, **params)
            
            # Create portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                **kwargs
            )
            
            # Calculate metrics
            stats = portfolio.stats()
            
            return {
                'total_return': float(stats['Total Return [%]']),
                'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
                'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
                'win_rate': float(stats.get('Win Rate [%]', 0)),
                'total_trades': int(stats.get('# Trades', 0)),
                'avg_trade_duration': str(stats.get('Avg Trade Duration', '0 days')),
                'calmar_ratio': float(stats.get('Calmar Ratio', 0)),
                'sortino_ratio': float(stats.get('Sortino Ratio', 0)),
                'profit_factor': float(stats.get('Profit Factor', 0)),
                'portfolio': portfolio
            }
            
        except Exception as e:
            logger.error(f"VectorBT backtest error: {e}")
            return {'error': str(e)}
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for VectorBT performance"""
        
        # Convert to optimal data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df.columns:
                # Use float32 instead of float64 for memory efficiency
                df[col] = df[col].astype('float32')
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        # Sort index for optimal access patterns
        df.sort_index(inplace=True)
        
        return df

class OptunaOptimizer:
    """Optimized Optuna hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = ModelMetrics()
        
    @performance_timer(category="optuna")
    async def optimize_parameters(
        self,
        objective_func: Callable,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        study_name: str = None
    ) -> Dict[str, Any]:
        """Run optimized parameter optimization"""
        
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available")
        
        start_time = time.time()
        
        # Create optimized study
        sampler = TPESampler(
            n_startup_trials=min(10, n_trials // 10),
            n_ei_candidates=24,
            seed=42
        )
        
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Run optimization in thread pool
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future = executor.submit(self._optimize_sync, study, objective_func, n_trials)
            await loop.run_in_executor(None, lambda: future.result())
        
        # Get results
        best_params = study.best_params
        best_value = study.best_value
        
        # Update metrics
        optimization_time = time.time() - start_time
        self.metrics.inference_time = optimization_time
        self.metrics.throughput_per_second = n_trials / optimization_time
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'study': study
        }
    
    def _optimize_sync(self, study, objective_func: Callable, n_trials: int):
        """Synchronous optimization execution"""
        
        def wrapped_objective(trial):
            try:
                return objective_func(trial)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial error: {e}")
                return float('-inf')
        
        study.optimize(wrapped_objective, n_trials=n_trials, n_jobs=1)

class MLMemoryManager:
    """Advanced memory management for ML operations"""
    
    def __init__(self):
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.cleanup_interval = 300  # 5 minutes
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def force_cleanup(self):
        """Force memory cleanup"""
        # Python garbage collection
        collected = gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear numba cache if available
        try:
            from numba import cuda
            if cuda.is_available():
                cuda.select_device(0)
                cuda.close()
        except ImportError:
            pass
        
        logger.info(f"Memory cleanup completed. Collected {collected} objects")
        
        return self.get_memory_usage()
    
    @contextmanager
    def memory_monitoring(self, operation_name: str):
        """Context manager for memory monitoring"""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            duration = time.time() - start_time
            
            memory_increase = end_memory['rss_mb'] - start_memory['rss_mb']
            
            if memory_increase > 100:  # Log if >100MB increase
                logger.warning(
                    f"Operation '{operation_name}' used {memory_increase:.1f}MB memory "
                    f"in {duration:.2f}s"
                )
            
            # Auto cleanup if memory usage is high
            if end_memory['rss_mb'] > self.memory_threshold_mb:
                logger.info(f"High memory usage detected ({end_memory['rss_mb']:.1f}MB), running cleanup")
                self.force_cleanup()

# Global instances
gpu_manager = GPUManager()
model_cache = ModelCache()
memory_manager = MLMemoryManager()

# Optimized ML processing functions
@performance_timer(category="ml_processing")
async def process_sentiment_optimized(texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    """Process sentiment with full optimization"""
    
    config = OptimizationConfig(batch_size=batch_size)
    optimizer = SentimentModelOptimizer(config)
    
    await optimizer.initialize()
    
    with memory_manager.memory_monitoring("sentiment_processing"):
        results = await optimizer.analyze_sentiment_batch(texts)
    
    return results

@performance_timer(category="ml_processing")
async def run_optimized_backtest(
    data: pd.DataFrame,
    strategy_func: Callable,
    params: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Run optimized backtest"""
    
    config = OptimizationConfig()
    optimizer = VectorBTOptimizer(config)
    
    with memory_manager.memory_monitoring("backtest_processing"):
        results = await optimizer.run_optimized_backtest(data, strategy_func, params, **kwargs)
    
    return results

async def initialize_ml_optimizations():
    """Initialize ML optimization components"""
    logger.info("Initializing ML optimizations")
    
    # Clear caches and prepare for optimal performance
    if gpu_manager.is_gpu_available:
        gpu_manager.clear_cache()
    
    memory_manager.force_cleanup()
    
    # Warm up models if configured
    if settings.LLM_CACHE_ENABLED:
        try:
            config = OptimizationConfig()
            optimizer = SentimentModelOptimizer(config)
            await optimizer.initialize()
            
            # Warm up with sample text
            await optimizer.analyze_sentiment_single("Sample text for model warmup")
            
            logger.info("ML models warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    logger.info("ML optimizations initialized")

def get_ml_performance_stats() -> Dict[str, Any]:
    """Get ML performance statistics"""
    return {
        'gpu_manager': {
            'device': str(gpu_manager.device),
            'gpu_available': gpu_manager.is_gpu_available,
            'memory_usage': gpu_manager.get_memory_usage()
        },
        'model_cache': model_cache.get_stats(),
        'memory_manager': memory_manager.get_memory_usage(),
        'optimization_config': {
            'gpu_enabled': settings.USE_GPU,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'vectorbt_available': VBT_AVAILABLE,
            'optuna_available': OPTUNA_AVAILABLE
        }
    }

# Export key components
__all__ = [
    'GPUManager', 'ModelCache', 'SentimentModelOptimizer', 'VectorBTOptimizer',
    'OptunaOptimizer', 'MLMemoryManager', 'OptimizationConfig',
    'process_sentiment_optimized', 'run_optimized_backtest', 
    'initialize_ml_optimizations', 'get_ml_performance_stats',
    'gpu_manager', 'model_cache', 'memory_manager'
]