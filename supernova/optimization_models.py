"""SuperNova Optimization Database Models

SQLAlchemy models for storing optimization studies, trials, and results.
Supports comprehensive tracking of hyperparameter optimization history.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import enum

from .db import Base

# Enums for study and trial states
class StudyStatus(enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class TrialState(enum.Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    PRUNED = "pruned" 
    FAILED = "failed"

class OptimizationPriority(enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class OptimizationStudyModel(Base):
    """
    Optimization study metadata and configuration.
    
    Stores high-level information about optimization runs including
    configuration, status, and aggregate results.
    """
    __tablename__ = "optimization_studies"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(String(255), unique=True, index=True, nullable=False)
    study_name = Column(String(255), index=True, nullable=False)
    
    # Target configuration
    symbol = Column(String(50), index=True, nullable=False)
    strategy_template = Column(String(100), index=True, nullable=False)
    
    # Study configuration (stored as JSON)
    configuration = Column(JSON, nullable=False)
    parameter_space = Column(JSON, nullable=True)
    
    # Optimization settings
    n_trials = Column(Integer, nullable=False)
    primary_objective = Column(String(100), nullable=False)
    secondary_objectives = Column(JSON, nullable=True)  # List of objectives
    
    # Study status and progress
    status = Column(SQLEnum(StudyStatus), default=StudyStatus.CREATED, index=True)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Trial statistics
    n_complete_trials = Column(Integer, default=0)
    n_pruned_trials = Column(Integer, default=0)
    n_failed_trials = Column(Integer, default=0)
    
    # Best results
    best_value = Column(Float, nullable=True)
    best_params = Column(JSON, nullable=True)
    best_trial_number = Column(Integer, nullable=True)
    best_metrics = Column(JSON, nullable=True)
    
    # Timing information
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Metadata
    user_attributes = Column(JSON, nullable=True)
    system_attributes = Column(JSON, nullable=True)
    
    # Risk and validation settings
    max_drawdown_limit = Column(Float, default=0.25)
    min_sharpe_ratio = Column(Float, default=0.5)
    min_win_rate = Column(Float, default=0.35)
    
    # Transaction cost settings
    include_transaction_costs = Column(Boolean, default=True)
    commission = Column(Float, default=0.001)
    slippage = Column(Float, default=0.001)
    
    # Advanced features
    walk_forward = Column(Boolean, default=False)
    validation_splits = Column(Integer, default=3)
    
    # Relationships
    trials = relationship("OptimizationTrialModel", back_populates="study", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_study_symbol_strategy', 'symbol', 'strategy_template'),
        Index('idx_study_status_created', 'status', 'created_at'),
        Index('idx_study_completion', 'completed_at'),
        CheckConstraint('progress >= 0.0 AND progress <= 1.0', name='progress_range_check'),
    )
    
    @hybrid_property
    def total_trials(self) -> int:
        """Total number of trials (complete + pruned + failed)"""
        return self.n_complete_trials + self.n_pruned_trials + self.n_failed_trials
    
    @hybrid_property
    def success_rate(self) -> float:
        """Percentage of successful trials"""
        total = self.total_trials
        if total == 0:
            return 0.0
        return (self.n_complete_trials / total) * 100.0
    
    @hybrid_property
    def duration_seconds(self) -> Optional[float]:
        """Study duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert study to dictionary representation"""
        return {
            'study_id': self.study_id,
            'study_name': self.study_name,
            'symbol': self.symbol,
            'strategy_template': self.strategy_template,
            'status': self.status.value if self.status else None,
            'progress': self.progress,
            'n_trials': self.n_trials,
            'n_complete_trials': self.n_complete_trials,
            'n_pruned_trials': self.n_pruned_trials,
            'n_failed_trials': self.n_failed_trials,
            'best_value': self.best_value,
            'best_params': self.best_params,
            'best_trial_number': self.best_trial_number,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'success_rate': self.success_rate,
            'configuration': self.configuration,
            'primary_objective': self.primary_objective,
            'secondary_objectives': self.secondary_objectives
        }
    
    def __repr__(self) -> str:
        return f"<OptimizationStudy(study_id='{self.study_id}', symbol='{self.symbol}', strategy='{self.strategy_template}', status='{self.status}')>"

class OptimizationTrialModel(Base):
    """
    Individual optimization trial results.
    
    Stores detailed information about each parameter combination tested
    during optimization, including parameters, metrics, and execution details.
    """
    __tablename__ = "optimization_trials"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    trial_id = Column(Integer, nullable=False)  # Trial number within study
    study_id = Column(String(255), ForeignKey('optimization_studies.study_id'), nullable=False, index=True)
    
    # Trial configuration
    params = Column(JSON, nullable=False)  # Parameter values tested
    
    # Trial results
    value = Column(Float, nullable=True)  # Primary objective value
    values = Column(JSON, nullable=True)  # Multi-objective values
    
    # Detailed metrics from backtesting
    metrics = Column(JSON, nullable=True)  # Full backtest metrics
    validation_metrics = Column(JSON, nullable=True)  # Out-of-sample validation
    
    # Trial status and execution info
    state = Column(SQLEnum(TrialState), default=TrialState.RUNNING, index=True)
    datetime_start = Column(DateTime, nullable=True)
    datetime_complete = Column(DateTime, nullable=True)
    
    # Intermediate values for pruning
    intermediate_values = Column(JSON, nullable=True)  # List of (step, value) tuples
    
    # Metadata
    user_attrs = Column(JSON, nullable=True)
    system_attrs = Column(JSON, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    
    # Relationship
    study = relationship("OptimizationStudyModel", back_populates="trials")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_trial_study_number', 'study_id', 'trial_id'),
        Index('idx_trial_state_value', 'state', 'value'),
        Index('idx_trial_completion', 'datetime_complete'),
        UniqueConstraint('study_id', 'trial_id', name='unique_study_trial'),
    )
    
    @hybrid_property
    def duration_seconds(self) -> Optional[float]:
        """Trial duration in seconds"""
        if self.datetime_start and self.datetime_complete:
            return (self.datetime_complete - self.datetime_start).total_seconds()
        return None
    
    @hybrid_property
    def is_successful(self) -> bool:
        """Whether trial completed successfully"""
        return self.state == TrialState.COMPLETE and self.value is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to dictionary representation"""
        return {
            'trial_id': self.trial_id,
            'study_id': self.study_id,
            'params': self.params,
            'value': self.value,
            'values': self.values,
            'state': self.state.value if self.state else None,
            'datetime_start': self.datetime_start.isoformat() if self.datetime_start else None,
            'datetime_complete': self.datetime_complete.isoformat() if self.datetime_complete else None,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics,
            'validation_metrics': self.validation_metrics,
            'intermediate_values': self.intermediate_values,
            'user_attrs': self.user_attrs,
            'system_attrs': self.system_attrs,
            'error_message': self.error_message,
            'is_successful': self.is_successful
        }
    
    def __repr__(self) -> str:
        return f"<OptimizationTrial(study_id='{self.study_id}', trial_id={self.trial_id}, state='{self.state}', value={self.value})>"

class WatchlistOptimizationModel(Base):
    """
    Batch optimization requests for multiple symbols/strategies.
    
    Tracks portfolio-level optimization jobs that span multiple symbols
    and strategy templates with coordinated execution.
    """
    __tablename__ = "watchlist_optimizations"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    optimization_id = Column(String(255), unique=True, index=True, nullable=False)
    profile_id = Column(Integer, nullable=False, index=True)
    
    # Configuration
    symbols = Column(JSON, nullable=False)  # List of symbols
    strategy_templates = Column(JSON, nullable=False)  # List of strategy templates
    
    # Optimization settings
    n_trials_per_symbol = Column(Integer, default=50)
    parallel_jobs = Column(Integer, default=4)
    data_timeframe = Column(String(20), default="1h")
    lookback_days = Column(Integer, default=365)
    
    # Risk management
    portfolio_risk_limit = Column(Float, default=0.20)
    correlation_threshold = Column(Float, default=0.70)
    
    # Status and progress
    status = Column(String(50), index=True, default="queued")
    progress = Column(Float, default=0.0)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    scheduled_for = Column(DateTime, nullable=True, index=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Resource allocation
    priority = Column(SQLEnum(OptimizationPriority), default=OptimizationPriority.NORMAL)
    parallel_jobs_used = Column(Integer, default=1)
    
    # Notification settings
    notify_on_completion = Column(Boolean, default=True)
    email_results = Column(Boolean, default=False)
    notification_sent = Column(Boolean, default=False)
    
    # Results summary (JSON structure with study_id -> result mapping)
    study_results = Column(JSON, nullable=True)
    aggregate_results = Column(JSON, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    failed_studies = Column(JSON, nullable=True)  # List of failed study IDs
    
    # Indexes
    __table_args__ = (
        Index('idx_watchlist_profile_status', 'profile_id', 'status'),
        Index('idx_watchlist_scheduled', 'scheduled_for'),
        Index('idx_watchlist_priority', 'priority', 'created_at'),
        CheckConstraint('progress >= 0.0 AND progress <= 1.0', name='watchlist_progress_range_check'),
    )
    
    @hybrid_property
    def total_combinations(self) -> int:
        """Total number of symbol-strategy combinations"""
        symbols_count = len(self.symbols) if self.symbols else 0
        strategies_count = len(self.strategy_templates) if self.strategy_templates else 0
        return symbols_count * strategies_count
    
    @hybrid_property
    def estimated_total_trials(self) -> int:
        """Estimated total number of trials across all combinations"""
        return self.total_combinations * self.n_trials_per_symbol
    
    @hybrid_property
    def duration_seconds(self) -> Optional[float]:
        """Total optimization duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'optimization_id': self.optimization_id,
            'profile_id': self.profile_id,
            'symbols': self.symbols,
            'strategy_templates': self.strategy_templates,
            'status': self.status,
            'progress': self.progress,
            'n_trials_per_symbol': self.n_trials_per_symbol,
            'parallel_jobs': self.parallel_jobs,
            'total_combinations': self.total_combinations,
            'estimated_total_trials': self.estimated_total_trials,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'scheduled_for': self.scheduled_for.isoformat() if self.scheduled_for else None,
            'duration_seconds': self.duration_seconds,
            'priority': self.priority.value if self.priority else None,
            'study_results': self.study_results,
            'aggregate_results': self.aggregate_results
        }
    
    def __repr__(self) -> str:
        return f"<WatchlistOptimization(optimization_id='{self.optimization_id}', profile_id={self.profile_id}, status='{self.status}')>"

class OptimizationParameterImportanceModel(Base):
    """
    Parameter importance scores from completed studies.
    
    Stores computed importance scores for parameters across different
    strategies and symbols for analysis and insights.
    """
    __tablename__ = "optimization_parameter_importance"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(String(255), ForeignKey('optimization_studies.study_id'), nullable=False, index=True)
    
    # Parameter information
    parameter_name = Column(String(100), nullable=False, index=True)
    importance_score = Column(Float, nullable=False)
    
    # Analysis context
    analysis_method = Column(String(50), default="fanova")  # fanova, permutation, etc.
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    confidence_interval = Column(JSON, nullable=True)  # [lower, upper] bounds
    rank = Column(Integer, nullable=True)  # Rank among all parameters in study
    
    # Relationships
    study = relationship("OptimizationStudyModel")
    
    __table_args__ = (
        Index('idx_param_importance_study_rank', 'study_id', 'rank'),
        Index('idx_param_importance_name_score', 'parameter_name', 'importance_score'),
        UniqueConstraint('study_id', 'parameter_name', 'analysis_method', name='unique_param_importance'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'study_id': self.study_id,
            'parameter_name': self.parameter_name,
            'importance_score': self.importance_score,
            'analysis_method': self.analysis_method,
            'rank': self.rank,
            'confidence_interval': self.confidence_interval,
            'computed_at': self.computed_at.isoformat() if self.computed_at else None
        }

class OptimizationComparisonModel(Base):
    """
    Comparisons between different optimization studies.
    
    Stores structured comparisons of parameter values, performance metrics,
    and statistical significance tests between studies.
    """
    __tablename__ = "optimization_comparisons"
    
    id = Column(Integer, primary_key=True, index=True)
    comparison_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Studies being compared
    study_ids = Column(JSON, nullable=False)  # List of study IDs
    
    # Comparison results
    parameter_comparison = Column(JSON, nullable=False)
    metric_comparison = Column(JSON, nullable=False)
    
    # Statistical analysis
    significance_tests = Column(JSON, nullable=True)
    correlation_analysis = Column(JSON, nullable=True)
    
    # Conclusions
    winner_study_id = Column(String(255), nullable=True, index=True)
    recommendations = Column(JSON, nullable=True)  # List of recommendation strings
    
    # Metadata
    comparison_criteria = Column(JSON, nullable=False)
    compared_at = Column(DateTime, default=datetime.utcnow)
    compared_by = Column(String(100), nullable=True)  # User or system identifier
    
    __table_args__ = (
        Index('idx_comparison_winner', 'winner_study_id'),
        Index('idx_comparison_date', 'compared_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'comparison_id': self.comparison_id,
            'study_ids': self.study_ids,
            'parameter_comparison': self.parameter_comparison,
            'metric_comparison': self.metric_comparison,
            'significance_tests': self.significance_tests,
            'correlation_analysis': self.correlation_analysis,
            'winner_study_id': self.winner_study_id,
            'recommendations': self.recommendations,
            'comparison_criteria': self.comparison_criteria,
            'compared_at': self.compared_at.isoformat() if self.compared_at else None,
            'compared_by': self.compared_by
        }

class OptimizationAlertModel(Base):
    """
    Alerts and notifications for optimization events.
    
    Tracks alerts triggered by optimization completion, significant findings,
    or system events.
    """
    __tablename__ = "optimization_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Alert type and context
    alert_type = Column(String(50), nullable=False, index=True)  # completion, anomaly, threshold, etc.
    severity = Column(String(20), default="info", index=True)  # info, warning, error, critical
    
    # Associated entities
    study_id = Column(String(255), nullable=True, index=True)
    optimization_id = Column(String(255), nullable=True, index=True)  # For watchlist optimizations
    profile_id = Column(Integer, nullable=True, index=True)
    
    # Alert content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)  # Additional structured data
    
    # Status and timing
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Delivery tracking
    notification_sent = Column(Boolean, default=False)
    email_sent = Column(Boolean, default=False)
    delivery_attempts = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_alert_profile_triggered', 'profile_id', 'triggered_at'),
        Index('idx_alert_unacknowledged', 'acknowledged_at'),  # NULL values for unacknowledged
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'study_id': self.study_id,
            'optimization_id': self.optimization_id,
            'profile_id': self.profile_id,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'notification_sent': self.notification_sent,
            'email_sent': self.email_sent
        }

# Utility functions for model operations

def create_optimization_study(
    study_id: str,
    study_name: str,
    symbol: str,
    strategy_template: str,
    configuration: Dict[str, Any],
    session
) -> OptimizationStudyModel:
    """Create a new optimization study record"""
    study = OptimizationStudyModel(
        study_id=study_id,
        study_name=study_name,
        symbol=symbol,
        strategy_template=strategy_template,
        configuration=configuration,
        n_trials=configuration.get('n_trials', 100),
        primary_objective=configuration.get('primary_objective', 'sharpe_ratio'),
        secondary_objectives=configuration.get('secondary_objectives', []),
        max_drawdown_limit=configuration.get('max_drawdown_limit', 0.25),
        min_sharpe_ratio=configuration.get('min_sharpe_ratio', 0.5),
        min_win_rate=configuration.get('min_win_rate', 0.35),
        include_transaction_costs=configuration.get('include_transaction_costs', True),
        commission=configuration.get('commission', 0.001),
        slippage=configuration.get('slippage', 0.001),
        walk_forward=configuration.get('walk_forward', False),
        validation_splits=configuration.get('validation_splits', 3)
    )
    
    session.add(study)
    session.commit()
    session.refresh(study)
    return study

def update_study_progress(
    study_id: str,
    progress: float,
    best_value: Optional[float] = None,
    best_params: Optional[Dict[str, Any]] = None,
    best_trial_number: Optional[int] = None,
    session=None
) -> bool:
    """Update study progress and best results"""
    if session is None:
        return False
    
    study = session.query(OptimizationStudyModel).filter(
        OptimizationStudyModel.study_id == study_id
    ).first()
    
    if not study:
        return False
    
    study.progress = max(0.0, min(1.0, progress))
    
    if best_value is not None:
        study.best_value = best_value
    if best_params is not None:
        study.best_params = best_params
    if best_trial_number is not None:
        study.best_trial_number = best_trial_number
    
    # Update status based on progress
    if progress >= 1.0:
        study.status = StudyStatus.COMPLETED
        study.completed_at = datetime.utcnow()
    elif progress > 0:
        study.status = StudyStatus.RUNNING
        if not study.started_at:
            study.started_at = datetime.utcnow()
    
    session.commit()
    return True

def record_optimization_trial(
    study_id: str,
    trial_id: int,
    params: Dict[str, Any],
    value: Optional[float] = None,
    values: Optional[List[float]] = None,
    state: TrialState = TrialState.COMPLETE,
    metrics: Optional[Dict[str, float]] = None,
    session=None
) -> Optional[OptimizationTrialModel]:
    """Record a completed optimization trial"""
    if session is None:
        return None
    
    trial = OptimizationTrialModel(
        study_id=study_id,
        trial_id=trial_id,
        params=params,
        value=value,
        values=values,
        state=state,
        metrics=metrics,
        datetime_complete=datetime.utcnow() if state in [TrialState.COMPLETE, TrialState.FAILED, TrialState.PRUNED] else None
    )
    
    session.add(trial)
    session.commit()
    session.refresh(trial)
    
    # Update study trial counts
    study = session.query(OptimizationStudyModel).filter(
        OptimizationStudyModel.study_id == study_id
    ).first()
    
    if study:
        if state == TrialState.COMPLETE:
            study.n_complete_trials += 1
        elif state == TrialState.PRUNED:
            study.n_pruned_trials += 1
        elif state == TrialState.FAILED:
            study.n_failed_trials += 1
        
        session.commit()
    
    return trial

def get_study_statistics(study_id: str, session) -> Optional[Dict[str, Any]]:
    """Get comprehensive statistics for a study"""
    study = session.query(OptimizationStudyModel).filter(
        OptimizationStudyModel.study_id == study_id
    ).first()
    
    if not study:
        return None
    
    # Get trial statistics
    trials = session.query(OptimizationTrialModel).filter(
        OptimizationTrialModel.study_id == study_id
    ).all()
    
    successful_trials = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
    
    stats = study.to_dict()
    
    if successful_trials:
        values = [t.value for t in successful_trials]
        stats.update({
            'value_statistics': {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0
            },
            'best_trials': [
                {
                    'trial_id': t.trial_id,
                    'value': t.value,
                    'params': t.params
                }
                for t in sorted(successful_trials, key=lambda x: x.value, reverse=True)[:5]
            ]
        })
    
    return stats

# Export all models and utility functions
__all__ = [
    "OptimizationStudyModel",
    "OptimizationTrialModel", 
    "WatchlistOptimizationModel",
    "OptimizationParameterImportanceModel",
    "OptimizationComparisonModel",
    "OptimizationAlertModel",
    "StudyStatus",
    "TrialState",
    "OptimizationPriority",
    "create_optimization_study",
    "update_study_progress",
    "record_optimization_trial",
    "get_study_statistics"
]