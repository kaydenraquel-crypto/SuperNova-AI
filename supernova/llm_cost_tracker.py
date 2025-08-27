"""
LLM Cost Tracking and Usage Monitoring System

This module provides:
- Real-time cost tracking per provider and user
- Usage quotas and spending limits
- Cost alerts and notifications
- Usage analytics and reporting
- Token usage optimization recommendations
- Budget management and forecasting
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, date
import asyncio
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
import statistics

# Core imports
from .config import settings
from .llm_key_manager import ProviderType
from .db import SessionLocal
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, Float, Date, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class AlertType(str, Enum):
    """Alert types for cost monitoring"""
    DAILY_LIMIT = "daily_limit"
    MONTHLY_LIMIT = "monthly_limit"
    HOURLY_SPIKE = "hourly_spike"
    PROVIDER_LIMIT = "provider_limit"
    USER_LIMIT = "user_limit"
    BUDGET_THRESHOLD = "budget_threshold"

class UsagePeriod(str, Enum):
    """Usage tracking periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    requests: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int, input_rate: float, output_rate: float):
        """Add usage to breakdown"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.input_cost += (input_tokens / 1000) * input_rate
        self.output_cost += (output_tokens / 1000) * output_rate
        self.total_cost = self.input_cost + self.output_cost
        self.requests += 1

class UsageRecord(Base):
    """Usage record for individual requests"""
    __tablename__ = "llm_usage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Token usage
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    
    # Cost tracking
    input_cost = Column(Float, nullable=False, default=0.0)
    output_cost = Column(Float, nullable=False, default=0.0)
    total_cost = Column(Float, nullable=False, default=0.0)
    
    # Request metadata
    request_type = Column(String(50), nullable=True)  # chat, analysis, summary, etc.
    response_time = Column(Float, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    date = Column(Date, default=date.today, index=True)
    hour = Column(Integer, default=lambda: datetime.utcnow().hour, index=True)
    
    # Additional metadata
    metadata = Column(Text, nullable=True)  # JSON

    __table_args__ = (
        Index('idx_provider_date', 'provider', 'date'),
        Index('idx_user_date', 'user_id', 'date'),
        Index('idx_timestamp_success', 'timestamp', 'success'),
    )

class CostAlert(Base):
    """Cost alert records"""
    __tablename__ = "llm_cost_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    provider = Column(String(50), nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    
    threshold_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    threshold_percentage = Column(Float, nullable=True)
    
    message = Column(Text, nullable=False)
    sent = Column(Boolean, nullable=False, default=False)
    acknowledged = Column(Boolean, nullable=False, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    sent_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    metadata = Column(Text, nullable=True)

class UsageSummary(Base):
    """Pre-aggregated usage summaries for fast reporting"""
    __tablename__ = "llm_usage_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    period_type = Column(String(20), nullable=False, index=True)  # hourly, daily, monthly
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Aggregated metrics
    total_requests = Column(Integer, nullable=False, default=0)
    successful_requests = Column(Integer, nullable=False, default=0)
    failed_requests = Column(Integer, nullable=False, default=0)
    
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    
    total_input_cost = Column(Float, nullable=False, default=0.0)
    total_output_cost = Column(Float, nullable=False, default=0.0)
    total_cost = Column(Float, nullable=False, default=0.0)
    
    avg_response_time = Column(Float, nullable=True)
    unique_users = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_provider_period', 'provider', 'period_type', 'date'),
    )

class LLMCostTracker:
    """
    Comprehensive LLM cost tracking and usage monitoring system
    """
    
    def __init__(self):
        """Initialize cost tracker"""
        self.session_factory = sessionmaker()
        self.engine = None
        
        # Cost tracking settings
        self.enabled = getattr(settings, 'COST_TRACKING_ENABLED', True)
        self.daily_limit = getattr(settings, 'LLM_DAILY_COST_LIMIT', 50.0)
        self.monthly_limit = getattr(settings, 'LLM_MONTHLY_COST_LIMIT', 1000.0)
        self.alert_threshold = getattr(settings, 'LLM_COST_ALERT_THRESHOLD', 0.8)
        
        # Provider-specific limits
        self.provider_limits = {
            ProviderType.OPENAI: getattr(settings, 'OPENAI_DAILY_COST_LIMIT', 30.0),
            ProviderType.ANTHROPIC: getattr(settings, 'ANTHROPIC_DAILY_COST_LIMIT', 20.0),
            ProviderType.HUGGINGFACE: 0.0,  # Free tier
            ProviderType.OLLAMA: 0.0  # Local models
        }
        
        # Current usage cache
        self.current_usage: Dict[str, CostBreakdown] = {}
        self.daily_usage: Dict[date, CostBreakdown] = {}
        self.hourly_usage: Dict[datetime, CostBreakdown] = {}
        
        # Alert tracking
        self.sent_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(hours=1)
        
        self._initialize_database()
        self._start_background_tasks()
        
        logger.info("LLM Cost Tracker initialized")
    
    def _initialize_database(self):
        """Initialize database connection and tables"""
        try:
            database_url = settings.DATABASE_URL
            if database_url.startswith("sqlite"):
                self.engine = create_engine(database_url, echo=False)
            else:
                self.engine = create_engine(database_url, pool_size=5, max_overflow=10)
            
            Base.metadata.create_all(bind=self.engine)
            self.session_factory.configure(bind=self.engine)
            
            logger.info("Cost tracking database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cost tracking database: {e}")
            raise
    
    def _get_session(self) -> Session:
        """Get database session"""
        return self.session_factory()
    
    def _start_background_tasks(self):
        """Start background aggregation and alert tasks"""
        if not self.enabled:
            return
        
        async def background_loop():
            while True:
                try:
                    await self._aggregate_usage_summaries()
                    await self._check_cost_alerts()
                    await self._cleanup_old_records()
                    await asyncio.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Background cost tracking task error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(background_loop())
        logger.info("Cost tracking background tasks started")
    
    async def record_usage(self,
                          provider: ProviderType,
                          model: str,
                          input_tokens: int,
                          output_tokens: int,
                          input_cost_per_1k: float,
                          output_cost_per_1k: float,
                          user_id: Optional[int] = None,
                          session_id: Optional[str] = None,
                          request_type: str = 'chat',
                          response_time: Optional[float] = None,
                          success: bool = True,
                          error_message: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record LLM usage with cost calculation
        
        Returns usage record with cost breakdown
        """
        if not self.enabled:
            return {"cost": 0.0, "tracked": False}
        
        try:
            # Calculate costs
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            total_tokens = input_tokens + output_tokens
            
            # Create usage record
            now = datetime.utcnow()
            usage_record = UsageRecord(
                provider=provider.value,
                model=model,
                user_id=user_id,
                session_id=session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                request_type=request_type,
                response_time=response_time,
                success=success,
                error_message=error_message,
                timestamp=now,
                date=now.date(),
                hour=now.hour,
                metadata=json.dumps(metadata) if metadata else None
            )
            
            # Store in database
            with self._get_session() as session:
                session.add(usage_record)
                session.commit()
                record_id = usage_record.id
            
            # Update cache
            self._update_usage_cache(provider, input_tokens, output_tokens, input_cost_per_1k, output_cost_per_1k)
            
            # Check if limits exceeded
            await self._check_usage_limits(provider, total_cost, user_id)
            
            logger.debug(f"Recorded usage: {provider.value} - ${total_cost:.4f}")
            
            return {
                "record_id": record_id,
                "cost": total_cost,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "tokens": total_tokens,
                "tracked": True
            }
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
            return {"cost": 0.0, "tracked": False, "error": str(e)}
    
    def _update_usage_cache(self, provider: ProviderType, input_tokens: int, output_tokens: int, 
                           input_rate: float, output_rate: float):
        """Update in-memory usage cache"""
        
        # Update provider cache
        if provider.value not in self.current_usage:
            self.current_usage[provider.value] = CostBreakdown()
        self.current_usage[provider.value].add_usage(input_tokens, output_tokens, input_rate, output_rate)
        
        # Update daily cache
        today = date.today()
        if today not in self.daily_usage:
            self.daily_usage[today] = CostBreakdown()
        self.daily_usage[today].add_usage(input_tokens, output_tokens, input_rate, output_rate)
        
        # Update hourly cache
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        if current_hour not in self.hourly_usage:
            self.hourly_usage[current_hour] = CostBreakdown()
        self.hourly_usage[current_hour].add_usage(input_tokens, output_tokens, input_rate, output_rate)
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Clean old cache entries"""
        # Keep only last 7 days of daily usage
        cutoff_date = date.today() - timedelta(days=7)
        self.daily_usage = {d: usage for d, usage in self.daily_usage.items() if d >= cutoff_date}
        
        # Keep only last 24 hours of hourly usage
        cutoff_hour = datetime.utcnow() - timedelta(hours=24)
        self.hourly_usage = {h: usage for h, usage in self.hourly_usage.items() if h >= cutoff_hour}
    
    async def _check_usage_limits(self, provider: ProviderType, cost: float, user_id: Optional[int] = None):
        """Check if usage limits are exceeded"""
        
        # Check daily provider limit
        provider_daily = await self.get_daily_cost(provider=provider)
        provider_limit = self.provider_limits.get(provider, 0.0)
        
        if provider_limit > 0 and provider_daily >= provider_limit:
            await self._create_alert(
                AlertType.PROVIDER_LIMIT,
                f"Daily cost limit exceeded for {provider.value}",
                provider_limit,
                provider_daily,
                provider=provider.value
            )
        
        # Check overall daily limit
        daily_total = await self.get_daily_cost()
        if daily_total >= self.daily_limit:
            await self._create_alert(
                AlertType.DAILY_LIMIT,
                f"Daily cost limit exceeded: ${daily_total:.2f}",
                self.daily_limit,
                daily_total
            )
        
        # Check monthly limit
        monthly_total = await self.get_monthly_cost()
        if monthly_total >= self.monthly_limit:
            await self._create_alert(
                AlertType.MONTHLY_LIMIT,
                f"Monthly cost limit exceeded: ${monthly_total:.2f}",
                self.monthly_limit,
                monthly_total
            )
        
        # Check threshold alerts (80% of limits)
        if daily_total >= self.daily_limit * self.alert_threshold:
            await self._create_alert(
                AlertType.BUDGET_THRESHOLD,
                f"Daily cost approaching limit: ${daily_total:.2f} / ${self.daily_limit:.2f}",
                self.daily_limit * self.alert_threshold,
                daily_total,
                threshold_percentage=self.alert_threshold * 100
            )
    
    async def _create_alert(self,
                           alert_type: AlertType,
                           message: str,
                           threshold: float,
                           current_value: float,
                           provider: Optional[str] = None,
                           user_id: Optional[int] = None,
                           threshold_percentage: Optional[float] = None):
        """Create cost alert"""
        
        # Check alert cooldown
        alert_key = f"{alert_type.value}:{provider or 'global'}:{user_id or 0}"
        if alert_key in self.sent_alerts:
            last_sent = self.sent_alerts[alert_key]
            if datetime.utcnow() - last_sent < self.alert_cooldown:
                return  # Skip alert due to cooldown
        
        try:
            with self._get_session() as session:
                alert = CostAlert(
                    alert_type=alert_type.value,
                    provider=provider,
                    user_id=user_id,
                    threshold_value=threshold,
                    current_value=current_value,
                    threshold_percentage=threshold_percentage,
                    message=message
                )
                
                session.add(alert)
                session.commit()
                
                # Update sent alerts cache
                self.sent_alerts[alert_key] = datetime.utcnow()
                
                logger.warning(f"Cost alert created: {message}")
                
                # Send alert (integrate with notification system)
                await self._send_alert_notification(alert)
                
        except Exception as e:
            logger.error(f"Failed to create cost alert: {e}")
    
    async def _send_alert_notification(self, alert: CostAlert):
        """Send alert notification (implement based on your notification system)"""
        # This is where you'd integrate with email, Slack, webhook, etc.
        logger.info(f"COST ALERT: {alert.message}")
        
        # Mark alert as sent
        try:
            with self._get_session() as session:
                session.query(CostAlert).filter(CostAlert.id == alert.id).update({
                    "sent": True,
                    "sent_at": datetime.utcnow()
                })
                session.commit()
        except Exception as e:
            logger.error(f"Failed to mark alert as sent: {e}")
    
    async def get_daily_cost(self, 
                           provider: Optional[ProviderType] = None,
                           user_id: Optional[int] = None,
                           target_date: Optional[date] = None) -> float:
        """Get daily cost total"""
        target_date = target_date or date.today()
        
        try:
            with self._get_session() as session:
                query = session.query(func.sum(UsageRecord.total_cost)).filter(
                    UsageRecord.date == target_date,
                    UsageRecord.success == True
                )
                
                if provider:
                    query = query.filter(UsageRecord.provider == provider.value)
                
                if user_id:
                    query = query.filter(UsageRecord.user_id == user_id)
                
                result = query.scalar()
                return float(result or 0.0)
                
        except Exception as e:
            logger.error(f"Failed to get daily cost: {e}")
            return 0.0
    
    async def get_monthly_cost(self,
                              provider: Optional[ProviderType] = None,
                              user_id: Optional[int] = None,
                              year: Optional[int] = None,
                              month: Optional[int] = None) -> float:
        """Get monthly cost total"""
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month
        
        try:
            with self._get_session() as session:
                query = session.query(func.sum(UsageRecord.total_cost)).filter(
                    func.extract('year', UsageRecord.date) == year,
                    func.extract('month', UsageRecord.date) == month,
                    UsageRecord.success == True
                )
                
                if provider:
                    query = query.filter(UsageRecord.provider == provider.value)
                
                if user_id:
                    query = query.filter(UsageRecord.user_id == user_id)
                
                result = query.scalar()
                return float(result or 0.0)
                
        except Exception as e:
            logger.error(f"Failed to get monthly cost: {e}")
            return 0.0
    
    async def get_usage_analytics(self,
                                 period: UsagePeriod = UsagePeriod.DAILY,
                                 days_back: int = 30,
                                 provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Get comprehensive usage analytics"""
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            with self._get_session() as session:
                base_query = session.query(UsageRecord).filter(
                    UsageRecord.date >= start_date,
                    UsageRecord.date <= end_date
                )
                
                if provider:
                    base_query = base_query.filter(UsageRecord.provider == provider.value)
                
                records = base_query.all()
                
                if not records:
                    return {"total_cost": 0.0, "total_requests": 0, "providers": {}}
                
                # Aggregate data
                analytics = {
                    "period": period.value,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_cost": 0.0,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_tokens": 0,
                    "providers": {},
                    "daily_breakdown": {},
                    "cost_trends": [],
                    "top_users": [],
                    "model_usage": {}
                }
                
                # Process records
                daily_costs = {}
                provider_stats = {}
                user_costs = {}
                model_usage = {}
                
                for record in records:
                    # Overall stats
                    analytics["total_requests"] += 1
                    if record.success:
                        analytics["successful_requests"] += 1
                        analytics["total_cost"] += record.total_cost
                        analytics["total_tokens"] += record.total_tokens
                    else:
                        analytics["failed_requests"] += 1
                    
                    # Daily breakdown
                    day_str = record.date.isoformat()
                    if day_str not in daily_costs:
                        daily_costs[day_str] = 0.0
                    if record.success:
                        daily_costs[day_str] += record.total_cost
                    
                    # Provider stats
                    if record.provider not in provider_stats:
                        provider_stats[record.provider] = {
                            "cost": 0.0, "requests": 0, "tokens": 0
                        }
                    provider_stats[record.provider]["requests"] += 1
                    if record.success:
                        provider_stats[record.provider]["cost"] += record.total_cost
                        provider_stats[record.provider]["tokens"] += record.total_tokens
                    
                    # User stats
                    if record.user_id:
                        if record.user_id not in user_costs:
                            user_costs[record.user_id] = 0.0
                        if record.success:
                            user_costs[record.user_id] += record.total_cost
                    
                    # Model usage
                    if record.model not in model_usage:
                        model_usage[record.model] = {"cost": 0.0, "requests": 0}
                    model_usage[record.model]["requests"] += 1
                    if record.success:
                        model_usage[record.model]["cost"] += record.total_cost
                
                # Add aggregated data to analytics
                analytics["daily_breakdown"] = daily_costs
                analytics["providers"] = provider_stats
                analytics["model_usage"] = model_usage
                
                # Cost trends (simple moving average)
                if len(daily_costs) > 7:
                    daily_values = list(daily_costs.values())
                    for i in range(6, len(daily_values)):
                        avg = statistics.mean(daily_values[i-6:i+1])
                        analytics["cost_trends"].append({
                            "date": list(daily_costs.keys())[i],
                            "cost": daily_values[i],
                            "trend": avg
                        })
                
                # Top users by cost
                analytics["top_users"] = [
                    {"user_id": uid, "cost": cost}
                    for uid, cost in sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get usage analytics: {e}")
            return {"error": str(e)}
    
    async def get_cost_forecast(self, days_ahead: int = 30) -> Dict[str, Any]:
        """Generate cost forecast based on historical usage"""
        
        try:
            # Get last 30 days of usage
            analytics = await self.get_usage_analytics(days_back=30)
            
            if not analytics.get("daily_breakdown"):
                return {"forecast": 0.0, "confidence": 0.0}
            
            daily_costs = list(analytics["daily_breakdown"].values())
            
            # Simple forecast using moving average
            if len(daily_costs) >= 7:
                recent_avg = statistics.mean(daily_costs[-7:])
                overall_avg = statistics.mean(daily_costs)
                trend_factor = recent_avg / overall_avg if overall_avg > 0 else 1.0
                
                # Calculate variance for confidence
                variance = statistics.variance(daily_costs) if len(daily_costs) > 1 else 0
                confidence = max(0.1, 1.0 - (variance / (overall_avg ** 2)) if overall_avg > 0 else 0.5)
                
                forecast = {
                    "daily_average": recent_avg,
                    "weekly_forecast": recent_avg * 7,
                    "monthly_forecast": recent_avg * 30,
                    "forecast_period_days": days_ahead,
                    "period_forecast": recent_avg * days_ahead,
                    "trend_factor": trend_factor,
                    "confidence_score": min(1.0, confidence),
                    "recommendations": []
                }
                
                # Add recommendations
                if trend_factor > 1.2:
                    forecast["recommendations"].append("Usage trending upward - consider reviewing cost optimization")
                elif trend_factor < 0.8:
                    forecast["recommendations"].append("Usage trending downward - good cost control")
                
                if recent_avg > self.daily_limit * 0.8:
                    forecast["recommendations"].append("Approaching daily cost limit - monitor usage closely")
                
                return forecast
            else:
                return {"forecast": 0.0, "confidence": 0.0, "error": "Insufficient data for forecast"}
                
        except Exception as e:
            logger.error(f"Failed to generate cost forecast: {e}")
            return {"error": str(e)}
    
    async def _aggregate_usage_summaries(self):
        """Create pre-aggregated usage summaries for fast reporting"""
        try:
            now = datetime.utcnow()
            
            # Aggregate hourly summaries
            await self._create_hourly_summaries(now)
            
            # Aggregate daily summaries (run once per day)
            if now.hour == 0:  # Run at midnight
                await self._create_daily_summaries(now.date() - timedelta(days=1))
            
        except Exception as e:
            logger.error(f"Failed to aggregate usage summaries: {e}")
    
    async def _create_hourly_summaries(self, target_time: datetime):
        """Create hourly usage summaries"""
        target_hour = target_time.replace(minute=0, second=0, microsecond=0)
        
        try:
            with self._get_session() as session:
                # Get hourly data for each provider
                for provider_type in ProviderType:
                    records = session.query(UsageRecord).filter(
                        UsageRecord.provider == provider_type.value,
                        UsageRecord.timestamp >= target_hour,
                        UsageRecord.timestamp < target_hour + timedelta(hours=1)
                    ).all()
                    
                    if not records:
                        continue
                    
                    # Calculate aggregates
                    total_requests = len(records)
                    successful_requests = len([r for r in records if r.success])
                    failed_requests = total_requests - successful_requests
                    
                    successful_records = [r for r in records if r.success]
                    if successful_records:
                        total_input_tokens = sum(r.input_tokens for r in successful_records)
                        total_output_tokens = sum(r.output_tokens for r in successful_records)
                        total_cost = sum(r.total_cost for r in successful_records)
                        avg_response_time = statistics.mean([r.response_time for r in successful_records if r.response_time])
                        unique_users = len(set(r.user_id for r in successful_records if r.user_id))
                    else:
                        total_input_tokens = 0
                        total_output_tokens = 0
                        total_cost = 0.0
                        avg_response_time = None
                        unique_users = 0
                    
                    # Check if summary already exists
                    existing = session.query(UsageSummary).filter(
                        UsageSummary.provider == provider_type.value,
                        UsageSummary.period_type == UsagePeriod.HOURLY.value,
                        UsageSummary.period_start == target_hour
                    ).first()
                    
                    if existing:
                        # Update existing summary
                        existing.total_requests = total_requests
                        existing.successful_requests = successful_requests
                        existing.failed_requests = failed_requests
                        existing.total_input_tokens = total_input_tokens
                        existing.total_output_tokens = total_output_tokens
                        existing.total_tokens = total_input_tokens + total_output_tokens
                        existing.total_cost = total_cost
                        existing.avg_response_time = avg_response_time
                        existing.unique_users = unique_users
                        existing.updated_at = datetime.utcnow()
                    else:
                        # Create new summary
                        summary = UsageSummary(
                            provider=provider_type.value,
                            period_type=UsagePeriod.HOURLY.value,
                            period_start=target_hour,
                            period_end=target_hour + timedelta(hours=1),
                            date=target_hour.date(),
                            total_requests=total_requests,
                            successful_requests=successful_requests,
                            failed_requests=failed_requests,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            total_tokens=total_input_tokens + total_output_tokens,
                            total_cost=total_cost,
                            avg_response_time=avg_response_time,
                            unique_users=unique_users
                        )
                        session.add(summary)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to create hourly summaries: {e}")
    
    async def _create_daily_summaries(self, target_date: date):
        """Create daily usage summaries"""
        try:
            with self._get_session() as session:
                for provider_type in ProviderType:
                    records = session.query(UsageRecord).filter(
                        UsageRecord.provider == provider_type.value,
                        UsageRecord.date == target_date
                    ).all()
                    
                    if not records:
                        continue
                    
                    # Calculate daily aggregates (similar to hourly)
                    total_requests = len(records)
                    successful_requests = len([r for r in records if r.success])
                    failed_requests = total_requests - successful_requests
                    
                    successful_records = [r for r in records if r.success]
                    if successful_records:
                        total_input_tokens = sum(r.input_tokens for r in successful_records)
                        total_output_tokens = sum(r.output_tokens for r in successful_records)
                        total_cost = sum(r.total_cost for r in successful_records)
                        avg_response_time = statistics.mean([r.response_time for r in successful_records if r.response_time])
                        unique_users = len(set(r.user_id for r in successful_records if r.user_id))
                    else:
                        total_input_tokens = 0
                        total_output_tokens = 0
                        total_cost = 0.0
                        avg_response_time = None
                        unique_users = 0
                    
                    # Create/update daily summary
                    period_start = datetime.combine(target_date, datetime.min.time())
                    period_end = period_start + timedelta(days=1)
                    
                    existing = session.query(UsageSummary).filter(
                        UsageSummary.provider == provider_type.value,
                        UsageSummary.period_type == UsagePeriod.DAILY.value,
                        UsageSummary.date == target_date
                    ).first()
                    
                    if existing:
                        existing.total_requests = total_requests
                        existing.successful_requests = successful_requests
                        existing.failed_requests = failed_requests
                        existing.total_input_tokens = total_input_tokens
                        existing.total_output_tokens = total_output_tokens
                        existing.total_tokens = total_input_tokens + total_output_tokens
                        existing.total_cost = total_cost
                        existing.avg_response_time = avg_response_time
                        existing.unique_users = unique_users
                        existing.updated_at = datetime.utcnow()
                    else:
                        summary = UsageSummary(
                            provider=provider_type.value,
                            period_type=UsagePeriod.DAILY.value,
                            period_start=period_start,
                            period_end=period_end,
                            date=target_date,
                            total_requests=total_requests,
                            successful_requests=successful_requests,
                            failed_requests=failed_requests,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            total_tokens=total_input_tokens + total_output_tokens,
                            total_cost=total_cost,
                            avg_response_time=avg_response_time,
                            unique_users=unique_users
                        )
                        session.add(summary)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to create daily summaries: {e}")
    
    async def _check_cost_alerts(self):
        """Check and process pending cost alerts"""
        try:
            with self._get_session() as session:
                pending_alerts = session.query(CostAlert).filter(
                    CostAlert.sent == False
                ).all()
                
                for alert in pending_alerts:
                    await self._send_alert_notification(alert)
                    
        except Exception as e:
            logger.error(f"Failed to check cost alerts: {e}")
    
    async def _cleanup_old_records(self):
        """Cleanup old usage records to manage database size"""
        try:
            # Keep detailed records for 90 days, summaries for 2 years
            cutoff_date = date.today() - timedelta(days=90)
            summary_cutoff = date.today() - timedelta(days=730)
            
            with self._get_session() as session:
                # Delete old usage records
                deleted_records = session.query(UsageRecord).filter(
                    UsageRecord.date < cutoff_date
                ).delete()
                
                # Delete old hourly summaries (keep daily)
                deleted_summaries = session.query(UsageSummary).filter(
                    UsageSummary.period_type == UsagePeriod.HOURLY.value,
                    UsageSummary.date < cutoff_date
                ).delete()
                
                # Delete very old daily summaries
                deleted_old_summaries = session.query(UsageSummary).filter(
                    UsageSummary.date < summary_cutoff
                ).delete()
                
                session.commit()
                
                if deleted_records > 0 or deleted_summaries > 0:
                    logger.info(f"Cleaned up {deleted_records} old usage records and {deleted_summaries} old summaries")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
    
    def get_current_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary from cache"""
        return {
            "daily_total": self.daily_usage.get(date.today(), CostBreakdown()).__dict__,
            "by_provider": {provider: usage.__dict__ for provider, usage in self.current_usage.items()},
            "hourly_trend": [
                {"hour": h.hour, "cost": usage.total_cost}
                for h, usage in sorted(self.hourly_usage.items())
            ]
        }

# Global cost tracker instance
cost_tracker: Optional[LLMCostTracker] = None

def get_cost_tracker() -> LLMCostTracker:
    """Get global cost tracker instance"""
    global cost_tracker
    if cost_tracker is None:
        cost_tracker = LLMCostTracker()
    return cost_tracker

async def record_llm_usage(provider: ProviderType,
                          model: str,
                          input_tokens: int,
                          output_tokens: int,
                          input_cost_per_1k: float,
                          output_cost_per_1k: float,
                          **kwargs) -> Dict[str, Any]:
    """Convenient function to record LLM usage"""
    tracker = get_cost_tracker()
    return await tracker.record_usage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1k=input_cost_per_1k,
        output_cost_per_1k=output_cost_per_1k,
        **kwargs
    )