"""
SuperNova AI Monitoring Dashboard System
========================================

Comprehensive monitoring dashboard with:
- Real-time system health visualization
- Performance metrics dashboards
- Business metrics and KPI tracking
- Alert management interface
- Custom dashboard creation
- Mobile-responsive monitoring
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# FastAPI and web imports
from fastapi import HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse

# Internal imports
try:
    from .health_monitor import health_monitor, HealthStatus
    from .performance_monitor import performance_collector, dashboard as perf_dashboard
    from .monitoring_alerts import alert_manager
    from .business_metrics import business_metrics
    from .database_monitoring import db_monitor
    from .auth import get_current_user, User
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of monitoring dashboards."""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    BUSINESS_METRICS = "business_metrics"
    ALERTS = "alerts"
    INFRASTRUCTURE = "infrastructure"
    CUSTOM = "custom"

class ChartType(Enum):
    """Chart types for dashboard widgets."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    TABLE = "table"
    STAT = "stat"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"

class TimeRange(Enum):
    """Time ranges for dashboard data."""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    CUSTOM = "custom"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    chart_type: ChartType
    data_source: str  # Metric path or data source identifier
    position: Dict[str, int]  # x, y, width, height
    time_range: TimeRange = TimeRange.LAST_HOUR
    refresh_interval: int = 30  # seconds
    options: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "chart_type": self.chart_type.value,
            "data_source": self.data_source,
            "position": self.position,
            "time_range": self.time_range.value,
            "refresh_interval": self.refresh_interval,
            "options": self.options,
            "filters": self.filters
        }

@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    layout: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "description": self.description,
            "type": self.dashboard_type.value,
            "widgets": [widget.to_dict() for widget in self.widgets],
            "layout": self.layout,
            "permissions": self.permissions,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class MonitoringDashboardManager:
    """Main dashboard management system."""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.data_sources: Dict[str, callable] = {}
        
        # Setup default dashboards and data sources
        self._setup_data_sources()
        self._create_default_dashboards()
    
    def _setup_data_sources(self):
        """Setup data source functions for dashboard widgets."""
        self.data_sources = {
            "health.overall_score": self._get_health_score,
            "health.component_status": self._get_component_status,
            "performance.cpu_usage": self._get_cpu_usage,
            "performance.memory_usage": self._get_memory_usage,
            "performance.api_response_time": self._get_api_response_time,
            "performance.request_rate": self._get_request_rate,
            "alerts.active_count": self._get_active_alerts_count,
            "alerts.by_severity": self._get_alerts_by_severity,
            "business.daily_active_users": self._get_daily_active_users,
            "business.revenue": self._get_revenue_metrics,
            "business.kpi_summary": self._get_kpi_summary,
            "database.query_performance": self._get_database_performance,
            "database.connection_pool": self._get_database_connections,
            "uptime": self._get_uptime_metric
        }
    
    def _create_default_dashboards(self):
        """Create default monitoring dashboards."""
        # System Health Dashboard
        health_dashboard = Dashboard(
            dashboard_id="system_health",
            name="System Health",
            description="Overall system health and component status",
            dashboard_type=DashboardType.SYSTEM_HEALTH,
            widgets=[
                DashboardWidget(
                    widget_id="overall_health_score",
                    title="Overall Health Score",
                    chart_type=ChartType.GAUGE,
                    data_source="health.overall_score",
                    position={"x": 0, "y": 0, "width": 6, "height": 4},
                    options={"min": 0, "max": 100, "thresholds": [70, 90]}
                ),
                DashboardWidget(
                    widget_id="component_status",
                    title="Component Health Status", 
                    chart_type=ChartType.TABLE,
                    data_source="health.component_status",
                    position={"x": 6, "y": 0, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_id="system_uptime",
                    title="System Uptime",
                    chart_type=ChartType.STAT,
                    data_source="uptime",
                    position={"x": 0, "y": 4, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_id="active_alerts",
                    title="Active Alerts",
                    chart_type=ChartType.STAT,
                    data_source="alerts.active_count",
                    position={"x": 3, "y": 4, "width": 3, "height": 2},
                    options={"color_thresholds": {"warning": 1, "critical": 5}}
                )
            ]
        )
        
        # Performance Dashboard
        performance_dashboard = Dashboard(
            dashboard_id="performance",
            name="Performance Metrics",
            description="System performance and resource utilization",
            dashboard_type=DashboardType.PERFORMANCE,
            widgets=[
                DashboardWidget(
                    widget_id="cpu_usage_chart",
                    title="CPU Usage",
                    chart_type=ChartType.LINE_CHART,
                    data_source="performance.cpu_usage",
                    position={"x": 0, "y": 0, "width": 6, "height": 4},
                    time_range=TimeRange.LAST_HOUR,
                    options={"yAxis": {"min": 0, "max": 100}}
                ),
                DashboardWidget(
                    widget_id="memory_usage_chart",
                    title="Memory Usage",
                    chart_type=ChartType.LINE_CHART,
                    data_source="performance.memory_usage",
                    position={"x": 6, "y": 0, "width": 6, "height": 4},
                    time_range=TimeRange.LAST_HOUR,
                    options={"yAxis": {"min": 0, "max": 100}}
                ),
                DashboardWidget(
                    widget_id="api_response_time",
                    title="API Response Time",
                    chart_type=ChartType.LINE_CHART,
                    data_source="performance.api_response_time",
                    position={"x": 0, "y": 4, "width": 6, "height": 4},
                    time_range=TimeRange.LAST_HOUR
                ),
                DashboardWidget(
                    widget_id="request_rate",
                    title="Request Rate",
                    chart_type=ChartType.BAR_CHART,
                    data_source="performance.request_rate",
                    position={"x": 6, "y": 4, "width": 6, "height": 4},
                    time_range=TimeRange.LAST_HOUR
                )
            ]
        )
        
        # Business Metrics Dashboard
        business_dashboard = Dashboard(
            dashboard_id="business_metrics",
            name="Business Metrics",
            description="Key business performance indicators and metrics",
            dashboard_type=DashboardType.BUSINESS_METRICS,
            widgets=[
                DashboardWidget(
                    widget_id="daily_active_users",
                    title="Daily Active Users",
                    chart_type=ChartType.STAT,
                    data_source="business.daily_active_users",
                    position={"x": 0, "y": 0, "width": 3, "height": 3}
                ),
                DashboardWidget(
                    widget_id="revenue_metrics",
                    title="Revenue Metrics",
                    chart_type=ChartType.LINE_CHART,
                    data_source="business.revenue",
                    position={"x": 3, "y": 0, "width": 6, "height": 3},
                    time_range=TimeRange.LAST_MONTH
                ),
                DashboardWidget(
                    widget_id="kpi_summary",
                    title="KPI Summary",
                    chart_type=ChartType.TABLE,
                    data_source="business.kpi_summary",
                    position={"x": 9, "y": 0, "width": 3, "height": 6}
                )
            ]
        )
        
        # Alerts Dashboard
        alerts_dashboard = Dashboard(
            dashboard_id="alerts",
            name="Alert Management",
            description="Active alerts and alert management",
            dashboard_type=DashboardType.ALERTS,
            widgets=[
                DashboardWidget(
                    widget_id="alerts_by_severity",
                    title="Alerts by Severity",
                    chart_type=ChartType.PIE_CHART,
                    data_source="alerts.by_severity",
                    position={"x": 0, "y": 0, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_id="alert_count_trend",
                    title="Alert Count Trend",
                    chart_type=ChartType.LINE_CHART,
                    data_source="alerts.active_count",
                    position={"x": 6, "y": 0, "width": 6, "height": 4},
                    time_range=TimeRange.LAST_DAY
                )
            ]
        )
        
        # Store default dashboards
        self.dashboards = {
            "system_health": health_dashboard,
            "performance": performance_dashboard,
            "business_metrics": business_dashboard,
            "alerts": alerts_dashboard
        }
    
    async def get_dashboard_data(self, dashboard_id: str, time_range: str = "1h") -> Dict[str, Any]:
        """Get dashboard data for rendering."""
        if dashboard_id not in self.dashboards:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        dashboard = self.dashboards[dashboard_id]
        widget_data = {}
        
        # Collect data for each widget
        for widget in dashboard.widgets:
            try:
                data_source_func = self.data_sources.get(widget.data_source)
                if data_source_func:
                    widget_time_range = time_range if widget.time_range == TimeRange.CUSTOM else widget.time_range.value
                    data = await data_source_func(widget_time_range)
                    widget_data[widget.widget_id] = {
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat(),
                        "widget_config": widget.to_dict()
                    }
                else:
                    logger.warning(f"Data source not found: {widget.data_source}")
                    widget_data[widget.widget_id] = {
                        "error": f"Data source '{widget.data_source}' not available",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error collecting data for widget {widget.widget_id}: {e}")
                widget_data[widget.widget_id] = {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return {
            "dashboard": dashboard.to_dict(),
            "widgets": widget_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # Data source implementations
    async def _get_health_score(self, time_range: str) -> Dict[str, Any]:
        """Get overall health score."""
        if not MONITORING_AVAILABLE:
            return {"value": 0, "status": "unavailable"}
        
        try:
            if health_monitor.health_history:
                latest = health_monitor.health_history[-1]
                return {
                    "value": latest.overall_score,
                    "status": latest.overall_status.value,
                    "timestamp": latest.timestamp.isoformat()
                }
            return {"value": 100, "status": "healthy"}
        except Exception as e:
            logger.error(f"Failed to get health score: {e}")
            return {"value": 0, "status": "error", "error": str(e)}
    
    async def _get_component_status(self, time_range: str) -> List[Dict[str, Any]]:
        """Get component health status."""
        if not MONITORING_AVAILABLE:
            return []
        
        try:
            if health_monitor.health_history:
                latest = health_monitor.health_history[-1]
                components = []
                for name, result in latest.components.items():
                    components.append({
                        "component": name,
                        "status": result.status.value,
                        "message": result.message,
                        "response_time_ms": result.response_time_ms,
                        "last_check": result.timestamp.isoformat()
                    })
                return components
            return []
        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return []
    
    async def _get_cpu_usage(self, time_range: str) -> List[Dict[str, Any]]:
        """Get CPU usage metrics."""
        try:
            if performance_collector:
                resource_stats = performance_collector.get_resource_stats()
                if resource_stats:
                    return [{
                        "timestamp": datetime.utcnow().isoformat(),
                        "value": resource_stats.get("avg_cpu_percent", 0)
                    }]
            return []
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return []
    
    async def _get_memory_usage(self, time_range: str) -> List[Dict[str, Any]]:
        """Get memory usage metrics."""
        try:
            if performance_collector:
                resource_stats = performance_collector.get_resource_stats()
                if resource_stats:
                    return [{
                        "timestamp": datetime.utcnow().isoformat(),
                        "value": resource_stats.get("avg_memory_percent", 0)
                    }]
            return []
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return []
    
    async def _get_api_response_time(self, time_range: str) -> List[Dict[str, Any]]:
        """Get API response time metrics."""
        try:
            if performance_collector:
                endpoint_stats = performance_collector.get_endpoint_stats()
                if endpoint_stats:
                    return [{
                        "timestamp": datetime.utcnow().isoformat(),
                        "avg_response_time": endpoint_stats.get("avg_response_time_ms", 0),
                        "p95_response_time": endpoint_stats.get("p95_response_time_ms", 0),
                        "p99_response_time": endpoint_stats.get("p99_response_time_ms", 0)
                    }]
            return []
        except Exception as e:
            logger.error(f"Failed to get API response time: {e}")
            return []
    
    async def _get_request_rate(self, time_range: str) -> List[Dict[str, Any]]:
        """Get API request rate metrics."""
        try:
            if performance_collector:
                endpoint_stats = performance_collector.get_endpoint_stats()
                if endpoint_stats:
                    return [{
                        "timestamp": datetime.utcnow().isoformat(),
                        "request_count": endpoint_stats.get("request_count", 0),
                        "error_rate": endpoint_stats.get("error_rate", 0)
                    }]
            return []
        except Exception as e:
            logger.error(f"Failed to get request rate: {e}")
            return []
    
    async def _get_active_alerts_count(self, time_range: str) -> Dict[str, Any]:
        """Get active alerts count."""
        try:
            if alert_manager:
                active_alerts = alert_manager.get_active_alerts()
                return {
                    "count": len(active_alerts),
                    "timestamp": datetime.utcnow().isoformat()
                }
            return {"count": 0}
        except Exception as e:
            logger.error(f"Failed to get active alerts count: {e}")
            return {"count": 0, "error": str(e)}
    
    async def _get_alerts_by_severity(self, time_range: str) -> List[Dict[str, Any]]:
        """Get alerts grouped by severity."""
        try:
            if alert_manager:
                stats = alert_manager.get_alert_stats()
                severity_counts = stats.get("active_alerts", {}).get("by_level", {})
                
                return [
                    {"severity": severity, "count": count}
                    for severity, count in severity_counts.items()
                ]
            return []
        except Exception as e:
            logger.error(f"Failed to get alerts by severity: {e}")
            return []
    
    async def _get_daily_active_users(self, time_range: str) -> Dict[str, Any]:
        """Get daily active users metric."""
        try:
            if business_metrics:
                user_metrics = await business_metrics.collect_user_metrics()
                return {
                    "value": user_metrics.active_users_today,
                    "timestamp": datetime.utcnow().isoformat()
                }
            return {"value": 0}
        except Exception as e:
            logger.error(f"Failed to get daily active users: {e}")
            return {"value": 0, "error": str(e)}
    
    async def _get_revenue_metrics(self, time_range: str) -> List[Dict[str, Any]]:
        """Get revenue metrics."""
        try:
            if business_metrics:
                financial_metrics = await business_metrics.collect_financial_metrics()
                return [{
                    "timestamp": datetime.utcnow().isoformat(),
                    "monthly_recurring_revenue": financial_metrics.monthly_recurring_revenue,
                    "revenue_today": financial_metrics.total_revenue_today,
                    "revenue_month": financial_metrics.total_revenue_month
                }]
            return []
        except Exception as e:
            logger.error(f"Failed to get revenue metrics: {e}")
            return []
    
    async def _get_kpi_summary(self, time_range: str) -> List[Dict[str, Any]]:
        """Get KPI summary data."""
        try:
            if business_metrics:
                kpi_dashboard = business_metrics.get_kpi_dashboard()
                kpis = kpi_dashboard.get("kpis", {})
                
                return [
                    {
                        "name": kpi_data["name"],
                        "current_value": kpi_data["current_value"],
                        "target_value": kpi_data["target_value"],
                        "achievement_percentage": kpi_data["achievement_percentage"],
                        "health_status": kpi_data["health_status"],
                        "unit": kpi_data["unit"]
                    }
                    for kpi_data in kpis.values()
                ]
            return []
        except Exception as e:
            logger.error(f"Failed to get KPI summary: {e}")
            return []
    
    async def _get_database_performance(self, time_range: str) -> List[Dict[str, Any]]:
        """Get database performance metrics."""
        try:
            if db_monitor:
                current_metrics = db_monitor.get_current_metrics()
                performance_data = []
                
                for db_name, metrics in current_metrics.items():
                    performance_data.append({
                        "database": db_name,
                        "avg_query_duration": getattr(metrics, 'avg_query_duration', 0),
                        "active_connections": getattr(metrics, 'active_connections', 0),
                        "cache_hit_ratio": getattr(metrics, 'cache_hit_ratio', 0),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                return performance_data
            return []
        except Exception as e:
            logger.error(f"Failed to get database performance: {e}")
            return []
    
    async def _get_database_connections(self, time_range: str) -> List[Dict[str, Any]]:
        """Get database connection metrics."""
        try:
            if db_monitor:
                current_metrics = db_monitor.get_current_metrics()
                connection_data = []
                
                for db_name, metrics in current_metrics.items():
                    connection_data.append({
                        "database": db_name,
                        "active_connections": getattr(metrics, 'active_connections', 0),
                        "max_connections": getattr(metrics, 'max_connections', 100),
                        "connection_utilization": getattr(metrics, 'connection_utilization', 0),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                return connection_data
            return []
        except Exception as e:
            logger.error(f"Failed to get database connections: {e}")
            return []
    
    async def _get_uptime_metric(self, time_range: str) -> Dict[str, Any]:
        """Get system uptime metric."""
        try:
            if health_monitor:
                uptime_seconds = (datetime.utcnow() - health_monitor.start_time).total_seconds()
                
                # Convert to human-readable format
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                
                return {
                    "uptime_seconds": uptime_seconds,
                    "uptime_formatted": f"{days}d {hours}h {minutes}m",
                    "started_at": health_monitor.start_time.isoformat(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            return {"uptime_seconds": 0, "uptime_formatted": "0d 0h 0m"}
        except Exception as e:
            logger.error(f"Failed to get uptime metric: {e}")
            return {"uptime_seconds": 0, "uptime_formatted": "Error", "error": str(e)}
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards."""
        return [
            {
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets),
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat()
            }
            for dashboard in self.dashboards.values()
        ]
    
    def create_custom_dashboard(self, name: str, description: str, widgets: List[Dict[str, Any]], 
                              user: str) -> str:
        """Create a custom dashboard."""
        dashboard_id = f"custom_{name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Convert widget dictionaries to DashboardWidget objects
        widget_objects = []
        for widget_dict in widgets:
            widget = DashboardWidget(
                widget_id=widget_dict.get("widget_id", f"widget_{len(widget_objects)}"),
                title=widget_dict.get("title", "Untitled Widget"),
                chart_type=ChartType(widget_dict.get("chart_type", "stat")),
                data_source=widget_dict.get("data_source", ""),
                position=widget_dict.get("position", {"x": 0, "y": 0, "width": 6, "height": 4}),
                time_range=TimeRange(widget_dict.get("time_range", "1h")),
                refresh_interval=widget_dict.get("refresh_interval", 30),
                options=widget_dict.get("options", {}),
                filters=widget_dict.get("filters", {})
            )
            widget_objects.append(widget)
        
        # Create dashboard
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            dashboard_type=DashboardType.CUSTOM,
            widgets=widget_objects,
            created_by=user
        )
        
        self.dashboards[dashboard_id] = dashboard
        logger.info(f"Created custom dashboard: {name} ({dashboard_id}) by {user}")
        
        return dashboard_id
    
    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any], user: str) -> bool:
        """Update an existing dashboard."""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Update allowed fields
        if "name" in updates:
            dashboard.name = updates["name"]
        if "description" in updates:
            dashboard.description = updates["description"]
        if "widgets" in updates:
            # Convert and update widgets
            widget_objects = []
            for widget_dict in updates["widgets"]:
                widget = DashboardWidget(
                    widget_id=widget_dict.get("widget_id", f"widget_{len(widget_objects)}"),
                    title=widget_dict.get("title", "Untitled Widget"),
                    chart_type=ChartType(widget_dict.get("chart_type", "stat")),
                    data_source=widget_dict.get("data_source", ""),
                    position=widget_dict.get("position", {"x": 0, "y": 0, "width": 6, "height": 4}),
                    time_range=TimeRange(widget_dict.get("time_range", "1h")),
                    refresh_interval=widget_dict.get("refresh_interval", 30),
                    options=widget_dict.get("options", {}),
                    filters=widget_dict.get("filters", {})
                )
                widget_objects.append(widget)
            dashboard.widgets = widget_objects
        
        dashboard.updated_at = datetime.utcnow()
        
        logger.info(f"Updated dashboard: {dashboard_id} by {user}")
        return True
    
    def delete_dashboard(self, dashboard_id: str, user: str) -> bool:
        """Delete a dashboard."""
        if dashboard_id not in self.dashboards:
            return False
        
        # Only allow deletion of custom dashboards
        dashboard = self.dashboards[dashboard_id]
        if dashboard.dashboard_type != DashboardType.CUSTOM:
            return False
        
        del self.dashboards[dashboard_id]
        logger.info(f"Deleted custom dashboard: {dashboard_id} by {user}")
        return True
    
    def generate_dashboard_html(self, dashboard_id: str) -> str:
        """Generate HTML for dashboard rendering."""
        if dashboard_id not in self.dashboards:
            return "<html><body><h1>Dashboard not found</h1></body></html>"
        
        dashboard = self.dashboards[dashboard_id]
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard.name} - SuperNova AI Monitoring</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gridstack@9.2.0/dist/gridstack-all.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/gridstack@9.2.0/dist/gridstack.min.css"/>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .dashboard-header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .dashboard-title {{
            font-size: 24px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
        }}
        .dashboard-description {{
            color: #718096;
            font-size: 14px;
        }}
        .grid-stack {{
            background: transparent;
        }}
        .grid-stack-item-content {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 16px;
            display: flex;
            flex-direction: column;
        }}
        .widget-title {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .widget-content {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #2b6cb0;
        }}
        .stat-label {{
            font-size: 0.875rem;
            color: #718096;
            margin-top: 4px;
        }}
        .health-status {{
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-healthy {{ background: #c6f6d5; color: #22543d; }}
        .status-warning {{ background: #fed7aa; color: #9c4221; }}
        .status-critical {{ background: #fbb6ce; color: #97266d; }}
        .refresh-indicator {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4299e1;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            display: none;
        }}
        .error-message {{
            color: #e53e3e;
            font-size: 14px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="refresh-indicator" id="refreshIndicator">Refreshing...</div>
    
    <div class="dashboard-header">
        <div class="dashboard-title">{dashboard.name}</div>
        <div class="dashboard-description">{dashboard.description}</div>
    </div>
    
    <div class="grid-stack" id="grid-stack">
        <!-- Widgets will be loaded here -->
    </div>

    <script>
        let grid;
        let refreshInterval;
        
        function initGrid() {{
            grid = GridStack.init({{
                cellHeight: 70,
                acceptWidgets: false,
                disableResize: true,
                disableDrag: true
            }});
        }}
        
        async function loadDashboardData() {{
            try {{
                const response = await fetch(`/api/dashboards/{dashboard_id}/data`);
                const data = await response.json();
                
                if (data.error) {{
                    console.error('Dashboard data error:', data.error);
                    return;
                }}
                
                updateWidgets(data.widgets);
            }} catch (error) {{
                console.error('Failed to load dashboard data:', error);
            }}
        }}
        
        function updateWidgets(widgets) {{
            // Clear existing widgets
            grid.removeAll();
            
            // Add widgets
            Object.entries(widgets).forEach(([widgetId, widgetData]) => {{
                const config = widgetData.widget_config;
                const content = generateWidgetContent(widgetData);
                
                const widgetHtml = `
                    <div class="grid-stack-item-content">
                        <div class="widget-title">${{config.title}}</div>
                        <div class="widget-content">${{content}}</div>
                    </div>
                `;
                
                grid.addWidget(widgetHtml, {{
                    x: config.position.x,
                    y: config.position.y,
                    w: config.position.width,
                    h: config.position.height,
                    id: widgetId
                }});
            }});
        }}
        
        function generateWidgetContent(widgetData) {{
            if (widgetData.error) {{
                return `<div class="error-message">Error: ${{widgetData.error}}</div>`;
            }}
            
            const config = widgetData.widget_config;
            const data = widgetData.data;
            
            switch (config.chart_type) {{
                case 'stat':
                    return generateStatWidget(data);
                case 'gauge':
                    return generateGaugeWidget(data);
                case 'table':
                    return generateTableWidget(data);
                default:
                    return `<div>Chart type ${{config.chart_type}} not implemented</div>`;
            }}
        }}
        
        function generateStatWidget(data) {{
            if (typeof data === 'object' && data.value !== undefined) {{
                return `
                    <div style="text-align: center;">
                        <div class="stat-value">${{data.value}}</div>
                        <div class="stat-label">${{data.status || ''}}</div>
                    </div>
                `;
            }}
            return `<div class="stat-value">${{data}}</div>`;
        }}
        
        function generateGaugeWidget(data) {{
            const value = typeof data === 'object' ? data.value : data;
            const max = 100;
            const percentage = (value / max) * 100;
            
            let color = '#22c55e'; // green
            if (percentage > 70) color = '#f59e0b'; // yellow
            if (percentage > 90) color = '#ef4444'; // red
            
            return `
                <div style="text-align: center; width: 100%;">
                    <div style="width: 120px; height: 120px; margin: 0 auto; position: relative;">
                        <svg viewBox="0 0 120 120" style="transform: rotate(-90deg);">
                            <circle cx="60" cy="60" r="50" fill="none" stroke="#e2e8f0" stroke-width="8"/>
                            <circle cx="60" cy="60" r="50" fill="none" stroke="${{color}}" stroke-width="8"
                                stroke-dasharray="${{Math.PI * 2 * 50}}" 
                                stroke-dashoffset="${{Math.PI * 2 * 50 * (1 - percentage / 100)}}"/>
                        </svg>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                            <div style="font-size: 18px; font-weight: bold;">${{Math.round(value)}}</div>
                        </div>
                    </div>
                </div>
            `;
        }}
        
        function generateTableWidget(data) {{
            if (!Array.isArray(data) || data.length === 0) {{
                return '<div>No data available</div>';
            }}
            
            const headers = Object.keys(data[0]);
            let html = '<table style="width: 100%; font-size: 12px;"><thead><tr>';
            
            headers.forEach(header => {{
                html += `<th style="padding: 4px; text-align: left; border-bottom: 1px solid #e2e8f0;">${{header}}</th>`;
            }});
            
            html += '</tr></thead><tbody>';
            
            data.forEach(row => {{
                html += '<tr>';
                headers.forEach(header => {{
                    let value = row[header];
                    if (header === 'status') {{
                        value = `<span class="health-status status-${{value}}">${{value}}</span>`;
                    }}
                    html += `<td style="padding: 4px;">${{value}}</td>`;
                }});
                html += '</tr>';
            }});
            
            html += '</tbody></table>';
            return html;
        }}
        
        function showRefreshIndicator() {{
            document.getElementById('refreshIndicator').style.display = 'block';
            setTimeout(() => {{
                document.getElementById('refreshIndicator').style.display = 'none';
            }}, 1000);
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            initGrid();
            loadDashboardData();
            
            // Setup auto-refresh
            refreshInterval = setInterval(() => {{
                showRefreshIndicator();
                loadDashboardData();
            }}, 30000); // Refresh every 30 seconds
        }});
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            if (refreshInterval) {{
                clearInterval(refreshInterval);
            }}
        }});
    </script>
</body>
</html>
        """
        
        return html_template

# Global dashboard manager
dashboard_manager = MonitoringDashboardManager()

# Export key components
__all__ = [
    'MonitoringDashboardManager', 'Dashboard', 'DashboardWidget', 'DashboardType',
    'ChartType', 'TimeRange', 'dashboard_manager'
]