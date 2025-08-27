"""
Database Optimization and Performance Management System

Comprehensive database optimization including:
- Query performance monitoring and optimization
- Index management and recommendations
- Connection pool optimization
- Slow query detection and analysis
- Database performance metrics collection
"""

from __future__ import annotations
import logging
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from sqlalchemy import text, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool
from sqlalchemy.event import listen
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import settings
    from .database_config import db_config, get_session
    from .db import Base
    from .sentiment_models import TimescaleBase
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    raise e

logger = logging.getLogger(__name__)

@dataclass
class QueryStats:
    """Query performance statistics."""
    query_hash: str
    query_text: str
    database: str
    execution_count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    last_executed: datetime
    table_names: Set[str]

@dataclass
class IndexRecommendation:
    """Database index recommendation."""
    table_name: str
    column_names: List[str]
    index_type: str  # btree, hash, gin, gist
    estimated_benefit: float
    query_patterns: List[str]
    estimated_size_mb: float
    priority: str  # high, medium, low

@dataclass
class PerformanceMetrics:
    """Database performance metrics."""
    timestamp: datetime
    database: str
    active_connections: int
    total_queries: int
    slow_queries: int
    avg_query_time: float
    cache_hit_ratio: float
    lock_waits: int
    deadlocks: int
    disk_usage_gb: float

class QueryMonitor:
    """Monitor and analyze database query performance."""
    
    def __init__(self, max_queries: int = 1000):
        self.max_queries = max_queries
        self.query_stats: Dict[str, QueryStats] = {}
        self.recent_queries: deque = deque(maxlen=100)
        self.slow_query_threshold = getattr(settings, 'SLOW_QUERY_THRESHOLD', 1.0)
        self.lock = threading.Lock()
        
    def start_monitoring(self, database: str = 'primary') -> None:
        """Start query monitoring for specified database."""
        try:
            if database not in db_config.engines:
                db_config.create_engine(database)
            
            engine = db_config.engines[database]
            
            # Add event listeners
            listen(engine, "before_cursor_execute", self._before_cursor_execute)
            listen(engine, "after_cursor_execute", self._after_cursor_execute)
            
            logger.info(f"Started query monitoring for database: {database}")
            
        except Exception as e:
            logger.error(f"Failed to start query monitoring for {database}: {e}")
    
    def _before_cursor_execute(self, conn, cursor, statement, parameters, context, executemany):
        """Called before query execution."""
        context._query_start_time = time.time()
    
    def _after_cursor_execute(self, conn, cursor, statement, parameters, context, executemany):
        """Called after query execution."""
        try:
            execution_time = time.time() - getattr(context, '_query_start_time', time.time())
            
            # Generate query hash for grouping similar queries
            query_hash = self._generate_query_hash(statement)
            
            # Extract table names
            table_names = self._extract_table_names(statement)
            
            with self.lock:
                # Update query statistics
                if query_hash in self.query_stats:
                    stats = self.query_stats[query_hash]
                    stats.execution_count += 1
                    stats.total_time += execution_time
                    stats.min_time = min(stats.min_time, execution_time)
                    stats.max_time = max(stats.max_time, execution_time)
                    stats.avg_time = stats.total_time / stats.execution_count
                    stats.last_executed = datetime.utcnow()
                else:
                    # Create new query stats
                    stats = QueryStats(
                        query_hash=query_hash,
                        query_text=statement[:500],  # Truncate long queries
                        database=getattr(conn.engine.url, 'database', 'unknown'),
                        execution_count=1,
                        total_time=execution_time,
                        min_time=execution_time,
                        max_time=execution_time,
                        avg_time=execution_time,
                        last_executed=datetime.utcnow(),
                        table_names=table_names
                    )
                    self.query_stats[query_hash] = stats
                
                # Track recent queries
                self.recent_queries.append({
                    'timestamp': datetime.utcnow(),
                    'query_hash': query_hash,
                    'execution_time': execution_time,
                    'is_slow': execution_time > self.slow_query_threshold
                })
                
                # Log slow queries
                if execution_time > self.slow_query_threshold:
                    logger.warning(
                        f"Slow query detected: {execution_time:.3f}s - {statement[:100]}..."
                    )
            
            # Cleanup old stats if needed
            if len(self.query_stats) > self.max_queries:
                self._cleanup_old_stats()
                
        except Exception as e:
            logger.error(f"Error in query monitoring: {e}")
    
    def _generate_query_hash(self, statement: str) -> str:
        """Generate hash for query grouping (normalize parameters)."""
        import hashlib
        import re
        
        # Normalize the query by removing parameters
        normalized = re.sub(r'\b\d+\b', '?', statement)  # Replace numbers with ?
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # Replace strings with '?'
        normalized = re.sub(r'\s+', ' ', normalized.strip().upper())  # Normalize whitespace
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _extract_table_names(self, statement: str) -> Set[str]:
        """Extract table names from SQL statement."""
        import re
        
        table_names = set()
        
        # Simple regex patterns for common SQL operations
        patterns = [
            r'\bFROM\s+(\w+)',
            r'\bJOIN\s+(\w+)',
            r'\bINTO\s+(\w+)',
            r'\bUPDATE\s+(\w+)',
            r'\bDELETE\s+FROM\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            table_names.update(matches)
        
        return table_names
    
    def _cleanup_old_stats(self) -> None:
        """Remove oldest query statistics."""
        sorted_stats = sorted(
            self.query_stats.items(),
            key=lambda x: x[1].last_executed
        )
        
        # Remove oldest 10% of queries
        remove_count = len(sorted_stats) // 10
        for query_hash, _ in sorted_stats[:remove_count]:
            del self.query_stats[query_hash]
    
    def get_slow_queries(self, limit: int = 20) -> List[QueryStats]:
        """Get slowest queries."""
        with self.lock:
            slow_queries = [
                stats for stats in self.query_stats.values()
                if stats.avg_time > self.slow_query_threshold
            ]
            
            return sorted(slow_queries, key=lambda x: x.avg_time, reverse=True)[:limit]
    
    def get_most_frequent_queries(self, limit: int = 20) -> List[QueryStats]:
        """Get most frequently executed queries."""
        with self.lock:
            return sorted(
                self.query_stats.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:limit]
    
    def get_query_stats_summary(self) -> Dict[str, Any]:
        """Get query statistics summary."""
        with self.lock:
            if not self.query_stats:
                return {
                    'total_queries': 0,
                    'slow_queries': 0,
                    'avg_execution_time': 0,
                    'recent_activity': 0
                }
            
            total_queries = sum(stats.execution_count for stats in self.query_stats.values())
            slow_queries = sum(
                1 for stats in self.query_stats.values()
                if stats.avg_time > self.slow_query_threshold
            )
            
            all_times = []
            for stats in self.query_stats.values():
                all_times.extend([stats.avg_time] * stats.execution_count)
            
            recent_activity = len([
                q for q in self.recent_queries
                if q['timestamp'] > datetime.utcnow() - timedelta(minutes=5)
            ])
            
            return {
                'total_queries': total_queries,
                'unique_queries': len(self.query_stats),
                'slow_queries': slow_queries,
                'avg_execution_time': statistics.mean(all_times) if all_times else 0,
                'min_execution_time': min(all_times) if all_times else 0,
                'max_execution_time': max(all_times) if all_times else 0,
                'recent_activity': recent_activity,
                'slow_query_threshold': self.slow_query_threshold
            }

class IndexOptimizer:
    """Analyze queries and recommend database indexes."""
    
    def __init__(self, query_monitor: QueryMonitor):
        self.query_monitor = query_monitor
        
    def analyze_missing_indexes(self, database: str = 'primary') -> List[IndexRecommendation]:
        """Analyze queries and recommend missing indexes."""
        recommendations = []
        
        try:
            with get_session(database) as session:
                # Get current indexes
                existing_indexes = self._get_existing_indexes(session, database)
                
                # Analyze slow queries for index opportunities
                slow_queries = self.query_monitor.get_slow_queries(50)
                
                for query_stats in slow_queries:
                    recommendations.extend(
                        self._analyze_query_for_indexes(
                            query_stats, existing_indexes, database
                        )
                    )
                
                # Remove duplicates and prioritize
                recommendations = self._prioritize_recommendations(recommendations)
                
        except Exception as e:
            logger.error(f"Error analyzing missing indexes for {database}: {e}")
        
        return recommendations
    
    def _get_existing_indexes(self, session: Session, database: str) -> Dict[str, List[str]]:
        """Get existing indexes for database tables."""
        existing_indexes = defaultdict(list)
        
        try:
            inspector = inspect(session.bind)
            
            for table_name in inspector.get_table_names():
                indexes = inspector.get_indexes(table_name)
                for index in indexes:
                    existing_indexes[table_name].append(
                        tuple(sorted(index['column_names']))
                    )
        except Exception as e:
            logger.error(f"Error getting existing indexes: {e}")
        
        return dict(existing_indexes)
    
    def _analyze_query_for_indexes(
        self, 
        query_stats: QueryStats, 
        existing_indexes: Dict[str, List[str]],
        database: str
    ) -> List[IndexRecommendation]:
        """Analyze a specific query for index opportunities."""
        recommendations = []
        
        # Simple heuristic: if query touches tables and is slow,
        # recommend indexes on commonly filtered/joined columns
        for table_name in query_stats.table_names:
            if table_name in existing_indexes:
                # Analyze WHERE clauses for potential index columns
                potential_columns = self._extract_where_columns(query_stats.query_text)
                
                for columns in potential_columns:
                    if tuple(sorted(columns)) not in existing_indexes[table_name]:
                        recommendations.append(IndexRecommendation(
                            table_name=table_name,
                            column_names=columns,
                            index_type='btree',
                            estimated_benefit=self._estimate_index_benefit(query_stats),
                            query_patterns=[query_stats.query_hash],
                            estimated_size_mb=self._estimate_index_size(table_name, columns),
                            priority=self._calculate_priority(query_stats)
                        ))
        
        return recommendations
    
    def _extract_where_columns(self, query: str) -> List[List[str]]:
        """Extract columns used in WHERE clauses."""
        import re
        
        potential_columns = []
        
        # Simple regex to find WHERE conditions
        where_pattern = r'\bWHERE\s+.*?(?=\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group()
            
            # Find column references (simplified)
            column_pattern = r'\b(\w+)\s*[=<>!]'
            columns = re.findall(column_pattern, where_clause)
            
            # Filter out common SQL keywords
            sql_keywords = {'and', 'or', 'not', 'true', 'false', 'null'}
            columns = [col for col in columns if col.lower() not in sql_keywords]
            
            if columns:
                potential_columns.append(columns[:3])  # Limit to 3 columns per index
        
        return potential_columns
    
    def _estimate_index_benefit(self, query_stats: QueryStats) -> float:
        """Estimate benefit of creating an index."""
        # Simple heuristic based on query frequency and execution time
        return (query_stats.execution_count * query_stats.avg_time) / 100.0
    
    def _estimate_index_size(self, table_name: str, columns: List[str]) -> float:
        """Estimate size of proposed index in MB."""
        # Rough estimate: assume 8 bytes per column per row, 10K rows
        return len(columns) * 8 * 10000 / (1024 * 1024)
    
    def _calculate_priority(self, query_stats: QueryStats) -> str:
        """Calculate priority for index recommendation."""
        if query_stats.avg_time > 5.0 and query_stats.execution_count > 100:
            return 'high'
        elif query_stats.avg_time > 2.0 or query_stats.execution_count > 50:
            return 'medium'
        else:
            return 'low'
    
    def _prioritize_recommendations(self, recommendations: List[IndexRecommendation]) -> List[IndexRecommendation]:
        """Sort and prioritize index recommendations."""
        # Remove duplicates
        unique_recommendations = {}
        
        for rec in recommendations:
            key = (rec.table_name, tuple(rec.column_names))
            if key not in unique_recommendations:
                unique_recommendations[key] = rec
            else:
                # Merge recommendations
                existing = unique_recommendations[key]
                existing.estimated_benefit += rec.estimated_benefit
                existing.query_patterns.extend(rec.query_patterns)
        
        # Sort by priority and estimated benefit
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        return sorted(
            unique_recommendations.values(),
            key=lambda x: (priority_order[x.priority], x.estimated_benefit),
            reverse=True
        )
    
    def create_index(self, recommendation: IndexRecommendation, database: str = 'primary') -> bool:
        """Create recommended index."""
        try:
            index_name = f"idx_{recommendation.table_name}_{'_'.join(recommendation.column_names)}"
            columns_str = ', '.join(recommendation.column_names)
            
            sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
                ON {recommendation.table_name} ({columns_str})
            """
            
            with get_session(database) as session:
                session.execute(text(sql))
                session.commit()
            
            logger.info(f"Created index {index_name} on {recommendation.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

class PerformanceCollector:
    """Collect database performance metrics."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
    
    def collect_metrics(self, database: str = 'primary') -> Optional[PerformanceMetrics]:
        """Collect current performance metrics."""
        try:
            with get_session(database) as session:
                metrics = PerformanceMetrics(
                    timestamp=datetime.utcnow(),
                    database=database,
                    active_connections=self._get_active_connections(session),
                    total_queries=self._get_total_queries(session),
                    slow_queries=self._get_slow_query_count(session),
                    avg_query_time=self._get_avg_query_time(session),
                    cache_hit_ratio=self._get_cache_hit_ratio(session),
                    lock_waits=self._get_lock_waits(session),
                    deadlocks=self._get_deadlock_count(session),
                    disk_usage_gb=self._get_disk_usage(session)
                )
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Cleanup old metrics
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting performance metrics for {database}: {e}")
            return None
    
    def _get_active_connections(self, session: Session) -> int:
        """Get number of active connections."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                ))
                return result.scalar() or 0
            else:
                # For SQLite, return 1 if connected
                return 1
        except Exception:
            return 0
    
    def _get_total_queries(self, session: Session) -> int:
        """Get total number of queries executed."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT sum(calls) FROM pg_stat_statements"
                ))
                return result.scalar() or 0
            else:
                return 0  # Not available for SQLite
        except Exception:
            return 0
    
    def _get_slow_query_count(self, session: Session) -> int:
        """Get number of slow queries."""
        # This would typically come from query monitoring
        return 0
    
    def _get_avg_query_time(self, session: Session) -> float:
        """Get average query execution time."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT avg(mean_time) FROM pg_stat_statements WHERE calls > 0"
                ))
                return result.scalar() or 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_cache_hit_ratio(self, session: Session) -> float:
        """Get database cache hit ratio."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    """
                    SELECT 
                        round(
                            (sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1)) * 100, 2
                        ) as hit_ratio
                    FROM pg_statio_user_tables
                    """
                ))
                return result.scalar() or 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_lock_waits(self, session: Session) -> int:
        """Get number of lock waits."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
                ))
                return result.scalar() or 0
            else:
                return 0
        except Exception:
            return 0
    
    def _get_deadlock_count(self, session: Session) -> int:
        """Get number of deadlocks."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT deadlocks FROM pg_stat_database WHERE datname = current_database()"
                ))
                return result.scalar() or 0
            else:
                return 0
        except Exception:
            return 0
    
    def _get_disk_usage(self, session: Session) -> float:
        """Get database disk usage in GB."""
        try:
            if 'postgresql' in str(session.bind.url):
                result = session.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                ))
                size_str = result.scalar() or "0 GB"
                # Parse size string (simplified)
                if 'GB' in size_str:
                    return float(size_str.replace(' GB', ''))
                elif 'MB' in size_str:
                    return float(size_str.replace(' MB', '')) / 1024
                else:
                    return 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in recent_metrics],
            'active_connections': [m.active_connections for m in recent_metrics],
            'avg_query_time': [m.avg_query_time for m in recent_metrics],
            'cache_hit_ratio': [m.cache_hit_ratio for m in recent_metrics],
            'lock_waits': [m.lock_waits for m in recent_metrics],
            'deadlocks': [m.deadlocks for m in recent_metrics],
            'disk_usage_gb': [m.disk_usage_gb for m in recent_metrics]
        }

class DatabaseOptimizer:
    """Main database optimization manager."""
    
    def __init__(self):
        self.query_monitor = QueryMonitor()
        self.index_optimizer = IndexOptimizer(self.query_monitor)
        self.performance_collector = PerformanceCollector()
        self.monitoring_active = False
    
    def start_optimization(self, databases: List[str] = None) -> None:
        """Start database optimization monitoring."""
        databases = databases or ['primary', 'timescale']
        
        for database in databases:
            if database in db_config.get_database_urls():
                self.query_monitor.start_monitoring(database)
        
        self.monitoring_active = True
        logger.info("Database optimization monitoring started")
    
    def stop_optimization(self) -> None:
        """Stop database optimization monitoring."""
        self.monitoring_active = False
        logger.info("Database optimization monitoring stopped")
    
    def get_optimization_report(self, database: str = 'primary') -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'database': database,
            'monitoring_active': self.monitoring_active
        }
        
        # Query performance summary
        report['query_performance'] = self.query_monitor.get_query_stats_summary()
        
        # Slow queries
        report['slow_queries'] = [
            asdict(stats) for stats in self.query_monitor.get_slow_queries(10)
        ]
        
        # Most frequent queries
        report['frequent_queries'] = [
            asdict(stats) for stats in self.query_monitor.get_most_frequent_queries(10)
        ]
        
        # Index recommendations
        report['index_recommendations'] = [
            asdict(rec) for rec in self.index_optimizer.analyze_missing_indexes(database)
        ]
        
        # Performance metrics
        current_metrics = self.performance_collector.collect_metrics(database)
        if current_metrics:
            report['current_performance'] = asdict(current_metrics)
        
        # Performance trends
        report['performance_trends'] = self.performance_collector.get_performance_trends(24)
        
        return report
    
    def apply_recommendations(self, database: str = 'primary', max_indexes: int = 5) -> Dict[str, Any]:
        """Apply optimization recommendations."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'database': database,
            'indexes_created': 0,
            'indexes_failed': 0,
            'recommendations_applied': []
        }
        
        # Get index recommendations
        recommendations = self.index_optimizer.analyze_missing_indexes(database)
        
        # Apply top recommendations
        for i, recommendation in enumerate(recommendations[:max_indexes]):
            if self.index_optimizer.create_index(recommendation, database):
                results['indexes_created'] += 1
                results['recommendations_applied'].append({
                    'table_name': recommendation.table_name,
                    'columns': recommendation.column_names,
                    'priority': recommendation.priority,
                    'status': 'created'
                })
            else:
                results['indexes_failed'] += 1
                results['recommendations_applied'].append({
                    'table_name': recommendation.table_name,
                    'columns': recommendation.column_names,
                    'priority': recommendation.priority,
                    'status': 'failed'
                })
        
        return results

# Global optimizer instance
db_optimizer = DatabaseOptimizer()

# Convenience functions
def start_database_optimization(databases: List[str] = None) -> None:
    """Start database optimization monitoring."""
    db_optimizer.start_optimization(databases)

def get_optimization_report(database: str = 'primary') -> Dict[str, Any]:
    """Get database optimization report."""
    return db_optimizer.get_optimization_report(database)

def apply_optimizations(database: str = 'primary', max_indexes: int = 5) -> Dict[str, Any]:
    """Apply database optimizations."""
    return db_optimizer.apply_recommendations(database, max_indexes)