"""
System monitoring and observability for AI Principles Gym.

Tracks key performance metrics, provides alerting thresholds,
and integrates with structured logging for comprehensive observability.
"""
import asyncio
import functools
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

from .config import settings

logger = structlog.get_logger(__name__)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])


class MetricType(str, Enum):
    """Types of metrics we track."""
    INFERENCE_LATENCY = "inference_latency"
    BEHAVIORAL_ENTROPY = "behavioral_entropy_distribution"
    PRINCIPLE_DISCOVERY_RATE = "principle_discovery_rate"
    SCENARIO_GENERATION_TIME = "scenario_generation_time"
    CONCURRENT_TRAINING = "concurrent_training_sessions"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    DB_QUERY_TIME = "database_query_time"
    API_RESPONSE_TIME = "api_response_time"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertThreshold:
    """Configuration for metric alerting."""
    metric: MetricType
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    level: AlertLevel
    message: str
    cooldown_minutes: int = 5  # Prevent alert spam


@dataclass
class MetricSnapshot:
    """Point-in-time metric value."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    
class MetricsCollector:
    """Centralized metrics collection and alerting."""
    
    def __init__(self):
        # Prometheus metrics
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.summaries: Dict[str, Summary] = {}
        
        # Internal tracking
        self._metrics_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._alert_history: Dict[str, datetime] = {}
        self._active_sessions: Set[str] = set()
        
        # Define alerting thresholds
        self.thresholds = [
            AlertThreshold(
                metric=MetricType.BEHAVIORAL_ENTROPY,
                threshold=0.8,
                comparison="gt",
                level=AlertLevel.WARNING,
                message="High entropy detected - agent behavior may be random"
            ),
            AlertThreshold(
                metric=MetricType.INFERENCE_LATENCY,
                threshold=1000,  # 1 second in ms
                comparison="gt",
                level=AlertLevel.ERROR,
                message="Slow inference detected - performance degradation"
            ),
            AlertThreshold(
                metric=MetricType.MEMORY_USAGE,
                threshold=80,  # percentage
                comparison="gt",
                level=AlertLevel.WARNING,
                message="High memory usage - potential memory leak"
            ),
            AlertThreshold(
                metric=MetricType.ERROR_RATE,
                threshold=1,  # percentage
                comparison="gt",
                level=AlertLevel.ERROR,
                message="High error rate - system issues detected"
            ),
            AlertThreshold(
                metric=MetricType.CONCURRENT_TRAINING,
                threshold=1000,
                comparison="gt",
                level=AlertLevel.WARNING,
                message="High concurrent training sessions - may impact performance"
            ),
        ]
        
        self._initialize_prometheus_metrics()
        
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metric collectors."""
        # Histograms for latency metrics
        self.histograms['inference_latency'] = Histogram(
            'inference_latency_seconds',
            'Time spent in principle inference',
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.histograms['scenario_generation_time'] = Histogram(
            'scenario_generation_seconds',
            'Time to generate scenarios',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
        )
        
        self.histograms['api_response_time'] = Histogram(
            'api_response_seconds',
            'API endpoint response times',
            ['method', 'endpoint', 'status']
        )
        
        # Gauges for current state metrics
        self.gauges['concurrent_training_sessions'] = Gauge(
            'concurrent_training_sessions',
            'Number of active training sessions'
        )
        
        self.gauges['memory_usage_percent'] = Gauge(
            'memory_usage_percent',
            'Current memory usage percentage'
        )
        
        self.gauges['behavioral_entropy'] = Gauge(
            'behavioral_entropy',
            'Current behavioral entropy distribution',
            ['agent_id']
        )
        
        # Counters for cumulative metrics
        self.counters['principles_discovered'] = Counter(
            'principles_discovered_total',
            'Total number of principles discovered'
        )
        
        self.counters['errors'] = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        self.counters['cache_hits'] = Counter(
            'cache_hits_total',
            'Cache hit count',
            ['cache_name']
        )
        
        self.counters['cache_misses'] = Counter(
            'cache_misses_total',
            'Cache miss count',
            ['cache_name']
        )
        
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value and check alert thresholds."""
        labels = labels or {}
        
        # Store in buffer
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels
        )
        self._metrics_buffer[metric_type].append(snapshot)
        
        # Update Prometheus metrics
        self._update_prometheus(metric_type, value, labels)
        
        # Check thresholds
        await self._check_thresholds(metric_type, value)
        
        # Log metric with structure
        logger.debug(
            "metric_recorded",
            metric=metric_type,
            value=value,
            labels=labels,
            request_id=labels.get("request_id"),
            agent_id=labels.get("agent_id"),
            session_id=labels.get("session_id")
        )
        
    def _update_prometheus(
        self,
        metric_type: MetricType,
        value: float,
        labels: Dict[str, str]
    ):
        """Update Prometheus metrics based on type."""
        if metric_type == MetricType.INFERENCE_LATENCY:
            self.histograms['inference_latency'].observe(value / 1000)  # Convert to seconds
        elif metric_type == MetricType.SCENARIO_GENERATION_TIME:
            self.histograms['scenario_generation_time'].observe(value / 1000)
        elif metric_type == MetricType.CONCURRENT_TRAINING:
            self.gauges['concurrent_training_sessions'].set(value)
        elif metric_type == MetricType.BEHAVIORAL_ENTROPY:
            agent_id = labels.get('agent_id', 'unknown')
            self.gauges['behavioral_entropy'].labels(agent_id=agent_id).set(value)
        elif metric_type == MetricType.MEMORY_USAGE:
            self.gauges['memory_usage_percent'].set(value)
        elif metric_type == MetricType.PRINCIPLE_DISCOVERY_RATE:
            self.counters['principles_discovered'].inc(value)
            
    async def _check_thresholds(self, metric_type: MetricType, value: float):
        """Check if metric value triggers any alerts."""
        for threshold in self.thresholds:
            if threshold.metric != metric_type:
                continue
                
            # Check if we're in cooldown
            last_alert = self._alert_history.get(f"{metric_type}:{threshold.level}")
            if last_alert:
                cooldown_end = last_alert + timedelta(minutes=threshold.cooldown_minutes)
                if datetime.utcnow() < cooldown_end:
                    continue
                    
            # Evaluate threshold
            triggered = False
            if threshold.comparison == "gt" and value > threshold.threshold:
                triggered = True
            elif threshold.comparison == "lt" and value < threshold.threshold:
                triggered = True
            elif threshold.comparison == "eq" and value == threshold.threshold:
                triggered = True
                
            if triggered:
                await self._trigger_alert(threshold, metric_type, value)
                
    async def _trigger_alert(
        self,
        threshold: AlertThreshold,
        metric_type: MetricType,
        value: float
    ):
        """Trigger an alert for threshold violation."""
        alert_key = f"{metric_type}:{threshold.level}"
        self._alert_history[alert_key] = datetime.utcnow()
        
        log_method = getattr(logger, threshold.level.value)
        log_method(
            "alert_triggered",
            metric=metric_type,
            value=value,
            threshold=threshold.threshold,
            message=threshold.message,
            alert_level=threshold.level
        )
        
        # In production, this would integrate with alerting systems
        # like PagerDuty, Slack, etc.
        
    def start_training_session(self, session_id: str):
        """Track a new training session."""
        self._active_sessions.add(session_id)
        self.gauges['concurrent_training_sessions'].set(len(self._active_sessions))
        
    def end_training_session(self, session_id: str):
        """Mark training session as complete."""
        self._active_sessions.discard(session_id)
        self.gauges['concurrent_training_sessions'].set(len(self._active_sessions))
        
    def record_cache_hit(self, cache_name: str):
        """Record a cache hit."""
        self.counters['cache_hits'].labels(cache_name=cache_name).inc()
        
    def record_cache_miss(self, cache_name: str):
        """Record a cache miss."""
        self.counters['cache_misses'].labels(cache_name=cache_name).inc()
        
    def record_error(self, error_type: str, component: str):
        """Record an error occurrence."""
        self.counters['errors'].labels(
            error_type=error_type,
            component=component
        ).inc()
        
    def get_metric_summary(
        self,
        metric_type: MetricType,
        minutes: int = 5
    ) -> Dict[str, Any]:
        """Get summary statistics for a metric over time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        values = [
            s.value for s in self._metrics_buffer[metric_type]
            if s.timestamp > cutoff
        ]
        
        if not values:
            return {
                "metric": metric_type,
                "window_minutes": minutes,
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "p50": None,
                "p95": None,
                "p99": None
            }
            
        values.sort()
        count = len(values)
        
        return {
            "metric": metric_type,
            "window_minutes": minutes,
            "count": count,
            "mean": sum(values) / count,
            "min": values[0],
            "max": values[-1],
            "p50": values[int(count * 0.5)],
            "p95": values[int(count * 0.95)],
            "p99": values[int(count * 0.99)]
        }


# Global metrics collector instance
metrics = MetricsCollector()


def monitor_performance(metric_name: str) -> Callable[[F], F]:
    """
    Decorator to automatically track function execution time.
    
    Usage:
        @monitor_performance("inference_latency")
        async def infer_principles():
            # function implementation
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            labels = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            # Extract common labels from kwargs if present
            for label_key in ["request_id", "agent_id", "session_id"]:
                if label_key in kwargs:
                    labels[label_key] = kwargs[label_key]
                    
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                metrics.record_error(
                    error_type=type(e).__name__,
                    component=func.__module__
                )
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                await metrics.record_metric(
                    MetricType(metric_name),
                    elapsed_ms,
                    labels
                )
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            labels = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            for label_key in ["request_id", "agent_id", "session_id"]:
                if label_key in kwargs:
                    labels[label_key] = kwargs[label_key]
                    
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                metrics.record_error(
                    error_type=type(e).__name__,
                    component=func.__module__
                )
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                # Run async record in background for sync functions
                asyncio.create_task(
                    metrics.record_metric(
                        MetricType(metric_name),
                        elapsed_ms,
                        labels
                    )
                )
                
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


@asynccontextmanager
async def monitor_operation(
    operation_name: str,
    metric_type: MetricType,
    labels: Optional[Dict[str, str]] = None
):
    """
    Context manager for monitoring operations.
    
    Usage:
        async with monitor_operation("database_query", MetricType.DB_QUERY_TIME):
            await db.execute(query)
    """
    start_time = time.perf_counter()
    labels = labels or {}
    labels["operation"] = operation_name
    
    try:
        yield
    except Exception as e:
        metrics.record_error(
            error_type=type(e).__name__,
            component=operation_name
        )
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        await metrics.record_metric(metric_type, elapsed_ms, labels)


async def check_system_health() -> Dict[str, Any]:
    """
    Perform system health check and return status.
    
    Returns dict with:
    - healthy: bool
    - checks: Dict of component health
    - metrics: Recent metric summaries
    """
    checks = {}
    
    # Check memory usage
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        await metrics.record_metric(MetricType.MEMORY_USAGE, memory_percent)
        checks["memory"] = {
            "healthy": memory_percent < 80,
            "value": memory_percent,
            "unit": "percent"
        }
    except ImportError:
        checks["memory"] = {"healthy": True, "value": None, "error": "psutil not installed"}
        
    # Check concurrent sessions
    session_count = len(metrics._active_sessions)
    checks["concurrent_sessions"] = {
        "healthy": session_count < 1000,
        "value": session_count,
        "unit": "sessions"
    }
    
    # Get recent metric summaries
    metric_summaries = {}
    for metric_type in [
        MetricType.INFERENCE_LATENCY,
        MetricType.SCENARIO_GENERATION_TIME,
        MetricType.API_RESPONSE_TIME
    ]:
        metric_summaries[metric_type] = metrics.get_metric_summary(metric_type)
        
    # Overall health
    all_healthy = all(check.get("healthy", False) for check in checks.values())
    
    return {
        "healthy": all_healthy,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "metrics": metric_summaries,
        "active_sessions": session_count
    }


# Export commonly used functions and objects
__all__ = [
    "monitor_performance",
    "monitor_operation", 
    "metrics",
    "MetricType",
    "AlertLevel",
    "check_system_health"
]
