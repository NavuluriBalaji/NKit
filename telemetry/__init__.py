"""Telemetry and observability for agent systems.

This module provides:
- Metrics collection (counters, gauges, histograms)
- Tracing for distributed operations
- Performance monitoring
- Cost tracking (token usage, API calls)

Architecture:
    - MetricsCollector: Central metrics aggregation
    - Tracer: Distributed tracing with spans
    - CostTracker: Token and API cost tracking
"""

import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"      # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values


@dataclass
class Metric:
    """Represents a metric measurement.
    
    Attributes:
        name: Metric name (e.g., "agent.tool_calls")
        type: Metric type (counter, gauge, histogram)
        value: Metric value
        tags: Metric labels/dimensions
        timestamp: Measurement time
    """
    name: str
    type: MetricType
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Span:
    """Represents a trace span for distributed tracing.
    
    Attributes:
        name: Operation name
        span_id: Unique span identifier
        parent_id: Parent span ID (for nested operations)
        start_time: Span start timestamp
        end_time: Span end timestamp
        duration: Operation duration in seconds
        metadata: Additional span data
        status: Span status (ok, error)
    """
    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    
    def finish(self, status: str = "ok") -> None:
        """Mark span as complete.
        
        Args:
            status: Span status ("ok" or "error")
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status


class MetricsCollector:
    """Centralized metrics collection and aggregation.
    
    Purpose:
        Track agent performance and behavior:
        - Operation counts (tool calls, LLM calls)
        - Latencies (agent runtime, tool execution time)
        - Resource usage (tokens, API calls)
    
    Features:
        - Multiple metric types
        - Tag-based filtering
        - Aggregation (sum, avg, percentiles)
        - Export to monitoring systems
    
    Reuse Patterns:
        - Monitoring: Track agent health
        - Optimization: Identify slow operations
        - Billing: Calculate API costs
        - Debugging: Understand system behavior
    
    Example:
        ```python
        collector = MetricsCollector()
        
        # Increment counter
        collector.increment("agent.tool_calls", tags={"tool": "web_search"})
        
        # Record gauge
        collector.gauge("agent.active_tasks", value=5)
        
        # Record histogram
        collector.histogram("agent.latency", value=1.25, tags={"agent": "analyst"})
        
        # Get metrics
        stats = collector.get_stats("agent.latency")
        print(f"Average latency: {stats['avg']:.2f}s")
        ```
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment amount (default 1.0)
            tags: Metric tags
        """
        metric_key = self._metric_key(name, tags)
        self.counters[metric_key] += value
        
        self.metrics.append(Metric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            tags=tags or {}
        ))
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric.
        
        Args:
            name: Metric name
            value: Current value
            tags: Metric tags
        """
        metric_key = self._metric_key(name, tags)
        self.gauges[metric_key] = value
        
        self.metrics.append(Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            tags=tags or {}
        ))
    
    def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value.
        
        Args:
            name: Metric name
            value: Measured value
            tags: Metric tags
        """
        metric_key = self._metric_key(name, tags)
        self.histograms[metric_key].append(value)
        
        self.metrics.append(Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            tags=tags or {}
        ))
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value.
        
        Args:
            name: Metric name
            tags: Metric tags
        
        Returns:
            Counter value
        """
        metric_key = self._metric_key(name, tags)
        return self.counters.get(metric_key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value.
        
        Args:
            name: Metric name
            tags: Metric tags
        
        Returns:
            Gauge value or None
        """
        metric_key = self._metric_key(name, tags)
        return self.gauges.get(metric_key)
    
    def get_stats(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get histogram statistics.
        
        Args:
            name: Metric name
            tags: Metric tags
        
        Returns:
            Statistics dict with count, sum, avg, min, max, p50, p95, p99
        """
        metric_key = self._metric_key(name, tags)
        values = self.histograms.get(metric_key, [])
        
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_values = sorted(values)
        count = len(values)
        
        return {
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": self._percentile(sorted_values, 0.50),
            "p95": self._percentile(sorted_values, 0.95),
            "p99": self._percentile(sorted_values, 0.99)
        }
    
    def _metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate metric key from name and tags.
        
        Args:
            name: Metric name
            tags: Metric tags
        
        Returns:
            Unique metric key
        """
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values.
        
        Args:
            sorted_values: Sorted list of values
            p: Percentile (0.0-1.0)
        
        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0
        
        idx = int(len(sorted_values) * p)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
    
    def reset(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()


class Tracer:
    """Distributed tracing for agent operations.
    
    Purpose:
        Track operation flow and timing:
        - Nested operations (agent -> task -> tool)
        - Latency measurement
        - Error tracking
        - Dependency visualization
    
    Features:
        - Parent-child span relationships
        - Context manager for automatic timing
        - Span metadata for debugging
    
    Reuse Patterns:
        - Performance: Identify slow operations
        - Debugging: Trace execution flow
        - Monitoring: Track error rates
    
    Example:
        ```python
        tracer = Tracer()
        
        # Manual span
        span = tracer.start_span("agent_run")
        try:
            # ... do work ...
            span.metadata["result"] = "success"
        finally:
            tracer.end_span(span)
        
        # Context manager
        with tracer.span("tool_call", metadata={"tool": "web_search"}):
            # ... execute tool ...
            pass
        
        # Get traces
        traces = tracer.get_traces()
        for span in traces:
            print(f"{span.name}: {span.duration:.3f}s")
        ```
    """
    
    def __init__(self):
        """Initialize tracer."""
        self.spans: List[Span] = []
        self.active_spans: Dict[str, Span] = {}
    
    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span.
        
        Args:
            name: Operation name
            parent_id: Parent span ID
            metadata: Additional span data
        
        Returns:
            New span
        """
        span = Span(
            name=name,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        self.active_spans[span.span_id] = span
        return span
    
    def end_span(self, span: Span, status: str = "ok") -> None:
        """End a span.
        
        Args:
            span: Span to end
            status: Span status ("ok" or "error")
        """
        span.finish(status=status)
        self.spans.append(span)
        self.active_spans.pop(span.span_id, None)
    
    @contextmanager
    def span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for automatic span timing.
        
        Args:
            name: Operation name
            parent_id: Parent span ID
            metadata: Additional span data
        
        Yields:
            Span object
        
        Example:
            ```python
            with tracer.span("llm_call") as span:
                response = llm.call(prompt)
                span.metadata["tokens"] = count_tokens(response)
            ```
        """
        span_obj = self.start_span(name, parent_id, metadata)
        try:
            yield span_obj
            self.end_span(span_obj, status="ok")
        except Exception as e:
            span_obj.metadata["error"] = str(e)
            self.end_span(span_obj, status="error")
            raise
    
    def get_traces(self, name: Optional[str] = None) -> List[Span]:
        """Get completed spans.
        
        Args:
            name: Filter by operation name
        
        Returns:
            List of spans
        """
        if name:
            return [s for s in self.spans if s.name == name]
        return self.spans.copy()
    
    def reset(self) -> None:
        """Clear all traces."""
        self.spans.clear()
        self.active_spans.clear()


class CostTracker:
    """Track API costs and token usage.
    
    Purpose:
        Monitor resource consumption:
        - Token usage (prompt, completion, total)
        - API call counts
        - Estimated costs
    
    Features:
        - Provider-specific pricing
        - Cost estimation
        - Usage aggregation
    
    Example:
        ```python
        tracker = CostTracker()
        tracker.set_price("gpt-4", prompt_cost=0.03, completion_cost=0.06)
        
        # Record usage
        tracker.record_usage(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        # Get costs
        print(f"Total cost: ${tracker.get_total_cost():.4f}")
        print(f"Total tokens: {tracker.get_total_tokens()}")
        ```
    """
    
    def __init__(self):
        """Initialize cost tracker."""
        self.usage: List[Dict[str, Any]] = []
        self.prices: Dict[str, Dict[str, float]] = {}
    
    def set_price(
        self,
        model: str,
        prompt_cost: float,
        completion_cost: float,
        per_tokens: int = 1000
    ) -> None:
        """Set pricing for a model.
        
        Args:
            model: Model name
            prompt_cost: Cost per prompt tokens
            completion_cost: Cost per completion tokens
            per_tokens: Pricing unit (default 1000 tokens)
        """
        self.prices[model] = {
            "prompt": prompt_cost,
            "completion": completion_cost,
            "per_tokens": per_tokens
        }
    
    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Record token usage.
        
        Args:
            model: Model name
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            metadata: Additional context
        
        Returns:
            Estimated cost for this usage
        """
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        cost = 0.0
        if model in self.prices:
            pricing = self.prices[model]
            per_tokens = pricing["per_tokens"]
            prompt_cost = (prompt_tokens / per_tokens) * pricing["prompt"]
            completion_cost = (completion_tokens / per_tokens) * pricing["completion"]
            cost = prompt_cost + completion_cost
        
        # Record usage
        usage = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.usage.append(usage)
        
        return cost
    
    def get_total_cost(self, model: Optional[str] = None) -> float:
        """Get total estimated cost.
        
        Args:
            model: Filter by model (None = all models)
        
        Returns:
            Total cost
        """
        filtered = self.usage
        if model:
            filtered = [u for u in filtered if u["model"] == model]
        
        return sum(u["cost"] for u in filtered)
    
    def get_total_tokens(self, model: Optional[str] = None) -> int:
        """Get total token usage.
        
        Args:
            model: Filter by model (None = all models)
        
        Returns:
            Total tokens
        """
        filtered = self.usage
        if model:
            filtered = [u for u in filtered if u["model"] == model]
        
        return sum(u["total_tokens"] for u in filtered)
    
    def get_usage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get usage summary by model.
        
        Returns:
            Dict of {model: {tokens, cost, calls}}
        """
        summary = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        
        for usage in self.usage:
            model = usage["model"]
            summary[model]["tokens"] += usage["total_tokens"]
            summary[model]["cost"] += usage["cost"]
            summary[model]["calls"] += 1
        
        return dict(summary)
    
    def reset(self) -> None:
        """Clear usage data."""
        self.usage.clear()


# Global instances
_global_metrics: Optional[MetricsCollector] = None
_global_tracer: Optional[Tracer] = None
_global_cost_tracker: Optional[CostTracker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector singleton."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_tracer() -> Tracer:
    """Get global tracer singleton."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker singleton."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
        # Set default OpenAI pricing (as of 2024)
        _global_cost_tracker.set_price("gpt-4", 0.03, 0.06)
        _global_cost_tracker.set_price("gpt-3.5-turbo", 0.001, 0.002)
    return _global_cost_tracker


__all__ = [
    "Metric",
    "MetricType",
    "Span",
    "MetricsCollector",
    "Tracer",
    "CostTracker",
    "get_metrics_collector",
    "get_tracer",
    "get_cost_tracker"
]
