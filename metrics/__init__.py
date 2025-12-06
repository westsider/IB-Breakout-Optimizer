"""
Metrics module for IB Breakout Optimizer.

Provides performance analysis and reporting.
"""

from .performance_metrics import PerformanceMetrics, calculate_metrics

__all__ = [
    "PerformanceMetrics",
    "calculate_metrics",
]
