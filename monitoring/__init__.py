"""
Monitoring Module - Phase 5: Continuous Learning

This module provides performance monitoring, regime detection,
degradation detection, and re-optimization triggers for the
IB Breakout Optimizer.
"""

from monitoring.performance_monitor import (
    PerformanceMonitor,
    RollingMetrics,
    TradeRecord,
    create_monitor_from_trades,
)
from monitoring.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeChange,
    VolatilityRegime,
    TrendRegime,
    CorrelationRegime,
)
from monitoring.degradation_detector import (
    DegradationDetector,
    DegradationAlert,
    DegradationThresholds,
    AlertType,
    AlertSeverity,
)
from monitoring.reoptimization_trigger import (
    ReoptimizationTrigger,
    ReoptimizationRequest,
    TriggerConfig,
    TriggerReason,
)
from monitoring.data_updater import (
    DataUpdater,
    DataUpdateConfig,
    UpdateResult,
)

__all__ = [
    # Performance Monitor
    "PerformanceMonitor",
    "RollingMetrics",
    "TradeRecord",
    "create_monitor_from_trades",
    # Regime Detector
    "RegimeDetector",
    "MarketRegime",
    "RegimeChange",
    "VolatilityRegime",
    "TrendRegime",
    "CorrelationRegime",
    # Degradation Detector
    "DegradationDetector",
    "DegradationAlert",
    "DegradationThresholds",
    "AlertType",
    "AlertSeverity",
    # Re-optimization Trigger
    "ReoptimizationTrigger",
    "ReoptimizationRequest",
    "TriggerConfig",
    "TriggerReason",
    # Data Updater
    "DataUpdater",
    "DataUpdateConfig",
    "UpdateResult",
]
