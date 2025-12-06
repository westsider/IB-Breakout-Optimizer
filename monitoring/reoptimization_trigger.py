"""
Re-optimization Trigger - Queue re-optimization when degradation detected.

Monitors degradation alerts and regime changes to determine when
strategy parameters should be re-optimized.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Callable
from enum import Enum
import json
import os

from monitoring.degradation_detector import (
    DegradationDetector, DegradationAlert, AlertType, AlertSeverity
)
from monitoring.regime_detector import RegimeDetector, MarketRegime


class TriggerReason(Enum):
    """Reasons for triggering re-optimization."""
    CRITICAL_DEGRADATION = "critical_degradation"
    SUSTAINED_POOR_PERFORMANCE = "sustained_poor_performance"
    REGIME_SHIFT = "regime_shift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA = "new_data"


@dataclass
class TriggerConfig:
    """Configuration for re-optimization triggers."""
    # Degradation triggers
    critical_alert_threshold: int = 2  # Trigger if N critical alerts
    warning_alert_threshold: int = 5  # Trigger if N warning alerts
    sustained_degradation_days: int = 5  # Trigger after N days of degradation

    # Regime triggers
    volatility_regime_change: bool = True  # Trigger on vol regime change
    trend_regime_change: bool = False  # Trigger on trend regime change

    # Scheduled triggers
    scheduled_interval_days: int = 30  # Re-optimize every N days
    min_trades_between_reopt: int = 20  # Minimum trades before re-optimizing

    # Data triggers
    new_data_threshold_days: int = 7  # Re-optimize after N days of new data

    # Cooldown
    cooldown_hours: int = 24  # Minimum hours between re-optimizations


@dataclass
class ReoptimizationRequest:
    """Request for re-optimization."""
    timestamp: datetime
    reason: TriggerReason
    priority: int  # 1-5, 5 being most urgent
    ticker: str
    details: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason.value,
            'priority': self.priority,
            'ticker': self.ticker,
            'details': self.details,
            'status': self.status,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
        }


class ReoptimizationTrigger:
    """
    Manage re-optimization triggers based on performance and market conditions.

    Monitors degradation detector and regime detector to determine when
    parameters should be re-optimized, then queues requests.
    """

    def __init__(
        self,
        degradation_detector: DegradationDetector,
        regime_detector: Optional[RegimeDetector] = None,
        config: Optional[TriggerConfig] = None,
        queue_file: Optional[str] = None,
    ):
        """
        Initialize the re-optimization trigger.

        Args:
            degradation_detector: Detector providing performance alerts
            regime_detector: Optional detector for market regimes
            config: Trigger configuration
            queue_file: Optional file to persist queue
        """
        self.degradation_detector = degradation_detector
        self.regime_detector = regime_detector
        self.config = config or TriggerConfig()
        self.queue_file = queue_file

        # Request queue
        self.pending_requests: List[ReoptimizationRequest] = []
        self.completed_requests: List[ReoptimizationRequest] = []

        # Tracking
        self.last_reopt_time: Optional[datetime] = None
        self.trades_since_reopt: int = 0
        self.degradation_start: Optional[datetime] = None
        self._last_regime: Optional[MarketRegime] = None

        # Callbacks
        self._trigger_callbacks: List[Callable[[ReoptimizationRequest], None]] = []

        # Load persisted queue
        if queue_file and os.path.exists(queue_file):
            self._load_queue()

    def register_callback(self, callback: Callable[[ReoptimizationRequest], None]):
        """Register callback for new re-optimization requests."""
        self._trigger_callbacks.append(callback)

    def record_trade(self, ticker: str = ""):
        """Record that a trade occurred (for counting between re-opts)."""
        self.trades_since_reopt += 1

    def check_triggers(self, ticker: str = "") -> Optional[ReoptimizationRequest]:
        """
        Check if re-optimization should be triggered.

        Args:
            ticker: Ticker symbol to check

        Returns:
            ReoptimizationRequest if triggered, None otherwise
        """
        # Check cooldown
        if self.last_reopt_time:
            hours_since = (datetime.now() - self.last_reopt_time).total_seconds() / 3600
            if hours_since < self.config.cooldown_hours:
                return None

        # Check minimum trades
        if self.trades_since_reopt < self.config.min_trades_between_reopt:
            return None

        request = None

        # Check critical degradation
        request = self._check_critical_degradation(ticker)
        if request:
            return self._queue_request(request)

        # Check sustained degradation
        request = self._check_sustained_degradation(ticker)
        if request:
            return self._queue_request(request)

        # Check regime shift
        request = self._check_regime_shift(ticker)
        if request:
            return self._queue_request(request)

        # Check scheduled re-optimization
        request = self._check_scheduled(ticker)
        if request:
            return self._queue_request(request)

        return None

    def _check_critical_degradation(self, ticker: str) -> Optional[ReoptimizationRequest]:
        """Check for critical performance degradation."""
        active_alerts = self.degradation_detector.get_active_alerts()

        critical_count = sum(
            1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL
        )
        warning_count = sum(
            1 for a in active_alerts if a.severity == AlertSeverity.WARNING
        )

        if critical_count >= self.config.critical_alert_threshold:
            return ReoptimizationRequest(
                timestamp=datetime.now(),
                reason=TriggerReason.CRITICAL_DEGRADATION,
                priority=5,
                ticker=ticker,
                details={
                    'critical_alerts': critical_count,
                    'warning_alerts': warning_count,
                    'alerts': [a.message for a in active_alerts[:5]],
                },
            )

        if warning_count >= self.config.warning_alert_threshold:
            return ReoptimizationRequest(
                timestamp=datetime.now(),
                reason=TriggerReason.CRITICAL_DEGRADATION,
                priority=4,
                ticker=ticker,
                details={
                    'warning_alerts': warning_count,
                    'alerts': [a.message for a in active_alerts[:5]],
                },
            )

        return None

    def _check_sustained_degradation(self, ticker: str) -> Optional[ReoptimizationRequest]:
        """Check for sustained poor performance over time."""
        health = self.degradation_detector.get_health_status()

        if health['status'] in ['degraded', 'critical']:
            if self.degradation_start is None:
                self.degradation_start = datetime.now()
            else:
                days_degraded = (datetime.now() - self.degradation_start).days
                if days_degraded >= self.config.sustained_degradation_days:
                    return ReoptimizationRequest(
                        timestamp=datetime.now(),
                        reason=TriggerReason.SUSTAINED_POOR_PERFORMANCE,
                        priority=3,
                        ticker=ticker,
                        details={
                            'days_degraded': days_degraded,
                            'health_status': health['status'],
                            'health_score': health['score'],
                        },
                    )
        else:
            # Performance recovered
            self.degradation_start = None

        return None

    def _check_regime_shift(self, ticker: str) -> Optional[ReoptimizationRequest]:
        """Check for significant market regime changes."""
        if not self.regime_detector or not self.regime_detector.current_regime:
            return None

        current = self.regime_detector.current_regime

        if self._last_regime is None:
            self._last_regime = current
            return None

        # Check for significant regime changes
        vol_changed = (
            self.config.volatility_regime_change and
            current.volatility != self._last_regime.volatility
        )
        trend_changed = (
            self.config.trend_regime_change and
            current.trend != self._last_regime.trend
        )

        if vol_changed or trend_changed:
            changes = []
            if vol_changed:
                changes.append(
                    f"Volatility: {self._last_regime.volatility.value} → {current.volatility.value}"
                )
            if trend_changed:
                changes.append(
                    f"Trend: {self._last_regime.trend.value} → {current.trend.value}"
                )

            self._last_regime = current

            return ReoptimizationRequest(
                timestamp=datetime.now(),
                reason=TriggerReason.REGIME_SHIFT,
                priority=2,
                ticker=ticker,
                details={
                    'changes': changes,
                    'current_regime': current.summary(),
                },
            )

        self._last_regime = current
        return None

    def _check_scheduled(self, ticker: str) -> Optional[ReoptimizationRequest]:
        """Check if scheduled re-optimization is due."""
        if self.last_reopt_time is None:
            return None

        days_since = (datetime.now() - self.last_reopt_time).days

        if days_since >= self.config.scheduled_interval_days:
            return ReoptimizationRequest(
                timestamp=datetime.now(),
                reason=TriggerReason.SCHEDULED,
                priority=1,
                ticker=ticker,
                details={
                    'days_since_last': days_since,
                    'interval': self.config.scheduled_interval_days,
                },
            )

        return None

    def trigger_new_data(self, ticker: str, data_days: int) -> Optional[ReoptimizationRequest]:
        """
        Manually trigger re-optimization due to new data.

        Args:
            ticker: Ticker symbol
            data_days: Number of new data days

        Returns:
            ReoptimizationRequest if threshold met
        """
        if data_days >= self.config.new_data_threshold_days:
            return self._queue_request(ReoptimizationRequest(
                timestamp=datetime.now(),
                reason=TriggerReason.NEW_DATA,
                priority=2,
                ticker=ticker,
                details={
                    'new_data_days': data_days,
                    'threshold': self.config.new_data_threshold_days,
                },
            ))
        return None

    def trigger_manual(self, ticker: str, reason: str = "") -> ReoptimizationRequest:
        """
        Manually trigger re-optimization.

        Args:
            ticker: Ticker symbol
            reason: Optional reason string

        Returns:
            ReoptimizationRequest
        """
        return self._queue_request(ReoptimizationRequest(
            timestamp=datetime.now(),
            reason=TriggerReason.MANUAL,
            priority=3,
            ticker=ticker,
            details={'manual_reason': reason},
        ))

    def _queue_request(self, request: ReoptimizationRequest) -> ReoptimizationRequest:
        """Add request to queue and notify callbacks."""
        self.pending_requests.append(request)

        # Sort by priority (highest first)
        self.pending_requests.sort(key=lambda r: r.priority, reverse=True)

        # Persist queue
        if self.queue_file:
            self._save_queue()

        # Notify callbacks
        for callback in self._trigger_callbacks:
            try:
                callback(request)
            except Exception:
                pass

        return request

    def get_next_request(self) -> Optional[ReoptimizationRequest]:
        """Get next pending request (highest priority)."""
        if self.pending_requests:
            return self.pending_requests[0]
        return None

    def start_request(self, request: ReoptimizationRequest):
        """Mark a request as running."""
        request.status = "running"
        if self.queue_file:
            self._save_queue()

    def complete_request(
        self,
        request: ReoptimizationRequest,
        success: bool = True,
        result: Dict = None
    ):
        """Mark a request as completed."""
        request.status = "completed" if success else "failed"
        request.completed_at = datetime.now()
        request.result = result

        # Move to completed
        if request in self.pending_requests:
            self.pending_requests.remove(request)
        self.completed_requests.append(request)

        # Update tracking
        if success:
            self.last_reopt_time = datetime.now()
            self.trades_since_reopt = 0
            self.degradation_start = None

        # Persist
        if self.queue_file:
            self._save_queue()

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self.pending_requests)

    def get_queue_summary(self) -> Dict:
        """Get summary of re-optimization queue."""
        return {
            'pending_count': len(self.pending_requests),
            'pending_requests': [r.to_dict() for r in self.pending_requests],
            'recent_completed': [
                r.to_dict() for r in self.completed_requests[-5:]
            ],
            'last_reopt': self.last_reopt_time.isoformat() if self.last_reopt_time else None,
            'trades_since_reopt': self.trades_since_reopt,
        }

    def _save_queue(self):
        """Persist queue to file."""
        try:
            data = {
                'pending': [r.to_dict() for r in self.pending_requests],
                'completed': [r.to_dict() for r in self.completed_requests[-20:]],
                'last_reopt': self.last_reopt_time.isoformat() if self.last_reopt_time else None,
                'trades_since_reopt': self.trades_since_reopt,
            }
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail on save errors

    def _load_queue(self):
        """Load queue from file."""
        try:
            with open(self.queue_file, 'r') as f:
                data = json.load(f)

            if data.get('last_reopt'):
                self.last_reopt_time = datetime.fromisoformat(data['last_reopt'])
            self.trades_since_reopt = data.get('trades_since_reopt', 0)

            # Note: We don't restore pending/completed requests on load
            # to avoid stale requests
        except Exception:
            pass


if __name__ == "__main__":
    # Test the re-optimization trigger
    import random
    from monitoring.performance_monitor import PerformanceMonitor, TradeRecord

    print("Testing Re-optimization Trigger")
    print("=" * 50)

    # Create components
    perf_monitor = PerformanceMonitor(windows=[10, 20])
    regime_detector = RegimeDetector()
    degradation_detector = DegradationDetector(perf_monitor, regime_detector)

    config = TriggerConfig(
        critical_alert_threshold=2,
        sustained_degradation_days=3,
        min_trades_between_reopt=10,
        cooldown_hours=1,
    )

    trigger = ReoptimizationTrigger(
        degradation_detector,
        regime_detector,
        config=config,
    )

    # Callback for trigger events
    def on_trigger(request: ReoptimizationRequest):
        print(f"\n  *** RE-OPTIMIZATION TRIGGERED ***")
        print(f"  Reason: {request.reason.value}")
        print(f"  Priority: {request.priority}")
        print(f"  Details: {request.details}")

    trigger.register_callback(on_trigger)

    # Generate trades that cause degradation
    random.seed(42)
    print("\nAdding trades with degrading performance...")

    for i in range(30):
        # Start good, end bad
        if i < 15:
            win_rate = 0.60
            pnl_win = (100, 300)
            pnl_loss = (-100, -50)
        else:
            win_rate = 0.30
            pnl_win = (50, 100)
            pnl_loss = (-200, -100)

        is_win = random.random() < win_rate
        pnl = random.uniform(*pnl_win) if is_win else random.uniform(*pnl_loss)

        trade = TradeRecord(pnl=pnl, timestamp=datetime.now())
        perf_monitor.add_trade(trade)
        trigger.record_trade("TSLA")

        # Check for degradation and triggers
        degradation_detector.check_degradation()
        request = trigger.check_triggers("TSLA")

        if request:
            print(f"\nTrade {i + 1}: P&L ${pnl:.2f}")

    print("\n\nQueue Summary:")
    summary = trigger.get_queue_summary()
    print(f"  Pending requests: {summary['pending_count']}")
    print(f"  Trades since last re-opt: {summary['trades_since_reopt']}")

    if summary['pending_requests']:
        print("\n  Pending:")
        for req in summary['pending_requests']:
            print(f"    - {req['reason']} (priority {req['priority']})")
