"""
Degradation Detector - Identify performance deterioration.

Monitors rolling metrics against configurable thresholds to detect
when strategy performance has degraded significantly.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Callable
from enum import Enum

from monitoring.performance_monitor import PerformanceMonitor, RollingMetrics
from monitoring.regime_detector import RegimeDetector, MarketRegime


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of degradation alerts."""
    SHARPE_LOW = "sharpe_low"
    WIN_RATE_LOW = "win_rate_low"
    PROFIT_FACTOR_LOW = "profit_factor_low"
    DRAWDOWN_HIGH = "drawdown_high"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    REGIME_CHANGE = "regime_change"
    PERFORMANCE_DIVERGENCE = "performance_divergence"


@dataclass
class DegradationThresholds:
    """Configurable thresholds for degradation detection."""
    min_sharpe: float = 0.5  # Minimum acceptable Sharpe ratio
    min_win_rate: float = 40.0  # Minimum win rate (%)
    min_profit_factor: float = 1.0  # Minimum profit factor
    max_drawdown: float = 10.0  # Maximum drawdown (% or $)
    max_consecutive_losses: int = 5  # Max losing streak

    # Divergence thresholds (recent vs long-term)
    win_rate_divergence: float = 10.0  # Alert if recent win rate is X% below long-term
    sharpe_divergence: float = 1.0  # Alert if recent Sharpe is X below long-term
    pf_divergence: float = 0.5  # Alert if recent PF is X below long-term


@dataclass
class DegradationAlert:
    """Record of a detected degradation."""
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    window_size: int = 0
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'window_size': self.window_size,
            'acknowledged': self.acknowledged,
        }


class DegradationDetector:
    """
    Detect performance degradation using rolling metrics.

    Monitors performance against configurable thresholds and
    generates alerts when degradation is detected.
    """

    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        regime_detector: Optional[RegimeDetector] = None,
        thresholds: Optional[DegradationThresholds] = None,
    ):
        """
        Initialize the degradation detector.

        Args:
            performance_monitor: Monitor providing rolling metrics
            regime_detector: Optional regime detector for context
            thresholds: Degradation thresholds (uses defaults if None)
        """
        self.performance_monitor = performance_monitor
        self.regime_detector = regime_detector
        self.thresholds = thresholds or DegradationThresholds()

        # Alert history
        self.alerts: List[DegradationAlert] = []
        self.active_alerts: Dict[AlertType, DegradationAlert] = {}

        # Callback for real-time notifications
        self._alert_callbacks: List[Callable[[DegradationAlert], None]] = []

        # Track last regime for change detection
        self._last_regime: Optional[MarketRegime] = None

    def register_callback(self, callback: Callable[[DegradationAlert], None]):
        """Register a callback to be notified of new alerts."""
        self._alert_callbacks.append(callback)

    def check_degradation(self) -> List[DegradationAlert]:
        """
        Check current metrics against thresholds.

        Returns:
            List of new alerts detected
        """
        new_alerts = []

        # Check each window size
        for window in self.performance_monitor.windows:
            metrics = self.performance_monitor.get_current_metrics(window)

            if metrics.total_trades < 5:
                continue  # Not enough trades for meaningful check

            # Check Sharpe ratio
            if metrics.sharpe_ratio < self.thresholds.min_sharpe:
                alert = self._create_alert(
                    AlertType.SHARPE_LOW,
                    self._get_severity_sharpe(metrics.sharpe_ratio),
                    f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) below threshold "
                    f"({self.thresholds.min_sharpe:.2f}) over {window} trades",
                    metrics.sharpe_ratio,
                    self.thresholds.min_sharpe,
                    window,
                )
                new_alerts.append(alert)

            # Check win rate
            if metrics.win_rate < self.thresholds.min_win_rate:
                alert = self._create_alert(
                    AlertType.WIN_RATE_LOW,
                    self._get_severity_win_rate(metrics.win_rate),
                    f"Win rate ({metrics.win_rate:.1f}%) below threshold "
                    f"({self.thresholds.min_win_rate:.1f}%) over {window} trades",
                    metrics.win_rate,
                    self.thresholds.min_win_rate,
                    window,
                )
                new_alerts.append(alert)

            # Check profit factor
            if metrics.profit_factor < self.thresholds.min_profit_factor:
                alert = self._create_alert(
                    AlertType.PROFIT_FACTOR_LOW,
                    self._get_severity_pf(metrics.profit_factor),
                    f"Profit factor ({metrics.profit_factor:.2f}) below threshold "
                    f"({self.thresholds.min_profit_factor:.2f}) over {window} trades",
                    metrics.profit_factor,
                    self.thresholds.min_profit_factor,
                    window,
                )
                new_alerts.append(alert)

            # Check drawdown
            if metrics.max_drawdown > self.thresholds.max_drawdown:
                alert = self._create_alert(
                    AlertType.DRAWDOWN_HIGH,
                    self._get_severity_drawdown(metrics.max_drawdown),
                    f"Max drawdown (${metrics.max_drawdown:.2f}) exceeds threshold "
                    f"(${self.thresholds.max_drawdown:.2f}) over {window} trades",
                    metrics.max_drawdown,
                    self.thresholds.max_drawdown,
                    window,
                )
                new_alerts.append(alert)

            # Check consecutive losses
            if metrics.consecutive_losses >= self.thresholds.max_consecutive_losses:
                alert = self._create_alert(
                    AlertType.CONSECUTIVE_LOSSES,
                    AlertSeverity.CRITICAL if metrics.consecutive_losses >= 7
                    else AlertSeverity.WARNING,
                    f"{metrics.consecutive_losses} consecutive losses detected",
                    metrics.consecutive_losses,
                    self.thresholds.max_consecutive_losses,
                    window,
                )
                new_alerts.append(alert)

        # Check for performance divergence (recent vs long-term)
        comparison = self.performance_monitor.get_metrics_comparison()
        if comparison and comparison.get('is_degrading'):
            # Win rate divergence
            if comparison.get('win_rate_diff', 0) < -self.thresholds.win_rate_divergence:
                alert = self._create_alert(
                    AlertType.PERFORMANCE_DIVERGENCE,
                    AlertSeverity.WARNING,
                    f"Recent win rate ({comparison['recent_window']} trades) is "
                    f"{abs(comparison['win_rate_diff']):.1f}% below longer-term average",
                    comparison['win_rate_diff'],
                    -self.thresholds.win_rate_divergence,
                )
                new_alerts.append(alert)

            # Sharpe divergence
            if comparison.get('sharpe_diff', 0) < -self.thresholds.sharpe_divergence:
                alert = self._create_alert(
                    AlertType.PERFORMANCE_DIVERGENCE,
                    AlertSeverity.WARNING,
                    f"Recent Sharpe ratio is {abs(comparison['sharpe_diff']):.2f} "
                    f"below longer-term average",
                    comparison['sharpe_diff'],
                    -self.thresholds.sharpe_divergence,
                )
                new_alerts.append(alert)

        # Check for regime change
        if self.regime_detector and self.regime_detector.current_regime:
            current_regime = self.regime_detector.current_regime
            if self._last_regime is not None:
                regime_changed = (
                    current_regime.volatility != self._last_regime.volatility or
                    current_regime.trend != self._last_regime.trend
                )
                if regime_changed:
                    alert = self._create_alert(
                        AlertType.REGIME_CHANGE,
                        AlertSeverity.INFO,
                        f"Market regime changed: Vol {self._last_regime.volatility.value} → "
                        f"{current_regime.volatility.value}, "
                        f"Trend {self._last_regime.trend.value} → {current_regime.trend.value}",
                        0,
                        0,
                    )
                    new_alerts.append(alert)
            self._last_regime = current_regime

        # Store alerts and notify callbacks
        for alert in new_alerts:
            self.alerts.append(alert)
            self.active_alerts[alert.alert_type] = alert
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass  # Don't let callback errors break the detector

        return new_alerts

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        current_value: float,
        threshold_value: float,
        window_size: int = 0,
    ) -> DegradationAlert:
        """Create a degradation alert."""
        return DegradationAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            window_size=window_size,
        )

    def _get_severity_sharpe(self, sharpe: float) -> AlertSeverity:
        """Determine severity based on Sharpe value."""
        if sharpe < 0:
            return AlertSeverity.CRITICAL
        elif sharpe < 0.3:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _get_severity_win_rate(self, win_rate: float) -> AlertSeverity:
        """Determine severity based on win rate."""
        if win_rate < 30:
            return AlertSeverity.CRITICAL
        elif win_rate < 35:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _get_severity_pf(self, pf: float) -> AlertSeverity:
        """Determine severity based on profit factor."""
        if pf < 0.5:
            return AlertSeverity.CRITICAL
        elif pf < 0.8:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _get_severity_drawdown(self, dd: float) -> AlertSeverity:
        """Determine severity based on drawdown."""
        if dd > 20:
            return AlertSeverity.CRITICAL
        elif dd > 15:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def acknowledge_alert(self, alert_type: AlertType):
        """Acknowledge an active alert."""
        if alert_type in self.active_alerts:
            self.active_alerts[alert_type].acknowledged = True

    def get_active_alerts(self, unacknowledged_only: bool = False) -> List[DegradationAlert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return sorted(alerts, key=lambda a: a.severity.value, reverse=True)

    def get_alert_history(self, days: int = 30) -> List[DegradationAlert]:
        """Get alert history for the past N days."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        return [a for a in self.alerts if a.timestamp > cutoff]

    def clear_resolved_alerts(self):
        """Clear alerts that are no longer triggered."""
        current_alerts = set()

        # Re-check all conditions to see what's still active
        for window in self.performance_monitor.windows:
            metrics = self.performance_monitor.get_current_metrics(window)

            if metrics.sharpe_ratio < self.thresholds.min_sharpe:
                current_alerts.add(AlertType.SHARPE_LOW)
            if metrics.win_rate < self.thresholds.min_win_rate:
                current_alerts.add(AlertType.WIN_RATE_LOW)
            if metrics.profit_factor < self.thresholds.min_profit_factor:
                current_alerts.add(AlertType.PROFIT_FACTOR_LOW)
            if metrics.max_drawdown > self.thresholds.max_drawdown:
                current_alerts.add(AlertType.DRAWDOWN_HIGH)
            if metrics.consecutive_losses >= self.thresholds.max_consecutive_losses:
                current_alerts.add(AlertType.CONSECUTIVE_LOSSES)

        # Remove resolved alerts
        resolved = set(self.active_alerts.keys()) - current_alerts
        for alert_type in resolved:
            del self.active_alerts[alert_type]

    def get_health_status(self) -> Dict:
        """
        Get overall strategy health status.

        Returns:
            Dict with health score and details
        """
        metrics = self.performance_monitor.get_current_metrics()
        active_alerts = self.get_active_alerts()

        # Calculate health score (0-100)
        score = 100

        # Deduct for active alerts
        for alert in active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                score -= 30
            elif alert.severity == AlertSeverity.WARNING:
                score -= 15
            else:
                score -= 5

        # Bonus/penalty based on metrics
        if metrics.sharpe_ratio > 1.5:
            score += 10
        if metrics.win_rate > 55:
            score += 5
        if metrics.profit_factor > 2.0:
            score += 5

        score = max(0, min(100, score))

        # Determine status
        if score >= 80:
            status = "healthy"
        elif score >= 60:
            status = "warning"
        elif score >= 40:
            status = "degraded"
        else:
            status = "critical"

        return {
            'score': score,
            'status': status,
            'active_alerts': len(active_alerts),
            'critical_alerts': sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL),
            'current_sharpe': metrics.sharpe_ratio,
            'current_win_rate': metrics.win_rate,
            'current_pf': metrics.profit_factor,
        }

    def export_status(self) -> Dict:
        """Export full degradation status."""
        return {
            'health': self.get_health_status(),
            'thresholds': {
                'min_sharpe': self.thresholds.min_sharpe,
                'min_win_rate': self.thresholds.min_win_rate,
                'min_profit_factor': self.thresholds.min_profit_factor,
                'max_drawdown': self.thresholds.max_drawdown,
                'max_consecutive_losses': self.thresholds.max_consecutive_losses,
            },
            'active_alerts': [a.to_dict() for a in self.get_active_alerts()],
            'recent_alerts': [a.to_dict() for a in self.get_alert_history(7)],
        }


if __name__ == "__main__":
    # Test the degradation detector
    import random
    from monitoring.performance_monitor import TradeRecord

    print("Testing Degradation Detector")
    print("=" * 50)

    # Create performance monitor
    monitor = PerformanceMonitor(windows=[10, 20, 50])

    # Create degradation detector
    thresholds = DegradationThresholds(
        min_sharpe=0.5,
        min_win_rate=45.0,
        max_consecutive_losses=4,
    )
    detector = DegradationDetector(monitor, thresholds=thresholds)

    # Add callback for alerts
    def on_alert(alert: DegradationAlert):
        print(f"  ALERT: [{alert.severity.value}] {alert.message}")

    detector.register_callback(on_alert)

    # Generate some trades - start good, then degrade
    random.seed(42)

    print("\nPhase 1: Good performance")
    for i in range(30):
        # 60% win rate, good R:R
        is_win = random.random() < 0.60
        pnl = random.uniform(100, 300) if is_win else random.uniform(-100, -50)

        trade = TradeRecord(pnl=pnl, timestamp=datetime.now())
        monitor.add_trade(trade)

    alerts = detector.check_degradation()
    print(f"  Alerts: {len(alerts)}")

    print("\nPhase 2: Degrading performance")
    for i in range(20):
        # 35% win rate, bad R:R
        is_win = random.random() < 0.35
        pnl = random.uniform(50, 100) if is_win else random.uniform(-150, -50)

        trade = TradeRecord(pnl=pnl, timestamp=datetime.now())
        monitor.add_trade(trade)

    alerts = detector.check_degradation()
    print(f"  New alerts: {len(alerts)}")

    print("\nPhase 3: Consecutive losses")
    for i in range(6):
        trade = TradeRecord(pnl=random.uniform(-100, -50), timestamp=datetime.now())
        monitor.add_trade(trade)

    alerts = detector.check_degradation()

    print(f"\nHealth Status:")
    health = detector.get_health_status()
    for key, value in health.items():
        print(f"  {key}: {value}")

    print(f"\nActive Alerts:")
    for alert in detector.get_active_alerts():
        print(f"  [{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
