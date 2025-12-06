"""
Performance Monitor - Track rolling performance metrics.

Calculates key metrics over sliding windows to detect recent
performance changes that all-time metrics might mask.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from collections import deque
import numpy as np


@dataclass
class RollingMetrics:
    """Rolling performance metrics over a window of trades."""

    window_size: int
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    avg_pnl: float
    total_pnl: float
    max_drawdown: float
    consecutive_losses: int
    consecutive_wins: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'window_size': self.window_size,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'avg_pnl': self.avg_pnl,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TradeRecord:
    """Simple trade record for monitoring."""
    pnl: float
    timestamp: datetime
    ticker: str = ""
    direction: str = ""  # "long" or "short"
    entry_price: float = 0.0
    exit_price: float = 0.0


class PerformanceMonitor:
    """
    Monitor rolling performance metrics over sliding windows.

    Tracks metrics like Sharpe ratio, win rate, and profit factor
    over configurable windows (e.g., last 20, 50, 100 trades) to
    detect performance degradation early.
    """

    def __init__(
        self,
        windows: List[int] = None,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the performance monitor.

        Args:
            windows: List of window sizes to track (default: [20, 50, 100])
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.windows = windows or [20, 50, 100]
        self.risk_free_rate = risk_free_rate

        # Store all trades
        self.trades: List[TradeRecord] = []

        # Track rolling metrics history for each window
        self.metrics_history: Dict[int, List[RollingMetrics]] = {
            w: [] for w in self.windows
        }

        # Cache for quick consecutive win/loss tracking
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._current_streak_type: Optional[str] = None  # "win" or "loss"

    def add_trade(self, trade: TradeRecord) -> Dict[int, RollingMetrics]:
        """
        Add a trade and recalculate rolling metrics.

        Args:
            trade: Trade record to add

        Returns:
            Dict mapping window size to updated RollingMetrics
        """
        self.trades.append(trade)

        # Update consecutive tracking
        if trade.pnl >= 0:
            if self._current_streak_type == "win":
                self._consecutive_wins += 1
            else:
                self._consecutive_wins = 1
                self._consecutive_losses = 0
            self._current_streak_type = "win"
        else:
            if self._current_streak_type == "loss":
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 1
                self._consecutive_wins = 0
            self._current_streak_type = "loss"

        # Calculate metrics for each window
        results = {}
        for window in self.windows:
            metrics = self._calculate_rolling_metrics(window)
            results[window] = metrics
            self.metrics_history[window].append(metrics)

        return results

    def add_trades_bulk(self, trades: List[TradeRecord]) -> Dict[int, RollingMetrics]:
        """
        Add multiple trades at once.

        Args:
            trades: List of trade records

        Returns:
            Final rolling metrics after all trades added
        """
        for trade in trades:
            results = self.add_trade(trade)
        return results

    def _calculate_rolling_metrics(self, window: int) -> RollingMetrics:
        """
        Calculate metrics over the last N trades.

        Args:
            window: Number of recent trades to include

        Returns:
            RollingMetrics for the window
        """
        # Get recent trades
        recent_trades = self.trades[-window:] if len(self.trades) >= window else self.trades

        if not recent_trades:
            return RollingMetrics(
                window_size=window,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                avg_pnl=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                consecutive_losses=0,
                consecutive_wins=0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
            )

        pnls = [t.pnl for t in recent_trades]

        # Basic stats
        total_trades = len(pnls)
        winners = [p for p in pnls if p >= 0]
        losers = [p for p in pnls if p < 0]

        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        if len(pnls) > 1:
            pnl_std = np.std(pnls)
            if pnl_std > 0:
                # Approximate annualization (252 trading days)
                daily_rf = self.risk_free_rate / 252
                excess_return = np.mean(pnls) - daily_rf
                sharpe_ratio = (excess_return / pnl_std) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Averages
        avg_pnl = np.mean(pnls)
        total_pnl = sum(pnls)
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        largest_win = max(winners) if winners else 0
        largest_loss = min(losers) if losers else 0

        return RollingMetrics(
            window_size=window,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 99.99,
            sharpe_ratio=sharpe_ratio,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            consecutive_losses=self._consecutive_losses,
            consecutive_wins=self._consecutive_wins,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )

    def get_current_metrics(self, window: int = None) -> RollingMetrics:
        """
        Get current rolling metrics for a specific window.

        Args:
            window: Window size (uses smallest if not specified)

        Returns:
            Current RollingMetrics
        """
        if window is None:
            window = min(self.windows)
        return self._calculate_rolling_metrics(window)

    def get_all_current_metrics(self) -> Dict[int, RollingMetrics]:
        """Get current metrics for all windows."""
        return {w: self._calculate_rolling_metrics(w) for w in self.windows}

    def get_metrics_comparison(self) -> Dict[str, Dict]:
        """
        Compare metrics across different windows.

        Useful for detecting divergence between recent and longer-term performance.

        Returns:
            Dict with comparison data
        """
        metrics = self.get_all_current_metrics()

        if len(self.windows) < 2:
            return {}

        smallest = min(self.windows)
        largest = max(self.windows)

        recent = metrics[smallest]
        longer = metrics[largest]

        return {
            'recent_window': smallest,
            'longer_window': largest,
            'win_rate_diff': recent.win_rate - longer.win_rate,
            'sharpe_diff': recent.sharpe_ratio - longer.sharpe_ratio,
            'pf_diff': recent.profit_factor - longer.profit_factor,
            'avg_pnl_diff': recent.avg_pnl - longer.avg_pnl,
            'is_degrading': (
                recent.win_rate < longer.win_rate - 5 or
                recent.sharpe_ratio < longer.sharpe_ratio - 0.5 or
                recent.profit_factor < longer.profit_factor - 0.3
            ),
        }

    def get_metrics_history(self, window: int) -> List[RollingMetrics]:
        """Get historical metrics for a window."""
        return self.metrics_history.get(window, [])

    def clear(self):
        """Clear all trade data and metrics."""
        self.trades = []
        self.metrics_history = {w: [] for w in self.windows}
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._current_streak_type = None

    def get_trade_count(self) -> int:
        """Get total number of trades tracked."""
        return len(self.trades)

    def export_metrics(self) -> Dict:
        """Export all current metrics as a dictionary."""
        return {
            'total_trades': len(self.trades),
            'windows': {
                w: self._calculate_rolling_metrics(w).to_dict()
                for w in self.windows
            },
            'comparison': self.get_metrics_comparison(),
        }


# Convenience function to create monitor from trade list
def create_monitor_from_trades(
    trades: List[Dict],
    windows: List[int] = None
) -> PerformanceMonitor:
    """
    Create a PerformanceMonitor from a list of trade dictionaries.

    Args:
        trades: List of trade dicts with 'pnl' and optionally 'timestamp', 'ticker', etc.
        windows: Window sizes to track

    Returns:
        Initialized PerformanceMonitor
    """
    monitor = PerformanceMonitor(windows=windows)

    for t in trades:
        record = TradeRecord(
            pnl=t.get('pnl', 0),
            timestamp=t.get('timestamp', datetime.now()),
            ticker=t.get('ticker', ''),
            direction=t.get('direction', ''),
            entry_price=t.get('entry_price', 0),
            exit_price=t.get('exit_price', 0),
        )
        monitor.add_trade(record)

    return monitor


if __name__ == "__main__":
    # Test the performance monitor
    import random

    print("Testing Performance Monitor")
    print("=" * 50)

    monitor = PerformanceMonitor(windows=[10, 20, 50])

    # Generate some random trades
    random.seed(42)
    for i in range(100):
        # 55% win rate with varying P&L
        is_win = random.random() < 0.55
        if is_win:
            pnl = random.uniform(50, 200)
        else:
            pnl = random.uniform(-150, -30)

        trade = TradeRecord(
            pnl=pnl,
            timestamp=datetime.now(),
            ticker="AAPL",
        )
        monitor.add_trade(trade)

    print(f"\nTotal trades: {monitor.get_trade_count()}")

    print("\nCurrent metrics by window:")
    for window, metrics in monitor.get_all_current_metrics().items():
        print(f"\n  Window {window}:")
        print(f"    Win Rate: {metrics.win_rate:.1f}%")
        print(f"    Profit Factor: {metrics.profit_factor:.2f}")
        print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"    Total P&L: ${metrics.total_pnl:.2f}")
        print(f"    Max Drawdown: ${metrics.max_drawdown:.2f}")

    print("\nComparison (recent vs longer-term):")
    comparison = monitor.get_metrics_comparison()
    for key, value in comparison.items():
        print(f"  {key}: {value}")
