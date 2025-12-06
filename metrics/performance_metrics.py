"""
Performance Metrics for IB Breakout Optimizer.

Calculates all standard trading performance metrics including:
- P&L metrics (total, gross profit/loss, avg trade)
- Win rate metrics (win%, consecutive wins/losses)
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis (max DD, duration, recovery)
- Trade analysis (avg bars, time in market)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict

from data.data_types import Trade, TradeDirection, ExitReason


@dataclass
class DrawdownInfo:
    """Drawdown statistics."""
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    time_to_recovery_days: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    Matches NinjaTrader's Strategy Analyzer output format.
    """
    # Period
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    even_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0

    # P&L metrics
    total_net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    commission: float = 0.0
    total_fees: float = 0.0

    # Ratios
    profit_factor: float = 0.0
    percent_profitable: float = 0.0
    ratio_avg_win_avg_loss: float = 0.0

    # Average trade metrics
    avg_trade: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0

    # Consecutive trades
    max_consecutive_winners: int = 0
    max_consecutive_losers: int = 0

    # Extremes
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0

    # Time metrics
    avg_time_in_market_minutes: float = 0.0
    avg_bars_in_trade: float = 0.0
    avg_trades_per_day: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    r_squared: float = 0.0

    # Recovery metrics
    max_time_to_recover_days: float = 0.0
    longest_flat_period_days: float = 0.0
    profit_per_month: float = 0.0

    # MAE/MFE
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    avg_etd: float = 0.0  # End Trade Drawdown

    # Probability (from NT - not sure what this is)
    probability: float = 0.0

    # Exit analysis
    exit_reasons: Dict[str, int] = field(default_factory=dict)

    # By direction
    long_net_profit: float = 0.0
    short_net_profit: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'Performance': {
                'Total net profit': f"${self.total_net_profit:,.2f}",
                'Gross profit': f"${self.gross_profit:,.2f}",
                'Gross loss': f"(${abs(self.gross_loss):,.2f})",
                'Commission': f"${self.commission:,.2f}",
                'Profit factor': f"{self.profit_factor:.2f}",
                'Max. drawdown': f"(${abs(self.max_drawdown):,.2f})",
                'Sharpe ratio': f"{self.sharpe_ratio:.2f}",
                'Sortino ratio': f"{self.sortino_ratio:.2f}",
                'Ulcer index': f"{self.ulcer_index:.2f}",
                'R squared': f"{self.r_squared:.2f}",
            },
            'Trades': {
                'Total # of trades': self.total_trades,
                'Percent profitable': f"{self.percent_profitable:.2f}%",
                '# of winning trades': self.winning_trades,
                '# of losing trades': self.losing_trades,
                '# of even trades': self.even_trades,
            },
            'Averages': {
                'Avg. trade': f"${self.avg_trade:.2f}",
                'Avg. winning trade': f"${self.avg_winning_trade:.2f}",
                'Avg. losing trade': f"(${abs(self.avg_losing_trade):.2f})",
                'Ratio avg. win / avg. loss': f"{self.ratio_avg_win_avg_loss:.2f}",
            },
            'Streaks': {
                'Max. consec. winners': self.max_consecutive_winners,
                'Max. consec. losers': self.max_consecutive_losers,
                'Largest winning trade': f"${self.largest_winning_trade:.2f}",
                'Largest losing trade': f"(${abs(self.largest_losing_trade):.2f})",
            },
            'Time': {
                'Avg. # of trades per day': f"{self.avg_trades_per_day:.2f}",
                'Avg. time in market': f"{self.avg_time_in_market_minutes:.2f} min",
                'Avg. bars in trade': f"{self.avg_bars_in_trade:.2f}",
                'Profit per month': f"${self.profit_per_month:.2f}",
                'Max. time to recover': f"{self.max_time_to_recover_days:.2f} days",
                'Longest flat period': f"{self.longest_flat_period_days:.2f} days",
            },
            'MAE/MFE': {
                'Avg. MAE': f"${self.avg_mae:.2f}",
                'Avg. MFE': f"${self.avg_mfe:.2f}",
                'Avg. ETD': f"${self.avg_etd:.2f}",
            }
        }

    def print_report(self):
        """Print formatted performance report matching NT Strategy Analyzer."""
        print("=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)

        if self.start_date and self.end_date:
            print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            print()

        print(f"{'Performance':<30} {'All trades':>12} {'Long':>12} {'Short':>12}")
        print("-" * 60)
        print(f"{'Total net profit':<30} ${self.total_net_profit:>10,.2f} ${self.long_net_profit:>10,.2f} ${self.short_net_profit:>10,.2f}")
        print(f"{'Gross profit':<30} ${self.gross_profit:>10,.2f}")
        print(f"{'Gross loss':<30} (${abs(self.gross_loss):>9,.2f})")
        print(f"{'Commission':<30} ${self.commission:>10,.2f}")
        print(f"{'Profit factor':<30} {self.profit_factor:>12.2f}")
        print(f"{'Max. drawdown':<30} (${abs(self.max_drawdown):>9,.2f})")
        print(f"{'Sharpe ratio':<30} {self.sharpe_ratio:>12.2f}")
        print(f"{'Sortino ratio':<30} {self.sortino_ratio:>12.2f}")
        print()

        print(f"{'Total # of trades':<30} {self.total_trades:>12}")
        print(f"{'Percent profitable':<30} {self.percent_profitable:>11.2f}%")
        print(f"{'# of winning trades':<30} {self.winning_trades:>12}")
        print(f"{'# of losing trades':<30} {self.losing_trades:>12}")
        print()

        print(f"{'Avg. trade':<30} ${self.avg_trade:>10,.2f}")
        print(f"{'Avg. winning trade':<30} ${self.avg_winning_trade:>10,.2f}")
        print(f"{'Avg. losing trade':<30} (${abs(self.avg_losing_trade):>9,.2f})")
        print(f"{'Ratio avg. win / avg. loss':<30} {self.ratio_avg_win_avg_loss:>12.2f}")
        print()

        print(f"{'Max. consec. winners':<30} {self.max_consecutive_winners:>12}")
        print(f"{'Max. consec. losers':<30} {self.max_consecutive_losers:>12}")
        print(f"{'Largest winning trade':<30} ${self.largest_winning_trade:>10,.2f}")
        print(f"{'Largest losing trade':<30} (${abs(self.largest_losing_trade):>9,.2f})")
        print()

        print(f"{'Avg. # of trades per day':<30} {self.avg_trades_per_day:>12.2f}")
        print(f"{'Avg. time in market':<30} {self.avg_time_in_market_minutes:>9.2f} min")
        print(f"{'Avg. bars in trade':<30} {self.avg_bars_in_trade:>12.2f}")
        print(f"{'Profit per month':<30} ${self.profit_per_month:>10,.2f}")
        print(f"{'Max. time to recover':<30} {self.max_time_to_recover_days:>9.2f} days")
        print()

        print(f"{'Avg. MAE':<30} ${self.avg_mae:>10,.2f}")
        print(f"{'Avg. MFE':<30} ${self.avg_mfe:>10,.2f}")
        print(f"{'Avg. ETD':<30} ${self.avg_etd:>10,.2f}")
        print("=" * 60)


def calculate_metrics(trades: List[Trade], initial_capital: float = 100000.0) -> PerformanceMetrics:
    """
    Calculate all performance metrics from a list of trades.

    Args:
        trades: List of completed trades
        initial_capital: Starting capital for percentage calculations

    Returns:
        PerformanceMetrics object with all calculated values
    """
    metrics = PerformanceMetrics()

    if not trades:
        return metrics

    # Sort trades by exit time
    trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)

    # Basic counts
    metrics.total_trades = len(trades)
    metrics.winning_trades = len([t for t in trades if t.pnl > 0])
    metrics.losing_trades = len([t for t in trades if t.pnl < 0])
    metrics.even_trades = len([t for t in trades if t.pnl == 0])

    long_trades = [t for t in trades if t.direction == TradeDirection.LONG]
    short_trades = [t for t in trades if t.direction == TradeDirection.SHORT]
    metrics.long_trades = len(long_trades)
    metrics.short_trades = len(short_trades)

    # Date range
    entry_times = [t.entry_time for t in trades if t.entry_time]
    exit_times = [t.exit_time for t in trades if t.exit_time]
    if entry_times:
        metrics.start_date = min(entry_times)
    if exit_times:
        metrics.end_date = max(exit_times)

    # P&L metrics
    pnls = [t.pnl for t in trades]
    winning_pnls = [t.pnl for t in trades if t.pnl > 0]
    losing_pnls = [t.pnl for t in trades if t.pnl < 0]

    metrics.total_net_profit = sum(pnls)
    metrics.gross_profit = sum(winning_pnls) if winning_pnls else 0.0
    metrics.gross_loss = sum(losing_pnls) if losing_pnls else 0.0
    metrics.commission = sum(t.commission for t in trades)

    # By direction
    metrics.long_net_profit = sum(t.pnl for t in long_trades)
    metrics.short_net_profit = sum(t.pnl for t in short_trades)

    # Profit factor
    if metrics.gross_loss != 0:
        metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)
    else:
        metrics.profit_factor = float('inf') if metrics.gross_profit > 0 else 0.0

    # Win rate
    metrics.percent_profitable = (metrics.winning_trades / metrics.total_trades * 100) if metrics.total_trades > 0 else 0.0

    # Average trades
    metrics.avg_trade = np.mean(pnls) if pnls else 0.0
    metrics.avg_winning_trade = np.mean(winning_pnls) if winning_pnls else 0.0
    metrics.avg_losing_trade = np.mean(losing_pnls) if losing_pnls else 0.0

    # Ratio avg win / avg loss
    if metrics.avg_losing_trade != 0:
        metrics.ratio_avg_win_avg_loss = abs(metrics.avg_winning_trade / metrics.avg_losing_trade)
    else:
        metrics.ratio_avg_win_avg_loss = float('inf') if metrics.avg_winning_trade > 0 else 0.0

    # Consecutive wins/losses
    metrics.max_consecutive_winners, metrics.max_consecutive_losers = _calculate_consecutive(trades)

    # Extremes
    metrics.largest_winning_trade = max(winning_pnls) if winning_pnls else 0.0
    metrics.largest_losing_trade = min(losing_pnls) if losing_pnls else 0.0

    # Time metrics
    bars_held = [t.bars_held for t in trades if t.bars_held > 0]
    metrics.avg_bars_in_trade = np.mean(bars_held) if bars_held else 0.0
    metrics.avg_time_in_market_minutes = metrics.avg_bars_in_trade  # 1-minute bars

    # Trades per day
    if metrics.start_date and metrics.end_date:
        days = (metrics.end_date - metrics.start_date).days
        if days > 0:
            metrics.avg_trades_per_day = metrics.total_trades / days
            months = days / 30.0
            metrics.profit_per_month = metrics.total_net_profit / months if months > 0 else 0.0

    # Drawdown analysis
    drawdown = _calculate_drawdown(trades, initial_capital)
    metrics.max_drawdown = drawdown.max_drawdown
    metrics.max_drawdown_pct = drawdown.max_drawdown_pct
    metrics.max_time_to_recover_days = drawdown.time_to_recovery_days

    # Risk-adjusted returns
    metrics.sharpe_ratio = _calculate_sharpe(trades)
    metrics.sortino_ratio = _calculate_sortino(trades)
    metrics.calmar_ratio = _calculate_calmar(metrics.total_net_profit, metrics.max_drawdown, metrics.start_date, metrics.end_date)
    metrics.ulcer_index = _calculate_ulcer_index(trades, initial_capital)
    metrics.r_squared = _calculate_r_squared(trades)

    # MAE/MFE
    mae_values = [t.mae for t in trades if t.mae > 0]
    mfe_values = [t.mfe for t in trades if t.mfe > 0]
    metrics.avg_mae = np.mean(mae_values) if mae_values else 0.0
    metrics.avg_mfe = np.mean(mfe_values) if mfe_values else 0.0
    metrics.avg_etd = metrics.avg_mfe - metrics.avg_trade if metrics.avg_mfe > 0 else 0.0

    # Exit reason analysis
    metrics.exit_reasons = defaultdict(int)
    for trade in trades:
        if trade.exit_reason:
            metrics.exit_reasons[trade.exit_reason.value] += 1

    return metrics


def _calculate_consecutive(trades: List[Trade]) -> Tuple[int, int]:
    """Calculate max consecutive winners and losers."""
    max_winners = 0
    max_losers = 0
    current_winners = 0
    current_losers = 0

    for trade in trades:
        if trade.pnl > 0:
            current_winners += 1
            current_losers = 0
            max_winners = max(max_winners, current_winners)
        elif trade.pnl < 0:
            current_losers += 1
            current_winners = 0
            max_losers = max(max_losers, current_losers)
        else:
            current_winners = 0
            current_losers = 0

    return max_winners, max_losers


def _calculate_drawdown(trades: List[Trade], initial_capital: float) -> DrawdownInfo:
    """Calculate drawdown statistics."""
    info = DrawdownInfo()

    if not trades:
        return info

    # Build equity curve
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0

    drawdown_start = None
    max_dd_duration = timedelta(0)
    max_recovery_time = timedelta(0)

    for trade in sorted(trades, key=lambda t: t.exit_time or datetime.min):
        equity += trade.pnl

        if equity > peak:
            # New peak - record recovery time if we were in drawdown
            if drawdown_start is not None:
                recovery_time = (trade.exit_time - drawdown_start) if trade.exit_time else timedelta(0)
                max_recovery_time = max(max_recovery_time, recovery_time)
            peak = equity
            drawdown_start = None
        else:
            # In drawdown
            if drawdown_start is None:
                drawdown_start = trade.exit_time

            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

    info.max_drawdown = max_dd
    info.max_drawdown_pct = max_dd_pct
    info.current_drawdown = peak - equity
    info.current_drawdown_pct = (info.current_drawdown / peak * 100) if peak > 0 else 0
    info.time_to_recovery_days = max_recovery_time.days

    return info


def _calculate_sharpe(trades: List[Trade], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    """
    if len(trades) < 2:
        return 0.0

    returns = [t.pnl_pct / 100 for t in trades]
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize (assume ~252 trading days, ~1 trade per day average)
    annualization_factor = np.sqrt(252)

    sharpe = (mean_return - risk_free_rate/252) / std_return * annualization_factor
    return sharpe


def _calculate_sortino(trades: List[Trade], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio.

    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
    """
    if len(trades) < 2:
        return 0.0

    returns = [t.pnl_pct / 100 for t in trades]
    mean_return = np.mean(returns)

    # Downside deviation (std of negative returns only)
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        return float('inf') if mean_return > 0 else 0.0

    downside_std = np.std(negative_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    annualization_factor = np.sqrt(252)
    sortino = (mean_return - risk_free_rate/252) / downside_std * annualization_factor
    return sortino


def _calculate_calmar(total_profit: float, max_drawdown: float,
                      start_date: Optional[datetime], end_date: Optional[datetime]) -> float:
    """
    Calculate Calmar ratio.

    Calmar = Annualized Return / Max Drawdown
    """
    if max_drawdown == 0 or not start_date or not end_date:
        return 0.0

    days = (end_date - start_date).days
    if days <= 0:
        return 0.0

    annualized_return = (total_profit / days) * 365
    calmar = abs(annualized_return / max_drawdown)
    return calmar


def _calculate_ulcer_index(trades: List[Trade], initial_capital: float) -> float:
    """
    Calculate Ulcer Index.

    Measures depth and duration of drawdowns.
    Lower is better.
    """
    if not trades:
        return 0.0

    # Build equity curve
    equity = initial_capital
    peak = equity
    squared_drawdowns = []

    for trade in sorted(trades, key=lambda t: t.exit_time or datetime.min):
        equity += trade.pnl
        if equity > peak:
            peak = equity

        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0
        squared_drawdowns.append(dd_pct ** 2)

    if not squared_drawdowns:
        return 0.0

    ulcer = np.sqrt(np.mean(squared_drawdowns))
    return ulcer


def _calculate_r_squared(trades: List[Trade]) -> float:
    """
    Calculate R-squared of equity curve.

    Measures how linear the equity growth is.
    Higher is better (more consistent growth).
    """
    if len(trades) < 3:
        return 0.0

    # Build cumulative P&L
    cum_pnl = []
    total = 0
    for trade in sorted(trades, key=lambda t: t.exit_time or datetime.min):
        total += trade.pnl
        cum_pnl.append(total)

    # Fit linear regression
    x = np.arange(len(cum_pnl))
    y = np.array(cum_pnl)

    # Calculate R-squared
    correlation_matrix = np.corrcoef(x, y)
    correlation = correlation_matrix[0, 1]
    r_squared = correlation ** 2

    return r_squared


if __name__ == "__main__":
    # Test with sample trades
    from data.data_types import Trade, TradeDirection, ExitReason
    from datetime import datetime, timedelta

    # Create sample trades matching the NT output
    # From stats.csv: 165 trades, 90 wins, 75 losses, 54.55% win rate
    sample_trades = []

    base_date = datetime(2022, 9, 29)

    # Create 90 winning trades
    for i in range(90):
        entry_time = base_date + timedelta(days=i*2, hours=10)
        trade = Trade(
            trade_id=f"W{i:03d}",
            ticker="TSLA",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=200.0,
            exit_time=entry_time + timedelta(minutes=53),
            exit_price=200.97,  # ~$97.38 avg winner on 100 shares
            quantity=100,
            exit_reason=ExitReason.PROFIT_TARGET,
            bars_held=51
        )
        trade.calculate_pnl()
        sample_trades.append(trade)

    # Create 75 losing trades
    for i in range(75):
        entry_time = base_date + timedelta(days=i*2 + 1, hours=10)
        trade = Trade(
            trade_id=f"L{i:03d}",
            ticker="TSLA",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=200.0,
            exit_time=entry_time + timedelta(minutes=53),
            exit_price=198.99,  # ~$101.01 avg loser on 100 shares
            quantity=100,
            exit_reason=ExitReason.STOP_LOSS,
            bars_held=51
        )
        trade.calculate_pnl()
        sample_trades.append(trade)

    # Calculate metrics
    metrics = calculate_metrics(sample_trades)
    metrics.print_report()
