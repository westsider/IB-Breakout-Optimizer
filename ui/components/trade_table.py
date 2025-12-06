"""
Trade Table Component - Reusable trade list display.
"""

import pandas as pd
from typing import List, Any, Optional, Dict, Callable
from datetime import datetime


def format_trade_row(trade: Any) -> Dict[str, Any]:
    """
    Format a trade object into a dictionary for display.

    Args:
        trade: Trade object with standard attributes

    Returns:
        Dictionary with formatted trade data
    """
    direction = getattr(trade, 'direction', None)
    direction_str = direction.value if direction else 'unknown'
    direction_emoji = '🟢 Long' if direction_str == 'long' else '🔴 Short'

    exit_reason = getattr(trade, 'exit_reason', None)
    exit_reason_str = exit_reason.value if exit_reason else ''

    entry_time = getattr(trade, 'entry_time', None)
    exit_time = getattr(trade, 'exit_time', None)
    entry_price = getattr(trade, 'entry_price', 0)
    exit_price = getattr(trade, 'exit_price', 0)
    pnl = getattr(trade, 'pnl', 0)
    pnl_pct = getattr(trade, 'pnl_pct', 0)
    bars_held = getattr(trade, 'bars_held', 0)
    ticker = getattr(trade, 'ticker', '')

    return {
        'Entry Time': entry_time.strftime('%Y-%m-%d %H:%M') if entry_time else '',
        'Exit Time': exit_time.strftime('%Y-%m-%d %H:%M') if exit_time else '',
        'Ticker': ticker,
        'Direction': direction_emoji,
        'Entry': f"${entry_price:.2f}",
        'Exit': f"${exit_price:.2f}" if exit_price else '',
        'P&L': f"${pnl:.2f}",
        'P&L %': f"{pnl_pct:.2f}%",
        'Exit Reason': exit_reason_str,
        'Bars': bars_held,
        # Raw values for filtering/sorting
        '_pnl': pnl,
        '_direction': direction_str,
        '_entry_time': entry_time,
        '_exit_reason': exit_reason_str
    }


def create_trade_table(
    trades: List[Any],
    columns: Optional[List[str]] = None,
    sort_by: str = 'Entry Time',
    ascending: bool = False,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a formatted DataFrame from a list of trades.

    Args:
        trades: List of trade objects
        columns: Columns to include (default: all display columns)
        sort_by: Column to sort by
        ascending: Sort order
        max_rows: Maximum rows to return (None for all)

    Returns:
        Formatted DataFrame
    """
    if not trades:
        return pd.DataFrame()

    # Format all trades
    trade_data = [format_trade_row(t) for t in trades]
    df = pd.DataFrame(trade_data)

    # Default display columns (exclude internal columns)
    default_columns = [
        'Entry Time', 'Exit Time', 'Ticker', 'Direction',
        'Entry', 'Exit', 'P&L', 'P&L %', 'Exit Reason', 'Bars'
    ]

    if columns is None:
        columns = [c for c in default_columns if c in df.columns]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    # Limit rows
    if max_rows is not None:
        df = df.head(max_rows)

    # Select display columns
    display_df = df[columns].copy()

    return display_df


def filter_trades(
    trades: List[Any],
    direction: Optional[List[str]] = None,
    outcome: Optional[List[str]] = None,
    exit_reasons: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_pnl: Optional[float] = None,
    max_pnl: Optional[float] = None
) -> List[Any]:
    """
    Filter trades based on criteria.

    Args:
        trades: List of trade objects
        direction: List of allowed directions ('long', 'short')
        outcome: List of allowed outcomes ('Winner', 'Loser', 'Break-even')
        exit_reasons: List of allowed exit reasons
        ticker: Filter by ticker
        date_from: Minimum entry date
        date_to: Maximum entry date
        min_pnl: Minimum P&L
        max_pnl: Maximum P&L

    Returns:
        Filtered list of trades
    """
    filtered = []

    for t in trades:
        # Direction filter
        if direction is not None:
            trade_dir = getattr(t, 'direction', None)
            trade_dir_str = trade_dir.value if trade_dir else 'unknown'
            if trade_dir_str not in direction:
                continue

        # Outcome filter
        if outcome is not None:
            pnl = getattr(t, 'pnl', 0)
            if pnl > 0 and 'Winner' not in outcome:
                continue
            if pnl < 0 and 'Loser' not in outcome:
                continue
            if pnl == 0 and 'Break-even' not in outcome:
                continue

        # Exit reason filter
        if exit_reasons is not None:
            exit_reason = getattr(t, 'exit_reason', None)
            exit_reason_str = exit_reason.value if exit_reason else ''
            if exit_reason_str not in exit_reasons:
                continue

        # Ticker filter
        if ticker is not None:
            trade_ticker = getattr(t, 'ticker', '')
            if trade_ticker != ticker:
                continue

        # Date filter
        entry_time = getattr(t, 'entry_time', None)
        if date_from is not None and entry_time:
            if entry_time < date_from:
                continue
        if date_to is not None and entry_time:
            if entry_time > date_to:
                continue

        # P&L filter
        pnl = getattr(t, 'pnl', 0)
        if min_pnl is not None and pnl < min_pnl:
            continue
        if max_pnl is not None and pnl > max_pnl:
            continue

        filtered.append(t)

    return filtered


def calculate_trade_stats(trades: List[Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics for a list of trades.

    Args:
        trades: List of trade objects

    Returns:
        Dictionary of statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_winner': 0,
            'avg_loser': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'max_winner': 0,
            'max_loser': 0,
            'avg_bars_held': 0
        }

    pnls = [getattr(t, 'pnl', 0) for t in trades]
    bars = [getattr(t, 'bars_held', 0) for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0

    return {
        'total_trades': len(trades),
        'total_pnl': total_pnl,
        'win_rate': (len(winners) / len(trades) * 100) if trades else 0,
        'avg_winner': (sum(winners) / len(winners)) if winners else 0,
        'avg_loser': (sum(losers) / len(losers)) if losers else 0,
        'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else float('inf'),
        'avg_trade': (total_pnl / len(trades)) if trades else 0,
        'max_winner': max(winners) if winners else 0,
        'max_loser': min(losers) if losers else 0,
        'avg_bars_held': (sum(bars) / len(bars)) if bars else 0,
        'winners': len(winners),
        'losers': len(losers),
        'breakeven': len([p for p in pnls if p == 0])
    }


def group_trades_by(
    trades: List[Any],
    group_by: str = 'day'
) -> Dict[str, List[Any]]:
    """
    Group trades by time period or attribute.

    Args:
        trades: List of trade objects
        group_by: Grouping method ('day', 'week', 'month', 'direction', 'exit_reason', 'ticker')

    Returns:
        Dictionary mapping group key to list of trades
    """
    groups = {}

    for t in trades:
        entry_time = getattr(t, 'entry_time', None)

        if group_by == 'day':
            key = entry_time.strftime('%Y-%m-%d') if entry_time else 'Unknown'
        elif group_by == 'week':
            key = entry_time.strftime('%Y-W%W') if entry_time else 'Unknown'
        elif group_by == 'month':
            key = entry_time.strftime('%Y-%m') if entry_time else 'Unknown'
        elif group_by == 'direction':
            direction = getattr(t, 'direction', None)
            key = direction.value if direction else 'unknown'
        elif group_by == 'exit_reason':
            exit_reason = getattr(t, 'exit_reason', None)
            key = exit_reason.value if exit_reason else 'Unknown'
        elif group_by == 'ticker':
            key = getattr(t, 'ticker', 'Unknown')
        elif group_by == 'dow':  # Day of week
            key = entry_time.strftime('%A') if entry_time else 'Unknown'
        elif group_by == 'hour':
            key = entry_time.strftime('%H:00') if entry_time else 'Unknown'
        else:
            key = 'All'

        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    return groups
