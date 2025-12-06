#!/usr/bin/env python3
"""
Quick Start Script - Run a backtest with the IB Breakout Strategy.

Usage:
    python scripts/run_backtest.py

This script will:
1. Load available data from Downloads folder
2. Run the IB Breakout strategy
3. Print performance metrics
4. Export results to CSV
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder
from strategy.ib_breakout import IBBreakoutStrategy, StrategyParams
from metrics.performance_metrics import calculate_metrics


def find_data_files(data_dir: str) -> dict:
    """Find available data files."""
    data_path = Path(data_dir)
    files = {}

    for f in data_path.iterdir():
        if f.is_file():
            name_upper = f.name.upper()
            if f.suffix.lower() in ['.txt', '.csv']:
                # Try to extract ticker from filename
                for ticker in ['TSLA', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'SPY']:
                    if ticker in name_upper:
                        files[ticker] = f
                        break

    return files


def run_single_backtest(
    data_file: Path,
    ticker: str,
    params: StrategyParams = None
) -> dict:
    """Run backtest on a single ticker."""

    params = params or StrategyParams(
        ib_duration_minutes=30,
        profit_target_percent=0.5,
        stop_loss_type="opposite_ib",
        trade_direction="both",  # Changed to allow both directions
        use_qqq_filter=False,
        trading_start_time="09:00",
        trading_end_time="15:00"
    )

    print(f"\n{'='*60}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*60}")
    print(f"Data file: {data_file.name}")

    # Load data
    loader = DataLoader(str(data_file.parent))
    df = loader.load_auto_detect(str(data_file), ticker)

    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Build sessions
    session_builder = SessionBuilder(
        market_open="09:30",
        market_close="16:00",
        ib_duration_minutes=params.ib_duration_minutes
    )
    sessions = session_builder.build_sessions_from_dataframe(df, ticker)
    print(f"Trading sessions: {len(sessions)}")

    # Create strategy
    strategy = IBBreakoutStrategy(params)

    print(f"\nStrategy Parameters:")
    print(f"  IB Duration: {params.ib_duration_minutes} minutes")
    print(f"  Profit Target: {params.profit_target_percent}%")
    print(f"  Stop Loss: {params.stop_loss_type}")
    print(f"  Direction: {params.trade_direction}")
    print(f"  QQQ Filter: {'ON' if params.use_qqq_filter else 'OFF'}")

    # Run backtest
    print(f"\nProcessing bars...")
    signal_count = 0

    for session in sessions:
        is_first = True
        for bar in session.bars:
            signal = strategy.process_bar(bar, is_first)
            is_first = False

            if signal:
                signal_count += 1

    # Get results
    trades = strategy.get_trades()
    summary = strategy.get_trade_summary()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total Signals: {signal_count}")
    print(f"Total Trades: {summary['total_trades']}")

    if summary['total_trades'] > 0:
        print(f"\nWin Rate: {summary['win_rate']:.1f}%")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"\nTotal P&L: ${summary['total_pnl']:,.2f}")
        print(f"Avg Trade: ${summary['avg_pnl']:,.2f}")
        print(f"Avg Winner: ${summary['avg_winner']:,.2f}")
        print(f"Avg Loser: ${summary['avg_loser']:,.2f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")

        # Calculate detailed metrics
        metrics = calculate_metrics(trades)

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: ${metrics.max_drawdown:,.2f}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")

        # Export trades
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)

        trades_file = output_dir / f"{ticker}_trades.csv"
        import pandas as pd

        trades_data = []
        for t in trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'direction': t.direction.value,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason.value if t.exit_reason else None,
                'bars_held': t.bars_held,
                'ib_high': t.ib.ib_high if t.ib else None,
                'ib_low': t.ib.ib_low if t.ib else None
            })

        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(trades_file, index=False)
        print(f"\nTrades exported to: {trades_file}")

    return {
        'ticker': ticker,
        'trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'total_pnl': summary['total_pnl'],
        'profit_factor': summary['profit_factor']
    }


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("IB BREAKOUT STRATEGY - PYTHON BACKTESTER")
    print("="*60)

    # Find data files
    data_dir = r"C:\Users\Warren\Downloads"
    print(f"\nSearching for data files in: {data_dir}")

    files = find_data_files(data_dir)

    if not files:
        print("\nNo data files found!")
        print("Please ensure you have files like 'TSLA_1min_NT.txt' in your Downloads folder.")
        print("\nSupported formats:")
        print("  - NinjaTrader: yyyyMMdd HHmmss;O;H;L;C;V")
        print("  - CSV: yyyy-mm-dd HH:MM,O,H,L,C,V")
        return

    print(f"\nFound data for: {', '.join(files.keys())}")

    # Run backtest for each ticker
    results = []

    for ticker, filepath in files.items():
        try:
            result = run_single_backtest(filepath, ticker)
            results.append(result)
        except Exception as e:
            print(f"\nError backtesting {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY - ALL TICKERS")
        print("="*60)
        print(f"{'Ticker':<10} {'Trades':>8} {'Win Rate':>10} {'Total P&L':>12} {'PF':>8}")
        print("-"*60)

        total_pnl = 0
        total_trades = 0

        for r in results:
            print(f"{r['ticker']:<10} {r['trades']:>8} {r['win_rate']:>9.1f}% ${r['total_pnl']:>10,.2f} {r['profit_factor']:>8.2f}")
            total_pnl += r['total_pnl']
            total_trades += r['trades']

        print("-"*60)
        print(f"{'TOTAL':<10} {total_trades:>8} {'-':>10} ${total_pnl:>10,.2f}")

    print("\n" + "="*60)
    print("Backtest complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
