"""
Comprehensive test to verify optimizer results match backtest results.

This test ensures that when you run an optimization and then double-click
a result, the equity curve, trade browser, and IB analysis tabs show
consistent data.

Run with: python -m pytest tests/test_optimizer_backtest_consistency.py -v
Or standalone: python tests/test_optimizer_backtest_consistency.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime
from typing import Dict, List, Tuple
import json


def run_optimization_and_backtest(
    ticker: str,
    use_qqq_filter: bool,
    preset: str = "quick",
    objective: str = "profit_factor"
) -> Tuple[Dict, Dict, List]:
    """
    Run optimization and then backtest with best params.

    Returns:
        Tuple of (optimization_result, backtest_metrics, backtest_trades)
    """
    from optimization.mmap_grid_search import MMapGridSearchOptimizer
    from optimization.parameter_space import create_parameter_space
    from backtester.backtest_runner import BacktestRunner
    from strategy.ib_breakout import StrategyParams

    # Get data directory
    data_dir = Path(__file__).parent.parent / "market_data"

    # Run optimization
    optimizer = MMapGridSearchOptimizer(str(data_dir))
    filter_ticker = "QQQ" if use_qqq_filter and ticker != "QQQ" else None
    optimizer.load_data(ticker, filter_ticker=filter_ticker)

    space = create_parameter_space(preset)
    results = optimizer.optimize(space, objective=objective)

    if not results.best_result:
        raise ValueError("No optimization results")

    best = results.best_result
    opt_result = {
        'params': best.params.copy(),
        'total_trades': best.total_trades,
        'win_rate': best.win_rate,
        'total_pnl': best.total_pnl,
        'profit_factor': best.profit_factor,
        'sharpe_ratio': best.sharpe_ratio,
        'max_drawdown': best.max_drawdown,
        'trade_pnls': best.trade_pnls.copy() if best.trade_pnls else [],
    }

    # Add use_qqq_filter to params if not present
    if 'use_qqq_filter' not in opt_result['params']:
        opt_result['params']['use_qqq_filter'] = use_qqq_filter

    # Run backtest with same params
    runner = BacktestRunner(str(data_dir))

    strategy_params = StrategyParams(
        ib_duration_minutes=opt_result['params'].get('ib_duration_minutes', 30),
        profit_target_percent=opt_result['params'].get('profit_target_percent', 0.5),
        stop_loss_type=opt_result['params'].get('stop_loss_type', 'opposite_ib'),
        trade_direction=opt_result['params'].get('trade_direction', 'both'),
        use_qqq_filter=use_qqq_filter,
        min_ib_range_percent=opt_result['params'].get('min_ib_range_percent', 0.0),
        max_ib_range_percent=opt_result['params'].get('max_ib_range_percent', 100.0),
        max_breakout_time=opt_result['params'].get('max_breakout_time', '14:00'),
        eod_exit_time=opt_result['params'].get('eod_exit_time', '15:55'),
        trailing_stop_enabled=opt_result['params'].get('trailing_stop_enabled', False),
        trailing_stop_atr_mult=opt_result['params'].get('trailing_stop_atr_mult', 2.0),
        break_even_enabled=opt_result['params'].get('break_even_enabled', False),
        break_even_pct=opt_result['params'].get('break_even_pct', 0.5),
        max_bars_enabled=opt_result['params'].get('max_bars_enabled', False),
        max_bars=opt_result['params'].get('max_bars', 60),
        fixed_share_size=opt_result['params'].get('fixed_share_size', 100),
    )

    if use_qqq_filter:
        bt_result, bt_metrics = runner.run_backtest_with_filter(
            ticker=ticker,
            filter_ticker="QQQ",
            params=strategy_params,
            verbose=False
        )
    else:
        bt_result, bt_metrics = runner.run_backtest(
            ticker=ticker,
            params=strategy_params,
            verbose=False
        )

    return opt_result, bt_metrics, bt_result.trades


def calculate_metrics_from_trades(trades: List) -> Dict:
    """Calculate metrics from trade list (as equity curve tab would)."""
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
        }

    pnls = [t.pnl for t in trades]
    winners = [p for p in pnls if p >= 0]
    losers = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.99 if gross_profit > 0 else 0)

    # Calculate max drawdown
    cumulative = []
    total = 0
    for pnl in pnls:
        total += pnl
        cumulative.append(total)

    max_dd = 0
    peak = 0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    return {
        'total_trades': len(trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
    }


class TestOptimizerBacktestConsistency(unittest.TestCase):
    """Test that optimizer and backtest produce consistent results."""

    @classmethod
    def setUpClass(cls):
        """Check if data files exist."""
        data_dir = Path(__file__).parent.parent / "market_data"
        cls.has_tsla = (data_dir / "TSLA_NT.txt").exists()
        cls.has_qqq = (data_dir / "QQQ_NT.txt").exists()
        cls.has_aapl = (data_dir / "AAPL_NT.txt").exists()

    def test_tsla_no_filter(self):
        """Test TSLA optimization without QQQ filter."""
        if not self.has_tsla:
            self.skipTest("TSLA data not available")

        opt_result, bt_metrics, trades = run_optimization_and_backtest(
            ticker="TSLA",
            use_qqq_filter=False,
            preset="quick"
        )

        self._compare_results("TSLA (no filter)", opt_result, bt_metrics, trades)

    def test_tsla_with_qqq_filter(self):
        """Test TSLA optimization with QQQ filter."""
        if not self.has_tsla or not self.has_qqq:
            self.skipTest("TSLA or QQQ data not available")

        opt_result, bt_metrics, trades = run_optimization_and_backtest(
            ticker="TSLA",
            use_qqq_filter=True,
            preset="quick"
        )

        self._compare_results("TSLA (with QQQ filter)", opt_result, bt_metrics, trades)

    def test_aapl_no_filter(self):
        """Test AAPL optimization without QQQ filter."""
        if not self.has_aapl:
            self.skipTest("AAPL data not available")

        opt_result, bt_metrics, trades = run_optimization_and_backtest(
            ticker="AAPL",
            use_qqq_filter=False,
            preset="quick"
        )

        self._compare_results("AAPL (no filter)", opt_result, bt_metrics, trades)

    def test_aapl_with_qqq_filter(self):
        """Test AAPL optimization with QQQ filter."""
        if not self.has_aapl or not self.has_qqq:
            self.skipTest("AAPL or QQQ data not available")

        opt_result, bt_metrics, trades = run_optimization_and_backtest(
            ticker="AAPL",
            use_qqq_filter=True,
            preset="quick"
        )

        self._compare_results("AAPL (with QQQ filter)", opt_result, bt_metrics, trades)

    def _compare_results(self, test_name: str, opt_result: Dict, bt_metrics, trades: List):
        """Compare optimization result with backtest metrics."""
        print(f"\n{'='*60}")
        print(f"Validating: {test_name}")
        print(f"{'='*60}")

        # Calculate metrics from trades (as UI tabs would)
        trade_metrics = calculate_metrics_from_trades(trades)

        # Trade count
        opt_trades = opt_result['total_trades']
        bt_trades = bt_metrics.total_trades
        ui_trades = trade_metrics['total_trades']

        print(f"\nTrade Count:")
        print(f"  Optimizer:  {opt_trades}")
        print(f"  Backtest:   {bt_trades}")
        print(f"  From Trades: {ui_trades}")

        self.assertEqual(opt_trades, bt_trades,
            f"Trade count mismatch: optimizer={opt_trades}, backtest={bt_trades}")
        self.assertEqual(bt_trades, ui_trades,
            f"Trade count mismatch: backtest={bt_trades}, from_trades={ui_trades}")

        # Total P&L (PerformanceMetrics uses total_net_profit)
        opt_pnl = opt_result['total_pnl']
        bt_pnl = bt_metrics.total_net_profit
        ui_pnl = trade_metrics['total_pnl']

        print(f"\nTotal P&L:")
        print(f"  Optimizer:  ${opt_pnl:,.2f}")
        print(f"  Backtest:   ${bt_pnl:,.2f}")
        print(f"  From Trades: ${ui_pnl:,.2f}")

        # Allow small tolerance for floating point
        self.assertAlmostEqual(opt_pnl, bt_pnl, delta=0.01,
            msg=f"P&L mismatch: optimizer=${opt_pnl:.2f}, backtest=${bt_pnl:.2f}")
        self.assertAlmostEqual(bt_pnl, ui_pnl, delta=0.01,
            msg=f"P&L mismatch: backtest=${bt_pnl:.2f}, from_trades=${ui_pnl:.2f}")

        # Win Rate (PerformanceMetrics uses percent_profitable)
        opt_wr = opt_result['win_rate']
        bt_wr = bt_metrics.percent_profitable
        ui_wr = trade_metrics['win_rate']

        print(f"\nWin Rate:")
        print(f"  Optimizer:  {opt_wr:.2f}%")
        print(f"  Backtest:   {bt_wr:.2f}%")
        print(f"  From Trades: {ui_wr:.2f}%")

        self.assertAlmostEqual(opt_wr, bt_wr, delta=0.1,
            msg=f"Win rate mismatch: optimizer={opt_wr:.2f}%, backtest={bt_wr:.2f}%")
        self.assertAlmostEqual(bt_wr, ui_wr, delta=0.1,
            msg=f"Win rate mismatch: backtest={bt_wr:.2f}%, from_trades={ui_wr:.2f}%")

        # Profit Factor
        opt_pf = opt_result['profit_factor']
        bt_pf = bt_metrics.profit_factor
        ui_pf = trade_metrics['profit_factor']

        print(f"\nProfit Factor:")
        print(f"  Optimizer:  {opt_pf:.2f}")
        print(f"  Backtest:   {bt_pf:.2f}")
        print(f"  From Trades: {ui_pf:.2f}")

        # PF can have larger variance due to calculation method
        self.assertAlmostEqual(opt_pf, bt_pf, delta=0.1,
            msg=f"Profit factor mismatch: optimizer={opt_pf:.2f}, backtest={bt_pf:.2f}")

        # Max Drawdown
        opt_dd = opt_result['max_drawdown']
        bt_dd = bt_metrics.max_drawdown
        ui_dd = trade_metrics['max_drawdown']

        print(f"\nMax Drawdown:")
        print(f"  Optimizer:  ${opt_dd:,.2f}")
        print(f"  Backtest:   ${bt_dd:,.2f}")
        print(f"  From Trades: ${ui_dd:,.2f}")

        self.assertAlmostEqual(opt_dd, bt_dd, delta=1.0,
            msg=f"Drawdown mismatch: optimizer=${opt_dd:.2f}, backtest=${bt_dd:.2f}")

        # Compare trade-by-trade P&Ls
        if opt_result['trade_pnls'] and trades:
            opt_pnls = opt_result['trade_pnls']
            bt_pnls = [t.pnl for t in trades]

            print(f"\nTrade-by-Trade P&L Comparison:")
            print(f"  Optimizer trade count: {len(opt_pnls)}")
            print(f"  Backtest trade count:  {len(bt_pnls)}")

            if len(opt_pnls) == len(bt_pnls):
                mismatches = []
                for i, (opt_p, bt_p) in enumerate(zip(opt_pnls, bt_pnls)):
                    if abs(opt_p - bt_p) > 0.01:
                        mismatches.append((i, opt_p, bt_p))

                if mismatches:
                    print(f"  MISMATCHES FOUND: {len(mismatches)}")
                    for idx, opt_p, bt_p in mismatches[:5]:  # Show first 5
                        print(f"    Trade {idx}: opt=${opt_p:.2f}, bt=${bt_p:.2f}, diff=${opt_p-bt_p:.2f}")
                    self.fail(f"Trade P&L mismatches found: {len(mismatches)} trades differ")
                else:
                    print(f"  All {len(opt_pnls)} trades match exactly!")
            else:
                self.fail(f"Trade count mismatch in P&L arrays")

        print(f"\n✓ All validations passed for {test_name}")


def run_quick_validation():
    """Run a quick validation without unittest framework."""
    print("="*70)
    print("OPTIMIZER vs BACKTEST CONSISTENCY VALIDATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path(__file__).parent.parent / "market_data"

    tests = []

    if (data_dir / "TSLA_NT.txt").exists():
        tests.append(("TSLA", False))
        if (data_dir / "QQQ_NT.txt").exists():
            tests.append(("TSLA", True))

    if (data_dir / "AAPL_NT.txt").exists():
        tests.append(("AAPL", False))
        if (data_dir / "QQQ_NT.txt").exists():
            tests.append(("AAPL", True))

    if not tests:
        print("\nNo data files found. Skipping tests.")
        return False

    all_passed = True

    for ticker, use_qqq in tests:
        filter_str = "with QQQ filter" if use_qqq else "no filter"
        test_name = f"{ticker} ({filter_str})"

        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print(f"{'='*60}")

        try:
            opt_result, bt_metrics, trades = run_optimization_and_backtest(
                ticker=ticker,
                use_qqq_filter=use_qqq,
                preset="quick"
            )

            # Compare key metrics
            errors = []

            # Trade count
            if opt_result['total_trades'] != bt_metrics.total_trades:
                errors.append(f"Trade count: opt={opt_result['total_trades']}, bt={bt_metrics.total_trades}")

            # P&L (PerformanceMetrics uses total_net_profit)
            if abs(opt_result['total_pnl'] - bt_metrics.total_net_profit) > 0.01:
                errors.append(f"P&L: opt=${opt_result['total_pnl']:.2f}, bt=${bt_metrics.total_net_profit:.2f}")

            # Win rate (PerformanceMetrics uses percent_profitable)
            if abs(opt_result['win_rate'] - bt_metrics.percent_profitable) > 0.1:
                errors.append(f"Win rate: opt={opt_result['win_rate']:.2f}%, bt={bt_metrics.percent_profitable:.2f}%")

            if errors:
                print(f"\n[FAILED] {test_name}")
                for err in errors:
                    print(f"   - {err}")
                all_passed = False
            else:
                print(f"\n[PASSED] {test_name}")
                print(f"   Trades: {opt_result['total_trades']}")
                print(f"   P&L: ${opt_result['total_pnl']:,.2f}")
                print(f"   Win Rate: {opt_result['win_rate']:.1f}%")
                print(f"   Profit Factor: {opt_result['profit_factor']:.2f}")

        except Exception as e:
            print(f"\n[ERROR] {test_name}")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate optimizer/backtest consistency")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.quick:
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        # Run full unittest suite
        unittest.main(verbosity=2 if args.verbose else 1)
