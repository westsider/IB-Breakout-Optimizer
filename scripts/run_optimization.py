#!/usr/bin/env python
"""
Standalone optimization script for parallel execution.
Run this directly for full CPU utilization with multiprocessing.

Usage:
    python run_optimization.py --ticker TSLA --preset standard --objective sharpe_ratio
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Run IB Breakout Optimization')
    parser.add_argument('--ticker', type=str, default='TSLA', help='Ticker symbol')
    parser.add_argument('--preset', type=str, default='standard',
                       choices=['quick', 'standard', 'full', 'exits_only'],
                       help='Parameter preset')
    parser.add_argument('--objective', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'sortino_ratio', 'profit_factor',
                               'total_profit', 'calmar_ratio', 'win_rate'],
                       help='Objective function to optimize')
    parser.add_argument('--data-dir', type=str, default=r'C:\Users\Warren\Downloads',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str,
                       default=r'C:\Users\Warren\Projects\ib_breakout_optimizer\output',
                       help='Output directory')
    parser.add_argument('--use-qqq-filter', action='store_true',
                       help='Enable QQQ filter')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 = all CPUs)')
    parser.add_argument('--progress-file', type=str, default=None,
                       help='File to write progress updates to (for UI integration)')

    args = parser.parse_args()

    # Import after setting up path
    from optimization.grid_search import GridSearchOptimizer
    from optimization.parameter_space import create_parameter_space

    print(f"\n{'='*60}")
    print("IB BREAKOUT GRID SEARCH OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Ticker: {args.ticker}")
    print(f"Preset: {args.preset}")
    print(f"Objective: {args.objective}")
    print(f"QQQ Filter: {args.use_qqq_filter}")
    print(f"Data Dir: {args.data_dir}")
    print(f"{'='*60}\n")

    # Create optimizer
    optimizer = GridSearchOptimizer(
        data_dir=args.data_dir,
        output_dir=f"{args.output_dir}/optimization",
        n_jobs=args.n_jobs
    )

    # Load data
    print(f"Loading data for {args.ticker}...")
    filter_ticker = 'QQQ' if args.use_qqq_filter and args.ticker != 'QQQ' else None
    optimizer.load_data(args.ticker, filter_ticker=filter_ticker)

    # Create parameter space
    space = create_parameter_space(args.preset)

    # Enable QQQ filter parameter if requested
    if args.use_qqq_filter and 'use_qqq_filter' in space.parameters:
        space.parameters['use_qqq_filter'].enabled = True

    # Progress callback for file-based updates
    def progress_callback(current, total, best):
        if args.progress_file:
            progress_data = {
                'current': current,
                'total': total,
                'percent': current / total * 100,
                'best_objective': best.objective_value if best else None,
                'best_trades': best.metrics.total_trades if best else None,
                'best_pnl': best.metrics.total_net_profit if best else None,
                'timestamp': datetime.now().isoformat()
            }
            with open(args.progress_file, 'w') as f:
                json.dump(progress_data, f)

    # Run optimization
    results = optimizer.optimize(
        parameter_space=space,
        objective=args.objective,
        progress_callback=progress_callback if args.progress_file else None
    )

    # Print summary
    print("\n" + results.summary())

    # Save final results to JSON for UI to read
    if args.progress_file:
        final_data = {
            'status': 'complete',
            'total_combinations': results.total_combinations,
            'completed': results.completed_combinations,
            'total_time': results.total_time_seconds,
            'best_params': results.best_result.params if results.best_result else None,
            'best_objective': results.best_result.objective_value if results.best_result else None,
            'best_metrics': {
                'total_trades': results.best_result.metrics.total_trades,
                'win_rate': results.best_result.metrics.percent_profitable,
                'total_pnl': results.best_result.metrics.total_net_profit,
                'profit_factor': results.best_result.metrics.profit_factor,
                'sharpe_ratio': results.best_result.metrics.sharpe_ratio,
                'max_drawdown': results.best_result.metrics.max_drawdown
            } if results.best_result else None
        }
        with open(args.progress_file, 'w') as f:
            json.dump(final_data, f, indent=2)

    return results


if __name__ == '__main__':
    # This guard is REQUIRED for multiprocessing on Windows
    main()
