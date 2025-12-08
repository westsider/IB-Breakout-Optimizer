"""
Filter Verification Diagnostic Script

This script verifies that all filter calculations are working correctly by:
1. Loading ticker data
2. Calculating filter values for each trading day
3. Showing distribution statistics
4. Demonstrating which days would be filtered by each filter mode

Usage:
    python scripts/verify_filters.py [ticker] [data_dir]

Example:
    python scripts/verify_filters.py TSLA market_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_daily_data(data_dir: str, ticker: str) -> pd.DataFrame:
    """Load and aggregate minute data into daily OHLC."""
    data_path = Path(data_dir) / f"{ticker}_NT.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load NinjaTrader format data
    df = pd.read_csv(data_path, sep=';', header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi'])

    # Parse date and time
    df['date'] = pd.to_datetime(df['datetime'].astype(str).str[:8], format='%Y%m%d')
    df['time'] = df['datetime'].astype(str).str[9:]

    # Aggregate to daily OHLC (using all bars, not just RTH)
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    return daily


def calculate_filter_values(daily: pd.DataFrame, trend_lookback: int = 3, range_lookback: int = 5) -> pd.DataFrame:
    """Calculate all filter values for each trading day."""

    # =========================================================================
    # GAP % CALCULATION
    # Gap = (Today's Open - Yesterday's Close) / Yesterday's Close * 100
    # =========================================================================
    daily['prior_close'] = daily['close'].shift(1)
    daily['gap_pct'] = (daily['open'] - daily['prior_close']) / daily['prior_close'] * 100

    # Gap direction for filter modes
    daily['gap_direction'] = 'neutral'
    daily.loc[daily['gap_pct'] > 0, 'gap_direction'] = 'up'
    daily.loc[daily['gap_pct'] < 0, 'gap_direction'] = 'down'

    # =========================================================================
    # PRIOR DAYS TREND CALCULATION
    # Count of bullish days (close > open) in the last N days
    # =========================================================================
    daily['bullish'] = (daily['close'] > daily['open']).astype(int)
    daily['bullish_count'] = daily['bullish'].shift(1).rolling(trend_lookback).sum()
    daily['bearish_count'] = trend_lookback - daily['bullish_count']

    # Trend direction
    daily['trend'] = 'neutral'
    daily.loc[daily['bullish_count'] > daily['bearish_count'], 'trend'] = 'bullish'
    daily.loc[daily['bearish_count'] > daily['bullish_count'], 'trend'] = 'bearish'

    # =========================================================================
    # DAILY RANGE % (VOLATILITY) CALCULATION
    # Average of (High - Low) / Low * 100 over the last N days
    # =========================================================================
    daily['range_pct'] = (daily['high'] - daily['low']) / daily['low'] * 100
    daily['avg_range_pct'] = daily['range_pct'].shift(1).rolling(range_lookback).mean()

    return daily


def print_distribution_stats(daily: pd.DataFrame):
    """Print distribution statistics for filter values."""
    print("\n" + "=" * 80)
    print("FILTER VALUE DISTRIBUTIONS")
    print("=" * 80)

    # Gap % distribution
    gap = daily['gap_pct'].dropna()
    print(f"\nGAP % (Today's Open vs Yesterday's Close):")
    print(f"  Count: {len(gap)} days")
    print(f"  Mean: {gap.mean():.3f}%")
    print(f"  Std Dev: {gap.std():.3f}%")
    print(f"  Min: {gap.min():.3f}%")
    print(f"  Max: {gap.max():.3f}%")
    print(f"  Percentiles:")
    print(f"    16th (1 std below): {gap.quantile(0.16):.3f}%")
    print(f"    50th (median): {gap.quantile(0.50):.3f}%")
    print(f"    84th (1 std above): {gap.quantile(0.84):.3f}%")

    # Gap direction counts
    gap_up = (daily['gap_direction'] == 'up').sum()
    gap_down = (daily['gap_direction'] == 'down').sum()
    print(f"  Gap Up days: {gap_up} ({gap_up/len(gap)*100:.1f}%)")
    print(f"  Gap Down days: {gap_down} ({gap_down/len(gap)*100:.1f}%)")

    # Trend distribution
    print(f"\nPRIOR DAYS TREND (3-day lookback):")
    bullish = (daily['trend'] == 'bullish').sum()
    bearish = (daily['trend'] == 'bearish').sum()
    neutral = (daily['trend'] == 'neutral').sum()
    total = bullish + bearish + neutral
    print(f"  Bullish trend days: {bullish} ({bullish/total*100:.1f}%)")
    print(f"  Bearish trend days: {bearish} ({bearish/total*100:.1f}%)")
    print(f"  Neutral trend days: {neutral} ({neutral/total*100:.1f}%)")

    # Range distribution
    rng = daily['avg_range_pct'].dropna()
    print(f"\nAVERAGE DAILY RANGE % (5-day lookback):")
    print(f"  Mean: {rng.mean():.3f}%")
    print(f"  Std Dev: {rng.std():.3f}%")
    print(f"  Min: {rng.min():.3f}%")
    print(f"  Max: {rng.max():.3f}%")
    print(f"  Percentiles:")
    print(f"    16th: {rng.quantile(0.16):.3f}%")
    print(f"    50th (median): {rng.quantile(0.50):.3f}%")
    print(f"    68th: {rng.quantile(0.68):.3f}%")
    print(f"    84th: {rng.quantile(0.84):.3f}%")


def demonstrate_filters(daily: pd.DataFrame):
    """Show how each filter mode would affect trade counts."""
    total_days = len(daily.dropna(subset=['gap_pct']))
    gap = daily['gap_pct'].dropna()

    # Get percentile thresholds
    gap_p16 = gap.quantile(0.16)
    gap_p84 = gap.quantile(0.84)

    rng = daily['avg_range_pct'].dropna()
    range_p16 = rng.quantile(0.16)
    range_p50 = rng.quantile(0.50)
    range_p68 = rng.quantile(0.68)
    range_p84 = rng.quantile(0.84)

    print("\n" + "=" * 80)
    print("FILTER MODE EFFECTS")
    print("=" * 80)

    # Gap filter modes
    print(f"\nGAP FILTER MODES (using p16={gap_p16:.2f}%, p84={gap_p84:.2f}%):")

    middle_68 = ((daily['gap_pct'] >= gap_p16) & (daily['gap_pct'] <= gap_p84)).sum()
    exclude_68 = ((daily['gap_pct'] < gap_p16) | (daily['gap_pct'] > gap_p84)).sum()

    print(f"  'any' (no filter): {total_days} days ({100:.1f}%)")
    print(f"  'middle_68' (normal gaps): {middle_68} days ({middle_68/total_days*100:.1f}%)")
    print(f"  'exclude_middle_68' (extreme gaps): {exclude_68} days ({exclude_68/total_days*100:.1f}%)")
    print(f"  'directional' (gap aligns with trade): varies by trade direction")
    print(f"  'reverse_directional' (fade the gap): varies by trade direction")

    # Trend filter modes
    print(f"\nTREND FILTER MODES:")
    bullish_days = (daily['trend'] == 'bullish').sum()
    bearish_days = (daily['trend'] == 'bearish').sum()

    print(f"  'any' (no filter): {total_days} days ({100:.1f}%)")
    print(f"  'with_trend' longs only on bullish: {bullish_days} days ({bullish_days/total_days*100:.1f}%)")
    print(f"  'with_trend' shorts only on bearish: {bearish_days} days ({bearish_days/total_days*100:.1f}%)")
    print(f"  'counter_trend' longs on bearish: {bearish_days} days ({bearish_days/total_days*100:.1f}%)")
    print(f"  'counter_trend' shorts on bullish: {bullish_days} days ({bullish_days/total_days*100:.1f}%)")

    # Range filter modes
    print(f"\nRANGE FILTER MODES (p16={range_p16:.2f}%, p50={range_p50:.2f}%, p68={range_p68:.2f}%, p84={range_p84:.2f}%):")

    rng_data = daily['avg_range_pct'].dropna()
    middle_68_rng = ((rng_data >= range_p16) & (rng_data <= range_p84)).sum()
    above_68_rng = (rng_data > range_p68).sum()
    below_median_rng = (rng_data < range_p50).sum()
    middle_or_below = (rng_data <= range_p84).sum()

    print(f"  'any' (no filter): {len(rng_data)} days ({100:.1f}%)")
    print(f"  'middle_68' (normal volatility): {middle_68_rng} days ({middle_68_rng/len(rng_data)*100:.1f}%)")
    print(f"  'above_68' (high volatility): {above_68_rng} days ({above_68_rng/len(rng_data)*100:.1f}%)")
    print(f"  'below_median' (low volatility): {below_median_rng} days ({below_median_rng/len(rng_data)*100:.1f}%)")
    print(f"  'middle_68_or_below': {middle_or_below} days ({middle_or_below/len(rng_data)*100:.1f}%)")


def print_sample_days(daily: pd.DataFrame, n: int = 20):
    """Print sample days with all filter values."""
    print("\n" + "=" * 80)
    print(f"SAMPLE DAYS (most recent {n})")
    print("=" * 80)

    cols = ['date', 'gap_pct', 'gap_direction', 'bullish_count', 'trend', 'avg_range_pct']
    sample = daily[cols].dropna().tail(n)

    # Format for display
    sample = sample.copy()
    sample['date'] = sample['date'].dt.strftime('%Y-%m-%d')
    sample['gap_pct'] = sample['gap_pct'].apply(lambda x: f"{x:+.2f}%")
    sample['bullish_count'] = sample['bullish_count'].astype(int).astype(str) + "/3"
    sample['avg_range_pct'] = sample['avg_range_pct'].apply(lambda x: f"{x:.2f}%")

    sample.columns = ['Date', 'Gap %', 'Gap Dir', 'Bullish', 'Trend', 'Avg Range']

    print(sample.to_string(index=False))


def main():
    # Parse arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "market_data"

    print(f"\n{'=' * 80}")
    print(f"FILTER VERIFICATION FOR {ticker}")
    print(f"Data directory: {data_dir}")
    print(f"{'=' * 80}")

    # Load and process data
    print("\nLoading data...")
    daily = load_daily_data(data_dir, ticker)
    print(f"Loaded {len(daily)} trading days")

    # Calculate filter values
    print("Calculating filter values...")
    daily = calculate_filter_values(daily, trend_lookback=3, range_lookback=5)

    # Print results
    print_distribution_stats(daily)
    demonstrate_filters(daily)
    print_sample_days(daily, n=20)

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nTo verify against optimizer:")
    print("1. Run optimization with 'gap_filter_mode' = 'middle_68'")
    print("2. Check that trades only occur on days where gap is within p16-p84")
    print("3. Compare trade dates with the sample days above")


if __name__ == "__main__":
    main()
