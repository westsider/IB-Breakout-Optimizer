"""
Feature Builder - Extract ML features from trades and market data.

Features include:
- IB characteristics (size, range %)
- Gap % (today's open vs yesterday's close)
- Prior days trend (bullish/bearish count)
- Daily range / volatility
- Time-based features (hour, day of week)
- Trade direction
- QQQ confirmation status
- Strategy parameters (profit target, stop type, etc.)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


@dataclass
class TradeFeatures:
    """Features extracted for a single trade."""
    # IB characteristics
    ib_range_percent: float  # IB range as % of IB low
    ib_duration_minutes: int  # IB window size

    # Gap features
    gap_percent: float  # (today open - yesterday close) / yesterday close * 100
    is_gap_up: bool
    is_gap_down: bool

    # Prior days trend
    prior_days_bullish_count: int  # How many of last N days closed > opened
    prior_days_lookback: int

    # Volatility
    avg_daily_range_percent: float  # Average (high-low)/low * 100 over last N days

    # Time features
    entry_hour: int  # Hour of entry (0-23)
    day_of_week: int  # 0=Monday, 4=Friday

    # Trade characteristics
    is_long: bool
    is_short: bool

    # QQQ filter
    qqq_confirmed: bool  # Whether QQQ broke its IB first

    # Distance from IB level at entry
    distance_from_ib_percent: float  # How far past IB level when entering

    # Strategy parameters (NEW)
    profit_target_percent: float  # Target as % of entry price
    stop_type_opposite_ib: bool  # Stop at opposite IB level
    stop_type_fixed: bool  # Fixed percentage stop
    stop_type_atr: bool  # ATR-based stop
    trailing_stop_enabled: bool
    break_even_enabled: bool

    # Target
    is_winner: bool  # True if trade was profitable
    pnl: float  # Actual P&L

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'ib_range_percent': self.ib_range_percent,
            'ib_duration_minutes': self.ib_duration_minutes,
            'gap_percent': self.gap_percent,
            'is_gap_up': int(self.is_gap_up),
            'is_gap_down': int(self.is_gap_down),
            'prior_days_bullish_count': self.prior_days_bullish_count,
            'prior_days_lookback': self.prior_days_lookback,
            'avg_daily_range_percent': self.avg_daily_range_percent,
            'entry_hour': self.entry_hour,
            'day_of_week': self.day_of_week,
            'is_long': int(self.is_long),
            'is_short': int(self.is_short),
            'qqq_confirmed': int(self.qqq_confirmed),
            'distance_from_ib_percent': self.distance_from_ib_percent,
            # Strategy parameters
            'profit_target_percent': self.profit_target_percent,
            'stop_type_opposite_ib': int(self.stop_type_opposite_ib),
            'stop_type_fixed': int(self.stop_type_fixed),
            'stop_type_atr': int(self.stop_type_atr),
            'trailing_stop_enabled': int(self.trailing_stop_enabled),
            'break_even_enabled': int(self.break_even_enabled),
            # Target
            'is_winner': int(self.is_winner),
            'pnl': self.pnl
        }


class FeatureBuilder:
    """
    Build ML features from backtest trades and market data.

    This class extracts features that can help predict trade outcomes.
    """

    # Feature columns used for training (excludes target and metadata)
    FEATURE_COLUMNS = [
        # Market condition features
        'ib_range_percent',
        'ib_duration_minutes',
        'gap_percent',
        'is_gap_up',
        'is_gap_down',
        'prior_days_bullish_count',
        'avg_daily_range_percent',
        'entry_hour',
        'day_of_week',
        'is_long',
        'is_short',
        'qqq_confirmed',
        'distance_from_ib_percent',
        # Strategy parameter features
        'profit_target_percent',
        'stop_type_opposite_ib',
        'stop_type_fixed',
        'stop_type_atr',
        'trailing_stop_enabled',
        'break_even_enabled',
    ]

    TARGET_COLUMN = 'is_winner'

    def __init__(self, prior_days_lookback: int = 3, daily_range_lookback: int = 5):
        """
        Initialize feature builder.

        Args:
            prior_days_lookback: Number of prior days to check for trend
            daily_range_lookback: Number of days to average for daily range
        """
        self.prior_days_lookback = prior_days_lookback
        self.daily_range_lookback = daily_range_lookback

    def build_features_from_backtest(
        self,
        trades: List[Dict[str, Any]],
        bars: np.ndarray,
        timestamps: np.ndarray,
        ib_duration_minutes: int = 30,
        qqq_filter_used: bool = False,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Build feature DataFrame from backtest trades.

        Args:
            trades: List of trade dictionaries from backtest
            bars: OHLCV bar data (N x 5 array: open, high, low, close, volume)
            timestamps: Timestamps for each bar
            ib_duration_minutes: IB window size used
            qqq_filter_used: Whether QQQ filter was enabled
            strategy_params: Strategy parameters used for the backtest (optional)
                - profit_target_percent: float
                - stop_loss_type: str ('opposite_ib', 'fixed_percent', 'atr')
                - trailing_stop_enabled: bool
                - break_even_enabled: bool

        Returns:
            DataFrame with features for each trade
        """
        # First, compute daily OHLC from minute bars
        daily_data = self._compute_daily_ohlc(bars, timestamps)

        # Default strategy params if not provided
        if strategy_params is None:
            strategy_params = {}

        features_list = []

        for trade in trades:
            try:
                features = self._extract_trade_features(
                    trade, bars, timestamps, daily_data,
                    ib_duration_minutes, qqq_filter_used, strategy_params
                )
                if features is not None:
                    features_list.append(features.to_dict())
            except Exception as e:
                # Skip trades where feature extraction fails
                continue

        if not features_list:
            return pd.DataFrame()

        return pd.DataFrame(features_list)

    def _compute_daily_ohlc(
        self,
        bars: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute daily OHLC from minute bars.

        Returns dict mapping date string to {open, high, low, close}.
        """
        daily_data = {}

        for i, ts in enumerate(timestamps):
            if isinstance(ts, (int, float, np.integer, np.floating)):
                dt = datetime.fromtimestamp(ts)
            else:
                dt = pd.Timestamp(ts).to_pydatetime()

            # Only use regular trading hours (9:30-16:00)
            hour_minute = dt.hour * 60 + dt.minute
            if hour_minute < 570 or hour_minute > 960:  # Before 9:30 or after 16:00
                continue

            date_str = dt.strftime('%Y-%m-%d')

            if date_str not in daily_data:
                daily_data[date_str] = {
                    'open': bars[i, 0],
                    'high': bars[i, 1],
                    'low': bars[i, 2],
                    'close': bars[i, 3],
                    'open_set': True
                }
            else:
                daily_data[date_str]['high'] = max(daily_data[date_str]['high'], bars[i, 1])
                daily_data[date_str]['low'] = min(daily_data[date_str]['low'], bars[i, 2])
                daily_data[date_str]['close'] = bars[i, 3]

        return daily_data

    def _extract_trade_features(
        self,
        trade: Dict[str, Any],
        bars: np.ndarray,
        timestamps: np.ndarray,
        daily_data: Dict[str, Dict[str, float]],
        ib_duration_minutes: int,
        qqq_filter_used: bool,
        strategy_params: Dict[str, Any]
    ) -> Optional[TradeFeatures]:
        """Extract features for a single trade."""

        # Get trade entry time
        entry_time = trade.get('entry_time') or trade.get('entry_timestamp')
        if entry_time is None:
            return None

        if isinstance(entry_time, str):
            entry_dt = pd.Timestamp(entry_time).to_pydatetime()
        elif isinstance(entry_time, (int, float)):
            entry_dt = datetime.fromtimestamp(entry_time)
        else:
            entry_dt = pd.Timestamp(entry_time).to_pydatetime()

        trade_date = entry_dt.strftime('%Y-%m-%d')

        # Get sorted dates for lookback
        sorted_dates = sorted(daily_data.keys())
        if trade_date not in sorted_dates:
            return None

        date_idx = sorted_dates.index(trade_date)

        # Get today's daily data
        today_data = daily_data.get(trade_date)
        if today_data is None:
            return None

        # Calculate IB range % (use trade's IB if available, else estimate)
        ib_high = trade.get('ib_high', today_data['high'])
        ib_low = trade.get('ib_low', today_data['low'])
        if ib_low > 0:
            ib_range_percent = (ib_high - ib_low) / ib_low * 100
        else:
            ib_range_percent = 0

        # Calculate gap %
        gap_percent = 0.0
        is_gap_up = False
        is_gap_down = False
        if date_idx > 0:
            yesterday_date = sorted_dates[date_idx - 1]
            yesterday_close = daily_data[yesterday_date]['close']
            today_open = today_data['open']
            if yesterday_close > 0:
                gap_percent = (today_open - yesterday_close) / yesterday_close * 100
                is_gap_up = gap_percent > 0.1
                is_gap_down = gap_percent < -0.1

        # Calculate prior days bullish count
        prior_bullish = 0
        lookback_start = max(0, date_idx - self.prior_days_lookback)
        for i in range(lookback_start, date_idx):
            d = sorted_dates[i]
            if daily_data[d]['close'] > daily_data[d]['open']:
                prior_bullish += 1

        # Calculate average daily range
        avg_range = 0.0
        range_lookback_start = max(0, date_idx - self.daily_range_lookback)
        range_count = 0
        for i in range(range_lookback_start, date_idx):
            d = sorted_dates[i]
            day_low = daily_data[d]['low']
            if day_low > 0:
                day_range = (daily_data[d]['high'] - day_low) / day_low * 100
                avg_range += day_range
                range_count += 1
        if range_count > 0:
            avg_range /= range_count

        # Time features
        entry_hour = entry_dt.hour
        day_of_week = entry_dt.weekday()

        # Trade direction
        direction = trade.get('direction', trade.get('side', 'long'))
        is_long = direction.lower() in ('long', 'buy')
        is_short = direction.lower() in ('short', 'sell')

        # Distance from IB at entry
        entry_price = trade.get('entry_price', 0)
        if is_long and ib_high > 0 and entry_price > 0:
            distance_from_ib = (entry_price - ib_high) / ib_high * 100
        elif is_short and ib_low > 0 and entry_price > 0:
            distance_from_ib = (ib_low - entry_price) / ib_low * 100
        else:
            distance_from_ib = 0

        # Strategy parameters
        profit_target = strategy_params.get('profit_target_percent', 1.0)
        stop_type = strategy_params.get('stop_loss_type', 'opposite_ib')
        stop_type_opposite_ib = stop_type == 'opposite_ib'
        stop_type_fixed = stop_type == 'fixed_percent'
        stop_type_atr = stop_type == 'atr'
        trailing_stop_enabled = strategy_params.get('trailing_stop_enabled', False)
        break_even_enabled = strategy_params.get('break_even_enabled', False)

        # Trade outcome
        pnl = trade.get('pnl', trade.get('profit', 0))
        is_winner = pnl > 0

        return TradeFeatures(
            ib_range_percent=ib_range_percent,
            ib_duration_minutes=ib_duration_minutes,
            gap_percent=gap_percent,
            is_gap_up=is_gap_up,
            is_gap_down=is_gap_down,
            prior_days_bullish_count=prior_bullish,
            prior_days_lookback=self.prior_days_lookback,
            avg_daily_range_percent=avg_range,
            entry_hour=entry_hour,
            day_of_week=day_of_week,
            is_long=is_long,
            is_short=is_short,
            qqq_confirmed=qqq_filter_used,
            distance_from_ib_percent=distance_from_ib,
            profit_target_percent=profit_target,
            stop_type_opposite_ib=stop_type_opposite_ib,
            stop_type_fixed=stop_type_fixed,
            stop_type_atr=stop_type_atr,
            trailing_stop_enabled=trailing_stop_enabled,
            break_even_enabled=break_even_enabled,
            is_winner=is_winner,
            pnl=pnl
        )

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature matrix X and target vector y from DataFrame.

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        X = df[self.FEATURE_COLUMNS].values
        y = df[self.TARGET_COLUMN].values
        return X, y

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.FEATURE_COLUMNS.copy()
