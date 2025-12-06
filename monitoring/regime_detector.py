"""
Regime Detector - Identify market environment changes.

Detects volatility regimes, trend states, and correlation changes
to understand when optimized parameters may need adjustment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime classification."""
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    RANGING = "ranging"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"


class CorrelationRegime(Enum):
    """Correlation regime with market."""
    HIGH_POSITIVE = "high_positive"
    MODERATE_POSITIVE = "moderate_positive"
    UNCORRELATED = "uncorrelated"
    MODERATE_NEGATIVE = "moderate_negative"
    HIGH_NEGATIVE = "high_negative"


@dataclass
class MarketRegime:
    """Current market regime assessment."""
    volatility: VolatilityRegime
    trend: TrendRegime
    correlation: CorrelationRegime
    volatility_percentile: float  # 0-100, where current vol sits historically
    trend_strength: float  # ADX-like measure, 0-100
    correlation_value: float  # -1 to 1
    atr_value: float  # Actual ATR value
    atr_percent: float  # ATR as % of price
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'volatility': self.volatility.value,
            'trend': self.trend.value,
            'correlation': self.correlation.value,
            'volatility_percentile': self.volatility_percentile,
            'trend_strength': self.trend_strength,
            'correlation_value': self.correlation_value,
            'atr_value': self.atr_value,
            'atr_percent': self.atr_percent,
            'timestamp': self.timestamp.isoformat(),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Volatility: {self.volatility.value} ({self.volatility_percentile:.0f}th percentile), "
            f"Trend: {self.trend.value} (strength: {self.trend_strength:.0f}), "
            f"Correlation: {self.correlation.value} ({self.correlation_value:.2f})"
        )


@dataclass
class RegimeChange:
    """Record of a regime change."""
    timestamp: datetime
    regime_type: str  # "volatility", "trend", or "correlation"
    old_value: str
    new_value: str
    significance: float  # How significant is this change (0-1)


class RegimeDetector:
    """
    Detect and track market regime changes.

    Monitors volatility, trend, and correlation to identify
    when market conditions have shifted significantly.
    """

    def __init__(
        self,
        atr_period: int = 14,
        trend_period: int = 20,
        correlation_period: int = 20,
        lookback_days: int = 252,  # 1 year for percentile calculations
    ):
        """
        Initialize the regime detector.

        Args:
            atr_period: Period for ATR calculation
            trend_period: Period for trend detection (moving averages)
            correlation_period: Period for correlation calculation
            lookback_days: Days of history for percentile calculations
        """
        self.atr_period = atr_period
        self.trend_period = trend_period
        self.correlation_period = correlation_period
        self.lookback_days = lookback_days

        # Store price history
        self.price_history: List[Dict] = []  # {timestamp, open, high, low, close, volume}
        self.market_history: List[Dict] = []  # QQQ/SPY for correlation

        # Regime history
        self.regime_history: List[MarketRegime] = []
        self.regime_changes: List[RegimeChange] = []

        # Current regime
        self.current_regime: Optional[MarketRegime] = None

        # ATR history for percentile calculation
        self.atr_history: List[float] = []

    def add_bar(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0,
        market_close: float = None  # QQQ/SPY close for correlation
    ) -> Optional[MarketRegime]:
        """
        Add a price bar and update regime detection.

        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            market_close: Market index close (for correlation)

        Returns:
            Updated MarketRegime if enough data, None otherwise
        """
        self.price_history.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        })

        if market_close is not None:
            self.market_history.append({
                'timestamp': timestamp,
                'close': market_close,
            })

        # Need minimum data
        min_bars = max(self.atr_period, self.trend_period, self.correlation_period) + 5
        if len(self.price_history) < min_bars:
            return None

        # Detect current regime
        new_regime = self._detect_regime()

        # Check for regime changes
        if self.current_regime is not None:
            self._check_regime_changes(new_regime)

        self.current_regime = new_regime
        self.regime_history.append(new_regime)

        return new_regime

    def add_bars_bulk(self, bars: List[Dict], market_bars: List[Dict] = None) -> Optional[MarketRegime]:
        """
        Add multiple bars at once.

        Args:
            bars: List of bar dicts with timestamp, open, high, low, close, volume
            market_bars: Optional list of market index bars for correlation

        Returns:
            Final regime after all bars added
        """
        market_dict = {}
        if market_bars:
            for bar in market_bars:
                market_dict[bar['timestamp']] = bar['close']

        for bar in bars:
            market_close = market_dict.get(bar['timestamp'])
            self.add_bar(
                timestamp=bar['timestamp'],
                open_price=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar.get('volume', 0),
                market_close=market_close,
            )

        return self.current_regime

    def _detect_regime(self) -> MarketRegime:
        """Detect current market regime from price history."""
        # Calculate ATR
        atr, atr_percent = self._calculate_atr()
        self.atr_history.append(atr)

        # Volatility regime
        vol_regime, vol_percentile = self._classify_volatility(atr)

        # Trend regime
        trend_regime, trend_strength = self._detect_trend()

        # Correlation regime
        corr_regime, corr_value = self._detect_correlation()

        return MarketRegime(
            volatility=vol_regime,
            trend=trend_regime,
            correlation=corr_regime,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            correlation_value=corr_value,
            atr_value=atr,
            atr_percent=atr_percent,
        )

    def _calculate_atr(self) -> Tuple[float, float]:
        """Calculate Average True Range."""
        if len(self.price_history) < self.atr_period + 1:
            return 0.0, 0.0

        recent = self.price_history[-(self.atr_period + 1):]

        true_ranges = []
        for i in range(1, len(recent)):
            curr = recent[i]
            prev = recent[i - 1]

            tr = max(
                curr['high'] - curr['low'],
                abs(curr['high'] - prev['close']),
                abs(curr['low'] - prev['close'])
            )
            true_ranges.append(tr)

        atr = np.mean(true_ranges)
        current_price = recent[-1]['close']
        atr_percent = (atr / current_price) * 100 if current_price > 0 else 0

        return atr, atr_percent

    def _classify_volatility(self, atr: float) -> Tuple[VolatilityRegime, float]:
        """Classify volatility regime based on ATR percentile."""
        if len(self.atr_history) < 20:
            # Not enough history, use simple thresholds
            percentile = 50.0
        else:
            # Use historical percentile
            lookback = self.atr_history[-min(len(self.atr_history), self.lookback_days):]
            percentile = (sum(1 for x in lookback if x <= atr) / len(lookback)) * 100

        # Classify based on percentile
        if percentile <= 25:
            regime = VolatilityRegime.LOW
        elif percentile <= 50:
            regime = VolatilityRegime.MEDIUM
        elif percentile <= 85:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        return regime, percentile

    def _detect_trend(self) -> Tuple[TrendRegime, float]:
        """Detect trend using moving average slope and ADX-like measure."""
        if len(self.price_history) < self.trend_period + 10:
            return TrendRegime.RANGING, 0.0

        closes = [bar['close'] for bar in self.price_history[-self.trend_period - 10:]]

        # Calculate short and long moving averages
        short_ma = np.mean(closes[-self.trend_period // 2:])
        long_ma = np.mean(closes[-self.trend_period:])

        # Price position relative to MAs
        current_price = closes[-1]

        # MA slope (normalized)
        ma_slope = (short_ma - long_ma) / long_ma * 100 if long_ma > 0 else 0

        # Trend strength (simplified ADX-like)
        # Using price range vs average range
        recent_range = max(closes[-self.trend_period:]) - min(closes[-self.trend_period:])
        avg_price = np.mean(closes[-self.trend_period:])
        trend_strength = (recent_range / avg_price * 100) if avg_price > 0 else 0
        trend_strength = min(100, trend_strength * 5)  # Scale to 0-100

        # Classify trend
        if ma_slope > 1.0 and current_price > short_ma:
            regime = TrendRegime.STRONG_UP
        elif ma_slope > 0.3:
            regime = TrendRegime.WEAK_UP
        elif ma_slope < -1.0 and current_price < short_ma:
            regime = TrendRegime.STRONG_DOWN
        elif ma_slope < -0.3:
            regime = TrendRegime.WEAK_DOWN
        else:
            regime = TrendRegime.RANGING

        return regime, trend_strength

    def _detect_correlation(self) -> Tuple[CorrelationRegime, float]:
        """Detect correlation with market index."""
        if len(self.market_history) < self.correlation_period:
            return CorrelationRegime.UNCORRELATED, 0.0

        # Get recent closes for both
        ticker_closes = [bar['close'] for bar in self.price_history[-self.correlation_period:]]
        market_closes = [bar['close'] for bar in self.market_history[-self.correlation_period:]]

        if len(ticker_closes) != len(market_closes):
            return CorrelationRegime.UNCORRELATED, 0.0

        # Calculate returns
        ticker_returns = np.diff(ticker_closes) / ticker_closes[:-1]
        market_returns = np.diff(market_closes) / market_closes[:-1]

        # Calculate correlation
        if len(ticker_returns) < 5:
            return CorrelationRegime.UNCORRELATED, 0.0

        correlation = np.corrcoef(ticker_returns, market_returns)[0, 1]

        # Handle NaN
        if np.isnan(correlation):
            return CorrelationRegime.UNCORRELATED, 0.0

        # Classify
        if correlation > 0.7:
            regime = CorrelationRegime.HIGH_POSITIVE
        elif correlation > 0.3:
            regime = CorrelationRegime.MODERATE_POSITIVE
        elif correlation > -0.3:
            regime = CorrelationRegime.UNCORRELATED
        elif correlation > -0.7:
            regime = CorrelationRegime.MODERATE_NEGATIVE
        else:
            regime = CorrelationRegime.HIGH_NEGATIVE

        return regime, correlation

    def _check_regime_changes(self, new_regime: MarketRegime):
        """Check for and record regime changes."""
        old = self.current_regime

        # Volatility change
        if new_regime.volatility != old.volatility:
            self.regime_changes.append(RegimeChange(
                timestamp=new_regime.timestamp,
                regime_type="volatility",
                old_value=old.volatility.value,
                new_value=new_regime.volatility.value,
                significance=abs(new_regime.volatility_percentile - old.volatility_percentile) / 100,
            ))

        # Trend change
        if new_regime.trend != old.trend:
            self.regime_changes.append(RegimeChange(
                timestamp=new_regime.timestamp,
                regime_type="trend",
                old_value=old.trend.value,
                new_value=new_regime.trend.value,
                significance=abs(new_regime.trend_strength - old.trend_strength) / 100,
            ))

        # Correlation change
        if new_regime.correlation != old.correlation:
            self.regime_changes.append(RegimeChange(
                timestamp=new_regime.timestamp,
                regime_type="correlation",
                old_value=old.correlation.value,
                new_value=new_regime.correlation.value,
                significance=abs(new_regime.correlation_value - old.correlation_value),
            ))

    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get current market regime."""
        return self.current_regime

    def get_recent_changes(self, days: int = 30) -> List[RegimeChange]:
        """Get recent regime changes."""
        cutoff = datetime.now() - timedelta(days=days)
        return [c for c in self.regime_changes if c.timestamp > cutoff]

    def get_regime_at_date(self, date: datetime) -> Optional[MarketRegime]:
        """Get regime that was active at a specific date."""
        for regime in reversed(self.regime_history):
            if regime.timestamp <= date:
                return regime
        return None

    def export_regime_summary(self) -> Dict:
        """Export current regime and recent history."""
        return {
            'current': self.current_regime.to_dict() if self.current_regime else None,
            'recent_changes': [
                {
                    'timestamp': c.timestamp.isoformat(),
                    'type': c.regime_type,
                    'from': c.old_value,
                    'to': c.new_value,
                    'significance': c.significance,
                }
                for c in self.regime_changes[-10:]
            ],
            'history_length': len(self.regime_history),
        }


if __name__ == "__main__":
    # Test the regime detector
    import random

    print("Testing Regime Detector")
    print("=" * 50)

    detector = RegimeDetector(atr_period=14, trend_period=20)

    # Generate some price data
    random.seed(42)
    price = 100.0
    timestamp = datetime.now() - timedelta(days=100)

    for i in range(100):
        # Simulate daily bars with some trending behavior
        trend = 0.001 if i < 50 else -0.001  # Up first half, down second
        volatility = 0.02 if i > 70 else 0.01  # Higher vol in last 30 days

        change = random.gauss(trend, volatility)
        price *= (1 + change)

        high = price * (1 + abs(random.gauss(0, volatility)))
        low = price * (1 - abs(random.gauss(0, volatility)))
        open_price = price * (1 + random.gauss(0, volatility / 2))

        # Market (QQQ) with correlation
        market_price = 400 * (1 + change * 0.8 + random.gauss(0, 0.005))

        regime = detector.add_bar(
            timestamp=timestamp,
            open_price=open_price,
            high=max(open_price, high, price),
            low=min(open_price, low, price),
            close=price,
            market_close=market_price,
        )

        timestamp += timedelta(days=1)

    if detector.current_regime:
        print(f"\nCurrent Regime:")
        print(f"  {detector.current_regime.summary()}")
        print(f"\n  ATR: {detector.current_regime.atr_value:.2f}")
        print(f"  ATR %: {detector.current_regime.atr_percent:.2f}%")

    print(f"\nRegime Changes (last 30 days):")
    for change in detector.get_recent_changes(30):
        print(f"  {change.timestamp.date()}: {change.regime_type} "
              f"{change.old_value} -> {change.new_value}")
