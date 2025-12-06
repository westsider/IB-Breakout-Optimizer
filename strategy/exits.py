"""
Exit Management for IB Breakout Optimizer.

Handles all exit logic including:
- Profit targets (percent, IB multiple, ATR-based)
- Stop losses (opposite IB, match target, ATR-based, percent)
- Trailing stops
- Break-even stops
- Max bars exit
- End-of-day exit
"""

from datetime import datetime, time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from data.data_types import (
    Bar, Position, InitialBalance, Trade, ExitReason,
    TradeDirection, StopLossType, TargetMode, PositionStatus
)


@dataclass
class ExitLevels:
    """
    Exit price levels for a position.

    Attributes:
        target_price: Profit target price
        stop_price: Initial stop loss price
        trailing_stop_price: Current trailing stop (if enabled)
        break_even_stop: Break-even stop price (if triggered)
    """
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    break_even_stop: Optional[float] = None

    def get_active_stop(self) -> Optional[float]:
        """Get the most restrictive active stop price."""
        stops = [s for s in [self.stop_price, self.trailing_stop_price, self.break_even_stop] if s is not None]
        if not stops:
            return None
        # For long, highest stop is most restrictive
        # This method needs position context - handled in ExitManager
        return stops[0] if len(stops) == 1 else None


class ExitManager:
    """
    Manages exit logic for positions.

    Matches NinjaTrader's exit behavior from QQQIBBreakout.cs.
    """

    def __init__(
        self,
        # Target parameters
        target_mode: str = "percent",
        target_percent: float = 0.5,
        target_ib_multiple: float = 1.0,
        target_atr_multiple: float = 2.0,

        # Stop parameters
        stop_mode: str = "opposite_ib",
        stop_percent: float = 0.5,
        stop_atr_multiple: float = 1.5,

        # Trailing stop
        trailing_stop_enabled: bool = False,
        trailing_stop_atr_mult: float = 2.0,
        trailing_stop_percent: float = 1.0,

        # Break-even stop
        break_even_enabled: bool = False,
        break_even_trigger_pct: float = 0.7,  # Move to BE when 70% to target

        # Max bars exit
        max_bars_enabled: bool = False,
        max_bars: int = 60,

        # EOD exit
        eod_exit_time: str = "15:55",

        # Tick size for rounding
        tick_size: float = 0.01
    ):
        """
        Initialize exit manager.

        Args:
            target_mode: "percent", "ib_multiple", or "atr"
            target_percent: Target as % of entry price
            target_ib_multiple: Target as multiple of IB range
            target_atr_multiple: Target as multiple of ATR

            stop_mode: "opposite_ib", "match_target", "percent", or "atr"
            stop_percent: Stop as % of entry price
            stop_atr_multiple: Stop as multiple of ATR

            trailing_stop_enabled: Enable trailing stop
            trailing_stop_atr_mult: Trail distance as ATR multiple
            trailing_stop_percent: Trail distance as percentage

            break_even_enabled: Enable break-even stop
            break_even_trigger_pct: Trigger BE at this % to target

            max_bars_enabled: Enable max bars exit
            max_bars: Exit after N bars

            eod_exit_time: Force exit time (HH:MM)

            tick_size: Minimum price increment
        """
        self.target_mode = target_mode
        self.target_percent = target_percent
        self.target_ib_multiple = target_ib_multiple
        self.target_atr_multiple = target_atr_multiple

        self.stop_mode = stop_mode
        self.stop_percent = stop_percent
        self.stop_atr_multiple = stop_atr_multiple

        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.trailing_stop_percent = trailing_stop_percent

        self.break_even_enabled = break_even_enabled
        self.break_even_trigger_pct = break_even_trigger_pct

        self.max_bars_enabled = max_bars_enabled
        self.max_bars = max_bars

        self.eod_exit_time = datetime.strptime(eod_exit_time, "%H:%M").time()

        self.tick_size = tick_size

    def round_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self.tick_size) * self.tick_size

    def calculate_exit_levels(
        self,
        entry_price: float,
        direction: TradeDirection,
        ib: InitialBalance,
        atr: Optional[float] = None
    ) -> ExitLevels:
        """
        Calculate initial exit levels for a new position.

        This matches NinjaTrader's OnExecutionUpdate logic.

        Args:
            entry_price: Position entry price
            direction: Long or short
            ib: Initial Balance levels
            atr: ATR value (required for ATR-based exits)

        Returns:
            ExitLevels with target and stop prices
        """
        is_long = direction == TradeDirection.LONG

        # Calculate target price
        target_price = self._calculate_target(entry_price, is_long, ib, atr)

        # Calculate stop price
        stop_price = self._calculate_stop(entry_price, target_price, is_long, ib, atr)

        return ExitLevels(
            target_price=target_price,
            stop_price=stop_price
        )

    def _calculate_target(
        self,
        entry_price: float,
        is_long: bool,
        ib: InitialBalance,
        atr: Optional[float]
    ) -> float:
        """Calculate profit target price."""

        if self.target_mode == "percent":
            # Target as percentage of entry price
            if is_long:
                target = entry_price * (1 + self.target_percent / 100.0)
            else:
                target = entry_price * (1 - self.target_percent / 100.0)

        elif self.target_mode == "ib_multiple":
            # Target as multiple of IB range
            target_distance = ib.ib_range * self.target_ib_multiple
            if is_long:
                target = entry_price + target_distance
            else:
                target = entry_price - target_distance

        elif self.target_mode == "atr" and atr is not None:
            # Target as multiple of ATR
            target_distance = atr * self.target_atr_multiple
            if is_long:
                target = entry_price + target_distance
            else:
                target = entry_price - target_distance

        else:
            # Fallback to percent
            if is_long:
                target = entry_price * (1 + self.target_percent / 100.0)
            else:
                target = entry_price * (1 - self.target_percent / 100.0)

        return self.round_to_tick(target)

    def _calculate_stop(
        self,
        entry_price: float,
        target_price: float,
        is_long: bool,
        ib: InitialBalance,
        atr: Optional[float]
    ) -> float:
        """Calculate initial stop loss price."""

        if self.stop_mode == "opposite_ib":
            # Stop at opposite IB level (NinjaTrader default)
            if is_long:
                stop = ib.ib_low
            else:
                stop = ib.ib_high

        elif self.stop_mode == "match_target":
            # Stop distance equals target distance
            target_distance = abs(target_price - entry_price)
            if is_long:
                stop = entry_price - target_distance
            else:
                stop = entry_price + target_distance

        elif self.stop_mode == "percent":
            # Stop as percentage of entry price
            if is_long:
                stop = entry_price * (1 - self.stop_percent / 100.0)
            else:
                stop = entry_price * (1 + self.stop_percent / 100.0)

        elif self.stop_mode == "atr" and atr is not None:
            # Stop as multiple of ATR
            stop_distance = atr * self.stop_atr_multiple
            if is_long:
                stop = entry_price - stop_distance
            else:
                stop = entry_price + stop_distance

        else:
            # Fallback to opposite IB
            if is_long:
                stop = ib.ib_low
            else:
                stop = ib.ib_high

        return self.round_to_tick(stop)

    def update_trailing_stop(
        self,
        position: Position,
        current_high: float,
        current_low: float,
        atr: Optional[float] = None
    ) -> Optional[float]:
        """
        Update trailing stop based on new prices.

        Args:
            position: Current position
            current_high: Current bar high
            current_low: Current bar low
            atr: ATR for ATR-based trailing

        Returns:
            New trailing stop price, or None if no update
        """
        if not self.trailing_stop_enabled:
            return None

        is_long = position.status == PositionStatus.LONG

        # Calculate trail distance
        if atr is not None:
            trail_distance = atr * self.trailing_stop_atr_mult
        else:
            trail_distance = position.entry_price * (self.trailing_stop_percent / 100.0)

        if is_long:
            # For longs, trail below the high
            new_trail = current_high - trail_distance

            # Only update if higher than current trailing stop
            if position.trailing_stop_price is None or new_trail > position.trailing_stop_price:
                return self.round_to_tick(new_trail)
        else:
            # For shorts, trail above the low
            new_trail = current_low + trail_distance

            # Only update if lower than current trailing stop
            if position.trailing_stop_price is None or new_trail < position.trailing_stop_price:
                return self.round_to_tick(new_trail)

        return None

    def check_break_even_trigger(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """
        Check if break-even stop should be triggered.

        Args:
            position: Current position
            current_price: Current price

        Returns:
            True if break-even should be triggered
        """
        if not self.break_even_enabled or position.break_even_triggered:
            return False

        if position.target_price is None:
            return False

        is_long = position.status == PositionStatus.LONG

        # Calculate distance to target
        total_distance = abs(position.target_price - position.entry_price)
        current_distance = current_price - position.entry_price if is_long else position.entry_price - current_price

        # Check if we've moved X% toward target
        if total_distance > 0:
            progress_pct = current_distance / total_distance

            if progress_pct >= self.break_even_trigger_pct:
                return True

        return False

    def check_exit_conditions(
        self,
        position: Position,
        bar: Bar,
        bars_held: int,
        atr: Optional[float] = None
    ) -> Tuple[bool, Optional[ExitReason], Optional[float]]:
        """
        Check all exit conditions for a position.

        Args:
            position: Current position
            bar: Current bar
            bars_held: Number of bars position has been held
            atr: ATR value for ATR-based exits

        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        is_long = position.status == PositionStatus.LONG

        # Check EOD exit first (highest priority)
        if bar.timestamp.time() >= self.eod_exit_time:
            return (True, ExitReason.EOD_EXIT, bar.close)

        # Check max bars exit
        if self.max_bars_enabled and bars_held >= self.max_bars:
            return (True, ExitReason.MAX_BARS, bar.close)

        # Get active stop level
        active_stop = self._get_active_stop(position, is_long)

        # Check stop loss
        if active_stop is not None:
            if is_long and bar.low <= active_stop:
                # Stop triggered - use stop price or low if gapped through
                exit_price = max(bar.open, active_stop) if bar.open >= active_stop else bar.open
                reason = self._get_stop_reason(position, active_stop)
                return (True, reason, exit_price)

            elif not is_long and bar.high >= active_stop:
                # Short stop triggered
                exit_price = min(bar.open, active_stop) if bar.open <= active_stop else bar.open
                reason = self._get_stop_reason(position, active_stop)
                return (True, reason, exit_price)

        # Check profit target
        if position.target_price is not None:
            if is_long and bar.high >= position.target_price:
                # Target hit - use target price or open if gapped through
                exit_price = min(bar.open, position.target_price) if bar.open <= position.target_price else position.target_price
                return (True, ExitReason.PROFIT_TARGET, exit_price)

            elif not is_long and bar.low <= position.target_price:
                # Short target hit
                exit_price = max(bar.open, position.target_price) if bar.open >= position.target_price else position.target_price
                return (True, ExitReason.PROFIT_TARGET, exit_price)

        return (False, None, None)

    def _get_active_stop(self, position: Position, is_long: bool) -> Optional[float]:
        """Get the most restrictive active stop price."""
        stops = []

        if position.stop_price is not None:
            stops.append(position.stop_price)
        if position.trailing_stop_price is not None:
            stops.append(position.trailing_stop_price)

        if not stops:
            return None

        # For longs, use the highest stop (most protective)
        # For shorts, use the lowest stop
        if is_long:
            return max(stops)
        else:
            return min(stops)

    def _get_stop_reason(self, position: Position, triggered_stop: float) -> ExitReason:
        """Determine which type of stop was triggered."""
        if position.trailing_stop_price == triggered_stop:
            return ExitReason.TRAILING_STOP
        elif position.break_even_triggered:
            return ExitReason.BREAK_EVEN_STOP
        else:
            return ExitReason.STOP_LOSS


if __name__ == "__main__":
    # Test exit manager
    from data.data_types import Position, InitialBalance, TradeDirection, PositionStatus

    # Create test IB
    ib = InitialBalance(
        date=datetime.now().date(),
        ticker="TSLA",
        session_start=datetime.now(),
        ib_end_time=datetime.now(),
        ib_high=250.00,
        ib_low=245.00,
        is_calculated=True
    )
    ib._calculate_range()

    # Test with default settings
    exit_mgr = ExitManager(
        target_mode="percent",
        target_percent=0.5,
        stop_mode="opposite_ib"
    )

    # Calculate exit levels for long entry
    entry_price = 250.50
    levels = exit_mgr.calculate_exit_levels(
        entry_price=entry_price,
        direction=TradeDirection.LONG,
        ib=ib
    )

    print("Exit Levels Test (LONG entry at $250.50):")
    print(f"  Target: ${levels.target_price:.2f} ({(levels.target_price/entry_price - 1)*100:.2f}%)")
    print(f"  Stop: ${levels.stop_price:.2f} (IB Low at ${ib.ib_low:.2f})")

    # Test with match_target stop
    exit_mgr2 = ExitManager(
        target_mode="percent",
        target_percent=0.5,
        stop_mode="match_target"
    )

    levels2 = exit_mgr2.calculate_exit_levels(
        entry_price=entry_price,
        direction=TradeDirection.LONG,
        ib=ib
    )

    print(f"\nWith match_target stop:")
    print(f"  Target: ${levels2.target_price:.2f}")
    print(f"  Stop: ${levels2.stop_price:.2f} (matches target distance)")
