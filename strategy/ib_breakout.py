"""
IB Breakout Strategy for Python Backtester.

This is the main strategy orchestrator that matches NinjaTrader's
QQQIBBreakout.cs logic exactly.

Key features:
- IB calculation matching NT logic
- Optional QQQ filter
- Configurable entries (exact breakout vs proximity)
- Configurable exits (target, stop, trailing, break-even, max bars, EOD)
- Day-of-week filters
- IB range filters
- Trading time window
"""

from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from data.data_types import (
    Bar, Position, InitialBalance, Trade, Signal, TradingSession,
    TradeDirection, PositionStatus, SignalType, ExitReason
)
from strategy.ib_calculator import IBCalculator, IBBreakoutDetector
from strategy.exits import ExitManager, ExitLevels


@dataclass
class StrategyState:
    """
    Current state of the strategy for a single ticker.

    Matches the state variables in QQQIBBreakout.cs.
    """
    # IB state
    ib: Optional[InitialBalance] = None
    ib_calculated: bool = False

    # Signal state
    signal_triggered: bool = False
    long_signal: bool = False
    signal_time: Optional[datetime] = None
    signal_price: float = 0.0

    # Trade state
    trade_taken_today: bool = False
    position: Optional[Position] = None

    # Bar tracking
    bars_in_position: int = 0

    def reset_daily(self):
        """Reset state for new session."""
        self.ib = None
        self.ib_calculated = False
        self.signal_triggered = False
        self.long_signal = False
        self.signal_time = None
        self.signal_price = 0.0
        self.trade_taken_today = False
        self.bars_in_position = 0


@dataclass
class StrategyParams:
    """
    All configurable strategy parameters.

    Defaults match NinjaTrader's QQQIBBreakout.cs defaults.
    """
    # IB parameters
    ib_duration_minutes: int = 30
    ib_proximity_percent: float = 0.0  # 0 = exact breakout

    # Entry parameters
    trade_direction: str = "both"  # "both", "long_only", "short_only"
    trading_start_time: str = "09:00"
    trading_end_time: str = "15:00"

    # Position sizing
    fixed_share_size: int = 100
    dollar_amount: float = 0.0  # 0 = use fixed_share_size

    # Exit parameters
    profit_target_percent: float = 0.5
    stop_loss_type: str = "opposite_ib"  # "opposite_ib" or "match_target"

    # Advanced exits
    trailing_stop_enabled: bool = False
    trailing_stop_atr_mult: float = 2.0
    break_even_enabled: bool = False
    break_even_pct: float = 0.7
    max_bars_enabled: bool = False
    max_bars: int = 60
    eod_exit_time: str = "15:55"

    # Filter parameters
    use_qqq_filter: bool = False
    min_ib_range_percent: float = 0.0
    max_ib_range_percent: float = 10.0
    max_breakout_time: str = "14:00"

    # Day of week filters
    trade_monday: bool = True
    trade_tuesday: bool = True
    trade_wednesday: bool = True
    trade_thursday: bool = True
    trade_friday: bool = True

    # Gap filters (open vs prior day close)
    gap_filter_enabled: bool = False
    min_gap_percent: float = -10.0  # Minimum gap % (negative = gap down allowed)
    max_gap_percent: float = 10.0   # Maximum gap % (positive = gap up allowed)
    gap_direction_filter: str = "any"  # "any", "gap_up_only", "gap_down_only", "with_trade"

    # Prior days trend filter (last N days close vs open)
    prior_days_filter_enabled: bool = False
    prior_days_lookback: int = 3  # Number of prior days to check
    prior_days_trend: str = "any"  # "any", "bullish", "bearish", "with_trade"

    # Daily range / volatility filter
    daily_range_filter_enabled: bool = False
    min_avg_daily_range_percent: float = 0.0  # Minimum average daily range %
    max_avg_daily_range_percent: float = 100.0  # Maximum average daily range %
    daily_range_lookback: int = 5  # Days to average for daily range


class IBBreakoutStrategy:
    """
    IB Breakout strategy implementation.

    This class processes bars and generates signals/orders matching
    NinjaTrader's QQQIBBreakout.cs behavior.
    """

    def __init__(self, params: Optional[StrategyParams] = None):
        """
        Initialize strategy.

        Args:
            params: Strategy parameters (uses defaults if None)
        """
        self.params = params or StrategyParams()

        # Initialize components - separate IB calculators for primary and filter tickers
        self.primary_ib_calc = IBCalculator(
            ib_duration_minutes=self.params.ib_duration_minutes,
            session_start="09:30"
        )
        self.qqq_ib_calc = IBCalculator(
            ib_duration_minutes=self.params.ib_duration_minutes,
            session_start="09:30"
        )

        self.breakout_detector = IBBreakoutDetector(
            ib_proximity_percent=self.params.ib_proximity_percent,
            allow_long=self.params.trade_direction in ["both", "long_only"],
            allow_short=self.params.trade_direction in ["both", "short_only"]
        )

        self.exit_manager = ExitManager(
            target_mode="percent",
            target_percent=self.params.profit_target_percent,
            stop_mode=self.params.stop_loss_type,
            trailing_stop_enabled=self.params.trailing_stop_enabled,
            trailing_stop_atr_mult=self.params.trailing_stop_atr_mult,
            break_even_enabled=self.params.break_even_enabled,
            break_even_trigger_pct=self.params.break_even_pct,
            max_bars_enabled=self.params.max_bars_enabled,
            max_bars=self.params.max_bars,
            eod_exit_time=self.params.eod_exit_time
        )

        # Parse time parameters
        self.trading_start = datetime.strptime(self.params.trading_start_time, "%H:%M").time()
        self.trading_end = datetime.strptime(self.params.trading_end_time, "%H:%M").time()
        self.max_breakout = datetime.strptime(self.params.max_breakout_time, "%H:%M").time()

        # State for primary ticker
        self.primary_state = StrategyState()

        # State for QQQ (if filter enabled)
        self.qqq_state = StrategyState()

        # Trade history
        self.trades: List[Trade] = []
        self.trade_counter = 0

    def reset(self):
        """Reset all strategy state."""
        self.primary_state = StrategyState()
        self.qqq_state = StrategyState()
        self.primary_ib_calc._reset_state()
        self.qqq_ib_calc._reset_state()
        self.trades = []
        self.trade_counter = 0

    def process_bar(
        self,
        bar: Bar,
        is_first_bar_of_session: bool,
        qqq_bar: Optional[Bar] = None
    ) -> Optional[Signal]:
        """
        Process a single bar and potentially generate a signal.

        This is the main entry point matching NinjaTrader's OnBarUpdate().

        Args:
            bar: Primary instrument bar
            is_first_bar_of_session: True if new session started
            qqq_bar: Optional QQQ bar for QQQ filter

        Returns:
            Signal if entry/exit generated, None otherwise
        """
        # Handle new session
        if is_first_bar_of_session:
            self._handle_new_session()

        # Process QQQ if filter enabled
        if self.params.use_qqq_filter and qqq_bar is not None:
            self._process_qqq_bar(qqq_bar, is_first_bar_of_session)

        # Process primary bar
        return self._process_primary_bar(bar, is_first_bar_of_session)

    def _handle_new_session(self):
        """Reset state for new trading session."""
        self.primary_state.reset_daily()
        self.qqq_state.reset_daily()
        self.primary_ib_calc._reset_state()
        self.qqq_ib_calc._reset_state()

    def _process_qqq_bar(self, bar: Bar, is_first_bar: bool):
        """
        Process QQQ bar for filter.

        Args:
            bar: QQQ bar
            is_first_bar: True if first bar of session
        """
        # Calculate QQQ IB using dedicated QQQ calculator
        ib_locked, ib = self.qqq_ib_calc.process_bar(bar, is_first_bar)

        if ib_locked:
            self.qqq_state.ib = ib
            self.qqq_state.ib_calculated = True

        # Check for QQQ breakout (only if IB calculated and no signal yet)
        if (self.qqq_state.ib_calculated and
            not self.qqq_state.signal_triggered and
            not self.qqq_state.trade_taken_today):

            breakout, is_long, desc = self.breakout_detector.check_breakout(
                bar, self.qqq_state.ib
            )

            if breakout:
                self.qqq_state.signal_triggered = True
                self.qqq_state.long_signal = is_long
                self.qqq_state.signal_time = bar.timestamp
                self.qqq_state.signal_price = bar.close

    def _process_primary_bar(
        self,
        bar: Bar,
        is_first_bar: bool
    ) -> Optional[Signal]:
        """
        Process primary instrument bar.

        Args:
            bar: Primary bar
            is_first_bar: True if first bar of session

        Returns:
            Signal if generated
        """
        # Calculate primary IB using dedicated primary calculator
        ib_locked, ib = self.primary_ib_calc.process_bar(bar, is_first_bar)

        if ib_locked:
            self.primary_state.ib = ib
            self.primary_state.ib_calculated = True

        # Update position if we have one
        if self.primary_state.position and not self.primary_state.position.is_flat():
            return self._process_position(bar)

        # Check for direct primary breakout (when QQQ filter disabled)
        if not self.params.use_qqq_filter:
            if (self.primary_state.ib_calculated and
                not self.primary_state.signal_triggered and
                not self.primary_state.trade_taken_today):

                breakout, is_long, desc = self.breakout_detector.check_breakout(
                    bar, self.primary_state.ib
                )

                if breakout:
                    self.primary_state.signal_triggered = True
                    self.primary_state.long_signal = is_long
                    self.primary_state.signal_time = bar.timestamp
                    self.primary_state.signal_price = bar.close

        # Check if we should enter (QQQ filter mode uses QQQ signal)
        if self.params.use_qqq_filter:
            signal_active = self.qqq_state.signal_triggered
            is_long = self.qqq_state.long_signal
        else:
            signal_active = self.primary_state.signal_triggered
            is_long = self.primary_state.long_signal

        # Entry logic
        if (signal_active and
            self.primary_state.ib_calculated and
            not self.primary_state.trade_taken_today and
            self._is_trading_time_allowed(bar.timestamp) and
            self._passes_filters(bar)):

            return self._generate_entry_signal(bar, is_long)

        return None

    def _process_position(self, bar: Bar) -> Optional[Signal]:
        """
        Process position for potential exits.

        Args:
            bar: Current bar

        Returns:
            Exit signal if position closed
        """
        position = self.primary_state.position
        self.primary_state.bars_in_position += 1

        # Update position price
        position.update_price(bar.close)
        position.bars_held = self.primary_state.bars_in_position

        # Update trailing stop if enabled
        if self.params.trailing_stop_enabled:
            new_trail = self.exit_manager.update_trailing_stop(
                position, bar.high, bar.low
            )
            if new_trail is not None:
                position.trailing_stop_price = new_trail

        # Check break-even trigger
        if self.params.break_even_enabled and not position.break_even_triggered:
            if self.exit_manager.check_break_even_trigger(position, bar.close):
                position.break_even_triggered = True
                position.stop_price = position.entry_price  # Move stop to entry

        # Check exit conditions
        should_exit, exit_reason, exit_price = self.exit_manager.check_exit_conditions(
            position, bar, self.primary_state.bars_in_position
        )

        if should_exit:
            return self._close_position(bar, exit_reason, exit_price)

        return None

    def _generate_entry_signal(self, bar: Bar, is_long: bool) -> Signal:
        """
        Generate entry signal and create position.

        Args:
            bar: Current bar
            is_long: True for long entry

        Returns:
            Entry signal
        """
        direction = TradeDirection.LONG if is_long else TradeDirection.SHORT

        # Calculate position size
        if self.params.dollar_amount > 0:
            quantity = int(self.params.dollar_amount / bar.close)
            quantity = max(1, quantity)
        else:
            quantity = self.params.fixed_share_size

        # Determine entry price (for backtesting, we use close)
        # In real trading, this would be the fill price
        entry_price = bar.close

        # Calculate exit levels
        exit_levels = self.exit_manager.calculate_exit_levels(
            entry_price=entry_price,
            direction=direction,
            ib=self.primary_state.ib
        )

        # Create position
        self.primary_state.position = Position(
            ticker=bar.ticker,
            status=PositionStatus.LONG if is_long else PositionStatus.SHORT,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=bar.timestamp,
            current_price=entry_price,
            stop_price=exit_levels.stop_price,
            target_price=exit_levels.target_price
        )

        self.primary_state.trade_taken_today = True
        self.primary_state.bars_in_position = 0

        # Create signal
        signal_type = SignalType.LONG_ENTRY if is_long else SignalType.SHORT_ENTRY
        signal = Signal(
            timestamp=bar.timestamp,
            ticker=bar.ticker,
            signal_type=signal_type,
            price=entry_price,
            ib=self.primary_state.ib,
            reason=f"{'Long' if is_long else 'Short'} entry at {entry_price:.2f}, Target: {exit_levels.target_price:.2f}, Stop: {exit_levels.stop_price:.2f}"
        )

        return signal

    def _close_position(
        self,
        bar: Bar,
        exit_reason: ExitReason,
        exit_price: float
    ) -> Signal:
        """
        Close position and record trade.

        Args:
            bar: Current bar
            exit_reason: Why position was closed
            exit_price: Exit price

        Returns:
            Exit signal
        """
        position = self.primary_state.position

        # Create trade record
        self.trade_counter += 1
        direction = TradeDirection.LONG if position.status == PositionStatus.LONG else TradeDirection.SHORT

        trade = Trade(
            trade_id=f"T{self.trade_counter:04d}",
            ticker=bar.ticker,
            direction=direction,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=bar.timestamp,
            exit_price=exit_price,
            quantity=position.quantity,
            exit_reason=exit_reason,
            ib=self.primary_state.ib,
            bars_held=self.primary_state.bars_in_position
        )
        trade.calculate_pnl()

        self.trades.append(trade)

        # Reset position
        self.primary_state.position = Position(ticker=bar.ticker)
        self.primary_state.bars_in_position = 0

        # Create exit signal
        signal = Signal(
            timestamp=bar.timestamp,
            ticker=bar.ticker,
            signal_type=SignalType.EXIT,
            price=exit_price,
            ib=self.primary_state.ib,
            reason=f"Exit {exit_reason.value}: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)"
        )

        return signal

    def _is_trading_time_allowed(self, timestamp: datetime) -> bool:
        """Check if current time is within trading window."""
        current_time = timestamp.time()
        return self.trading_start <= current_time <= self.trading_end

    def _passes_filters(self, bar: Bar) -> bool:
        """
        Check if trade passes all filters.

        Args:
            bar: Current bar

        Returns:
            True if all filters pass
        """
        # Day of week filter
        day = bar.timestamp.weekday()  # 0=Monday
        day_filters = [
            self.params.trade_monday,
            self.params.trade_tuesday,
            self.params.trade_wednesday,
            self.params.trade_thursday,
            self.params.trade_friday
        ]
        if not day_filters[day]:
            return False

        # IB range filter
        if self.primary_state.ib:
            ib_range_pct = self.primary_state.ib.ib_range_pct
            if ib_range_pct < self.params.min_ib_range_percent:
                return False
            if ib_range_pct > self.params.max_ib_range_percent:
                return False

        # Max breakout time filter
        if self.params.use_qqq_filter and self.qqq_state.signal_time:
            if self.qqq_state.signal_time.time() > self.max_breakout:
                return False
        elif self.primary_state.signal_time:
            if self.primary_state.signal_time.time() > self.max_breakout:
                return False

        return True

    def get_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.trades

    def get_trade_summary(self) -> Dict:
        """Get summary statistics of trades."""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

        winning = [t for t in self.trades if t.is_winner]
        losing = [t for t in self.trades if t.is_loser]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': sum(t.pnl for t in self.trades),
            'avg_pnl': sum(t.pnl for t in self.trades) / len(self.trades),
            'avg_winner': sum(t.pnl for t in winning) / len(winning) if winning else 0,
            'avg_loser': sum(t.pnl for t in losing) / len(losing) if losing else 0,
            'profit_factor': abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing and sum(t.pnl for t in losing) != 0 else float('inf')
        }


if __name__ == "__main__":
    # Test the strategy
    from data.data_loader import DataLoader
    from data.session_builder import SessionBuilder
    import os

    test_dir = r"C:\Users\Warren\Downloads"

    # Load data
    loader = DataLoader(test_dir)

    for filename in os.listdir(test_dir):
        if filename.endswith("_NT.txt") and "TSLA" in filename.upper():
            filepath = os.path.join(test_dir, filename)
            ticker = filename.split("_")[0].upper()

            try:
                df = loader.load_ninjatrader_file(filepath, ticker)

                # Build sessions
                session_builder = SessionBuilder(ib_duration_minutes=30)
                sessions = session_builder.build_sessions_from_dataframe(df, ticker)

                # Create strategy with default params
                params = StrategyParams(
                    use_qqq_filter=False,  # Direct IB breakout
                    profit_target_percent=0.5,
                    stop_loss_type="opposite_ib"
                )
                strategy = IBBreakoutStrategy(params)

                print(f"\nRunning strategy on {ticker}: {len(sessions)} sessions")

                # Process each session
                for session in sessions:
                    is_first = True
                    for bar in session.bars:
                        signal = strategy.process_bar(bar, is_first)
                        is_first = False

                        if signal:
                            print(f"  {signal}")

                # Print summary
                summary = strategy.get_trade_summary()
                print(f"\nStrategy Summary:")
                print(f"  Total Trades: {summary['total_trades']}")
                print(f"  Win Rate: {summary['win_rate']:.1f}%")
                print(f"  Total P&L: ${summary['total_pnl']:.2f}")
                print(f"  Avg P&L: ${summary['avg_pnl']:.2f}")
                print(f"  Profit Factor: {summary['profit_factor']:.2f}")

                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
