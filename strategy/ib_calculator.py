"""
Initial Balance (IB) Calculator for IB Breakout Optimizer.

This module calculates IB levels to EXACTLY match NinjaTrader's logic.

Key matching requirements from QQQIBBreakout.cs:
1. IB period starts at session start (e.g., 9:30 AM)
2. IB period ends after IBDurationMinutes (default 30 min, so 10:00 AM)
3. Bars STRICTLY BEFORE IB end time are included in IB calculation
4. The bar TIMESTAMPED at IB end time (e.g., 10:00) is the FIRST bar AFTER IB
5. IB is "locked in" on the first bar at or after IB end time
6. Breakout can occur on the same bar that locks in IB

From NinjaTrader code:
```csharp
// Calculate PRIMARY IB during IB period (STRICTLY BEFORE IB end time)
if (Time[0] >= primarySessionStart && Time[0] < primaryIbEndTime && !primaryIbCalculated)
{
    if (High[0] > primaryIbHigh)
        primaryIbHigh = High[0];
    if (Low[0] < primaryIbLow)
        primaryIbLow = Low[0];
}

// Lock in PRIMARY IB on the FIRST bar AT or AFTER IB end time
if (Time[0] >= primaryIbEndTime && !primaryIbCalculated && primaryIbHigh > 0)
{
    primaryIbCalculated = true;
}
```
"""

from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from data.data_types import Bar, InitialBalance, TradingSession


class IBCalculator:
    """
    Calculates Initial Balance levels matching NinjaTrader logic.

    The IB is calculated during the first N minutes of a trading session.
    For a 30-minute IB starting at 9:30 AM:
    - IB period includes bars from 9:30:00 to 9:59:00 (timestamps)
    - Bar timestamped 10:00:00 is the FIRST bar AFTER IB
    - IB is locked in when the 10:00:00 bar arrives
    """

    def __init__(
        self,
        ib_duration_minutes: int = 30,
        session_start: str = "09:30"
    ):
        """
        Initialize IB Calculator.

        Args:
            ib_duration_minutes: Duration of IB period in minutes
            session_start: Session start time (HH:MM)
        """
        self.ib_duration_minutes = ib_duration_minutes
        self.session_start = datetime.strptime(session_start, "%H:%M").time()

        # State for tracking current IB calculation
        self._reset_state()

    def _reset_state(self):
        """Reset internal state for a new session."""
        self.current_ib_high = 0.0
        self.current_ib_low = float('inf')
        self.is_calculated = False
        self.current_session_start: Optional[datetime] = None
        self.current_ib_end_time: Optional[datetime] = None

    def get_ib_end_time(self, session_start: datetime) -> datetime:
        """
        Calculate IB end time from session start.

        Args:
            session_start: Session start datetime

        Returns:
            IB end datetime
        """
        return session_start + timedelta(minutes=self.ib_duration_minutes)

    def process_bar(self, bar: Bar, is_first_bar_of_session: bool = False) -> Tuple[bool, Optional[InitialBalance]]:
        """
        Process a single bar for IB calculation.

        This is the core method that matches NinjaTrader's OnBarUpdate logic.

        Args:
            bar: Current bar
            is_first_bar_of_session: Whether this is the first bar of a new session

        Returns:
            Tuple of (ib_just_locked_in, InitialBalance if locked)

        The returned InitialBalance is only non-None when IB is locked in
        on this bar (i.e., the first bar at or after IB end time).
        """
        # Handle new session
        if is_first_bar_of_session:
            self._reset_state()
            self.current_session_start = datetime.combine(
                bar.timestamp.date(),
                self.session_start
            )
            self.current_ib_end_time = self.get_ib_end_time(self.current_session_start)

        # Skip if session not initialized
        if self.current_session_start is None:
            return (False, None)

        bar_time = bar.timestamp

        # Calculate IB during IB period (STRICTLY BEFORE IB end time)
        # This matches: Time[0] >= primarySessionStart && Time[0] < primaryIbEndTime
        if (bar_time >= self.current_session_start and
            bar_time < self.current_ib_end_time and
            not self.is_calculated):

            if bar.high > self.current_ib_high:
                self.current_ib_high = bar.high
            if bar.low < self.current_ib_low:
                self.current_ib_low = bar.low

            return (False, None)

        # Lock in IB on the FIRST bar AT or AFTER IB end time
        # This matches: Time[0] >= primaryIbEndTime && !primaryIbCalculated && primaryIbHigh > 0
        if (bar_time >= self.current_ib_end_time and
            not self.is_calculated and
            self.current_ib_high > 0):

            self.is_calculated = True

            # Create InitialBalance object
            ib = InitialBalance(
                date=bar.timestamp.date(),
                ticker=bar.ticker,
                session_start=self.current_session_start,
                ib_end_time=self.current_ib_end_time,
                ib_high=self.current_ib_high,
                ib_low=self.current_ib_low,
                is_calculated=True
            )
            ib._calculate_range()

            return (True, ib)

        # IB already calculated, nothing to do
        return (False, None)

    def get_current_ib(self) -> Optional[InitialBalance]:
        """
        Get current IB state (even if not finalized).

        Returns:
            Current InitialBalance state, or None if not started
        """
        if self.current_session_start is None:
            return None

        ib = InitialBalance(
            date=self.current_session_start.date(),
            ticker="",
            session_start=self.current_session_start,
            ib_end_time=self.current_ib_end_time,
            ib_high=self.current_ib_high,
            ib_low=self.current_ib_low if self.current_ib_low < float('inf') else 0.0,
            is_calculated=self.is_calculated
        )
        ib._calculate_range()
        return ib

    def calculate_ib_for_session(self, session: TradingSession) -> InitialBalance:
        """
        Calculate IB for an entire session.

        Args:
            session: Trading session with bars

        Returns:
            InitialBalance for the session
        """
        self._reset_state()

        result_ib = None

        for i, bar in enumerate(session.bars):
            is_first = (i == 0)
            ib_locked, ib = self.process_bar(bar, is_first)

            if ib_locked:
                result_ib = ib

        # If IB was never locked (session ended early), return current state
        if result_ib is None:
            result_ib = self.get_current_ib()
            if result_ib:
                result_ib.ticker = session.ticker

        return result_ib

    def calculate_ibs_for_all_sessions(
        self,
        sessions: List[TradingSession]
    ) -> Dict[datetime, InitialBalance]:
        """
        Calculate IB for multiple sessions.

        Args:
            sessions: List of trading sessions

        Returns:
            Dict mapping date -> InitialBalance
        """
        ibs = {}
        for session in sessions:
            ib = self.calculate_ib_for_session(session)
            if ib:
                ibs[session.date] = ib
                # Also store IB in session
                session.ib = ib
        return ibs


class IBBreakoutDetector:
    """
    Detects IB breakouts matching NinjaTrader logic.

    Breakout conditions (from QQQIBBreakout.cs):
    - Long: Close > IB High (exact breakout) or Close >= IB High * (1 - proximity%) (proximity)
    - Short: Close < IB Low (exact breakout) or Close <= IB Low * (1 + proximity%) (proximity)
    """

    def __init__(
        self,
        ib_proximity_percent: float = 0.0,
        allow_long: bool = True,
        allow_short: bool = True
    ):
        """
        Initialize breakout detector.

        Args:
            ib_proximity_percent: Trigger when within X% of IB level (0 = exact breakout)
            allow_long: Whether to detect long breakouts
            allow_short: Whether to detect short breakouts
        """
        self.ib_proximity_percent = ib_proximity_percent
        self.allow_long = allow_long
        self.allow_short = allow_short

    def check_breakout(
        self,
        bar: Bar,
        ib: InitialBalance
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Check if bar breaks out of IB.

        Args:
            bar: Current bar
            ib: Initial Balance levels

        Returns:
            Tuple of (breakout_occurred, is_long, description)
        """
        if not ib.is_calculated:
            return (False, False, None)

        close = bar.close

        # Calculate trigger levels
        if self.ib_proximity_percent > 0:
            # Proximity mode: trigger when within X% of IB level
            long_trigger = ib.ib_high * (1 - self.ib_proximity_percent / 100.0)
            short_trigger = ib.ib_low * (1 + self.ib_proximity_percent / 100.0)
        else:
            # Exact breakout mode
            long_trigger = ib.ib_high
            short_trigger = ib.ib_low

        # Check for LONG breakout (use HIGH for intra-bar breakout detection)
        if self.allow_long:
            if self.ib_proximity_percent > 0:
                # Proximity: close >= trigger level
                long_condition = close >= long_trigger
            else:
                # Exact: high > IB High (matches NinjaTrader and mmap optimizer)
                long_condition = bar.high > ib.ib_high

            if long_condition:
                trigger_type = f"within {self.ib_proximity_percent}% of" if self.ib_proximity_percent > 0 else "above"
                desc = f"LONG breakout {trigger_type} IB High {ib.ib_high:.2f} at {bar.high:.2f}"
                return (True, True, desc)

        # Check for SHORT breakout (use LOW for intra-bar breakout detection)
        if self.allow_short:
            if self.ib_proximity_percent > 0:
                # Proximity: close <= trigger level
                short_condition = close <= short_trigger
            else:
                # Exact: low < IB Low (matches NinjaTrader and mmap optimizer)
                short_condition = bar.low < ib.ib_low

            if short_condition:
                trigger_type = f"within {self.ib_proximity_percent}% of" if self.ib_proximity_percent > 0 else "below"
                desc = f"SHORT breakout {trigger_type} IB Low {ib.ib_low:.2f} at {bar.low:.2f}"
                return (True, False, desc)

        return (False, False, None)


def validate_ib_against_nt(
    python_ibs: Dict[datetime, InitialBalance],
    nt_reference_file: str
) -> List[Dict]:
    """
    Validate Python IB calculations against NinjaTrader reference.

    This function is used to ensure our IB calculations exactly match
    NinjaTrader's output.

    Args:
        python_ibs: Dict of date -> InitialBalance from Python calculator
        nt_reference_file: Path to file with NT IB values (from strategy prints)

    Returns:
        List of discrepancies found
    """
    # This would parse a reference file from NT debug output
    # and compare against Python calculations
    # TODO: Implement when we have reference data
    discrepancies = []
    return discrepancies


if __name__ == "__main__":
    # Test IB Calculator
    from data.data_loader import DataLoader
    from data.session_builder import SessionBuilder
    import os

    test_dir = r"C:\Users\Warren\Downloads"

    # Load data
    loader = DataLoader(test_dir)

    for filename in os.listdir(test_dir):
        if filename.endswith("_NT.txt"):
            filepath = os.path.join(test_dir, filename)
            ticker = filename.split("_")[0].upper()

            try:
                df = loader.load_ninjatrader_file(filepath, ticker)

                # Build sessions
                session_builder = SessionBuilder(ib_duration_minutes=30)
                sessions = session_builder.build_sessions_from_dataframe(df, ticker)

                # Calculate IBs
                ib_calc = IBCalculator(ib_duration_minutes=30)
                ibs = ib_calc.calculate_ibs_for_all_sessions(sessions)

                print(f"\n{ticker}: {len(ibs)} IB levels calculated")

                # Show first few
                for date, ib in list(ibs.items())[:5]:
                    print(f"  {date}: H={ib.ib_high:.2f} L={ib.ib_low:.2f} Range={ib.ib_range_pct:.2f}%")

                # Test breakout detection
                if sessions:
                    breakout_detector = IBBreakoutDetector(ib_proximity_percent=0)
                    first_session = sessions[0]
                    if first_session.ib:
                        print(f"\nTesting breakout detection for {first_session.date}:")
                        for bar in first_session.bars[30:35]:  # Check bars after IB
                            breakout, is_long, desc = breakout_detector.check_breakout(bar, first_session.ib)
                            if breakout:
                                print(f"  {bar.timestamp}: {desc}")

                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
