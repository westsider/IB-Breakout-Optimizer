"""
Session Builder for IB Breakout Optimizer.

Constructs trading sessions from bar data, handling:
- Session boundaries (9:30 AM - 4:00 PM ET for US equities)
- Session detection (first bar of session)
- Multi-day data organization
"""

from datetime import datetime, time, timedelta
from typing import List, Dict, Optional, Iterator
import pandas as pd

from .data_types import Bar, TradingSession, InitialBalance
from .data_loader import DataLoader


class SessionBuilder:
    """
    Builds trading sessions from bar data.

    A session represents one trading day, containing all bars
    and providing session-level state management.
    """

    def __init__(
        self,
        market_open: str = "09:30",
        market_close: str = "16:00",
        ib_duration_minutes: int = 30
    ):
        """
        Initialize session builder.

        Args:
            market_open: Market open time (HH:MM) in Eastern Time
            market_close: Market close time (HH:MM) in Eastern Time
            ib_duration_minutes: Duration of Initial Balance period
        """
        self.market_open = datetime.strptime(market_open, "%H:%M").time()
        self.market_close = datetime.strptime(market_close, "%H:%M").time()
        self.ib_duration_minutes = ib_duration_minutes

    def build_sessions(self, bars: List[Bar], ticker: str) -> List[TradingSession]:
        """
        Build trading sessions from a list of bars.

        Args:
            bars: List of Bar objects (must be sorted by timestamp)
            ticker: Ticker symbol

        Returns:
            List of TradingSession objects
        """
        if not bars:
            return []

        sessions: List[TradingSession] = []
        current_session: Optional[TradingSession] = None
        current_date = None

        for bar in bars:
            bar_date = bar.timestamp.date()
            bar_time = bar.timestamp.time()

            # Check if this is a new session
            if current_date != bar_date:
                # Finalize previous session if exists
                if current_session is not None:
                    sessions.append(current_session)

                # Create new session
                session_start = datetime.combine(bar_date, self.market_open)
                session_end = datetime.combine(bar_date, self.market_close)

                current_session = TradingSession(
                    date=bar_date,
                    ticker=ticker,
                    session_start=session_start,
                    session_end=session_end
                )
                current_date = bar_date

            # Add bar to current session if within regular hours
            if self.market_open <= bar_time < self.market_close:
                current_session.add_bar(bar)

        # Don't forget the last session
        if current_session is not None and current_session.bars:
            sessions.append(current_session)

        return sessions

    def build_sessions_from_dataframe(self, df: pd.DataFrame, ticker: str) -> List[TradingSession]:
        """
        Build sessions from a DataFrame.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            ticker: Ticker symbol

        Returns:
            List of TradingSession objects
        """
        bars = []
        for _, row in df.iterrows():
            bar = Bar(
                timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                ticker=ticker
            )
            bars.append(bar)

        return self.build_sessions(bars, ticker)

    def is_first_bar_of_session(self, bar: Bar, prev_bar: Optional[Bar] = None) -> bool:
        """
        Check if a bar is the first bar of a new session.

        This matches NinjaTrader's Bars.IsFirstBarOfSession logic.

        Args:
            bar: Current bar
            prev_bar: Previous bar (None if first bar ever)

        Returns:
            True if this is the first bar of a new session
        """
        if prev_bar is None:
            return True

        # New session if different date
        if bar.timestamp.date() != prev_bar.timestamp.date():
            return True

        # New session if current bar is at or after market open
        # and previous bar was before market open (pre-market -> RTH transition)
        bar_time = bar.timestamp.time()
        prev_time = prev_bar.timestamp.time()

        if bar_time >= self.market_open and prev_time < self.market_open:
            return True

        return False

    def get_session_start(self, bar_timestamp: datetime) -> datetime:
        """
        Get the session start time for a given bar timestamp.

        Args:
            bar_timestamp: Timestamp of any bar

        Returns:
            Session start datetime
        """
        return datetime.combine(bar_timestamp.date(), self.market_open)

    def get_ib_end_time(self, session_start: datetime) -> datetime:
        """
        Get the IB end time for a session.

        Args:
            session_start: Session start datetime

        Returns:
            IB end datetime
        """
        return session_start + timedelta(minutes=self.ib_duration_minutes)

    def iterate_bars_with_session_info(
        self,
        bars: List[Bar]
    ) -> Iterator[tuple]:
        """
        Iterate through bars with session information.

        Yields tuples of (bar, is_first_bar, session_start, ib_end_time).

        This is useful for event-driven processing where you need
        session context for each bar.

        Args:
            bars: List of bars

        Yields:
            (bar, is_first_bar, session_start, ib_end_time) tuples
        """
        prev_bar = None
        current_session_start = None
        current_ib_end = None

        for bar in bars:
            is_first = self.is_first_bar_of_session(bar, prev_bar)

            if is_first:
                current_session_start = self.get_session_start(bar.timestamp)
                current_ib_end = self.get_ib_end_time(current_session_start)

            yield (bar, is_first, current_session_start, current_ib_end)
            prev_bar = bar


class MultiTickerSessionBuilder:
    """
    Manages sessions for multiple tickers simultaneously.

    Useful for strategies that need to track multiple instruments
    (e.g., QQQ filter for TSLA trading).
    """

    def __init__(
        self,
        market_open: str = "09:30",
        market_close: str = "16:00",
        ib_duration_minutes: int = 30
    ):
        """
        Initialize multi-ticker session builder.

        Args:
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)
            ib_duration_minutes: IB duration
        """
        self.builder = SessionBuilder(market_open, market_close, ib_duration_minutes)
        self.sessions_by_ticker: Dict[str, List[TradingSession]] = {}

    def build_sessions(self, loader: DataLoader, tickers: List[str]) -> Dict[str, List[TradingSession]]:
        """
        Build sessions for multiple tickers.

        Args:
            loader: DataLoader with ticker data loaded
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker -> list of sessions
        """
        for ticker in tickers:
            if loader.is_loaded(ticker):
                df = loader.get_dataframe(ticker)
                sessions = self.builder.build_sessions_from_dataframe(df, ticker)
                self.sessions_by_ticker[ticker] = sessions
                print(f"Built {len(sessions)} sessions for {ticker}")
            else:
                print(f"Warning: {ticker} not loaded in DataLoader")

        return self.sessions_by_ticker

    def get_sessions_for_date(self, date: datetime) -> Dict[str, TradingSession]:
        """
        Get all ticker sessions for a specific date.

        Args:
            date: Target date

        Returns:
            Dict mapping ticker -> session for that date
        """
        target_date = date.date() if isinstance(date, datetime) else date
        result = {}

        for ticker, sessions in self.sessions_by_ticker.items():
            for session in sessions:
                if session.date == target_date:
                    result[ticker] = session
                    break

        return result

    def get_aligned_sessions(self) -> List[Dict[str, TradingSession]]:
        """
        Get sessions aligned by date across all tickers.

        Returns:
            List of dicts, each containing all ticker sessions for one date
        """
        # Collect all unique dates
        all_dates = set()
        for sessions in self.sessions_by_ticker.values():
            for session in sessions:
                all_dates.add(session.date)

        # Sort dates
        all_dates = sorted(all_dates)

        # Build aligned list
        aligned = []
        for date in all_dates:
            date_sessions = self.get_sessions_for_date(date)
            if date_sessions:
                aligned.append(date_sessions)

        return aligned

    def summary(self) -> Dict:
        """Get summary of built sessions."""
        summary = {}
        for ticker, sessions in self.sessions_by_ticker.items():
            if sessions:
                dates = [s.date for s in sessions]
                summary[ticker] = {
                    'sessions': len(sessions),
                    'start': min(dates),
                    'end': max(dates),
                    'total_bars': sum(len(s.bars) for s in sessions)
                }
        return summary


if __name__ == "__main__":
    # Test session builder
    from .data_loader import DataLoader
    import os

    test_dir = r"C:\Users\Warren\Downloads"
    loader = DataLoader(test_dir)

    # Try to load test data
    for filename in os.listdir(test_dir):
        if filename.endswith("_NT.txt"):
            filepath = os.path.join(test_dir, filename)
            ticker = filename.split("_")[0].upper()

            try:
                df = loader.load_ninjatrader_file(filepath, ticker)

                # Build sessions
                builder = SessionBuilder(ib_duration_minutes=30)
                sessions = builder.build_sessions_from_dataframe(df, ticker)

                print(f"\n{ticker}: {len(sessions)} sessions")
                if sessions:
                    first = sessions[0]
                    print(f"  First session: {first.date}, {len(first.bars)} bars")
                    print(f"  Session start: {first.session_start}")

                    # Show first few bars
                    for bar in first.bars[:5]:
                        print(f"    {bar}")

                break
            except Exception as e:
                print(f"Error: {e}")
