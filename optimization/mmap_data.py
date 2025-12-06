"""
Memory-Mapped Data Manager for IB Breakout Optimizer.

Converts trading session data to NumPy memory-mapped arrays that can be
shared across worker processes without copying. This dramatically reduces
memory usage during parallel optimization.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class MMapArrayPaths:
    """Paths to memory-mapped array files."""
    base_dir: str
    timestamps: str
    opens: str
    highs: str
    lows: str
    closes: str
    volumes: str
    date_indices: str
    hours: str
    minutes: str
    n_bars: int
    n_dates: int
    ticker: str
    # Filter data (optional)
    filter_timestamps: Optional[str] = None
    filter_opens: Optional[str] = None
    filter_highs: Optional[str] = None
    filter_lows: Optional[str] = None
    filter_closes: Optional[str] = None
    filter_date_indices: Optional[str] = None
    filter_hours: Optional[str] = None
    filter_minutes: Optional[str] = None
    filter_n_bars: int = 0
    filter_ticker: str = ""


class MMapDataManager:
    """
    Manages memory-mapped NumPy arrays for trading data.

    Converts session/bar data to mmap files that can be shared across
    processes without copying, reducing memory usage by 80-90%.
    """

    def __init__(self, sessions: List, filter_bars_dict: Optional[Dict] = None,
                 ticker: str = "", filter_ticker: str = ""):
        """
        Initialize and create memory-mapped arrays from session data.

        Args:
            sessions: List of TradingSession objects
            filter_bars_dict: Optional dict mapping timestamp -> Bar for filter (e.g., QQQ)
            ticker: Primary ticker symbol
            filter_ticker: Filter ticker symbol (e.g., "QQQ")
        """
        self.ticker = ticker
        self.filter_ticker = filter_ticker
        self._temp_dir = None
        self._paths = None

        # Create mmap files
        self._create_mmap_arrays(sessions, filter_bars_dict)

    def _create_mmap_arrays(self, sessions: List, filter_bars_dict: Optional[Dict]):
        """Convert session data to memory-mapped arrays."""
        # Create temp directory for mmap files
        self._temp_dir = tempfile.mkdtemp(prefix="ib_optimizer_mmap_")
        base_path = Path(self._temp_dir)

        # Collect all bars from sessions
        all_bars = []
        for session in sessions:
            all_bars.extend(session.bars)

        n_bars = len(all_bars)

        # Extract data into numpy arrays
        timestamps = np.zeros(n_bars, dtype=np.int64)
        opens = np.zeros(n_bars, dtype=np.float64)
        highs = np.zeros(n_bars, dtype=np.float64)
        lows = np.zeros(n_bars, dtype=np.float64)
        closes = np.zeros(n_bars, dtype=np.float64)
        volumes = np.zeros(n_bars, dtype=np.int64)
        hours = np.zeros(n_bars, dtype=np.int32)
        minutes = np.zeros(n_bars, dtype=np.int32)

        for i, bar in enumerate(all_bars):
            # Convert datetime to int64 timestamp (nanoseconds)
            ts = bar.timestamp
            if isinstance(ts, datetime):
                timestamps[i] = int(ts.timestamp() * 1e9)
                hours[i] = ts.hour
                minutes[i] = ts.minute
            else:
                # Already a pandas Timestamp or similar
                timestamps[i] = int(pd.Timestamp(ts).timestamp() * 1e9)
                hours[i] = pd.Timestamp(ts).hour
                minutes[i] = pd.Timestamp(ts).minute

            opens[i] = bar.open
            highs[i] = bar.high
            lows[i] = bar.low
            closes[i] = bar.close
            volumes[i] = bar.volume

        # Create date indices (map each bar to unique date)
        dates = np.array([pd.Timestamp(ts, unit='ns').date() for ts in timestamps])
        unique_dates = np.unique(dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        date_indices = np.array([date_to_idx[d] for d in dates], dtype=np.int32)

        n_dates = len(unique_dates)

        # Save as memory-mapped files
        def save_mmap(arr: np.ndarray, name: str) -> str:
            path = str(base_path / f"{name}.mmap")
            mmap = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
            mmap[:] = arr[:]
            mmap.flush()
            del mmap
            return path

        paths = MMapArrayPaths(
            base_dir=str(base_path),
            timestamps=save_mmap(timestamps, "timestamps"),
            opens=save_mmap(opens, "opens"),
            highs=save_mmap(highs, "highs"),
            lows=save_mmap(lows, "lows"),
            closes=save_mmap(closes, "closes"),
            volumes=save_mmap(volumes, "volumes"),
            date_indices=save_mmap(date_indices, "date_indices"),
            hours=save_mmap(hours, "hours"),
            minutes=save_mmap(minutes, "minutes"),
            n_bars=n_bars,
            n_dates=n_dates,
            ticker=self.ticker
        )

        # Process filter data if provided
        if filter_bars_dict:
            paths = self._create_filter_mmap(filter_bars_dict, paths, base_path)

        self._paths = paths

    def _create_filter_mmap(self, filter_bars_dict: Dict, paths: MMapArrayPaths,
                           base_path: Path) -> MMapArrayPaths:
        """Create memory-mapped arrays for filter ticker data."""
        # Sort by timestamp
        sorted_items = sorted(filter_bars_dict.items(), key=lambda x: x[0])
        n_bars = len(sorted_items)

        timestamps = np.zeros(n_bars, dtype=np.int64)
        opens = np.zeros(n_bars, dtype=np.float64)
        highs = np.zeros(n_bars, dtype=np.float64)
        lows = np.zeros(n_bars, dtype=np.float64)
        closes = np.zeros(n_bars, dtype=np.float64)
        hours = np.zeros(n_bars, dtype=np.int32)
        minutes = np.zeros(n_bars, dtype=np.int32)

        for i, (ts, bar) in enumerate(sorted_items):
            if isinstance(ts, datetime):
                timestamps[i] = int(ts.timestamp() * 1e9)
                hours[i] = ts.hour
                minutes[i] = ts.minute
            else:
                timestamps[i] = int(pd.Timestamp(ts).timestamp() * 1e9)
                hours[i] = pd.Timestamp(ts).hour
                minutes[i] = pd.Timestamp(ts).minute

            opens[i] = bar.open
            highs[i] = bar.high
            lows[i] = bar.low
            closes[i] = bar.close

        # Create date indices
        dates = np.array([pd.Timestamp(ts, unit='ns').date() for ts in timestamps])
        unique_dates = np.unique(dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        date_indices = np.array([date_to_idx[d] for d in dates], dtype=np.int32)

        def save_mmap(arr: np.ndarray, name: str) -> str:
            path = str(base_path / f"filter_{name}.mmap")
            mmap = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
            mmap[:] = arr[:]
            mmap.flush()
            del mmap
            return path

        paths.filter_timestamps = save_mmap(timestamps, "timestamps")
        paths.filter_opens = save_mmap(opens, "opens")
        paths.filter_highs = save_mmap(highs, "highs")
        paths.filter_lows = save_mmap(lows, "lows")
        paths.filter_closes = save_mmap(closes, "closes")
        paths.filter_date_indices = save_mmap(date_indices, "date_indices")
        paths.filter_hours = save_mmap(hours, "hours")
        paths.filter_minutes = save_mmap(minutes, "minutes")
        paths.filter_n_bars = n_bars
        paths.filter_ticker = self.filter_ticker

        return paths

    def get_paths(self) -> MMapArrayPaths:
        """Get paths to memory-mapped array files."""
        return self._paths

    def cleanup(self):
        """Delete temporary mmap files."""
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def load_mmap_arrays(paths: MMapArrayPaths) -> Dict[str, np.ndarray]:
    """
    Load memory-mapped arrays from paths (read-only, shared across processes).

    Args:
        paths: MMapArrayPaths containing file paths

    Returns:
        Dictionary of numpy arrays (memory-mapped, not copied)
    """
    arrays = {
        'timestamps': np.memmap(paths.timestamps, dtype=np.int64, mode='r',
                               shape=(paths.n_bars,)),
        'opens': np.memmap(paths.opens, dtype=np.float64, mode='r',
                          shape=(paths.n_bars,)),
        'highs': np.memmap(paths.highs, dtype=np.float64, mode='r',
                          shape=(paths.n_bars,)),
        'lows': np.memmap(paths.lows, dtype=np.float64, mode='r',
                         shape=(paths.n_bars,)),
        'closes': np.memmap(paths.closes, dtype=np.float64, mode='r',
                           shape=(paths.n_bars,)),
        'volumes': np.memmap(paths.volumes, dtype=np.int64, mode='r',
                            shape=(paths.n_bars,)),
        'date_indices': np.memmap(paths.date_indices, dtype=np.int32, mode='r',
                                  shape=(paths.n_bars,)),
        'hours': np.memmap(paths.hours, dtype=np.int32, mode='r',
                          shape=(paths.n_bars,)),
        'minutes': np.memmap(paths.minutes, dtype=np.int32, mode='r',
                            shape=(paths.n_bars,)),
        'n_bars': paths.n_bars,
        'n_dates': paths.n_dates,
        'ticker': paths.ticker
    }

    # Load filter arrays if present
    if paths.filter_timestamps:
        arrays['filter'] = {
            'timestamps': np.memmap(paths.filter_timestamps, dtype=np.int64, mode='r',
                                   shape=(paths.filter_n_bars,)),
            'opens': np.memmap(paths.filter_opens, dtype=np.float64, mode='r',
                              shape=(paths.filter_n_bars,)),
            'highs': np.memmap(paths.filter_highs, dtype=np.float64, mode='r',
                              shape=(paths.filter_n_bars,)),
            'lows': np.memmap(paths.filter_lows, dtype=np.float64, mode='r',
                             shape=(paths.filter_n_bars,)),
            'closes': np.memmap(paths.filter_closes, dtype=np.float64, mode='r',
                               shape=(paths.filter_n_bars,)),
            'date_indices': np.memmap(paths.filter_date_indices, dtype=np.int32, mode='r',
                                      shape=(paths.filter_n_bars,)),
            'hours': np.memmap(paths.filter_hours, dtype=np.int32, mode='r',
                              shape=(paths.filter_n_bars,)),
            'minutes': np.memmap(paths.filter_minutes, dtype=np.int32, mode='r',
                                shape=(paths.filter_n_bars,)),
            'n_bars': paths.filter_n_bars,
            'ticker': paths.filter_ticker
        }

    return arrays


if __name__ == "__main__":
    # Test the mmap manager
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.data_loader import DataLoader
    from data.session_builder import SessionBuilder

    print("Testing MMapDataManager...")

    # Load data
    loader = DataLoader(r"C:\Users\Warren\Downloads")
    df = loader.load_auto_detect(
        r"C:\Users\Warren\Downloads\TSLA_1min_20231204_to_20241204_NT.txt",
        "TSLA"
    )

    # Build sessions
    builder = SessionBuilder()
    sessions = builder.build_sessions_from_dataframe(df, "TSLA")

    print(f"Loaded {len(sessions)} sessions")

    # Create mmap manager
    manager = MMapDataManager(sessions, ticker="TSLA")
    paths = manager.get_paths()

    print(f"\nMMap files created in: {paths.base_dir}")
    print(f"  n_bars: {paths.n_bars:,}")
    print(f"  n_dates: {paths.n_dates:,}")

    # Load arrays
    arrays = load_mmap_arrays(paths)

    print(f"\nLoaded arrays:")
    print(f"  opens shape: {arrays['opens'].shape}")
    print(f"  first 5 opens: {arrays['opens'][:5]}")
    print(f"  first 5 highs: {arrays['highs'][:5]}")

    # Cleanup
    manager.cleanup()
    print("\nCleanup complete")
