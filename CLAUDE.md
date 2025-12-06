# CLAUDE.md - IB Breakout Optimizer

This file provides context and guidance for Claude Code when working with this project.

---

## Project Overview

**Purpose**: Custom Python backtester with continuous learning/self-optimization for the IB (Initial Balance) Breakout trading strategy. Optimized parameters are exported to NinjaTrader for live trading.

**Status**: Phase 5 complete with continuous learning/monitoring system. Ready for Phase 6 (ML Filter) or Phase 7 (Portfolio View).

---

## Implementation Progress

### Completed Phases

#### Phase 1: Foundation (Complete)
- Data pipeline with NT format file loading
- Daily parquet caching
- Session builder with timezone handling (UTC → ET)
- IB Calculator matching NinjaTrader logic
- Basic metrics: P&L, win rate, trade count, Sharpe, Sortino

#### Phase 2: Backtester Core (Complete)
- Event engine with order queue
- Order simulator (market, stop, limit fills)
- Position tracking with P&L
- Advanced exits: trailing stop, break-even stop, ATR-based stops, max bars
- QQQ filter with configurable time window
- Multi-ticker portfolio manager
- Full IB breakout strategy implementation
- **Validated**: Backtests match NT Strategy Analyzer

#### Phase 3: Optimization (Complete)
- Grid search for exhaustive parameter exploration
- **Memory-mapped grid search** (mmap) for 80-90% memory reduction
- Bayesian optimization via Optuna
- Two-phase optimizer (coarse grid + Bayesian refinement)
- Walk-forward analysis framework
- IB size analysis
- Parameter export to NT (JSON + CSV)
- **QQQ filter fully wired as optimizable parameter**

#### Phase 4: Interactive UI (Complete)

**Desktop App (PySide6) - Primary**
- Native desktop application with professional dark theme
- 6 tabs: Optimization, Equity Curve, Trade Browser, IB Analysis, Monitoring, Download
- QThread workers for non-blocking background processing
- Live progress updates with elapsed time and ETA
- Two-phase optimization (Grid + Bayesian) with presets (quick, standard, full, thorough)
- Delta metrics showing improvement vs previous run (ΔP&L, ΔWin%, ΔPF)
- Equity curve display for selected optimization results
- Results persisted between app launches
- Desktop shortcut and batch file launcher
- K-Ratio optimization objective for smooth equity curves
- Double-click optimization result to populate Trade Browser, Equity Curve, IB Analysis, and Monitoring tabs

**Streamlit App - Alternative**
- Streamlit app scaffold (`ui/app.py`)
- Backtest page with parameter configuration
- Trade browser with candlestick charts
- Equity curve and drawdown visualization
- Monthly returns heatmap
- Performance by day-of-week and entry hour
- Optimization page (Grid, Bayesian, Walk-Forward)
- Results viewer for loading saved optimizations
- Reusable chart components

#### Phase 5: Continuous Learning (Complete)

**Monitoring Module** (`monitoring/`)
- `performance_monitor.py` - Rolling metrics over configurable windows (20, 50, 100 trades)
  - Sharpe ratio, win rate, profit factor, max drawdown
  - Consecutive win/loss tracking
  - Recent vs long-term performance comparison
- `regime_detector.py` - Market regime detection
  - Volatility regime (LOW, MEDIUM, HIGH, EXTREME) based on ATR percentile
  - Trend regime (STRONG_UP to STRONG_DOWN) using MA slopes
  - Correlation regime with market index (QQQ)
- `degradation_detector.py` - Performance degradation alerts
  - Configurable thresholds (min Sharpe, min win rate, max drawdown, etc.)
  - Alert severity levels (INFO, WARNING, CRITICAL)
  - Health score calculation (0-100)
- `reoptimization_trigger.py` - Automated re-optimization triggers
  - Critical degradation trigger
  - Sustained poor performance trigger
  - Regime shift trigger
  - Scheduled interval trigger
  - New data trigger
- `data_updater.py` - Polygon.io data update automation
  - Auto-detect last data date in files
  - Incremental updates (append new bars)
  - Rate-limited API calls for free tier

**Monitoring Tab** in Desktop App
- Health score dashboard with color-coded status
- Rolling metrics cards with window selection
- Market regime display (volatility, trend, correlation)
- Active alerts table with acknowledge function
- Data update status and controls

### Pending Phases

#### Phase 6: ML Filter (Not Started)
- Feature builder (IB size, time-of-day, volume, QQQ confirm)
- Train win/loss classifier (LightGBM/XGBoost)
- Integrate as trade filter with probability threshold

#### Phase 7: Portfolio View Tab (Not Started)
- New "Portfolio" tab in desktop app
- Multi-ticker selection (checkboxes for AAPL, TSLA, MSFT, NVDA, etc.)
- Load best optimization results for each selected ticker
- Combined equity curve showing aggregate P&L
- Portfolio-level statistics:
  - Total P&L across all tickers
  - Combined Sharpe ratio
  - Max portfolio drawdown
  - Correlation matrix between ticker returns
- Per-ticker breakdown table showing individual contributions

---

## Directory Structure

```
C:\Users\Warren\Projects\ib_breakout_optimizer\
│
├── config/                    # Configuration files
│   ├── settings.yaml          # Global settings
│   └── tickers.yaml           # Ticker-specific settings
│
├── data/                      # Data pipeline
│   ├── data_loader.py         # CSV/NT file loader
│   ├── cache_manager.py       # Parquet caching
│   ├── session_builder.py     # Trading session construction
│   └── data_types.py          # Bar, Session, Trade dataclasses
│
├── strategy/                  # Strategy logic
│   ├── ib_calculator.py       # IB high/low/range calculation
│   ├── signal_generator.py    # Breakout signal detection
│   ├── filters.py             # QQQ filter, time filters
│   ├── exits.py               # Exit logic (trailing, break-even, etc.)
│   └── ib_breakout.py         # Main strategy orchestrator
│
├── backtester/                # Core backtesting engine
│   ├── event_engine.py        # Event loop
│   ├── order_simulator.py     # Fill simulation
│   ├── position_tracker.py    # Position/P&L management
│   └── backtest_runner.py     # Main backtest orchestrator
│
├── metrics/                   # Performance analysis
│   ├── performance_metrics.py # Sharpe, Sortino, profit factor
│   └── trade_analyzer.py      # Per-trade statistics
│
├── optimization/              # Optimization engine
│   ├── parameter_space.py     # Parameter definitions + presets
│   ├── mmap_data.py           # Memory-mapped data manager
│   ├── mmap_grid_search.py    # Memory-efficient grid search (default)
│   ├── grid_search.py         # Legacy exhaustive search
│   ├── bayesian_optuna.py     # Optuna integration
│   ├── two_phase_optimizer.py # Grid + Bayesian hybrid
│   └── walk_forward.py        # Walk-forward analysis
│
├── monitoring/                # Phase 5: Continuous Learning
│   ├── __init__.py            # Module exports
│   ├── performance_monitor.py # Rolling metrics tracker
│   ├── regime_detector.py     # Market regime detection
│   ├── degradation_detector.py # Performance alerts
│   ├── reoptimization_trigger.py # Re-opt triggers
│   └── data_updater.py        # Polygon.io data updates
│
├── desktop_ui/                # PySide6 Desktop App (Primary)
│   ├── main.py                # App entry point
│   ├── main_window.py         # Main window with tabs
│   ├── tabs/
│   │   ├── optimization_tab.py # Grid search with live progress
│   │   ├── equity_curve_tab.py # Equity/drawdown charts
│   │   ├── trade_browser_tab.py # Trade list with charts
│   │   ├── ib_analysis_tab.py # IB size/day analysis
│   │   ├── monitoring_tab.py  # Phase 5 monitoring dashboard
│   │   └── download_tab.py    # Polygon.io data download
│   ├── workers/
│   │   ├── backtest_worker.py # Background backtest thread
│   │   └── optimization_worker.py # Background optimization thread
│   └── widgets/
│       ├── metrics_panel.py   # Metric display cards
│       └── chart_widget.py    # Plotly chart container
│
├── market_data/               # Downloaded market data (TICKER_NT.txt files)
│
├── ui/                        # Streamlit UI (Alternative)
│   ├── app.py                 # Main Streamlit app
│   ├── pages/
│   │   ├── backtest.py        # Backtest configuration
│   │   ├── trade_browser.py   # Trade viewing with charts
│   │   ├── equity_curve.py    # Performance visualization
│   │   ├── optimization.py    # Optimization configuration
│   │   └── results_viewer.py  # Load saved results
│   └── components/
│       ├── candlestick_chart.py   # Reusable chart components
│       └── trade_table.py         # Trade list utilities
│
├── output/                    # Results storage
│   └── optimization/          # Saved optimization results
│
├── scripts/                   # CLI tools
│   ├── run_backtest.py
│   └── run_optimization.py
│
├── run_optimizer.bat          # Desktop app launcher
└── create_shortcut.ps1        # Creates desktop shortcut
```

---

## Key Technical Details

### IB Calculator Logic

The IB (Initial Balance) is calculated during the first N minutes after market open (default: 30 min):

```python
# IB window: 09:30 - 10:00 ET (for 30-min IB)
# IB High = highest high during IB window
# IB Low = lowest low during IB window
# IB Range = IB High - IB Low
# IB Range % = (IB Range / IB Low) * 100
```

### QQQ Filter

When enabled, TSLA (or other ticker) trades require QQQ to break its IB first:

```python
# For long entry: QQQ must have broken above its IB high before TSLA breaks its IB high
# For short entry: QQQ must have broken below its IB low before TSLA breaks its IB low
```

**Critical Implementation Detail**: The strategy uses SEPARATE IB calculators for primary ticker and QQQ filter:
- `self.primary_ib_calc` - tracks primary ticker (e.g., TSLA)
- `self.qqq_ib_calc` - tracks QQQ for filter confirmation

### Data Location

Market data is stored in the `market_data/` folder within the app directory:
```
C:\Users\Warren\Projects\ib_breakout_optimizer\market_data\
├── TSLA_NT.txt
├── QQQ_NT.txt
├── AAPL_NT.txt
└── ...
```

Use the **Download** tab to fetch data from Polygon.io API.

### Data Format

The system expects NinjaTrader exported data format (`*_NT.txt`):

```
20240102 093000;239.25;239.50;239.00;239.45;125000;0
# Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume;OpenInterest
```

Or standard CSV with headers:
```
timestamp,open,high,low,close,volume
```

### Strategy Parameters

Key parameters (all optimizable):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ib_duration_minutes` | 30 | IB window size |
| `profit_target_percent` | 1.0 | Target as % of entry |
| `stop_loss_type` | "opposite_ib" | Stop at opposite IB level |
| `trade_direction` | "long_only" | long_only, short_only, both |
| `use_qqq_filter` | False | Require QQQ confirmation |
| `trailing_stop_enabled` | False | Enable trailing stop |
| `break_even_enabled` | False | Move stop to entry |
| `max_bars` | 120 | Exit after N bars |

---

## Running the Application

### Desktop App (Recommended)
```bash
# Option 1: Desktop shortcut
# Double-click "IB Breakout Optimizer" on desktop

# Option 2: Command line
cd C:\Users\Warren\Projects\ib_breakout_optimizer
python -m desktop_ui.main

# Option 3: Batch file
run_optimizer.bat
```

### Streamlit UI (Alternative)
```bash
cd C:\Users\Warren\Projects\ib_breakout_optimizer
streamlit run ui/app.py
```

### Run CLI Backtest
```bash
python scripts/run_backtest.py --ticker TSLA --use-qqq-filter
```

### Run Optimization
```bash
python scripts/run_optimization.py --mode grid --ticker TSLA
```

### Optimization Presets
The desktop app uses Two-Phase optimization (Grid + Bayesian refinement) with these presets:

| Preset | Combinations | Time (~8 cores) | Description |
|--------|-------------|-----------------|-------------|
| quick | 96 | ~5 sec | IB duration, direction, target, stop |
| standard | 288 | ~8 sec | Core params + IB range filter (recommended) |
| full | 1,152 | ~20 sec | All params including trailing/break-even |
| thorough | 2,592 | ~40 sec | Finer profit target grid (0.2% steps) |

Each preset runs the grid search phase followed by 50 Bayesian trials for refinement.

### Memory-Mapped Optimization
The default optimizer (`MMapGridSearchOptimizer`) uses memory-mapped NumPy arrays:
- **80-90% memory reduction** vs legacy grid search
- Data shared across workers without copying
- Handles 1000+ combinations without OOM
- Files: `optimization/mmap_data.py`, `optimization/mmap_grid_search.py`

---

## Common Tasks

### Adding a New Parameter
1. Add to `optimization/parameter_space.py` in `create_parameter_space()`
2. Add to `strategy/ib_breakout.py` in `StrategyParams` dataclass
3. Implement logic in appropriate strategy module

### Modifying IB Calculation
- Edit `strategy/ib_calculator.py`
- Ensure it matches NinjaTrader logic exactly (see `QQQIBBreakout.cs`)

### Adding New Chart Type
1. Create function in `ui/components/candlestick_chart.py`
2. Call from appropriate page in `ui/pages/`

### Validating Against NinjaTrader
1. Export same parameters to both systems
2. Run backtest on same date range
3. Compare trade-by-trade: entry/exit times, prices, P&L
4. Allow <1% tolerance for P&L differences due to fill timing

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
optuna>=3.0
plotly>=5.0
streamlit>=1.28
pyarrow>=12.0
pyyaml>=6.0
requests>=2.31
PySide6>=6.5
joblib>=1.3
```

Install with:
```bash
pip install pandas numpy optuna plotly streamlit pyarrow pyyaml requests PySide6 joblib
```

---

## Known Issues / Gotchas

1. **File Finder Priority**: The system prefers files with `_NT` in the name (NinjaTrader format). If both `TSLA.csv` and `TSLA_NT.txt` exist, it picks the NT file.

2. **QQQ Filter Requires Both Files**: When using QQQ filter for TSLA, both `TSLA_NT.txt` and `QQQ_NT.txt` must exist in the data directory.

3. **Timezone Handling**: Data should be in Eastern Time. The system handles UTC-to-ET conversion if needed.

4. **Session State**: Streamlit session_state holds backtest results between page navigations. Results persist until the browser tab is closed.

5. **Memory**: The mmap optimizer handles large grids efficiently. Legacy grid search (>10K combinations) can consume significant memory.

6. **Desktop App Threading**: The desktop app runs optimization directly in a QThread (not subprocess) for reliable progress updates on Windows. Uses Qt Signals for thread-safe UI updates.

7. **Trade Browser Date Filter**: Auto-adjusts date range to match actual trade dates when trades are loaded.

8. **Error Logging**: Optimization errors are logged to `logs/optimization_YYYYMMDD.log`. Check this file for detailed stack traces when optimization fails.

---

## Related Files

- **NinjaTrader Strategy**: `C:\Users\Warren\Documents\NinjaTrader 8\bin\Custom\Strategies\QQQIBBreakout.cs`
- **NT Documentation**: `C:\Users\Warren\Documents\NinjaTrader 8\Documentation\nt_docs_organized\`
- **Data Files**: `C:\Users\Warren\Downloads\` (TSLA_NT.txt, QQQ_NT.txt, etc.)
