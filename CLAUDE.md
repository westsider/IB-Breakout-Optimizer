# CLAUDE.md - IB Breakout Optimizer

This file provides context and guidance for Claude Code when working with this project.

---

## Project Overview

**Purpose**: Custom Python backtester with continuous learning/self-optimization for the IB (Initial Balance) Breakout trading strategy. Optimized parameters are exported to NinjaTrader for live trading.

**Status**: Phase 6 complete - ML Filter with ensemble models, insights generator, and optimizer integration. Statistical filters and Saved Tests tab added.

---

## Implementation Progress

### Completed Phases

#### Phase 1: Foundation (Complete)
- Data pipeline with NT format file loading
- Daily parquet caching
- Session builder with timezone handling (UTC → ET)
- IB Calculator matching NinjaTrader logic
- Basic metrics: P&L, win rate, trade count, Sharpe, Sortino, Move Capture %

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
- **Trade filters** for improved signal quality:
  - Gap % filter (today's open vs yesterday's close)
  - Prior days trend filter (last N days bullish/bearish bias)
  - Daily range % filter (volatility/ATR filter)
  - "with_trade" mode aligns filter direction with trade direction
- **Statistical filters** using pre-computed normal distribution percentiles:
  - Gap filter modes: any, middle_68, exclude_middle_68, directional, reverse_directional
  - Trend filter modes: any, with_trend, counter_trend
  - Range filter modes: any, middle_68, above_68, below_median, middle_68_or_below
  - Distribution stats cached per ticker in `data/stats_cache/`

#### Phase 4: Interactive UI (Complete)

**Desktop App (PySide6) - Primary**
- Native desktop application with professional dark theme
- 12 tabs: Optimization, Walk-Forward, Portfolio, Forward Tests, Equity Curve, Trade Browser, IB Analysis, Filter Analysis, Monitoring, ML Filter, Saved Tests, Download
- QThread workers for non-blocking background processing
- Live progress updates with elapsed time and ETA
- Two-phase optimization (Grid + Bayesian) with presets (quick, standard, full, thorough)
- Delta metrics showing improvement vs previous run (ΔP&L, ΔWin%, ΔPF)
- Equity curve display for selected optimization results
- Results persisted between app launches
- Desktop shortcut and batch file launcher
- K-Ratio optimization objective for smooth equity curves
- **Move Capture %** metric - measures trading efficiency (how much of available move was captured)
  - Displayed in Equity Curve tab, Trade Browser, and Optimization results
  - Formula: For longs: (exit-entry)/(day_high-ib_high); For shorts: (entry-exit)/(ib_low-day_low)
  - Color coded: Green (≥50%), Yellow (25-49%), Red (<25%)
- Double-click optimization result to populate Trade Browser, Equity Curve, IB Analysis, and Monitoring tabs
- **Saved Tests tab** for storing and reviewing best results per instrument
  - Results persisted to `output/saved_tests/saved_results.json`
  - Save button in Optimization tab to save current best
  - Shows parameters, equity curve, P&L, PF, Win%, Sharpe
  - Load saved params back to optimizer for re-testing
- **Portfolio tab** for multi-ticker portfolio simulation
  - Select multiple tickers from saved tests with checkboxes
  - Combined equity curve showing aggregate performance
  - Portfolio-level statistics: Total P&L, Combined PF, Combined Sharpe, Max DD
  - Per-ticker breakdown table with contribution percentages
  - Individual ticker curves overlaid on combined chart
  - **Monthly P&L bar chart** showing gain/loss per month with win/loss month stats
- **Walk-Forward Analysis tab** for out-of-sample strategy validation
  - Rolling optimization with configurable training window (3, 6, 9, 12 months)
  - Configurable test window (1, 2, 4 weeks)
  - Trains on historical data, tests "blind" on following period
  - Aggregates all out-of-sample results into single equity curve
  - Key metrics: OOS P&L, OOS Profit Factor, Efficiency Ratio (OOS/IS)
  - Period win rate shows % of test periods that were profitable
  - In-sample vs out-of-sample comparison bar chart
  - Same filters and objectives as Optimization tab
- **Forward Tests tab** for live market validation of saved optimizations
  - Start forward tests from saved optimization results
  - Run periodic forward tests as new market data is loaded
  - Tracks cumulative forward P&L vs backtest P&L
  - Run history with per-period breakdown (dates, trades, P&L, win%)
  - Cumulative P&L chart and per-run bar chart
  - Consistency metric (% profitable forward test runs)
  - Results persisted to `output/forward_tests/forward_tests.json`
- **Statistical filter dropdowns** in Optimization tab for gap, trend, and range filters
- **Rebuild Stats button** in Download tab to recompute distribution stats after new data
- **Filter Analysis tab** for visualizing filter calculations
  - Interactive daily candlestick chart with horizontal scrolling
  - Gap % visualization with shaded areas between prior close and open
  - Prior days trend indicators (bullish/bearish dots)
  - Daily range % bars showing volatility
  - Crosshair with detailed data panel showing all filter values
  - Mouse wheel zoom (5-30 days visible)
  - Shows both prior 5-day average volatility AND today's range for clarity
- **Walk-Forward tab enhancements**
  - Save button to export results as JSON
  - Settings persistence (ticker, windows, filters) via QSettings
  - OOS Length metric showing total out-of-sample months tested
  - $/Month metric showing average monthly P&L
  - Help menu: "How to Walk-Forward Test and Trade" guide
- **Download tab enhancements**
  - Custom date range inputs (Start Date / End Date) with calendar picker
  - "Auto-Detect Gap" button to find 1 year prior to existing data
  - "Download & Merge" button to prepend historical data
  - Automatic 2-year limit clamping for Polygon.io free tier
  - Prominent download progress display with status header and ETA

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

#### Phase 6: ML Filter (Complete)

**ML Filter Module** (`ml_filter/`)
- `feature_builder.py` - Extract ML features from backtest trades
  - IB characteristics (range %, duration)
  - Gap % (today's open vs yesterday's close)
  - Prior days trend (bullish/bearish count)
  - Daily range/volatility
  - Time features (hour, day of week)
  - Trade direction (long/short)
  - QQQ confirmation status
  - Distance from IB at entry
  - Strategy parameters (profit target, stop type, trailing/break-even)
- `model_trainer.py` - ML classifiers for win/loss prediction
  - **Ensemble model**: LightGBM + Random Forest + Logistic Regression (40/40/20 weights)
  - Reduced LightGBM complexity to prevent overfitting (`num_leaves=15`, `max_depth=4`)
  - **TimeSeriesSplit** cross-validation (preserves temporal order)
  - Model metrics: accuracy, precision, recall, F1, ROC AUC
  - Feature importance analysis
  - **Insights generator**: actionable recommendations based on data analysis
    - Day of week win rate analysis
    - IB range impact (narrow/wide IB days)
    - Gap direction effects
    - Entry hour patterns
    - Prior days trend impact
  - Model persistence (save/load pickle files with ensemble support)

**ML Filter Tab** in Desktop App
- Ticker and parameter selection
- **Model type dropdown**: Ensemble or LightGBM only
- **Probability threshold slider** (0.50 - 0.70)
- One-click training (runs backtest → extracts features → trains model)
- Model performance metrics with **tooltips explaining each metric**
- **Confusion matrix with color coding and explanations** (TN/FP/FN/TP)
- Feature importance bar chart
- **Insights panel** showing actionable ML recommendations
- **"Train from Best" button**: trains using best params from optimization
- **Optimizer integration**: best results auto-populate ML tab when optimization completes
- Save/load trained models

#### Phase 7: Portfolio View Tab (Complete)
- New "Portfolio" tab in desktop app
- Multi-ticker selection via checkboxes loading from Saved Tests
- Best results for each ticker auto-selected (highest P&L)
- Combined equity curve showing aggregate P&L with individual ticker overlays
- Portfolio-level statistics:
  - Total P&L across all tickers
  - Combined Sharpe ratio (trade-weighted average)
  - Combined Profit Factor
  - Max portfolio drawdown
- Per-ticker breakdown table showing:
  - Individual P&L, trades, win rate, PF, Sharpe
  - Contribution percentage to total portfolio

### Pending Phases

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
│   ├── data_types.py          # Bar, Session, Trade dataclasses
│   └── distribution_stats.py  # Statistical distribution calculator (gap%, range%)
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
├── ml_filter/                 # Phase 6: ML Trade Filter
│   ├── __init__.py            # Module exports
│   ├── feature_builder.py     # Extract features from trades
│   └── model_trainer.py       # LightGBM classifier training
│
├── desktop_ui/                # PySide6 Desktop App (Primary)
│   ├── main.py                # App entry point
│   ├── main_window.py         # Main window with tabs
│   ├── tabs/
│   │   ├── optimization_tab.py # Grid search with live progress
│   │   ├── portfolio_tab.py   # Multi-ticker portfolio simulation
│   │   ├── equity_curve_tab.py # Equity/drawdown charts
│   │   ├── trade_browser_tab.py # Trade list with charts
│   │   ├── ib_analysis_tab.py # IB size/day analysis
│   │   ├── monitoring_tab.py  # Phase 5 monitoring dashboard
│   │   ├── ml_filter_tab.py   # Phase 6 ML filter training
│   │   ├── saved_tests_tab.py # Saved test results storage
│   │   ├── forward_tests_tab.py # Forward testing with live data
│   │   ├── walk_forward_tab.py # Walk-forward analysis with OOS testing
│   │   ├── filter_analysis_tab.py # Filter visualization and verification
│   │   └── download_tab.py    # Polygon.io data download + prepend historical
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
│   ├── optimization/          # Saved optimization results
│   └── saved_tests/           # Saved test results (JSON)
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

**Trade Filter Parameters** (new):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gap_filter_enabled` | False | Filter by gap % |
| `gap_direction_filter` | "any" | any, gap_up_only, gap_down_only, with_trade |
| `prior_days_filter_enabled` | False | Filter by prior days trend |
| `prior_days_trend` | "any" | any, bullish, bearish, with_trade |
| `prior_days_lookback` | 3 | Days to check for trend |
| `daily_range_filter_enabled` | False | Filter by daily range % |
| `min_avg_daily_range_percent` | 0.0 | Minimum avg daily range required |
| `daily_range_lookback` | 5 | Days to average for range |

**Statistical Filter Parameters** (mode-based, using distribution percentiles):

| Parameter | Default | Options |
|-----------|---------|---------|
| `gap_filter_mode` | "any" | any, middle_68, exclude_middle_68, directional, reverse_directional |
| `trend_filter_mode` | "any" | any, with_trend, counter_trend |
| `range_filter_mode` | "any" | any, middle_68, above_68, below_median, middle_68_or_below |

Statistical filters use pre-computed normal distribution percentiles (p16, p50, p68, p84) from historical data to determine if today's gap/range is "normal" or "extreme".

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
lightgbm>=4.0
scikit-learn>=1.3
```

Install with:
```bash
pip install pandas numpy optuna plotly streamlit pyarrow pyyaml requests PySide6 joblib lightgbm scikit-learn
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
