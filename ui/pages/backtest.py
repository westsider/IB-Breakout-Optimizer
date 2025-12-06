"""
Backtest Page - Run backtests with custom parameters.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtester.backtest_runner import BacktestRunner
from strategy.ib_breakout import StrategyParams


def render_backtest_page():
    """Render the backtest configuration and execution page."""

    # Two columns for parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Basic Parameters")

        ticker = st.selectbox(
            "Ticker",
            ["TSLA", "QQQ", "AAPL", "NVDA", "MSFT"],
            index=0
        )

        trade_direction = st.selectbox(
            "Trade Direction",
            ["long_only", "short_only", "both"],
            index=0
        )

        profit_target = st.slider(
            "Profit Target %",
            min_value=0.3,
            max_value=3.0,
            value=1.0,
            step=0.1
        )

        stop_loss_type = st.selectbox(
            "Stop Loss Type",
            ["opposite_ib", "match_target"],
            index=0
        )

        ib_duration = st.selectbox(
            "IB Duration (minutes)",
            [15, 30, 45, 60],
            index=1
        )

    with col2:
        st.subheader("🔧 Filters & Options")

        use_qqq_filter = st.checkbox(
            "Use QQQ Filter",
            value=True,
            help="Require QQQ to break its IB before entering TSLA trades"
        )

        min_ib_range = st.slider(
            "Min IB Range %",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )

        max_ib_range = st.slider(
            "Max IB Range %",
            min_value=1.0,
            max_value=10.0,
            value=10.0,
            step=0.5
        )

        max_breakout_time = st.selectbox(
            "Max Breakout Time",
            ["12:00", "13:00", "14:00", "15:00"],
            index=2
        )

        eod_exit_time = st.selectbox(
            "EOD Exit Time",
            ["15:30", "15:45", "15:55"],
            index=2
        )

    # Advanced options in expander
    with st.expander("🔬 Advanced Exit Options"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)

        with adv_col1:
            trailing_stop = st.checkbox("Trailing Stop", value=False)
            trailing_atr = st.slider("Trail ATR Mult", 1.0, 4.0, 2.0, 0.5, disabled=not trailing_stop)

        with adv_col2:
            break_even = st.checkbox("Break Even Stop", value=False)
            break_even_pct = st.slider("BE Trigger %", 0.3, 0.9, 0.7, 0.1, disabled=not break_even)

        with adv_col3:
            max_bars = st.checkbox("Max Bars Exit", value=False)
            max_bars_val = st.slider("Max Bars", 10, 120, 60, 10, disabled=not max_bars)

    st.markdown("---")

    # Run button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest(
            ticker=ticker,
            trade_direction=trade_direction,
            profit_target=profit_target,
            stop_loss_type=stop_loss_type,
            ib_duration=ib_duration,
            use_qqq_filter=use_qqq_filter,
            min_ib_range=min_ib_range,
            max_ib_range=max_ib_range,
            max_breakout_time=max_breakout_time,
            eod_exit_time=eod_exit_time,
            trailing_stop=trailing_stop,
            trailing_atr=trailing_atr,
            break_even=break_even,
            break_even_pct=break_even_pct,
            max_bars_enabled=max_bars,
            max_bars_val=max_bars_val
        )


def run_backtest(**kwargs):
    """Execute the backtest with given parameters."""

    data_dir = st.session_state.get('data_dir', r"C:\Users\Warren\Downloads")

    with st.spinner(f"Running backtest for {kwargs['ticker']}..."):
        try:
            runner = BacktestRunner(data_dir)

            params = StrategyParams(
                ib_duration_minutes=kwargs['ib_duration'],
                profit_target_percent=kwargs['profit_target'],
                stop_loss_type=kwargs['stop_loss_type'],
                trade_direction=kwargs['trade_direction'],
                use_qqq_filter=kwargs['use_qqq_filter'],
                min_ib_range_percent=kwargs['min_ib_range'],
                max_ib_range_percent=kwargs['max_ib_range'],
                max_breakout_time=kwargs['max_breakout_time'],
                eod_exit_time=kwargs['eod_exit_time'],
                trailing_stop_enabled=kwargs['trailing_stop'],
                trailing_stop_atr_mult=kwargs['trailing_atr'],
                break_even_enabled=kwargs['break_even'],
                break_even_pct=kwargs['break_even_pct'],
                max_bars_enabled=kwargs['max_bars_enabled'],
                max_bars=kwargs['max_bars_val']
            )

            # Run with or without QQQ filter
            if kwargs['use_qqq_filter'] and kwargs['ticker'] != 'QQQ':
                result, metrics = runner.run_backtest_with_filter(
                    ticker=kwargs['ticker'],
                    filter_ticker='QQQ',
                    params=params,
                    verbose=False
                )
            else:
                result, metrics = runner.run_backtest(
                    ticker=kwargs['ticker'],
                    params=params,
                    verbose=False
                )

            # Store results in session state
            st.session_state['last_result'] = result
            st.session_state['last_metrics'] = metrics
            st.session_state['last_params'] = params

            # Display results
            display_results(result, metrics)

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_results(result, metrics):
    """Display backtest results."""

    st.success("✅ Backtest Complete!")

    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pnl_color = "normal" if metrics.total_net_profit >= 0 else "inverse"
        st.metric("Total P&L", f"${metrics.total_net_profit:,.2f}", delta_color=pnl_color)

    with col2:
        st.metric("Total Trades", metrics.total_trades)

    with col3:
        st.metric("Win Rate", f"{metrics.percent_profitable:.1f}%")

    with col4:
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

    with col5:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

    # Second row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Max Drawdown", f"${metrics.max_drawdown:,.2f}")

    with col2:
        st.metric("Avg Trade", f"${metrics.avg_trade:.2f}")

    with col3:
        st.metric("Avg Winner", f"${metrics.avg_winning_trade:.2f}")

    with col4:
        st.metric("Avg Loser", f"${metrics.avg_losing_trade:.2f}")

    with col5:
        st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

    # Trade list
    st.markdown("### 📋 Trade List")

    if result.trades:
        import pandas as pd

        trade_data = []
        for t in result.trades:
            trade_data.append({
                'Entry Time': t.entry_time.strftime('%Y-%m-%d %H:%M'),
                'Exit Time': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else '',
                'Direction': '🟢 Long' if t.direction.value == 'long' else '🔴 Short',
                'Entry': f"${t.entry_price:.2f}",
                'Exit': f"${t.exit_price:.2f}" if t.exit_price else '',
                'P&L': f"${t.pnl:.2f}",
                'P&L %': f"{t.pnl_pct:.2f}%",
                'Exit Reason': t.exit_reason.value if t.exit_reason else '',
                'Bars': t.bars_held
            })

        df = pd.DataFrame(trade_data)

        # Color P&L column
        def color_pnl(val):
            if isinstance(val, str) and val.startswith('$-'):
                return 'color: #ff4444'
            elif isinstance(val, str) and val.startswith('$'):
                return 'color: #00ff00'
            return ''

        try:
            # Try newer pandas API first
            styled_df = df.style.map(color_pnl, subset=['P&L'])
        except AttributeError:
            # Fall back to older API
            styled_df = df.style.applymap(color_pnl, subset=['P&L'])

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades CSV",
            data=csv,
            file_name=f"trades_{result.tickers[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No trades generated with these parameters.")
