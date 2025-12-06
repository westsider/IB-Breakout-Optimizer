"""
Trade Browser Page - View individual trades with charts.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_trade_browser():
    """Render the trade browser page."""

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    # Check for trades in session state
    if 'last_result' not in st.session_state or not st.session_state['last_result'].trades:
        st.info("No trades to display. Run a backtest first!")

        # Option to load from file
        st.markdown("### Or load trades from file:")
        output_dir = Path(st.session_state.get('output_dir', ''))

        trade_files = list(output_dir.glob("*_trades.csv")) if output_dir.exists() else []

        if trade_files:
            selected_file = st.selectbox("Select trade file", trade_files)
            if st.button("Load Trades"):
                load_trades_from_file(selected_file)
        else:
            st.warning("No trade files found in output directory.")
        return

    result = st.session_state['last_result']
    trades = result.trades

    # Filters
    st.sidebar.markdown("### 🔍 Filters")

    # Direction filter
    direction_filter = st.sidebar.multiselect(
        "Direction",
        ["long", "short"],
        default=["long", "short"]
    )

    # Win/Loss filter
    outcome_filter = st.sidebar.multiselect(
        "Outcome",
        ["Winner", "Loser", "Break-even"],
        default=["Winner", "Loser", "Break-even"]
    )

    # Exit reason filter
    exit_reasons = list(set(t.exit_reason.value for t in trades if t.exit_reason))
    exit_filter = st.sidebar.multiselect(
        "Exit Reason",
        exit_reasons,
        default=exit_reasons
    )

    # Apply filters
    filtered_trades = []
    for t in trades:
        if t.direction.value not in direction_filter:
            continue

        if t.pnl > 0 and "Winner" not in outcome_filter:
            continue
        if t.pnl < 0 and "Loser" not in outcome_filter:
            continue
        if t.pnl == 0 and "Break-even" not in outcome_filter:
            continue

        if t.exit_reason and t.exit_reason.value not in exit_filter:
            continue

        filtered_trades.append(t)

    st.markdown(f"### Showing {len(filtered_trades)} of {len(trades)} trades")

    if not filtered_trades:
        st.warning("No trades match the current filters.")
        return

    # Trade selector
    trade_options = {
        f"#{i+1} | {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {'Long' if t.direction.value == 'long' else 'Short'} | ${t.pnl:.2f}": i
        for i, t in enumerate(filtered_trades)
    }

    selected_trade_label = st.selectbox("Select Trade", list(trade_options.keys()))
    selected_idx = trade_options[selected_trade_label]
    selected_trade = filtered_trades[selected_idx]

    # Display trade details
    display_trade_details(selected_trade)

    # Display trade chart
    display_trade_chart(selected_trade)

    # Trade statistics summary
    st.markdown("### 📊 Filtered Trade Statistics")
    display_trade_stats(filtered_trades)


def display_trade_details(trade):
    """Display details for a single trade."""

    st.markdown("### 📝 Trade Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        direction_emoji = "🟢" if trade.direction.value == "long" else "🔴"
        st.metric("Direction", f"{direction_emoji} {trade.direction.value.title()}")

    with col2:
        st.metric("Entry Price", f"${trade.entry_price:.2f}")

    with col3:
        st.metric("Exit Price", f"${trade.exit_price:.2f}" if trade.exit_price else "Open")

    with col4:
        pnl_color = "normal" if trade.pnl >= 0 else "inverse"
        st.metric("P&L", f"${trade.pnl:.2f}", delta_color=pnl_color)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Entry Time", trade.entry_time.strftime('%Y-%m-%d %H:%M'))

    with col2:
        st.metric("Exit Time", trade.exit_time.strftime('%H:%M') if trade.exit_time else "")

    with col3:
        st.metric("Bars Held", trade.bars_held)

    with col4:
        st.metric("Exit Reason", trade.exit_reason.value if trade.exit_reason else "")

    # IB details if available
    if trade.ib:
        st.markdown("#### Initial Balance")
        ib_col1, ib_col2, ib_col3 = st.columns(3)

        with ib_col1:
            st.metric("IB High", f"${trade.ib.ib_high:.2f}")

        with ib_col2:
            st.metric("IB Low", f"${trade.ib.ib_low:.2f}")

        with ib_col3:
            st.metric("IB Range %", f"{trade.ib.ib_range_pct:.2f}%")


def display_trade_chart(trade):
    """Display candlestick chart for a trade."""

    st.markdown("### 📈 Trade Chart")

    # Load price data around the trade
    data_dir = Path(st.session_state.get('data_dir', r"C:\Users\Warren\Downloads"))

    try:
        from data.data_loader import DataLoader

        loader = DataLoader(str(data_dir))

        # Find data file for ticker
        ticker = trade.ticker
        filepath = None
        for f in data_dir.iterdir():
            if f.is_file() and ticker.upper() in f.name.upper():
                if '_NT' in f.name.upper() and f.suffix.lower() == '.txt':
                    filepath = f
                    break

        if not filepath:
            st.warning(f"Could not find data file for {ticker}")
            return

        df = loader.load_auto_detect(str(filepath), ticker)

        # Filter to trade day
        trade_date = trade.entry_time.date()
        day_start = datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=0))
        day_end = datetime.combine(trade_date, datetime.min.time().replace(hour=16, minute=0))

        mask = (df['timestamp'] >= day_start) & (df['timestamp'] <= day_end)
        day_df = df[mask].copy()

        if day_df.empty:
            st.warning("No data available for trade date")
            return

        # Create candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
            subplot_titles=('', 'Volume')
        )

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=day_df['timestamp'],
                open=day_df['open'],
                high=day_df['high'],
                low=day_df['low'],
                close=day_df['close'],
                name='Price',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )

        # IB High/Low lines
        if trade.ib:
            fig.add_hline(
                y=trade.ib.ib_high,
                line_dash="dash",
                line_color="cyan",
                annotation_text=f"IB High: ${trade.ib.ib_high:.2f}",
                row=1, col=1
            )
            fig.add_hline(
                y=trade.ib.ib_low,
                line_dash="dash",
                line_color="cyan",
                annotation_text=f"IB Low: ${trade.ib.ib_low:.2f}",
                row=1, col=1
            )

        # Entry marker
        entry_color = "#00ff00" if trade.direction.value == "long" else "#ff4444"
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade.direction.value == 'long' else 'triangle-down',
                    size=15,
                    color=entry_color,
                    line=dict(width=2, color='white')
                ),
                name='Entry',
                hovertemplate=f"Entry: ${trade.entry_price:.2f}<br>Time: {trade.entry_time.strftime('%H:%M')}"
            ),
            row=1, col=1
        )

        # Exit marker
        if trade.exit_time and trade.exit_price:
            exit_color = "#00ff00" if trade.pnl >= 0 else "#ff4444"
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color=exit_color,
                        line=dict(width=2, color='white')
                    ),
                    name='Exit',
                    hovertemplate=f"Exit: ${trade.exit_price:.2f}<br>P&L: ${trade.pnl:.2f}"
                ),
                row=1, col=1
            )

        # Volume bars
        if 'volume' in day_df.columns:
            colors = ['#00ff00' if c >= o else '#ff4444'
                     for c, o in zip(day_df['close'], day_df['open'])]
            fig.add_trace(
                go.Bar(
                    x=day_df['timestamp'],
                    y=day_df['volume'],
                    marker_color=colors,
                    name='Volume',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"{ticker} - {trade_date.strftime('%Y-%m-%d')}",
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[20, 4], pattern="hour")
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def display_trade_stats(trades):
    """Display statistics for filtered trades."""

    if not trades:
        return

    col1, col2, col3, col4 = st.columns(4)

    total_pnl = sum(t.pnl for t in trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl < 0]

    with col1:
        st.metric("Total P&L", f"${total_pnl:,.2f}")

    with col2:
        win_rate = len(winners) / len(trades) * 100 if trades else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col3:
        avg_win = sum(t.pnl for t in winners) / len(winners) if winners else 0
        st.metric("Avg Winner", f"${avg_win:.2f}")

    with col4:
        avg_loss = sum(t.pnl for t in losers) / len(losers) if losers else 0
        st.metric("Avg Loser", f"${avg_loss:.2f}")


def load_trades_from_file(filepath):
    """Load trades from a CSV file."""
    st.info(f"Loading from {filepath}... (Not yet implemented)")
    # TODO: Implement trade loading from CSV
