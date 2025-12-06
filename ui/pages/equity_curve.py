"""
Equity Curve Page - Performance visualization.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_equity_page():
    """Render the equity curve and performance page."""

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    # Check for results in session state
    if 'last_result' not in st.session_state or not st.session_state['last_result'].trades:
        st.info("No backtest results to display. Run a backtest first!")
        return

    result = st.session_state['last_result']
    metrics = st.session_state.get('last_metrics')
    trades = result.trades

    # Build equity curve data
    equity_data = build_equity_data(trades)

    # Main equity curve chart
    st.markdown("### 📈 Equity Curve")
    display_equity_chart(equity_data)

    # Drawdown chart
    st.markdown("### 📉 Drawdown")
    display_drawdown_chart(equity_data)

    # Monthly returns heatmap
    st.markdown("### 📅 Monthly Returns")
    display_monthly_returns(equity_data)

    # Win/Loss distribution
    st.markdown("### 📊 Trade Distribution")
    display_trade_distribution(trades)

    # Performance by day of week
    st.markdown("### 📆 Performance by Day of Week")
    display_dow_performance(trades)

    # Performance by hour
    st.markdown("### ⏰ Performance by Entry Hour")
    display_hourly_performance(trades)


def build_equity_data(trades):
    """Build equity curve dataframe from trades."""

    if not trades:
        return pd.DataFrame()

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)

    data = []
    cumulative_pnl = 0
    peak = 0

    for t in sorted_trades:
        cumulative_pnl += t.pnl
        peak = max(peak, cumulative_pnl)
        drawdown = cumulative_pnl - peak

        data.append({
            'date': t.exit_time.date() if t.exit_time else t.entry_time.date(),
            'datetime': t.exit_time if t.exit_time else t.entry_time,
            'pnl': t.pnl,
            'cumulative_pnl': cumulative_pnl,
            'peak': peak,
            'drawdown': drawdown,
            'drawdown_pct': (drawdown / peak * 100) if peak > 0 else 0,
            'direction': t.direction.value,
            'ticker': t.ticker
        })

    return pd.DataFrame(data)


def display_equity_chart(equity_data):
    """Display the main equity curve chart."""

    if equity_data.empty:
        st.warning("No equity data available.")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Cumulative P&L', 'Trade P&L')
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_data['datetime'],
            y=equity_data['cumulative_pnl'],
            mode='lines',
            name='Equity',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        row=1, col=1
    )

    # Peak line
    fig.add_trace(
        go.Scatter(
            x=equity_data['datetime'],
            y=equity_data['peak'],
            mode='lines',
            name='Peak',
            line=dict(color='cyan', width=1, dash='dot')
        ),
        row=1, col=1
    )

    # Individual trade P&L bars
    colors = ['#00ff00' if pnl >= 0 else '#ff4444' for pnl in equity_data['pnl']]
    fig.add_trace(
        go.Bar(
            x=equity_data['datetime'],
            y=equity_data['pnl'],
            name='Trade P&L',
            marker_color=colors
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade P&L ($)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def display_drawdown_chart(equity_data):
    """Display drawdown chart."""

    if equity_data.empty:
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_data['datetime'],
            y=equity_data['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)'
        )
    )

    # Mark max drawdown
    max_dd_idx = equity_data['drawdown'].idxmin()
    if pd.notna(max_dd_idx):
        max_dd_row = equity_data.loc[max_dd_idx]
        fig.add_annotation(
            x=max_dd_row['datetime'],
            y=max_dd_row['drawdown'],
            text=f"Max DD: ${max_dd_row['drawdown']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='white',
            font=dict(color='white')
        )

    fig.update_layout(
        template='plotly_dark',
        height=300,
        showlegend=False,
        yaxis_title="Drawdown ($)"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_monthly_returns(equity_data):
    """Display monthly returns heatmap."""

    if equity_data.empty:
        st.info("Not enough data for monthly returns.")
        return

    # Group by month
    equity_data['year'] = pd.to_datetime(equity_data['date']).dt.year
    equity_data['month'] = pd.to_datetime(equity_data['date']).dt.month

    monthly = equity_data.groupby(['year', 'month'])['pnl'].sum().reset_index()
    monthly_pivot = monthly.pivot(index='year', columns='month', values='pnl')

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pivot.columns = [month_names[m-1] for m in monthly_pivot.columns]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values,
        x=monthly_pivot.columns,
        y=monthly_pivot.index,
        colorscale=[
            [0, '#ff4444'],
            [0.5, '#1e1e1e'],
            [1, '#00ff00']
        ],
        zmid=0,
        text=[[f"${v:.0f}" if pd.notna(v) else "" for v in row]
              for row in monthly_pivot.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>P&L: $%{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        template='plotly_dark',
        height=200,
        xaxis_title="Month",
        yaxis_title="Year"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_trade_distribution(trades):
    """Display trade P&L distribution."""

    if not trades:
        return

    pnls = [t.pnl for t in trades]

    col1, col2 = st.columns(2)

    with col1:
        # P&L histogram
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=pnls,
                nbinsx=30,
                marker_color='#00aaff',
                name='P&L Distribution'
            )
        )

        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=np.mean(pnls), line_dash="dot", line_color="yellow",
                     annotation_text=f"Mean: ${np.mean(pnls):.2f}")

        fig.update_layout(
            template='plotly_dark',
            height=300,
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Win/Loss pie chart
        winners = len([t for t in trades if t.pnl > 0])
        losers = len([t for t in trades if t.pnl < 0])
        breakeven = len([t for t in trades if t.pnl == 0])

        fig = go.Figure(data=[go.Pie(
            labels=['Winners', 'Losers', 'Break-even'],
            values=[winners, losers, breakeven],
            marker_colors=['#00ff00', '#ff4444', '#888888'],
            hole=0.4,
            textinfo='percent+value'
        )])

        fig.update_layout(
            template='plotly_dark',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)


def display_dow_performance(trades):
    """Display performance by day of week."""

    if not trades:
        return

    # Group by day of week
    dow_data = {}
    for t in trades:
        dow = t.entry_time.strftime('%A')
        if dow not in dow_data:
            dow_data[dow] = {'pnl': 0, 'count': 0, 'wins': 0}
        dow_data[dow]['pnl'] += t.pnl
        dow_data[dow]['count'] += 1
        if t.pnl > 0:
            dow_data[dow]['wins'] += 1

    # Order days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    days = [d for d in day_order if d in dow_data]
    pnls = [dow_data[d]['pnl'] for d in days]
    counts = [dow_data[d]['count'] for d in days]
    win_rates = [dow_data[d]['wins'] / dow_data[d]['count'] * 100 if dow_data[d]['count'] > 0 else 0
                 for d in days]

    col1, col2 = st.columns(2)

    with col1:
        colors = ['#00ff00' if p >= 0 else '#ff4444' for p in pnls]
        fig = go.Figure(data=[
            go.Bar(x=days, y=pnls, marker_color=colors, name='P&L')
        ])
        fig.update_layout(
            template='plotly_dark',
            height=250,
            title="P&L by Day",
            yaxis_title="P&L ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[
            go.Bar(x=days, y=counts, marker_color='#00aaff', name='Trades')
        ])
        fig.update_layout(
            template='plotly_dark',
            height=250,
            title="Trade Count by Day",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_hourly_performance(trades):
    """Display performance by entry hour."""

    if not trades:
        return

    # Group by hour
    hour_data = {}
    for t in trades:
        hour = t.entry_time.hour
        if hour not in hour_data:
            hour_data[hour] = {'pnl': 0, 'count': 0}
        hour_data[hour]['pnl'] += t.pnl
        hour_data[hour]['count'] += 1

    hours = sorted(hour_data.keys())
    pnls = [hour_data[h]['pnl'] for h in hours]
    counts = [hour_data[h]['count'] for h in hours]
    hour_labels = [f"{h}:00" for h in hours]

    col1, col2 = st.columns(2)

    with col1:
        colors = ['#00ff00' if p >= 0 else '#ff4444' for p in pnls]
        fig = go.Figure(data=[
            go.Bar(x=hour_labels, y=pnls, marker_color=colors)
        ])
        fig.update_layout(
            template='plotly_dark',
            height=250,
            title="P&L by Entry Hour",
            yaxis_title="P&L ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[
            go.Bar(x=hour_labels, y=counts, marker_color='#00aaff')
        ])
        fig.update_layout(
            template='plotly_dark',
            height=250,
            title="Trades by Entry Hour",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
