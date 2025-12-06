"""
Candlestick Chart Component - Reusable Plotly candlestick charts.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    height: int = 500,
    show_volume: bool = True,
    markers: Optional[List[Dict[str, Any]]] = None,
    hlines: Optional[List[Dict[str, Any]]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None
) -> Optional[go.Figure]:
    """
    Create a candlestick chart with optional volume, markers, and horizontal lines.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume (optional)
        title: Chart title
        height: Chart height in pixels
        show_volume: Whether to show volume subplot
        markers: List of marker dicts with keys: time, price, symbol, color, name, text
        hlines: List of horizontal line dicts with keys: y, color, dash, text
        annotations: List of annotation dicts

    Returns:
        Plotly Figure object or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None

    if df.empty:
        return None

    has_volume = 'volume' in df.columns and show_volume

    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
            subplot_titles=('', 'Volume')
        )
    else:
        fig = go.Figure()

    # Candlesticks
    candlestick = go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff00',
        decreasing_fillcolor='#ff4444'
    )

    if has_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)

    # Volume bars
    if has_volume:
        colors = ['#00ff00' if c >= o else '#ff4444'
                 for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False
            ),
            row=2, col=1
        )

    # Add horizontal lines
    if hlines:
        for hline in hlines:
            fig.add_hline(
                y=hline['y'],
                line_dash=hline.get('dash', 'dash'),
                line_color=hline.get('color', 'cyan'),
                annotation_text=hline.get('text', ''),
                row=1 if has_volume else None,
                col=1 if has_volume else None
            )

    # Add markers
    if markers:
        for marker in markers:
            scatter = go.Scatter(
                x=[marker['time']],
                y=[marker['price']],
                mode='markers',
                marker=dict(
                    symbol=marker.get('symbol', 'circle'),
                    size=marker.get('size', 12),
                    color=marker.get('color', '#00aaff'),
                    line=dict(width=2, color='white')
                ),
                name=marker.get('name', ''),
                hovertemplate=marker.get('text', f"Price: {marker['price']:.2f}")
            )
            if has_volume:
                fig.add_trace(scatter, row=1, col=1)
            else:
                fig.add_trace(scatter)

    # Add annotations
    if annotations:
        for ann in annotations:
            fig.add_annotation(
                x=ann.get('x'),
                y=ann.get('y'),
                text=ann.get('text', ''),
                showarrow=ann.get('showarrow', True),
                arrowhead=ann.get('arrowhead', 2),
                arrowcolor=ann.get('arrowcolor', 'white'),
                font=dict(color=ann.get('fontcolor', 'white'))
            )

    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Remove weekend gaps
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour")
        ]
    )

    return fig


def create_trade_chart(
    df: pd.DataFrame,
    trade: Any,
    ib_high: Optional[float] = None,
    ib_low: Optional[float] = None,
    title: str = "",
    height: int = 600
) -> Optional[go.Figure]:
    """
    Create a candlestick chart specifically for displaying a trade.

    Args:
        df: DataFrame with OHLCV data for the trading day
        trade: Trade object with entry_time, entry_price, exit_time, exit_price, direction, pnl
        ib_high: IB high level to display
        ib_low: IB low level to display
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None

    markers = []
    hlines = []

    # Entry marker
    if hasattr(trade, 'entry_time') and hasattr(trade, 'entry_price'):
        direction = getattr(trade, 'direction', None)
        is_long = direction.value == 'long' if direction else True

        markers.append({
            'time': trade.entry_time,
            'price': trade.entry_price,
            'symbol': 'triangle-up' if is_long else 'triangle-down',
            'color': '#00ff00' if is_long else '#ff4444',
            'size': 15,
            'name': 'Entry',
            'text': f"Entry: ${trade.entry_price:.2f}<br>Time: {trade.entry_time.strftime('%H:%M')}"
        })

    # Exit marker
    if hasattr(trade, 'exit_time') and hasattr(trade, 'exit_price'):
        if trade.exit_time and trade.exit_price:
            pnl = getattr(trade, 'pnl', 0)
            markers.append({
                'time': trade.exit_time,
                'price': trade.exit_price,
                'symbol': 'x',
                'color': '#00ff00' if pnl >= 0 else '#ff4444',
                'size': 12,
                'name': 'Exit',
                'text': f"Exit: ${trade.exit_price:.2f}<br>P&L: ${pnl:.2f}"
            })

    # IB levels
    if ib_high is not None:
        hlines.append({
            'y': ib_high,
            'color': 'cyan',
            'dash': 'dash',
            'text': f'IB High: ${ib_high:.2f}'
        })

    if ib_low is not None:
        hlines.append({
            'y': ib_low,
            'color': 'cyan',
            'dash': 'dash',
            'text': f'IB Low: ${ib_low:.2f}'
        })

    return create_candlestick_chart(
        df=df,
        title=title,
        height=height,
        show_volume=True,
        markers=markers,
        hlines=hlines
    )


def create_equity_chart(
    equity_data: pd.DataFrame,
    height: int = 400,
    show_peak: bool = True,
    show_trades: bool = True
) -> Optional[go.Figure]:
    """
    Create an equity curve chart.

    Args:
        equity_data: DataFrame with columns: datetime, cumulative_pnl, peak, pnl
        height: Chart height
        show_peak: Whether to show peak line
        show_trades: Whether to show individual trade bars

    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE or equity_data.empty:
        return None

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
    if show_peak and 'peak' in equity_data.columns:
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

    # Individual trade bars
    if show_trades and 'pnl' in equity_data.columns:
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
        height=height,
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

    return fig


def create_drawdown_chart(
    equity_data: pd.DataFrame,
    height: int = 300
) -> Optional[go.Figure]:
    """
    Create a drawdown chart.

    Args:
        equity_data: DataFrame with columns: datetime, drawdown
        height: Chart height

    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE or equity_data.empty:
        return None

    if 'drawdown' not in equity_data.columns:
        return None

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
        height=height,
        showlegend=False,
        yaxis_title="Drawdown ($)"
    )

    return fig
