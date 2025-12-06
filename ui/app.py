"""
IB Breakout Optimizer - Main Streamlit App.

Run with: streamlit run ui/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="IB Breakout Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .positive { color: #00ff00; }
    .negative { color: #ff4444; }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main app entry point."""

    # Sidebar navigation
    st.sidebar.title("📈 IB Breakout Optimizer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Run Backtest", "🔍 Trade Browser", "📈 Equity Curve",
         "⚙️ Optimization", "📋 Results Viewer"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    # Global settings in sidebar
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=r"C:\Users\Warren\Downloads",
        help="Directory containing price data files"
    )

    output_dir = st.sidebar.text_input(
        "Output Directory",
        value=r"C:\Users\Warren\Projects\ib_breakout_optimizer\output",
        help="Directory for backtest results"
    )

    # Store in session state
    st.session_state['data_dir'] = data_dir
    st.session_state['output_dir'] = output_dir

    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Run Backtest":
        show_backtest_page()
    elif page == "🔍 Trade Browser":
        show_trade_browser_page()
    elif page == "📈 Equity Curve":
        show_equity_page()
    elif page == "⚙️ Optimization":
        show_optimization_page()
    elif page == "📋 Results Viewer":
        show_results_page()


def show_home_page():
    """Home page with overview and quick stats."""
    st.title("🏠 IB Breakout Strategy Optimizer")

    st.markdown("""
    Welcome to the IB Breakout Strategy Optimizer! This tool helps you:

    - **Run Backtests**: Test the IB breakout strategy on historical data
    - **Browse Trades**: View individual trades with entry/exit markers on charts
    - **Analyze Performance**: View equity curves, drawdowns, and key metrics
    - **Optimize Parameters**: Find optimal strategy parameters using grid search or Bayesian optimization
    - **Review Results**: Load and compare previous optimization results

    ### Quick Start
    1. Go to **Run Backtest** to test the strategy
    2. View trades in **Trade Browser**
    3. Check **Equity Curve** for performance visualization
    4. Use **Optimization** to find better parameters
    """)

    # Show available data files
    st.markdown("### 📁 Available Data Files")
    data_dir = Path(st.session_state.get('data_dir', r"C:\Users\Warren\Downloads"))

    if data_dir.exists():
        data_files = []
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.txt', '.csv']:
                if any(t in f.name.upper() for t in ['TSLA', 'QQQ', 'AAPL', 'NVDA', 'MSFT']):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    data_files.append({
                        'File': f.name,
                        'Size (MB)': f"{size_mb:.1f}",
                        'Ticker': f.stem.split('_')[0].upper()
                    })

        if data_files:
            import pandas as pd
            st.dataframe(pd.DataFrame(data_files), use_container_width=True)
        else:
            st.warning("No data files found in the specified directory.")
    else:
        st.error(f"Data directory not found: {data_dir}")

    # Show recent results if available
    output_dir = Path(st.session_state.get('output_dir', ''))
    optimization_dir = output_dir / "optimization"

    if optimization_dir.exists():
        st.markdown("### 📊 Recent Optimization Results")
        result_files = sorted(optimization_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]

        if result_files:
            for f in result_files:
                st.text(f"  • {f.name}")
        else:
            st.info("No optimization results found yet.")


def show_backtest_page():
    """Run backtest page."""
    st.title("📊 Run Backtest")

    from ui.pages.backtest import render_backtest_page
    render_backtest_page()


def show_trade_browser_page():
    """Trade browser page."""
    st.title("🔍 Trade Browser")

    from ui.pages.trade_browser import render_trade_browser
    render_trade_browser()


def show_equity_page():
    """Equity curve page."""
    st.title("📈 Equity Curve & Performance")

    from ui.pages.equity_curve import render_equity_page
    render_equity_page()


def show_optimization_page():
    """Optimization page."""
    st.title("⚙️ Parameter Optimization")

    from ui.pages.optimization import render_optimization_page
    render_optimization_page()


def show_results_page():
    """Results viewer page."""
    st.title("📋 Results Viewer")

    from ui.pages.results_viewer import render_results_page
    render_results_page()


if __name__ == "__main__":
    main()
