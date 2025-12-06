"""
Results Viewer Page - Load and analyze saved optimization results.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_results_page():
    """Render the results viewer page."""

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    output_dir = Path(st.session_state.get('output_dir', r"C:\Users\Warren\Projects\ib_breakout_optimizer\output"))
    opt_dir = output_dir / "optimization"

    st.markdown("""
    Load and analyze saved optimization results from previous runs.
    """)

    # Find available result files
    result_files = []
    if opt_dir.exists():
        result_files.extend(opt_dir.glob("*.json"))
        result_files.extend(opt_dir.glob("*.csv"))

    if not result_files:
        st.warning(f"No optimization results found in {opt_dir}")
        st.info("Run an optimization first to generate results.")
        return

    # Group files by type
    json_files = [f for f in result_files if f.suffix == '.json']
    csv_files = [f for f in result_files if f.suffix == '.csv']

    # File selector
    st.subheader("Select Results File")

    col1, col2 = st.columns(2)

    with col1:
        file_type = st.radio("File Type", ["JSON (Full Results)", "CSV (Summary)"])

    with col2:
        if file_type == "JSON (Full Results)":
            if json_files:
                selected_file = st.selectbox(
                    "Select File",
                    json_files,
                    format_func=lambda x: f"{x.stem} ({x.stat().st_mtime:.0f})"
                )
            else:
                st.warning("No JSON files found")
                selected_file = None
        else:
            if csv_files:
                selected_file = st.selectbox(
                    "Select File",
                    csv_files,
                    format_func=lambda x: f"{x.stem} ({x.stat().st_mtime:.0f})"
                )
            else:
                st.warning("No CSV files found")
                selected_file = None

    if selected_file is None:
        return

    # Load button
    if st.button("Load Results", type="primary"):
        load_and_display_results(selected_file)


def load_and_display_results(filepath):
    """Load results from file and display."""

    filepath = Path(filepath)
    st.markdown(f"### Results from: `{filepath.name}`")

    try:
        if filepath.suffix == '.json':
            display_json_results(filepath)
        else:
            display_csv_results(filepath)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def display_json_results(filepath):
    """Display JSON optimization results."""

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Detect result type
    if 'best_params' in data:
        display_bayesian_json(data)
    elif 'periods' in data:
        display_walk_forward_json(data)
    elif 'results' in data:
        display_grid_search_json(data)
    else:
        # Generic JSON display
        st.json(data)


def display_grid_search_json(data):
    """Display grid search results from JSON."""

    st.markdown("### Grid Search Results")

    # Meta info
    if 'meta' in data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker", data['meta'].get('ticker', 'N/A'))
        with col2:
            st.metric("Total Combinations", data['meta'].get('total_combinations', 'N/A'))
        with col3:
            st.metric("Objective", data['meta'].get('objective', 'N/A'))

    # Best result
    if 'best' in data:
        st.markdown("### Best Parameters")
        best = data['best']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Objective", f"{best.get('objective_value', 0):.4f}")
        with col2:
            st.metric("Total Trades", best.get('total_trades', 0))
        with col3:
            st.metric("Total P&L", f"${best.get('total_pnl', 0):,.2f}")

        # Parameters table
        if 'params' in best:
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v}
                for k, v in best['params'].items()
            ])
            st.dataframe(params_df, use_container_width=True)

    # All results
    if 'results' in data and data['results']:
        st.markdown("### All Results")

        results_df = pd.DataFrame(data['results'])

        # Sort by objective
        if 'objective_value' in results_df.columns:
            results_df = results_df.sort_values('objective_value', ascending=False)

        st.dataframe(results_df, use_container_width=True, height=400)

        # Visualization
        st.markdown("### Objective Distribution")

        if 'objective_value' in results_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results_df['objective_value'],
                nbinsx=30,
                marker_color='#00aaff'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                xaxis_title='Objective Value',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)


def display_bayesian_json(data):
    """Display Bayesian optimization results from JSON."""

    st.markdown("### Bayesian Optimization Results")

    # Best parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Value", f"{data.get('best_value', 0):.4f}")
    with col2:
        st.metric("Total Trials", data.get('n_trials', 0))
    with col3:
        if 'best_metrics' in data:
            st.metric("Best P&L", f"${data['best_metrics'].get('total_pnl', 0):,.2f}")

    # Best parameters
    if 'best_params' in data:
        st.markdown("### Best Parameters")
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in data['best_params'].items()
        ])
        st.dataframe(params_df, use_container_width=True)

    # Trial history
    if 'all_trials' in data and data['all_trials']:
        st.markdown("### Trial History")

        trials_df = pd.DataFrame(data['all_trials'])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Trial Values', 'Best So Far'))

        if 'value' in trials_df.columns:
            fig.add_trace(
                go.Scatter(y=trials_df['value'], mode='markers',
                          marker=dict(color='#00aaff', size=6),
                          name='Trial Value'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(y=trials_df['value'].cummax(), mode='lines',
                          line=dict(color='#00ff00', width=2),
                          name='Best So Far'),
                row=2, col=1
            )

        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Parameter importance (if available)
        display_parameter_analysis(trials_df, data.get('best_params', {}))


def display_walk_forward_json(data):
    """Display walk-forward results from JSON."""

    st.markdown("### Walk-Forward Analysis Results")

    # Summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Periods", len(data.get('periods', [])))
    with col2:
        st.metric("Efficiency Ratio", f"{data.get('efficiency_ratio', 0):.2%}")
    with col3:
        if 'combined_metrics' in data:
            st.metric("Combined P&L", f"${data['combined_metrics'].get('total_pnl', 0):,.2f}")
    with col4:
        if 'combined_metrics' in data:
            st.metric("Combined Sharpe", f"{data['combined_metrics'].get('sharpe_ratio', 0):.2f}")

    # Period breakdown
    if 'periods' in data and data['periods']:
        st.markdown("### Period Breakdown")

        periods_df = pd.DataFrame(data['periods'])
        st.dataframe(periods_df, use_container_width=True)

        # Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('In-Sample vs Out-of-Sample', 'Out-of-Sample P&L'))

        period_labels = [f"P{i+1}" for i in range(len(data['periods']))]

        # IS vs OS comparison
        if 'in_sample_objective' in periods_df.columns:
            fig.add_trace(
                go.Bar(x=period_labels, y=periods_df['in_sample_objective'],
                      name='In-Sample', marker_color='#00aaff'),
                row=1, col=1
            )
        if 'out_sample_objective' in periods_df.columns:
            fig.add_trace(
                go.Bar(x=period_labels, y=periods_df['out_sample_objective'],
                      name='Out-of-Sample', marker_color='#00ff00'),
                row=1, col=1
            )

        # OS P&L
        if 'os_pnl' in periods_df.columns:
            colors = ['#00ff00' if p >= 0 else '#ff4444' for p in periods_df['os_pnl']]
            fig.add_trace(
                go.Bar(x=period_labels, y=periods_df['os_pnl'],
                      name='OS P&L', marker_color=colors),
                row=2, col=1
            )

        fig.update_layout(
            template='plotly_dark',
            height=500,
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)


def display_csv_results(filepath):
    """Display CSV results."""

    df = pd.read_csv(filepath)

    st.markdown(f"### CSV Results ({len(df)} rows)")

    # Basic stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        if 'objective_value' in df.columns:
            st.metric("Best Objective", f"{df['objective_value'].max():.4f}")

    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df, use_container_width=True, height=400)

    # Column analysis
    st.markdown("### Column Analysis")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select column to analyze", numeric_cols)

        col1, col2 = st.columns(2)

        with col1:
            # Distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[selected_col],
                nbinsx=30,
                marker_color='#00aaff'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                title=f'{selected_col} Distribution',
                xaxis_title=selected_col,
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Stats
            st.markdown(f"**Statistics for {selected_col}:**")
            stats = df[selected_col].describe()
            stats_df = pd.DataFrame({
                'Statistic': stats.index,
                'Value': stats.values
            })
            st.dataframe(stats_df, use_container_width=True)

    # Download button
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name=filepath.name,
        mime="text/csv"
    )


def display_parameter_analysis(trials_df, best_params):
    """Analyze parameter importance from trial data."""

    if trials_df.empty:
        return

    st.markdown("### Parameter Analysis")

    # Extract parameter columns (exclude system columns)
    system_cols = ['trial_number', 'value', 'state', 'datetime']
    param_cols = [c for c in trials_df.columns if c not in system_cols]

    if not param_cols:
        return

    # Correlation with objective
    if 'value' in trials_df.columns:
        correlations = {}
        for col in param_cols:
            if trials_df[col].dtype in ['float64', 'int64', 'bool']:
                corr = trials_df[col].astype(float).corr(trials_df['value'])
                if pd.notna(corr):
                    correlations[col] = corr

        if correlations:
            corr_df = pd.DataFrame([
                {'Parameter': k, 'Correlation': v}
                for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            ])

            fig = go.Figure()
            colors = ['#00ff00' if c >= 0 else '#ff4444' for c in corr_df['Correlation']]
            fig.add_trace(go.Bar(
                x=corr_df['Parameter'],
                y=corr_df['Correlation'],
                marker_color=colors
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                title='Parameter Correlation with Objective',
                xaxis_title='Parameter',
                yaxis_title='Correlation'
            )
            st.plotly_chart(fig, use_container_width=True)

