"""
Optimization Page - Run parameter optimization.
"""

import streamlit as st
import sys
import subprocess
import json
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output"


def render_optimization_page():
    """Render the optimization configuration page."""

    st.markdown("""
    Run parameter optimization to find the best strategy parameters.
    Choose between Grid Search (exhaustive) or Bayesian Optimization (efficient).
    """)

    # Optimization mode
    mode = st.radio(
        "Optimization Mode",
        ["Grid Search", "Bayesian (Optuna)", "Walk-Forward"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Data Settings")

        ticker = st.selectbox(
            "Ticker",
            ["TSLA", "QQQ", "AAPL", "NVDA", "MSFT"],
            index=0
        )

        use_qqq_filter = st.checkbox(
            "Include QQQ Filter in optimization",
            value=True,
            help="Test with and without QQQ filter"
        )

        objective = st.selectbox(
            "Objective Function",
            ["sharpe_ratio", "sortino_ratio", "profit_factor", "k_ratio",
             "total_profit", "calmar_ratio", "win_rate"],
            index=0
        )

    with col2:
        st.subheader("🔧 Optimization Settings")

        preset = st.selectbox(
            "Parameter Preset",
            ["quick", "standard", "full", "exits_only"],
            index=1,
            help="Quick: 3 params, Standard: 6 params, Full: all params"
        )

        if mode == "Bayesian (Optuna)":
            n_trials = st.slider(
                "Number of Trials",
                min_value=20,
                max_value=500,
                value=100,
                step=10
            )

        if mode == "Walk-Forward":
            wf_col1, wf_col2 = st.columns(2)
            with wf_col1:
                in_sample_days = st.number_input(
                    "In-Sample Days",
                    min_value=30,
                    max_value=365,
                    value=90
                )
            with wf_col2:
                out_sample_days = st.number_input(
                    "Out-of-Sample Days",
                    min_value=7,
                    max_value=90,
                    value=30
                )

            anchored = st.checkbox(
                "Anchored (Expanding Window)",
                value=False,
                help="If checked, uses expanding window instead of rolling"
            )

    # Parameter customization
    with st.expander("🔬 Customize Parameter Ranges"):
        st.markdown("Override default parameter ranges:")

        param_col1, param_col2, param_col3 = st.columns(3)

        with param_col1:
            profit_range = st.slider(
                "Profit Target % Range",
                min_value=0.3,
                max_value=3.0,
                value=(0.5, 2.0),
                step=0.1
            )

            directions = st.multiselect(
                "Trade Directions",
                ["long_only", "short_only", "both"],
                default=["long_only", "both"]
            )

        with param_col2:
            stop_types = st.multiselect(
                "Stop Loss Types",
                ["opposite_ib", "match_target"],
                default=["opposite_ib", "match_target"]
            )

            ib_durations = st.multiselect(
                "IB Durations",
                [15, 30, 45, 60],
                default=[30]
            )

        with param_col3:
            include_trailing = st.checkbox("Include Trailing Stop", value=False)
            include_breakeven = st.checkbox("Include Break-Even", value=False)
            include_maxbars = st.checkbox("Include Max Bars", value=False)

    st.markdown("---")

    # Run button
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        run_optimization(
            mode=mode,
            ticker=ticker,
            use_qqq_filter=use_qqq_filter,
            objective=objective,
            preset=preset,
            n_trials=n_trials if mode == "Bayesian (Optuna)" else 50,
            in_sample_days=in_sample_days if mode == "Walk-Forward" else 90,
            out_sample_days=out_sample_days if mode == "Walk-Forward" else 30,
            anchored=anchored if mode == "Walk-Forward" else False,
            profit_range=profit_range,
            directions=directions,
            stop_types=stop_types
        )


def run_optimization(**kwargs):
    """Execute the optimization."""

    data_dir = st.session_state.get('data_dir', r"C:\Users\Warren\Downloads")
    output_dir = st.session_state.get('output_dir', r"C:\Users\Warren\Projects\ib_breakout_optimizer\output")

    mode = kwargs['mode']

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if mode == "Grid Search":
            run_grid_search(data_dir, output_dir, progress_bar, status_text, **kwargs)

        elif mode == "Bayesian (Optuna)":
            run_bayesian(data_dir, output_dir, progress_bar, status_text, **kwargs)

        elif mode == "Walk-Forward":
            run_walk_forward(data_dir, output_dir, progress_bar, status_text, **kwargs)

    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    progress_bar.progress(100)


def run_grid_search(data_dir, output_dir, progress_bar, status_text, **kwargs):
    """Run grid search optimization using subprocess for full CPU utilization.

    Uses a non-blocking approach with st.status for live progress updates.
    """

    import uuid

    # Create unique progress file for this run
    run_id = uuid.uuid4().hex[:8]
    progress_file = OUTPUT_DIR / f"optimization_progress_{run_id}.json"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build command
    script_path = SCRIPTS_DIR / "run_optimization.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--ticker", kwargs['ticker'],
        "--preset", kwargs['preset'],
        "--objective", kwargs['objective'],
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--progress-file", str(progress_file)
    ]

    if kwargs['use_qqq_filter']:
        cmd.append("--use-qqq-filter")

    # Use st.status for expandable progress container with live updates
    with st.status(f"🚀 Running Grid Search Optimization for {kwargs['ticker']}...", expanded=True) as status:
        st.write(f"**Preset:** {kwargs['preset']} | **Objective:** {kwargs['objective']}")
        st.write(f"**QQQ Filter:** {'Enabled' if kwargs['use_qqq_filter'] else 'Disabled'}")

        # Progress display elements
        progress_text = st.empty()
        metrics_cols = st.columns(4)
        col_completed = metrics_cols[0].empty()
        col_objective = metrics_cols[1].empty()
        col_trades = metrics_cols[2].empty()
        col_pnl = metrics_cols[3].empty()

        progress_text.write("⏳ Launching parallel workers...")

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT)
        )

        progress_bar.progress(10)

        # Poll for progress with streamlit-compatible updates
        poll_count = 0
        last_current = 0

        while process.poll() is None:
            poll_count += 1

            # Check for progress file
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        content = f.read()
                        if content.strip():
                            progress_data = json.loads(content)

                            # Check if optimization is complete
                            if progress_data.get('status') == 'complete':
                                break

                            # Extract progress info
                            current = progress_data.get('current', 0)
                            total = progress_data.get('total', 1)
                            percent = progress_data.get('percent', 0)
                            best_obj = progress_data.get('best_objective')
                            best_trades = progress_data.get('best_trades')
                            best_pnl = progress_data.get('best_pnl')

                            # Update progress bar (10-90 range)
                            pct = 10 + int(80 * percent / 100)
                            progress_bar.progress(min(pct, 90))

                            # Calculate speed
                            speed = ""
                            if current > last_current:
                                speed = f" ({current - last_current} combos/update)"
                                last_current = current

                            # Update progress text
                            progress_text.write(
                                f"**Progress:** {current:,} / {total:,} combinations ({percent:.1f}%){speed}"
                            )

                            # Update metric displays
                            if best_obj is not None:
                                col_completed.metric("Completed", f"{current:,}")
                                col_objective.metric(f"Best {kwargs['objective']}", f"{best_obj:.4f}")
                                col_trades.metric("Trades", best_trades or 0)
                                col_pnl.metric("P&L", f"${best_pnl:,.2f}" if best_pnl else "$0")

                except (json.JSONDecodeError, FileNotFoundError, ValueError):
                    pass

            # Sleep between polls
            time.sleep(0.5)

        # Wait for process to complete
        stdout, stderr = process.communicate()

        progress_bar.progress(95)

        # Check for errors
        if process.returncode != 0:
            status.update(label="❌ Optimization Failed", state="error")
            st.error(f"Optimization failed with return code {process.returncode}")
            if stderr:
                st.code(stderr)
            return

        # Update status to complete
        status.update(label="✅ Optimization Complete!", state="complete")

    # Load and display final results
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                final_data = json.load(f)

            # Display results from the JSON data
            display_subprocess_results(final_data, kwargs['objective'])

            # Clean up progress file
            try:
                os.remove(progress_file)
            except:
                pass

        except Exception as e:
            st.error(f"Error loading results: {e}")
            if stdout:
                st.code(stdout)
    else:
        st.warning("Results file not found. Check console output.")
        if stdout:
            st.code(stdout)


def display_subprocess_results(results_data, objective):
    """Display results from subprocess optimization."""

    st.success("✅ Grid Search Complete!")

    best_params = results_data.get('best_params')
    best_metrics = results_data.get('best_metrics')
    best_objective = results_data.get('best_objective')

    if best_params and best_metrics:
        st.markdown("### 🏆 Best Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Objective", f"{best_objective:.4f}")

        with col2:
            st.metric("Total Trades", best_metrics.get('total_trades', 0))

        with col3:
            st.metric("Total P&L", f"${best_metrics.get('total_pnl', 0):,.2f}")

        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Win Rate", f"{best_metrics.get('win_rate', 0):.1f}%")

        with col2:
            st.metric("Profit Factor", f"{best_metrics.get('profit_factor', 0):.2f}")

        with col3:
            st.metric("Sharpe Ratio", f"{best_metrics.get('sharpe_ratio', 0):.2f}")

        with col4:
            st.metric("Max Drawdown", f"${best_metrics.get('max_drawdown', 0):,.2f}")

        # Best parameters table
        st.markdown("#### Parameters:")
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in best_params.items()
            if not k.startswith('trade_') or k == 'trade_direction'
        ])
        st.dataframe(params_df, use_container_width=True, height=300)

        # Summary stats
        st.markdown("#### Optimization Summary")
        summary_data = {
            'Total Combinations': results_data.get('total_combinations', 0),
            'Completed': results_data.get('completed', 0),
            'Total Time (seconds)': f"{results_data.get('total_time', 0):.1f}"
        }
        st.json(summary_data)

    else:
        st.warning("No valid results found. The optimization may have produced no profitable combinations.")


def run_bayesian(data_dir, output_dir, progress_bar, status_text, **kwargs):
    """Run Bayesian optimization with Optuna."""

    from optimization.bayesian_optuna import BayesianOptimizer
    from optimization.parameter_space import create_parameter_space

    status_text.text("Initializing Bayesian optimization...")

    optimizer = BayesianOptimizer(
        data_dir=data_dir,
        output_dir=f"{output_dir}/optimization"
    )

    status_text.text(f"Loading data for {kwargs['ticker']}...")
    progress_bar.progress(10)

    filter_ticker = 'QQQ' if kwargs['use_qqq_filter'] and kwargs['ticker'] != 'QQQ' else None
    optimizer.load_data(kwargs['ticker'], filter_ticker=filter_ticker)

    progress_bar.progress(20)

    space = create_parameter_space(kwargs['preset'])

    if kwargs['use_qqq_filter']:
        space.parameters['use_qqq_filter'].enabled = True

    status_text.text(f"Running Optuna optimization ({kwargs['n_trials']} trials)...")

    results = optimizer.optimize(
        parameter_space=space,
        n_trials=kwargs['n_trials'],
        objective=kwargs['objective']
    )

    progress_bar.progress(95)
    status_text.text("Optimization complete!")

    display_bayesian_results(results)


def run_walk_forward(data_dir, output_dir, progress_bar, status_text, **kwargs):
    """Run walk-forward analysis."""

    from optimization.walk_forward import WalkForwardAnalyzer
    from optimization.parameter_space import create_parameter_space

    status_text.text("Initializing walk-forward analysis...")

    analyzer = WalkForwardAnalyzer(
        data_dir=data_dir,
        output_dir=f"{output_dir}/optimization"
    )

    status_text.text(f"Loading data for {kwargs['ticker']}...")
    progress_bar.progress(10)

    filter_ticker = 'QQQ' if kwargs['use_qqq_filter'] and kwargs['ticker'] != 'QQQ' else None
    analyzer.load_data(kwargs['ticker'], filter_ticker=filter_ticker)

    progress_bar.progress(20)

    space = create_parameter_space(kwargs['preset'])

    if kwargs['use_qqq_filter']:
        space.parameters['use_qqq_filter'].enabled = True

    status_text.text("Running walk-forward analysis...")

    results = analyzer.analyze(
        in_sample_days=kwargs['in_sample_days'],
        out_sample_days=kwargs['out_sample_days'],
        anchored=kwargs['anchored'],
        parameter_space=space,
        objective=kwargs['objective']
    )

    progress_bar.progress(95)
    status_text.text("Analysis complete!")

    display_walk_forward_results(results)


def display_optimization_results(results):
    """Display grid search results."""

    st.success("✅ Grid Search Complete!")

    if results.best_result:
        st.markdown("### 🏆 Best Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Objective", f"{results.best_result.objective_value:.4f}")

        with col2:
            st.metric("Total Trades", results.best_result.metrics.total_trades)

        with col3:
            st.metric("Total P&L", f"${results.best_result.metrics.total_net_profit:,.2f}")

        # Best parameters
        st.markdown("#### Parameters:")
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in results.best_result.params.items()
            if not k.startswith('trade_') or k == 'trade_direction'
        ])
        st.dataframe(params_df, use_container_width=True, height=300)

    # Top 10 results
    st.markdown("### 📊 Top 10 Results")

    top_results = results.get_top_n(10)
    if top_results:
        top_df = pd.DataFrame([r.to_dict() for r in top_results])
        display_cols = ['trade_direction', 'profit_target_percent', 'stop_loss_type',
                       'use_qqq_filter', 'objective_value', 'total_trades',
                       'win_rate', 'total_pnl', 'sharpe_ratio']
        available_cols = [c for c in display_cols if c in top_df.columns]
        st.dataframe(top_df[available_cols], use_container_width=True)


def display_bayesian_results(results):
    """Display Bayesian optimization results."""

    st.success("✅ Bayesian Optimization Complete!")

    st.markdown("### 🏆 Best Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Value", f"{results.best_value:.4f}")

    with col2:
        if results.best_metrics:
            st.metric("Total Trades", results.best_metrics.total_trades)

    with col3:
        if results.best_metrics:
            st.metric("Total P&L", f"${results.best_metrics.total_net_profit:,.2f}")

    # Best parameters
    st.markdown("#### Best Parameters:")
    params_df = pd.DataFrame([
        {'Parameter': k, 'Value': v}
        for k, v in results.best_params.items()
    ])
    st.dataframe(params_df, use_container_width=True)

    # Trial history
    if results.all_trials:
        st.markdown("### 📈 Trial History")

        trial_df = pd.DataFrame(results.all_trials)
        if 'value' in trial_df.columns:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=trial_df['value'],
                mode='lines+markers',
                name='Trial Value'
            ))
            fig.add_trace(go.Scatter(
                y=trial_df['value'].cummax(),
                mode='lines',
                name='Best So Far',
                line=dict(dash='dash', color='green')
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                xaxis_title='Trial',
                yaxis_title='Objective Value'
            )
            st.plotly_chart(fig, use_container_width=True)


def display_walk_forward_results(results):
    """Display walk-forward analysis results."""

    st.success("✅ Walk-Forward Analysis Complete!")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Periods", len(results.periods))

    with col2:
        efficiency = results.get_efficiency_ratio()
        st.metric("Efficiency Ratio", f"{efficiency:.2%}")

    with col3:
        if results.combined_metrics:
            st.metric("Combined P&L", f"${results.combined_metrics.total_net_profit:,.2f}")

    with col4:
        if results.combined_metrics:
            st.metric("Combined Sharpe", f"{results.combined_metrics.sharpe_ratio:.2f}")

    # Period-by-period results
    st.markdown("### 📊 Period Results")

    period_data = [p.to_dict() for p in results.periods]
    if period_data:
        period_df = pd.DataFrame(period_data)

        display_cols = ['period_id', 'in_sample_objective', 'out_sample_objective',
                       'is_trades', 'os_trades', 'os_pnl']
        available_cols = [c for c in display_cols if c in period_df.columns]

        st.dataframe(period_df[available_cols], use_container_width=True)

        # Chart
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"P{p['period_id']}" for p in period_data],
            y=[p['in_sample_objective'] for p in period_data],
            name='In-Sample',
            marker_color='#00aaff'
        ))
        fig.add_trace(go.Bar(
            x=[f"P{p['period_id']}" for p in period_data],
            y=[p['out_sample_objective'] for p in period_data],
            name='Out-of-Sample',
            marker_color='#00ff00'
        ))

        fig.update_layout(
            template='plotly_dark',
            height=300,
            barmode='group',
            xaxis_title='Period',
            yaxis_title='Objective Value'
        )

        st.plotly_chart(fig, use_container_width=True)
