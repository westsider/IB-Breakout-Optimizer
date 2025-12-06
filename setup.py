"""
Setup script for IB Breakout Optimizer.
"""

from setuptools import setup, find_packages

setup(
    name="ib_breakout_optimizer",
    version="0.1.0",
    description="Custom Python backtester with self-optimization for IB Breakout strategy",
    author="Warren",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.6.0",
        ],
        "optimization": [
            "optuna>=3.4.0",
            "scikit-learn>=1.3.0",
        ],
        "ui": [
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
        ],
        "ml": [
            "lightgbm>=4.1.0",
            "xgboost>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ib-backtest=backtester.backtest_runner:main",
        ],
    },
)
