"""
Strategy module for IB Breakout Optimizer.

Contains IB calculation, signal generation, and exit logic.
"""

from .ib_calculator import IBCalculator, IBBreakoutDetector

__all__ = [
    "IBCalculator",
    "IBBreakoutDetector",
]
