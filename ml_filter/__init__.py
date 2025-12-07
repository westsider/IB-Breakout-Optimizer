"""
ML Filter Module - Machine learning trade filter for IB Breakout Strategy.

This module provides:
- Feature extraction from trades and market data
- LightGBM classifier for win/loss prediction
- Model persistence (save/load)
- Integration with the optimizer
"""

from ml_filter.feature_builder import FeatureBuilder
from ml_filter.model_trainer import MLTradeFilter, TrainingResult

__all__ = ['FeatureBuilder', 'MLTradeFilter', 'TrainingResult']
