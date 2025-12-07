"""
ML Trade Filter - LightGBM classifier for predicting trade outcomes.

This module provides:
- Model training with cross-validation
- Feature importance analysis
- Model persistence (save/load)
- Prediction interface for the optimizer
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from ml_filter.feature_builder import FeatureBuilder


@dataclass
class TrainingResult:
    """Results from model training."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    n_samples: int
    n_winners: int
    n_losers: int
    train_date: str

    def summary(self) -> str:
        """Get summary string of results."""
        lines = [
            "=" * 50,
            "ML TRADE FILTER TRAINING RESULTS",
            "=" * 50,
            f"Training Date: {self.train_date}",
            f"Samples: {self.n_samples} ({self.n_winners} wins, {self.n_losers} losses)",
            "",
            "Model Performance:",
            f"  Accuracy:  {self.accuracy:.1%}",
            f"  Precision: {self.precision:.1%}",
            f"  Recall:    {self.recall:.1%}",
            f"  F1 Score:  {self.f1:.3f}",
            f"  ROC AUC:   {self.roc_auc:.3f}",
            "",
            f"Cross-Validation (5-fold):",
            f"  Mean: {self.cv_mean:.1%} (+/- {self.cv_std:.1%})",
            "",
            "Confusion Matrix:",
            f"  TN: {self.confusion_matrix[0,0]:4d}  FP: {self.confusion_matrix[0,1]:4d}",
            f"  FN: {self.confusion_matrix[1,0]:4d}  TP: {self.confusion_matrix[1,1]:4d}",
            "",
            "Top 5 Feature Importance:",
        ]

        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for name, importance in sorted_features:
            lines.append(f"  {name}: {importance:.3f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'n_samples': self.n_samples,
            'n_winners': self.n_winners,
            'n_losers': self.n_losers,
            'train_date': self.train_date
        }


class MLTradeFilter:
    """
    LightGBM-based trade filter for predicting win/loss outcomes.

    Usage:
        # Training
        filter = MLTradeFilter()
        result = filter.train(features_df)
        filter.save('models/tsla_filter.pkl')

        # Prediction
        filter = MLTradeFilter.load('models/tsla_filter.pkl')
        probability = filter.predict_proba(features)
    """

    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'random_state': 42
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ML trade filter.

        Args:
            params: LightGBM parameters (uses defaults if not provided)
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_builder = FeatureBuilder()
        self.feature_names: List[str] = []
        self.training_result: Optional[TrainingResult] = None
        self.ticker: str = ""
        self.train_date: str = ""

    def train(
        self,
        features_df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> TrainingResult:
        """
        Train the ML model on feature data.

        Args:
            features_df: DataFrame with features and target column
            ticker: Ticker symbol (for metadata)
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds

        Returns:
            TrainingResult with metrics and feature importance
        """
        self.ticker = ticker
        self.train_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.feature_names = self.feature_builder.get_feature_names()

        # Extract features and target
        X, y = self.feature_builder.get_feature_matrix(features_df)

        if len(X) < 20:
            raise ValueError(f"Not enough samples for training: {len(X)} (need at least 20)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Create and train model
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = 0.5  # If only one class in test set

        cm = confusion_matrix(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            lgb.LGBMClassifier(**self.params),
            X, y, cv=cv_folds, scoring='accuracy'
        ).tolist()

        # Feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))

        # Store result
        self.training_result = TrainingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            cv_scores=cv_scores,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            feature_importance=importance,
            confusion_matrix=cm,
            n_samples=len(X),
            n_winners=int(y.sum()),
            n_losers=int(len(y) - y.sum()),
            train_date=self.train_date
        )

        return self.training_result

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict win/loss for given features.

        Args:
            features: Feature array (n_samples x n_features)

        Returns:
            Array of predictions (0 = loss, 1 = win)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict win probability for given features.

        Args:
            features: Feature array (n_samples x n_features)

        Returns:
            Array of win probabilities (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(features)[:, 1]

    def predict_single(
        self,
        ib_range_percent: float,
        gap_percent: float,
        prior_days_bullish_count: int,
        avg_daily_range_percent: float,
        entry_hour: int,
        day_of_week: int,
        is_long: bool,
        qqq_confirmed: bool = False,
        distance_from_ib_percent: float = 0.0,
        ib_duration_minutes: int = 30
    ) -> float:
        """
        Predict win probability for a single trade.

        Returns:
            Win probability (0.0 to 1.0)
        """
        features = np.array([[
            ib_range_percent,
            ib_duration_minutes,
            gap_percent,
            1 if gap_percent > 0.1 else 0,  # is_gap_up
            1 if gap_percent < -0.1 else 0,  # is_gap_down
            prior_days_bullish_count,
            avg_daily_range_percent,
            entry_hour,
            day_of_week,
            1 if is_long else 0,
            0 if is_long else 1,  # is_short
            1 if qqq_confirmed else 0,
            distance_from_ib_percent
        ]])

        return self.predict_proba(features)[0]

    def save(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model (pickle format)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        save_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params,
            'ticker': self.ticker,
            'train_date': self.train_date,
            'training_result': self.training_result.to_dict() if self.training_result else None
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        # Also save metadata as JSON for easy inspection
        meta_path = path.with_suffix('.json')
        meta = {
            'ticker': self.ticker,
            'train_date': self.train_date,
            'feature_names': self.feature_names,
            'params': {k: v for k, v in self.params.items() if not callable(v)},
            'training_result': self.training_result.to_dict() if self.training_result else None
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MLTradeFilter':
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            MLTradeFilter instance with loaded model
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        instance = cls(params=save_data.get('params'))
        instance.model = save_data['model']
        instance.feature_names = save_data['feature_names']
        instance.ticker = save_data.get('ticker', '')
        instance.train_date = save_data.get('train_date', '')

        # Restore training result if available
        if save_data.get('training_result'):
            result_dict = save_data['training_result']
            instance.training_result = TrainingResult(
                accuracy=result_dict['accuracy'],
                precision=result_dict['precision'],
                recall=result_dict['recall'],
                f1=result_dict['f1'],
                roc_auc=result_dict['roc_auc'],
                cv_scores=result_dict['cv_scores'],
                cv_mean=result_dict['cv_mean'],
                cv_std=result_dict['cv_std'],
                feature_importance=result_dict['feature_importance'],
                confusion_matrix=np.array(result_dict['confusion_matrix']),
                n_samples=result_dict['n_samples'],
                n_winners=result_dict['n_winners'],
                n_losers=result_dict['n_losers'],
                train_date=result_dict['train_date']
            )

        return instance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        if self.model is None:
            return {}

        return dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for display."""
        return {
            'ticker': self.ticker,
            'train_date': self.train_date,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_trained': self.model is not None,
            'training_result': self.training_result.to_dict() if self.training_result else None
        }
