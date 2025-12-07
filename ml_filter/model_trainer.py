"""
ML Trade Filter - Ensemble classifier for predicting trade outcomes.

This module provides:
- Ensemble model training (LightGBM + Random Forest + Logistic Regression)
- Time-series cross-validation for proper temporal validation
- Feature importance analysis
- Actionable insights generation
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
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
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
    insights: List[str] = field(default_factory=list)  # Actionable recommendations
    model_type: str = "lightgbm"  # lightgbm or ensemble

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
            'train_date': self.train_date,
            'insights': self.insights,
            'model_type': self.model_type
        }


class MLTradeFilter:
    """
    Ensemble-based trade filter for predicting win/loss outcomes.

    Supports:
    - Single LightGBM model (faster, simpler)
    - Ensemble of LightGBM + Random Forest + Logistic Regression (more robust)

    Usage:
        # Training with ensemble
        filter = MLTradeFilter(use_ensemble=True)
        result = filter.train(features_df)
        filter.save('models/tsla_filter.pkl')

        # Prediction
        filter = MLTradeFilter.load('models/tsla_filter.pkl')
        probability = filter.predict_proba(features)
    """

    # Reduced complexity params to prevent overfitting on small datasets
    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,           # Reduced from 31
        'max_depth': 4,             # Added depth limit
        'learning_rate': 0.05,
        'feature_fraction': 0.7,    # Reduced from 0.8
        'bagging_fraction': 0.7,    # Reduced from 0.8
        'bagging_freq': 5,
        'min_child_samples': 20,    # Prevent overfitting
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 1.0,          # L2 regularization
        'verbose': -1,
        'n_estimators': 50,         # Reduced from 100
        'random_state': 42
    }

    # Random Forest params for ensemble
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_leaf': 10,
        'min_samples_split': 20,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.4,
        'random_forest': 0.4,
        'logistic': 0.2
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, use_ensemble: bool = False):
        """
        Initialize ML trade filter.

        Args:
            params: LightGBM parameters (uses defaults if not provided)
            use_ensemble: If True, use ensemble of models (more robust)
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.use_ensemble = use_ensemble
        self.model = None  # Can be LGBMClassifier or VotingClassifier
        self.lgb_model: Optional[lgb.LGBMClassifier] = None  # Keep reference for feature importance
        self.feature_builder = FeatureBuilder()
        self.feature_names: List[str] = []
        self.training_result: Optional[TrainingResult] = None
        self.ticker: str = ""
        self.train_date: str = ""
        self.features_df: Optional[pd.DataFrame] = None  # Store for insights

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
            TrainingResult with metrics, feature importance, and insights
        """
        self.ticker = ticker
        self.train_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.feature_names = self.feature_builder.get_feature_names()
        self.features_df = features_df.copy()  # Store for insights

        # Extract features and target
        X, y = self.feature_builder.get_feature_matrix(features_df)

        if len(X) < 20:
            raise ValueError(f"Not enough samples for training: {len(X)} (need at least 20)")

        # Use time-based split (last 20% as test) for proper temporal validation
        # This is more realistic for trading - we always predict future from past
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create and train model(s)
        if self.use_ensemble:
            # Create ensemble of models
            lgb_model = lgb.LGBMClassifier(**self.params)
            rf_model = RandomForestClassifier(**self.RF_PARAMS)
            lr_model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )

            # Voting classifier with soft voting (uses probabilities)
            self.model = VotingClassifier(
                estimators=[
                    ('lgb', lgb_model),
                    ('rf', rf_model),
                    ('lr', lr_model)
                ],
                voting='soft',
                weights=[
                    self.ENSEMBLE_WEIGHTS['lightgbm'],
                    self.ENSEMBLE_WEIGHTS['random_forest'],
                    self.ENSEMBLE_WEIGHTS['logistic']
                ]
            )
            self.model.fit(X_train, y_train)

            # Get the fitted LightGBM model from inside the ensemble for feature importance
            self.lgb_model = self.model.named_estimators_['lgb']
            importance = dict(zip(
                self.feature_names,
                self.lgb_model.feature_importances_.tolist()
            ))
            model_type = "ensemble"
        else:
            # Single LightGBM model
            self.lgb_model = lgb.LGBMClassifier(**self.params)
            self.model = self.lgb_model
            self.model.fit(X_train, y_train)

            # Feature importance
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            model_type = "lightgbm"

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

        # Time-series cross-validation (proper for trading data)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        if self.use_ensemble:
            # For ensemble, use a fresh LightGBM for CV (faster)
            cv_model = lgb.LGBMClassifier(**self.params)
        else:
            cv_model = lgb.LGBMClassifier(**self.params)

        cv_scores = cross_val_score(
            cv_model, X, y, cv=tscv, scoring='accuracy'
        ).tolist()

        # Generate actionable insights
        insights = self._generate_insights(features_df, importance)

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
            train_date=self.train_date,
            insights=insights,
            model_type=model_type
        )

        return self.training_result

    def _generate_insights(
        self,
        features_df: pd.DataFrame,
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """
        Generate actionable insights from the training data.

        Analyzes win rates by different dimensions and generates
        human-readable recommendations.
        """
        insights = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Need minimum samples for reliable insights
        if len(features_df) < 30:
            insights.append("Not enough data for reliable insights (need 30+ trades)")
            return insights

        overall_win_rate = features_df['is_winner'].mean()

        # Day of week analysis
        if 'day_of_week' in features_df.columns:
            day_win_rates = features_df.groupby('day_of_week')['is_winner'].agg(['mean', 'count'])
            for day_idx, row in day_win_rates.iterrows():
                if row['count'] >= 10:  # Need enough samples
                    day_name = day_names[int(day_idx)] if int(day_idx) < 5 else f"Day {day_idx}"
                    if row['mean'] < overall_win_rate - 0.10:  # 10% worse than average
                        insights.append(
                            f"Consider avoiding {day_name}s - only {row['mean']:.0%} win rate "
                            f"({int(row['count'])} trades)"
                        )
                    elif row['mean'] > overall_win_rate + 0.10:  # 10% better than average
                        insights.append(
                            f"{day_name}s show strong results - {row['mean']:.0%} win rate "
                            f"({int(row['count'])} trades)"
                        )

        # IB range analysis
        if 'ib_range_percent' in features_df.columns:
            median_ib = features_df['ib_range_percent'].median()
            if median_ib > 0:
                wide_ib = features_df[features_df['ib_range_percent'] > median_ib * 1.5]
                narrow_ib = features_df[features_df['ib_range_percent'] < median_ib * 0.5]

                if len(wide_ib) >= 10:
                    wide_wr = wide_ib['is_winner'].mean()
                    if wide_wr < overall_win_rate - 0.08:
                        insights.append(
                            f"Wide IB days (>{median_ib*1.5:.1f}%) underperform - "
                            f"{wide_wr:.0%} win rate ({len(wide_ib)} trades)"
                        )
                    elif wide_wr > overall_win_rate + 0.08:
                        insights.append(
                            f"Wide IB days (>{median_ib*1.5:.1f}%) perform well - "
                            f"{wide_wr:.0%} win rate ({len(wide_ib)} trades)"
                        )

                if len(narrow_ib) >= 10:
                    narrow_wr = narrow_ib['is_winner'].mean()
                    if narrow_wr < overall_win_rate - 0.08:
                        insights.append(
                            f"Narrow IB days (<{median_ib*0.5:.1f}%) underperform - "
                            f"{narrow_wr:.0%} win rate ({len(narrow_ib)} trades)"
                        )

        # Gap analysis
        if 'gap_percent' in features_df.columns:
            gap_up = features_df[features_df['gap_percent'] > 0.5]
            gap_down = features_df[features_df['gap_percent'] < -0.5]

            if len(gap_up) >= 10:
                gap_up_wr = gap_up['is_winner'].mean()
                if gap_up_wr < overall_win_rate - 0.08:
                    insights.append(
                        f"Gap-up days (>0.5%) show weaker results - "
                        f"{gap_up_wr:.0%} win rate ({len(gap_up)} trades)"
                    )
                elif gap_up_wr > overall_win_rate + 0.08:
                    insights.append(
                        f"Gap-up days (>0.5%) show strong results - "
                        f"{gap_up_wr:.0%} win rate ({len(gap_up)} trades)"
                    )

            if len(gap_down) >= 10:
                gap_down_wr = gap_down['is_winner'].mean()
                if gap_down_wr < overall_win_rate - 0.08:
                    insights.append(
                        f"Gap-down days (<-0.5%) show weaker results - "
                        f"{gap_down_wr:.0%} win rate ({len(gap_down)} trades)"
                    )
                elif gap_down_wr > overall_win_rate + 0.08:
                    insights.append(
                        f"Gap-down days (<-0.5%) show strong results - "
                        f"{gap_down_wr:.0%} win rate ({len(gap_down)} trades)"
                    )

        # Entry hour analysis
        if 'entry_hour' in features_df.columns:
            hour_win_rates = features_df.groupby('entry_hour')['is_winner'].agg(['mean', 'count'])
            for hour, row in hour_win_rates.iterrows():
                if row['count'] >= 10:
                    if row['mean'] < overall_win_rate - 0.12:
                        insights.append(
                            f"Avoid entries at {int(hour)}:00 - only {row['mean']:.0%} win rate "
                            f"({int(row['count'])} trades)"
                        )
                    elif row['mean'] > overall_win_rate + 0.12:
                        insights.append(
                            f"Best entry time: {int(hour)}:00 - {row['mean']:.0%} win rate "
                            f"({int(row['count'])} trades)"
                        )

        # Prior days trend analysis
        if 'prior_days_bullish_count' in features_df.columns:
            bullish_trend = features_df[features_df['prior_days_bullish_count'] >= 2]
            bearish_trend = features_df[features_df['prior_days_bullish_count'] == 0]

            if len(bullish_trend) >= 10:
                bull_wr = bullish_trend['is_winner'].mean()
                if abs(bull_wr - overall_win_rate) > 0.08:
                    trend_word = "better" if bull_wr > overall_win_rate else "worse"
                    insights.append(
                        f"After bullish days (2+ up), results are {trend_word} - "
                        f"{bull_wr:.0%} win rate ({len(bullish_trend)} trades)"
                    )

        # Top feature insight
        if feature_importance:
            top_feature = max(feature_importance, key=feature_importance.get)
            readable_name = top_feature.replace('_', ' ').title()
            insights.append(f"Most predictive feature: {readable_name}")

        # Overall assessment
        if overall_win_rate < 0.45:
            insights.insert(0, f"Warning: Overall win rate is low ({overall_win_rate:.0%}) - consider different parameters")
        elif overall_win_rate > 0.60:
            insights.insert(0, f"Strong base win rate: {overall_win_rate:.0%}")

        return insights

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
        ib_duration_minutes: int = 30,
        # Strategy parameters
        profit_target_percent: float = 1.0,
        stop_loss_type: str = 'opposite_ib',
        trailing_stop_enabled: bool = False,
        break_even_enabled: bool = False
    ) -> float:
        """
        Predict win probability for a single trade.

        Returns:
            Win probability (0.0 to 1.0)
        """
        features = np.array([[
            # Market condition features
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
            distance_from_ib_percent,
            # Strategy parameters
            profit_target_percent,
            1 if stop_loss_type == 'opposite_ib' else 0,
            1 if stop_loss_type == 'fixed_percent' else 0,
            1 if stop_loss_type == 'atr' else 0,
            1 if trailing_stop_enabled else 0,
            1 if break_even_enabled else 0,
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
            'lgb_model': self.lgb_model,
            'feature_names': self.feature_names,
            'params': self.params,
            'ticker': self.ticker,
            'train_date': self.train_date,
            'use_ensemble': self.use_ensemble,
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
            'use_ensemble': self.use_ensemble,
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

        instance = cls(
            params=save_data.get('params'),
            use_ensemble=save_data.get('use_ensemble', False)
        )
        instance.model = save_data['model']
        instance.lgb_model = save_data.get('lgb_model')
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
                train_date=result_dict['train_date'],
                insights=result_dict.get('insights', []),
                model_type=result_dict.get('model_type', 'lightgbm')
            )

        return instance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        if self.lgb_model is not None:
            # Use LightGBM model for feature importance (works for both ensemble and single)
            return dict(zip(
                self.feature_names,
                self.lgb_model.feature_importances_.tolist()
            ))
        elif self.model is not None and hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
        return {}

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
