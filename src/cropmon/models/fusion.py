"""Fused spectral + micro-climate risk model.

A gradient-boosted classifier is used deliberately: with tabular fused features
and a few hundred zones it trains in under a second, is deterministic under a
fixed seed, and exposes permutation importance for agronomists. A hybrid
CNN-LSTM remains available in :mod:`cropmon.models.deep` for raw cubes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cropmon.data.dataset import FEATURE_COLUMNS, RISK_CLASS_NAMES

MODEL_VERSION = "1.0.0"


@dataclass
class RiskModel:
    """Wrapper that keeps feature order, defaults, and metadata with the model."""

    seed: int = 42
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    feature_defaults: dict[str, float] = field(default_factory=dict)
    version: str = MODEL_VERSION
    estimator: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RiskModel:
        frame = X[self.feature_names]
        self.feature_defaults = {name: float(frame[name].median()) for name in self.feature_names}
        self.estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        )
        self.estimator.fit(frame, y)
        return self

    def _check_fitted(self) -> HistGradientBoostingClassifier:
        if self.estimator is None:
            raise RuntimeError("RiskModel is not fitted; call fit() or load() first")
        return self.estimator

    def prepare(self, records: pd.DataFrame | list[dict]) -> pd.DataFrame:
        """Align arbitrary records to the training schema, filling gaps."""
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        for name in self.feature_names:
            if name not in frame.columns:
                frame[name] = self.feature_defaults.get(name, 0.0)
        return frame[self.feature_names].astype(float)

    def predict_proba(self, records: pd.DataFrame | list[dict]) -> np.ndarray:
        return self._check_fitted().predict_proba(self.prepare(records))

    def predict(self, records: pd.DataFrame | list[dict]) -> np.ndarray:
        return self._check_fitted().predict(self.prepare(records))

    def expected_risk(self, records: pd.DataFrame | list[dict]) -> np.ndarray:
        """Probability-weighted severity on a 0-1 scale."""
        probabilities = self.predict_proba(records)
        weights = np.linspace(0.0, 1.0, probabilities.shape[1])
        return probabilities @ weights

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> RiskModel:
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"unexpected artifact type: {type(model)!r}")
        return model

    def metadata(self) -> dict:
        return {
            "version": self.version,
            "algorithm": "HistGradientBoostingClassifier",
            "classes": {str(k): v for k, v in RISK_CLASS_NAMES.items()},
            "feature_names": list(self.feature_names),
            "feature_defaults": self.feature_defaults,
        }


def train_risk_model(
    dataset: pd.DataFrame, seed: int = 42, test_size: float = 0.3
) -> tuple[RiskModel, dict, pd.DataFrame, pd.Series]:
    """Fit on a stratified split and return the model, metrics, and holdout."""
    X = dataset[FEATURE_COLUMNS]
    y = dataset["risk_class"].astype(int)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = RiskModel(seed=seed).fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["class_support"] = {RISK_CLASS_NAMES[k]: int(v) for k, v in y.value_counts().sort_index().items()}
    metrics["data_source"] = "synthetic simulation (physics-inspired), not field measurements"
    return model, metrics, X_test, y_test


def evaluate_model(model: RiskModel, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    labels = np.arange(probabilities.shape[1])

    metrics: dict = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
    }
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y, probabilities, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        metrics["roc_auc_ovr"] = float("nan")

    importance = permutation_importance(
        model._check_fitted(), X[model.feature_names], y, n_repeats=5, random_state=0, scoring="accuracy"
    )
    order = np.argsort(importance.importances_mean)[::-1]
    metrics["feature_importance"] = [
        {"feature": model.feature_names[i], "importance": float(importance.importances_mean[i])}
        for i in order
    ]
    return metrics
