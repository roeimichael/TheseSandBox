from __future__ import annotations
from typing import Dict, Tuple
from collections import OrderedDict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_logreg(cfg: dict) -> LogisticRegression:
    """Build Logistic Regression model with configuration."""
    mcfg = cfg.get("logistic_regression", {})
    return LogisticRegression(
        C=mcfg.get("C", 1.0),
        penalty=mcfg.get("penalty", "l2"),
        max_iter=mcfg.get("max_iter", 200),
        class_weight=mcfg.get("class_weight", None),
        solver="lbfgs",
        random_state=42
    )


def build_mlp(cfg: dict) -> MLPClassifier:
    """Build MLP model with configuration."""
    mcfg = cfg.get("mlp", {})
    return MLPClassifier(
        hidden_layer_sizes=mcfg.get("hidden_layer_sizes", (100, 50)),
        activation=mcfg.get("activation", "relu"),
        solver=mcfg.get("solver", "adam"),
        alpha=mcfg.get("alpha", 0.0001),
        learning_rate=mcfg.get("learning_rate", "adaptive"),
    max_iter=mcfg.get("max_iter", 200),
    random_state=mcfg.get("random_state", 42),
    # Training control knobs (optional; defaults mirror sklearn)
    early_stopping=mcfg.get("early_stopping", False),
    n_iter_no_change=mcfg.get("n_iter_no_change", 10),
    validation_fraction=mcfg.get("validation_fraction", 0.1),
    tol=mcfg.get("tol", 1e-4),
    learning_rate_init=mcfg.get("learning_rate_init", 0.001),
    )


def get_models(cfg: dict) -> Dict[str, object]:
    models = OrderedDict()

    # Logistic Regression (well-calibrated baseline)
    if cfg.get("logistic_regression", {}).get("enabled", True):
        models["logreg"] = build_logreg(cfg)

    # MLP (neural network for comparison)
    if cfg.get("mlp", {}).get("enabled", True):
        models["mlp"] = build_mlp(cfg)

    return models


def fit_and_predict(
        models: Dict[str, object],
        X_train, y_train, X_val, y_val, X_test
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      fitted_models
      val_proba: dict[name] -> (n_val, n_classes)
      test_proba: dict[name] -> (n_test, n_classes)
      val_pred: dict[name] -> (n_val,)
    """
    fitted = {}
    val_proba, test_proba, val_pred = {}, {}, {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
        # probabilities (ensure has predict_proba)
        if hasattr(clf, "predict_proba"):
            val_proba[name] = clf.predict_proba(X_val)
            test_proba[name] = clf.predict_proba(X_test)
        else:
            # fallback: decision_function → sigmoid-ish (binary)
            def _sigmoid(x: np.ndarray) -> np.ndarray:
                return 1.0 / (1.0 + np.exp(-x))
            dv_val = clf.decision_function(X_val)
            dv_test = clf.decision_function(X_test)
            # binary only here:
            val_proba[name] = np.vstack([1 - _sigmoid(dv_val), _sigmoid(dv_val)]).T
            test_proba[name] = np.vstack([1 - _sigmoid(dv_test), _sigmoid(dv_test)]).T

        val_pred[name] = np.argmax(val_proba[name], axis=1)
    return fitted, val_proba, test_proba, val_pred
