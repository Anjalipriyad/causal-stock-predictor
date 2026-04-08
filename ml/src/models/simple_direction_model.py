"""
simple_direction_model.py
------------------------
A lightweight, robust logistic fallback used when regime training data
is too small for a full LightGBM ensemble.

Interface mimics the parts of `Ensemble` used by `RegimeAwareEnsemble`:
    - `train_all(df, ticker, causal_features)` -> returns X_test, y_test
    - `predict_historical(df, causal_features)` -> DataFrame with
       `predicted_return` and `actual_return` (if present)

This keeps the rest of the pipeline unchanged while providing a stable
fallback that needs far less data (100-200 rows) to behave sensibly.
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config


class SimpleDirectionModel:
    """Very small logistic-regression direction classifier.

    Predicted return is returned as `probability - 0.5` so that the sign
    matches the ensemble contract (positive = UP).
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg if cfg is not None else _load_config(config_path)
        self.model_name = "simple_dir"
        self._is_fitted = False

        # Lazy import scikit-learn objects to avoid hard dependency during quick imports
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._clf = LogisticRegression(C=0.1, solver="liblinear", random_state=self.cfg["project"]["random_seed"])  # type: ignore
        self._scaler = StandardScaler()

        # Default simple features (will pick intersection with available columns)
        self._candidate_features = [
            "momentum_5d",
            "momentum_20d",
            "india_vix",
            "vix_change_5d",
            "pe_change_1d",
        ]

    # -----------------------------------------------------------------------
    # Training helper used by RegimeAwareEnsemble
    # -----------------------------------------------------------------------

    def train_all(self, df: pd.DataFrame, ticker: str, causal_features: List[str]):
        """Train on the provided DataFrame and return X_test (raw) and y_test.

        Args:
            df: Full feature matrix for the regime
            ticker: regime-specific ticker name (for compatibility)
            causal_features: not used but accepted for API parity

        Returns:
            X_test (raw, unscaled DataFrame), y_test (Series of real returns)
        """
        ticker = ticker.upper()
        n = len(df)
        train_end = int(n * self.cfg["model"]["train_ratio"])
        val_end = int(n * (self.cfg["model"]["train_ratio"] + self.cfg["model"]["val_ratio"]))

        features = [f for f in self._candidate_features if f in df.columns]
        if not features:
            raise ValueError("No simple features available for SimpleDirectionModel")

        X = df[features].fillna(0.0)
        y = df[self.cfg["model"]["target"]]

        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

        # Fit scaler + classifier on train
        self._scaler.fit(X_train)
        X_train_s = pd.DataFrame(self._scaler.transform(X_train), index=X_train.index, columns=X_train.columns)

        # Binary target
        y_train_cls = (y_train >= 0).astype(int)
        self._clf.fit(X_train_s, y_train_cls)
        self._is_fitted = True

        return X_test, y_test

    # -----------------------------------------------------------------------
    # Inference (historical)
    # -----------------------------------------------------------------------

    def predict_historical(self, df: pd.DataFrame, causal_features: Optional[List[str]] = None) -> pd.DataFrame:
        """Predict on a DataFrame and return a tidy results DataFrame.

        Returns DataFrame with at least `predicted_return` and, if present,
        `actual_return`.
        """
        if not self._is_fitted:
            raise RuntimeError("SimpleDirectionModel not fitted.")

        features = [f for f in self._candidate_features if f in df.columns]
        X = df[features].fillna(0.0)
        X_s = pd.DataFrame(self._scaler.transform(X), index=X.index, columns=X.columns)

        proba = self._clf.predict_proba(X_s)[:, 1]
        pred_return = proba - 0.5  # map [0,1] -> [-0.5, +0.5]

        out = pd.DataFrame(index=df.index)
        out["predicted_return"] = pred_return
        out["direction_pred"] = (pred_return >= 0).astype(int)

        target = self.cfg["model"]["target"]
        if target in df.columns:
            out["actual_return"] = df[target].values
            out["direction_actual"] = (out["actual_return"] >= 0).astype(int)

        return out

    # -----------------------------------------------------------------------
    # Live prediction (optional helper)
    # -----------------------------------------------------------------------

    def predict_live(self, live_features: pd.Series, ticker: str, current_price: float):
        """Produce a live PredictionResult-like dict for compatibility.
        Minimal output: predicted_return and confidence-like proxy.
        """
        if not self._is_fitted:
            raise RuntimeError("SimpleDirectionModel not fitted.")

        features = [f for f in self._candidate_features if f in live_features.index]
        X = pd.DataFrame([live_features[features].fillna(0.0)])
        X_s = pd.DataFrame(self._scaler.transform(X), columns=X.columns)
        proba = float(self._clf.predict_proba(X_s)[:, 1][0])
        pred_return = proba - 0.5
        return {"predicted_return": pred_return, "confidence": proba}

    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    def save(self, name: str, models_dir: Optional[str] = None) -> None:
        """Save model artifacts (scaler + classifier) to disk using joblib."""
        import joblib

        root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[3] / self.cfg["saved_models"]["dir"]
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}_{self.model_name}.pkl"
        joblib.dump({"scaler": self._scaler, "clf": self._clf, "cfg": self.cfg}, path)

    @classmethod
    def load(cls, name: str, models_dir: Optional[str] = None):
        """Load a previously saved SimpleDirectionModel instance."""
        import joblib

        # Create a blank instance (cfg will be overwritten by saved one)
        inst = cls()
        root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[3] / inst.cfg["saved_models"]["dir"]
        path = root / f"{name}_{inst.model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Saved SimpleDirectionModel not found: {path}")
        data = joblib.load(path)
        inst._scaler = data["scaler"]
        inst._clf = data["clf"]
        inst.cfg = data.get("cfg", inst.cfg)
        inst._is_fitted = True
        return inst
