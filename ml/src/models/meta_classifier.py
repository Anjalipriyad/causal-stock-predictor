"""
meta_classifier.py
------------------
Train a small logistic-regression classifier on base-model predictions
(presented as features) to directly predict direction. This complements
the regressor meta-learner by providing a classification-based routing
option and enables per-model probability thresholds.

API:
    MetaClassifier(cfg)
    fit(preds_dict_or_array, y_true)
    predict_proba(preds_dict_or_array) -> np.ndarray (shape [n_samples,])
    predict_raw(...) -> mapped pseudo-return (prob-0.5)*scale
    save(ticker, models_dir)
    load(ticker, models_dir)

The classifier expects either a 2D numpy array of features (n x k) or a
`dict` mapping model names ('lgbm','xgb','arima','lstm') -> 1D arrays.
"""

from typing import Optional, Union, Dict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config


class MetaClassifier:
    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg if cfg is not None else _load_config(config_path)
        self.model_name = "meta_classifier"
        self._is_fitted = False

        # Lazy import to avoid heavy deps on import
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(C=1.0, solver="liblinear", random_state=self.cfg["project"]["random_seed"])  # type: ignore
        self._feature_order = None

    def _to_matrix(self, preds: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]) -> np.ndarray:
        """Normalize input into a 2D numpy array (n_samples x n_features)."""
        if isinstance(preds, np.ndarray):
            if preds.ndim == 1:
                return preds.reshape(-1, 1)
            return preds
        if isinstance(preds, pd.DataFrame):
            return preds.values
        if isinstance(preds, dict):
            # Preserve deterministic column order
            keys = sorted(preds.keys())
            self._feature_order = keys
            return np.vstack([np.asarray(preds[k]) for k in keys]).T
        raise ValueError("Unsupported preds type for MetaClassifier")

    def fit(self, preds: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame], y_true: np.ndarray) -> None:
        X = self._to_matrix(preds)
        y_cls = (np.asarray(y_true) >= 0).astype(int)
        self._clf.fit(X, y_cls)
        self._is_fitted = True

    def predict_proba(self, preds: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("MetaClassifier not fitted")
        X = self._to_matrix(preds)
        return self._clf.predict_proba(X)[:, 1]

    def predict_raw(self, preds: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]) -> np.ndarray:
        """Return pseudo-return in same contract as other models: (p - 0.5) * scale"""
        p = self.predict_proba(preds)
        scale = float(self.cfg["model"].get("classifier_return_scale", 0.02))
        return (p - 0.5) * scale

    def save(self, ticker: str, models_dir: Optional[str] = None) -> None:
        root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[3] / self.cfg["saved_models"]["dir"]
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"meta_clf_{ticker}.pkl"
        joblib.dump({"clf": self._clf, "feature_order": self._feature_order, "cfg": self.cfg}, path)

    @classmethod
    def load(cls, ticker: str, models_dir: Optional[str] = None):
        inst = cls()
        root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[3] / inst.cfg["saved_models"]["dir"]
        path = root / f"meta_clf_{ticker}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"MetaClassifier not found: {path}")
        data = joblib.load(path)
        inst._clf = data["clf"]
        inst._feature_order = data.get("feature_order")
        inst.cfg = data.get("cfg", inst.cfg)
        inst._is_fitted = True
        return inst
