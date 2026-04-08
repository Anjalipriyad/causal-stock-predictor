"""
xgb_model.py
------------
XGBoost forecasting model — comparison model in the ensemble.

Weight in ensemble: 0.35 (from config)

Usage:
    from ml.src.models.xgb_model import XGBModel
    model = XGBModel()
    model.fit(X_train, y_train, X_val, y_val)
    result = model.predict(X_live, ticker="AAPL", current_price=213.40)
    model.save("AAPL")
    model.load("AAPL")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from ml.src.models.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class XGBModel(BaseModel):
    """
    XGBoost gradient boosting model for 5-day return prediction.
    Comparison model — second highest weight in the ensemble.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        super().__init__(config_path, cfg)
        self.model_name = "xgb"
        self._model: Optional[xgb.Booster] = None
        self._feature_names: list[str] = []

        p = self.cfg["model"]["xgb"]
        self._params = {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "learning_rate":    p["learning_rate"],
            "max_depth":        p["max_depth"],
            "subsample":        p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
            "reg_alpha":        p["reg_alpha"],
            "reg_lambda":       p["reg_lambda"],
            "verbosity":        p["verbosity"],
            "seed":             self.cfg["project"]["random_seed"],
        }
        self._n_estimators   = p["n_estimators"]
        self._early_stopping = p["early_stopping_rounds"]

    # -----------------------------------------------------------------------
    # Abstract implementations
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Train XGBoost model with optional early stopping."""
        logger.info(
            f"[xgb] Training on {len(X_train)} rows, "
            f"{len(X_train.columns)} features ..."
        )

        self._feature_names = list(X_train.columns)

        # Support optional sample weights for crash upweighting
        if sample_weight is not None:
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self._feature_names, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self._feature_names)

        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self._feature_names)
            evals = [(dval, "val")]

        callbacks = []
        if evals:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self._early_stopping,
                    metric_name="rmse",
                    save_best=True,
                )
            )

        self._model = xgb.train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=self._n_estimators,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=50 if evals else False,
        )

        self._is_fitted = True
        logger.info(
            f"[xgb] Training complete. "
            f"Best iteration: {self._model.best_iteration}"
        )

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw log return predictions as numpy array."""
        if self._model is None:
            raise RuntimeError("[xgb] Model not fitted.")
        dmat = xgb.DMatrix(
            X[self._feature_names],
            feature_names=self._feature_names,
        )
        return self._model.predict(
            dmat,
            iteration_range=(0, self._model.best_iteration + 1)
            if hasattr(self._model, "best_iteration") else (0, 0),
        )

    def save(self, ticker: str) -> None:
        """Save model to saved_models/."""
        if not self._is_fitted:
            raise RuntimeError("[xgb] Cannot save — model not fitted.")
        path = self._model_path(ticker, "xgb_filename")
        joblib.dump(
            {"model": self._model, "feature_names": self._feature_names},
            path,
        )
        logger.info(f"[xgb] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load model from saved_models/."""
        path = self._model_path(ticker, "xgb_filename")
        if not path.exists():
            raise FileNotFoundError(
                f"No XGBoost model found for {ticker} at {path}."
            )
        data = joblib.load(path)
        self._model         = data["model"]
        self._feature_names = data["feature_names"]
        self._is_fitted     = True
        logger.info(f"[xgb] Model loaded from {path.name}")

    # -----------------------------------------------------------------------
    # Feature importance
    # -----------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        Return feature importance.
        importance_type: "gain" | "weight" | "cover"
        """
        if not self._is_fitted:
            raise RuntimeError("[xgb] Model not fitted.")
        scores = self._model.get_score(importance_type=importance_type)
        return (
            pd.Series(scores)
            .reindex(self._feature_names)
            .fillna(0)
            .sort_values(ascending=False)
        )

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values for XGBoost model."""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            dmat      = xgb.DMatrix(
                X[self._feature_names],
                feature_names=self._feature_names,
            )
            shap_vals = explainer.shap_values(X[self._feature_names])
            return pd.DataFrame(
                shap_vals, index=X.index, columns=self._feature_names
            )
        except Exception as e:
            logger.warning(f"[xgb] SHAP failed: {e}")
            return pd.DataFrame()

    def _extract_drivers(
        self,
        X_row: pd.DataFrame,
        causal_features: Optional[list[str]],
    ) -> list[dict]:
        """Use SHAP values for driver extraction if available."""
        shap_df = self.shap_values(X_row)
        if shap_df.empty:
            return super()._extract_drivers(X_row, causal_features)

        row     = shap_df.iloc[0]
        drivers = []
        for feat, shap_val in row.items():
            drivers.append({
                "feature": feat,
                "value":   round(float(X_row[feat].iloc[0]), 4),
                "shap":    round(float(shap_val), 4),
                "impact":  "positive" if shap_val > 0 else "negative",
            })
        drivers.sort(key=lambda d: abs(d["shap"]), reverse=True)
        return drivers[:5]