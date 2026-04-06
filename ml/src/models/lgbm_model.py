"""
lgbm_model.py
-------------
LightGBM forecasting model — primary model in the ensemble.
Trained on causal features selected by PCMCI + Granger.

Weight in ensemble: 0.50 (from config)

Usage:
    from ml.src.models.lgbm_model import LGBMModel
    model = LGBMModel()
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
import lightgbm as lgb

from ml.src.models.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """
    LightGBM gradient boosting model for 5-day return prediction.
    Primary model — highest weight in the ensemble.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        super().__init__(config_path, cfg)
        self.model_name = "lgbm"
        self._model: Optional[lgb.Booster] = None
        self._feature_names: list[str] = []

        # Load hyperparams from config
        p = self.cfg["model"]["lgbm"]
        self._params = {
            "objective":        "regression",
            "metric":           "rmse",
            "learning_rate":    p["learning_rate"],
            "max_depth":        p["max_depth"],
            "num_leaves":       p["num_leaves"],
            "min_child_samples": p["min_child_samples"],
            "subsample":        p["subsample"],
            "subsample_freq":   p["subsample_freq"],
            "colsample_bytree": p["colsample_bytree"],
            "reg_alpha":        p["reg_alpha"],
            "reg_lambda":       p["reg_lambda"],
            "verbose":          p["verbose"],
            "random_state":     self.cfg["project"]["random_seed"],
        }
        self._n_estimators       = p["n_estimators"]
        self._early_stopping     = p["early_stopping_rounds"]

    # -----------------------------------------------------------------------
    # Abstract implementations
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Train LightGBM model with optional early stopping on validation set."""
        logger.info(
            f"[lgbm] Training on {len(X_train)} rows, "
            f"{len(X_train.columns)} features ..."
        )

        self._feature_names = list(X_train.columns)

        dtrain = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.log_evaluation(period=50),
        ]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self._early_stopping, verbose=False)
            )
            self._model = lgb.train(
                params=self._params,
                train_set=dtrain,
                num_boost_round=self._n_estimators,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            logger.info(
                f"[lgbm] Best iteration: {self._model.best_iteration}"
            )
        else:
            self._model = lgb.train(
                params=self._params,
                train_set=dtrain,
                num_boost_round=self._n_estimators,
                callbacks=callbacks,
            )

        self._is_fitted = True
        logger.info("[lgbm] Training complete.")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw log return predictions as numpy array."""
        if self._model is None:
            raise RuntimeError("[lgbm] Model not fitted.")
        return self._model.predict(
            X[self._feature_names],
            num_iteration=self._model.best_iteration
            if hasattr(self._model, "best_iteration") else -1,
        )

    def save(self, ticker: str) -> None:
        """Save model + feature names to saved_models/."""
        if not self._is_fitted:
            raise RuntimeError("[lgbm] Cannot save — model not fitted.")
        path = self._model_path(ticker, "lgbm_filename")
        joblib.dump(
            {"model": self._model, "feature_names": self._feature_names},
            path,
        )
        logger.info(f"[lgbm] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load model + feature names from saved_models/."""
        path = self._model_path(ticker, "lgbm_filename")
        if not path.exists():
            raise FileNotFoundError(
                f"No LightGBM model found for {ticker} at {path}. "
                "Run fit() + save() first."
            )
        data = joblib.load(path)
        self._model        = data["model"]
        self._feature_names = data["feature_names"]
        self._is_fitted    = True
        logger.info(f"[lgbm] Model loaded from {path.name}")

    # -----------------------------------------------------------------------
    # SHAP feature importance
    # -----------------------------------------------------------------------

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for interpretability.
        Returns DataFrame with same shape as X — one SHAP value per feature per row.
        Used for causal driver extraction in predict().
        """
        try:
            import shap
            explainer   = shap.TreeExplainer(self._model)
            shap_vals   = explainer.shap_values(X[self._feature_names])
            return pd.DataFrame(shap_vals, index=X.index, columns=self._feature_names)
        except Exception as e:
            logger.warning(f"[lgbm] SHAP computation failed: {e}")
            return pd.DataFrame()

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        Return feature importance from the trained model.
        importance_type: "gain" | "split"
        """
        if not self._is_fitted:
            raise RuntimeError("[lgbm] Model not fitted.")
        importance = self._model.feature_importance(importance_type=importance_type)
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)

    # -----------------------------------------------------------------------
    # Override _extract_drivers to use SHAP
    # -----------------------------------------------------------------------

    def _extract_drivers(
        self,
        X_row: pd.DataFrame,
        causal_features: Optional[list[str]],
    ) -> list[dict]:
        """Use SHAP values for driver extraction if available."""
        shap_df = self.shap_values(X_row)

        if shap_df.empty:
            # Fall back to base class implementation
            return super()._extract_drivers(X_row, causal_features)

        row = shap_df.iloc[0]
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