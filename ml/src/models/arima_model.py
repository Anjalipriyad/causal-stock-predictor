"""
arima_model.py
--------------
ARIMA classical baseline model — lowest weight in the ensemble.
Uses pmdarima's auto_arima to automatically select optimal (p, d, q) order.

Unlike LightGBM and XGBoost, ARIMA is trained on the TARGET time series
only (log returns). It does not use causal features — it is a univariate
baseline. This is intentional:
    - It represents the "naive" time series baseline
    - Its inclusion in the ensemble adds stability during low-signal periods
    - The paper compares causal feature models vs pure ARIMA

Weight in ensemble: 0.15 (from config)

Usage:
    from ml.src.models.arima_model import ARIMAModel
    model = ARIMAModel()
    model.fit(X_train, y_train)       # X_train ignored — univariate
    result = model.predict(X_live, ticker="AAPL", current_price=213.40)
    model.save("AAPL")
    model.load("AAPL")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from ml.src.models.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """
    Auto-ARIMA baseline model.
    Predicts 5-day forward log return from the return time series alone.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.model_name = "arima"
        self._model     = None
        self._y_train: Optional[pd.Series] = None

        p = self.cfg["model"]["arima"]
        self._max_p  = p["max_p"]
        self._max_q  = p["max_q"]
        self._max_d  = p["max_d"]
        self._seasonal = p["seasonal"]
        self._ic     = p["information_criterion"]

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
        """
        Fit auto-ARIMA on the target return series.
        X_train is accepted for interface compatibility but ignored.
        If X_val/y_val provided, training series is extended to include val.
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima required. pip install pmdarima")

        # ARIMA uses full y series (train + val) for more data
        if y_val is not None:
            y_series = pd.concat([y_train, y_val])
        else:
            y_series = y_train

        logger.info(
            f"[arima] Fitting auto-ARIMA on {len(y_series)} observations "
            f"(max_p={self._max_p}, max_q={self._max_q}, ic={self._ic}) ..."
        )

        self._model = auto_arima(
            y_series,
            max_p=self._max_p,
            max_q=self._max_q,
            max_d=self._max_d,
            seasonal=self._seasonal,
            information_criterion=self._ic,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            random_state=self.cfg["project"]["random_seed"],
        )
        self._y_train    = y_series
        self._is_fitted  = True

        logger.info(
            f"[arima] Fitted ARIMA{self._model.order}. "
            f"AIC={self._model.aic():.2f}"
        )

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return n-step ARIMA forecasts.
        X is used only to determine how many predictions to return (len(X)).
        Each call re-forecasts from the end of the training series.
        """
        if self._model is None:
            raise RuntimeError("[arima] Model not fitted.")

        n = len(X)
        # Forecast horizon_days ahead, return the last n predictions
        forecasts = self._model.predict(n_periods=self.horizon)

        # pmdarima may return a pandas Series with DatetimeIndex — use iloc
        if hasattr(forecasts, "iloc"):
            last_val = float(forecasts.iloc[-1])
        else:
            last_val = float(forecasts[-1])

        return np.full(n, last_val)

    def save(self, ticker: str) -> None:
        """Save ARIMA model to saved_models/."""
        if not self._is_fitted:
            raise RuntimeError("[arima] Cannot save — model not fitted.")
        path = self._model_path(ticker, "arima_filename")
        joblib.dump({"model": self._model}, path)
        logger.info(f"[arima] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load ARIMA model from saved_models/."""
        path = self._model_path(ticker, "arima_filename")
        if not path.exists():
            raise FileNotFoundError(
                f"No ARIMA model found for {ticker} at {path}."
            )
        data         = joblib.load(path)
        self._model  = data["model"]
        self._is_fitted = True
        logger.info(
            f"[arima] Model loaded from {path.name} "
            f"(order={self._model.order})"
        )

    # -----------------------------------------------------------------------
    # Update (online learning)
    # -----------------------------------------------------------------------

    def update(self, new_observations: pd.Series) -> None:
        """
        Update ARIMA model with new observations without full refit.
        Useful for keeping the model current without expensive retraining.
        """
        if not self._is_fitted:
            raise RuntimeError("[arima] Model not fitted.")
        self._model.update(new_observations)
        logger.info(
            f"[arima] Model updated with {len(new_observations)} new observations."
        )

    # -----------------------------------------------------------------------
    # ARIMA doesn't have causal features — return empty drivers
    # -----------------------------------------------------------------------

    def _extract_drivers(
        self,
        X_row: pd.DataFrame,
        causal_features: Optional[list[str]],
    ) -> list[dict]:
        """ARIMA is univariate — no feature-based drivers."""
        return [
            {
                "feature": "arima_forecast",
                "value":   round(float(self.predict_raw(X_row)[-1]), 4),
                "impact":  "positive" if self.predict_raw(X_row)[-1] > 0 else "negative",
            }
        ]

    def model_summary(self) -> str:
        """Return ARIMA model summary string."""
        if not self._is_fitted:
            return "ARIMA model not fitted."
        return str(self._model.summary())