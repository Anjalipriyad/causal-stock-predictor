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

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        super().__init__(config_path, cfg)
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

    def predict_raw(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> np.ndarray:
        """
        Produce rolling/recursive ARIMA forecasts for each row in X.

        If `y_true` is provided (e.g., validation or historical test series),
        the function will simulate online forecasting by updating the ARIMA
        state with the observed true value after each prediction. If
        `y_true` is not provided (live inference), the method will perform
        recursive forecasting updating the local model with its own forecasts
        so each step moves forward (no leakage).
        """
        if self._model is None:
            raise RuntimeError("[arima] Model not fitted.")

        import copy

        n = len(X)
        # Work on a copy of the fitted model so we don't mutate the persisted one
        try:
            model = copy.deepcopy(self._model)
        except Exception:
            # Fallback: operate on the fitted model (best-effort)
            model = self._model

        preds = []

        # Align y_true if provided
        if y_true is not None:
            # Accept Series or array-like; use iloc for positional access
            y_series = pd.Series(y_true).reset_index(drop=True)
        else:
            y_series = None
        # Special-case: if auto_arima selected a trivial model (0,0,0)
        # then fallback to a simple persistence-style online forecast so
        # historical rolling predictions can vary when `y_true` is supplied.
        try:
            order = getattr(model, "order", None)
        except Exception:
            order = None

        if order == (0, 0, 0):
            last_obs = None
            if hasattr(self, "_y_train") and self._y_train is not None and len(self._y_train) > 0:
                try:
                    last_obs = float(self._y_train.iloc[-1])
                except Exception:
                    last_obs = 0.0
            else:
                last_obs = 0.0

            for i in range(n):
                pred = float(last_obs)
                preds.append(pred)
                # update with observed true value if available, else use prediction
                if y_series is not None and i < len(y_series):
                    try:
                        last_obs = float(y_series.iloc[i])
                    except Exception:
                        last_obs = pred
                else:
                    last_obs = pred

            return np.array(preds)

        for i in range(n):
            # Forecast horizon steps ahead and take the horizon-th element
            forecasts = model.predict(n_periods=self.horizon)
            if hasattr(forecasts, "iloc"):
                pred = float(forecasts.iloc[-1])
            else:
                pred = float(forecasts[-1])

            preds.append(pred)

            # Update the local model with the observed true value if available,
            # otherwise update with the model's own forecast so the window moves.
            try:
                if y_series is not None and i < len(y_series):
                    obs = float(y_series.iloc[i])
                    model.update(np.array([obs]))
                else:
                    # No true observation available (live), update with prediction
                    model.update(np.array([pred]))
            except Exception:
                # If update fails, continue — predictions will fall back to
                # repeated multi-step forecasts from the current model state.
                continue

        return np.array(preds)

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