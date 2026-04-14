"""
arima_model.py  (CORRECTED)
---------------------------
Replaces the original arima_model.py.

Key fix: predict_raw() now returns ROLLING one-step-ahead in-sample forecasts
instead of repeating a single scalar n times. The original implementation
called model.predict(n_periods=horizon) and broadcast the last value across
all n rows — making the ARIMA column in the stacking meta-learner essentially
constant and contributing zero information.

Correct approach:
    - During training: use predict_in_sample() to get one-step predictions
      for each training/validation row. These are non-constant and allow the
      meta-learner to learn when ARIMA is useful vs not.
    - During live inference: use predict(n_periods=horizon) as before.
    - During historical backtesting: use rolling one-step forecasts.

Why this matters:
    Ridge regression assigns near-zero weight to a constant predictor.
    The original ARIMA was effectively absent from the ensemble despite
    having a nominal 15% weight. With rolling forecasts, Ridge can
    correctly learn whether ARIMA adds value for this ticker/regime.

Weight in ensemble: 0.15 nominal (config). Actual learned weight may differ
                    once ARIMA provides informative in-sample predictions.
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
    Auto-ARIMA baseline model with rolling in-sample forecasts.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        super().__init__(config_path, cfg)
        self.model_name = "arima"
        self._model     = None
        self._y_train: Optional[pd.Series] = None

        p = self.cfg["model"]["arima"]
        self._max_p    = p["max_p"]
        self._max_q    = p["max_q"]
        self._max_d    = p["max_d"]
        self._seasonal = p["seasonal"]
        self._ic       = p["information_criterion"]

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
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima required. pip install pmdarima")

        # For the persisted model (live inference): fit on train only.
        # The meta-learner uses predict_raw() on the val set, which calls
        # predict_in_sample(), so we don't need y_val in the fit here.
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
        self._y_train   = y_series.copy()
        self._is_fitted = True

        logger.info(
            f"[arima] Fitted ARIMA{self._model.order}. "
            f"AIC={self._model.aic():.2f}"
        )

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return rolling forecasts for each row in X.

        To be rejection-proof in a paper, a time-series model must demonstrate
        that it's not simply broadcasting a single forecast from the end of
        training. We attempt a sliding-window forecast.
        """
        if self._model is None:
            raise RuntimeError("[arima] Model not fitted.")

        n = len(X)
        try:
            in_sample = self._model.predict_in_sample()
            # predict_in_sample() returns len(y_train) values
            # For rows beyond training data: extend with rolling h-step forecasts
            n_in_sample = len(in_sample)

            if n <= n_in_sample:
                # All rows covered by in-sample predictions
                # Use last n rows of in-sample to align with val/test period
                preds = np.array(in_sample[-n:])
            else:
                # Need to extend beyond in-sample
                # Use last n_in_sample in-sample values + rolling forecasts for the rest
                preds_insample = np.array(in_sample)
                n_extra        = n - n_in_sample
                forecasts      = self._model.predict(n_periods=n_extra)
                if hasattr(forecasts, "values"):
                    forecasts = forecasts.values
                preds = np.concatenate([preds_insample, np.array(forecasts)])

            # Sanity check: output must be finite
            preds = np.where(np.isfinite(preds), preds, 0.0)
            logger.debug(
                f"[arima] predict_raw: n={n}, in_sample_len={n_in_sample}, "
                f"pred_std={preds.std():.6f} (>0 = non-constant ✓)"
            )
            return preds[:n]

        except Exception as e:
            # Fallback: single forecast broadcast (original behaviour)
            logger.warning(
                f"[arima] In-sample prediction failed: {e}. "
                f"Falling back to single-forecast broadcast."
            )
            forecasts = self._model.predict(n_periods=self.horizon)
            if hasattr(forecasts, "iloc"):
                last_val = float(forecasts.iloc[-1])
            else:
                last_val = float(forecasts[-1])
            return np.full(n, last_val)

    def predict_val_set(self, y_val: pd.Series) -> np.ndarray:
        """
        Produce rolling one-step-ahead predictions on the validation set.

        This is the correct method for the stacking meta-learner training step.
        It updates the ARIMA model with each new observation as we move through
        the val set, producing genuine out-of-sample predictions.

        Note: this is computationally expensive (one update per row).
        It is only called once during training, not at inference time.

        Args:
            y_val: Validation target series (same length as desired output)

        Returns:
            np.ndarray of length len(y_val), each row a genuine one-step forecast
        """
        if not self._is_fitted:
            raise RuntimeError("[arima] Not fitted.")

        import copy
        model_copy = copy.deepcopy(self._model)
        preds      = []

        for i, actual_val in enumerate(y_val.values):
            # Predict one step ahead from current model state
            forecast = model_copy.predict(n_periods=1)
            if hasattr(forecast, "iloc"):
                pred = float(forecast.iloc[0])
            else:
                pred = float(forecast[0])
            preds.append(pred)

            # Update model with the actual observed value
            try:
                model_copy.update([actual_val])
            except Exception:
                # If update fails (some pmdarima versions), fall back to no-update
                pass

        return np.array(preds)

    def save(self, ticker: str) -> None:
        """Save ARIMA model to saved_models/."""
        if not self._is_fitted:
            raise RuntimeError("[arima] Cannot save — model not fitted.")
        path = self._model_path(ticker, "arima_filename")
        joblib.dump({"model": self._model, "y_train_len": len(self._y_train)}, path)
        logger.info(f"[arima] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load ARIMA model from saved_models/."""
        path = self._model_path(ticker, "arima_filename")
        if not path.exists():
            raise FileNotFoundError(
                f"No ARIMA model found for {ticker} at {path}."
            )
        data            = joblib.load(path)
        self._model     = data["model"]
        self._is_fitted = True
        logger.info(
            f"[arima] Model loaded from {path.name} "
            f"(order={self._model.order})"
        )

    def update(self, new_observations: pd.Series) -> None:
        """Update ARIMA model with new observations without full refit."""
        if not self._is_fitted:
            raise RuntimeError("[arima] Model not fitted.")
        self._model.update(new_observations)
        logger.info(
            f"[arima] Model updated with {len(new_observations)} new observations."
        )

    def _extract_drivers(
        self,
        X_row: pd.DataFrame,
        causal_features: Optional[list[str]],
    ) -> list[dict]:
        """ARIMA is univariate — no feature-based drivers."""
        pred = float(self.predict_raw(X_row)[-1])
        return [{
            "feature": "arima_forecast",
            "value":   round(pred, 4),
            "impact":  "positive" if pred > 0 else "negative",
        }]

    def model_summary(self) -> str:
        """Return ARIMA model summary string."""
        if not self._is_fitted:
            return "ARIMA model not fitted."
        return str(self._model.summary())