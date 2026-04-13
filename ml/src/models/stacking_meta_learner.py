"""
stacking_meta_learner.py
------------------------
Level-1 meta-learner for the stacking ensemble.

Replaces the fixed hardcoded weights (lgbm=0.50, xgb=0.35, arima=0.15)
with a Ridge Regression that LEARNS the optimal combination of base model
predictions from the validation set.

Why this is better than fixed weights:
    - The optimal blend varies by ticker, regime, and market condition
    - Some regimes (e.g. rate_hike) favour LightGBM more; others favour ARIMA
    - Ridge regularization prevents the meta-learner from overfitting to val set noise
    - Falls back to config weights gracefully if not yet trained

Why Ridge not plain LinearRegression:
    - Base model predictions are highly correlated (all trained on same features)
    - High correlation → unstable coefficient estimates without regularization
    - Ridge shrinks the weights toward equal weighting, which is the safe default

Training protocol (CRITICAL — prevents leakage):
    - Meta-learner MUST be trained on VALIDATION set predictions, not train set
    - If trained on train set predictions: base models have memorized the data
      → their predictions are near-perfect → meta-learner learns nonsense weights
    - Validation set predictions come from models that haven't seen that data

Usage:
    from ml.src.models.stacking_meta_learner import StackingMetaLearner

    meta = StackingMetaLearner()

    # Train on validation-set predictions from base models
    meta.fit(
        lgbm_val_preds  = lgbm.predict_raw(X_val_scaled),
        xgb_val_preds   = xgb.predict_raw(X_val_scaled),
        arima_val_preds = arima.predict_raw(X_val),
        y_val           = y_val.values,
    )

    # Blend at inference time
    final_pred = meta.predict(pred_lgbm, pred_xgb, pred_arima)

    # Save / load
    meta.save("AAPL", models_dir)
    meta.load("AAPL", models_dir)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class StackingMetaLearner:
    """
    Ridge Regression meta-learner for stacking ensemble.

    Input:  predictions from LightGBM, XGBoost, ARIMA (and optionally LSTM)
    Output: optimally weighted blend of those predictions

    Graceful fallback:
        If not fitted (first run, or load fails), falls back to the
        config-specified fixed weights: lgbm=0.50, xgb=0.35, arima=0.15.
        This means the ensemble.py changes are backward-compatible — existing
        saved models continue to work without retraining.
    """

    # Fallback weights used when meta-learner is not fitted
    DEFAULT_WEIGHTS = {"lgbm": 0.50, "xgb": 0.35, "arima": 0.15}

    def __init__(
        self,
        alpha: float = 1.0,
        default_weights: Optional[dict] = None,
    ):
        """
        Args:
            alpha:           Ridge regularization strength. Higher = more shrinkage
                             toward equal weights. Default 1.0 works well in practice.
            default_weights: Override the fallback weights (used before fitting).
                             Must be a dict with keys: lgbm, xgb, arima.
                             If None, uses DEFAULT_WEIGHTS above.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.alpha          = alpha
        self._meta_model    = Ridge(alpha=alpha, fit_intercept=True)
        self._scaler        = StandardScaler()
        self._is_fitted     = False
        self._n_base_models = 3   # lgbm, xgb, arima (4 if LSTM added)
        self._model_names   = ["lgbm", "xgb", "arima"]

        self._default_weights = default_weights or self.DEFAULT_WEIGHTS
        self._coef_se = None

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(
        self,
        lgbm_preds:  np.ndarray,
        xgb_preds:   np.ndarray,
        arima_preds: np.ndarray,
        y_true:      np.ndarray,
        lstm_preds:  Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the meta-learner on base model validation-set predictions.

        MUST be called with VALIDATION SET predictions — never training set.
        See module docstring for explanation of why.

        Args:
            lgbm_preds:   LightGBM predictions on validation set
            xgb_preds:    XGBoost predictions on validation set
            arima_preds:  ARIMA predictions on validation set
            y_true:       Actual returns for validation set
            lstm_preds:   Optional LSTM predictions (adds a 4th base learner)
        """
        preds_list   = [lgbm_preds, xgb_preds, arima_preds]
        model_names  = ["lgbm", "xgb", "arima"]

        if lstm_preds is not None:
            preds_list.append(lstm_preds)
            model_names.append("lstm")

        self._model_names   = model_names
        self._n_base_models = len(preds_list)

        # Stack into (n_samples, n_models) matrix
        X_meta = np.column_stack(preds_list)

        # Scale inputs — Ridge is sensitive to input scale
        X_meta_scaled = self._scaler.fit_transform(X_meta)

        self._meta_model.fit(X_meta_scaled, y_true)
        self._is_fitted = True

        # Log learned weights for interpretability and paper reporting
        coefs = self._meta_model.coef_
        bias  = self._meta_model.intercept_
        weight_str = " | ".join(
            f"{name}={coef:.3f}" for name, coef in zip(model_names, coefs)
        )
        logger.info(
            f"[meta_learner] Learned weights — {weight_str} | bias={bias:.4f}"
        )

        # Compute validation set R² for quality check
        y_hat  = self._meta_model.predict(X_meta_scaled)
        ss_res = np.sum((y_true - y_hat) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        logger.info(f"[meta_learner] Validation R² = {r2:.4f}")

        # ------------------------------------------------------------------
        # Bootstrap standard errors for coefficients (diagnostic)
        # Useful to report variance of learned weights — if large, the
        # meta-learner is unstable on small validation sets and fixed
        # fallback weights may be preferable.
        try:
            from sklearn.linear_model import Ridge

            n_samples = X_meta_scaled.shape[0]
            B = min(200, max(20, max(20, n_samples // 2)))
            coefs_boot = np.zeros((B, self._n_base_models))
            rng = np.random.default_rng(42)
            for b in range(B):
                idx = rng.integers(0, n_samples, n_samples)
                Xb = X_meta_scaled[idx]
                yb = y_true[idx]
                model_b = Ridge(alpha=self.alpha, fit_intercept=True)
                model_b.fit(Xb, yb)
                coefs = model_b.coef_
                # If LSTM was not present during fit, coefs length matches
                coefs_boot[b, : len(coefs)] = coefs[: self._n_base_models]

            coef_se = coefs_boot.std(axis=0)
            self._coef_se = {
                name: float(se) for name, se in zip(self._model_names, coef_se)
            }
            logger.info(f"[meta_learner] Coef SE: {self._coef_se}")
        except Exception:
            self._coef_se = None

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        lgbm_pred:  float,
        xgb_pred:   float,
        arima_pred: float,
        lstm_pred:  Optional[float] = None,
    ) -> float:
        """
        Blend base model predictions into a single final prediction.

        Args:
            lgbm_pred:  LightGBM scalar prediction
            xgb_pred:   XGBoost scalar prediction
            arima_pred: ARIMA scalar prediction
            lstm_pred:  Optional LSTM scalar prediction

        Returns:
            Blended prediction as float.
        """
        if not self._is_fitted:
            return self._fallback_blend(lgbm_pred, xgb_pred, arima_pred, lstm_pred)

        preds = [lgbm_pred, xgb_pred, arima_pred]
        if lstm_pred is not None and self._n_base_models == 4:
            preds.append(lstm_pred)

        X = np.array(preds).reshape(1, -1)

        # If shape mismatch (e.g. LSTM added after meta-learner was trained without it),
        # fall back gracefully rather than crashing at inference time
        if X.shape[1] != self._n_base_models:
            logger.warning(
                f"[meta_learner] Shape mismatch: expected {self._n_base_models} "
                f"inputs, got {X.shape[1]}. Using fallback weights."
            )
            return self._fallback_blend(lgbm_pred, xgb_pred, arima_pred, lstm_pred)

        X_scaled = self._scaler.transform(X)
        return float(self._meta_model.predict(X_scaled)[0])

    def predict_batch(
        self,
        lgbm_preds:  np.ndarray,
        xgb_preds:   np.ndarray,
        arima_preds: np.ndarray,
        lstm_preds:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Batch version of predict() for use in predict_historical().

        Args:
            lgbm_preds:  Array of LightGBM predictions
            xgb_preds:   Array of XGBoost predictions
            arima_preds: Array of ARIMA predictions
            lstm_preds:  Optional array of LSTM predictions

        Returns:
            Array of blended predictions.
        """
        if not self._is_fitted:
            return self._fallback_blend_batch(lgbm_preds, xgb_preds, arima_preds, lstm_preds)

        preds_list = [lgbm_preds, xgb_preds, arima_preds]
        if lstm_preds is not None and self._n_base_models == 4:
            preds_list.append(lstm_preds)

        X = np.column_stack(preds_list)
        if X.shape[1] != self._n_base_models:
            logger.warning(
                f"[meta_learner] Shape mismatch in batch predict. Using fallback weights."
            )
            return self._fallback_blend_batch(lgbm_preds, xgb_preds, arima_preds, lstm_preds)

        X_scaled = self._scaler.transform(X)
        return self._meta_model.predict(X_scaled)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, ticker: str, models_dir: Path) -> None:
        """Save meta-learner to saved_models/."""
        if not self._is_fitted:
            logger.warning("[meta_learner] Not fitted — nothing to save.")
            return
        path = models_dir / f"meta_learner_{ticker.upper()}.pkl"
        joblib.dump({
            "meta_model":    self._meta_model,
            "scaler":        self._scaler,
            "model_names":   self._model_names,
            "n_base_models": self._n_base_models,
            "coef_se":       self._coef_se,
        }, path)
        logger.info(f"[meta_learner] Saved → {path.name}")

    def load(self, ticker: str, models_dir: Path) -> None:
        """Load meta-learner from saved_models/."""
        path = models_dir / f"meta_learner_{ticker.upper()}.pkl"
        if not path.exists():
            logger.warning(
                f"[meta_learner] No saved meta-learner for {ticker} at {path}. "
                f"Using fallback weights: {self._default_weights}"
            )
            return
        data = joblib.load(path)
        self._meta_model    = data["meta_model"]
        self._scaler        = data["scaler"]
        self._model_names   = data.get("model_names", ["lgbm", "xgb", "arima"])
        self._n_base_models = data.get("n_base_models", 3)
        self._coef_se       = data.get("coef_se", None)
        self._is_fitted     = True
        logger.info(
            f"[meta_learner] Loaded from {path.name} "
            f"({self._n_base_models} base models: {self._model_names})"
        )

    def learned_weights_with_se(self) -> dict:
        """
        Return learned weights and their bootstrap standard errors (if available).
        """
        weights = self.learned_weights()
        if not self._is_fitted:
            return {"weights": weights, "se": None}
        return {"weights": weights, "se": self._coef_se}

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def learned_weights(self) -> dict:
        """
        Return the learned Ridge coefficients as a named dict.
        Useful for paper reporting and ablation studies.

        Note: these are the Ridge coefficients on the SCALED inputs,
        so they are not directly interpretable as "fraction of weight"
        the way fixed weights (0.50, 0.35, 0.15) are. Use them for
        relative comparison only.
        """
        if not self._is_fitted:
            return {f"fallback_{k}": v for k, v in self._default_weights.items()}
        return {
            name: float(coef)
            for name, coef in zip(self._model_names, self._meta_model.coef_)
        }

    # -----------------------------------------------------------------------
    # Private fallback helpers
    # -----------------------------------------------------------------------

    def _fallback_blend(
        self,
        lgbm_pred:  float,
        xgb_pred:   float,
        arima_pred: float,
        lstm_pred:  Optional[float] = None,
    ) -> float:
        """Fallback to config default weights when meta-learner not fitted."""
        w = self._default_weights
        result = (
            w["lgbm"]  * lgbm_pred  +
            w["xgb"]   * xgb_pred   +
            w["arima"] * arima_pred
        )
        if lstm_pred is not None and "lstm" in w:
            result += w["lstm"] * lstm_pred
        return float(result)

    def _fallback_blend_batch(
        self,
        lgbm_preds:  np.ndarray,
        xgb_preds:   np.ndarray,
        arima_preds: np.ndarray,
        lstm_preds:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch fallback to default weights."""
        w = self._default_weights
        result = (
            w["lgbm"]  * lgbm_preds  +
            w["xgb"]   * xgb_preds   +
            w["arima"] * arima_preds
        )
        if lstm_preds is not None and "lstm" in w:
            result += w["lstm"] * lstm_preds
        return result