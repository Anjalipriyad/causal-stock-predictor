"""
ensemble.py
-----------
Blends LightGBM + XGBoost + ARIMA (+ optional LSTM) using a learned
Ridge Regression meta-learner rather than fixed hardcoded weights.

Architecture change from v1:
    BEFORE: pred = 0.50*lgbm + 0.35*xgb + 0.15*arima  (fixed weights)
    AFTER:  pred = Ridge(lgbm_pred, xgb_pred, arima_pred)  (learned weights)

The StackingMetaLearner is trained on VALIDATION SET predictions from the
base models, so it learns which model to trust more for this ticker/regime
without any data leakage. Falls back to fixed weights gracefully if the
meta-learner hasn't been trained yet (backward compatible with saved models).

LSTM is optional — if PyTorch is not installed, the ensemble runs as
LightGBM + XGBoost + ARIMA + fixed-weight fallback, identical to v1.

The ensemble is the ONLY entry point used by:
    - backtester.py         (evaluation)
    - prediction_service.py (backend inference)
    - run_pipeline.py       (CLI)

Usage:
    from ml.src.ensemble import Ensemble

    # Load pre-trained models
    ensemble = Ensemble()
    ensemble.load("AAPL")

    # Live inference
    result = ensemble.predict_live(
        live_features=feature_row,
        ticker="AAPL",
        current_price=213.40,
        causal_features=["vix_change_1d", "sentiment_ma_5d", ...]
    )
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.models.base_model import PredictionResult
from ml.src.models.lgbm_model import LGBMModel
from ml.src.models.xgb_model import XGBModel
from ml.src.models.arima_model import ARIMAModel
from ml.src.models.stacking_meta_learner import StackingMetaLearner
from ml.src.models.meta_classifier import MetaClassifier
from ml.src.causal.selector import CausalSelector
from ml.src.models.tuner import HyperparameterTuner

# Optional LSTM import — fails gracefully if PyTorch not installed
try:
    from ml.src.models.lstm_model import LSTMModel
    _LSTM_IMPORT_OK = True
except ImportError:
    _LSTM_IMPORT_OK = False

# Optional TFT import
try:
    from ml.src.models.tft_model import TFTModel
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class Ensemble:
    """
    Stacking ensemble: LightGBM + XGBoost + ARIMA + optional LSTM.
    Blended via a learned Ridge Regression meta-learner.

    Backward compatibility:
        If no meta_learner_{ticker}.pkl exists on disk, falls back to the
        config-specified fixed weights (0.50/0.35/0.15). Existing saved
        models continue to work without retraining.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        # Allow callers to pass an already-loaded config dict (overrides file)
        self.cfg      = cfg if cfg is not None else _load_config(config_path)
        self.root     = Path(__file__).resolve().parents[2]

        self.confidence_z = self.cfg["model"]["ensemble"]["confidence_z"]
        self.horizon      = self.cfg["model"]["horizon_days"]
        self.target_col   = self.cfg["model"]["target"]

        # ── Base models ────────────────────────────────────────────────────
        # Pass the loaded config to submodels so runtime overrides are honoured
        self.lgbm  = LGBMModel(config_path, self.cfg)
        self.xgb   = XGBModel(config_path, self.cfg)
        self.arima = ARIMAModel(config_path, self.cfg)

        # ── LSTM base learner (optional) ───────────────────────────────────
        if _LSTM_IMPORT_OK:
            self.lstm = LSTMModel(config_path, self.cfg)
            self._lstm_enabled = self.lstm.is_available()
        else:
            self.lstm          = None
            self._lstm_enabled = False

        # ── Meta-learner (replaces fixed weights) ─────────────────────────
        # Initialize with config default weights as fallback
        w = self.cfg["model"]["ensemble"]["weights"]
        self.meta_learner = StackingMetaLearner(
            alpha=1.0,
            default_weights=w,
        )

        # ── Support infrastructure ─────────────────────────────────────────
        self.selector = CausalSelector(config_path, self.cfg)
        models_dir    = self.cfg["saved_models"]["dir"]
        self.models_dir = self.root / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._ticker:           Optional[str]       = None
        self._causal_features:  Optional[list[str]] = None
        self._is_loaded:        bool                = False
        self.meta_classifier:    Optional[MetaClassifier] = None
        self._thresholds:        dict = {}

    # -----------------------------------------------------------------------
    # Load / train
    # -----------------------------------------------------------------------

    def load(self, ticker: str) -> None:
        """
        Load all trained models + meta-learner + causal feature list.
        Must be called before predict_live() or predict_historical().
        """
        ticker = ticker.upper()
        logger.info(f"[ensemble] Loading models for {ticker} ...")

        self.lgbm.load(ticker)
        self.xgb.load(ticker)
        self.arima.load(ticker)
        self.lgbm.load_scaler(ticker)
        self.xgb.load_scaler(ticker)

        # Load LSTM if available and saved
        if self._lstm_enabled and self.lstm is not None:
            try:
                self.lstm.load(ticker)
                logger.info(f"[ensemble] LSTM model loaded for {ticker}.")
            except FileNotFoundError:
                logger.info(
                    f"[ensemble] No LSTM model found for {ticker} — "
                    "LSTM will be skipped at inference."
                )
                self._lstm_enabled = False

        # Load meta-learner (falls back to fixed weights if not found)
        self.meta_learner.load(ticker, self.models_dir)

        # Attempt to load an optional meta-classifier (classification fallback)
        try:
            self.meta_classifier = MetaClassifier.load(ticker, models_dir=self.models_dir)
            logger.info(f"[ensemble] Meta-classifier loaded for {ticker}.")
        except Exception:
            self.meta_classifier = None

        # Load tuned thresholds if present
        try:
            import json
            path = self.models_dir / f"thresholds_{ticker}.json"
            if path.exists():
                with open(path) as f:
                    self._thresholds = json.load(f)
                    logger.info(f"[ensemble] Loaded thresholds for {ticker}: {self._thresholds}")
        except Exception:
            self._thresholds = {}

        self._causal_features = self.selector.load(ticker)
        self._ticker          = ticker
        self._is_loaded       = True

        logger.info(
            f"[ensemble] Loaded. Causal features ({len(self._causal_features)}): "
            f"{self._causal_features}"
        )

    def train_all(self, df, ticker, causal_features, refit_arima=False):
        """
        Train all base models and the stacking meta-learner.

        ARIMA is fitted on the TRAIN set only; the meta-learner receives
        ARIMA's rolling one-step-ahead forecasts on the VALIDATION set via
        `arima.predict_val_set(y_val)` to avoid broadcasting a constant.
        """
        ticker = ticker.upper()
        logger.info(f"[ensemble] Training all models for {ticker} ...")

        # ── 1. Data split ──────────────────────────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.lgbm.prepare_data(df, causal_features)

        # ── 2. Scale ─────────────────────────────────────────────────────
        X_train_lgbm_s, X_val_lgbm_s, X_test_lgbm_s = self.lgbm.scale(X_train, X_val, X_test, ticker)
        X_train_xgb_s, X_val_xgb_s, X_test_xgb_s = self.xgb.scale(X_train, X_val, X_test, ticker)

        # ── 3. Optional Optuna tuning ───────────────────────────────────
        tuner = HyperparameterTuner()
        if tuner.enabled:
            logger.info("[ensemble] Running Optuna hyperparameter tuning ...")
            best_lgbm = tuner.tune_lgbm(X_train_lgbm_s, y_train, X_val_lgbm_s, y_val, ticker)
            best_xgb  = tuner.tune_xgb(X_train_xgb_s, y_train, X_val_xgb_s, y_val, ticker)
            self.lgbm._params.update(best_lgbm)
            self.xgb._params.update(best_xgb)

        # ── 4. Train base models ─────────────────────────────────────────
        self.lgbm.fit(X_train_lgbm_s, y_train, X_val_lgbm_s, y_val)
        self.lgbm.save(ticker)

        self.xgb.fit(X_train_xgb_s, y_train, X_val_xgb_s, y_val)
        self.xgb.save(ticker)

        # ARIMA fitted on training data only (not val)
        self.arima.fit(X_train, y_train, X_val=None, y_val=None)

        # ── 5. Optional LSTM training ───────────────────────────────────
        lstm_val_preds = None
        if self._lstm_enabled and self.lstm is not None:
            try:
                self.lstm.fit(X_train_lgbm_s, y_train, X_val_lgbm_s, y_val)
                if self.lstm._is_fitted:
                    self.lstm.save(ticker)
                    lstm_val_preds = self.lstm.predict_raw(X_val_lgbm_s)
                    logger.info("[ensemble] LSTM trained and saved.")
            except Exception as e:
                logger.warning(f"[ensemble] LSTM training failed: {e}. Skipping.")
                self._lstm_enabled = False

        # ── 6. Train meta-learner on VALIDATION SET predictions ──────────
        val_lgbm  = self.lgbm.predict_raw(X_val_lgbm_s)
        val_xgb   = self.xgb.predict_raw(X_val_xgb_s)

        # Key fix: rolling one-step forecasts from ARIMA on the validation set
        val_arima = self.arima.predict_val_set(y_val)

        # Sanity check for non-constant ARIMA forecasts
        if val_arima.std() < 1e-10:
            logger.warning(
                "[ensemble] ARIMA val predictions are near-constant; "
                "meta-learner may assign near-zero weight to ARIMA."
            )
        else:
            logger.info(
                f"[ensemble] ARIMA val prediction std={val_arima.std():.5f} ✓"
            )

        self.meta_learner.fit(
            lgbm_preds  = val_lgbm,
            xgb_preds   = val_xgb,
            arima_preds = val_arima,
            y_true      = y_val.values,
            lstm_preds  = lstm_val_preds,
        )
        self.meta_learner.save(ticker, self.models_dir)

        self._ticker          = ticker
        self._causal_features = causal_features
        self._is_loaded       = True

        logger.info(f"[ensemble] All models trained and saved for {ticker}.")

        try:
            lw = self.meta_learner.learned_weights_with_se()
            logger.info(
                f"[ensemble] Meta-learner learned weights: {lw['weights']} (se={lw['se']})"
            )
        except Exception:
            logger.info(
                f"[ensemble] Meta-learner learned weights: {self.meta_learner.learned_weights()}"
            )

        # Final ARIMA refit on train+val for the persisted live-inference model
        if refit_arima:
            try:
                self.arima.fit(X_train, y_train, X_val, y_val)
                self.arima.save(ticker)
                logger.info("[ensemble] ARIMA refit on train+val and saved.")
                logger.info(
                    "[ensemble] Note: meta-learner was trained on ARIMA val-forecasts "
                    "produced by the train-only ARIMA. Final ARIMA refit on train+val "
                    "is intentional for live inference (standard practice)."
                )
            except Exception as e:
                logger.warning(f"[ensemble] ARIMA final refit failed: {e}")
        else:
            try:
                # Save the train-only ARIMA without refitting (Issue #9 strict protocol)
                self.arima.save(ticker)
                logger.info("[ensemble] ARIMA saved without train+val refit "
                            "(strict meta-learner alignment protocol).")
            except Exception as e:
                logger.warning(f"[ensemble] ARIMA save failed: {e}")

        return X_test_lgbm_s, y_test
    # -----------------------------------------------------------------------
    # Inference — live
    # -----------------------------------------------------------------------

    def predict_live(
        self,
        live_features:   pd.Series,
        ticker:          str,
        current_price:   float,
        causal_features: Optional[list[str]] = None,
    ) -> PredictionResult:
        """
        Produce a single blended PredictionResult for live inference.
        Uses the learned meta-learner weights (or fixed fallback).
        """
        self._check_loaded()
        ticker   = ticker.upper()
        features = causal_features or self._causal_features

        # Convert Series to single-row DataFrame
        X = live_features.to_frame().T
        X = X[[c for c in features if c in X.columns]]

        # Scale
        X_lgbm = self.lgbm.transform(X)
        X_xgb  = self.xgb.transform(X)

        # Base model raw predictions
        pred_lgbm  = float(self.lgbm.predict_raw(X_lgbm)[0])
        pred_xgb   = float(self.xgb.predict_raw(X_xgb)[0])
        pred_arima = float(self.arima.predict_raw(X)[0])

        # Optional LSTM prediction
        pred_lstm = None
        if self._lstm_enabled and self.lstm is not None and self.lstm._is_fitted:
            try:
                # LSTM needs context rows — if only 1 row available, skip
                pred_lstm = float(self.lstm.predict_raw(X_lgbm)[0])
            except Exception as e:
                logger.debug(f"[ensemble] LSTM live prediction failed: {e}")

        # Choose classifier-based direction if configured and available
        use_clf = bool(self.cfg["model"].get("use_meta_classifier", False)) and (self.meta_classifier is not None)
        if use_clf:
            preds_row = {"lgbm": np.array([pred_lgbm]), "xgb": np.array([pred_xgb]), "arima": np.array([pred_arima])}
            if pred_lstm is not None:
                preds_row["lstm"] = np.array([pred_lstm])
            prob = float(self.meta_classifier.predict_proba(preds_row)[0])
            clf_thresh = float(self._thresholds.get("classifier_threshold", 0.5))
            scale = float(self.cfg["model"].get("classifier_return_scale", 0.02))
            pred_blended = (prob - clf_thresh) * scale
            direction = "UP" if prob >= clf_thresh else "DOWN"
        else:
            pred_blended = self.meta_learner.predict(
                lgbm_pred  = pred_lgbm,
                xgb_pred   = pred_xgb,
                arima_pred = pred_arima,
                lstm_pred  = pred_lstm,
            )
            reg_thresh = float(self._thresholds.get("regressor_threshold", 0.0))
            direction = "UP" if pred_blended >= reg_thresh else "DOWN"

        # Predicted price
        pred_price = current_price * np.exp(pred_blended)

        # Confidence & bands — prefer split-conformal intervals if available
        try:
            conf_q = float(self._thresholds.get("conformal_q")) if self._thresholds else None
        except Exception:
            conf_q = None

        if conf_q is not None and conf_q > 0:
            # Use conformal interval on log-return scale
            alpha = float(self.cfg["model"]["ensemble"].get("conformal_alpha", 0.10))
            conf_scale = float(self.cfg["model"]["ensemble"].get("conformal_conf_scale", 3.0))
            # Confidence decreases as predicted magnitude grows relative to conformal width
            confidence = 1.0 - min(abs(pred_blended) / (conf_scale * conf_q + 1e-12), 1.0)
            confidence = float(np.clip(confidence, 0.3, 0.9))
            upper = current_price * np.exp(pred_blended + conf_q)
            lower = current_price * np.exp(pred_blended - conf_q)
        else:
            # Fallback: original volatility-based bands and blended confidences
            conf_lgbm = self.lgbm._compute_confidence(np.array([pred_lgbm]),  X_lgbm)
            conf_xgb  = self.xgb._compute_confidence(np.array([pred_xgb]),   X_xgb)
            # Prefer meta-learner's learned emphasis for confidence weighting
            try:
                if getattr(self.meta_learner, "_is_fitted", False):
                    learned = self.meta_learner.learned_weights()
                    # Convert to non-negative importance and normalise
                    pos = {k: max(0.0, float(learned.get(k, 0.0))) for k in ["lgbm", "xgb", "arima"]}
                    total = sum(pos.values())
                    if total > 0:
                        norm = {k: pos[k] / total for k in pos}
                    else:
                        norm = self.cfg["model"]["ensemble"]["weights"]
                else:
                    norm = self.cfg["model"]["ensemble"]["weights"]
            except Exception:
                norm = self.cfg["model"]["ensemble"]["weights"]

            arima_w = norm.get("arima", 0.15)
            confidence = (
                norm.get("lgbm", 0.0) * conf_lgbm
                + norm.get("xgb", 0.0) * conf_xgb
                + arima_w * 0.5
            )
            confidence = float(np.clip(confidence, 0.3, 0.9))

            # Confidence band — volatility based
            vol   = self.lgbm._estimate_volatility(X_lgbm)
            z     = self.confidence_z
            upper = current_price * np.exp(pred_blended + z * vol * np.sqrt(self.horizon))
            lower = current_price * np.exp(pred_blended - z * vol * np.sqrt(self.horizon))

        # Causal drivers from LightGBM (SHAP-based, highest interpretability)
        drivers = self.lgbm._extract_drivers(X_lgbm, features)

        from datetime import datetime
        return PredictionResult(
            ticker=ticker,
            predicted_return=round(pred_blended, 6),
            predicted_price=round(pred_price, 2),
            current_price=round(current_price, 2),
            direction=direction,
            confidence=round(confidence, 3),
            upper_band=round(upper, 2),
            lower_band=round(lower, 2),
            causal_drivers=drivers,
            model_name="ensemble(lgbm+xgb+arima+meta)",
            horizon_days=self.horizon,
            prediction_date=datetime.today().strftime("%Y-%m-%d"),
        )

    # -----------------------------------------------------------------------
    # Inference — historical (backtesting)
    # -----------------------------------------------------------------------

    def predict_historical(
        self,
        df: pd.DataFrame,
        causal_features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for an entire DataFrame.
        Used by backtester.py for walk-forward and regime evaluation.

        Returns:
            DataFrame with columns:
                predicted_return  — meta-learner blended prediction
                actual_return     — actual log_return target
                direction_pred    — 1=UP, 0=DOWN
                direction_actual  — 1=UP, 0=DOWN
                lgbm_pred         — individual model outputs (for ablation)
                xgb_pred
                arima_pred
                lstm_pred         — only present if LSTM trained
        """
        self._check_loaded()
        features  = causal_features or self._causal_features
        feat_cols = [c for c in features if c in df.columns]
        X         = df[feat_cols]
        y         = df[self.target_col] if self.target_col in df.columns else None

        # Scale
        X_lgbm = self.lgbm.transform(X)
        X_xgb  = self.xgb.transform(X)

        # Base model predictions
        pred_lgbm  = self.lgbm.predict_raw(X_lgbm)
        pred_xgb   = self.xgb.predict_raw(X_xgb)
        # ARIMA: Use rolling predict-then-update forecasting when actual
        # targets are available. This is standard online forecasting — the
        # prediction at time t is made BEFORE seeing y[t], then y[t] is used
        # to update the model state for the next prediction. NOT leakage.
        # Falls back to predict_raw(X) when no targets available (live mode).
        pred_arima = self.arima.predict_val_set(y) if y is not None else self.arima.predict_raw(X)

        # Optional LSTM
        pred_lstm = None
        if self._lstm_enabled and self.lstm is not None and self.lstm._is_fitted:
            try:
                pred_lstm = self.lstm.predict_raw(X_lgbm)
            except Exception as e:
                logger.debug(f"[ensemble] LSTM historical predict failed: {e}")

        # Choose classifier-based direction if configured and available
        use_clf = bool(self.cfg["model"].get("use_meta_classifier", False)) and (self.meta_classifier is not None)
        if use_clf:
            preds_dict = {"lgbm": pred_lgbm, "xgb": pred_xgb, "arima": pred_arima}
            if pred_lstm is not None:
                preds_dict["lstm"] = pred_lstm
            probs = self.meta_classifier.predict_proba(preds_dict)
            clf_thresh = float(self._thresholds.get("classifier_threshold", 0.5))
            scale = float(self.cfg["model"].get("classifier_return_scale", 0.02))
            pred_blended = (probs - clf_thresh) * scale
        else:
            # Meta-learner blend
            pred_blended = self.meta_learner.predict_batch(
                lgbm_preds  = pred_lgbm,
                xgb_preds   = pred_xgb,
                arima_preds = pred_arima,
                lstm_preds  = pred_lstm,
            )

        result = pd.DataFrame(index=df.index)
        result["predicted_return"] = pred_blended
        result["lgbm_pred"]        = pred_lgbm
        result["xgb_pred"]         = pred_xgb
        result["arima_pred"]       = pred_arima
        if use_clf:
            result["direction_pred"] = (probs >= clf_thresh).astype(int)
        else:
            # If regressor blend used, convert via tuned regressor threshold (default 0)
            reg_thresh = float(self._thresholds.get("regressor_threshold", 0.0))
            result["direction_pred"] = (pred_blended >= reg_thresh).astype(int)

        if pred_lstm is not None:
            result["lstm_pred"] = pred_lstm

        if y is not None:
            result["actual_return"]    = y.values
            result["direction_actual"] = (y.values >= 0).astype(int)

        return result

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def set_weights(self, lgbm: float, xgb: float, arima: float) -> None:
        """
        Override ensemble fallback weights at runtime.
        Used for ablation studies. Does NOT affect the learned meta-learner.
        Must sum to 1.0.
        """
        assert abs(lgbm + xgb + arima - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.meta_learner._default_weights = {
            "lgbm": lgbm, "xgb": xgb, "arima": arima
        }
        logger.info(
            f"[ensemble] Fallback weights updated: "
            f"lgbm={lgbm}, xgb={xgb}, arima={arima}"
        )

    def _check_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                "Ensemble models not loaded. "
                "Call ensemble.load(ticker) or ensemble.train_all() first."
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ml.src.causal.selector import CausalSelector
    from ml.src.evaluation.metrics import Metrics
    from ml.src.data.loader import _load_config

    parser = argparse.ArgumentParser(description="Train ensemble for a ticker")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. AAPL or NIFTY")
    args = parser.parse_args()

    cfg    = _load_config()
    ticker = args.ticker.upper()
    root   = Path(__file__).resolve().parents[2]

    feat_path   = root / cfg["data"]["processed_dir"] / "features" / f"{ticker}_features.csv"
    causal_path = root / cfg["saved_models"]["dir"] / f"causal_features_{ticker}.json"

    if not feat_path.exists():
        print(f"ERROR: No feature matrix at {feat_path}")
        exit(1)
    if not causal_path.exists():
        print(f"ERROR: No causal features at {causal_path}")
        exit(1)

    df      = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    features = CausalSelector().load(ticker)

    print(f"\nTraining stacking ensemble for {ticker} ...")
    print(f"Causal features ({len(features)}): {features}")

    ensemble = Ensemble()
    X_test, y_test = ensemble.train_all(df, ticker, features)

    try:
        lw = ensemble.meta_learner.learned_weights_with_se()
        print(f"\nMeta-learner learned weights: {lw['weights']} (se={lw['se']})")
    except Exception:
        print(f"\nMeta-learner learned weights: {ensemble.meta_learner.learned_weights()}")

    test_df = pd.concat([X_test, y_test], axis=1)
    preds   = ensemble.predict_historical(test_df, features)
    metrics = Metrics()
    scores  = metrics.compute_all(
        preds["predicted_return"], preds["actual_return"], label="Test set"
    )

    print(f"\n=== TEST SET RESULTS ===")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"\nModels saved to: saved_models/")