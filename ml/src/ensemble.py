"""
ensemble.py
-----------
Blends LightGBM + XGBoost + ARIMA into a single PredictionResult.

Weights from config (must sum to 1.0):
    lgbm:  0.50
    xgb:   0.35
    arima: 0.15

The ensemble is the ONLY entry point used by:
    - backtester.py       (evaluation)
    - prediction_service.py (backend inference)

Nobody outside ml/ ever calls individual model files directly.

Two modes:
    predict_historical(df, ticker, causal_features)
        → returns pd.Series of predictions aligned to df index
        → used by backtester for walk-forward evaluation

    predict_live(live_features, ticker, current_price, causal_features)
        → returns single PredictionResult
        → used by backend at inference time

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
    print(result)
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
try:
    from ml.src.models.tft_model import TFTModel
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
from ml.src.causal.selector import CausalSelector
from ml.src.models.tuner import HyperparameterTuner

logger = logging.getLogger(__name__)


class Ensemble:
    """
    Weighted ensemble of LightGBM + XGBoost + ARIMA.

    Responsibilities:
        - Load all three models from saved_models/
        - Load causal feature list from saved_models/
        - Blend raw predictions using configured weights
        - Produce a single PredictionResult with blended confidence band
        - Extract causal drivers from the highest-weight model (LightGBM)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg      = _load_config(config_path)
        self.root     = Path(__file__).resolve().parents[2]

        # Weights
        w = self.cfg["model"]["ensemble"]["weights"]
        self.weights = {
            "lgbm":  w["lgbm"],
            "xgb":   w["xgb"],
            "arima": w["arima"],
        }
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"

        self.confidence_z = self.cfg["model"]["ensemble"]["confidence_z"]
        self.horizon      = self.cfg["model"]["horizon_days"]
        self.target_col   = self.cfg["model"]["target"]

        # Model instances
        self.lgbm  = LGBMModel(config_path)
        self.xgb   = XGBModel(config_path)
        self.arima = ARIMAModel(config_path)

        # Causal selector for loading feature list
        self.selector = CausalSelector(config_path)

        self._ticker: Optional[str] = None
        self._causal_features: Optional[list[str]] = None
        self._is_loaded: bool = False

    # -----------------------------------------------------------------------
    # Load / train
    # -----------------------------------------------------------------------

    def load(self, ticker: str) -> None:
        """
        Load all three trained models + causal feature list from saved_models/.
        Must be called before predict_live() or predict_historical().
        """
        ticker = ticker.upper()
        logger.info(f"[ensemble] Loading models for {ticker} ...")

        self.lgbm.load(ticker)
        self.xgb.load(ticker)
        self.arima.load(ticker)
        self.lgbm.load_scaler(ticker)
        self.xgb.load_scaler(ticker)

        self._causal_features = self.selector.load(ticker)
        self._ticker          = ticker
        self._is_loaded       = True

        logger.info(
            f"[ensemble] Loaded. Causal features ({len(self._causal_features)}): "
            f"{self._causal_features}"
        )

    def train_all(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
    ) -> None:
        """
        Train all three models on the feature matrix.
        Saves all models + scaler to saved_models/.

        Args:
            df:               Full feature matrix (from pipeline.build())
            ticker:           e.g. "AAPL"
            causal_features:  Output of CausalSelector.select()
        """
        ticker = ticker.upper()
        logger.info(f"[ensemble] Training all models for {ticker} ...")

        # Prepare splits using base_model helper
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.lgbm.prepare_data(df, causal_features)

        # Scale — fit on train, apply to all splits
        X_train_s, X_val_s, X_test_s = self.lgbm.scale(
            X_train, X_val, X_test, ticker
        )
        # XGBoost uses same scaler
        _, _, _ = self.xgb.scale(X_train, X_val, X_test, ticker)

        # Optuna tuning (if enabled in config)
        tuner = HyperparameterTuner()
        if tuner.enabled:
            logger.info("[ensemble] Running Optuna hyperparameter tuning...")
            best_lgbm = tuner.tune_lgbm(X_train_s, y_train, X_val_s, y_val, ticker)
            best_xgb  = tuner.tune_xgb(X_train_s, y_train, X_val_s, y_val, ticker)
            # Apply tuned params
            self.lgbm._params.update(best_lgbm)
            self.xgb._params.update(best_xgb)

        # Train LightGBM
        self.lgbm.fit(X_train_s, y_train, X_val_s, y_val)
        self.lgbm.save(ticker)

        # Train XGBoost
        self.xgb.fit(X_train_s, y_train, X_val_s, y_val)
        self.xgb.save(ticker)

        # Train ARIMA (univariate — uses y only)
        self.arima.fit(X_train, y_train, X_val, y_val)
        self.arima.save(ticker)

        self._ticker          = ticker
        self._causal_features = causal_features
        self._is_loaded       = True

        logger.info(f"[ensemble] All models trained and saved for {ticker}.")
        return X_test_s, y_test   # return test set for immediate evaluation

    # -----------------------------------------------------------------------
    # Inference — live
    # -----------------------------------------------------------------------

    def predict_live(
        self,
        live_features: pd.Series,
        ticker: str,
        current_price: float,
        causal_features: Optional[list[str]] = None,
    ) -> PredictionResult:
        """
        Produce a single blended PredictionResult for live inference.

        Args:
            live_features:   Single feature row (pd.Series) from pipeline.build_live()
            ticker:          Stock ticker
            current_price:   Latest close price
            causal_features: Override causal feature list (uses loaded list if None)

        Returns:
            PredictionResult with blended prediction + confidence band + drivers
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

        # Raw predictions from each model
        pred_lgbm  = float(self.lgbm.predict_raw(X_lgbm)[0])
        pred_xgb   = float(self.xgb.predict_raw(X_xgb)[0])
        pred_arima = float(self.arima.predict_raw(X)[0])

        # Weighted blend
        pred_blended = (
            self.weights["lgbm"]  * pred_lgbm  +
            self.weights["xgb"]   * pred_xgb   +
            self.weights["arima"] * pred_arima
        )

        # TFT blend — only if installed and trained
        tft_enabled = getattr(self, "_tft_enabled", False)
        if tft_enabled and getattr(self, "tft", None) is not None:
            try:
                if self.tft.is_available():
                    pred_tft     = float(self.tft.predict_raw(X)[0])
                    pred_blended = (
                        self.weights["lgbm"]  * pred_lgbm  +
                        self.weights["xgb"]   * pred_xgb   +
                        self.weights["arima"] * pred_arima +
                        self.weights.get("tft", 0) * pred_tft
                    )
            except Exception as e:
                logger.warning(f"[ensemble] TFT prediction failed: {e}")

        # Predicted price
        pred_price = current_price * np.exp(pred_blended)

        # Direction
        direction = "UP" if pred_blended >= 0 else "DOWN"

        # Confidence — blend individual confidences
        conf_lgbm  = self.lgbm._compute_confidence(np.array([pred_lgbm]),  X_lgbm)
        conf_xgb   = self.xgb._compute_confidence(np.array([pred_xgb]),    X_xgb)
        confidence = (
            self.weights["lgbm"]  * conf_lgbm +
            self.weights["xgb"]   * conf_xgb  +
            self.weights["arima"] * 0.5         # ARIMA has no confidence signal
        )

        # Confidence band — use blended volatility estimate
        vol   = self.lgbm._estimate_volatility(X_lgbm)
        z     = self.confidence_z
        upper = current_price * np.exp(pred_blended + z * vol * np.sqrt(self.horizon))
        lower = current_price * np.exp(pred_blended - z * vol * np.sqrt(self.horizon))

        # Causal drivers from LightGBM (highest weight + SHAP)
        drivers = self.lgbm._extract_drivers(X_lgbm, features)

        from datetime import datetime
        return PredictionResult(
            ticker=ticker,
            predicted_return=round(pred_blended, 6),
            predicted_price=round(pred_price, 2),
            current_price=round(current_price, 2),
            direction=direction,
            confidence=round(float(confidence), 3),
            upper_band=round(upper, 2),
            lower_band=round(lower, 2),
            causal_drivers=drivers,
            model_name="ensemble(lgbm+xgb+arima)",
            horizon_days=self.horizon,
            prediction_date=datetime.today().strftime("%Y-%m-%d"),
        )

    # -----------------------------------------------------------------------
    # Inference — historical (for backtesting)
    # -----------------------------------------------------------------------

    def predict_historical(
        self,
        df: pd.DataFrame,
        causal_features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for an entire DataFrame.
        Used by backtester.py for walk-forward evaluation.

        Args:
            df:              Feature matrix with DatetimeIndex
            causal_features: Override causal feature list

        Returns:
            DataFrame with columns:
                predicted_return  — blended log return prediction
                actual_return     — actual log_return_5d (target)
                direction_pred    — predicted direction (1=UP, 0=DOWN)
                direction_actual  — actual direction
                lgbm_pred         — individual model predictions
                xgb_pred
                arima_pred
        """
        self._check_loaded()
        features = causal_features or self._causal_features

        # Extract features + target
        feat_cols = [c for c in features if c in df.columns]
        X = df[feat_cols]
        y = df[self.target_col] if self.target_col in df.columns else None

        # Scale
        X_lgbm = self.lgbm.transform(X)
        X_xgb  = self.xgb.transform(X)

        # Predictions
        pred_lgbm  = self.lgbm.predict_raw(X_lgbm)
        pred_xgb   = self.xgb.predict_raw(X_xgb)
        pred_arima = self.arima.predict_raw(X)

        # Blend
        pred_blended = (
            self.weights["lgbm"]  * pred_lgbm  +
            self.weights["xgb"]   * pred_xgb   +
            self.weights["arima"] * pred_arima
        )

        result = pd.DataFrame(index=df.index)
        result["predicted_return"] = pred_blended
        result["lgbm_pred"]        = pred_lgbm
        result["xgb_pred"]         = pred_xgb
        result["arima_pred"]       = pred_arima
        result["direction_pred"]   = (pred_blended >= 0).astype(int)

        if y is not None:
            result["actual_return"]    = y.values
            result["direction_actual"] = (y.values >= 0).astype(int)

        return result

    # -----------------------------------------------------------------------
    # Model weights management
    # -----------------------------------------------------------------------

    def set_weights(
        self, lgbm: float, xgb: float, arima: float
    ) -> None:
        """
        Override ensemble weights at runtime.
        Useful for ablation studies — test different weight combinations.
        Must sum to 1.0.
        """
        assert abs(lgbm + xgb + arima - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.weights = {"lgbm": lgbm, "xgb": xgb, "arima": arima}
        logger.info(
            f"[ensemble] Weights updated: "
            f"lgbm={lgbm}, xgb={xgb}, arima={arima}"
        )

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

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
    import pandas as pd
    from ml.src.causal.selector import CausalSelector
    from ml.src.models.tuner import HyperparameterTuner
    from ml.src.evaluation.metrics import Metrics
    from ml.src.data.loader import _load_config

    parser = argparse.ArgumentParser(description="Train ensemble for a ticker")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. AAPL")
    args = parser.parse_args()

    cfg    = _load_config()
    ticker = args.ticker.upper()
    root   = Path(__file__).resolve().parents[2]

    feat_path   = root / cfg["data"]["processed_dir"] / "features" / f"{ticker}_features.csv"
    causal_path = root / cfg["saved_models"]["dir"] / f"causal_features_{ticker}.json"

    if not feat_path.exists():
        print(f"ERROR: No feature matrix at {feat_path}")
        print(f"Run: python -m ml.src.features.pipeline --ticker {ticker}")
        exit(1)

    if not causal_path.exists():
        print(f"ERROR: No causal features at {causal_path}")
        print(f"Run: python -m ml.src.causal.selector --ticker {ticker}")
        exit(1)

    df       = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    selector = CausalSelector()
    causal_features = selector.load(ticker)

    print(f"\nTraining ensemble for {ticker}...")
    print(f"Causal features ({len(causal_features)}): {causal_features}")

    ensemble = Ensemble()
    X_test, y_test = ensemble.train_all(df, ticker, causal_features)

    # Evaluate on test set
    test_df = pd.concat([X_test, y_test], axis=1)
    preds   = ensemble.predict_historical(test_df, causal_features)
    metrics = Metrics()
    scores  = metrics.compute_all(
        preds["predicted_return"], preds["actual_return"], label="Test set"
    )

    print(f"\n=== TEST SET RESULTS ===")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"\nModels saved to: saved_models/")